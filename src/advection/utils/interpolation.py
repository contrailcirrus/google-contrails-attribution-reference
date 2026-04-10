"""Functions for interpolating xarray data."""

import abc
from collections.abc import Callable, Sequence
import enum
from typing import Optional, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate as sp_interpolate
import tensorflow_probability.substrates.jax as tfp
import xarray as xr

from . import ce_types
from . import conform
from . import dataset_util as dsu


config = jax.config
Array = ce_types.Array

config.update('jax_default_matmul_precision', 'float32')
config.update('jax_enable_x64', True)


class OutOfBoundsError(Exception):
  """Raise if dimension values are out of bounds of an interpolation table."""

  def __init__(self, message, oob_points, dim_limits):
    super().__init__(message)
    self.oob_points = oob_points
    self.dim_limits = dim_limits


class GridError(Exception):
  """Raise if a grid does not conform to expectations."""


class InterpolationEngine(enum.Enum):
  SCIPY = 'SCIPY'
  TFP_REGULAR = 'TFP_REGULAR'
  TFP_RECTILINEAR = 'TFP_RECTILINEAR'


EXTRAPOLATE = 'extrapolate'
CONSTANT_EXTENSION = 'constant_extension'


class DatasetInterpolator:
  """Interpolate xarray.Dataset on a regular grid."""

  def __init__(
      self,
      ds: xr.Dataset,
      variables: Sequence[str],
      batch_dims: Optional[Sequence[str]] = None,
      logspace_dims: Optional[Sequence[str]] = None,
      logspace_vars: Optional[Sequence[str]] = None,
      interpolation_engine: InterpolationEngine = InterpolationEngine.SCIPY,
      fill_value: Union[str, Array] = np.nan,
      nan_safe: bool = False,
      jit: bool = False,
  ):
    batch_dims = batch_dims or ()
    logspace_dims = logspace_dims or ()
    logspace_vars = logspace_vars or ()

    self._dims = tuple(
        cast(str, d) for d in ds[variables[0]].dims if d not in batch_dims
    )
    self._batch_dims = tuple(batch_dims)
    ds = ds.transpose(*(self.dims + self.batch_dims))

    dsu.assert_valid_dataset_dims(
        ds,
        expected_dims=self.dims + self.batch_dims,
        variables=variables,
        expect_sorted=True,
    )
    dim_points = {d: ds[d].values for d in self.dims}
    self._dim_limits = {
        d: (np.min(v), np.max(v)) for d, v in dim_points.items()
    }

    table_kwargs = {}
    if interpolation_engine == InterpolationEngine.SCIPY:
      self._lib = np
      table_cls = sp_interpolate.RegularGridInterpolator
      table_kwargs['bounds_error'] = False
    elif interpolation_engine == InterpolationEngine.TFP_RECTILINEAR:
      self._lib = jnp
      table_cls = TFPRectilinearGridInterpolator
      table_kwargs['jit'] = jit
    elif interpolation_engine == InterpolationEngine.TFP_REGULAR:
      self._lib = jnp
      table_cls = TFPRegularGridInterpolator
      table_kwargs['jit'] = jit
    else:
      raise NotImplementedError(
          f'`interpolation_engine` {interpolation_engine} not handled.'
      )

    self._interpolation_engine = interpolation_engine
    if nan_safe:
      table_cls = _nan_safe_wrapper(table_cls)

    self._table = table_cls(  # type: ignore
        points=tuple(
            self._maybe_ln(k, v, logspace_dims) for k, v in dim_points.items()
        ),
        values=np.stack(
            [self._maybe_ln(k, ds[k].values, logspace_vars) for k in variables],
            axis=-1,
        ),
        fill_value=self._get_fill_value_for_engine_and_set_should_clip_points(
            fill_value, interpolation_engine
        ),
        **table_kwargs,
    )
    self._variables = tuple(variables)
    self._logspace_dims = tuple(logspace_dims)
    self._logspace_vars = tuple(logspace_vars)
    self._batch_coords = {d: ds[d].copy() for d in self.batch_dims}
    self._dim_points = {d: dim_points[d].copy() for d in self.dims}

  def _clip_points_to_make_in_bounds(self, points: Array) -> Array:
    """Clip points so that they are in bounds."""
    new_points = []
    for i, (lower, upper) in enumerate(self._dim_limits.values()):
      new_points.append(self._lib.clip(points[..., i], lower, upper))
    return self._lib.stack(new_points, axis=-1)

  def _get_fill_value_for_engine_and_set_should_clip_points(
      self,
      fill_value: Union[str, Array],
      interpolation_engine: InterpolationEngine,
  ) -> Optional[Union[str, Array]]:
    """Processes fill_value and sets should_clip_points for the chosen engine.

    Parameters
    ----------
    fill_value : Union[str, Array]
        The desired fill value behavior. Can be a numeric value,
        'extrapolate', or 'constant_extension'.
    interpolation_engine : InterpolationEngine
        The engine used for interpolation.

    Returns
    -------
    Optional[Union[str, Array]]
        The fill_value to pass to the underlying interpolation engine.
    """
    self._should_clip_points = False
    if fill_value is None:
      raise ValueError('`fill_value` must be provided.')

    if interpolation_engine == InterpolationEngine.SCIPY:
      if fill_value == EXTRAPOLATE:
        return None
      elif fill_value == CONSTANT_EXTENSION:
        self._should_clip_points = True
        return None
      else:
        return np.asarray(fill_value)
    elif interpolation_engine in {
        InterpolationEngine.TFP_REGULAR,
        InterpolationEngine.TFP_RECTILINEAR,
    }:
      if fill_value == EXTRAPOLATE:
        raise ValueError(
            f'fill_value={EXTRAPOLATE} not supported for engine'
            f' {interpolation_engine}.'
        )
      return fill_value
    return fill_value

  def _maybe_ln(self, key, values, logspace_keys):
    return self._lib.log(values) if key in logspace_keys else values

  def _maybe_exp(self, key, values, exp_keys):
    return self._lib.exp(values) if key in exp_keys else values

  @property
  def batch_dims(self) -> tuple[str, ...]:
    return self._batch_dims

  @property
  def dims(self) -> tuple[str, ...]:
    return self._dims

  def points(self, **dims) -> Array:
    if sorted(dims) != sorted(self.dims):
      raise ValueError(
          f'Dimension mismatch: {sorted(dims)} vs. {sorted(self.dims)}'
      )
    dtype = conform.common_dtype(
        list(dims.values()), preferred_dtype=np.float32
    )
    return self._lib.stack(
        [self._lib.asarray(dims[d], dtype=dtype) for d in self.dims], axis=-1
    )

  def eval(
      self,
      points: Array,
      variable_or_variables: Union[str, list[str], tuple[str, ...]],
  ) -> Union[Array, tuple[Array, ...]]:
    """Evaluates the interpolator at the given points.

    Parameters
    ----------
    points : Array
        The points to evaluate the interpolator at. The last dimension
        must match the number of dimensions in the interpolator.
    variable_or_variables : Union[str, list[str], tuple[str, ...]]
        The variable or variables to interpolate.

    Returns
    -------
    Union[Array, tuple[Array, ...]]
        The interpolated values. If a single variable is requested, a single array
        is returned. If a sequence of variables is requested, a tuple of arrays is
        returned.
    """
    if points.shape[-1] != len(self.dims):
      raise ValueError('`points` mismatch')

    if self._should_clip_points:
      points = self._clip_points_to_make_in_bounds(points)

    if self._logspace_dims:
      idx = [self._dims.index(d) for d in self._logspace_dims]
      if isinstance(points, np.ndarray):
        points = points.copy()
        points[..., idx] = np.log(points[..., idx])
      elif isinstance(points, jax.Array):
        points = points.at[..., idx].set(jnp.log(points[..., idx]))

    y = self._table(points)
    is_single_var = isinstance(variable_or_variables, str)
    if is_single_var:
      variable_or_variables = [variable_or_variables]
    results = tuple(
        self._maybe_exp(
            key, y[..., self._variables.index(key)], self._logspace_vars
        )
        for key in variable_or_variables
    )
    if self.batch_dims:
      n_total_dims = results[0].ndim
      p_ndim = n_total_dims - len(self.batch_dims)
      axis = tuple(range(p_ndim, n_total_dims)) + tuple(range(p_ndim))
      results = [self._lib.transpose(x, axis) for x in results]
    return results[0] if is_single_var else results


class BaseTFPGridInterpolator(abc.ABC):
  """Base class for TFP-based grid interpolators."""

  def __init__(
      self,
      points: Sequence[Array],
      values: Array,
      fill_value: Union[str, Array] = np.nan,
      jit: bool = False,
  ):
    self._dtype = conform.common_dtype(list(points), preferred_dtype=np.float32)
    self._interp_impl_fn = None
    self._fill_value = fill_value
    self._jit = jit
    self._ndim = len(points)
    self._interp_kwargs = {}

  @property
  @abc.abstractmethod
  def _tfp_interp_func(self):
    pass

  @property
  def _interp_impl(self) -> Callable[[Array], jax.Array]:
    """Returns a function that implements the interpolation."""
    if self._interp_impl_fn is None:

      def interp_impl_fn(points):
        points = jnp.asarray(points, dtype=self._dtype)
        assert self._interp_kwargs['axis'] == 0
        ndim_eq_1 = points.ndim == 1
        if ndim_eq_1:
          points = points[jnp.newaxis]
        points_nobatch = jnp.reshape(points, (-1, self._ndim))
        interpolated_values_nobatch = self._tfp_interp_func(
            x=points_nobatch, **self._interp_kwargs
        )
        interpolated_values = jnp.reshape(
            interpolated_values_nobatch,
            points.shape[:-1] + interpolated_values_nobatch.shape[1:],
        )
        if ndim_eq_1:
          interpolated_values = interpolated_values[0]
        return interpolated_values

      self._interp_impl_fn = interp_impl_fn
      if self._jit:
        self._interp_impl_fn = jax.jit(self._interp_impl_fn)
    return self._interp_impl_fn

  def __call__(self, points: Array) -> jax.Array:
    points = jnp.asarray(points)
    return self._interp_impl(points)


class TFPRectilinearGridInterpolator(BaseTFPGridInterpolator):
  """TFP-based rectilinear grid interpolator."""

  def __init__(
      self,
      points: Sequence[Array],
      values: Array,
      fill_value: Union[str, Array] = np.nan,
      jit: bool = False,
  ):
    super().__init__(points, values, fill_value=fill_value, jit=jit)
    self._interp_kwargs = dict(
        x_grid_points=tuple(np.asarray(p, dtype=self._dtype) for p in points),
        y_ref=values.astype(self._dtype),
        axis=0,
        fill_value=self._fill_value,
    )

  @property
  def _tfp_interp_func(self):
    return tfp.math.batch_interp_rectilinear_nd_grid


class TFPRegularGridInterpolator(BaseTFPGridInterpolator):
  """TFP-based regular grid interpolator."""

  def __init__(
      self,
      points: Sequence[Array],
      values: Array,
      fill_value: Union[str, Array] = np.nan,
      jit: bool = False,
  ):
    super().__init__(points, values, fill_value=fill_value, jit=jit)
    self._interp_kwargs = dict(
        x_ref_min=np.array([p[0] for p in points], dtype=self._dtype),
        x_ref_max=np.array([p[-1] for p in points], dtype=self._dtype),
        y_ref=values.astype(self._dtype),
        axis=0,
        fill_value=self._fill_value,
    )

  @property
  def _tfp_interp_func(self):
    return tfp.math.batch_interp_regular_nd_grid


def _nan_safe_wrapper(cls):
  """Returns a wrapper that handles NaNs in the interpolated values.

  The wrapper works by interpolating both the values (with NaNs set to zero)
  and a mask (1.0 for non-NaN, 0.0 for NaN). The result is the ratio of
  the two interpolated values.

  Parameters
  ----------
  cls : type
      The interpolator class to wrap.

  Returns
  -------
  type
      A class that wraps `cls` to be NaN-safe.
  """

  class NaNSafeWrapper:
    """A wrapper that handles NaNs in interpolated values.

    This wrapper normalizes by a mask of non-NaN values.
    """

    def __init__(
        self, points: Sequence[Array], values: Array, **interpolator_kwargs
    ):
      is_nan = np.isnan(values)
      self._interpolator_with_nan_set_to_zero = cls(
          points, values=np.where(is_nan, 0.0, values), **interpolator_kwargs
      )
      self._non_nan_frac = cls(
          points, values=np.where(is_nan, 0.0, 1.0), **interpolator_kwargs
      )

    def __call__(self, points: Array) -> Array:
      return self._interpolator_with_nan_set_to_zero(
          points
      ) / self._non_nan_frac(points)

  return NaNSafeWrapper
