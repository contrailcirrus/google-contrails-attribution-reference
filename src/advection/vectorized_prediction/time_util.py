"""Functions and classes for time initialization and manipulation."""

import abc

import numpy as np
from scipy import interpolate as sp_interpolate

from . import constants as cs

D = cs.Dims


class TimeChange(abc.ABC):
  """Base class for representing change time scales to solve IVP."""

  @abc.abstractmethod
  def t(self, s: np.ndarray) -> np.ndarray:
    """Time for all parcels, used to index an interpolation table."""

  @abc.abstractmethod
  def dt_ds(self, s: np.ndarray) -> np.ndarray:
    """Derivative for all parcels, used to rescale the ODE."""

  @property
  @abc.abstractmethod
  def s_span(self) -> tuple[np.ndarray, np.ndarray]:
    """The (min, max) integration span passed as `t_span` to solve_ivp."""

  @property
  @abc.abstractmethod
  def s_eval(self) -> np.ndarray:
    """The 1D array of evaluation points passed as `t_eval` to solve_ivp."""

  @property
  @abc.abstractmethod
  def ds_dt_max(self) -> float:
    """The maximal derivative ds/dt over all parcels."""

  @property
  @abc.abstractmethod
  def ds_dt_min(self) -> float:
    """The minimal derivative ds/dt over all parcels."""


class IdentityTimeChange(TimeChange):
  """Change time scales for solve_ivp via the identity transformation."""

  def __init__(self, t_eval: np.ndarray):
    """Initialize an IdentityTimeChange."""
    t_eval = np.asarray(t_eval)
    if t_eval.ndim != 1:
      raise ValueError(f'`t_eval.ndims`={t_eval.ndim} is not supported.')

    self._s_span = (t_eval[0], t_eval[-1])
    self._sorted_s_span = sorted(self._s_span)
    self._s_eval = t_eval

  def t(self, s: np.ndarray) -> np.ndarray:
    """Scalar time for all parcels."""
    s = np.clip(s, *self._sorted_s_span)
    return np.asarray(s)

  def dt_ds(self, s: np.ndarray) -> np.ndarray:
    """Scalar derivative for all parcels."""
    s = np.asarray(s)
    return np.array(1, dtype=s.dtype)

  @property
  def s_span(self) -> tuple[np.ndarray, np.ndarray]:
    """The (first, last) integration span passed as `t_span` to solve_ivp."""
    return self._s_span

  @property
  def s_eval(self) -> np.ndarray:
    """The 1D array of evaluation points passed as `t_eval` to solve_ivp."""
    return self._s_eval

  @property
  def ds_dt_max(self) -> float:
    """The maximal derivative ds/dt for all parcels."""
    return 1.0

  @property
  def ds_dt_min(self) -> float:
    """The minimal derivative ds/dt for all parcels."""
    return 1.0


class LinearTimeChange(TimeChange):
  """Change time scales for solve_ivp by linear rescaling between eval times."""

  def __init__(self, t_eval: np.ndarray):
    """Initialize a LinearTimeChange."""
    t_eval = np.asarray(t_eval)
    if t_eval.ndim != 2:
      raise ValueError(f'`t_eval.ndims`={t_eval.ndim} is not supported.')

    self._parity = get_time_parity(t_eval, strict=False)
    first = np.min if self._parity > 0 else np.max

    self._s_eval = first(t_eval, axis=-1)
    self._s_span = (self.s_eval[0], self.s_eval[-1])
    self._sorted_s_span = sorted(self._s_span)

    self._t = sp_interpolate.interp1d(self.s_eval, t_eval, axis=0)

    delta_s = np.diff(self.s_eval, axis=0)
    delta_t = np.diff(t_eval, axis=0)
    self._delta_t_div_delta_s = delta_t / delta_s[:, np.newaxis]
    self._delta_t_div_delta_s_min = np.min(self._delta_t_div_delta_s)
    self._delta_t_div_delta_s_max = np.max(self._delta_t_div_delta_s)
    self._n_steps = t_eval.shape[0]

  def t(self, s: np.typing.ArrayLike) -> np.ndarray:
    """Shape [n_parcels] array giving time for each parcel."""
    s = np.clip(s, *self._sorted_s_span)
    return self._t(s)

  def dt_ds(self, s: np.typing.ArrayLike) -> np.ndarray:
    """Shape [n_parcels] array giving dt/ds for each parcel."""
    raw_idx = (
        np.searchsorted(
            self._parity * self.s_eval, self._parity * s, side='left'
        )
        - 1
    )
    idx = np.clip(raw_idx, 0, self._n_steps - 1)
    return self._delta_t_div_delta_s[idx]

  @property
  def s_span(self) -> tuple[np.ndarray, np.ndarray]:
    """The (first, last) integration span passed as `t_span` to solve_ivp."""
    return self._s_span

  @property
  def s_eval(self) -> np.ndarray:
    """The 1D array of evaluation points passed as `t_eval` to solve_ivp."""
    return self._s_eval

  @property
  def ds_dt_max(self) -> float:
    """The maximal derivative ds/dt over all parcels."""
    return float(1 / self._delta_t_div_delta_s_min)

  @property
  def ds_dt_min(self) -> float:
    """The minimal derivative ds/dt over all parcels."""
    return float(1 / self._delta_t_div_delta_s_max)


def get_time_parity(times: np.typing.ArrayLike, strict: bool) -> int:
  """+1/-1 if times are [strictly] increasing/decreasing on axis=0."""
  times = np.array(times)
  t_diffs = np.diff(times, axis=0)

  cmp = np.less if strict else np.less_equal

  if np.all(cmp(0, t_diffs)):
    return +1
  elif np.all(cmp(t_diffs, 0)):
    return -1
  else:
    raise ValueError('`times` must be increasing or decreasing.')


def t_eval_homogeneous(
    t_start: np.typing.ArrayLike,
    advection_length_hours: float,
    step_size_secs: float,
) -> np.ndarray:
  """Integration evaluation times that are the same for every starting time."""
  if step_size_secs <= 0:
    raise ValueError(f'{step_size_secs=} was <= 0 but should be positive.')
  parity = np.sign(advection_length_hours)
  t_start = np.asarray(t_start)
  if t_start.ndim > 1:
    raise ValueError(f't_start.ndim={t_start.ndim} not yet supported.')

  vector_t_start = t_start.ndim == 1
  if not vector_t_start:
    t_start = t_start.ravel()

  step_size = step_size_secs / cs.TIME_UNIT.seconds_per_unit
  advection_length_secs = advection_length_hours * 3600
  advection_length = advection_length_secs / cs.TIME_UNIT.seconds_per_unit

  delta = np.arange(0, np.abs(advection_length), step_size)
  t_final = t_start + advection_length
  t_eval = t_start + delta[:, np.newaxis] * parity

  if np.any(t_eval[-1] * parity < t_final * parity):
    t_eval = np.concatenate((t_eval, [t_final]))

  return t_eval if vector_t_start else np.squeeze(t_eval, axis=-1)


def t_eval_stagger_start(
    t_lower: float,
    t_start: np.ndarray,
    advection_length_hours: float,
    step_size_secs: float,
) -> np.ndarray:
  """Evaluation times with staggered start times.

  These evaluation times start different and are evaluated at the same points.

  Parameters
  ----------
  t_lower : float
      Evaluation times lower bound.
  t_start : np.ndarray
      Evaluated times at the beginning of the advection.
  advection_length_hours : float
      Advection length in hours.
  step_size_secs : float
      The required step size for the evaluation in seconds.

  Returns
  -------
  np.ndarray
      A (n_steps, n_parcels) array of evaluation times.
  """
  t_start = np.asarray(t_start)
  if t_start.ndim != 1:
    raise ValueError(f'`t_start.ndim`={t_start.ndim} is not supported.')

  if t_lower >= np.min(t_start):
    raise ValueError(f'`t_lower`={t_lower} >= {np.min(t_start)} but should not')

  t_eval_baseline = t_eval_homogeneous(
      t_start=t_lower,
      advection_length_hours=advection_length_hours,
      step_size_secs=step_size_secs,
  )
  n_parcels = t_start.shape[0]
  n_steps = len(t_eval_baseline)

  t_eval = np.tile(t_eval_baseline, n_parcels).reshape(n_parcels, n_steps).T
  t_eval[0, :] = t_start

  return t_eval
