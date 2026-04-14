"""Vectorized advection for contrails."""

from collections.abc import Mapping
from typing import Any, Optional, TypeVar, Union

import jax.numpy as jnp
import numpy as np
from pycontrails.core import flight
from pycontrails.physics import units
from scipy import integrate as sp_integrate
import tree_math
import xarray as xr

from ..utils import casting
from ..utils import conform
from ..utils import interpolation
from . import constants as cs
from . import moist_advect
from . import physics
from . import time_util

pl_to_m = units.pl_to_m
Flight = flight.Flight
D = cs.Dims

# Monkey patch in a max-norm for RK to use. This means step size is chosen based
# on the component with maximum difference between higher/lower order method.
# The default was the mean. In practice, there is very little difference.
# The use of nanmax prevents NaN states from influencing others.
sp_integrate._ivp.rk.norm = lambda x: np.nanmax(np.abs(x))  # pylint: disable=protected-access

# For the .from_* classmethods
_AdvectionStateType = TypeVar('_AdvectionStateType', bound='AdvectionState')

_MIN_SANITIZED_PRESSURE = 1e-3

# Typings
MaxFrac = Union[float, Mapping[str, float]]


@tree_math.struct
class AdvectionState:
  """State of a set of parcels being advected in 3D space and time.

  This class uses Gaussian coordinates (X, Y, Z on a sphere) for spatial
  representation, which is more accurate than latitude/longitude and avoids
  singularities at the Earth's poles.

  Attributes
  ----------
  time : Any
      Time in hours since a reference epoch (e.g., 1970-01-01). Can be a
      scalar or a 1D/2D array.
  pressure : Any
      Air pressure in hPa (hectopascals), used as the vertical coordinate.
  gaussian : physics.Gaussian
      A physics.Gaussian object containing the X, Y, Z coordinates.
  """

  time: Any
  pressure: Any
  gaussian: physics.Gaussian

  def __post_init__(self):
    """Validate shapes and types of state variables."""
    time, pressure = self.time, self.pressure
    x, y, z = self.gaussian.x, self.gaussian.y, self.gaussian.z

    # Ensure all components use the same floating-point type (e.g., float64)
    conform.common_dtype([time, x, y, z, pressure])

    # Ensure spatial components and pressure all have the same shape
    conform.static_shape(y, expect_shape=x.shape)
    conform.static_shape(z, expect_shape=x.shape)
    conform.static_shape(pressure, expect_shape=x.shape)

    # Validate that time dimensions make sense compared to spatial dimensions
    if np.ndim(time) not in {0, 1, 2}:
      raise ValueError(
          f'time should have 0, 1, or 2 dims. Found {np.ndim(time)}'
      )
    if np.ndim(x) not in {1, 2}:
      raise ValueError(
          f'x coordinate should have 1 or 2 dims. Found {np.ndim(x)}'
      )

  @property
  def latitude(self) -> Any:
    """The latitude of the parcels in degrees."""
    return physics.gaussian_to_latlon(self.gaussian).latitude

  @property
  def longitude(self) -> Any:
    """The longitude of the parcels in degrees."""
    return physics.gaussian_to_latlon(self.gaussian).longitude

  @property
  def n_steps(self) -> Optional[int]:
    """The number of time steps in this state (if 2D), or None (if 1D)."""
    return self.gaussian.x.shape[0] if self.gaussian.x.ndim == 2 else None

  @property
  def n_parcels(self) -> int:
    """The number of individual parcels tracked in this state."""
    return self.gaussian.x.shape[-1]

  def isfinite(self) -> xr.DataArray:
    """Returns a boolean mask indicating which parcel states are finite."""
    isfinite_in_space = np.all(np.isfinite(self.pressure_xyz()), axis=-1)
    isfinite_in_time = np.isfinite(self.time)

    if self.n_steps and self.n_parcels and np.ndim(self.time) < 2:
      # Broadcast time if it's not already per-parcel/step
      isfinite_in_time = isfinite_in_time[:, np.newaxis]

    dims = (D.STEP, D.PARCEL) if self.n_steps else (D.PARCEL,)
    return xr.DataArray(data=isfinite_in_time & isfinite_in_space, dims=dims)

  def pressure_xyz(self) -> jnp.ndarray:
    """Combines pressure and Gaussian X, Y, Z into a single [..., 4] array."""
    return jnp.stack(
        [self.pressure, self.gaussian.x, self.gaussian.y, self.gaussian.z],
        axis=-1,
    )

  @classmethod
  def from_time_and_pressure_xyz(
      cls: type[_AdvectionStateType],
      time: Any,
      pressure_xyz: Any,
  ) -> _AdvectionStateType:
    """Creates an AdvectionState from time and a combined pressure/XYZ array."""
    gaussian = physics.Gaussian(
        x=pressure_xyz[..., 1],
        y=pressure_xyz[..., 2],
        z=pressure_xyz[..., 3],
    )
    # Ensure parcels stay on the Earth's surface
    gaussian = physics.project_gaussian_to_earths_surface(gaussian)
    return cls(time=time, pressure=pressure_xyz[..., 0], gaussian=gaussian)

  @classmethod
  def from_time_pressure_lat_lon(
      cls: type[_AdvectionStateType],
      time: Any,
      pressure: Any,
      latitude: Any,
      longitude: Any,
  ) -> _AdvectionStateType:
    """Initializes a state from standard lat/lon coordinates."""
    gaussian = physics.latlon_to_gaussian(
        physics.LatLon(latitude=latitude, longitude=longitude)
    )
    return cls(time=time, pressure=pressure, gaussian=gaussian)

  def points(self) -> jnp.ndarray:
    """Returns points in (time, pressure, lat, lon) format for interpolation."""
    if self.n_steps:
      raise AssertionError('points() should only be called on a single step.')

    latlon = physics.gaussian_to_latlon(self.gaussian)
    # Broadcast time to match the number of parcels
    time = jnp.broadcast_to(self.time, latlon.latitude.shape)
    return jnp.stack(
        [time, self.pressure, latlon.latitude, latlon.longitude], axis=-1
    )

  def as_sanitized_state(self) -> _AdvectionStateType:
    """Prevents numerical issues by ensuring pressure is slightly positive."""
    return AdvectionState(
        time=self.time,
        pressure=jnp.maximum(self.pressure, _MIN_SANITIZED_PRESSURE),
        gaussian=self.gaussian,
    )

  @classmethod
  def from_flight(
      cls: type[_AdvectionStateType], input_flight: Flight
  ) -> _AdvectionStateType:
    """Creates an AdvectionState from a pycontrails Flight object."""
    # Flight.level returns pressure in hPa, calculating it from either
    # altitude or altitude_ft.
    pressure = input_flight.level
    time = casting.to_float(
        input_flight.data['time'],
        reference_time=cs.REFERENCE_TIME,
        unit=cs.TIME_UNIT,
    )
    return cls.from_time_pressure_lat_lon(
        time=time,
        pressure=pressure,
        latitude=input_flight.data['latitude'],
        longitude=input_flight.data['longitude'],
    )

  def to_flight(self, source_flight: Optional[Flight] = None) -> Flight:
    """Converts the AdvectionState back to a pycontrails Flight object."""
    time_dt = casting.float_to_datetime64(
        self.time, reference_time=cs.REFERENCE_TIME, unit=cs.TIME_UNIT
    )

    latlon = physics.gaussian_to_latlon(self.gaussian)

    # Handle both 1D (single step) and 2D (multiple steps) states
    if self.n_steps is not None:
      # Axis 0 is steps, axis 1 is parcels.
      # ravel() is step-major: [s0p0, s0p1, ..., s0pN, s1p0, s1p1, ...]
      time_arr = time_dt
      if time_arr.ndim == 1:
        time_arr = np.repeat(time_dt[:, np.newaxis], self.n_parcels, axis=1)

      data = {
          'time': time_arr.ravel(),
          'latitude': latlon.latitude.ravel(),
          'longitude': latlon.longitude.ravel(),
          'level': self.pressure.ravel(),
          'altitude': pl_to_m(self.pressure.ravel()),
          'parcel_id': np.tile(np.arange(self.n_parcels), self.n_steps),
      }

      if source_flight is not None:
        # Replicate original columns for each timestep
        for col in source_flight.data:
          if col not in data and col != 'altitude':
            # Use tile to repeat [p0, p1, ..., pN] for each step
            data[col] = np.tile(source_flight.data[col], self.n_steps)
        attrs = source_flight.attrs.copy()
      else:
        attrs = {}
    else:
      data = {
          'time': time_dt,
          'latitude': latlon.latitude,
          'longitude': latlon.longitude,
          'level': self.pressure,
          'altitude': pl_to_m(self.pressure),
          'parcel_id': np.arange(self.n_parcels),
      }
      if source_flight is not None:
        for col in source_flight.data:
          if col not in data and col != 'altitude':
            data[col] = source_flight.data[col]
        attrs = source_flight.attrs.copy()
      else:
        attrs = {}

    return Flight(data=data, attrs=attrs, drop_duplicated_times=True)


class LagrangianAdvector:
  """Advects parcels through a 3D wind field using a Lagrangian approach.

  This advector uses numerical integration (solve_ivp) to track how parcels
  move over time. It handles coordinate transformations and fetches the
  necessary weather data slices automatically.
  """

  def __init__(
      self,
      weather: xr.Dataset,
      max_frac_oob: MaxFrac = 0.01,
      max_frac_nan_weather: MaxFrac = 0.01,
      sanitize_state_for_interpolation: bool = True,
      dtype: Any = np.float64,
  ):
    """Initializes the advector with weather data and configuration.

    Parameters
    ----------
    weather : xr.Dataset
        The xarray Dataset containing wind components 'u', 'v', 'w'.
    max_frac_oob : MaxFrac, optional
        Max fraction of parcels allowed to go out of
        spatial/temporal bounds before failing. Defaults to 0.01.
    max_frac_nan_weather : MaxFrac, optional
        Max fraction of NaN values allowed in weather data. Defaults to 0.01.
    sanitize_state_for_interpolation : bool, optional
        If True, ensures pressure remains valid. Defaults to True.
    dtype : Any, optional
        Numerical precision to use for calculations. Defaults to np.float64.
    """
    self._weather = weather.assign_coords(
        time=casting.to_float(
            weather.time, reference_time=cs.REFERENCE_TIME, unit=cs.TIME_UNIT
        )
    )
    self._max_frac_oob = max_frac_oob
    self._max_frac_nan_weather = max_frac_nan_weather
    self._sanitize_state_for_interpolation = sanitize_state_for_interpolation
    self._dtype = dtype

  def advect(
      self,
      source: Union[AdvectionState, Flight],
      t_eval: Any,
      use_moist_advection: bool = False,
      rtol: float = 1e-3,
      atol: float = 1e-6,
  ) -> Union[AdvectionState, Flight]:
    """Core advection method.

    Parameters
    ----------
    source : Union[AdvectionState, Flight]
        The starting positions and times. Can be an AdvectionState or a
        pycontrails Flight.
    t_eval : Any
        The target times to calculate positions for. Can be float hours
        from epoch or datetime-like objects.
    use_moist_advection : bool, optional
        If True, enables `moist advection`, which is a form
        of parcel advection in which the parcel's vertical velocity is corrected
        by a terminal velocity term. This model assumes the mean ice crystal
        size according to the CoCIP-based growth model. Defaults to False.
    rtol : float, optional
        Relative tolerance for the numerical solver. Defaults to 1e-3.
    atol : float, optional
        Absolute tolerance for the numerical solver. Defaults to 1e-6.

    Returns
    -------
    Union[AdvectionState, Flight]
        An AdvectionState (or Flight if input was Flight) containing positions at
        all target times.
    """
    is_flight = isinstance(source, Flight)
    if is_flight:
      initial_state = AdvectionState.from_flight(source)
    else:
      initial_state = source

    t_eval = casting.to_float(
        t_eval, reference_time=cs.REFERENCE_TIME, unit=cs.TIME_UNIT
    )

    # 1. Prepare weather data for the requested region and time
    # (In a real scenario, this would involve materialization. For now, we
    # assume the dataset provided to __init__ is sufficient or we use
    # interpolators).

    # For simplicity in this OSS version, we define three interpolators for
    # U, V, W wind.
    # Note: Canonical order is (time, pressure, latitude, longitude)
    weather_variables = (
        cs.ADVECTION_WEATHER_VARS_WITH_TEMPERATURE
        if use_moist_advection
        else cs.ADVECTION_WEATHER_VARS
    )
    uvw_interp = interpolation.DatasetInterpolator(
        self._weather,
        variables=weather_variables,
        fill_value=interpolation.CONSTANT_EXTENSION,
    )

    # Ensure start time is included in evaluation points for 1D grids.
    # For 2D grids (staggered starts), we assume evaluations are already
    # correctly structured.
    if t_eval.ndim == 1:
      t_eval = np.unique(np.concatenate([initial_state.time, t_eval]))
    result = self._advection_loop(
        initial_state,
        uvw_interp,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        use_moist_advection=use_moist_advection,
    )
    if is_flight:
      return result.to_flight(source_flight=source)
    return result

  def _advection_loop(
      self,
      initial_state: AdvectionState,
      uvw_interp: interpolation.DatasetInterpolator,
      t_eval: np.ndarray,
      rtol: float,
      atol: float,
      use_moist_advection: bool = False,
  ) -> AdvectionState:
    """Internal loop that solves the Ordinary Differential Equation (ODE)."""

    # We use a LinearTimeChange to handle mapping integration steps to real
    # time.
    if t_eval.ndim == 1:
      time_change = time_util.IdentityTimeChange(t_eval)
    else:
      time_change = time_util.LinearTimeChange(t_eval)

    def dy_ds_fun(s, y_flat):
      """Calculates the rate of change for (Pressure, X, Y, Z) at step s."""
      # Reshape the flat solver state back into [n_parcels, 4] (P, X, Y, Z)
      state = AdvectionState.from_time_and_pressure_xyz(
          time=np.asarray(time_change.t(s), dtype=y_flat.dtype),
          pressure_xyz=np.reshape(y_flat, (-1, 4)),
      )

      if self._sanitize_state_for_interpolation:
        state = state.as_sanitized_state()

      # Fetch wind speeds at the current parcel positions
      # u = Eastward (m/s), v = Northward (m/s), w = Vertical (Pa/s)
      if not use_moist_advection:
        u, v, w = uvw_interp.eval(state.points(), ['u', 'v', 'w'])
        hpa_per_second = w / 100.0
      else:
        u, v, w, t = uvw_interp.eval(state.points(), ['u', 'v', 'w', 't'])
        pa_per_sec = w + moist_advect.dp_correction(
            pressure_pa=state.pressure * 100.0,
            temperature=t,
            t=state.time,
            t0=initial_state.time,
        )
        hpa_per_second = pa_per_sec / 100.0

      # Convert lat/lon wind speeds into changes in Gaussian X, Y, Z
      xyz_per_second = physics.orthographic_to_gaussian_delta(
          state.gaussian, physics.Orthographic(u=u, v=v)
      )

      # Scaling: Convert rates from "per-second" to "per-TIME_UNIT" (usually
      # hours)
      seconds_per_unit = cs.TIME_UNIT.seconds_per_unit
      dgaussian_ds = xyz_per_second * seconds_per_unit
      dpressure_ds = hpa_per_second * seconds_per_unit

      # Combine and rescale based on how 's' relates back to real 't'
      dpressure_xyz_dt = jnp.stack(
          [dpressure_ds, dgaussian_ds.x, dgaussian_ds.y, dgaussian_ds.z],
          axis=-1,
      )

      # Rescale derivatives by dt/ds (chain rule)
      dt_ds = np.reshape(time_change.dt_ds(s), (-1, 1))
      res = (dpressure_xyz_dt * dt_ds).ravel()
      return res

    # Initial integration time step estimate
    first_step = min(
        time_change.ds_dt_min * (300 / cs.TIME_UNIT.seconds_per_unit),
        abs(time_change.s_eval[1] - time_change.s_eval[0]),
    )

    # Solve the system of differential equations
    solver_outputs = sp_integrate.solve_ivp(
        dy_ds_fun,
        t_span=time_change.s_span,
        y0=initial_state.pressure_xyz().ravel(),
        method='RK23',
        t_eval=time_change.s_eval,
        first_step=first_step,
        rtol=rtol,
        atol=atol,
    )

    # Check solver status
    if solver_outputs.status < 0:
      raise RuntimeError(f'ODE solver failed: {solver_outputs.message}')

    # Reshape results from [steps, n_parcels * 4] to [steps, n_parcels, 4]
    y_results = solver_outputs.y.T
    pressure_xyz_results = np.reshape(
        y_results, (len(time_change.s_eval), initial_state.n_parcels, 4)
    )

    advected_states = AdvectionState.from_time_and_pressure_xyz(
        t_eval, pressure_xyz_results
    )

    # Project coordinates back to the Earth's surface for consistency
    return AdvectionState(
        time=advected_states.time,
        pressure=advected_states.pressure,
        gaussian=physics.project_gaussian_to_earths_surface(
            advected_states.gaussian
        ),
    )

  def __call__(
      self,
      source: Union[AdvectionState, Flight],
      advection_length_hours: float,
      step_size_secs: float = 600,
  ) -> Union[AdvectionState, Flight]:
    """Alternative call syntax for simple constant-duration advection."""
    is_flight = isinstance(source, Flight)
    if is_flight:
      initial_state = AdvectionState.from_flight(source)
    else:
      initial_state = source

    t_eval = time_util.t_eval_homogeneous(
        t_start=initial_state.time,
        advection_length_hours=advection_length_hours,
        step_size_secs=step_size_secs,
    )
    # Strip the first step as t_eval_homogeneous includes it but we want the
    # target positions usually excluding the very first start position if using
    # this syntax. Actually, solve_ivp with t_eval returns all.
    return self.advect(source, t_eval)
