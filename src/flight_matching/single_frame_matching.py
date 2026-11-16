"""Flight matching in JAX."""

import collections
from collections.abc import Sequence
import dataclasses
import functools
import typing
from typing import NamedTuple
import warnings

import geojson
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np
import optax
import pandas as pd
from pycontrails.core import flight
import xarray as xr

from src.advection.vectorized_prediction import advection as oss_advection
from src.flight_matching.utils import projection

Flight = flight.Flight
MIN_CONTRAIL_LENGTH_KM = 15
MIN_CONTRAIL_AGE_HRS = 5 / 60
MIN_OVERLAP_SECONDS = 60
# These default values are derived from analysis conducted by
# Google Research as part of
# https://iopscience.iop.org/article/10.1088/2515-7620/ad11ab.
C_FIT = 2.0
C_SHIFT = 20.0
C_ANGLE = 10.0
C_AGE = 1.0
# Flight-contrail pairs with a match score less than this are
# considered matches.
MATCH_SCORE_THRESHOLD = 3.0
# When comparing flight-contrail pairs, a pair with a match score that is lower
# than another pair's by at least this amount is considered a much better match.
# Otherwise, both pairs are generally considered equally viable matches.
BETTER_SCORE_MARGIN = 1.0
# Fixed batch size to maintain static array shapes for Jax cache hits.
JAX_BATCH_SIZE = 8

SATELLITES_TO_LONGITUDES_MAP = collections.defaultdict(
    float,
    {
        'GOES-EAST-FULL-DISK': -75.0,
        'GOES-WEST-FULL-DISK': -137.0,
        'MTG-000-FULL-DISK': 0.0,
        'HIMAWARI-FULL-DISK': 140.7,
    },
)
# Pad flight out to a fixed parcel length with a mask so that we hit the JAX
# JIT cache as often as possible. Assuming 2hr advection and 1min flight
# sampling, we'd expect a maximum of ~120 parcels for a single flight at a
# single timestamp, so we'll pad it out to 150 to be safe. Note that this will
# need to be updated if the advection time or sampling rate changes.
JIT_PADDED_LENGTH = 150


@dataclasses.dataclass(frozen=True)
class ScoreComponents:
  """Components of the flight match score."""

  fit_score: jax.typing.ArrayLike
  shift_score: jax.typing.ArrayLike
  angle_score: jax.typing.ArrayLike
  age_score: jax.typing.ArrayLike

  v_offset: jax.typing.ArrayLike
  w_offset: jax.typing.ArrayLike
  theta: jax.typing.ArrayLike
  age_hrs: jax.typing.ArrayLike

  def total(self) -> jax.typing.ArrayLike:
    return self.fit_score + self.shift_score + self.angle_score + self.age_score

  def with_scalar_values(self) -> 'ScoreComponents':
    kwargs = dataclasses.asdict(self)
    for k, v in kwargs.items():
      if isinstance(v, (jax.Array, np.number, np.ndarray)):
        kwargs[k] = v.item()
    return ScoreComponents(**kwargs)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearContrail:
  """Represents a linear contrail.

  Attributes:
    lon1: Longitude of the western endpoint, in degrees.
    lat1: Latitude of the western endpoint, in degrees.
    lon2: Longitude of the eastern endpoint, in degrees.
    lat2: Latitude of the eastern endpoint, in degrees.
    timestamp: Detection time as a Unix timestamp in seconds.
    altitude_m: Contrail altitude in meters, or None if not supplied.
    satellite_longitude: Longitude of the satellite that detected the contrail,
      in degrees.
  """

  lon1: float
  lat1: float
  lon2: float
  lat2: float
  timestamp: int
  altitude_m: float | None
  satellite_longitude: float

  def __post_init__(self):
    # Force the convention that lon1 is west of lon2.
    if not dateline_safe_is_west_check(self.lon1, self.lon2):
      lon1, lat1 = self.lon1, self.lat1
      object.__setattr__(self, 'lon1', self.lon2)
      object.__setattr__(self, 'lat1', self.lat2)
      object.__setattr__(self, 'lon2', lon1)
      object.__setattr__(self, 'lat2', lat1)


@dataclasses.dataclass(frozen=True)
class ContrailMatchScore:
  """The result of matching a contrail to a flight.

  Attributes:
    contrail: The linear contrail that was matched.
    score: The total match score (lower values indicate a better match).
    flight_id: Identifier of the matched flight.
    overlap_start_timestamp: Earliest original flight waypoint timestamp used
      for calculation of the flight-contrail match score.
    overlap_end_timestamp: Latest original flight waypoint timestamp used for
      calculation of the flight-contrail match score.
    score_components: Individual loss terms and optimized parameters that make
      up the total match score.
  """

  contrail: LinearContrail
  score: float
  flight_id: str
  overlap_start_timestamp: int
  overlap_end_timestamp: int
  score_components: ScoreComponents


class Segment(NamedTuple):
  """The geometry of a contrail, or of a batch of contrails.

  For a single contrail `length_km` is a scalar and `v_hat` / `w_hat` have
  shape (3,). `_compute_segments` produces a batch with a leading contrail
  axis: shapes (num_contrails,) and (num_contrails, 3) respectively.

  Attributes:
    length_km: The length of the contrail, in kilometers.
    v_hat: The 3D unit vector (units are KM, length per unit vector = 1km)
      pointing along the contrail, with the origin at the contrail's center.
    w_hat: The 3D unit vector (units are KM, length per unit vector = 1km) that
      is normal to the contrail and to the line from the center of the Earth to
      the contrail's center, with the origin at the contrail's center.
  """

  length_km: np.ndarray
  v_hat: np.ndarray
  w_hat: np.ndarray


class WeightOptions(NamedTuple):
  """The age-dependent weights of the different loss terms."""

  # Penalty for residual linearity error. Advected non-linear flight waypoints
  # are unlikely to have created a linear contrail.
  c_fit: float = C_FIT
  # Expected rate of lateral advection error increase.
  c_shift: float = C_SHIFT
  # Expected rate of rotational advection error increase.
  c_angle: float = C_ANGLE
  # Penalty applied to the match score based on the advected duration
  # of a flight (longer flights incur a greater penalty).
  c_age: float = C_AGE
  # If populated, specifies the maximum absolute vertical difference between
  # that altitude and the advected flight waypoint altitudes in order for this
  # to be a candidate match.
  altitude_difference_threshold_m: float | None = None


class Weights:
  """The weights of the different loss terms, given the flight age."""

  def __init__(
      self,
      age_hours: jax.typing.ArrayLike,
      weight_options: WeightOptions,
  ):
    if isinstance(age_hours, (jax.Array, np.ndarray)) and age_hours.shape:
      raise ValueError('age_hours must be a scalar')

    np_impl = jnp if isinstance(age_hours, jax.Array) else np

    wind_offset_error_km = age_hours * weight_options.c_shift
    wind_angle_error_radians = np_impl.deg2rad(
        age_hours * weight_options.c_angle
    )
    # Assuming error = 2*sigma, then variance = sigma^2 = 0.25 * error^2
    measurement_variance_km = 0.25 * weight_options.c_fit * weight_options.c_fit
    wind_offset_variance_km = 0.25 * wind_offset_error_km * wind_offset_error_km
    wind_angle_variance_radians = 1 - np_impl.cos(wind_angle_error_radians / 2)

    self.match_weight = 1 / (measurement_variance_km + 1e-6)
    self.shift_weight = 1 / (wind_offset_variance_km + 1e-6)
    self.rotation_weight = 1 / (wind_angle_variance_radians + 1e-6)
    self.age_weight = age_hours * weight_options.c_age


class ProcessedMatchingInputs(NamedTuple):
  """Pre-processed inputs for matching a candidate contrail to a flight.

  Attributes:
    contrail: The candidate linear contrail being matched.
    vs: Along-contrail coordinates of each flight waypoint in the contrail's (v,
      w) frame, in kilometers, with shape (num_waypoints,).
    ws: Normal-to-contrail coordinates of each flight waypoint in the contrail's
      (v, w) frame, in kilometers, with shape (num_waypoints,).
    keep_mask: Boolean mask of shape (num_waypoints,) indicating which flight
      waypoints are close enough along the contrail to participate in the fit.
    overlap_start_timestamp: Earliest original flight waypoint timestamp (as a
      Unix timestamp in seconds) that falls directly within the contrail's
      V-axis span.
    overlap_end_timestamp: Latest original flight waypoint timestamp (as a Unix
      timestamp in seconds) that falls directly within the contrail's V-axis
      span.
  """

  contrail: LinearContrail
  vs: np.ndarray
  ws: np.ndarray
  keep_mask: np.ndarray
  overlap_start_timestamp: int
  overlap_end_timestamp: int


class PaddedJaxData(NamedTuple):
  vs: jax.Array
  ws: jax.Array
  age_hrs: jax.Array
  mask: jax.Array


def _pad_to_length_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
  # This is about 20x faster than np.pad
  assert len(arr.shape) == 1
  # Using np.empty is even faster, but you have to be very careful about what
  # happens with the uninitialized values, so leaving it as full for now.
  out = np.full(target_len, 0, dtype=arr.dtype)
  out[0 : len(arr)] = arr
  return out


def pad_parcels_and_jaxify(
    vs: np.ndarray,
    ws: np.ndarray,
    age_hrs: np.ndarray,
    mask: np.ndarray,
    padded_length: int,
) -> PaddedJaxData:
  """Pad parcel arrays to a fixed length and convert to JAX arrays.

  In order to hit the JAX JIT cache as often as possible, we want the inputs
  to the optimization to always be the same shape. The problem is that flights
  have different numbers of parcels. To resolve this, we pad them all out to a
  fixed number of parcels that should be an upper bound on the number a single
  flight would ever have, and also supply a mask of the same shape so that we
  know which values are real vs padded. The JIT-ed scoring function then
  ignores the padded data.

  If we ever encounter a flight that is longer than our padded length, we do
  not truncate it. We keep it as-is and allow it to miss the cache. This
  would likely manifest as the matching code becoming very slow, if many
  flights were missing the cache.

  Args:
    vs: The v parameters of the flight parcels.
    ws: The w parameters of the flight parcels.
    age_hrs: The ages of the flight parcels.
    mask: A mask of the same shape as vs, ws, and age_hrs, where the value is 1
      where the values in the other arrays are valid and 0 where there are
      padded values.
    padded_length: The length to pad the arrays to.

  Returns:
    The padded and jaxified arrays.
  """

  num_parcels = len(vs)
  if num_parcels > padded_length:
    warnings.warn(
        f'{num_parcels} parcels for a flight at a single timestamp.'
        ' This will result in breaking the JAX JIT cache.',
    )
  else:
    vs = _pad_to_length_1d(vs, padded_length)
    ws = _pad_to_length_1d(ws, padded_length)
    age_hrs = _pad_to_length_1d(age_hrs, padded_length)
    mask = _pad_to_length_1d(mask, padded_length)
  return PaddedJaxData(
      jnp.asarray(vs),
      jnp.asarray(ws),
      jnp.asarray(age_hrs),
      jnp.asarray(mask),
  )


def _pad_matching_inputs(
    match_data: ProcessedMatchingInputs,
    flight_age_hrs: np.ndarray,
    optimization_age_hrs: float | None = None,
) -> PaddedJaxData:
  """Pads a contrail's matching inputs for JAX optimization."""
  age_hrs = (
      np.full_like(match_data.vs, optimization_age_hrs)
      if optimization_age_hrs is not None
      else flight_age_hrs
  )
  return pad_parcels_and_jaxify(
      vs=match_data.vs,
      ws=match_data.ws,
      age_hrs=age_hrs,
      mask=match_data.keep_mask,
      padded_length=JIT_PADDED_LENGTH,
  )


def dateline_safe_west_east(lon1: float, lon2: float) -> tuple[float, float]:
  """Assign two longitudes east/west to minimize the distance between them.

  Args:
    lon1: Longitude, in degrees.
    lon2: Longitude, in degrees.

  Returns:
    lon1, lon2, ordered such that calling the first one 'west' minimizes the
    distance between them.
  """
  min_lon = np.minimum(lon1, lon2)
  max_lon = np.maximum(lon1, lon2)
  if (max_lon - min_lon) < (min_lon - max_lon + 360):
    return min_lon, max_lon
  else:
    return max_lon, min_lon


def dateline_safe_is_west_check(
    lon_tocheck: float, lon_reference: float
) -> bool:
  """Check if one longitude is west of the other.

  Args:
    lon_tocheck: Longitude for which to check whether it's the westernmost, in
      degrees.
    lon_reference: Reference longitude to which to compare lon_tocheck, in
      degrees.

  Returns:
    bool, such that when True the distance between the longitudes is smaller
    when lon_tocheck is considered west of lon_reference.
  """
  lon_west, lon_east = dateline_safe_west_east(lon_tocheck, lon_reference)
  return (lon_west < lon_east) == (lon_tocheck < lon_reference)


def to_gaussian(
    lats: jax.typing.ArrayLike, lngs: jax.typing.ArrayLike
) -> np.ndarray:
  """Converts latitude and longitude coordinates to 3D geocentric coordinates.

  Args:
    lats: Latitude(s) in degrees.
    lngs: Longitude(s) in degrees.

  Returns:
    A numpy array of 3D geocentric (x, y, z) coordinates in meters with shape
    `(..., 3)`, assuming a spherical Earth.
  """
  lats_rad = np.deg2rad(lats)
  lngs_rad = np.deg2rad(lngs)
  sin_lats = np.sin(lats_rad)
  sin_lngs = np.sin(lngs_rad)
  cos_lats = np.cos(lats_rad)
  cos_lngs = np.cos(lngs_rad)
  return np.stack(
      [
          projection.EARTH_RADIUS_M * cos_lats * cos_lngs,
          projection.EARTH_RADIUS_M * cos_lats * sin_lngs,
          projection.EARTH_RADIUS_M * sin_lats,
      ],
      axis=-1,
  )


def normalize(v: np.ndarray) -> np.ndarray:
  """Normalizes vector(s) to unit length along the last axis.

  Args:
    v: Input vector or batch of vectors as a numpy array.

  Returns:
    A numpy array with the same shape as `v`, normalized to unit length along
    the last axis.
  """
  return v / np.linalg.norm(v, axis=-1, keepdims=True)


def lat_lngs_to_segment(
    lat1: jax.typing.ArrayLike,
    lng1: jax.typing.ArrayLike,
    lat2: jax.typing.ArrayLike,
    lng2: jax.typing.ArrayLike,
) -> Segment:
  """Computes a `Segment` from contrail endpoints defined by lat/lngs.

  Args:
    lat1: Latitude(s) of the first contrail endpoint in degrees.
    lng1: Longitude(s) of the first contrail endpoint in degrees.
    lat2: Latitude(s) of the second contrail endpoint in degrees.
    lng2: Longitude(s) of the second contrail endpoint in degrees.

  Returns:
    A `Segment` containing the contrail length in kilometers and the `v_hat`
    and `w_hat` unit basis vectors.
  """
  begin = to_gaussian(lat1, lng1)
  end = to_gaussian(lat2, lng2)
  v_hat = normalize(end - begin)
  return Segment(
      length_km=np.linalg.norm(end - begin, axis=-1) / 1000,
      v_hat=v_hat,
      w_hat=normalize(np.cross((begin + end) / 2, v_hat)),
  )


def advect_flight_to_time(
    flight_to_advect: Flight, time: pd.Timestamp, weather: xr.Dataset
) -> Flight:
  """Helper method to advect a flight to a given time.

  Args:
    flight_to_advect: The flight to advect.
    time: A timestamp object that represents the time to advect the flight to.
    weather: Weather data to be used for the advection process.

  Returns:
    A Flight object that is advected to the target time.
  """
  if 'advected_time' in flight_to_advect.data:
    warnings.warn('Advecting a flight that has already been advected.')
  initial_state = oss_advection.AdvectionState.from_flight(flight_to_advect)
  target_hrs = time.timestamp() / 3600
  t_eval = np.stack([
      initial_state.time,
      np.full_like(initial_state.time, target_hrs),
  ])
  advector = oss_advection.LagrangianAdvector(weather)
  _, advected_flight = advector.advect(  # pyrefly: ignore[not-iterable]
      flight_to_advect, t_eval, use_moist_advection=True
  )
  return advected_flight


def _score_components(
    params: jax.typing.ArrayLike,
    vs: jax.typing.ArrayLike,
    ws: jax.typing.ArrayLike,
    age_hrs: jax.typing.ArrayLike,
    weight_options: WeightOptions,
    mask: jax.typing.ArrayLike | None = None,
) -> ScoreComponents:
  """Computes the components of the match score.

  All args, unless otherwise specified, can be either JAX or numpy arrays, but
  obviously if you want the score gradients they need to be JAX arrays.

  Args:
    params: The current learned param values at this step of the optimization.
    vs: A (possibly padded) array of the v parameters of the flight parcels.
    ws: A (possibly padded) array of the w parameters of the flight parcels.
    age_hrs: A (possibly padded) array of the ages of the flight parcels.
    weight_options: The age-dependent weights of the different loss terms.
    mask: An array that is the same shape as vs, ws, and age_hrs, and has the
      value 1 where the values in the other arrays are valid and 0 where there
      are padded values. If this is None, the vs, ws, an age_hrs are assumed not
      to be padded.

  Returns:
    The ScoreComponents object derived from the input values, which can be used
    to compute a scalar match score.
  """
  np_impl = jnp if isinstance(params, jax.Array) else np

  if mask is None:
    mask = np_impl.ones_like(vs)

  params_arr = np_impl.asarray(params)
  v_offset, w_offset, theta_in = params_arr
  # Avg age of parcels, excluding padding.
  avg_age = np_impl.sum(mask * age_hrs) / np_impl.sum(mask)
  weights = Weights(
      age_hours=avg_age,
      weight_options=weight_options,
  )
  theta = np_impl.pi * np_impl.tanh(theta_in)
  sin_theta = np_impl.sin(theta)
  cos_theta = np_impl.cos(theta)
  mean_w_squared = np_impl.sum(
      mask * (sin_theta * (vs + v_offset) + cos_theta * (ws + w_offset)) ** 2
  ) / np_impl.sum(mask)

  return ScoreComponents(
      fit_score=weights.match_weight * mean_w_squared,
      shift_score=(
          weights.shift_weight * (v_offset * v_offset + w_offset * w_offset)
      ),
      angle_score=weights.rotation_weight * (1 - cos_theta),
      age_score=weights.age_weight,
      v_offset=v_offset,
      w_offset=w_offset,
      theta=theta,
      age_hrs=avg_age,
  )


def _objective_func(
    params: jax.Array,
    vs: jax.Array,
    ws: jax.Array,
    age_hrs: jax.Array,
    mask: jax.Array,
    weight_options: WeightOptions,
) -> jax.Array:
  """The objective function of the scoring optimization.

  Args:
    params: The current learned param values at this step of the optimization.
    vs: A padded JAX array of the v parameters of the flight parcels.
    ws: A padded JAX array of the w parameters of the flight parcels.
    age_hrs: A padded JAX array of the ages of the flight parcels.
    mask: A JAX array that is the same shape as vs, ws, and age_hrs, and has the
      value 1 where the values in the other arrays are valid and 0 where there
      are padded values.
    weight_options: The age-dependent weights of the different loss terms.

  Returns:
    The match score (which is also the loss).
  """
  return typing.cast(
      jax.Array,
      _score_components(params, vs, ws, age_hrs, weight_options, mask).total(),
  )


@jax.jit
def _run_optimization(
    vs: jax.Array,
    ws: jax.Array,
    age_hrs: jax.Array,
    mask: jax.Array,
    weight_options: WeightOptions,
) -> tuple[jax.Array, optax.OptState]:
  """Runs the scoring optimization.

  Args:
    vs: A padded JAX array of the v parameters of the flight parcels.
    ws: A padded JAX array of the w parameters of the flight parcels.
    age_hrs: A padded JAX array of the ages of the flight parcels.
    mask: A JAX array that is the same shape as vs, ws, and age_hrs, and has the
      value 1 where the values in the other arrays are valid and 0 where there
      are padded values.
    weight_options: The age-dependent weights of the different loss terms.

  Returns:
    * The learned (v_offset, w_offset, theta) params in a JAX array.
    * The optimizer state.
  """
  jax.config.update("jax_enable_x64", True)
  loss_and_grad_fn = jax.value_and_grad(
      functools.partial(
          _objective_func,
          vs=vs.astype(jnp.float64),
          ws=ws.astype(jnp.float64),
          age_hrs=age_hrs.astype(jnp.float64),
          mask=mask,
          weight_options=weight_options,
      )
  )
  loss_fn = lambda p: loss_and_grad_fn(p)[0]
  optimizer = optax.lbfgs()

  max_iters = 150
  tol = 1e-6

  def should_continue_fn(state):
    _, _, step, grad_norm = state
    return jnp.logical_and(step < max_iters, grad_norm > tol)

  def body_fn(state):
    params, opt_state, step, _ = state
    loss, grad = loss_and_grad_fn(params)
    updates, opt_state = optimizer.update(
        grad, opt_state, params, value=loss, grad=grad, value_fn=loss_fn
    )
    params = optax.apply_updates(params, updates)
    grad_norm = jnp.linalg.norm(grad)
    return params, opt_state, step + 1, grad_norm

  params_init = jnp.zeros(3, dtype=jnp.float64)
  state_init = optimizer.init(params_init)
  init_state = (params_init, state_init, 0, jnp.array(1e9, dtype=jnp.float64))

  final_state = jax.lax.while_loop(should_continue_fn, body_fn, init_state)
  return final_state[0], final_state[1]


_run_optimization_batched = jax.jit(
    jax.vmap(
        _run_optimization,
        in_axes=(0, 0, 0, 0, None),
    )
)


def convert_geojson(
    json: geojson.Feature, satellite_longitude: float | None = None
) -> LinearContrail:
  """Helper to convert a linear contrail in GeoJSON to a LinearContrail object.

  Args:
    json: A geojson.Feature object. The format of the GeoJSON must match the
      format sent by Google's GetDetections API; see
      https://developers.google.com/contrails/reference/v2/detections/getDetections
        and https://www.rfc-editor.org/info/rfc7946/#section-3.
    satellite_longitude: The longitude of the satellite that detected the
      contrail. If not specified, will use default values from GOES, MTG and
      Himawari satellites.

  Returns:
    A LinearContrail object.
  """
  match json:
    case {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': [
                # Note that if supplied, alt1 = alt2 (at least for Google's API)
                [int() | float() as lon1, int() | float() as lat1, *alt1],
                [int() | float() as lon2, int() | float() as lat2, *_],
            ],
        },
        'properties': {
            'time': str(time_str),
            'satellite_origin': str(satellite_origin),
        },
    }:
      return LinearContrail(
          lon1=float(lon1),
          lat1=float(lat1),
          lon2=float(lon2),
          lat2=float(lat2),
          altitude_m=float(alt1[0]) if alt1 else None,
          timestamp=int(pd.Timestamp(time_str).timestamp()),
          satellite_longitude=satellite_longitude
          if satellite_longitude is not None
          else SATELLITES_TO_LONGITUDES_MAP[satellite_origin],
      )
    case _:
      raise ValueError('Ill-formatted GeoJSON object.')


def _get_age_hrs(
    advected_flight: Flight,
) -> np.ndarray:
  """Return the age of waypoints, calculated by the duration of advection."""
  if 'advected_time' not in advected_flight.data:
    raise ValueError(
        'Flight has not been advected. Advect the flight using the '
        'advect_flight_to_time helper method.'
    )
  time_diffs = (
      advected_flight.data['advected_time'] - advected_flight.data['time']
  )
  return time_diffs / np.timedelta64(1, 'h')


def _get_slop_km(
    age_hrs: np.ndarray,
    wind_offset_error_kph: float = C_SHIFT,
    measurement_error_km: float = C_FIT,
) -> np.ndarray:
  """Calculate 'slop' via waypoint age, wind offset and measurement error."""
  return np.hypot(
      age_hrs * wind_offset_error_kph, age_hrs * measurement_error_km
  )


def _compute_segments(
    contrails: Sequence[LinearContrail], expected_longitude: float
) -> Segment:
  """Computes the Segment geometry of many contrails in one batch.

  Calls `lat_lngs_to_segment` once on the stacked endpoints of all `contrails`
  instead of once per contrail, since a satellite frame can contain hundreds
  of candidate contrails per flight.

  Args:
    contrails: The contrails to compute geometry for.
    expected_longitude: The longitude of the satellite being matched. All
      supplied contrails must be in the same longitude.

  Returns:
    A Segment whose fields have a leading contrail axis: `length_km` has shape
    (num_contrails,) and `v_hat` / `w_hat` have shape (num_contrails, 3).

  Raises:
    ValueError: If any contrail's parallax type differs from
      `expected_longitude`, because its coordinates would then not be
      comparable with the flight's satellite-perspective coordinates.
  """
  for contrail in contrails:
    if contrail.satellite_longitude != expected_longitude:
      raise ValueError(
          f'Parallax mismatch for contrail at timestamp {contrail.timestamp}:'
          f' Contrail longitude {contrail.satellite_longitude} does not match'
          f' satellite longitude {expected_longitude}'
      )
  # The reshape keeps the shape (num_contrails, 4) even for an empty list.
  endpoints = np.array(
      [(c.lat1, c.lon1, c.lat2, c.lon2) for c in contrails], dtype=np.float64
  ).reshape(-1, 4)
  return lat_lngs_to_segment(
      endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], endpoints[:, 3]
  )


def _batch_and_pad_contrails(
    vs_list: Sequence[jax.Array],
    ws_list: Sequence[jax.Array],
    age_hrs_list: Sequence[jax.Array],
    mask_list: Sequence[jax.Array],
    batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Batches and pads contrail inputs to a fixed batch size.

  Pads by replicating the first element in each list.

  Args:
    vs_list: List of vs arrays.
    ws_list: List of ws arrays.
    age_hrs_list: List of age_hrs arrays.
    mask_list: List of mask arrays.
    batch_size: The target batch size.

  Returns:
    A tuple of stacked and padded (vs, ws, age_hrs, mask) JAX arrays.
  """
  num_to_pad = batch_size - (len(vs_list) % batch_size)
  if num_to_pad == batch_size:
    num_to_pad = 0

  extended_vs = list(vs_list)
  extended_ws = list(ws_list)
  extended_age_hrs = list(age_hrs_list)
  extended_mask = list(mask_list)

  if num_to_pad > 0:
    first_vs = vs_list[0]
    first_ws = ws_list[0]
    first_age_hrs = age_hrs_list[0]
    first_mask = mask_list[0]

    extended_vs.extend([first_vs] * num_to_pad)
    extended_ws.extend([first_ws] * num_to_pad)
    extended_age_hrs.extend([first_age_hrs] * num_to_pad)
    extended_mask.extend([first_mask] * num_to_pad)

  return (
      jnp.stack(extended_vs),
      jnp.stack(extended_ws),
      jnp.stack(extended_age_hrs),
      jnp.stack(extended_mask),
  )


class SingleFrameContrailMatcher:
  """Matcher for single-frame contrail matching using Jax.

  This is an implementation of the algorithm outlined by Geraedts et. al. (2024)
  as outlined in https://iopscience.iop.org/article/10.1088/2515-7620/ad11ab.
  """

  def __init__(
      self,
      satellite_longitude: float,
      satellite_altitude_km: float,
      weight_options: WeightOptions = WeightOptions(),
  ):
    """Create a ContrailMatcher object.

    Args:
       satellite_longitude: A float for satellite longitude to project a flight
         into a satellite frame (thus ensuring both contrails and flights are in
         the same flight coordinate system).
       satellite_altitude_km: A float for satellite altitude to project a flight
         into a satellite frame.
       weight_options: A WeightOptions instance to adjust for lateral/rotational
         advection error, residual linearity error, advection duration error,
         and optionally vertical differences.
    """
    self.satellite_longitude = satellite_longitude
    self.satellite_altitude_km = satellite_altitude_km
    self.weight_options = weight_options
    self.counters: collections.Counter[str] = collections.Counter()

  def _reject_contrails(
      self, remaining: np.ndarray, *, rejected: np.ndarray, reason: str
  ) -> np.ndarray:
    """Counts the contrails newly rejected by a check and returns the remaining.

    Only contrails still in `remaining` are counted, so each rejected contrail
    is counted once, under the first check it fails.

    Args:
      remaining: Boolean mask of the contrails that passed every earlier check,
        shape (num_contrails,).
      rejected: Boolean mask of the contrails that fail this check, shape
        (num_contrails,).
      reason: Name of the counter to increment.

    Returns:
      Boolean mask of the contrails in `remaining` that pass this check.
    """
    newly_rejected = rejected & remaining
    if newly_rejected.any():
      self.counters[reason] += int(newly_rejected.sum())
    return remaining & ~rejected

  def _compute_gaussians(self, flight_parcel: Flight) -> np.ndarray:
    lats, lngs = projection.geostationary_parallax_correction(
        a_lat_degrees=flight_parcel.data['latitude'],
        a_lng_degrees=flight_parcel.data['longitude'],
        a_altitude_km=flight_parcel.data['altitude'] / 1000,
        b_altitude_km=0,
        satellite_longitude=self.satellite_longitude,
        satellite_altitude_km=self.satellite_altitude_km,
    )
    return to_gaussian(lats, lngs)

  def _validation_checks(
      self,
      contrails: Sequence[LinearContrail],
      flight_to_check: Flight,
      flight_age_hrs: np.ndarray,
      gaussians: np.ndarray | None = None,
  ) -> Sequence[ProcessedMatchingInputs]:
    """Applies per contrail-flight pair heuristic filtering and builds scoring inputs.

    Evaluates all rejection heuristics vectorized over `contrails`, returning
    early whenever no candidates remain:

      1. Length: contrails shorter than MIN_CONTRAIL_LENGTH_KM are too small
         to match reliably.
      2. Normal-to-contrail distance: if every parcel lies more than `slop_km`
         from the contrail's great circle, the flight cannot have formed it.
         `slop_km` grows with parcel age to allow for advection error.
      3. Along-contrail overlap: at least one parcel must fall within the
         contrail's extent (plus `slop_km`) along its direction; otherwise the
         flight passed beside the contrail rather than along it.
      4. Average parcel age: the parcels taking part in the fit must have an
         average age of at least MIN_CONTRAIL_AGE_HRS.
      5. Altitude difference: when configured, the contrail's speculated
         altitude must be within `altitude_difference_threshold_m` of the mean
         flight altitude over the kept parcels.
      6. Overlap duration: parcels directly within the contrail's extent must
         span at least `MIN_OVERLAP_SECONDS`.
      7. Spans contrail: the youngest parcel must be past the contrail's
         extent (minus `slop_km`).

    Args:
      contrails: Candidate contrails from a single satellite frame.
      flight_to_check: The single flight to match against, with a single step.
      flight_age_hrs: Age of each parcel in hours, shape (num_parcels,).
      gaussians: Pre-computed geocentric (x, y, z) coordinates in meters of each
        parcel of `flight_to_check`, shape (num_parcels, 3). Computed from
        `flight_to_check` if None.

    Returns:
      The scoring inputs of each contrail that passes every heuristic, in the
      order of `contrails`.
    """
    if not contrails:
      return []
    segments = _compute_segments(contrails, self.satellite_longitude)

    # Contrails that have passed every heuristic so far.
    remaining = np.ones(len(contrails), dtype=bool)
    # Check 1: Reject contrails below a certain length, given they're too small
    # to reliably match.
    remaining = self._reject_contrails(
        remaining,
        rejected=segments.length_km < MIN_CONTRAIL_LENGTH_KM,
        reason='small contrail',
    )
    if not remaining.any():
      return []

    if gaussians is None:
      gaussians = self._compute_gaussians(flight_to_check)
    # Project every parcel into every contrail's (v, w) frame with two matrix
    # products: (num_parcels, 3) @ (3, num_contrails) -> (num_parcels,
    # num_contrails), in km.
    vs = gaussians @ segments.v_hat.T / 1000
    ws = gaussians @ segments.w_hat.T / 1000
    slop_km = _get_slop_km(flight_age_hrs)[:, np.newaxis]

    # Check 2: Reject flight if it is are too far away from contrails.
    # The flight is considered too far away if every parcel is more than
    # slop_km from the contrail, on either side. w_hat always points north
    # (endpoints are ordered west to east), so `ws > slop_km` alone would
    # only ever reject flights north of the contrail.
    remaining = self._reject_contrails(
        remaining,
        rejected=np.all(np.abs(ws) > slop_km, axis=0),
        reason='too far away',
    )
    if not remaining.any():
      return []

    half_lengths = segments.length_km / 2
    # Flight parcel close enough along each contrail to take part in its fit.
    keep_masks = np.abs(vs) <= half_lengths + slop_km
    keep_counts = keep_masks.sum(axis=0)
    # Check 3: at least one parcel must fall within the contrail's extent
    # (plus `slop_km`) along its direction; otherwise the
    # flight passed beside the contrail rather than along it.
    remaining = self._reject_contrails(
        remaining, rejected=keep_counts == 0, reason='no overlap'
    )
    if not remaining.any():
      return []

    # Average age of the parcels that take part in each fit. Contrails with
    # `keep_counts == 0` are already False in `remaining`; clamping the
    # denominator avoids a 0/0 RuntimeWarning on those columns.
    safe_keep_counts = np.maximum(keep_counts, 1)
    mean_age_hrs = (
        np.where(keep_masks, flight_age_hrs[:, np.newaxis], 0).sum(axis=0)
        / safe_keep_counts
    )
    # Check 4: The parcels taking part in the fit must have an
    # average age of at least MIN_CONTRAIL_AGE_HRS.
    remaining = self._reject_contrails(
        remaining,
        rejected=mean_age_hrs < MIN_CONTRAIL_AGE_HRS,
        reason='young average parcel age',
    )
    if not remaining.any():
      return []

    # Check 5: If specified, the contrail's altitude must be within
    # `altitude_difference_threshold_m` of the mean flight altitude
    # over the kept parcels.
    threshold_m = self.weight_options.altitude_difference_threshold_m
    if threshold_m is not None:
      contrail_altitudes_m = np.array(
          [np.nan if c.altitude_m is None else c.altitude_m for c in contrails],
          dtype=np.float64,
      )
      altitude_is_nan = np.array(
          [
              c.altitude_m is not None and np.isnan(c.altitude_m)
              for c in contrails
          ],
          dtype=bool,
      )
      if (altitude_is_nan & remaining).any():
        self.counters['contrail altitude is nan'] += int(
            (altitude_is_nan & remaining).sum()
        )
      if (~np.isnan(contrail_altitudes_m) & remaining).any():
        flight_altitudes_m = flight_to_check.data['altitude']
        mean_altitudes_m = (
            np.where(keep_masks, flight_altitudes_m[:, np.newaxis], 0).sum(
                axis=0
            )
            / safe_keep_counts
        )
        remaining = self._reject_contrails(
            remaining,
            rejected=(
                np.abs(mean_altitudes_m - contrail_altitudes_m) > threshold_m
            ),
            reason=f'altitude difference larger than {threshold_m}m',
        )
        if not remaining.any():
          return []

    # Each contrail overlaps a different part of the flight, so we cannot slice
    # `timestamps` into a 2D array directly. Instead, find the index of the
    # first and last overlapping parcel in each column: `np.argmax` on a boolean
    # column returns the index of its first True, and running it on the
    # reversed rows (`overlap_masks[::-1]`) gives the offset of the last True
    # from the end (`num_parcels - 1`).
    overlap_masks = np.abs(vs) <= half_lengths
    timestamps = (
        flight_to_check.data['time'].astype('datetime64[s]').astype(np.int64)
    )
    num_parcels = len(timestamps)
    first_overlap = np.argmax(overlap_masks, axis=0)
    last_overlap = num_parcels - 1 - np.argmax(overlap_masks[::-1], axis=0)
    # Check 6: Parcels directly within the contrail's extent must
    # span at least `MIN_OVERLAP_SECONDS`.
    remaining = self._reject_contrails(
        remaining,
        rejected=(
            ~overlap_masks.any(axis=0)
            | (
                timestamps[last_overlap] - timestamps[first_overlap]
                < MIN_OVERLAP_SECONDS
            )
        ),
        reason='too short overlap',
    )
    if not remaining.any():
      return []

    youngest_parcel_idx = np.argmin(flight_age_hrs)
    v_of_youngest_parcel = vs[youngest_parcel_idx]
    # Check 7: The youngest part of the flight must be past the contrail's
    # extent (minus `slop_km`).
    remaining = self._reject_contrails(
        remaining,
        rejected=(
            np.abs(v_of_youngest_parcel)
            < half_lengths - _get_slop_km(flight_age_hrs[youngest_parcel_idx])
        ),
        reason="flight doesn't span contrail",
    )
    if not remaining.any():
      return []

    oldest_parcel_idx = np.argmax(flight_age_hrs)
    v_of_oldest_parcel = vs[oldest_parcel_idx]
    flip_sign = v_of_oldest_parcel > v_of_youngest_parcel
    # This is equivalent to having the invariant that endpoint1
    # of the contrail is older than endpoint2 of the contrail.
    # We can't actually determine that at detection time, so
    # initially the endpoint ordering is established based on lng1 < lng2. This
    # can cause instability in the signs of the resulting v and w offsets for
    # contrails oriented north/south or for contrails that advect across the
    # international dateline. Note that this doesn't actually matter for the
    # matching done here (the match scores all use squares of the v and w
    # offsets) but for downstream analysis of the v and w offsets
    # over the course of multiple matches for the same flight,
    # this is important.
    vs = np.where(flip_sign, -vs, vs)
    ws = np.where(flip_sign, -ws, ws)

    match_data_list = []
    for i in np.flatnonzero(remaining).tolist():
      match_data_list.append(
          ProcessedMatchingInputs(
              contrail=contrails[i],
              vs=vs[:, i],
              ws=ws[:, i],
              keep_mask=keep_masks[:, i],
              overlap_start_timestamp=int(timestamps[first_overlap[i]]),
              overlap_end_timestamp=int(timestamps[last_overlap[i]]),
          )
      )
    return match_data_list

  def _build_match_score(
      self,
      match_data: ProcessedMatchingInputs,
      params: jax.typing.ArrayLike,
      flight_age_hrs: np.ndarray,
      flight_id: str,
  ) -> ContrailMatchScore:
    """Computes unpadded score components and builds a ContrailMatchScore."""
    np_keep_mask = np.asarray(match_data.keep_mask)
    score_components = _score_components(
        np.asarray(params),
        np.asarray(match_data.vs)[np_keep_mask],
        np.asarray(match_data.ws)[np_keep_mask],
        np.asarray(flight_age_hrs)[np_keep_mask],
        weight_options=self.weight_options,
    ).with_scalar_values()
    return ContrailMatchScore(
        contrail=match_data.contrail,
        score=typing.cast(float, score_components.total()),
        flight_id=flight_id,
        overlap_start_timestamp=match_data.overlap_start_timestamp,
        overlap_end_timestamp=match_data.overlap_end_timestamp,
        score_components=score_components,
    )

  def match_score(
      self,
      advected_flight: Flight,
      contrail: geojson.Feature | LinearContrail,
      optimization_age_hrs: float | None = None,
      gaussians: np.ndarray | None = None,
  ) -> ContrailMatchScore | None:
    """Computes the match score for a single flight and a single contrail.

    Args:
      advected_flight: The advected flight data.
      contrail: The contrail, which can either be a LinearContrail instance or
        GeoJSON (in which case it will be coerced into the former).
      optimization_age_hrs: The age to use during the matching optimization for
        all segment-contrail pairs, regardless of true age. The actual age will
        be used only in the final score calculation. If this is None, the true
        age is used for the optimization also.
      gaussians: Pre-computed Gaussians for the advected flight data.

    Returns:
      A ContrailMatchScore object that holds the results of
      the scoring optimization and the overlapping flight timestamp range, or
      None if the optimization didn't run.

    Raises:
      ValueError: If the contrail parallax type does not match the satellite
        parallax type.
    """
    if isinstance(contrail, geojson.Feature):
      contrail = convert_geojson(contrail, self.satellite_longitude)
    flight_age_hrs = _get_age_hrs(advected_flight)
    match_data_list = self._validation_checks(
        [contrail], advected_flight, flight_age_hrs, gaussians
    )
    if not match_data_list:
      return None
    match_data = match_data_list[0]
    padded_data = _pad_matching_inputs(
        match_data, flight_age_hrs, optimization_age_hrs
    )
    params, _ = _run_optimization(
        vs=padded_data.vs,
        ws=padded_data.ws,
        age_hrs=padded_data.age_hrs,
        mask=padded_data.mask,
        weight_options=self.weight_options,
    )
    return self._build_match_score(
        match_data,
        params,
        flight_age_hrs,
        flight_id=str(advected_flight.attrs.get('flight_id', '')),
    )

  def _better_match_found(
      self,
      contrail: LinearContrail,
      flights: Sequence[Flight],
      flights_gaussians: Sequence[np.ndarray],
      threshold: float,
      optimization_age_hrs: float | None,
  ) -> bool:
    """Returns True if any flight in `flights` scores below `threshold`."""
    for fl, gaussians in zip(flights, flights_gaussians, strict=True):
      contrail_match_score = self.match_score(
          fl,
          contrail,
          optimization_age_hrs=optimization_age_hrs,
          gaussians=gaussians,
      )
      if contrail_match_score is not None:
        if contrail_match_score.score < threshold:
          return True
    return False

  def match_to_contrails(
      self,
      potential_matches: Sequence[geojson.Feature | LinearContrail],
      advected_flight: Flight,
      consider_better_matches: bool,
      flights: Sequence[Flight] | None = None,
      contrail_optimization_age_hrs: float | None = None,
  ) -> list[ContrailMatchScore]:
    """Matches an advected flight to a list of candidate contrails.

    If `consider_better_matches` is True, candidate matches with scores
    below `MATCH_SCORE_THRESHOLD` are compared against all flights in `flights`,
    and any contrail for which another flight has a score lower by more than
    `BETTER_SCORE_MARGIN` is excluded.

    Args:
      potential_matches: Candidate contrails to match against `advected_flight`.
      advected_flight: The advected flight data.
      consider_better_matches: Whether to exclude candidate matches when another
        flight in `flights` matches the contrail with a score lower by more than
        `BETTER_SCORE_MARGIN`.
      flights: Advected flights to compare against when
        `consider_better_matches` is True. If None or empty while
        `consider_better_matches` is True, no matches are returned.
      contrail_optimization_age_hrs: The age to use during the matching
        optimization for all segment-contrail pairs, regardless of true age. The
        actual age will be used only in the final score calculation. If this is
        None, the true age is used for the optimization also.

    Returns:
      A list of ContrailMatchScore objects for the contrails that pass
      validation checks. Any contrail that passes validation is returned. If
      consider_better_matches is True, will return ContrailMatchScores only if
      they are not excluded by a better match.
      Callers doing single-frame matching should filter the returned list by
      score < MATCH_SCORE_THRESHOLD.

    Raises:
      ValueError: If `advected_flight` has not been advected, or if any
        contrail's satellite longitude does not match the matcher's satellite
        longitude.
    """
    if not advected_flight:
      return []
    if consider_better_matches and not flights:
      return []
    potential_contrail_matches = []
    for contrail in potential_matches:
      if isinstance(contrail, geojson.Feature):
        contrail = convert_geojson(contrail)
      potential_contrail_matches.append(contrail)
    flight_id = str(advected_flight.attrs.get('flight_id', ''))
    flight_age_hrs = _get_age_hrs(advected_flight)
    if (flight_age_hrs < MIN_CONTRAIL_AGE_HRS).all():
      return []

    gaussians = self._compute_gaussians(advected_flight)
    match_data_list = self._validation_checks(
        potential_contrail_matches, advected_flight, flight_age_hrs, gaussians
    )
    if not match_data_list:
      return []

    # Batch and run JAX
    match_results = []
    flights_gaussians = (
        [self._compute_gaussians(fl) for fl in flights]
        if consider_better_matches and flights
        else []
    )

    for i in range(0, len(match_data_list), JAX_BATCH_SIZE):
      chunk = match_data_list[i : i + JAX_BATCH_SIZE]

      padded_chunk = [
          _pad_matching_inputs(
              match_data, flight_age_hrs, contrail_optimization_age_hrs
          )
          for match_data in chunk
      ]
      vs_batched, ws_batched, age_hrs_batched, mask_batched = (
          _batch_and_pad_contrails(
              [padded.vs for padded in padded_chunk],
              [padded.ws for padded in padded_chunk],
              [padded.age_hrs for padded in padded_chunk],
              [padded.mask for padded in padded_chunk],
              JAX_BATCH_SIZE,
          )
      )

      params_batched, _ = _run_optimization_batched(
          vs_batched,
          ws_batched,
          age_hrs_batched,
          mask_batched,
          self.weight_options,
      )

      for j, match_data in enumerate(chunk):
        match_score = self._build_match_score(
            match_data,
            params_batched[j],
            flight_age_hrs,
            flight_id=flight_id,
        )

        if (
            consider_better_matches
            and flights is not None
            and match_score.score < MATCH_SCORE_THRESHOLD
            and match_score.score - BETTER_SCORE_MARGIN > 0
        ):
          if self._better_match_found(
              match_data.contrail,
              flights,
              flights_gaussians,
              threshold=match_score.score - BETTER_SCORE_MARGIN,
              optimization_age_hrs=contrail_optimization_age_hrs,
          ):
            continue

        match_results.append(match_score)

    return match_results
