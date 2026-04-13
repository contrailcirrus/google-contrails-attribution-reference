"""Physical assumptions and functions for advections."""

import jax
import jax.numpy as jnp
import numpy as np
import tree_math

from ..utils import ce_types

config = jax.config
Array = ce_types.Array

config.update('jax_default_matmul_precision', 'float32')
config.update('jax_enable_x64', True)


@tree_math.struct
class Velocities:
  """Velocities in latitude/longitude/pressure(hPa) per hour."""

  latitude: float
  longitude: float
  pressure: float

  def __post_init__(self):
    if self.latitude < 0:
      raise ValueError(
          f'Latitude velocity must be non-negative. Found {self.latitude}'
      )
    if self.longitude < 0:
      raise ValueError(
          f'Longitude velocity must be non-negative. Found {self.longitude}'
      )
    if self.pressure < 0:
      raise ValueError(
          f'Pressure velocity must be non-negative. Found {self.pressure}'
      )


CONUS_LARGE_VELOCITIES = Velocities(latitude=4, longitude=4, pressure=30)

EARTH_RADIUS_M = 6371000


@tree_math.struct
class Gaussian:
  """Gaussian coordinate system."""

  x: Array
  y: Array
  z: Array


@tree_math.struct
class LatLon:
  """Latitude and longitude coordinates in degrees."""

  latitude: Array
  longitude: Array


@tree_math.struct
class Orthographic:
  """Orthographic coordinates for tangent vectors on the Earth's surface."""

  u: Array
  v: Array


def orthographic_to_gaussian_delta(
    g0: Gaussian,
    uv: Orthographic,
) -> Gaussian:
  """Converts orthographic tangent vector to a Gaussian delta."""
  s0 = jnp.hypot(g0.x, g0.y)
  dx = -g0.y * uv.u / s0 - g0.z * g0.x * uv.v / (EARTH_RADIUS_M * s0)
  dy = g0.x * uv.u / s0 - g0.z * g0.y * uv.v / (EARTH_RADIUS_M * s0)
  dz = s0 * uv.v / EARTH_RADIUS_M

  eps = np.finfo(jnp.asarray(g0.x).dtype).eps * EARTH_RADIUS_M
  dx = jnp.where(s0 > eps, dx, uv.u)
  dy = jnp.where(s0 > eps, dy, uv.v)
  dz = jnp.where(s0 > eps, dz, 0.0)

  return Gaussian(dx, dy, dz)


def gaussian_pair_to_orthographic(
    g0: Gaussian,
    g: Gaussian,
) -> Orthographic:
  """Map a base and perturbed Gaussian pair to orthographic tangent vector."""
  s0 = jnp.hypot(g0.x, g0.y)
  u = (g.y * g0.x - g.x * g0.y) / s0
  v = s0 * g.z / EARTH_RADIUS_M - g0.z * (g.x * g0.x + g.y * g0.y) / (
      EARTH_RADIUS_M * s0
  )

  eps = np.finfo(jnp.asarray(g0.x).dtype).eps * EARTH_RADIUS_M
  u = jnp.where(s0 > eps, u, g.x)
  v = jnp.where(s0 > eps, v, g.y)
  return Orthographic(u, v)


def latlon_to_gaussian(latlon: LatLon, altitude_m: float = 0) -> Gaussian:
  """Convert LatLon to Gaussian."""
  lat = jnp.asarray(latlon.latitude) * jnp.pi / 180  # Radians
  lon = jnp.asarray(latlon.longitude) * jnp.pi / 180  # Radians
  cos_lat = jnp.cos(lat)
  radius = EARTH_RADIUS_M + altitude_m
  return Gaussian(
      radius * cos_lat * jnp.cos(lon),
      radius * cos_lat * jnp.sin(lon),
      radius * jnp.sin(lat),
  )


def gaussian_to_latlon(g: Gaussian) -> LatLon:
  """Convert Gaussian to LatLon."""
  lat = jnp.arctan2(g.z, jnp.hypot(g.x, g.y))  # Radians
  lon = jnp.arctan2(g.y, g.x)  # Radians
  c = 180 / jnp.pi
  return LatLon(latitude=lat * c, longitude=lon * c)


def project_gaussian_to_earths_surface(g: Gaussian) -> Gaussian:
  """Project `g` onto the surface of the earth."""
  multiplier = EARTH_RADIUS_M / jnp.sqrt(g.x**2 + g.y**2 + g.z**2)
  return Gaussian(x=g.x * multiplier, y=g.y * multiplier, z=g.z * multiplier)
