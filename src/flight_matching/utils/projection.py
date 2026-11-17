"""Projection library."""

from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

EARTH_RADIUS_M = 6_371_000  # Mean radius of Earth, ignoring oblateness.


def geostationary_parallax_correction(
    a_lat_degrees: npt.ArrayLike,
    a_lng_degrees: npt.ArrayLike,
    a_altitude_km: npt.ArrayLike,
    b_altitude_km: npt.ArrayLike,
    satellite_longitude: float,
    satellite_altitude_km: float,
) -> Tuple[np.ndarray, np.ndarray]:
  """Parallax-correct a 3D geodetic point to a satellite surface point (or vice-versa).

  IMPORTANT NOTES BEFORE USAGE:
  - This method is treating the earth as a perfect sphere. This
  means the parallax corrections may be off by up to 3%. For GOES
  the parallax correction will be at its largest ~5 pixels, so these errors
  are less than one pixel in size. If parallax correction precision is more
  important (e.g. stereovision altitude estimation), calculations can instead
  be done with an ellipsoidal earth model by using pyproj.

  Parallax causes objects which are higher than the earth's surface
  (e.g. contrails) to appear further from the satellite than they actually are.
  This method can correct for this.

  In particular, given object 'a' at coordinates a_lat_degrees,
  a_lng_degrees, a_altitude_km, compute the coordinates of object 'b', which
  is at altitude b_altitude_km, such that a and b would appear identically to a
  satellite at given coordinates.

  Concretely, if you have the geodetic location, including geometric altitude,
  of an advected flight waypoint, this can tell you the surface location where
  the satellite would observe a contrail that it formed.

  Similarly, if you have the satellite-observed surface location of a contrail
  and a geometric altitude estimate for it, this can tell you its
  geodetic location.

  Args:
    a_lat_degrees: latitude of a
    a_lng_degrees: longitude of a
    a_altitude_km: altitude of a (distance from earth's surface)
    b_altitude_km: altitude of b (distance from earth's surface)
    satellite_longitude: longitude of satellite
    satellite_altitude_km: altitude of satellite

  Returns:
    lat and lon of b, such that a and b appear the same to a satellite
  """
  a_lat_degrees = np.array(a_lat_degrees, dtype=float)
  a_lng_degrees = np.array(a_lng_degrees, dtype=float)
  a_altitude_km = np.array(a_altitude_km, dtype=float)
  b_altitude_km = np.array(b_altitude_km, dtype=float)

  apparent_lat_radians = a_lat_degrees / 180 * np.pi
  cphi_a = np.cos(apparent_lat_radians)
  ctheta = np.cos((satellite_longitude - a_lng_degrees) / 180 * np.pi)
  s = satellite_altitude_km + (EARTH_RADIUS_M / 1000)
  r = a_altitude_km + (EARTH_RADIUS_M / 1000)
  h = b_altitude_km + (EARTH_RADIUS_M / 1000)

  soln_exists = s * cphi_a * ctheta > (EARTH_RADIUS_M / 1000)
  t = (
      (s**2 - s * r * cphi_a * ctheta)
      - np.sqrt(
          s**2 * r**2 * (cphi_a**2 * ctheta**2 - 1)
          + h**2 * (r**2 + s**2 - 2 * r * s * cphi_a * ctheta)
      )
  ) / (r**2 + s**2 - 2 * s * r * cphi_a * ctheta)

  sin_phi_c = t * r / h * np.sin(apparent_lat_radians)
  b_latitude_degrees = np.arcsin(sin_phi_c) * 180 / np.pi
  b_latitude_degrees = np.where(soln_exists, b_latitude_degrees, np.nan)

  theta_s = satellite_longitude / 180 * np.pi
  theta_a = a_lng_degrees / 180 * np.pi
  b_longitude_radians = np.arctan2(
      s * np.sin(theta_s)
      + t * (r * np.sin(theta_a) * cphi_a - s * np.sin(theta_s)),
      s * np.cos(theta_s)
      + t * (r * np.cos(theta_a) * cphi_a - s * np.cos(theta_s)),
  )
  b_longitude_radians = np.where(soln_exists, b_longitude_radians, np.nan)
  if np.any(np.isnan(b_latitude_degrees)) or np.any(
      np.isnan(b_longitude_radians)
  ):
    warnings.warn(
        'One or more points could not be parallax corrected because they'
        'were not visible from the satellite.'
    )

  return b_latitude_degrees, b_longitude_radians * 180 / np.pi
