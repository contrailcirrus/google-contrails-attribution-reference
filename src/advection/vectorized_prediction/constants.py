"""Constants for advection."""

from typing import Any, Union

import numpy as np

from ..utils import casting


class Dims:
  """Dimension names."""

  LATITUDE = 'latitude'
  LONGITUDE = 'longitude'
  NUMBER = 'number'
  PARCEL = 'parcel'
  STEP = 'step'
  TIME = 'time'
  TIMESTAMP = 'timestamp'
  ORIGINAL_TIMESTAMP = 'original_timestamp'
  ORIGINAL_LATITUDE = 'original_latitude'
  ORIGINAL_LONGITUDE = 'original_longitude'
  ORIGINAL_PRESSURE = 'original_pressure'
  CONTRAIL = 'contrail'
  AGE_HOURS = 'age_hours'
  COCIP_STEP = 'cocip_step'
  ATTACHING_STEP = 'attaching_step'

  HEIGHT_HPA = 'height_hpa'
  LEVEL = 'level'
  PRESSURE = 'pressure'
  HYBRID = 'hybrid'

  CONTEXT = 'context'
  RAYPOINT = 'raypoint'
  X = 'x'
  Y = 'y'
  ROLLOUT = 'rollout'


D = Dims


class Units:
  HPA = 'hPa'
  K = 'K'
  KG_INV_KG = 'kg kg**-1'


CANONICAL_DIMS = (D.TIME, D.PRESSURE, D.LATITUDE, D.LONGITUDE)
WEATHER_DIMS = (D.LEVEL, D.TIME, D.LATITUDE, D.LONGITUDE)
ADVECTION_WEATHER_VARS = ('u', 'v', 'w')
ADVECTION_WEATHER_VARS_WITH_TEMPERATURE = ('u', 'v', 'w', 't')

LOGSPACE_DIMS = (Dims.PRESSURE,)
LOGSPACE_VARS = ('q',)

DTYPE = np.float64
TIME_UNIT = casting.Units.HOURS
REFERENCE_TIME = casting.REFERENCE_TIME_1970

WeatherTypeCfg = Union[dict[str, Any], list[dict[str, Any]]]
