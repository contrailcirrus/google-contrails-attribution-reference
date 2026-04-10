"""Utilities for casting numpy/JAX types."""

import datetime
import enum
from typing import Union

import dateutil
import numpy as np
import pandas as pd

from . import ce_types

TimeType = ce_types.TimeType
NPDtype = ce_types.NPDtype
ArrayLike = ce_types.ArrayLike

# For converting datetime64 indices to floats.
REFERENCE_TIME_1970 = pd.to_datetime('1970-01-01T00:00:00').tz_localize('UTC')

_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.float128)


class Units(enum.Enum):
  """Units for time to float conversion."""

  DAYS = 'DAYS'
  HOURS = 'HOURS'
  SECONDS = 'SECONDS'

  @property
  def seconds_per_unit(self) -> int:
    return {
        'SECONDS': 1,
        'HOURS': 60 * 60,
        'DAYS': 24 * 60 * 60,
    }[self.value]

  @property
  def hours_per_unit(self) -> float:
    return self.seconds_per_unit / 3600


def _assert_allowed_float_type(dtype: NPDtype) -> None:
  if dtype not in _FLOAT_TYPES:
    raise TypeError(f'`dtype` {dtype} was not in {_FLOAT_TYPES}')


def maybe_localize_datetimeindex(
    dt: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
  """Localizes a DatetimeIndex to UTC only if it is currently timezone naive."""
  if hasattr(dt, 'tz') and dt.tz is None:
    dt = dt.tz_localize(dateutil.tz.UTC)
  elif not hasattr(dt, 'tz'):
    dt = pd.to_datetime(dt)
    if dt.tz is None:
      dt = dt.tz_localize(dateutil.tz.UTC)
  return dt


def timetype_to_float(
    time: TimeType,
    reference_time: datetime.datetime = REFERENCE_TIME_1970,
    unit: Units = Units.SECONDS,
    dtype: NPDtype = np.float64,
) -> np.ndarray:
  """Convert time types to float (units since `reference_time`)."""
  _assert_allowed_float_type(dtype)
  dt = pd.to_datetime(time)
  if isinstance(dt, pd.Series):
    dt = pd.DatetimeIndex(dt)
  return np.asarray(
      (maybe_localize_datetimeindex(dt) - reference_time).total_seconds()
      / unit.seconds_per_unit,
      dtype=dtype,
  )


def float_to_datetime(
    x: ArrayLike,
    reference_time: datetime.datetime = REFERENCE_TIME_1970,
    unit: Units = Units.SECONDS,
) -> Union[pd.DatetimeIndex, pd.Timestamp]:
  """Convert float (units since reference_time) to pandas datetime."""
  x_arr = np.asarray(x)
  # pd.to_timedelta does not support N-D arrays (N > 1).
  dt = pd.to_datetime(
      reference_time + pd.to_timedelta(x_arr.ravel(), unit=unit.value)
  )
  if x_arr.ndim == 0:
    return maybe_localize_datetimeindex(dt)[0]
  return maybe_localize_datetimeindex(dt)


def float_to_datetime64(
    x: ArrayLike,
    reference_time: datetime.datetime = REFERENCE_TIME_1970,
    unit: Units = Units.SECONDS,
) -> np.ndarray:
  """Convert float (units since reference_time) to datetime64."""
  x_arr = np.asarray(x)
  dt = float_to_datetime(x_arr, reference_time=reference_time, unit=unit)
  if isinstance(dt, pd.DatetimeIndex):
    return dt.values.reshape(x_arr.shape)
  elif isinstance(dt, pd.Timestamp):
    return dt.to_datetime64()
  else:
    raise AssertionError(f'Unexpected converted type {type(dt)} for {dt}.')


def to_float(
    x: ArrayLike,
    reference_time: datetime.datetime = REFERENCE_TIME_1970,
    unit: Units = Units.SECONDS,
    dtype: NPDtype = np.float64,
) -> np.ndarray:
  """Convert `x` to float, using `timetype_to_float` to handle times."""
  _assert_allowed_float_type(dtype)
  x_arr = np.asarray(x)
  if np.issubdtype(x_arr.dtype, np.datetime64):
    return timetype_to_float(
        x_arr, reference_time=reference_time, unit=unit, dtype=dtype
    )

  # Check for pandas types that np.asarray might have turned into objects
  if isinstance(x, (pd.DatetimeIndex, pd.Series, pd.Timestamp)):
    return timetype_to_float(
        x, reference_time=reference_time, unit=unit, dtype=dtype
    )

  return np.asarray(x, dtype=dtype)
