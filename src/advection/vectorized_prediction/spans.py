"""Definitions of an manipulations of space and time spans."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..utils import casting
from ..utils import ce_types
from ..utils import dataset_util as dsu
from . import constants as cs

XData = ce_types.XData
TimeType = ce_types.TimeType
D = cs.Dims


class BadSpanError(Exception):
  """Raise if a Space Span is invalid."""


@dataclasses.dataclass
class SpaceSpan:
  """Defines an axis-aligned 3D box in space."""

  latitude: Optional[tuple[float, float]] = None
  longitude: Optional[tuple[float, float]] = None
  pressure: Optional[tuple[float, float]] = None


@dataclasses.dataclass
class DropValues:
  """Specifies values to drop in slice_dataset_with_spans."""

  latitude: Optional[Sequence[float]] = None
  longitude: Optional[Sequence[float]] = None
  pressure: Optional[Sequence[float]] = None


class TimeSpan:
  """Minimum and maximum times in various formats."""

  def __init__(self, times: tuple[TimeType, TimeType]):
    self._times: tuple[pd.Timestamp, pd.Timestamp] = tuple(
        casting.maybe_localize_datetimeindex(pd.to_datetime(t)) for t in times
    )

  @property
  def times(self) -> tuple[pd.Timestamp, pd.Timestamp]:
    return self._times

  @property
  def datetimes(self) -> tuple[datetime.datetime, datetime.datetime]:
    return tuple(t.to_pydatetime() for t in self.times)

  @property
  def datetime64s(self) -> tuple[np.datetime64, np.datetime64]:
    return tuple(t.to_datetime64() for t in self.times)

  def subdivide(
      self,
      timestep: Union[
          Callable[[datetime.datetime], datetime.timedelta], datetime.timedelta
      ],
      allow_partial_slices: bool = False,
  ) -> list[TimeSpan]:
    """Divides a time span into smaller time spans."""
    start_dt, end_dt = self.datetimes
    outputs = []
    if callable(timestep):
      if timestep(start_dt) != timestep(end_dt):
        raise ValueError(
            'Time step changes within a single time span are not allowed.'
        )
      timestep = timestep(start_dt)

    dt = start_dt
    next_dt = dt + timestep
    while next_dt <= end_dt:
      outputs.append(TimeSpan((dt, next_dt)))
      dt = next_dt
      next_dt = dt + timestep

    if dt != end_dt:
      if allow_partial_slices:
        outputs.append(TimeSpan((dt, end_dt)))
      else:
        raise ValueError(
            f'Time ranges must be blocks evenly divisible by {timestep}'
        )

    return outputs

  def to_datetime_range(
      self,
      timestep: Union[
          Callable[[datetime.datetime], datetime.timedelta], datetime.timedelta
      ],
      allow_partial_slices: bool = False,
  ) -> list[datetime.datetime]:
    """Gets a range of datetimes at the specified interval within a timespan."""
    subspans = self.subdivide(timestep, allow_partial_slices)
    return [subspan.datetimes[0] for subspan in subspans]

  def contains(self, dt: datetime.datetime) -> bool:
    return self.datetimes[0] <= dt <= self.datetimes[1]


def space_span_from_data(data: XData) -> SpaceSpan:
  """SpaceSpan with same bounds as `data`."""
  latitude = np.asarray(data.latitude)
  longitude = np.asarray(data.longitude)
  pressure = np.asarray(data.pressure)
  return SpaceSpan(
      latitude=(np.nanmin(latitude), np.nanmax(latitude)),
      longitude=(np.nanmin(longitude), np.nanmax(longitude)),
      pressure=(np.nanmin(pressure), np.nanmax(pressure)),
  )


def time_span_from_data(data: XData) -> TimeSpan:
  """TimeSpan with same bounds as `data`."""
  if D.TIME in data:
    time = np.ravel(data.time)
  elif D.TIMESTAMP in data:
    # Simplified for demo
    time = pd.to_datetime(
        np.ravel(data.timestamp), unit='s', origin='unix'
    ).values
  else:
    raise ValueError('Could not extract time from `data`')
  t_min, t_max = np.min(time), np.max(time)
  return TimeSpan(times=(t_min, t_max))


def slice_dataset_with_spans(
    data: XData,
    time_span: Optional[TimeSpan] = None,
    space_span: Optional[SpaceSpan] = None,
    drop_values: Optional[DropValues] = None,
    floor_below: bool = True,
    ceil_above: bool = True,
) -> XData:
  """Get a slice of a Dataset."""
  span = {}
  if space_span is not None:
    span.update(dataclasses.asdict(space_span))
  if time_span is not None:
    span.update({D.TIME: time_span.datetime64s})

  sliced = dsu.slice_data(
      data=data,
      span=span,
      floor_below=floor_below,
      ceil_above=ceil_above,
  )

  if drop_values is not None:
    drop_values = dataclasses.asdict(drop_values)
    sliced = sliced.drop_sel(drop_values, errors='ignore')

  return sliced
