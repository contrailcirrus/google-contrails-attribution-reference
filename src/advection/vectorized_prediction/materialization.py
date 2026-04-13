"""Materialization and preparation of weather data for advection."""

import dataclasses
import datetime
import typing
from typing import Any, Optional

import xarray as xr

from ..utils import ce_types
from ..utils import dataset_util as dsu
from . import constants as cs
from . import spans

TimeType = ce_types.TimeType
Array = ce_types.Array
D = cs.Dims


@dataclasses.dataclass
class MaterializedAdvectionDatasets:
  """Container for materialized weather datasets used in advection."""

  u: xr.Dataset
  v: xr.Dataset
  w: xr.Dataset
  t: Optional[xr.Dataset] = None


def prepare_datasets_for_advection(
    ds: xr.Dataset,
) -> MaterializedAdvectionDatasets:
  """Prepare weather datasets for advection."""
  return MaterializedAdvectionDatasets(
      u=ds[['u']],
      v=ds[['v']],
      w=ds[['w']],
      t=ds[['t']] if 't' in ds else None,
  )


def slice_cast_swap_sort_weather(
    ds: xr.Dataset,
    time_span: Optional[spans.TimeSpan] = None,
    space_span: Optional[spans.SpaceSpan] = None,
    dtype: Any = cs.DTYPE,
    time_unit: Any = cs.TIME_UNIT,
    reference_time: datetime.datetime = cs.REFERENCE_TIME,
) -> xr.Dataset:
  """Slice, cast, swap and sort a weather dataset."""
  ds = spans.slice_dataset_with_spans(
      ds, time_span=time_span, space_span=space_span
  )
  ds = dsu.to_float(
      ds, dtype=dtype, unit=time_unit, reference_time=reference_time
  )
  return typing.cast(xr.Dataset, ds)
