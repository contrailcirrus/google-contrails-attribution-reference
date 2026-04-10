"""Utilities for working with XArray Datasets and DataArrays."""

from collections.abc import Hashable, Sequence
import datetime
from typing import Optional

import numpy as np
import xarray as xr

from . import casting
from . import ce_types


XData = ce_types.XData
NPDtype = ce_types.NPDtype
ArrayLike = ce_types.ArrayLike


def assert_dtype(data: XData, dtype: NPDtype) -> None:
  """Assert all data vars and dims have specified dtype."""

  def one_assert(obj, name=''):
    if obj.dtype != dtype:
      raise TypeError(
          f'Expected {name} to have dtype {dtype} but found {obj.dtype}.'
      )

  if isinstance(data, xr.DataArray):
    one_assert(data, 'argument `data`')
  else:
    for k, v in data.data_vars.items():
      one_assert(v, f'data var {k}')

  for d in data.dims:
    one_assert(data[d], f'Dimension {d}')


class InvalidXDataDimsError(Exception):
  """Raised if dimensions of DataArray/Dataset fail a validity check."""


def assert_valid_dataarray_dims(
    da: xr.DataArray,
    expected_dims: Sequence[str],
    expect_sorted: bool = False,
    check_dim_order: bool = True,
) -> None:
  """Check dimensions of `da` are as expected."""
  if check_dim_order:
    expected_dims = tuple(expected_dims)
    dims = tuple(da.dims)
  else:
    expected_dims = set(expected_dims)
    dims = set(da.dims)
  if dims != expected_dims:
    raise InvalidXDataDimsError(
        f'Wrong dimensions: dims for {da.name} = {dims} != {expected_dims}.'
    )
  if expect_sorted:
    for d in expected_dims:
      if not np.all(np.diff(da[d]) >= 0):
        raise InvalidXDataDimsError(f'Dimension {d} was not sorted.')


def assert_valid_dataset_dims(
    ds: xr.Dataset,
    expected_dims: Sequence[str],
    variables: Sequence[str],
    expect_sorted: bool = False,
    check_dim_order: bool = True,
) -> None:
  """Check dimensions of `ds` are as expected."""
  for var in variables:
    assert_valid_dataarray_dims(
        ds[var],
        expected_dims,
        expect_sorted=expect_sorted,
        check_dim_order=check_dim_order,
    )


def slice_data(
    data: XData,
    span: Optional[dict[Hashable, tuple[ArrayLike, ArrayLike]]] = None,
    floor_below: bool = True,
    ceil_above: bool = True,
) -> XData:
  """Select a subset of `data` by slicing."""
  data = data.copy()
  indexers = {}

  def _set_indexer(name, min_val, max_val):
    values = data[name].values.copy()
    ascending = values[0] <= values[1]
    if not ascending:
      values = values[::-1]

    min_val = values[0] if min_val is None else min_val
    max_val = values[-1] if max_val is None else max_val
    if min_val > max_val:
      raise ValueError(f'Span for {name} had min_val={min_val} > {max_val}')
    if floor_below:
      min_val = values[
          max(0, np.searchsorted(values, min_val, side='right') - 1)
      ]
    if ceil_above:
      max_val = values[
          min(np.searchsorted(values, max_val, side='left'), len(values) - 1)
      ]
    indexers[name] = slice(min_val, max_val)

    if not ascending:
      indexers[name] = slice(indexers[name].stop, indexers[name].start)

  if span is not None:
    for name in span:
      _set_indexer(name, span[name][0], span[name][1])
    data = data.sel(indexers)

  return data


def to_float(
    ds: XData,
    reference_time: datetime.datetime = casting.REFERENCE_TIME_1970,
    unit: casting.Units = casting.Units.SECONDS,
    dtype: NPDtype = np.float64,
) -> XData:
  """Convert variables and dimensions in `ds` to float."""
  ds = ds.copy(deep=True)
  ds = ds.astype(dtype)
  new_coords = {}
  for d in ds.dims:
    new_coords[d] = xr.DataArray(
        data=casting.to_float(
            ds[d].values,
            dtype=dtype,
            reference_time=reference_time,
            unit=unit,
        ),
        dims=ds[d].dims,
        attrs=ds[d].attrs,
    )
  return ds.assign_coords(new_coords)
