"""Utilities for validating np.arrays conform to expectations."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np

from . import ce_types

NPDtype = ce_types.NPDtype
ArrayLike = ce_types.ArrayLike
ArrayLikeOrNone = Union[ArrayLike, None]


def common_dtype(
    arrays: Union[Mapping[Any, ArrayLikeOrNone], Sequence[ArrayLikeOrNone]],
    preferred_dtype: Optional[NPDtype] = None,
    message: Optional[str] = None,
) -> Optional[NPDtype]:
  """Returns explicit shared dtype if there is one."""
  if isinstance(arrays, Mapping):
    arrays = list(arrays.values())

  message = message + ':  ' if message else ''
  dtype = None
  for a in arrays:
    if a is None or not hasattr(a, 'dtype'):
      continue
    dt = a.dtype
    if dtype is None:
      dtype = dt
    elif dtype != dt:
      raise TypeError(f'{message}Found incompatible dtypes, {dtype} and {dt}.')
  return preferred_dtype if dtype is None else dtype


def static_shape(
    array: Any,
    expect_shape: Optional[Sequence[Optional[int]]] = None,
    expect_ndim: Optional[int] = None,
    message: Optional[str] = None,
) -> tuple[int, ...]:
  """Validates that `array` has the expected static shape and/or ndim."""
  shape = np.shape(array)
  if expect_ndim is not None and len(shape) != expect_ndim:
    raise ValueError(
        f'{message or ""}: Expected ndim {expect_ndim}, but found {len(shape)}.'
    )
  if expect_shape is not None:
    if len(shape) != len(expect_shape):
      raise ValueError(
          f'{message or ""}: Expected shape {expect_shape}, but found {shape}.'
      )
    for i, (actual, expected) in enumerate(zip(shape, expect_shape)):
      if expected is not None and actual != expected:
        raise ValueError(
            f'{message or ""}: Expected shape {expect_shape}, but found {shape}'
            f' (mismatch at index {i}).'
        )
  return shape
