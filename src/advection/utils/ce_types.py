"""Common types for Contrails attribution open-sourcing."""

from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr

Array = Any
ArrayLike = Any
NPDtype = Any
TimeType = Union[pd.Timestamp, np.datetime64, str]
XData = Union[xr.Dataset, xr.DataArray]
