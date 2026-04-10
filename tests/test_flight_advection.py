"""Tests for advection functions interacting with `Flight` objects."""

from absl.testing import absltest
import numpy as np
import pandas as pd
from pycontrails.core import flight
import xarray as xr

from src.advection.vectorized_prediction import advection


Flight = flight.Flight


class FlightAdvectionTest(absltest.TestCase):
  """Tests for advection functions interacting with `Flight` objects."""

  def test_flight_roundtrip(self):
    """Tests converting a Flight to AdvectionState and back."""
    df = pd.DataFrame({
        "longitude": [0, 1],
        "latitude": [0, 1],
        "altitude": [10000, 10000],
        "time": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:01:00"]),
        "extra_col": ["a", "b"],
    })
    fl = Flight(data=df, flight_id="test_flight")

    state = advection.AdvectionState.from_flight(fl)
    self.assertEqual(state.n_parcels, 2)
    self.assertEqual(state.pressure.shape, (2,))

    fl_back = state.to_flight(source_flight=fl)
    self.assertIsInstance(fl_back, Flight)
    self.assertEqual(fl_back.attrs["flight_id"], "test_flight")
    self.assertIn("extra_col", fl_back.data)
    self.assertEqual(list(fl_back.data["extra_col"]), ["a", "b"])

  def test_advection_flight(self):
    """Tests advecting a Flight object.

    Checks the output structure and values.
    """
    # Simple 2x2x2x2 weather
    # times in hours since 1970-01-01
    times = np.array([0.0, 10.0])
    levels = np.array([100.0, 1000.0])
    lats = np.array([-80.0, 80.0])
    lons = np.array([-180.0, 180.0])

    weather_ds = xr.Dataset(
        {
            "u": (
                ["time", "level", "latitude", "longitude"],
                np.ones((2, 2, 2, 2)) * 10.0,
            ),  # 10 m/s East
            "v": (
                ["time", "level", "latitude", "longitude"],
                np.zeros((2, 2, 2, 2)),
            ),
            "w": (
                ["time", "level", "latitude", "longitude"],
                np.zeros((2, 2, 2, 2)),
            ),
        },
        coords={
            "time": times,
            "level": levels,
            "latitude": lats,
            "longitude": lons,
        },
    )

    advector = advection.LagrangianAdvector(weather_ds)

    df = pd.DataFrame({
        "longitude": [0.0],
        "latitude": [0.0],
        "level": [250.0],
        "time": [pd.Timestamp("1970-01-01", tz="UTC")],
        "metadata": ["keep_me"],
    })
    fl = Flight(data=df)

    # Advect for 1 hour
    t_eval = pd.to_datetime(["1970-01-01 01:00:00"], utc=True)
    out_fl = advector.advect(fl, t_eval)

    self.assertIsInstance(out_fl, Flight)
    # 2 waypoints in output (initial + eval) because of advect's logic to
    # include start in evaluations if 1D.
    self.len_out_fl = len(out_fl)
    self.assertEqual(self.len_out_fl, 2)
    self.assertIn("metadata", out_fl.data)
    self.assertEqual(out_fl.data["metadata"][0], "keep_me")
    self.assertEqual(out_fl.data["metadata"][1], "keep_me")

    # Check longitude change (10 m/s * 3600 s = 36000 m = 36 km)
    # At equator 1 deg lon approx 111.32 km. So 36/111.32 approx 0.323 deg.
    self.assertLess(0.3, out_fl.data["longitude"][1])
    self.assertLess(out_fl.data["longitude"][1], 0.4)


if __name__ == "__main__":
  absltest.main()
