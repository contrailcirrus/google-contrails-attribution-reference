"""Tests for moist advection."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from pycontrails.core import flight
import xarray as xr

from src.advection.vectorized_prediction import advection
from src.advection.vectorized_prediction import moist_advect


class MoistAdvectionTest(parameterized.TestCase):
  """Tests for moist advection functionality."""

  def test_terminal_velocity(self):
    # Compare values with Figure 1 of:
    # "Terminal Velocity of Spherical Particles and Its Numerical
    # Representation"
    # by Michael G. Laurence
    v = moist_advect.terminal_velocity(
        radius=50e-6,  # 100 micron diameter
        pressure=101325,  # 1 atm
        temperature=293.15,  # 20 C
        particle_density=1000,  # water
    )
    self.assertAlmostEqual(float(v), 0.25, delta=0.05)

  def test_crystal_radius(self):
    r = moist_advect.crystal_radius(time_sec=3600)
    self.assertGreater(float(r), 0)
    self.assertLess(float(r), 1e-3)

  def test_lagrangian_advector_moist(self):
    """Tests that moist advection causes sinking due to terminal velocity."""
    # Create synthetic weather with temperature
    times = pd.date_range('2023-01-01', periods=3, freq='h')
    weather = xr.Dataset(
        data_vars={
            'u': (
                ('time', 'level', 'latitude', 'longitude'),
                np.zeros((3, 2, 2, 2)),
            ),
            'v': (
                ('time', 'level', 'latitude', 'longitude'),
                np.zeros((3, 2, 2, 2)),
            ),
            'w': (
                ('time', 'level', 'latitude', 'longitude'),
                np.zeros((3, 2, 2, 2)),
            ),
            't': (
                ('time', 'level', 'latitude', 'longitude'),
                np.full((3, 2, 2, 2), 220.0),
            ),
        },
        coords={
            'time': times,
            'level': [200, 250],
            'latitude': [0, 1],
            'longitude': [0, 1],
        },
    )

    advector = advection.LagrangianAdvector(weather)

    # Create a flight
    fl = flight.Flight(
        data={
            'time': [times[0].to_datetime64()],
            'latitude': [0.5],
            'longitude': [0.5],
            'level': [225],
        }
    )

    # Dry advection (w=0, should stay at same level)
    dry_result = advector.advect(fl, times[:2])
    self.assertAlmostEqual(float(dry_result.dataframe['level'].iloc[0]), 225.0)

    # Moist advection (should sink due to terminal velocity)
    moist_result = advector.advect(fl, times[:2], use_moist_advection=True)
    # Pressure should increase (sinking)
    self.assertGreater(float(moist_result.dataframe['level'].iloc[-1]), 225.0)


if __name__ == '__main__':
  absltest.main()
