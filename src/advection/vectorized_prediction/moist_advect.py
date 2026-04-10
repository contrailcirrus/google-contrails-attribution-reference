"""Fluid mechanics utility code for moist advection."""

import math
from typing import Any, Union

import numpy as np

# Typings
FloatOrArray = Union[float, np.ndarray]

# constants
# https://en.wikipedia.org/wiki/Density_of_air
MOLAR_MASS_AIR = 0.0289652
IDEAL_GAS_CONSTANT = 8.314462
BOLTZMANN_CONSTANT = 1.38e-23
AVOGADROS_NUMBER = 6.022e23
ICE_DENSITY = 971  # density of ice, kg/m^3
G = 9.81  # gravitational acceleration


def lennard_jones_viscosity(temperature: FloatOrArray) -> np.ndarray:
  """Compute dynamic viscosity of air at a temperature.

  Based on the Lennard Jones molecular interaction force field.

  https://en.wikipedia.org/wiki/Temperature_dependence_of_viscosity

  Parameters
  ----------
  temperature : FloatOrArray
      Temperature in Kelvin.

  Returns
  -------
  np.ndarray
      Viscosity of air in Pa-s.
  """
  temperature = np.asarray(temperature)
  if np.any(temperature < 0.0):
    raise ValueError('Temperature must be non-negative.')
  kb = BOLTZMANN_CONSTANT
  m = MOLAR_MASS_AIR / AVOGADROS_NUMBER
  sigma = 3.617e-10  # radius of air particles
  scaled_temp = temperature / 97.0
  scale = 5.0 / 16.0 / math.sqrt(math.pi) * np.sqrt(m * kb) / sigma**2
  denom = 1.16145 * scaled_temp**-0.14874
  denom += 0.52487 * np.exp(-0.77320 * scaled_temp)
  denom += 2.16178 * np.exp(-2.43787 * scaled_temp)
  return scale * np.sqrt(temperature) / denom


def creeping_terminal_velocity(
    radius: FloatOrArray,
    viscosity: FloatOrArray,
    particle_density: FloatOrArray,
) -> FloatOrArray:
  """Terminal velocity of spherical particle in low Reynolds number limit.

  Based on Stokes' law, valid for Re < ~0.25.
  https://en.wikipedia.org/wiki/Stokes%27_law

  Parameters
  ----------
  radius : FloatOrArray
      Radius of particle (m).
  viscosity : FloatOrArray
      Dynamic viscosity of air (Pa-s).
  particle_density : FloatOrArray
      Density of particle (kg/m^3).

  Returns
  -------
  FloatOrArray
      Terminal velocity (m/s).
  """
  return 2 * G * radius * radius * particle_density / (9 * viscosity)


def air_density(
    pressure: FloatOrArray,
    temperature: FloatOrArray,
) -> FloatOrArray:
  """Density of air according to ideal gas law.

  https://en.wikipedia.org/wiki/Density_of_air

  Parameters
  ----------
  pressure : FloatOrArray
      Pressure in Pa.
  temperature : FloatOrArray
      Temperature in Kelvin.

  Returns
  -------
  FloatOrArray
      Density in kg/m^3.
  """
  return pressure * MOLAR_MASS_AIR / (temperature * IDEAL_GAS_CONSTANT)


def reynolds_number(
    rho: FloatOrArray,
    velocity: FloatOrArray,
    radius: FloatOrArray,
    viscosity: FloatOrArray,
) -> FloatOrArray:
  """Reynolds number of sphere in fluid.

  Parameters
  ----------
  rho : FloatOrArray
      Air density in kg/m^3.
  velocity : FloatOrArray
      Velocity in m/s, relative to particle.
  radius : FloatOrArray
      Radius in m, of particle.
  viscosity : FloatOrArray
      Viscosity in Pa-s, of air.

  Returns
  -------
  FloatOrArray
      Reynolds number (dimensionless), measure of turbulent flow.
  """
  return rho * velocity * 2 * radius / viscosity


def diameter_from_reynolds(
    reynolds: FloatOrArray,
    viscosity: FloatOrArray,
    particle_density: FloatOrArray,
    rho: FloatOrArray,
) -> FloatOrArray:
  """Diameter of sphere that attains a certain Reynolds number.

  Note this formula only works when particle_density >> air_density.
  This can be derived by combining the above formulas for creeping terminal
  velocity and reynolds number.

  Parameters
  ----------
  reynolds : FloatOrArray
      Reynolds number.
  viscosity : FloatOrArray
      Viscosity in Pa-s, of air.
  particle_density : FloatOrArray
      Particle density in kg/m^3.
  rho : FloatOrArray
      Air density in kg/m^3.

  Returns
  -------
  FloatOrArray
      Diameter of particle, in m.
  """
  diameter_cubed = (
      18 * reynolds * viscosity * viscosity / (G * rho * particle_density)
  )
  diameter_cubed = np.maximum(0, diameter_cubed)
  return diameter_cubed ** (1 / 3)


def terminal_velocity(
    radius: FloatOrArray,
    pressure: FloatOrArray,
    temperature: FloatOrArray,
    particle_density: FloatOrArray,
) -> FloatOrArray:
  """Compute terminal velocity of small particle in air.

  Parameters
  ----------
  radius : FloatOrArray
      Radius in m, of particle.
  pressure : FloatOrArray
      Pressure in Pa, of air.
  temperature : FloatOrArray
      Temperature in K, of air.
  particle_density : FloatOrArray
      Particle density in kg/m^3.

  Returns
  -------
  FloatOrArray
      Terminal velocity in m/s.
  """
  # First compute terminal velocity in low reynolds number limit
  viscosity = lennard_jones_viscosity(temperature)
  creeping_velocity = creeping_terminal_velocity(
      radius, viscosity, particle_density
  )
  rho = air_density(pressure, temperature)
  # Given that terminal velocity, is the Re low enough to be valid?
  reynolds = reynolds_number(rho, creeping_velocity, radius, viscosity)

  # Where reynolds < 0.25, we will return creeping_velocity.
  # Otherwise assume terminal velocity grows linearly with radius
  # beyond critical diameter (where Re=0.25)
  critical_diameter = diameter_from_reynolds(
      0.25, viscosity, particle_density, rho
  )
  critical_velocity = creeping_terminal_velocity(
      critical_diameter / 2, viscosity, particle_density
  )
  slope = 2 / critical_diameter * critical_velocity
  v = (2 * radius - critical_diameter) * slope + critical_velocity
  return np.where(reynolds <= 0.25, creeping_velocity, v)


def crystal_radius(time_sec: FloatOrArray) -> np.ndarray:
  """Function that emulates CoCIP ice crystal radius growth.

  Function comes from fitting a quadratic to the
  log(crystal growth)/log(contrail age) data shown in bottom left of
  Figure 7 of Schumann's CoCIP paper
  https://gmd.copernicus.org/articles/5/543/2012/gmd-5-543-2012.pdf

  We assumed that the spread in crystal size was -3/+3 sigma, which yielded
  an estimate of the mean and standard deviation. User supplies z.

  NOTE: This function should only be invoked when relative humidity > 100%

  Parameters
  ----------
  time_sec : FloatOrArray
      Time since contrail formation.

  Returns
  -------
  np.ndarray
      Ice crystal radius in m.
  """
  z = 0.0
  time_hr = (time_sec + 60) / 3600.0  # avoid log(0)
  log_time = np.log(time_hr)
  mean_coef = [0.61185313, 0.08065007]
  mean_intercept = 1.48711988
  std_coef = [0.16493902, 0.0255061]
  std_intercept = 0.66280488
  coef_0 = mean_coef[0] + (z * std_coef[0])
  coef_1 = mean_coef[1] + (z * std_coef[1])
  intercept = mean_intercept + (z * std_intercept)
  return (
      np.exp(log_time * coef_0 + log_time * log_time * coef_1 + intercept)
      * 1e-6
  )


def dp_correction(
    pressure_pa: Any,
    temperature: Any,
    t: Any,
    t0: Any,
) -> np.ndarray:
  """Term to add to pressure derivate for moist advection.

  Parameters
  ----------
  pressure_pa : Any
      Pressure in Pascals.
  temperature : Any
      Temperature in Kelvin.
  t : Any
      Current time.
  t0 : Any
      Crystal formation time.

  Returns
  -------
  np.ndarray
      Term to add to dry advect pressure derivative (w.r.t time in seconds).
      Has units of Pa / sec.
  """
  pressure_pa = np.asarray(pressure_pa)
  temperature = np.asarray(temperature)
  t = np.asarray(t)
  t0 = np.asarray(t0)

  density = air_density(pressure_pa, temperature)

  radius = crystal_radius(t - t0)
  velocity = terminal_velocity(radius, pressure_pa, temperature, ICE_DENSITY)
  return np.asarray(G * density * velocity)
