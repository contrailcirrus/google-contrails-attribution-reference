# Advection Library for Contrail Modeling

This library provides a high-performance, vectorized implementation of
Lagrangian advection for atmospheric parcels, specifically optimized for
contrail modeling and prediction.

## Overview

Advection is the process by which properties of the atmosphere (such as heat,
moisture, or in our case, contrails) are transported by the bulk movement of the
air (wind).

This library follows a **Lagrangian approach**, meaning we track individual
parcels of air as they move through a 3D wind field over time. This is in
contrast to an **Eulerian approach**, which looks at a fixed grid and observes
how properties change at those fixed locations.

## Physics & Methodology

### 3D Wind Fields

The movement of a parcel is determined by three components of the wind:

-   **$u$(Eastward)**: Movement along the longitudinal axis.
-   **$v$ (Northward)**: Movement along the latitudinal axis.
-   **$w$ (Vertical)**: Movement between different atmospheric pressure levels
    (expressed in Pascals per second, $Pa/s$).

### Moist Advection

In addition to transport by the wind, this library supports **Moist Advection**,
which accounts for **ice crystal sedimentation**.

Atmospheric parcels containing ice crystals (like contrails) do not strictly
follow the air's vertical velocity ($w$). Instead, they "sink" relative to the
surrounding air due to gravity. The effective vertical velocity ($w_{eff}$) is:
$$w_{eff} = w + w_{term}$$ where $w_{term}$ is the **terminal velocity** of the
ice crystals, calculated based on:

-   Air density and viscosity (derived from temperature and pressure).
-   Ice crystal size (modeled using the mean trajectory).
-   Local environmental conditions.

This correction is crucial for predicting the long-term vertical evolution and
eventual sublimation of contrails.

### Numerical Integration

To find the position of a parcel at a future (or past) time, we solve an
Ordinary Differential Equation (ODE): $$\frac{dy}{dt} = f(t, y)$$ where $y$ is
the state of the parcel (Position and Pressure).

Unlike simpler models that use fixed-step Euler integration, this library uses
**Higher-Order Runge-Kutta methods** (specifically `RK23` via
`scipy.solve_ivp`). This provides significant benefits:

-   **Adaptive Time-Stepping**: The solver automatically adjusts the step size
    to maintain a requested error tolerance.
-   **Precision**: Higher-order accuracy compared to linear (Euler)
    approximations.

## Mathematical Implementation

### Gaussian Coordinate System

Most geospatial libraries operate in Latitude and Longitude. However, Lat/Lon
coordinates have a **singularity at the poles** (the North and South poles),
which can cause numerical instability and "jagged" paths during global
advection.

Our library uses **Gaussian Coordinates** ($X, Y, Z$ on a sphere) for all
internal calculations, which:

-   Treats the Earth as a sphere (or ellipsoid).
-   Calculates parcel movement as a vector tangent to the sphere's surface.
-   Is robust at all latitudes, including the poles.

### Vectorized Execution with JAX

The library is built on top of **JAX**, allowing it to:

-   Run in a highly vectorized manner, processing thousands of parcels in
    parallel.
-   Execute seamlessly on **CPU, GPU, or TPU** hardware.
-   Support **automatic differentiation**, which is useful for sensitivity
    analysis or optimization tasks.

## Comparison with `pycontrails`

While both libraries serve the contrail community, they have different
architectural priorities:

| Feature                | This Library           | `pycontrails.DryAdvection` |
| :--------------------- | :--------------------- | :------------------------- |
| **Integrator**         | Higher-order           | Euler (Fixed Step)         |
:                        : Runge-Kutta (`RK23`)   :                            :
| **State                | Gaussian ($X, Y, Z$)   | Geographic (Lat/Lon)       |
: Representation**       :                        :                            :
| **Hardware Targets**   | CPU, GPU, TPU (via     | CPU (via NumPy)            |
:                        : JAX)                   :                            :
| **Numerical            | Robust at poles        | Potential issues at poles  |
: Stability**            :                        :                            :
| **Vertical Dimension** | $P$ (hPa) as a primary | Levels/Altitude segments   |
:                        : coordinate             :                            :
| **Moist Advection**    | Supported (Ice         | Limited/None               |
:                        : sedimentation)         :                            :

## Quick Start

### Basic Advection (Dry)

```python
from src.advection.vectorized_prediction import advection

# Create an initial state for your parcels
initial_state = advection.AdvectionState.from_time_pressure_lat_lon(
    time=0,
    pressure=250.0,  # hPa
    latitude=37.0,
    longitude=-122.0
)

# Initialize the advector with weather data (xarray Dataset with u, v, w)
advector = advection.LagrangianAdvector(weather_ds)

# Advect for 2 hours
final_state = advector.advect(initial_state, advection_length_hours=2.0)
```

### Moist Advection

To account for ice crystal sedimentation (particle sinking), set the
`use_moist_advection` flag to `True`.

```python
# Advect with moist advection (sinking parcels)
final_state = advector.advect(
    initial_state,
    advection_length_hours=2.0,
    use_moist_advection=True
)
```
