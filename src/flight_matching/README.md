# Flight Matching Library for Contrail Modeling

This library implements single-frame flight-contrail matching as outlined in
[Geraedts _et al_ 2024](https://iopscience.iop.org/article/10.1088/2515-7620/ad11ab),
enabling users to match flights to contrails.

Multiframe matching is also planned to be implemented.

## Overview

The flight matching process assigns match scores between any arbitrary flight-
contrail pairing, based on a number of heuristic evaluations which account for
flight advection age, wind offset and measurement error.

The library is built on top of **JAX**, allowing it to:

-   Run in a highly vectorized manner, processing thousands of parcels in
    parallel.
-   Execute seamlessly on **CPU, GPU, or TPU** hardware.
-   Support **automatic differentiation**, which is useful for sensitivity
    analysis or optimization tasks.

## Quick Start

See the attached notebook for implementation and usage details.