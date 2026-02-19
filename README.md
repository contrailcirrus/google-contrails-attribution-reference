# Google Contrails Attributions Reference

## Overview

This repository contains the key components that generate contrail attributed flights. Attributions are composed of three parts:

1. **Flight loading:** Fetch and process flight trajectory data (ADS-B)
2. **Flight advection:** Advect flight waypoints forward in time using weather data
3. **Flight matching:** Match advected flight segments to observed contrails.

This release focuses on providing core library functions and demostration notebooks, rather than a fully operational pipeline. 

## Features

*   **Flight Data Handling:**
    *   Utilities to fetch ADS-B data from public sources (e.g., [contrails.org](https://contrails.org)).
    *   Functions to clean, segment, and group waypoints into individual flights.
    *   Algorithms for imputing missing flight identifiers.
*   **Flight Advection:**
    *   ETA: End of March 2026
    *   Integration with PyContrails data structures for weather
    *   Advection models to simulate the movement of flight segments along wind trajectories.
*   **Contrail Matching:**
    *   ETA: Summer/Fall 2026
    *   Performs flight matching over advected flight segments to a set of contrail detections.
 
<img width="909" height="403" alt="Screenshot 2026-02-19 at 1 25 48 PM" src="https://github.com/user-attachments/assets/58b4b6ff-7736-4b5d-b76f-4b1a52cb0bde" />


## Data Requirements

To run the notebooks and utilize the libraries, you will typically need:

*   **ECMWF Weather Data:** Accessible via PyContrails utilities. You may need a Climate Data Store (CDS) API key.
*   **Flight ADS-B Data:** Can be fetched using the provided tools/notebooks from sources like [contrails.org](https://contrails.org).
*   **Contrail Detections:** Sample contrail detection data will be provided for demonstration purposes in the notebooks once flight matching is available.

## Notebook Demonstrations

We provide Jupyter notebooks to illustrate the usage of the core components:

1.  **Flight Loading:** Demonstrates fetching ADS-B data, processing it, and preparing it for advection.
    *   `notebooks/01_flight_loading.ipynb`
2.  **Flight Advection:** Shows how to load weather data and advect flight paths.
    *   Notebook TBD
3.  **Flight to Contrail Matching:** Illustrates the process of matching advected flight segments with contrail observations.
    *   Notebook TBD

## Core Library Functions

The key functionalities are organized within the PyContrails library structure:

*   `src.adsb`: ADS-B data fetching and processing via contrails.org provided [ADS-B API](https://apidocs.contrails.org/notebooks/adsb_api.html).
*   Under construction: advection and flight matching


## Disclaimer

This open-source release provides libraries and examples showcasing the key components behind Google's attributions system, not the full [ContrailWatch](https://developers.google.com/contrails/v1/ContrailWatch-description) production pipeline. The aim is to foster research and transparency in contrail attribution methodology.

## Contributing

We welcome contributions! Please see the main [PyContrails contributing guide](https://py.contrails.org/contributing.html).

## License

This code is released under the Apache 2.0 License.
