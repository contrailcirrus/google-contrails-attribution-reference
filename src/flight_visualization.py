"""Utilities for visualizing flights data."""

import plotly.graph_objects as go

from pycontrails import Flight

def plot_flight_on_globe(flight: Flight):
    """Plot a pycontrails Flight object on a 3D Plotly globe centered and zoomed on the trajectory."""
    df = flight.dataframe
    # Access flight_id from the attributes dictionary
    fid = flight.attrs.get("flight_id", "Unknown")

    # Calculate center point for the camera
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    # Calculate the spread to determine zoom level
    lat_range = df["latitude"].max() - df["latitude"].min()
    lon_range = df["longitude"].max() - df["longitude"].min()
    max_range = max(lat_range, lon_range, 0.1)

    # Heuristic for projection scale: 1.0 is the full globe (~180 degrees)
    # We scale such that the trajectory occupies a significant part of the frame.
    # Adjusted from 120.0 to 100.0 to zoom out slightly more.
    zoom_scale = 1.0 / (max_range / 100.0)
    zoom_scale = max(1.0, min(zoom_scale, 20.0))  # Limit zoom to stay within reasonable bounds

    fig = go.Figure()

    # Add the flight path
    fig.add_trace(
        go.Scattergeo(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="lines+markers",
            line=dict(width=2, color="red"),
            marker=dict(size=5, color="blue"),
            name=f"Flight {fid}",
            hovertext=df["time"].dt.strftime("%H:%M:%S"),
        )
    )

    # Configure the globe layout and center/zoom it
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=center_lon, lat=center_lat, roll=0),
        projection_scale=zoom_scale,
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="#E5ECF6",
        showocean=True,
        oceancolor="#f9f9f9",
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
    )

    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title=f"Trajectory for Flight ID: {fid} (Zoom: {zoom_scale:.3f}x)",
    )

    fig.show()