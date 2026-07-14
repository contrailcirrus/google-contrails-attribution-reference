"""Utils for Flight Attribution Data Fetching and Processing"""

import asyncio
import io
import time
from datetime import date, datetime, timedelta

import aiohttp
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pycontrails import Flight
from pycontrails.core import flight

_API_BASE_URL = "https://api.contrails.org/v1/observations/google/geostationary/L4/attributions"

def generate_flight_identifier(
    carrier: str,
    flight_number: str,
    departure_date: str,
    depature_airport_iata: str,
) -> str:
    """Generate a flight identifier string given various flight details.

    Parameters
    ----------
    carrier : datetime
        The 2-character IATA airline designator (case-sensitive).
    flight_number : datetime
        The numeric portion only (no carrier prefix).
    departure_date : str
        The departure date of the flight, formatted in form 'YYYY-MM-DD'.
    departure_airport_iata : int
        The 3-character IATA code (case-sensitive).

    Returns
    -------
    str
        The generated flight identifier.
    """
    if len(carrier) != 2:
        raise ValueError("Carrier must be a two-character IATA airline designator.")
    if len(depature_airport_iata) != 3:
        raise ValueError("Departure airport IATA must be a three-character IATA departure airport code.")
    return f'{carrier}~{flight_number}~{departure_date}~{depature_airport_iata}'

async def fetch_flight_attribution(flight_identifier: str, api_key: str):
    """Fetch flight attribution for the flight in question."""

    headers = {"x-api-key": api_key}
    params = {"flight": [flight_identifier]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(_API_BASE_URL, headers=headers, params=params) as response:
                response.raise_for_status()
                print(response.content)
                content = await response.read()
                if not content:
                    print(f"No content received for {flight_identifier}")
                    return None
                return content
    except aiohttp.ClientError as e:
        print(f"Error fetching data for {flight_identifier}: {e}")
        return None
    except Exception as e:
        print(f"Error processing data for {flight_identifier}: {e}")
        return None