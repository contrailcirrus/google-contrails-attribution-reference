"""Utils for ADS-B Data Fetching and Processing"""

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

_API_BASE_URL = "https://api.contrails.org/v1/adsb/telemetry"

def generate_flight_id(
    start_timestamp: datetime,
    end_timestamp: datetime,
    icao_address: str,
    midnight_threshold_mins: int,
) -> str:
    """Generate a flight ID for a flight waypoint based on timestamp and ICAO address.

    Flight IDs are generated based on the flight's start and end timestamps and its
    ICAO address. All IDs are prefixed with SPIRE-INFERRED-{icao_address}-.
    The rest of the ID depends on the time of day:

        1. Midnight Rollover/Holdover: Special formatting is applied if the flight
        period crosses midnight within a certain threshold (midnight_threshold_mins).
            * If the flight ends just after midnight (a "holdover"), the ID includes
            the dates of the day before the start and the start date, formatted as:
            {start_date - 1 day}-rollover-{start_date}.
            * If the flight starts just before midnight (a "rollover"), the ID
            includes the start date and the day after the end date, formatted as:
            {start_date}-rollover-{end_date + 1 day}.

        2. Standard: If the flight period doesn't cross the midnight threshold, the
        ID is generated using the Unix timestamp (in seconds) of the start and end
        times: {int(start_timestamp)}-{int(end_timestamp)}.

    Parameters
    ----------
    start_timestamp : datetime
        The start timestamp of the waypoint group.
    end_timestamp : datetime
        The end timestamp of the waypoint group.
    icao_address : str
        The ICAO address of the flight.
    midnight_threshold_mins : int
        The number of minutes before/after midnight to consider for generating
        a rollover/holdover ID.

    Returns
    -------
    str
        The generated flight ID.

    Examples
    --------
    Holdover: SPIRE-INFERRED-ABC123-2026-02-03-rollover-2026-02-04
    Rollover: SPIRE-INFERRED-ABC123-2026-02-04-rollover-2026-02-05
    Standard: SPIRE-INFERRED-ABC123-1760035200-1760042400
    """
    is_rollover = (
        start_timestamp.time()
        >= (pd.to_datetime("23:59:59") - pd.Timedelta(minutes=midnight_threshold_mins)).time()
    )
    is_holdover = (
        end_timestamp.time()
        <= (pd.to_datetime("00:00:00") + pd.Timedelta(minutes=midnight_threshold_mins)).time()
    )
    if is_holdover:
        generated_id = (
            f"SPIRE-INFERRED-{icao_address}-"
            f"{start_timestamp.date() - pd.Timedelta(days=1)}-rollover-"
            f"{start_timestamp.date()}"
        )
    elif is_rollover:
        generated_id = (
            f"SPIRE-INFERRED-{icao_address}-"
            f"{start_timestamp.date()}-rollover-"
            f"{end_timestamp.date() + pd.Timedelta(days=1)}"
        )
    else:
        generated_id = (
            f"SPIRE-INFERRED-{icao_address}-"
            f"{int(start_timestamp.timestamp())}-{int(end_timestamp.timestamp())}"
        )

    return generated_id


async def fetch_adsb_data_hour(
    session: aiohttp.ClientSession, dt_hour: datetime, api_key: str
) -> pd.DataFrame | None:
    """Asynchronously fetch ADS-B data for a single hour."""
    headers = {"accept": "application/vnd.apache.parquet", "x-api-key": api_key}
    # The /telemetry endpoint uses 'date' param for the start of the hour
    params = {"date": dt_hour.strftime("%Y-%m-%dT%H")}

    try:
        async with session.get(_API_BASE_URL, headers=headers, params=params) as response:
            response.raise_for_status()
            content = await response.read()
            if not content:
                print(f"No content received for {dt_hour}")
                return None
            # Load Parquet from response content
            return pd.read_parquet(io.BytesIO(content))
    except aiohttp.ClientError as e:
        print(f"Error fetching data for {dt_hour}: {e}")
        return None
    except Exception as e:
        print(f"Error processing data for {dt_hour}: {e}")
        return None

async def fetch_all_day_data(target_date: date, api_key: str) -> pd.DataFrame:
    """Fetch ADS-B data for the entire day asynchronously."""
    start_datetime = datetime(target_date.year, target_date.month, target_date.day)
    tasks = []

    async with aiohttp.ClientSession() as session:
        for hour in range(24):
            dt_hour = start_datetime + timedelta(hours=hour)
            tasks.append(fetch_adsb_data_hour(session, dt_hour, api_key))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    dataframes = []
    total_resp_size = 0
    for res in results:
        if isinstance(res, pd.DataFrame) and not res.empty:
            dataframes.append(res)
            total_resp_size += res.memory_usage(deep=True).sum()
        elif isinstance(res, Exception):
            print(f"An exception occurred during fetch: {res}")

    if not dataframes:
        raise ValueError("No data fetched. Check API key and date range.")

    print(f"Total response size: {round(total_resp_size / 1000000, 2)} MB")
    return pd.concat(dataframes, ignore_index=True)

def clean_adsb_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the raw ADS-B DataFrame."""
    if df.empty:
        return df
    df = df.copy()

    # Rename columns to match pycontrails expectations
    # The API returns 'altitude_baro', 'icao_address', and 'timestamp'
    # Pycontrails expects 'altitude', 'icao', and 'time'
    rename_map = {
        "altitude_baro": "altitude",
        "icao_address": "icao",
        "timestamp": "time",
    }
    df = df.rename(columns=rename_map)

    # Ensure time is datetime object
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Select necessary columns
    columns = [
        "time",
        "latitude",
        "longitude",
        "altitude",
        "icao",
        "flight_id",
        "tail_number",
        "collection_type",
    ]
    # Keep only columns that exist in the dataframe
    return df[df.columns.intersection(columns)]


def impute_flight_ids(
    df: pd.DataFrame,
    time_threshold_mins: int = 20,
    midnight_threshold_mins: int = 20,
) -> pd.DataFrame:
    """Impute missing flight IDs for ADS-B waypoints.

    There is no guarantee that incoming waypoints have a flight ID. To properly
    group waypoints into flights we want to ensure all waypoints have flight ID.

    We thus impute missing flight ID waypoints following these methods:

        - First, group waypoints w/ missing flight IDs by ICAO address and timestamp.
        Each waypoint in the group is at most [waypoint_ground_threshold_mins]
        minutes from its temporal neighbor.

        - For each group, check for a terrestrial waypoint that has a flight ID within
        [time_threshold_mins] minutes. If one such waypoint exists, backfill/
        forward-fill flight ID using that waypoint.
        If multiple qualifying waypoints exist *before* and *after* the group, use
        the one that's temporally closest.

        - Otherwise, impute a custom flight ID based on the group's start + end
        timestamp and the ICAO address.

    Parameters
    ----------
    df : pd.DataFrame
        Flight formatted dataframe of waypoints
    time_threshold_mins : int
        A group of NULL-flight ID waypoints is considered
        temporally "linked" to a waypoint with a flight ID if the latter is
        within `time_threshold_mins` of the former.
    midnight_threshold_mins : int
        If a group of NULL-flight ID waypoints *starts* or
        *ends* within `midnight_threshold_mins` of midnight, *and* an inferred ID
        has to be generated (rather than derived from waypoints with flight IDs),
        use a custom 'rollover' flight ID rather than deriving from timestamps.

    Returns
    -------
    pd.Dataframe
        A pandas DataFrame with missing flight IDs imputed.
    """
    # Early return if there are no missing flight IDs (no need to impute).
    if not df["flight_id"].isna().any():
        return df

    # Drop:
    # - Duplicated waypoints
    # - Waypoints without an ICAO address or timestamp (precautionary step).
    df_imputed = df.copy()
    df_imputed = df_imputed.drop_duplicates()
    df_imputed = df_imputed.dropna(subset=["icao_address", "timestamp"])
    df_imputed = df_imputed.reset_index()

    df_missing = df_imputed[df_imputed["flight_id"].isna()].sort_values(
        ["icao_address", "timestamp"]
    )
    df_valid = df_imputed[~df_imputed["flight_id"].isna()].sort_values("timestamp")

    # Group missing flight ID waypoints by ICAO and timestamps.
    is_new_group = (df_missing["icao_address"] != df_missing["icao_address"].shift()) | (
        df_missing["timestamp"].diff() > pd.Timedelta(minutes=20)
    )
    df_missing["group_id"] = is_new_group.cumsum()
    groups = (
        df_missing.groupby("group_id")
        .agg(
            icao_address=("icao_address", "first"),
            group_start=("timestamp", "min"),
            group_end=("timestamp", "max"),
        )
        .reset_index()
    )

    # Search for waypoints with a non-null ID that are:
    # - In between the start/end timestamp of the group.
    # - Within time_threshold_mins of the start/end of the group.
    start_group = groups.sort_values("group_start")
    internal_match = pd.merge_asof(
        start_group,
        df_valid[["timestamp", "flight_id", "icao_address"]],
        left_on="group_start",
        right_on="timestamp",
        by="icao_address",
        direction="forward",
        suffixes=("", "_internal"),
    )
    backward_match = pd.merge_asof(
        start_group,
        df_valid[["timestamp", "flight_id", "icao_address"]],
        left_on="group_start",
        right_on="timestamp",
        by="icao_address",
        direction="backward",
        suffixes=("", "_prev"),
    )
    forward_match = pd.merge_asof(
        groups.sort_values("group_end"),
        df_valid[["timestamp", "flight_id", "icao_address"]],
        left_on="group_end",
        right_on="timestamp",
        by="icao_address",
        direction="forward",
        suffixes=("", "_next"),
    )

    # Align all merges on group_id for comparison.
    groups = groups.set_index("group_id")
    internal_match = internal_match.set_index("group_id")
    backward_match = backward_match.set_index("group_id")
    forward_match = forward_match.set_index("group_id")

    groups["internal_id"] = internal_match["flight_id"]
    groups["internal_ts"] = internal_match["timestamp"]
    groups["prev_id"] = backward_match["flight_id"]
    groups["prev_ts"] = backward_match["timestamp"]
    groups["next_id"] = forward_match["flight_id"]
    groups["next_ts"] = forward_match["timestamp"]

    # Check if a group has a waypoint inside its start/end timestamp range with
    # a valid flight ID (since if it does, we want to use it).
    has_internal = groups["internal_ts"] <= groups["group_end"]

    groups["dist_prev"] = (groups["group_start"] - groups["prev_ts"]).abs()
    groups["dist_next"] = (groups["next_ts"] - groups["group_end"]).abs()
    groups["valid_prev"] = groups["dist_prev"] <= pd.Timedelta(minutes=time_threshold_mins)
    groups["valid_next"] = groups["dist_next"] <= pd.Timedelta(minutes=time_threshold_mins)

    # Begin assigning flight IDs.
    groups["final_flight_id"] = pd.Series(np.nan, index=groups.index, dtype=object)
    groups.loc[has_internal, "final_flight_id"] = groups.loc[has_internal, "internal_id"]
    # If no internal IDs, use the chronologically closest ID within 20m of the
    # start/end range.
    missing_mask = groups["final_flight_id"].isna()

    if missing_mask.any():
        sub = groups.loc[missing_mask]
        use_prev = sub["valid_prev"] & (~sub["valid_next"] | (sub["dist_prev"] <= sub["dist_next"]))
        use_next = sub["valid_next"] & (~sub["valid_prev"] | (sub["dist_next"] < sub["dist_prev"]))
        groups.loc[sub.index[use_prev], "final_flight_id"] = sub.loc[use_prev, "prev_id"]
        groups.loc[sub.index[use_next], "final_flight_id"] = sub.loc[use_next, "next_id"]

    # For groups that have neither internal waypoints nor waypoints within
    # time_threshold_mins with a non-null ID, generate an ID for the
    # group.
    needs_gen_mask = groups["final_flight_id"].isna()
    if needs_gen_mask.any():
        generated_ids = groups[needs_gen_mask].apply(
            lambda row: generate_flight_id(
                row["group_start"],
                row["group_end"],
                row["icao_address"],
                midnight_threshold_mins,
            ),
            axis=1,
        )
        groups.loc[needs_gen_mask, "final_flight_id"] = generated_ids
    df_missing["flight_id"] = df_missing["group_id"].map(groups["final_flight_id"])
    df_imputed.loc[df_missing.index, "flight_id"] = df_missing["flight_id"]

    num_not_imputed = len(df_imputed[df_imputed["flight_id"].isna()])
    if num_not_imputed > 0:
        raise ValueError(
            f"Unexpected error; {num_not_imputed} waypoint(s) without flight IDs"
            " were not imputed despite running impute_flight_ids."
        )
    return df_imputed
