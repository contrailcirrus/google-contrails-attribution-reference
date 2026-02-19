import datetime
import math
from typing import Any
from unittest import mock

import pandas as pd
import pytest

from pycontrails import Flight
from src import adsb

##########
# Fixtures
##########


@pytest.fixture(scope="module")
def adsb_waypoints() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": [datetime.datetime(2025, 1, 24, 0, 0, 0, 0)],
            "latitude": [32.405804],
            "longitude": [-110.98243],
            "collection_type": ["terrestrial"],
            "altitude_baro": [15675],
            "altitude_gnss": [math.nan],
            "icao_address": ["A00537"],
            "flight_id": ["0e781b4b-4ae6-4a7d-ba58-9dab71185127"],
            "callsign": ["N100FF"],
            "tail_number": ["N100FF"],
            "flight_number": [None],
            "aircraft_type_icao": ["E50P"],
            "airline_iata": [None],
            "departure_airport_icao": ["KTUS"],
            "departure_scheduled_time": [None],
            "arrival_airport_icao": ["KDVT"],
            "arrival_scheduled_time": [None],
            "nic": [None],
            "nacp": [12],
        }
    )
    # In practice Contrails.org's timestamp is in microseconds, not nanoseconds.
    df["timestamp"] = df["timestamp"].astype("datetime64[us]")
    df["latitude"] = df["latitude"].astype("float64")
    df["longitude"] = df["longitude"].astype("float64")
    return df


##########
# Tests
##########


def test_does_not_impute_if_all_ids_filled_out(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    new_row = df.iloc[0].copy()
    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
    with mock.patch.object(adsb, "generate_flight_id", autospec=True) as mock_generate_flight_id:
        adsb.impute_flight_ids(df)
        mock_generate_flight_id.assert_not_called()


def test_impute_backfills_if_temporal_alignment(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    new_row = df.iloc[0].copy()
    new_row.collection_type = "satellite"
    new_row.flight_id = None
    new_row.timestamp -= pd.Timedelta(minutes=20)
    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)

    df = adsb.impute_flight_ids(df)

    assert df.iloc[0]["flight_id"] == df.iloc[1]["flight_id"]


def test_impute_handles_rollover_flights(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    df["collection_type"] = "satellite"
    df["flight_id"] = None
    df["timestamp"] = datetime.datetime(2025, 1, 24, 23, 59, 59, 0)

    df = adsb.impute_flight_ids(df)

    assert (
        df.iloc[0]["flight_id"]
        == f"SPIRE-INFERRED-{df.iloc[0]['icao_address']}-2025-01-24-rollover-2025-01-25"
    )


def test_imput_handles_rollover_missing_terrestrial_flight_id(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    df["collection_type"] = "terrestrial"
    df["flight_id"] = None
    df["timestamp"] = datetime.datetime(2025, 1, 24, 23, 59, 59, 0)

    df = adsb.impute_flight_ids(df)

    assert (
        df.iloc[0]["flight_id"]
        == f"SPIRE-INFERRED-{df.iloc[0]['icao_address']}-2025-01-24-rollover-2025-01-25"
    )


def test_impute_handles_holdover_flights(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    df["collection_type"] = "satellite"
    df["flight_id"] = None
    df["timestamp"] = datetime.datetime(2025, 1, 24, 0, 0, 1, 0)

    df = adsb.impute_flight_ids(df)

    assert (
        df.iloc[0]["flight_id"]
        == f"SPIRE-INFERRED-{df.iloc[0]['icao_address']}-2025-01-23-rollover-2025-01-24"
    )


def test_impute_generates_on_icao_and_timestamp(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    df = pd.concat([pd.DataFrame([df.iloc[0].copy()]), df], ignore_index=True)
    df["collection_type"] = "satellite"
    df["flight_id"] = None
    df.loc[1, "timestamp"] += pd.Timedelta(minutes=15)

    df = adsb.impute_flight_ids(df)

    assert df.iloc[0]["flight_id"] == df.iloc[1]["flight_id"]
    assert (
        df.iloc[1]["flight_id"]
        == f"SPIRE-INFERRED-{df.iloc[0]['icao_address']}-2025-01-23-rollover-2025-01-24"
    )


def test_impute_generates_if_no_temporal_alignment(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    new_row = df.iloc[0].copy()
    new_row.collection_type = "satellite"
    new_row.flight_id = None
    new_row.timestamp -= pd.Timedelta(hours=20)
    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)

    df = adsb.impute_flight_ids(df)

    assert (
        df.iloc[0]["flight_id"] == f"SPIRE-INFERRED-{df.iloc[0]['icao_address']}-"
        f"{int(new_row.timestamp.timestamp())}-"
        f"{int(new_row.timestamp.timestamp())}"
    )
    assert df.iloc[1]["flight_id"] == "0e781b4b-4ae6-4a7d-ba58-9dab71185127"


def test_impute_prioritizes_waypoints_inside_group_range(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    df = pd.concat(
        [
            pd.DataFrame(
                [
                    df.iloc[0].copy(),
                    df.iloc[0].copy(),
                    df.iloc[0].copy(),
                    df.iloc[0].copy(),
                    df.iloc[0].copy(),
                ]
            ),
            df,
        ],
        ignore_index=True,
    )
    df["collection_type"] = "satellite"
    df["flight_id"] = None
    df.loc[1, "timestamp"] += pd.Timedelta(minutes=15)
    df.loc[2, "timestamp"] += pd.Timedelta(minutes=30)
    df.loc[2, "collection_type"] = "terrestrial"
    df.loc[2, "flight_id"] = "0e781b4b-4ae6-4a7d-ba58-9dab71185127"
    df.loc[3, "timestamp"] += pd.Timedelta(minutes=45)
    df.loc[4, "timestamp"] += pd.Timedelta(minutes=60)
    df.loc[4, "collection_type"] = "terrestrial"
    df.loc[4, "flight_id"] = "should-not-be-used"
    df = adsb.impute_flight_ids(df)
    assert df.iloc[0]["flight_id"] == "0e781b4b-4ae6-4a7d-ba58-9dab71185127"
    assert df.iloc[0]["flight_id"] == df.iloc[1]["flight_id"]
    assert df.iloc[1]["flight_id"] == df.iloc[3]["flight_id"]


def test_impute_handles_multiple_distinct_satellite_groups(adsb_waypoints: pd.DataFrame) -> None:
    df = adsb_waypoints.copy()
    df = pd.concat(
        [
            pd.DataFrame(
                [
                    df.iloc[0].copy(),
                    df.iloc[0].copy(),
                    df.iloc[0].copy(),
                    df.iloc[0].copy(),
                    df.iloc[0].copy(),
                ]
            ),
            df,
        ],
        ignore_index=True,
    )
    df["collection_type"] = "satellite"
    df["flight_id"] = None
    # 0 and 1 should be generated due to temporal non-proximity to 2.
    df.loc[1, "timestamp"] += pd.Timedelta(minutes=15)
    # 2 is terrestrial and should remain untouched.
    df.loc[2, "collection_type"] = "terrestrial"
    df.loc[2, "flight_id"] = "0e781b4b-4ae6-4a7d-ba58-9dab71185127"
    df.loc[2, "timestamp"] += pd.Timedelta(minutes=60)
    # 3, 4 and 5 should be overwritten due to temporal proximity to 2.
    # The update type should not matter.
    df.loc[3, "timestamp"] += pd.Timedelta(minutes=80)
    df.loc[4, "timestamp"] += pd.Timedelta(minutes=90)
    df.loc[5, "timestamp"] += pd.Timedelta(minutes=100)
    df.loc[5, "collection_type"] = "terrestrial"

    df = adsb.impute_flight_ids(df)

    assert (
        df.iloc[0]["flight_id"]
        == f"SPIRE-INFERRED-{df.iloc[0]['icao_address']}-2025-01-23-rollover-2025-01-24"
    )
    assert df.iloc[0]["flight_id"] == df.iloc[1]["flight_id"]
    assert df.iloc[2]["flight_id"] == "0e781b4b-4ae6-4a7d-ba58-9dab71185127"
    assert df.iloc[2]["flight_id"] == df.iloc[3]["flight_id"]
    assert df.iloc[3]["flight_id"] == df.iloc[4]["flight_id"]
    assert df.iloc[4]["flight_id"] == df.iloc[5]["flight_id"]