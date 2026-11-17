"""Tests for single-frame flight-contrail matching."""

from absl.testing import absltest
from absl.testing import parameterized
import geojson
import numpy as np
import pandas as pd
from pycontrails.core import flight
import xarray as xr

from src.flight_matching import single_frame_matching as matching_lib
from src.flight_matching.utils import projection


class MatchingLibTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='vanilla', lon1=10.0, lon2=11.0, lon1_west=True),
      dict(testcase_name='negative', lon1=-10.0, lon2=-30.0, lon1_west=False),
      dict(testcase_name='dateline1', lon1=-179.0, lon2=179.0, lon1_west=False),
      dict(testcase_name='dateline2', lon1=170.0, lon2=-150.0, lon1_west=True),
  )
  def test_dateline_safe_west_east(self, lon1, lon2, lon1_west):
    west, east = matching_lib.dateline_safe_west_east(lon1, lon2)
    if lon1_west:
      self.assertAlmostEqual(west, lon1)
      self.assertAlmostEqual(east, lon2)
    else:
      self.assertAlmostEqual(west, lon2)
      self.assertAlmostEqual(east, lon1)

  @parameterized.named_parameters(
      dict(
          testcase_name='dateline_west_east',
          lon_tocheck=175.0,
          lon_reference=-175.0,
          expected=True,
      ),
      dict(
          testcase_name='dateline_east_west',
          lon_tocheck=-175.0,
          lon_reference=175.0,
          expected=False,
      ),
      dict(
          testcase_name='east_west_positive',
          lon_tocheck=175.0,
          lon_reference=5.0,
          expected=False,
      ),
      dict(
          testcase_name='east_west_negative',
          lon_tocheck=-5.0,
          lon_reference=-174.0,
          expected=False,
      ),
      dict(
          testcase_name='west_east',
          lon_tocheck=-4.0,
          lon_reference=175.0,
          expected=True,
      ),
  )
  def test_dateline_safe_is_west_check(
      self, lon_tocheck, lon_reference, expected
  ):
    got = matching_lib.dateline_safe_is_west_check(
        lon_tocheck=lon_tocheck, lon_reference=lon_reference
    )
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name='already_ordered',
          lon1=-80.0,
          lat1=25.0,
          lon2=-79.0,
          lat2=26.0,
          expected_lon1=-80.0,
          expected_lat1=25.0,
          expected_lon2=-79.0,
          expected_lat2=26.0,
      ),
      dict(
          testcase_name='reversed_endpoints',
          lon1=-79.0,
          lat1=26.0,
          lon2=-80.0,
          lat2=25.0,
          expected_lon1=-80.0,
          expected_lat1=25.0,
          expected_lon2=-79.0,
          expected_lat2=26.0,
      ),
      dict(
          testcase_name='dateline_reversed',
          lon1=-175.0,
          lat1=26.0,
          lon2=175.0,
          lat2=25.0,
          expected_lon1=175.0,
          expected_lat1=25.0,
          expected_lon2=-175.0,
          expected_lat2=26.0,
      ),
      dict(
          testcase_name='equal_longitude_no_swap',
          lon1=-80.0,
          lat1=26.0,
          lon2=-80.0,
          lat2=25.0,
          expected_lon1=-80.0,
          expected_lat1=26.0,
          expected_lon2=-80.0,
          expected_lat2=25.0,
      ),
  )
  def test_linear_contrail_orders_endpoints_west_to_east(
      self,
      lon1,
      lat1,
      lon2,
      lat2,
      expected_lon1,
      expected_lat1,
      expected_lon2,
      expected_lat2,
  ):
    contrail = matching_lib.LinearContrail(
        lon1=lon1,
        lat1=lat1,
        lon2=lon2,
        lat2=lat2,
        altitude_m=10000.0,
        timestamp=1672531200,
        satellite_longitude=-75.0,
    )
    self.assertEqual(contrail.lon1, expected_lon1)
    self.assertEqual(contrail.lat1, expected_lat1)
    self.assertEqual(contrail.lon2, expected_lon2)
    self.assertEqual(contrail.lat2, expected_lat2)

  @parameterized.named_parameters(
      dict(
          testcase_name='atlantic_ocean',
          lat=0.0,
          lng=0.0,
          expected_unit_gaussian=(1.0, 0.0, 0.0),
      ),
      dict(
          testcase_name='indian_ocean',
          lat=0.0,
          lng=90.0,
          expected_unit_gaussian=(0.0, 1.0, 0.0),
      ),
      dict(
          testcase_name='pacific_ocean',
          lat=0.0,
          lng=180.0,
          expected_unit_gaussian=(-1.0, 0.0, 0.0),
      ),
      dict(
          testcase_name='galapagos',
          lat=0.0,
          lng=-90.0,
          expected_unit_gaussian=(0.0, -1.0, 0.0),
      ),
      dict(
          testcase_name='north_pole',
          lat=90.0,
          lng=0.0,
          expected_unit_gaussian=(0.0, 0.0, 1.0),
      ),
      dict(
          testcase_name='south_pole',
          lat=-90.0,
          lng=0.0,
          expected_unit_gaussian=(0.0, 0.0, -1.0),
      ),
  )
  def test_to_gaussian(self, lat, lng, expected_unit_gaussian):
    np.testing.assert_allclose(
        matching_lib.to_gaussian(lat, lng),
        np.array(expected_unit_gaussian) * projection.EARTH_RADIUS_M,
        rtol=0.0,
        atol=1e-8,
    )

  def test_get_age_hrs(self):
    advected_flight = flight.Flight(
        data={
            'longitude': [0.0, 1.0, 2.0],
            'latitude': [0.0, 1.0, 2.0],
            'altitude': [10000.0, 10000.0, 10000.0],
            'time': pd.to_datetime([
                '2020-01-01 00:00:00',
                '2020-01-01 00:30:00',
                '2020-01-01 01:00:00',
            ]),
            'advected_time': pd.to_datetime([
                '2020-01-01 02:00:00',
                '2020-01-01 02:00:00',
                '2020-01-01 02:00:00',
            ]),
        }
    )

    age_hrs = matching_lib._get_age_hrs(advected_flight)

    np.testing.assert_allclose(age_hrs, [2.0, 1.5, 1.0])

  def test_get_age_hrs_missing_advected_time_raises(self):
    unadvected_flight = flight.Flight(
        data={
            'longitude': [0.0, 1.0],
            'latitude': [0.0, 1.0],
            'altitude': [10000.0, 10000.0],
            'time': pd.to_datetime([
                '2020-01-01 00:00:00',
                '2020-01-01 01:00:00',
            ]),
        }
    )

    with self.assertRaisesRegex(ValueError, 'Flight has not been advected.'):
      matching_lib._get_age_hrs(unadvected_flight)

  def test_advect_flight_to_time_advects_waypoints_to_target(self):
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
            'level': [200.0, 250.0],
            'latitude': [0.0, 1.0],
            'longitude': [0.0, 1.0],
        },
    )
    unadvected_flight = flight.Flight(
        data={
            'time': pd.to_datetime([
                '2023-01-01 00:00:00',
                '2023-01-01 00:30:00',
                '2023-01-01 01:00:00',
            ]),
            'latitude': [0.2, 0.4, 0.6],
            'longitude': [0.2, 0.4, 0.6],
            'level': [225.0, 225.0, 225.0],
        }
    )

    advected_flight = matching_lib.advect_flight_to_time(
        unadvected_flight, pd.Timestamp('2023-01-01 02:00:00'), weather
    )
    self.assertIn('advected_time', advected_flight.data)
    self.assertSameElements(
        [np.datetime64('2023-01-01 02:00:00')],
        advected_flight.data['advected_time'],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='with_altitude',
          coordinates=[(-80.0, 25.0, 10000.0), (-79.0, 26.0, 10000.0)],
          expected_altitude_m=10000.0,
      ),
      dict(
          testcase_name='without_altitude',
          coordinates=[(-80.0, 25.0), (-79.0, 26.0)],
          expected_altitude_m=None,
      ),
  )
  def test_convert_geojson(self, coordinates, expected_altitude_m):
    feature = geojson.Feature(
        geometry=geojson.LineString(coordinates),
        properties={
            'time': '2023-01-01T00:00:00Z',
            'satellite_origin': 'GOES-EAST-FULL-DISK',
        },
    )
    contrail = matching_lib.convert_geojson(feature)
    self.assertEqual(
        contrail,
        matching_lib.LinearContrail(
            lon1=-80.0,
            lat1=25.0,
            lon2=-79.0,
            lat2=26.0,
            altitude_m=expected_altitude_m,
            timestamp=1672531200,
            satellite_longitude=-75.0,
        ),
    )

  def test_convert_geojson_invalid_raises(self):
    invalid_feature = geojson.Feature(
        geometry=geojson.Point((-80.0, 25.0)),
        properties={'time': '2023-01-01T00:00:00Z'},
    )

    with self.assertRaisesRegex(ValueError, 'Ill-formatted GeoJSON object.'):
      matching_lib.convert_geojson(invalid_feature)

  def test_compute_segments_parallax_mismatch_raises(self):
    contrail = matching_lib.LinearContrail(
        lon1=-80.0,
        lat1=25.0,
        lon2=-79.0,
        lat2=26.0,
        altitude_m=10000.0,
        timestamp=1672531200,
        satellite_longitude=-75.0,
    )

    with self.assertRaisesRegex(ValueError, 'Parallax mismatch for contrail'):
      matching_lib._compute_segments([contrail], expected_longitude=0.0)

  @parameterized.named_parameters(
      dict(
          testcase_name='small_contrail',
          contrail_kwargs=dict(lon1=-75.01, lon2=-75.0),
          expected_reason='small contrail',
      ),
      dict(
          testcase_name='too_far_away',
          contrail_kwargs=dict(lat1=10.0, lat2=10.0),
          expected_reason='too far away',
      ),
      dict(
          testcase_name='no_overlap',
          contrail_kwargs=dict(lon1=-70.2, lon2=-69.8),
          expected_reason='no overlap',
      ),
      dict(
          testcase_name='young_average_parcel_age',
          flight_age_hrs=[0.05, 0.01],
          expected_reason='young average parcel age',
      ),
      dict(
          testcase_name='altitude_difference_too_large',
          contrail_kwargs=dict(altitude_m=12000.0),
          expected_reason='altitude difference larger than 1000.0m',
      ),
      dict(
          testcase_name='too_short_overlap',
          flight_times=['2023-01-01 00:00:00', '2023-01-01 00:00:30'],
          expected_reason='too short overlap',
      ),
      dict(
          testcase_name='flight_does_not_span_contrail',
          contrail_kwargs=dict(lon1=-76.0, lon2=-74.0),
          expected_reason="flight doesn't span contrail",
      ),
  )
  def test_validation_checks_rejects_invalid_contrails(
      self,
      expected_reason,
      contrail_kwargs=None,
      flight_age_hrs=None,
      flight_times=None,
  ):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
        weight_options=matching_lib.WeightOptions(
            altitude_difference_threshold_m=1000.0
        ),
    )
    test_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.0, 0.0]),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                flight_times or ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
        }
    )
    age_hrs = np.array(
        flight_age_hrs if flight_age_hrs is not None else [1.0, 0.9]
    )
    contrail_params = dict(
        lon1=-75.2,
        lat1=0.0,
        lon2=-74.8,
        lat2=0.0,
        altitude_m=10000.0,
        timestamp=1672531200,
        satellite_longitude=-75.0,
    )
    if contrail_kwargs:
      contrail_params.update(contrail_kwargs)
    contrail = matching_lib.LinearContrail(**contrail_params)

    result = matcher._validation_checks([contrail], test_flight, age_hrs)

    self.assertEmpty(result)
    self.assertEqual(matcher.counters, {expected_reason: 1})

  def test_validation_checks_returns_processed_matching_inputs(self):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
        weight_options=matching_lib.WeightOptions(
            altitude_difference_threshold_m=1000.0
        ),
    )
    test_flight = flight.Flight(
        data={
            'longitude': np.array([-76.0, -75.15, -74.805, -74.795]),
            'latitude': np.array([0.01, 0.02, -0.01, -0.02]),
            'altitude': np.array([10000.0, 10000.0, 10000.0, 10000.0]),
            'time': pd.to_datetime([
                '2023-01-01 00:00:00',
                '2023-01-01 00:05:00',
                '2023-01-01 00:10:00',
                '2023-01-01 00:15:00',
            ]),
        }
    )
    age_hrs = np.array([1.2, 1.1, 1.0, 0.9])
    contrail = matching_lib.LinearContrail(
        lon1=-75.2,
        lat1=0.0,
        lon2=-74.8,
        lat2=0.0,
        altitude_m=10000.0,
        timestamp=1672531200,
        satellite_longitude=-75.0,
    )

    results = matcher._validation_checks([contrail], test_flight, age_hrs)

    self.assertEmpty(matcher.counters)
    self.assertLen(results, 1)
    result = results[0]
    self.assertEqual(result.contrail, contrail)
    expected_segment = matching_lib.lat_lngs_to_segment(
        contrail.lat1, contrail.lon1, contrail.lat2, contrail.lon2
    )
    gaussians = matcher._compute_gaussians(test_flight)
    np.testing.assert_allclose(
        result.vs, gaussians @ expected_segment.v_hat / 1000
    )
    np.testing.assert_allclose(
        result.ws, gaussians @ expected_segment.w_hat / 1000
    )
    np.testing.assert_array_equal(result.keep_mask, [False, True, True, True])
    self.assertEqual(result.overlap_start_timestamp, 1672531500)
    self.assertEqual(result.overlap_end_timestamp, 1672531800)

  def test_compute_gaussians(self):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0, satellite_altitude_km=35786.023
    )
    test_flight = flight.Flight(
        data={
            'longitude': [-75.0, -80.0],
            'latitude': [0.0, 10.0],
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime([
                '2023-01-01 00:00:00',
                '2023-01-01 01:00:00',
            ]),
        }
    )

    gaussians = matcher._compute_gaussians(test_flight)

    expected_lats, expected_lngs = projection.geostationary_parallax_correction(
        test_flight.data['latitude'],
        test_flight.data['longitude'],
        test_flight.data['altitude'] / 1000,
        0,
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.023,
    )
    expected_gaussians = matching_lib.to_gaussian(expected_lats, expected_lngs)
    self.assertEqual(gaussians.shape, (2, 3))
    np.testing.assert_allclose(gaussians, expected_gaussians)
    np.testing.assert_allclose(
        gaussians[0], matching_lib.to_gaussian(0.0, -75.0)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='perfect_match',
          latitudes=[0.0, 0.0],
          expected_fit_score=0.0,
          expected_shift_score=0.0,
          expected_angle_score=0.0,
          expected_age_score=(1.0 + 55.0 / 60.0) / 2.0,
          expected_v_offset=0.0,
          expected_w_offset=0.0,
          expected_theta=0.0,
      ),
      dict(
          testcase_name='shifted_and_angled',
          latitudes=[0.02, 0.06],
          expected_fit_score=0.56044,
          expected_shift_score=0.20993,
          expected_angle_score=1.11238,
          expected_age_score=(1.0 + 55.0 / 60.0) / 2.0,
          expected_v_offset=0.38686,
          expected_w_offset=-4.37381,
          expected_theta=-0.08822,
      ),
  )
  def test_match_score_returns_expected_score_components(
      self,
      latitudes,
      expected_fit_score,
      expected_shift_score,
      expected_angle_score,
      expected_age_score,
      expected_v_offset,
      expected_w_offset,
      expected_theta,
  ):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    advected_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array(latitudes),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_1',
    )
    contrail = matching_lib.LinearContrail(
        lon1=-75.2,
        lat1=0.0,
        lon2=-74.8,
        lat2=0.0,
        altitude_m=10000.0,
        timestamp=1672534800,
        satellite_longitude=-75.0,
    )

    result = matcher.match_score(advected_flight, contrail)

    expected_age_hrs = (1.0 + 55.0 / 60.0) / 2.0
    self.assertIsNotNone(result)
    self.assertEqual(result.contrail, contrail)
    self.assertEqual(result.flight_id, 'flight_1')
    self.assertEqual(result.overlap_start_timestamp, 1672531200)
    self.assertEqual(result.overlap_end_timestamp, 1672531500)
    score_components = result.score_components
    with self.subTest('fit_score'):
      np.testing.assert_allclose(
          score_components.fit_score, expected_fit_score, atol=1e-4
      )
    with self.subTest('shift_score'):
      np.testing.assert_allclose(
          score_components.shift_score, expected_shift_score, atol=1e-4
      )
    with self.subTest('angle_score'):
      np.testing.assert_allclose(
          score_components.angle_score, expected_angle_score, atol=1e-4
      )
    with self.subTest('age_score'):
      np.testing.assert_allclose(
          score_components.age_score, expected_age_score, atol=1e-4
      )
    with self.subTest('total'):
      expected_total = (
          expected_fit_score
          + expected_shift_score
          + expected_angle_score
          + expected_age_score
      )
      np.testing.assert_allclose(
          score_components.total(), expected_total, atol=1e-4
      )
      np.testing.assert_allclose(result.score, expected_total, atol=1e-4)
    with self.subTest('v_offset'):
      np.testing.assert_allclose(
          score_components.v_offset, expected_v_offset, atol=1e-4
      )
    with self.subTest('w_offset'):
      np.testing.assert_allclose(
          score_components.w_offset, expected_w_offset, atol=1e-4
      )
    with self.subTest('theta'):
      np.testing.assert_allclose(
          score_components.theta, expected_theta, atol=1e-4
      )
    with self.subTest('age_hrs'):
      np.testing.assert_allclose(
          score_components.age_hrs, expected_age_hrs, atol=1e-4
      )

  def test_match_score_accepts_geojson_feature(self):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    advected_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.0, 0.0]),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_1',
    )
    feature = geojson.Feature(
        geometry=geojson.LineString(
            [(-75.2, 0.0, 10000.0), (-74.8, 0.0, 10000.0)]
        ),
        properties={
            'time': '2023-01-01T01:00:00Z',
            'satellite_origin': 'GOES-EAST-FULL-DISK',
        },
    )

    result = matcher.match_score(advected_flight, feature)

    expected_age_hrs = (1.0 + 55.0 / 60.0) / 2.0
    self.assertIsNotNone(result)
    self.assertEqual(result.flight_id, 'flight_1')
    self.assertEqual(result.overlap_start_timestamp, 1672531200)
    self.assertEqual(result.overlap_end_timestamp, 1672531500)
    np.testing.assert_allclose(result.score, expected_age_hrs, atol=1e-5)
    np.testing.assert_allclose(
        result.score_components.total(), expected_age_hrs, atol=1e-5
    )

  def test_match_score_returns_none_when_validation_fails(self):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    advected_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.0, 0.0]),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_1',
    )
    short_contrail = matching_lib.LinearContrail(
        lon1=-75.01,
        lat1=0.0,
        lon2=-75.0,
        lat2=0.0,
        altitude_m=10000.0,
        timestamp=1672534800,
        satellite_longitude=-75.0,
    )

    result = matcher.match_score(advected_flight, short_contrail)

    self.assertIsNone(result)
    self.assertEqual(matcher.counters, {'small contrail': 1})

  def test_match_score_with_fixed_optimization_age_hrs(self):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    flight_1hr = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.05, 0.05]),
            'altitude': np.array([0.0, 0.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_1hr',
    )
    flight_2hr = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.05, 0.05]),
            'altitude': np.array([0.0, 0.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 02:00:00', '2023-01-01 02:00:00']
            ),
        },
        flight_id='flight_2hr',
    )
    contrail = matching_lib.LinearContrail(
        lon1=-75.2,
        lat1=0.0,
        lon2=-74.8,
        lat2=0.0,
        altitude_m=0.0,
        timestamp=1672534800,
        satellite_longitude=-75.0,
    )

    result_1hr = matcher.match_score(
        flight_1hr, contrail, optimization_age_hrs=0.5
    )
    result_2hr = matcher.match_score(
        flight_2hr, contrail, optimization_age_hrs=0.5
    )

    self.assertIsNotNone(result_1hr)
    self.assertIsNotNone(result_2hr)
    score_1hr = result_1hr.score_components
    score_2hr = result_2hr.score_components
    np.testing.assert_allclose(score_1hr.v_offset, score_2hr.v_offset)
    np.testing.assert_allclose(score_1hr.w_offset, score_2hr.w_offset)
    np.testing.assert_allclose(score_1hr.theta, score_2hr.theta)
    np.testing.assert_allclose(score_1hr.age_hrs, (1.0 + 55.0 / 60.0) / 2.0)
    np.testing.assert_allclose(score_2hr.age_hrs, (2.0 + 115.0 / 60.0) / 2.0)
    self.assertNotAlmostEqual(
        np.asarray(result_2hr.score).item()
        - np.asarray(result_1hr.score).item(),
        1.0,
    )

  def test_match_score_parallax_mismatch_raises(self):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    advected_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.0, 0.0]),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        }
    )
    mismatched_contrail = matching_lib.LinearContrail(
        lon1=-75.2,
        lat1=0.0,
        lon2=-74.8,
        lat2=0.0,
        altitude_m=10000.0,
        timestamp=1672534800,
        satellite_longitude=0.0,
    )

    with self.assertRaisesRegex(ValueError, 'Parallax mismatch for contrail'):
      matcher.match_score(advected_flight, mismatched_contrail)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_advected_flight',
          use_empty_flight=True,
          consider_better_matches=False,
      ),
      dict(
          testcase_name='consider_better_matches_with_none_flights',
          consider_better_matches=True,
          flights=None,
      ),
      dict(
          testcase_name='consider_better_matches_with_empty_flights',
          consider_better_matches=True,
          flights=[],
      ),
      dict(
          testcase_name='young_flight',
          advected_times=['2023-01-01 00:04:59', '2023-01-01 00:04:59'],
          consider_better_matches=False,
      ),
      dict(
          testcase_name='empty_potential_matches',
          empty_contrails=True,
          consider_better_matches=False,
      ),
      dict(
          testcase_name='all_contrails_fail_validation',
          contrail_lon2=-75.19,
          consider_better_matches=False,
      ),
  )
  def test_match_to_contrails_returns_empty_when_preconditions_not_met(
      self,
      consider_better_matches,
      use_empty_flight=False,
      flights=None,
      advected_times=None,
      empty_contrails=False,
      contrail_lon2=-74.8,
  ):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    if use_empty_flight:
      advected_flight = flight.Flight()
    else:
      advected_flight = flight.Flight(
          data={
              'longitude': np.array([-75.15, -74.85]),
              'latitude': np.array([0.0, 0.0]),
              'altitude': np.array([10000.0, 10000.0]),
              'time': pd.to_datetime(
                  ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
              ),
              'advected_time': pd.to_datetime(
                  advected_times
                  or ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
              ),
          },
          flight_id='flight_1',
      )
    if empty_contrails:
      contrails = []
    else:
      contrails = [
          matching_lib.LinearContrail(
              lon1=-75.2,
              lat1=0.0,
              lon2=contrail_lon2,
              lat2=0.0,
              altitude_m=10000.0,
              timestamp=1672534800,
              satellite_longitude=-75.0,
          )
      ]

    results = matcher.match_to_contrails(
        potential_matches=contrails,
        advected_flight=advected_flight,
        consider_better_matches=consider_better_matches,
        flights=flights,
    )

    self.assertEmpty(results)

  @parameterized.named_parameters(
      dict(
          testcase_name='better_matches_false_keeps_match',
          consider_better_matches=False,
          other_flight_latitudes=[0.0, 0.0],
          expected_num_matches=1,
      ),
      dict(
          testcase_name='better_matches_true_excludes_when_better_exists',
          consider_better_matches=True,
          other_flight_latitudes=[0.0, 0.0],
          expected_num_matches=0,
      ),
      dict(
          testcase_name='better_matches_true_keeps_when_no_better_exists',
          consider_better_matches=True,
          other_flight_latitudes=[0.02, 0.06],
          expected_num_matches=1,
      ),
  )
  def test_match_to_contrails_single_contrail_better_match_filtering(
      self,
      consider_better_matches,
      other_flight_latitudes,
      expected_num_matches,
  ):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    advected_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.02, 0.06]),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_1',
    )
    other_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array(other_flight_latitudes),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_2',
    )
    contrail = matching_lib.LinearContrail(
        lon1=-75.2,
        lat1=0.0,
        lon2=-74.8,
        lat2=0.0,
        altitude_m=10000.0,
        timestamp=1672534800,
        satellite_longitude=-75.0,
    )

    results = matcher.match_to_contrails(
        potential_matches=[contrail],
        advected_flight=advected_flight,
        consider_better_matches=consider_better_matches,
        flights=[other_flight],
    )

    self.assertLen(results, expected_num_matches)
    if expected_num_matches:
      self.assertEqual(results[0].contrail, contrail)
      self.assertEqual(results[0].flight_id, 'flight_1')

  @parameterized.named_parameters(
      dict(
          testcase_name='consider_better_matches_false_returns_all_valid',
          consider_better_matches=False,
          expected_matched_latitudes=[-0.02, 0.02],
      ),
      dict(
          testcase_name='consider_better_matches_true_keeps_only_unbeaten',
          consider_better_matches=True,
          expected_matched_latitudes=[0.02],
      ),
  )
  def test_match_to_contrails_some_match_and_some_do_not(
      self,
      consider_better_matches,
      expected_matched_latitudes,
  ):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    advected_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.0, 0.0]),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_1',
    )
    # Co-located with contrail_beaten (south of advected_flight), and twice as
    # far/angled from contrail_kept (north of advected_flight).
    better_flight_for_first = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([-0.02, -0.06]),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_2',
    )
    contrail_beaten = matching_lib.LinearContrail(
        lon1=-75.2,
        lat1=-0.02,
        lon2=-74.8,
        lat2=-0.06,
        altitude_m=10000.0,
        timestamp=1672534800,
        satellite_longitude=-75.0,
    )
    contrail_kept = matching_lib.LinearContrail(
        lon1=-75.2,
        lat1=0.02,
        lon2=-74.8,
        lat2=0.06,
        altitude_m=10000.0,
        timestamp=1672534800,
        satellite_longitude=-75.0,
    )
    contrail_too_far = matching_lib.LinearContrail(
        lon1=-75.2,
        lat1=10.0,
        lon2=-74.8,
        lat2=10.0,
        altitude_m=10000.0,
        timestamp=1672534800,
        satellite_longitude=-75.0,
    )

    results = matcher.match_to_contrails(
        potential_matches=[contrail_beaten, contrail_kept, contrail_too_far],
        advected_flight=advected_flight,
        consider_better_matches=consider_better_matches,
        flights=[better_flight_for_first],
    )

    self.assertEqual(
        [r.contrail.lat1 for r in results],
        expected_matched_latitudes,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='exact_batch_size',
          num_contrails=matching_lib.JAX_BATCH_SIZE,
      ),
      dict(
          testcase_name='multiple_batches_with_partial_remainder',
          num_contrails=matching_lib.JAX_BATCH_SIZE + 2,
      ),
  )
  def test_match_to_contrails_batches_full_and_partial_chunks(
      self, num_contrails
  ):
    matcher = matching_lib.SingleFrameContrailMatcher(
        satellite_longitude=-75.0,
        satellite_altitude_km=35786.0,
    )
    advected_flight = flight.Flight(
        data={
            'longitude': np.array([-75.15, -74.85]),
            'latitude': np.array([0.0, 0.0]),
            'altitude': np.array([10000.0, 10000.0]),
            'time': pd.to_datetime(
                ['2023-01-01 00:00:00', '2023-01-01 00:05:00']
            ),
            'advected_time': pd.to_datetime(
                ['2023-01-01 01:00:00', '2023-01-01 01:00:00']
            ),
        },
        flight_id='flight_1',
    )
    latitudes = np.linspace(-0.05, 0.05, num_contrails)
    contrails = [
        matching_lib.LinearContrail(
            lon1=-75.2,
            lat1=float(lat),
            lon2=-74.8,
            lat2=float(lat),
            altitude_m=10000.0,
            timestamp=1672534800,
            satellite_longitude=-75.0,
        )
        for lat in latitudes
    ]

    results = matcher.match_to_contrails(
        potential_matches=contrails,
        advected_flight=advected_flight,
        consider_better_matches=False,
    )

    self.assertLen(results, num_contrails)
    self.assertEqual([r.contrail for r in results], contrails)
    for contrail, result in zip(contrails, results, strict=True):
      single_result = matcher.match_score(advected_flight, contrail)
      self.assertIsNotNone(single_result)
      np.testing.assert_allclose(result.score, single_result.score, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
