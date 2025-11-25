"""
Test for Real Structure Data Generator (Korean Traffic Safety Board format)

Tests for generate_production_dataset_real_structure.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.generate_production_dataset_real_structure import (
    generate_trip,
    detect_harsh_acceleration,
    simulate_gps_movement,
    VehicleSimulator,
    SEOUL_GPS_RANGE,
    ANOMALY_TYPES,
)


class TestRealStructureGenerator:
    """Test suite for real structure data generator"""

    def test_seoul_gps_range_valid(self):
        """Test Seoul GPS range is valid"""
        assert SEOUL_GPS_RANGE['lon_min'] == 126.88
        assert SEOUL_GPS_RANGE['lon_max'] == 126.90
        assert SEOUL_GPS_RANGE['lat_min'] == 37.51
        assert SEOUL_GPS_RANGE['lat_max'] == 37.58

    def test_anomaly_types_count(self):
        """Test 8 anomaly types are defined"""
        assert len(ANOMALY_TYPES) == 8
        assert 'harsh_acceleration_20' in ANOMALY_TYPES
        assert 'harsh_acceleration_40' in ANOMALY_TYPES
        assert 'harsh_braking' in ANOMALY_TYPES
        assert 'sharp_left_turn' in ANOMALY_TYPES

    def test_detect_harsh_acceleration_20km(self):
        """Test harsh acceleration detection for 20 km/h"""
        # Speed increase from 0 to 20 km/h in 1 second
        count_20, count_40, count_60 = detect_harsh_acceleration(
            speed_prev=0, speed_cur=20, dt=1.0
        )
        assert count_20 == 1
        assert count_40 == 0
        assert count_60 == 0

    def test_detect_harsh_acceleration_40km(self):
        """Test harsh acceleration detection for 40 km/h"""
        count_20, count_40, count_60 = detect_harsh_acceleration(
            speed_prev=0, speed_cur=40, dt=1.0
        )
        assert count_20 == 1
        assert count_40 == 1
        assert count_60 == 0

    def test_detect_harsh_acceleration_60km(self):
        """Test harsh acceleration detection for 60 km/h"""
        count_20, count_40, count_60 = detect_harsh_acceleration(
            speed_prev=0, speed_cur=60, dt=1.0
        )
        assert count_20 == 1
        assert count_40 == 1
        assert count_60 == 1

    def test_simulate_gps_movement_seoul_range(self):
        """Test GPS movement stays within Seoul range"""
        # Start in Seoul center
        lat, lon = 37.545, 126.89
        speed = 50  # km/h
        heading = 90  # East

        new_lat, new_lon = simulate_gps_movement(lat, lon, speed, heading, dt=1.0)

        # Check still in Seoul range
        assert SEOUL_GPS_RANGE['lat_min'] <= new_lat <= SEOUL_GPS_RANGE['lat_max']
        assert SEOUL_GPS_RANGE['lon_min'] <= new_lon <= SEOUL_GPS_RANGE['lon_max']

    def test_vehicle_simulator_init(self):
        """Test VehicleSimulator initialization"""
        simulator = VehicleSimulator(pattern='city_traffic')
        assert simulator.pattern == 'city_traffic'
        assert simulator.lat >= SEOUL_GPS_RANGE['lat_min']
        assert simulator.lat <= SEOUL_GPS_RANGE['lat_max']
        assert simulator.lon >= SEOUL_GPS_RANGE['lon_min']
        assert simulator.lon <= SEOUL_GPS_RANGE['lon_max']

    def test_vehicle_simulator_step(self):
        """Test VehicleSimulator single timestep"""
        simulator = VehicleSimulator(pattern='city_traffic')
        state = simulator.step(inject_anomaly=None)

        # Check all 27 features exist
        assert len(state) == 27
        assert 'vehicle_speed' in state
        assert 'gps_x' in state
        assert 'gps_y' in state
        assert '급가속건수20KM이상' in state
        assert '급감속건수' in state

    def test_generate_trip_normal(self):
        """Test generate_trip with normal pattern (no anomaly)"""
        df = generate_trip(
            trip_id=1,
            pattern='city_traffic',
            duration_seconds=60,
            anomaly_ratio=0.0
        )

        # Check shape
        assert len(df) == 60  # 60 seconds
        assert len(df.columns) == 27  # 27 features

        # Check no anomalies
        assert df['급가속건수20KM이상'].sum() == 0
        assert df['급가속건수40KM이상'].sum() == 0
        assert df['급감속건수'].sum() == 0

    def test_generate_trip_with_anomaly(self):
        """Test generate_trip with anomaly injection"""
        df = generate_trip(
            trip_id=1,
            pattern='city_traffic',
            duration_seconds=60,
            anomaly_ratio=0.2  # 20% anomaly
        )

        # Check shape
        assert len(df) == 60

        # Check some anomalies exist (at least one type)
        total_anomalies = (
            df['급가속건수20KM이상'].sum() +
            df['급감속건수'].sum() +
            df['급좌회전건수'].sum()
        )
        assert total_anomalies > 0

    def test_feature_columns_match_real_data(self):
        """Test generated features match Korean Traffic Safety Board data"""
        expected_features = [
            'trip_id',
            'timestamp',
            'time_seconds',
            'gps_x',
            'gps_y',
            'vehicle_speed',
            '급가속건수20KM이상',
            '급가속시간20KM이상',
            '급가속건수40KM이상',
            '급가속시간40KM이상',
            '급가속건수60KM이상',
            '급가속시간60KM이상',
            '급출발건수',
            '급출발시간',
            '급감속건수',
            '급정지건수',
            '급좌회전건수',
            '우회전건수',
            '급정거건수',
            '급유턴운전건수',
            '급차로변경건수',
            'engine_rpm',
            'throttle_position',
            'brake_pressure',
            'fuel_consumption',
            'carbon_emission',
            'label',
        ]

        df = generate_trip(trip_id=1, pattern='city_traffic', duration_seconds=60)

        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"

    def test_gps_consistency(self):
        """Test GPS coordinates are consistent with speed and time"""
        df = generate_trip(
            trip_id=1,
            pattern='highway_cruise',
            duration_seconds=60
        )

        # Calculate total distance from GPS
        total_distance_gps = 0
        for i in range(1, len(df)):
            lat1, lon1 = df.iloc[i-1]['gps_y'], df.iloc[i-1]['gps_x']
            lat2, lon2 = df.iloc[i]['gps_y'], df.iloc[i]['gps_x']

            # Haversine distance (simplified)
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 +
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371000 * c  # Earth radius in meters
            total_distance_gps += distance

        # Calculate total distance from speed
        total_distance_speed = df['vehicle_speed'].mean() / 3.6 * 60  # m

        # Should be roughly similar (within 50% tolerance due to turns)
        ratio = total_distance_gps / total_distance_speed if total_distance_speed > 0 else 0
        assert 0.5 <= ratio <= 1.5, f"GPS distance inconsistent: {ratio:.2f}"

    def test_statistics_match_real_data(self):
        """Test generated data statistics match real Korean traffic data"""
        df = generate_trip(
            trip_id=1,
            pattern='city_traffic',
            duration_seconds=600  # 10 minutes for better statistics
        )

        # Check speed range (Korean city traffic: 0-60 km/h)
        assert df['vehicle_speed'].min() >= 0
        assert df['vehicle_speed'].max() <= 120
        assert df['vehicle_speed'].mean() >= 10  # Not all stopped
        assert df['vehicle_speed'].mean() <= 50  # Not all highway

        # Check RPM range (typical: 600-3000 RPM)
        assert df['engine_rpm'].min() >= 0
        assert df['engine_rpm'].max() <= 4000

        # Check fuel consumption is positive
        assert df['fuel_consumption'].min() >= 0
        assert df['fuel_consumption'].mean() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
