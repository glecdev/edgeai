"""
GLEC DTG - Feature Extraction Accuracy Test

Tests that Python feature extraction logic matches Kotlin FeatureExtractor.kt
for cross-platform validation and debugging.

Red-Green-Refactor (TDD):
- RED: This test will initially fail (no Python implementation yet)
- GREEN: Implement Python version matching Kotlin logic
- REFACTOR: Optimize and validate against synthetic data
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


class CANDataSample:
    """Python representation of Kotlin CANData.kt"""

    def __init__(self, **kwargs):
        self.timestamp = kwargs.get('timestamp', 0)
        self.vehicle_speed = kwargs.get('vehicle_speed', 0.0)
        self.engine_rpm = kwargs.get('engine_rpm', 0)
        self.throttle_position = kwargs.get('throttle_position', 0.0)
        self.brake_position = kwargs.get('brake_position', 0.0)
        self.fuel_level = kwargs.get('fuel_level', 50.0)
        self.coolant_temp = kwargs.get('coolant_temp', 90.0)
        self.engine_load = kwargs.get('engine_load', 0.0)
        self.intake_air_temp = kwargs.get('intake_air_temp', 25.0)
        self.maf_rate = kwargs.get('maf_rate', 0.0)
        self.battery_voltage = kwargs.get('battery_voltage', 12.6)
        self.acceleration_x = kwargs.get('acceleration_x', 0.0)
        self.acceleration_y = kwargs.get('acceleration_y', 0.0)
        self.acceleration_z = kwargs.get('acceleration_z', 0.0)
        self.gyro_x = kwargs.get('gyro_x', 0.0)
        self.gyro_y = kwargs.get('gyro_y', 0.0)
        self.gyro_z = kwargs.get('gyro_z', 0.0)
        self.gps_latitude = kwargs.get('gps_latitude', 37.5665)
        self.gps_longitude = kwargs.get('gps_longitude', 126.9780)
        self.gps_altitude = kwargs.get('gps_altitude', 0.0)
        self.gps_speed = kwargs.get('gps_speed', 0.0)
        self.gps_heading = kwargs.get('gps_heading', 0.0)

    def calculate_fuel_consumption(self) -> float:
        """
        Calculate instantaneous fuel consumption (L/100km)
        Matches Kotlin CANData.calculateFuelConsumption()
        """
        if self.vehicle_speed < 1.0 or self.maf_rate < 0.1:
            return 0.0

        # Stoichiometric air-fuel ratio for gasoline: 14.7:1
        fuel_flow_rate = self.maf_rate / 14.7  # g/s

        # Convert to L/h: (g/s) * 3600 / (density of gasoline ~750 g/L)
        fuel_flow_liter_per_hour = (fuel_flow_rate * 3600.0) / 750.0

        # Convert to L/100km: (L/h) / (km/h) * 100
        return (fuel_flow_liter_per_hour / self.vehicle_speed) * 100.0


class PythonFeatureExtractor:
    """
    Python implementation of Kotlin FeatureExtractor.kt

    Extracts 18-dimensional feature vectors from 60-second windows.
    Must produce identical results to Kotlin version for validation.
    """

    FEATURE_DIMENSION = 18
    DEFAULT_WINDOW_SIZE = 60

    FEATURE_NAMES = [
        "speed_mean", "speed_std", "speed_max", "speed_min",
        "rpm_mean", "rpm_std",
        "throttle_mean", "throttle_std", "throttle_max",
        "brake_mean", "brake_std", "brake_max",
        "accel_x_mean", "accel_x_std", "accel_x_max",
        "accel_y_mean", "accel_y_std",
        "fuel_consumption"
    ]

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.window: List[CANDataSample] = []

    def add_sample(self, sample: CANDataSample):
        """Add new sample to sliding window"""
        self.window.append(sample)

        # Remove oldest sample if window exceeds size (FIFO)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def is_window_ready(self) -> bool:
        """Check if window has enough samples"""
        return len(self.window) == self.window_size

    def get_sample_count(self) -> int:
        """Get current sample count"""
        return len(self.window)

    def extract_features(self) -> np.ndarray:
        """
        Extract 18-dimensional feature vector

        Returns:
            numpy array of shape (18,) or None if window not ready
        """
        if not self.is_window_ready():
            return None

        # Extract raw values from window
        speeds = [s.vehicle_speed for s in self.window]
        rpms = [float(s.engine_rpm) for s in self.window]
        throttles = [s.throttle_position for s in self.window]
        brakes = [s.brake_position for s in self.window]
        accels_x = [s.acceleration_x for s in self.window]
        accels_y = [s.acceleration_y for s in self.window]
        fuel_consumptions = [s.calculate_fuel_consumption() for s in self.window]

        # Calculate features (18 dimensions) - matching Kotlin logic exactly
        features = np.array([
            # [0-3] Speed statistics
            np.mean(speeds),
            np.std(speeds, ddof=0),  # Population std (ddof=0) to match Kotlin
            np.max(speeds),
            np.min(speeds),

            # [4-5] RPM statistics
            np.mean(rpms),
            np.std(rpms, ddof=0),

            # [6-8] Throttle statistics
            np.mean(throttles),
            np.std(throttles, ddof=0),
            np.max(throttles),

            # [9-11] Brake statistics
            np.mean(brakes),
            np.std(brakes, ddof=0),
            np.max(brakes),

            # [12-14] Acceleration X statistics
            np.mean(accels_x),
            np.std(accels_x, ddof=0),
            np.max(accels_x),

            # [15-16] Acceleration Y statistics
            np.mean(accels_y),
            np.std(accels_y, ddof=0),

            # [17] Fuel consumption (mean)
            np.mean(fuel_consumptions)
        ], dtype=np.float32)

        return features

    def reset(self):
        """Clear window"""
        self.window.clear()


class TestFeatureExtractionAccuracy(unittest.TestCase):
    """
    Test suite for validating Python FeatureExtractor against Kotlin logic

    Phase 1.5 - Quality Assurance
    """

    def setUp(self):
        """Setup test fixtures"""
        self.extractor = PythonFeatureExtractor(window_size=60)

    def test_extractor_initialization(self):
        """Test FeatureExtractor initializes correctly"""
        self.assertEqual(self.extractor.window_size, 60)
        self.assertEqual(self.extractor.get_sample_count(), 0)
        self.assertFalse(self.extractor.is_window_ready())

    def test_window_filling(self):
        """Test sliding window fills correctly"""
        # Add 30 samples
        for i in range(30):
            sample = CANDataSample(
                timestamp=i * 1000,
                vehicle_speed=50.0,
                engine_rpm=2000
            )
            self.extractor.add_sample(sample)

        self.assertEqual(self.extractor.get_sample_count(), 30)
        self.assertFalse(self.extractor.is_window_ready())

        # Add 30 more samples (total 60)
        for i in range(30, 60):
            sample = CANDataSample(
                timestamp=i * 1000,
                vehicle_speed=50.0,
                engine_rpm=2000
            )
            self.extractor.add_sample(sample)

        self.assertEqual(self.extractor.get_sample_count(), 60)
        self.assertTrue(self.extractor.is_window_ready())

    def test_sliding_window_fifo(self):
        """Test sliding window removes oldest samples (FIFO)"""
        # Fill window with 60 samples (speed = index)
        for i in range(60):
            sample = CANDataSample(
                timestamp=i * 1000,
                vehicle_speed=float(i)
            )
            self.extractor.add_sample(sample)

        self.assertEqual(self.extractor.get_sample_count(), 60)

        # First sample should have speed=0
        self.assertEqual(self.extractor.window[0].vehicle_speed, 0.0)

        # Add one more sample (speed=60)
        sample = CANDataSample(timestamp=60000, vehicle_speed=60.0)
        self.extractor.add_sample(sample)

        # Window should still be 60, but first sample should now be speed=1 (oldest removed)
        self.assertEqual(self.extractor.get_sample_count(), 60)
        self.assertEqual(self.extractor.window[0].vehicle_speed, 1.0)
        self.assertEqual(self.extractor.window[-1].vehicle_speed, 60.0)

    def test_feature_extraction_constant_speed(self):
        """Test feature extraction with constant speed (baseline)"""
        # Fill window with constant values
        for i in range(60):
            sample = CANDataSample(
                timestamp=i * 1000,
                vehicle_speed=80.0,
                engine_rpm=2500,
                throttle_position=40.0,
                brake_position=0.0,
                acceleration_x=0.0,
                acceleration_y=0.0,
                maf_rate=15.0
            )
            self.extractor.add_sample(sample)

        features = self.extractor.extract_features()

        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (18,))

        # Check constant speed features
        self.assertAlmostEqual(features[0], 80.0, places=2)  # speed_mean
        self.assertAlmostEqual(features[1], 0.0, places=2)   # speed_std (constant)
        self.assertAlmostEqual(features[2], 80.0, places=2)  # speed_max
        self.assertAlmostEqual(features[3], 80.0, places=2)  # speed_min

        # Check RPM features
        self.assertAlmostEqual(features[4], 2500.0, places=2)  # rpm_mean
        self.assertAlmostEqual(features[5], 0.0, places=2)     # rpm_std (constant)

    def test_feature_extraction_eco_driving(self):
        """Test feature extraction for ECO_DRIVING behavior"""
        # Simulate eco driving: steady speed, low throttle, no harsh events
        for i in range(60):
            sample = CANDataSample(
                timestamp=i * 1000,
                vehicle_speed=60.0 + np.random.normal(0, 2.0),  # Steady speed ±2 km/h
                engine_rpm=1800 + int(np.random.normal(0, 100)),
                throttle_position=25.0 + np.random.normal(0, 5.0),
                brake_position=0.0,
                acceleration_x=np.random.normal(0, 0.5),  # Low acceleration variance
                acceleration_y=np.random.normal(0, 0.3),
                maf_rate=10.0 + np.random.normal(0, 1.0)
            )
            self.extractor.add_sample(sample)

        features = self.extractor.extract_features()

        self.assertIsNotNone(features)

        # Eco driving characteristics
        self.assertGreater(features[0], 50.0)  # speed_mean > 50 km/h
        self.assertLess(features[1], 5.0)      # speed_std < 5 (steady)
        self.assertLess(features[7], 10.0)     # throttle_std < 10 (smooth)
        self.assertLess(features[10], 5.0)     # brake_std < 5 (minimal braking)

    def test_feature_extraction_aggressive_driving(self):
        """Test feature extraction for AGGRESSIVE behavior"""
        # Simulate aggressive driving: high variance, harsh events
        for i in range(60):
            # Fluctuating speed
            speed = 80.0 + np.random.normal(0, 15.0)

            # High throttle with sudden changes
            throttle = 70.0 + np.random.normal(0, 20.0)

            # Occasional harsh braking
            brake = 50.0 if i % 10 == 0 else 0.0

            # High acceleration variance
            accel_x = np.random.normal(0, 2.0)
            if i % 15 == 0:
                accel_x = 4.0  # Harsh acceleration

            sample = CANDataSample(
                timestamp=i * 1000,
                vehicle_speed=max(0, min(speed, 120.0)),
                engine_rpm=3000 + int(np.random.normal(0, 500)),
                throttle_position=max(0, min(throttle, 100.0)),
                brake_position=brake,
                acceleration_x=accel_x,
                acceleration_y=np.random.normal(0, 1.0),
                maf_rate=25.0 + np.random.normal(0, 5.0)
            )
            self.extractor.add_sample(sample)

        features = self.extractor.extract_features()

        self.assertIsNotNone(features)

        # Aggressive driving characteristics
        self.assertGreater(features[1], 10.0)   # speed_std > 10 (high variance)
        self.assertGreater(features[7], 15.0)   # throttle_std > 15
        self.assertGreater(features[11], 40.0)  # brake_max > 40
        self.assertGreater(features[13], 2.0)   # accel_x_std > 2.0

    def test_fuel_consumption_calculation(self):
        """Test fuel consumption calculation matches Kotlin formula"""
        sample = CANDataSample(
            vehicle_speed=80.0,  # km/h
            maf_rate=15.0        # g/s
        )

        fuel = sample.calculate_fuel_consumption()

        # Manual calculation to verify
        # fuel_flow_rate = 15.0 / 14.7 = 1.0204 g/s
        # fuel_flow_L_per_h = (1.0204 * 3600) / 750 = 4.898 L/h
        # fuel_L_per_100km = (4.898 / 80) * 100 = 6.122 L/100km

        self.assertAlmostEqual(fuel, 6.122, places=2)

    def test_fuel_consumption_zero_speed(self):
        """Test fuel consumption returns 0 for zero speed (edge case)"""
        sample = CANDataSample(
            vehicle_speed=0.0,
            maf_rate=5.0
        )

        fuel = sample.calculate_fuel_consumption()
        self.assertEqual(fuel, 0.0)

    def test_fuel_consumption_zero_maf(self):
        """Test fuel consumption returns 0 for zero MAF (edge case)"""
        sample = CANDataSample(
            vehicle_speed=60.0,
            maf_rate=0.0
        )

        fuel = sample.calculate_fuel_consumption()
        self.assertEqual(fuel, 0.0)

    def test_feature_dimension(self):
        """Test feature vector has correct dimension (18)"""
        for i in range(60):
            sample = CANDataSample(
                timestamp=i * 1000,
                vehicle_speed=50.0
            )
            self.extractor.add_sample(sample)

        features = self.extractor.extract_features()

        self.assertEqual(len(features), PythonFeatureExtractor.FEATURE_DIMENSION)
        self.assertEqual(len(features), 18)

    def test_reset_functionality(self):
        """Test reset clears window"""
        # Fill window
        for i in range(60):
            sample = CANDataSample(timestamp=i * 1000)
            self.extractor.add_sample(sample)

        self.assertTrue(self.extractor.is_window_ready())

        # Reset
        self.extractor.reset()

        self.assertEqual(self.extractor.get_sample_count(), 0)
        self.assertFalse(self.extractor.is_window_ready())

    def test_feature_extraction_returns_none_when_not_ready(self):
        """Test extract_features returns None when window not ready"""
        # Add only 30 samples (window requires 60)
        for i in range(30):
            sample = CANDataSample(timestamp=i * 1000)
            self.extractor.add_sample(sample)

        features = self.extractor.extract_features()

        self.assertIsNone(features)


class TestFeatureExtractionWithSyntheticData(unittest.TestCase):
    """
    Test feature extraction with real synthetic dataset

    Validates that features can be extracted from production data
    and match expected statistical properties.
    """

    def setUp(self):
        """Load synthetic dataset if available"""
        self.dataset_path = Path(__file__).parent.parent / "datasets" / "train.csv"
        self.extractor = PythonFeatureExtractor(window_size=60)

    def test_load_synthetic_dataset(self):
        """Test that synthetic dataset can be loaded"""
        if not self.dataset_path.exists():
            self.skipTest(f"Synthetic dataset not found at {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)

        # Verify expected columns exist
        required_cols = ['vehicle_speed', 'engine_rpm', 'throttle_position',
                        'brake_pressure', 'acceleration_x', 'acceleration_y', 'fuel_consumption']

        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

        print(f"\n✅ Loaded {len(df)} samples from synthetic dataset")

    def test_extract_features_from_synthetic_data(self):
        """Test feature extraction from real synthetic dataset"""
        if not self.dataset_path.exists():
            self.skipTest(f"Synthetic dataset not found at {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)

        # Extract first 60 samples
        samples = []
        for i in range(min(60, len(df))):
            row = df.iloc[i]

            # Note: Synthetic dataset uses 'brake_pressure' not 'brake_position'
            # and has pre-calculated 'fuel_consumption' instead of 'maf_rate'
            # We'll approximate maf_rate from fuel_consumption for testing
            speed = row['vehicle_speed']
            fuel_consumption = row['fuel_consumption']

            # Reverse calculate maf_rate from fuel_consumption
            # fuel_consumption = (maf / 14.7 * 3600 / 750 / speed) * 100
            # maf = fuel_consumption * speed * 750 * 14.7 / (3600 * 100)
            if speed > 1.0 and fuel_consumption > 0.1:
                maf_rate = fuel_consumption * speed * 750.0 * 14.7 / (3600.0 * 100.0)
            else:
                maf_rate = 0.0

            sample = CANDataSample(
                timestamp=i * 1000,
                vehicle_speed=speed,
                engine_rpm=int(row['engine_rpm']),
                throttle_position=row['throttle_position'],
                brake_position=row['brake_pressure'],  # Use brake_pressure as brake_position
                maf_rate=maf_rate,
                acceleration_x=row['acceleration_x'],
                acceleration_y=row['acceleration_y']
            )
            samples.append(sample)
            self.extractor.add_sample(sample)

        if self.extractor.is_window_ready():
            features = self.extractor.extract_features()

            self.assertIsNotNone(features)
            self.assertEqual(features.shape, (18,))

            # Print features for debugging
            print("\n📊 Extracted features from synthetic data:")
            for i, (name, value) in enumerate(zip(PythonFeatureExtractor.FEATURE_NAMES, features)):
                print(f"  [{i:2d}] {name:20s}: {value:8.3f}")
        else:
            print(f"\n⚠️ Dataset has only {len(df)} samples, need 60 for window")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
