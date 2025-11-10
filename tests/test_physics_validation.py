#!/usr/bin/env python3
"""
GLEC DTG - Physics Validation Tests

Tests production-ported physics-based plausibility validation:
- Newton's laws of motion
- Energy conservation
- Thermodynamic constraints
- Sensor fault detection
"""

import sys
import os
from pathlib import Path

# Add ai-models to path for import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "ai-models"))

import unittest
import time
from validation.physics_validator import PhysicsValidator, ValidationResult
from inference.realtime_integration import RealtimeCANData


class TestPhysicsValidator(unittest.TestCase):
    """Test physics-based validation system"""

    def setUp(self):
        """Initialize validator for each test"""
        self.validator = PhysicsValidator(vehicle_type="truck")

    def test_normal_driving(self):
        """Test validation with normal driving data"""
        data1 = self._create_test_data(
            timestamp=1000,
            speed=60.0,
            rpm=2000,
            throttle=40.0,
            brake=0.0,
            accel_x=0.0
        )

        data2 = self._create_test_data(
            timestamp=2000,
            speed=65.0,
            rpm=2200,
            throttle=50.0,
            brake=0.0,
            accel_x=1.4  # (65-60)/3.6/1.0 = 1.39 m/s²
        )

        # First validation (no previous data)
        result1 = self.validator.validate(data1)
        self.assertTrue(result1.is_valid)

        # Second validation (with previous data)
        result2 = self.validator.validate(data2)
        self.assertTrue(result2.is_valid)
        self.assertEqual(result2.confidence, 1.0)

    def test_impossible_acceleration(self):
        """Test detection of impossible acceleration (sensor fault)"""
        data1 = self._create_test_data(
            timestamp=1000,
            speed=60.0,
            rpm=2000
        )

        # Impossible: 60 → 150 km/h in 1 second (25 m/s²!)
        data2 = self._create_test_data(
            timestamp=2000,
            speed=150.0,
            rpm=2200,
            accel_x=1.0  # IMU says 1.0 m/s² but speed says 25 m/s²
        )

        self.validator.validate(data1)
        result = self.validator.validate(data2)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "SENSOR_MALFUNCTION")
        self.assertIn("Impossible acceleration", result.reason)

    def test_impossible_deceleration(self):
        """Test detection of impossible deceleration"""
        data1 = self._create_test_data(
            timestamp=1000,
            speed=80.0,
            rpm=2500
        )

        # Impossible: 80 → 0 km/h in 0.5 seconds (-44 m/s²!)
        data2 = self._create_test_data(
            timestamp=1500,
            speed=0.0,
            rpm=800
        )

        self.validator.validate(data1)
        result = self.validator.validate(data2)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "SENSOR_MALFUNCTION")
        self.assertIn("Impossible deceleration", result.reason)

    def test_speed_imu_mismatch(self):
        """Test detection of speed/IMU accelerometer mismatch"""
        data1 = self._create_test_data(
            timestamp=1000,
            speed=60.0,
            rpm=2000
        )

        # Speed says 1.4 m/s² but IMU says 0.1 m/s² (mismatch)
        data2 = self._create_test_data(
            timestamp=2000,
            speed=65.0,
            rpm=2200,
            accel_x=0.1  # Should be ~1.4 m/s²
        )

        self.validator.validate(data1)
        result = self.validator.validate(data2)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "SENSOR_CORRELATION_ERROR")
        self.assertIn("Speed/IMU acceleration mismatch", result.reason)

    def test_acceleration_without_throttle(self):
        """Test detection of acceleration without throttle input"""
        data1 = self._create_test_data(
            timestamp=1000,
            speed=60.0,
            rpm=2000,
            throttle=5.0  # Very low throttle
        )

        # Accelerating but throttle is low
        data2 = self._create_test_data(
            timestamp=2000,
            speed=65.0,
            rpm=2200,
            throttle=5.0,  # Still low throttle
            accel_x=1.4
        )

        self.validator.validate(data1)
        result = self.validator.validate(data2)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "SENSOR_CORRELATION_ERROR")
        self.assertIn("Acceleration without throttle", result.reason)

    def test_negative_speed(self):
        """Test detection of negative speed (sensor malfunction)"""
        data = self._create_test_data(speed=-10.0)

        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "SENSOR_MALFUNCTION")
        self.assertIn("Negative speed", result.reason)

    def test_speed_limit_exceeded(self):
        """Test detection of speed limit violation"""
        data = self._create_test_data(speed=150.0)  # > 120 km/h truck limit

        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "SPEED_LIMITER_FAILURE")

    def test_negative_rpm(self):
        """Test detection of negative RPM"""
        data = self._create_test_data(rpm=-500)

        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "SENSOR_MALFUNCTION")
        self.assertIn("Negative RPM", result.reason)

    def test_rpm_overspeed(self):
        """Test detection of RPM redline exceeded"""
        data = self._create_test_data(rpm=5000)  # > 4000 RPM truck limit

        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "ENGINE_OVERSPEED")

    def test_rpm_speed_ratio(self):
        """Test RPM/speed ratio validation (gear ratio check)"""
        # Unrealistic ratio: 100 km/h at 1000 RPM (should be ~2000-3000 RPM)
        data = self._create_test_data(
            speed=100.0,
            rpm=1000
        )

        result = self.validator.validate(data)

        # This should fail due to unlikely gear ratio
        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "TRANSMISSION_ERROR")
        self.assertIn("RPM/speed ratio", result.reason)

    def test_fuel_consumption_plausibility(self):
        """Test fuel consumption physics validation"""
        data = self._create_test_data(
            maf_rate=5.0,  # 5 g/s MAF
            throttle=40.0
        )

        result = self.validator.validate(data)

        # Should pass: 5 g/s MAF is reasonable
        self.assertTrue(result.is_valid)

    def test_impossible_fuel_rate(self):
        """Test detection of impossible fuel consumption"""
        data = self._create_test_data(
            maf_rate=200.0,  # 200 g/s MAF (absurd for a truck!)
            throttle=40.0
        )

        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "FUEL_SENSOR_ERROR")
        self.assertIn("Implausible fuel rate", result.reason)

    def test_high_fuel_at_low_throttle(self):
        """Test detection of high fuel consumption at idle"""
        data = self._create_test_data(
            maf_rate=20.0,  # High MAF
            throttle=3.0  # But throttle is low
        )

        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "FUEL_LEAK_OR_SENSOR_ERROR")

    def test_battery_voltage_range(self):
        """Test battery voltage validation"""
        # Too low
        data1 = self._create_test_data(battery=8.0)
        result1 = self.validator.validate(data1)
        self.assertFalse(result1.is_valid)
        self.assertEqual(result1.anomaly_type, "ELECTRICAL_SYSTEM_FAULT")

        # Normal
        data2 = self._create_test_data(battery=12.6)
        result2 = self.validator.validate(data2)
        self.assertTrue(result2.is_valid)

        # Too high
        data3 = self._create_test_data(battery=18.0)
        result3 = self.validator.validate(data3)
        self.assertFalse(result3.is_valid)

    def test_coolant_temperature_range(self):
        """Test coolant temperature validation"""
        # Too cold
        data1 = self._create_test_data(coolant=-50)
        result1 = self.validator.validate(data1)
        self.assertFalse(result1.is_valid)
        self.assertEqual(result1.anomaly_type, "TEMPERATURE_SENSOR_FAULT")

        # Normal
        data2 = self._create_test_data(coolant=85)
        result2 = self.validator.validate(data2)
        self.assertTrue(result2.is_valid)

        # Overheating
        data3 = self._create_test_data(coolant=130)
        result3 = self.validator.validate(data3)
        self.assertFalse(result3.is_valid)

    def test_thermodynamic_consistency(self):
        """Test engine temperature vs load correlation"""
        # High RPM but cold engine (thermostat stuck or sensor fault)
        data = self._create_test_data(
            rpm=3500,
            coolant=50  # Should be >80°C at high RPM
        )

        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.anomaly_type, "COOLING_SYSTEM_FAULT")

    def test_validation_result_structure(self):
        """Test ValidationResult dataclass structure"""
        result = ValidationResult(
            is_valid=False,
            confidence=0.9,
            reason="Test failure",
            anomaly_type="TEST_ANOMALY"
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.reason, "Test failure")
        self.assertEqual(result.anomaly_type, "TEST_ANOMALY")

    def _create_test_data(
        self,
        timestamp=None,
        speed=60.0,
        rpm=2000,
        throttle=40.0,
        brake=0.0,
        coolant=85,
        maf_rate=5.0,
        battery=12.6,
        accel_x=0.0
    ):
        """Helper: Create test CAN data"""
        if timestamp is None:
            timestamp = int(time.time() * 1000)

        return RealtimeCANData(
            timestamp=timestamp,
            vehicle_speed=speed,
            engine_rpm=rpm,
            fuel_level=75.0,
            throttle_position=throttle,
            brake_position=brake,
            coolant_temp=coolant,
            maf_rate=maf_rate,
            battery_voltage=battery,
            acceleration_x=accel_x,
            acceleration_y=0.0,
            acceleration_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            latitude=37.5665,
            longitude=126.9780,
            altitude=50.0,
            heading=45.0
        )


class TestPhysicsConstants(unittest.TestCase):
    """Test physics constants and calculations"""

    def test_constants(self):
        """Test physical constants"""
        validator = PhysicsValidator()

        self.assertEqual(validator.GRAVITY, 9.81)
        self.assertEqual(validator.AIR_FUEL_RATIO, 14.7)
        self.assertEqual(validator.FUEL_DENSITY, 750)

    def test_vehicle_limits(self):
        """Test vehicle physical limits"""
        validator = PhysicsValidator(vehicle_type="truck")

        self.assertEqual(validator.MAX_ACCELERATION, 3.5)
        self.assertEqual(validator.MAX_DECELERATION, -8.0)
        self.assertEqual(validator.MAX_SPEED, 120.0)
        self.assertEqual(validator.MAX_RPM, 4000)


if __name__ == '__main__':
    unittest.main()
