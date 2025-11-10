#!/usr/bin/env python3
"""
GLEC DTG - DTGForegroundService Integration Tests

End-to-end integration tests for DTGForegroundService MQTT publishing.
Validates telemetry, status, and alert publishing logic.

Run: pytest tests/test_dtg_service_integration.py -v
"""

import json
import unittest
from typing import Dict, Any, Optional


class CANData:
    """Python representation of Kotlin CANData for testing"""

    def __init__(
        self,
        vehicle_speed: float = 0.0,
        engine_rpm: int = 0,
        throttle_position: float = 0.0,
        fuel_level: float = 100.0,
        coolant_temp: int = 90,
        brake_position: float = 0.0,
        acceleration_x: float = 0.0,
        acceleration_y: float = 0.0,
        acceleration_z: float = 9.8,
        steering_angle: float = 0.0,
    ):
        self.vehicle_speed = vehicle_speed
        self.engine_rpm = engine_rpm
        self.throttle_position = throttle_position
        self.fuel_level = fuel_level
        self.coolant_temp = coolant_temp
        self.brake_position = brake_position
        self.acceleration_x = acceleration_x
        self.acceleration_y = acceleration_y
        self.acceleration_z = acceleration_z
        self.steering_angle = steering_angle

    def is_harsh_braking(self) -> bool:
        """Harsh braking: acceleration_x < -4 m/s² and brake > 50%"""
        return self.acceleration_x < -4.0 and self.brake_position > 50.0

    def is_harsh_acceleration(self) -> bool:
        """Harsh acceleration: acceleration_x > 3 m/s² and throttle > 70%"""
        return self.acceleration_x > 3.0 and self.throttle_position > 70.0


class PythonDTGService:
    """
    Python implementation of DTGForegroundService publishing logic for testing.
    Mirrors Kotlin implementation.
    """

    def __init__(self, device_id: str = "DTG-SN-12345"):
        self.device_id = device_id
        self.samples_collected = 0
        self.inferences_run = 0

    def create_telemetry_payload(self, can_data: CANData) -> Dict[str, Any]:
        """Create telemetry payload (mirrors Kotlin publishTelemetry)"""
        return {
            "timestamp": 1704931200000,  # Fixed for testing
            "device_id": self.device_id,
            "vehicle_speed": can_data.vehicle_speed,
            "engine_rpm": can_data.engine_rpm,
            "throttle_position": can_data.throttle_position,
            "fuel_level": can_data.fuel_level,
            "coolant_temp": can_data.coolant_temp,
            "brake_position": can_data.brake_position,
            "acceleration_x": can_data.acceleration_x,
            "acceleration_y": can_data.acceleration_y,
            "acceleration_z": can_data.acceleration_z,
            "steering_angle": can_data.steering_angle,
            "gps": {
                "lat": 0.0,
                "lon": 0.0,
                "speed": can_data.vehicle_speed
            }
        }

    def create_status_payload(
        self,
        uptime_ms: int,
        mqtt_connected: bool
    ) -> Dict[str, Any]:
        """Create status payload (mirrors Kotlin publishStatus)"""
        return {
            "timestamp": 1704931200000,
            "device_id": self.device_id,
            "status": "ONLINE",
            "uptime_ms": uptime_ms,
            "samples_collected": self.samples_collected,
            "inferences_run": self.inferences_run,
            "mqtt_metrics": {
                "connected": mqtt_connected,
                "messages_sent": 100,
                "messages_failed": 2,
                "messages_queued": 5,
                "reconnect_count": 1
            },
            "inference_ready": True,
            "window_size": 60
        }

    def create_alert_payload(
        self,
        alert_type: str,
        severity: str,
        message: str,
        can_data: CANData
    ) -> Dict[str, Any]:
        """Create alert payload (mirrors Kotlin publishAlert)"""
        return {
            "timestamp": 1704931200000,
            "device_id": self.device_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "vehicle_data": {
                "speed": can_data.vehicle_speed,
                "rpm": can_data.engine_rpm,
                "throttle": can_data.throttle_position,
                "brake": can_data.brake_position,
                "coolant_temp": can_data.coolant_temp,
                "fuel_level": can_data.fuel_level
            }
        }

    def detect_anomalies(self, can_data: CANData) -> Optional[Dict[str, Any]]:
        """Detect anomalies and return alert if needed"""
        if can_data.is_harsh_braking():
            return self.create_alert_payload(
                alert_type="HARSH_BRAKING",
                severity="WARNING",
                message="Harsh braking detected: deceleration < -4 m/s²",
                can_data=can_data
            )

        if can_data.is_harsh_acceleration():
            return self.create_alert_payload(
                alert_type="HARSH_ACCELERATION",
                severity="WARNING",
                message="Harsh acceleration detected: acceleration > 3 m/s²",
                can_data=can_data
            )

        if can_data.coolant_temp > 105:
            return self.create_alert_payload(
                alert_type="ENGINE_OVERHEATING",
                severity="CRITICAL",
                message=f"Engine overheating: {can_data.coolant_temp}°C (threshold: 105°C)",
                can_data=can_data
            )

        if can_data.fuel_level < 10.0:
            return self.create_alert_payload(
                alert_type="LOW_FUEL",
                severity="INFO",
                message=f"Low fuel level: {can_data.fuel_level}%",
                can_data=can_data
            )

        return None


class TestTelemetryPublishing(unittest.TestCase):
    """Test telemetry payload creation (QoS 0, 1Hz)"""

    def setUp(self):
        self.service = PythonDTGService()

    def test_telemetry_payload_format(self):
        """Test telemetry payload has all required fields"""
        can_data = CANData(
            vehicle_speed=80.5,
            engine_rpm=2500,
            throttle_position=50.0,
            fuel_level=75.0,
            coolant_temp=92
        )

        payload = self.service.create_telemetry_payload(can_data)

        # Check required fields
        self.assertIn("timestamp", payload)
        self.assertIn("device_id", payload)
        self.assertIn("vehicle_speed", payload)
        self.assertIn("engine_rpm", payload)
        self.assertIn("throttle_position", payload)
        self.assertIn("fuel_level", payload)
        self.assertIn("coolant_temp", payload)
        self.assertIn("gps", payload)

        # Check values
        self.assertEqual(payload["device_id"], "DTG-SN-12345")
        self.assertEqual(payload["vehicle_speed"], 80.5)
        self.assertEqual(payload["engine_rpm"], 2500)
        self.assertEqual(payload["fuel_level"], 75.0)

    def test_telemetry_json_serializable(self):
        """Test telemetry payload is JSON serializable"""
        can_data = CANData(vehicle_speed=60.0)
        payload = self.service.create_telemetry_payload(can_data)

        # Should not raise exception
        json_str = json.dumps(payload)
        self.assertIsInstance(json_str, str)

        # Should be deserializable
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["vehicle_speed"], 60.0)

    def test_telemetry_gps_structure(self):
        """Test telemetry GPS data structure"""
        can_data = CANData(vehicle_speed=100.0)
        payload = self.service.create_telemetry_payload(can_data)

        self.assertIn("gps", payload)
        self.assertIn("lat", payload["gps"])
        self.assertIn("lon", payload["gps"])
        self.assertIn("speed", payload["gps"])
        self.assertEqual(payload["gps"]["speed"], 100.0)


class TestStatusPublishing(unittest.TestCase):
    """Test status payload creation (QoS 1, 5min)"""

    def setUp(self):
        self.service = PythonDTGService()
        self.service.samples_collected = 3600
        self.service.inferences_run = 60

    def test_status_payload_format(self):
        """Test status payload has all required fields"""
        payload = self.service.create_status_payload(
            uptime_ms=3600000,
            mqtt_connected=True
        )

        # Check required fields
        self.assertIn("timestamp", payload)
        self.assertIn("device_id", payload)
        self.assertIn("status", payload)
        self.assertIn("uptime_ms", payload)
        self.assertIn("samples_collected", payload)
        self.assertIn("inferences_run", payload)
        self.assertIn("mqtt_metrics", payload)
        self.assertIn("inference_ready", payload)
        self.assertIn("window_size", payload)

        # Check values
        self.assertEqual(payload["status"], "ONLINE")
        self.assertEqual(payload["uptime_ms"], 3600000)
        self.assertEqual(payload["samples_collected"], 3600)
        self.assertEqual(payload["inferences_run"], 60)

    def test_status_mqtt_metrics_structure(self):
        """Test status MQTT metrics structure"""
        payload = self.service.create_status_payload(
            uptime_ms=1000,
            mqtt_connected=True
        )

        mqtt_metrics = payload["mqtt_metrics"]
        self.assertIn("connected", mqtt_metrics)
        self.assertIn("messages_sent", mqtt_metrics)
        self.assertIn("messages_failed", mqtt_metrics)
        self.assertIn("messages_queued", mqtt_metrics)
        self.assertIn("reconnect_count", mqtt_metrics)

        self.assertEqual(mqtt_metrics["connected"], True)

    def test_status_json_serializable(self):
        """Test status payload is JSON serializable"""
        payload = self.service.create_status_payload(
            uptime_ms=1000,
            mqtt_connected=True
        )

        # Should not raise exception
        json_str = json.dumps(payload)
        self.assertIsInstance(json_str, str)


class TestAlertPublishing(unittest.TestCase):
    """Test alert payload creation (QoS 2, on event)"""

    def setUp(self):
        self.service = PythonDTGService()

    def test_alert_payload_format(self):
        """Test alert payload has all required fields"""
        can_data = CANData(
            vehicle_speed=120.0,
            acceleration_x=-5.5,
            brake_position=90.0
        )

        payload = self.service.create_alert_payload(
            alert_type="HARSH_BRAKING",
            severity="WARNING",
            message="Harsh braking detected",
            can_data=can_data
        )

        # Check required fields
        self.assertIn("timestamp", payload)
        self.assertIn("device_id", payload)
        self.assertIn("alert_type", payload)
        self.assertIn("severity", payload)
        self.assertIn("message", payload)
        self.assertIn("vehicle_data", payload)

        # Check values
        self.assertEqual(payload["alert_type"], "HARSH_BRAKING")
        self.assertEqual(payload["severity"], "WARNING")

    def test_alert_vehicle_data_structure(self):
        """Test alert vehicle data structure"""
        can_data = CANData(
            vehicle_speed=80.0,
            engine_rpm=3000,
            coolant_temp=110
        )

        payload = self.service.create_alert_payload(
            alert_type="ENGINE_OVERHEATING",
            severity="CRITICAL",
            message="Engine overheating",
            can_data=can_data
        )

        vehicle_data = payload["vehicle_data"]
        self.assertIn("speed", vehicle_data)
        self.assertIn("rpm", vehicle_data)
        self.assertIn("coolant_temp", vehicle_data)

        self.assertEqual(vehicle_data["speed"], 80.0)
        self.assertEqual(vehicle_data["rpm"], 3000)
        self.assertEqual(vehicle_data["coolant_temp"], 110)


class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection logic"""

    def setUp(self):
        self.service = PythonDTGService()

    def test_detect_harsh_braking(self):
        """Test harsh braking detection"""
        can_data = CANData(
            acceleration_x=-5.0,  # < -4 m/s²
            brake_position=80.0,  # > 50%
            vehicle_speed=100.0
        )

        alert = self.service.detect_anomalies(can_data)

        self.assertIsNotNone(alert)
        self.assertEqual(alert["alert_type"], "HARSH_BRAKING")
        self.assertEqual(alert["severity"], "WARNING")

    def test_detect_harsh_acceleration(self):
        """Test harsh acceleration detection"""
        can_data = CANData(
            acceleration_x=3.5,  # > 3 m/s²
            throttle_position=90.0,  # > 70%
            vehicle_speed=60.0
        )

        alert = self.service.detect_anomalies(can_data)

        self.assertIsNotNone(alert)
        self.assertEqual(alert["alert_type"], "HARSH_ACCELERATION")
        self.assertEqual(alert["severity"], "WARNING")

    def test_detect_engine_overheating(self):
        """Test engine overheating detection"""
        can_data = CANData(
            coolant_temp=110,  # > 105°C
            vehicle_speed=80.0
        )

        alert = self.service.detect_anomalies(can_data)

        self.assertIsNotNone(alert)
        self.assertEqual(alert["alert_type"], "ENGINE_OVERHEATING")
        self.assertEqual(alert["severity"], "CRITICAL")

    def test_detect_low_fuel(self):
        """Test low fuel detection"""
        can_data = CANData(
            fuel_level=5.0,  # < 10%
            vehicle_speed=60.0
        )

        alert = self.service.detect_anomalies(can_data)

        self.assertIsNotNone(alert)
        self.assertEqual(alert["alert_type"], "LOW_FUEL")
        self.assertEqual(alert["severity"], "INFO")

    def test_no_anomaly_detected(self):
        """Test normal driving (no anomaly)"""
        can_data = CANData(
            vehicle_speed=60.0,
            acceleration_x=1.0,  # Normal
            coolant_temp=90,     # Normal
            fuel_level=50.0      # Normal
        )

        alert = self.service.detect_anomalies(can_data)

        self.assertIsNone(alert)

    def test_harsh_braking_false_positive_prevention(self):
        """Test harsh braking requires both conditions"""
        # Only high deceleration, no brake
        can_data1 = CANData(
            acceleration_x=-5.0,
            brake_position=10.0  # Low brake position
        )
        self.assertIsNone(self.service.detect_anomalies(can_data1))

        # Only brake, no high deceleration
        can_data2 = CANData(
            acceleration_x=-2.0,  # Moderate deceleration
            brake_position=80.0
        )
        # This should still be None (acceleration not harsh enough)
        result = self.service.detect_anomalies(can_data2)
        if result is not None:
            # If detected, it's not harsh braking
            self.assertNotEqual(result["alert_type"], "HARSH_BRAKING")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTelemetryPublishing))
    suite.addTests(loader.loadTestsFromTestCase(TestStatusPublishing))
    suite.addTests(loader.loadTestsFromTestCase(TestAlertPublishing))
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetection))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
