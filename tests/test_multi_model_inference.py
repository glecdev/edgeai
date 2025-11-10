#!/usr/bin/env python3
"""
GLEC DTG - Multi-Model AI Inference Tests

Tests for multi-model AI inference pipeline (LightGBM + TCN + LSTM-AE).
Validates fuel efficiency prediction, anomaly detection, and orchestration.

Run: pytest tests/test_multi_model_inference.py -v
"""

import unittest
import numpy as np
from typing import List, Tuple


class InferenceResult:
    """Python representation of Kotlin InferenceResult for testing"""

    def __init__(
        self,
        behavior: str,
        confidence: float = 1.0,
        fuel_efficiency: float = 0.0,
        anomaly_score: float = 0.0,
        is_anomaly: bool = False,
        latency_ms: int = 0,
        timestamp: int = 0
    ):
        self.behavior = behavior
        self.confidence = confidence
        self.fuel_efficiency = fuel_efficiency
        self.anomaly_score = anomaly_score
        self.is_anomaly = is_anomaly
        self.latency_ms = latency_ms
        self.timestamp = timestamp

    def meets_latency_target(self) -> bool:
        """Multi-model inference should complete in < 50ms"""
        return self.latency_ms < 50

    def is_realistic_fuel_efficiency(self) -> bool:
        """Fuel efficiency should be in realistic range (3-20 L/100km)"""
        return 3.0 <= self.fuel_efficiency <= 20.0


class TCNEngine:
    """
    Python implementation of TCNEngine for testing.
    Mirrors Kotlin stub prediction logic.
    """

    SEQUENCE_LENGTH = 60
    FEATURE_DIM = 10

    def predict_fuel_efficiency(self, sequence: np.ndarray) -> float:
        """
        Predict fuel efficiency from temporal sequence

        Physics-based estimation: Fuel (L/100km) ≈ (RPM × throttle × 0.01) / (speed + 1)

        Args:
            sequence: 60×10 temporal feature array

        Returns:
            Predicted fuel efficiency in L/100km
        """
        if sequence.shape[0] == 0 or sequence.shape[1] < 3:
            return 8.0  # Default average

        total_fuel = 0.0
        valid_samples = 0

        for features in sequence:
            speed = features[0]      # km/h
            rpm = features[1]        # RPM
            throttle = features[2]   # %

            # Simplified fuel formula (calibrated for realistic stub values)
            fuel_rate = (rpm * throttle * 0.01) / (speed + 1.0)
            total_fuel += fuel_rate
            valid_samples += 1

        avg_fuel = total_fuel / valid_samples if valid_samples > 0 else 8.0

        # Clamp to realistic range (3-20 L/100km)
        return max(3.0, min(20.0, avg_fuel))


class LSTMAEEngine:
    """
    Python implementation of LSTMAEEngine for testing.
    Mirrors Kotlin stub detection logic.
    """

    SEQUENCE_LENGTH = 60
    FEATURE_DIM = 10
    ANOMALY_THRESHOLD = 0.15

    def detect_anomalies(self, sequence: np.ndarray) -> Tuple[float, bool]:
        """
        Detect anomalies in temporal sequence

        Args:
            sequence: 60×10 temporal feature array

        Returns:
            Tuple of (anomaly_score, is_anomaly)
        """
        if sequence.shape[0] == 0 or sequence.shape[1] < 3:
            return (0.0, False)

        anomaly_score = 0.0
        anomaly_count = 0

        # Check for sudden changes in key features
        for i in range(1, len(sequence)):
            prev = sequence[i - 1]
            curr = sequence[i]

            # Speed deviation
            speed_change = abs(curr[0] - prev[0])
            if speed_change > 30.0:  # > 30 km/h sudden change
                anomaly_score += 0.3
                anomaly_count += 1

            # RPM deviation
            rpm_change = abs(curr[1] - prev[1])
            if rpm_change > 1000.0:  # > 1000 RPM sudden change
                anomaly_score += 0.2
                anomaly_count += 1

            # Throttle spike
            throttle = curr[2]
            if throttle > 95.0:  # Full throttle
                anomaly_score += 0.1

        # Calculate variance across sequence
        speed_variance = np.var(sequence[:, 0])
        if speed_variance > 200.0:  # High speed variance
            anomaly_score += 0.2
            anomaly_count += 1

        # Normalize score (divide by max reasonable score to keep in [0, 1] range)
        # Max score estimation: ~3-5 anomalies × 0.3 each = ~1.5
        normalized_score = min(1.0, max(0.0, anomaly_score / 2.0))

        is_anomaly = normalized_score > self.ANOMALY_THRESHOLD or anomaly_count > 3

        return (normalized_score, is_anomaly)


class MultiModelInferenceService:
    """
    Python implementation of multi-model inference orchestration for testing.
    Mirrors Kotlin EdgeAIInferenceService behavior.
    """

    def __init__(self):
        self.tcn_engine = TCNEngine()
        self.lstmae_engine = LSTMAEEngine()

    def run_inference(self, temporal_sequence: np.ndarray) -> InferenceResult:
        """
        Run multi-model inference

        Args:
            temporal_sequence: 60×10 temporal feature array

        Returns:
            InferenceResult with all model outputs
        """
        import time

        start_time = time.time()

        # 1. TCN: Fuel efficiency prediction
        fuel_efficiency = self.tcn_engine.predict_fuel_efficiency(temporal_sequence)

        # 2. LSTM-AE: Anomaly detection
        anomaly_score, is_anomaly = self.lstmae_engine.detect_anomalies(temporal_sequence)

        # 3. LightGBM: Behavior classification (simulated)
        behavior = "NORMAL"
        confidence = 0.95

        latency_ms = int((time.time() - start_time) * 1000)

        return InferenceResult(
            behavior=behavior,
            confidence=confidence,
            fuel_efficiency=fuel_efficiency,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            latency_ms=latency_ms,
            timestamp=int(time.time() * 1000)
        )


class TestTCNEngine(unittest.TestCase):
    """Test TCN fuel efficiency prediction"""

    def setUp(self):
        self.engine = TCNEngine()

    def test_normal_driving_fuel_efficiency(self):
        """Test fuel efficiency for normal highway driving"""
        # Normal highway: 80 km/h, 2000 RPM, 30% throttle
        sequence = np.array([
            [80.0, 2000.0, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        fuel = self.engine.predict_fuel_efficiency(sequence)

        # Highway driving should have low fuel consumption
        self.assertGreater(fuel, 3.0, "Fuel too low")
        self.assertLess(fuel, 12.0, "Fuel too high for highway")

    def test_city_driving_fuel_efficiency(self):
        """Test fuel efficiency for city driving"""
        # City: 40 km/h, 1500 RPM, 40% throttle
        sequence = np.array([
            [40.0, 1500.0, 40.0, 60.0, 88.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        fuel = self.engine.predict_fuel_efficiency(sequence)

        # City driving should have moderate fuel consumption
        self.assertGreater(fuel, 6.0, "Fuel too low for city")
        self.assertLess(fuel, 18.0, "Fuel too high")

    def test_aggressive_driving_fuel_efficiency(self):
        """Test fuel efficiency for aggressive driving"""
        # Aggressive: 120 km/h, 4000 RPM, 80% throttle
        sequence = np.array([
            [120.0, 4000.0, 80.0, 50.0, 95.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        fuel = self.engine.predict_fuel_efficiency(sequence)

        # Aggressive driving should have high fuel consumption
        self.assertGreater(fuel, 12.0, "Fuel too low for aggressive driving")
        self.assertLessEqual(fuel, 20.0, "Fuel exceeds max realistic value")

    def test_idle_fuel_efficiency(self):
        """Test fuel efficiency at idle (zero speed)"""
        # Idle: 0 km/h, 800 RPM, 0% throttle
        sequence = np.array([
            [0.0, 800.0, 0.0, 100.0, 85.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        fuel = self.engine.predict_fuel_efficiency(sequence)

        # Should return realistic value (not crash on division by zero)
        # With 0 throttle, formula gives 0, clamped to minimum 3.0
        self.assertGreaterEqual(fuel, 3.0, "Fuel below minimum")
        self.assertLessEqual(fuel, 20.0, "Fuel too high")

    def test_fuel_efficiency_range(self):
        """Test fuel efficiency is always in realistic range"""
        # Random driving scenarios
        for _ in range(10):
            speed = np.random.uniform(0, 150)
            rpm = np.random.uniform(600, 5000)
            throttle = np.random.uniform(0, 100)

            sequence = np.array([
                [speed, rpm, throttle, 70.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0]
                for _ in range(60)
            ])

            fuel = self.engine.predict_fuel_efficiency(sequence)

            # Must be in realistic range
            self.assertGreaterEqual(fuel, 3.0, f"Fuel {fuel} below minimum")
            self.assertLessEqual(fuel, 20.0, f"Fuel {fuel} above maximum")


class TestLSTMAEEngine(unittest.TestCase):
    """Test LSTM-AE anomaly detection"""

    def setUp(self):
        self.engine = LSTMAEEngine()

    def test_normal_driving_no_anomaly(self):
        """Test normal driving should not trigger anomaly"""
        # Normal: smooth speed, moderate RPM, steady throttle
        sequence = np.array([
            [80.0 + i * 0.1, 2000.0 + i * 2, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for i in range(60)
        ])

        score, is_anomaly = self.engine.detect_anomalies(sequence)

        self.assertLess(score, 0.15, "Normal driving flagged as anomaly")
        self.assertFalse(is_anomaly, "Normal driving should not be anomaly")

    def test_sudden_speed_change_anomaly(self):
        """Test sudden speed change triggers anomaly"""
        sequence = []
        for i in range(30):
            sequence.append([80.0, 2000.0, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0])
        for i in range(30):
            # Sudden jump to 120 km/h (40 km/h change)
            sequence.append([120.0, 3000.0, 60.0, 70.0, 92.0, 0.0, 0.0, 0.0, 9.8, 0.0])

        sequence = np.array(sequence)
        score, is_anomaly = self.engine.detect_anomalies(sequence)

        self.assertGreater(score, 0.0, "Speed change not detected")
        self.assertTrue(is_anomaly, "Sudden speed change should be anomaly")

    def test_sudden_rpm_change_anomaly(self):
        """Test sudden RPM change triggers anomaly"""
        sequence = []
        # Create multiple sudden RPM jumps
        for i in range(15):
            sequence.append([80.0, 2000.0, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0])
        for i in range(15):
            # Sudden jump to 3500 RPM (1500 RPM change)
            sequence.append([80.0, 3500.0, 60.0, 70.0, 92.0, 0.0, 0.0, 0.0, 9.8, 0.0])
        for i in range(15):
            # Jump back to 2000 RPM
            sequence.append([80.0, 2000.0, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0])
        for i in range(15):
            # Jump to 3500 again
            sequence.append([80.0, 3500.0, 60.0, 70.0, 92.0, 0.0, 0.0, 0.0, 9.8, 0.0])

        sequence = np.array(sequence)
        score, is_anomaly = self.engine.detect_anomalies(sequence)

        self.assertGreater(score, 0.0, "RPM change not detected")
        # With 3 transitions of 1500 RPM each, should trigger anomaly
        self.assertTrue(is_anomaly, "Multiple sudden RPM changes should trigger anomaly")

    def test_full_throttle_anomaly(self):
        """Test full throttle triggers anomaly score increase"""
        # Full throttle for extended period
        sequence = np.array([
            [100.0, 4000.0, 98.0, 60.0, 95.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        score, _ = self.engine.detect_anomalies(sequence)

        # Full throttle should increase score
        self.assertGreater(score, 0.0, "Full throttle not detected")

    def test_high_variance_anomaly(self):
        """Test high speed variance triggers anomaly"""
        # Highly variable speed (city stop-and-go)
        sequence = []
        for i in range(60):
            speed = 60.0 if i % 2 == 0 else 10.0  # Alternating speeds
            sequence.append([speed, 2000.0, 40.0, 70.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0])

        sequence = np.array(sequence)
        score, is_anomaly = self.engine.detect_anomalies(sequence)

        self.assertGreater(score, 0.0, "High variance not detected")

    def test_anomaly_score_range(self):
        """Test anomaly score is always in valid range [0, 1]"""
        # Random scenarios
        for _ in range(10):
            sequence = np.random.uniform(
                low=[0, 600, 0, 0, 40, 0, -10, -10, 0, -45],
                high=[150, 5000, 100, 100, 120, 100, 10, 10, 20, 45],
                size=(60, 10)
            )

            score, _ = self.engine.detect_anomalies(sequence)

            self.assertGreaterEqual(score, 0.0, f"Score {score} below 0")
            self.assertLessEqual(score, 1.0, f"Score {score} above 1")


class TestMultiModelInference(unittest.TestCase):
    """Test multi-model inference orchestration"""

    def setUp(self):
        self.service = MultiModelInferenceService()

    def test_multi_model_inference_normal_driving(self):
        """Test multi-model inference for normal driving"""
        # Normal highway driving
        sequence = np.array([
            [80.0, 2000.0, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        result = self.service.run_inference(sequence)

        # Check all fields populated
        self.assertIsNotNone(result.behavior)
        self.assertGreater(result.confidence, 0.0)
        self.assertGreater(result.fuel_efficiency, 0.0)
        self.assertGreaterEqual(result.anomaly_score, 0.0)
        self.assertIsInstance(result.is_anomaly, bool)

        # Check realistic values
        self.assertTrue(result.is_realistic_fuel_efficiency())
        self.assertFalse(result.is_anomaly, "Normal driving should not be anomaly")

    def test_multi_model_inference_aggressive_driving(self):
        """Test multi-model inference for aggressive driving"""
        # Aggressive: high speed, high RPM, full throttle
        sequence = np.array([
            [120.0, 4000.0, 95.0, 50.0, 95.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        result = self.service.run_inference(sequence)

        # Should detect high fuel consumption
        self.assertGreater(result.fuel_efficiency, 12.0, "Aggressive driving should have high fuel")

        # May trigger anomaly
        self.assertGreaterEqual(result.anomaly_score, 0.0)

    def test_multi_model_inference_with_anomaly(self):
        """Test multi-model inference with anomaly scenario"""
        # Create sequence with sudden speed change
        sequence = []
        for i in range(30):
            sequence.append([60.0, 2000.0, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0])
        for i in range(30):
            sequence.append([110.0, 3500.0, 80.0, 70.0, 95.0, 0.0, 0.0, 0.0, 9.8, 0.0])

        sequence = np.array(sequence)
        result = self.service.run_inference(sequence)

        # Should detect anomaly
        self.assertGreater(result.anomaly_score, 0.0, "Anomaly not detected")
        self.assertTrue(result.is_anomaly, "Sudden change should trigger anomaly")

    def test_inference_result_latency_target(self):
        """Test multi-model inference meets latency target"""
        # Normal sequence
        sequence = np.array([
            [80.0, 2000.0, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        # Run multiple times and check latency
        latencies = []
        for _ in range(10):
            result = self.service.run_inference(sequence)
            latencies.append(result.latency_ms)

        avg_latency = sum(latencies) / len(latencies)

        # Python stub should be very fast (< 10ms)
        # Real ONNX multi-model target: < 50ms
        self.assertLess(avg_latency, 50, f"Average latency {avg_latency}ms exceeds target")

    def test_inference_result_structure(self):
        """Test InferenceResult has all required fields"""
        sequence = np.array([
            [80.0, 2000.0, 30.0, 75.0, 90.0, 0.0, 0.0, 0.0, 9.8, 0.0]
            for _ in range(60)
        ])

        result = self.service.run_inference(sequence)

        # Check all fields exist
        self.assertTrue(hasattr(result, 'behavior'))
        self.assertTrue(hasattr(result, 'confidence'))
        self.assertTrue(hasattr(result, 'fuel_efficiency'))
        self.assertTrue(hasattr(result, 'anomaly_score'))
        self.assertTrue(hasattr(result, 'is_anomaly'))
        self.assertTrue(hasattr(result, 'latency_ms'))
        self.assertTrue(hasattr(result, 'timestamp'))

        # Check types
        self.assertIsInstance(result.behavior, str)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.fuel_efficiency, float)
        self.assertIsInstance(result.anomaly_score, float)
        self.assertIsInstance(result.is_anomaly, bool)
        self.assertIsInstance(result.latency_ms, int)
        self.assertIsInstance(result.timestamp, int)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTCNEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMAEEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiModelInference))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
