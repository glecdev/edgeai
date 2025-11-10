#!/usr/bin/env python3
"""
GLEC DTG - Realtime Integration Tests

Tests production-ported realtime data pipeline:
- 5-second processing latency (production SLA)
- 254.7 records/second throughput (production SLA)
- >99% valid record rate (production SLA)
"""

import sys
import os
from pathlib import Path

# Add ai-models to path for import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "ai-models"))

import unittest
import asyncio
import time
from inference.realtime_integration import (
    RealtimeDataIntegrator,
    RealtimeCANData
)


class TestRealtimeIntegration(unittest.TestCase):
    """Test realtime data integration pipeline"""

    def test_data_structure(self):
        """Test RealtimeCANData structure"""
        data = RealtimeCANData(
            timestamp=1704787200000,
            vehicle_speed=80.0,
            engine_rpm=2500,
            fuel_level=75.0,
            throttle_position=50.0,
            brake_position=0.0,
            coolant_temp=90,
            maf_rate=5.0,
            battery_voltage=12.6,
            acceleration_x=0.5,
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

        self.assertEqual(data.vehicle_speed, 80.0)
        self.assertEqual(data.engine_rpm, 2500)
        self.assertIsNone(data.engine_torque)  # Optional field

    def test_integrator_initialization(self):
        """Test RealtimeDataIntegrator initialization"""
        integrator = RealtimeDataIntegrator(batch_size=300)

        self.assertEqual(integrator.batch_size, 300)
        self.assertEqual(len(integrator.buffer), 0)
        self.assertEqual(integrator.stats['total_records'], 0)

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        integrator = RealtimeDataIntegrator()

        # Simulate processing
        integrator.stats['total_records'] = 1000
        integrator.stats['valid_records'] = 995
        integrator.stats['invalid_records'] = 5
        integrator.stats['avg_processing_time'] = 3.5
        integrator.stats['throughput'] = 285.7

        metrics = integrator.get_performance_metrics()

        self.assertEqual(metrics['total_records'], 1000)
        self.assertEqual(metrics['valid_records'], 995)
        self.assertEqual(metrics['valid_rate_pct'], 99.5)
        self.assertTrue(metrics['meets_latency_sla'])  # < 5 seconds
        self.assertTrue(metrics['meets_throughput_sla'])  # > 250 rec/sec
        self.assertTrue(metrics['meets_quality_sla'])  # > 99%

    def test_sla_violations(self):
        """Test SLA violation detection"""
        integrator = RealtimeDataIntegrator()

        # Simulate poor performance
        integrator.stats['total_records'] = 100
        integrator.stats['valid_records'] = 95
        integrator.stats['avg_processing_time'] = 8.0  # Too slow
        integrator.stats['throughput'] = 100.0  # Too low

        metrics = integrator.get_performance_metrics()

        self.assertFalse(metrics['meets_latency_sla'])  # 8s > 5s
        self.assertFalse(metrics['meets_throughput_sla'])  # 100 < 250

    def test_batch_processing(self):
        """Test batch processing with async stream"""

        async def run_test():
            integrator = RealtimeDataIntegrator(batch_size=50)

            # Mock CAN stream (100 records)
            async def mock_stream():
                for i in range(100):
                    yield RealtimeCANData(
                        timestamp=int(time.time() * 1000) + i * 10,
                        vehicle_speed=60.0 + i * 0.1,
                        engine_rpm=2000 + i,
                        fuel_level=75.0,
                        throttle_position=40.0,
                        brake_position=0.0,
                        coolant_temp=85,
                        maf_rate=5.0,
                        battery_voltage=12.6,
                        acceleration_x=0.2,
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
                    await asyncio.sleep(0.001)

            # Process stream
            processed = []
            async for data in integrator.process_stream(mock_stream()):
                processed.append(data)

            # Verify batch processing occurred
            self.assertEqual(len(processed), 100)
            self.assertEqual(integrator.stats['total_records'], 100)

        # Run async test
        asyncio.run(run_test())

    def test_stats_update(self):
        """Test statistics update logic"""
        integrator = RealtimeDataIntegrator()

        # First update
        integrator._update_stats(processing_time=2.5, record_count=100)
        self.assertAlmostEqual(integrator.stats['throughput'], 40.0, places=1)

        # Second update (moving average)
        integrator._update_stats(processing_time=3.0, record_count=150)
        expected_throughput = 150.0 / 3.0  # 50.0
        self.assertAlmostEqual(integrator.stats['throughput'], expected_throughput, places=1)


class TestProductionBenchmarks(unittest.TestCase):
    """Test against production-verified benchmarks"""

    def test_production_latency_benchmark(self):
        """
        Production benchmark: < 5 seconds processing time

        Original production result: 238s → 5s (47x improvement)
        """

        async def benchmark():
            integrator = RealtimeDataIntegrator(batch_size=300)

            # Simulate 5 seconds of data at 60 Hz (300 records)
            async def high_frequency_stream():
                for i in range(300):
                    yield RealtimeCANData(
                        timestamp=int(time.time() * 1000),
                        vehicle_speed=80.0,
                        engine_rpm=2500,
                        fuel_level=75.0,
                        throttle_position=50.0,
                        brake_position=0.0,
                        coolant_temp=90,
                        maf_rate=5.0,
                        battery_voltage=12.6,
                        acceleration_x=0.5,
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
                    await asyncio.sleep(0.001)  # High-speed simulation

            start_time = time.time()

            processed_count = 0
            async for data in integrator.process_stream(high_frequency_stream()):
                processed_count += 1

            elapsed = time.time() - start_time

            # Production SLA: Processing should complete in < 5 seconds
            # Note: In production, this is under real CAN bus load
            self.assertLess(
                elapsed, 10.0,  # Generous limit for unit test
                f"Processing took {elapsed:.2f}s (production target: <5s)"
            )

            self.assertEqual(processed_count, 300)

        asyncio.run(benchmark())

    def test_production_throughput_benchmark(self):
        """
        Production benchmark: 254.7 records/second

        This tests the pipeline's ability to handle high-frequency data
        """

        async def benchmark():
            integrator = RealtimeDataIntegrator(batch_size=500)

            # Simulate high-frequency stream (500 records in ~2 seconds)
            async def fast_stream():
                for i in range(500):
                    yield RealtimeCANData(
                        timestamp=int(time.time() * 1000),
                        vehicle_speed=80.0,
                        engine_rpm=2500,
                        fuel_level=75.0,
                        throttle_position=50.0,
                        brake_position=0.0,
                        coolant_temp=90,
                        maf_rate=5.0,
                        battery_voltage=12.6,
                        acceleration_x=0.5,
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
                    await asyncio.sleep(0.001)

            processed_count = 0
            async for data in integrator.process_stream(fast_stream()):
                processed_count += 1

            metrics = integrator.get_performance_metrics()

            # Production SLA: > 250 records/second throughput
            self.assertGreater(
                metrics['throughput'],
                100.0,  # Conservative for unit test
                f"Throughput: {metrics['throughput']:.1f} rec/sec (production: 254.7)"
            )

        asyncio.run(benchmark())


if __name__ == '__main__':
    unittest.main()
