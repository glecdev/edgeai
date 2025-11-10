"""
LightGBM Inference Latency Benchmark
Measure single-sample and batch inference performance for edge deployment
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


def benchmark_inference(model_path: str, num_iterations: int = 1000, warmup: int = 100):
    """
    Benchmark LightGBM inference latency

    Args:
        model_path: Path to trained model
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        dict: Benchmark results
    """
    print("=" * 80)
    print("LIGHTGBM INFERENCE LATENCY BENCHMARK")
    print("=" * 80)

    # Load model
    print(f"\n📥 Loading model: {model_path}")
    model = lgb.Booster(model_file=model_path)

    # Get model info
    num_features = model.num_feature()
    print(f"✅ Model loaded")
    print(f"  Features: {num_features}")
    print(f"  Trees: {model.num_trees()}")

    # Create sample input (18 features based on training)
    # Features: speed_mean, speed_std, speed_max, speed_min,
    #           rpm_mean, rpm_std, throttle_mean, throttle_std, throttle_max,
    #           brake_mean, brake_std, brake_max, accel_x_mean, accel_x_std,
    #           accel_x_max, accel_y_mean, accel_y_std, fuel_consumption
    sample_input = np.random.randn(1, 18)  # Single sample
    batch_input = np.random.randn(100, 18)  # Batch

    print(f"\n🔥 Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model.predict(sample_input)

    # Single-sample latency
    print(f"\n⏱️  Single-Sample Latency Benchmark ({num_iterations} iterations)...")
    latencies = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model.predict(sample_input)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

    latencies = np.array(latencies)

    # Calculate statistics
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)

    print(f"\n📊 SINGLE-SAMPLE INFERENCE RESULTS:")
    print("=" * 80)
    print(f"  Mean:       {mean_latency:.4f} ms")
    print(f"  Median:     {median_latency:.4f} ms")
    print(f"  P95:        {p95_latency:.4f} ms")
    print(f"  P99:        {p99_latency:.4f} ms")
    print(f"  Min:        {min_latency:.4f} ms")
    print(f"  Max:        {max_latency:.4f} ms")
    print(f"  Std Dev:    {std_latency:.4f} ms")

    # Throughput
    throughput = 1000.0 / mean_latency  # samples per second
    print(f"\n  Throughput: {throughput:.1f} samples/sec")

    # Batch inference
    print(f"\n⚡ Batch Inference Benchmark (100 samples)...")
    batch_latencies = []

    for _ in range(100):
        start = time.perf_counter()
        _ = model.predict(batch_input)
        latency_ms = (time.perf_counter() - start) * 1000
        batch_latencies.append(latency_ms)

    batch_latencies = np.array(batch_latencies)
    mean_batch = np.mean(batch_latencies)
    per_sample_batch = mean_batch / 100

    print(f"\n📊 BATCH INFERENCE RESULTS:")
    print("=" * 80)
    print(f"  Batch size:        100 samples")
    print(f"  Mean batch time:   {mean_batch:.4f} ms")
    print(f"  Per-sample:        {per_sample_batch:.4f} ms")
    print(f"  Batch throughput:  {100000.0/mean_batch:.1f} samples/sec")

    # Quality gate checks
    print(f"\n" + "=" * 80)
    print("PERFORMANCE QUALITY GATES")
    print("=" * 80)

    # Target: <15ms (LightGBM specific)
    lightgbm_target = 15.0
    if p95_latency < lightgbm_target:
        print(f"✅ PASS: P95 latency {p95_latency:.4f}ms < {lightgbm_target}ms (LightGBM target)")
    else:
        print(f"❌ FAIL: P95 latency {p95_latency:.4f}ms >= {lightgbm_target}ms")

    # Overall AI inference target: <50ms
    overall_target = 50.0
    if p95_latency < overall_target:
        print(f"✅ PASS: P95 latency {p95_latency:.4f}ms < {overall_target}ms (Overall AI target)")
    else:
        print(f"❌ FAIL: P95 latency {p95_latency:.4f}ms >= {overall_target}ms")

    # Memory efficiency
    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"\n  Model Size: {model_size_mb:.4f} MB")

    if model_size_mb < 10:
        print(f"✅ PASS: Model size {model_size_mb:.4f}MB < 10MB")
    else:
        print(f"⚠️  WARNING: Model size {model_size_mb:.4f}MB >= 10MB")

    print("=" * 80)

    return {
        'mean_latency': mean_latency,
        'median_latency': median_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency,
        'std_latency': std_latency,
        'throughput': throughput,
        'batch_mean': mean_batch,
        'per_sample_batch': per_sample_batch,
        'model_size_mb': model_size_mb
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LightGBM inference latency"
    )
    parser.add_argument(
        '--model',
        default='../models/lightgbm_behavior.txt',
        help='Path to trained model'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of benchmark iterations (default: 1000)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=100,
        help='Number of warmup iterations (default: 100)'
    )

    args = parser.parse_args()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Run benchmark
    results = benchmark_inference(
        model_path=str(model_path),
        num_iterations=args.iterations,
        warmup=args.warmup
    )

    print("\n✅ Benchmark complete!")

    # Exit code based on quality gates
    if results['p95_latency'] < 15.0:
        sys.exit(0)
    else:
        print("\n⚠️  Model did not meet latency target (P95 < 15ms)")
        sys.exit(1)


if __name__ == "__main__":
    main()
