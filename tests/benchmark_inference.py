#!/usr/bin/env python3
"""
GLEC DTG - AI Model Inference Benchmark

Measures inference performance:
- Latency (ms)
- Throughput (samples/second)
- Memory usage (MB)
- Model size (MB)
- Power consumption (if available)

Usage:
    python benchmark_inference.py --model tcn --iterations 1000
"""

import argparse
import time
import numpy as np
import psutil
import os
import sys
from typing import Dict, List
import torch


class InferenceBenchmark:
    """AI model inference benchmark"""

    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'

        self.results: Dict = {
            'model_name': model_name,
            'model_path': model_path,
            'model_size_mb': 0,
            'device': self.device,
            'latencies_ms': [],
            'memory_usage_mb': [],
            'throughput': 0
        }

    def load_model(self):
        """Load AI model"""
        print(f"Loading model: {self.model_path}")

        try:
            # Get model size
            self.results['model_size_mb'] = os.path.getsize(self.model_path) / (1024 * 1024)
            print(f"  Model size: {self.results['model_size_mb']:.2f} MB")

            # Load model (PyTorch example)
            self.model = torch.jit.load(self.model_path)
            self.model.eval()

            # Check CUDA availability
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model = self.model.cuda()
                print(f"  Device: {self.device} (GPU)")
            else:
                print(f"  Device: {self.device} (CPU)")

            self.results['device'] = self.device

            print("✓ Model loaded successfully")

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            sys.exit(1)

    def warmup(self, iterations: int = 10):
        """Warmup model"""
        print(f"\nWarming up model ({iterations} iterations)...")

        # Generate dummy input
        dummy_input = self._generate_dummy_input()

        for _ in range(iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)

        print("✓ Warmup complete")

    def benchmark(self, iterations: int = 1000, batch_size: int = 1):
        """Run inference benchmark"""
        print(f"\nRunning benchmark ({iterations} iterations, batch_size={batch_size})...")

        latencies = []
        memory_usage = []

        # Generate input batch
        input_batch = self._generate_dummy_input(batch_size)

        start_time = time.time()

        for i in range(iterations):
            # Measure latency
            iter_start = time.time()

            with torch.no_grad():
                _ = self.model(input_batch)

            iter_end = time.time()
            latency_ms = (iter_end - iter_start) * 1000
            latencies.append(latency_ms)

            # Measure memory usage every 100 iterations
            if i % 100 == 0:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_usage.append(memory_mb)

                # Progress update
                print(f"  Progress: {i}/{iterations} iterations, "
                      f"Latency: {latency_ms:.2f}ms, "
                      f"Memory: {memory_mb:.1f}MB")

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate metrics
        self.results['latencies_ms'] = latencies
        self.results['memory_usage_mb'] = memory_usage
        self.results['throughput'] = (iterations * batch_size) / total_time

        # Statistical analysis
        self.results['latency_mean'] = float(np.mean(latencies))
        self.results['latency_std'] = float(np.std(latencies))
        self.results['latency_min'] = float(np.min(latencies))
        self.results['latency_max'] = float(np.max(latencies))
        self.results['latency_p50'] = float(np.percentile(latencies, 50))
        self.results['latency_p95'] = float(np.percentile(latencies, 95))
        self.results['latency_p99'] = float(np.percentile(latencies, 99))

        self.results['memory_mean'] = float(np.mean(memory_usage))
        self.results['memory_max'] = float(np.max(memory_usage))

        print("✓ Benchmark complete")

    def _generate_dummy_input(self, batch_size: int = 1):
        """Generate dummy input for model"""
        if self.model_name == 'tcn':
            # TCN input: (batch_size, sequence_length, features)
            input_shape = (batch_size, 60, 18)  # 60 samples, 18 features
        elif self.model_name == 'lstm_ae':
            # LSTM-AE input: (batch_size, sequence_length, features)
            input_shape = (batch_size, 60, 18)
        elif self.model_name == 'lightgbm':
            # LightGBM input: (batch_size, features)
            input_shape = (batch_size, 20)  # 20 statistical features
        else:
            # Default
            input_shape = (batch_size, 60, 18)

        dummy_input = torch.randn(*input_shape)

        if self.device == 'cuda':
            dummy_input = dummy_input.cuda()

        return dummy_input

    def print_report(self):
        """Print benchmark report"""
        print("\n" + "="*60)
        print("INFERENCE BENCHMARK REPORT")
        print("="*60)

        print(f"\nModel: {self.results['model_name']}")
        print(f"Path: {self.results['model_path']}")
        print(f"Size: {self.results['model_size_mb']:.2f} MB")
        print(f"Device: {self.results['device']}")

        print("\nLatency Statistics (ms):")
        print(f"  Mean:       {self.results['latency_mean']:7.2f}")
        print(f"  Std Dev:    {self.results['latency_std']:7.2f}")
        print(f"  Min:        {self.results['latency_min']:7.2f}")
        print(f"  Max:        {self.results['latency_max']:7.2f}")
        print(f"  P50:        {self.results['latency_p50']:7.2f}")
        print(f"  P95:        {self.results['latency_p95']:7.2f}")
        print(f"  P99:        {self.results['latency_p99']:7.2f}")

        print(f"\nThroughput: {self.results['throughput']:.2f} samples/second")

        print("\nMemory Usage (MB):")
        print(f"  Mean:       {self.results['memory_mean']:7.1f}")
        print(f"  Peak:       {self.results['memory_max']:7.1f}")

        # Check against targets
        print("\nTarget Comparison:")

        if self.model_name == 'tcn':
            target_latency = 25
            target_size = 4
        elif self.model_name == 'lstm_ae':
            target_latency = 35
            target_size = 3
        elif self.model_name == 'lightgbm':
            target_latency = 15
            target_size = 10
        else:
            target_latency = 50
            target_size = 100

        latency_status = "✓ PASS" if self.results['latency_mean'] < target_latency else "✗ FAIL"
        size_status = "✓ PASS" if self.results['model_size_mb'] < target_size else "✗ FAIL"

        print(f"  Latency:    {latency_status} (target: <{target_latency}ms, actual: {self.results['latency_mean']:.2f}ms)")
        print(f"  Model Size: {size_status} (target: <{target_size}MB, actual: {self.results['model_size_mb']:.2f}MB)")

        print("\n" + "="*60)

    def save_results(self, output_path: str):
        """Save benchmark results to JSON"""
        import json

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='GLEC DTG Inference Benchmark')
    parser.add_argument('--model', type=str, required=True,
                        choices=['tcn', 'lstm_ae', 'lightgbm'],
                        help='Model to benchmark')
    parser.add_argument('--model-path', type=str,
                        help='Path to model file (auto-detected if not specified)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')
    parser.add_argument('--save-results', type=str,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Auto-detect model path if not specified
    if not args.model_path:
        model_paths = {
            'tcn': 'ai-models/models/tcn_fuel_int8.pt',
            'lstm_ae': 'ai-models/models/lstm_ae_int8.pt',
            'lightgbm': 'ai-models/models/lightgbm_behavior.txt'
        }
        args.model_path = model_paths.get(args.model, f'ai-models/models/{args.model}.pt')

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"✗ Model not found: {args.model_path}")
        print(f"  Please train the model first or specify correct path with --model-path")
        sys.exit(1)

    # Create benchmark
    benchmark = InferenceBenchmark(args.model, args.model_path)

    # Run benchmark
    benchmark.load_model()
    benchmark.warmup(args.warmup)
    benchmark.benchmark(args.iterations, args.batch_size)

    # Print report
    benchmark.print_report()

    # Save results if requested
    if args.save_results:
        benchmark.save_results(args.save_results)


if __name__ == '__main__':
    main()
