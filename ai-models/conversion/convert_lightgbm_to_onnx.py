"""
LightGBM to ONNX Converter
Convert trained LightGBM model to ONNX format for mobile deployment

Phase 1 PoC: LightGBM → ONNX (web-compatible, no GPU required)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort


def convert_lightgbm_to_onnx(
    model_path: str,
    output_path: str,
    num_features: int = 18,
    num_classes: int = 3
) -> str:
    """
    Convert LightGBM model to ONNX format

    Args:
        model_path: Path to LightGBM text model (.txt)
        output_path: Path to save ONNX model (.onnx)
        num_features: Number of input features (default: 18)
        num_classes: Number of output classes (default: 3)

    Returns:
        Path to exported ONNX model
    """
    print("=" * 80)
    print("LIGHTGBM TO ONNX CONVERTER")
    print("=" * 80)

    # Load LightGBM model
    print(f"\n📥 Loading LightGBM model: {model_path}")
    model = lgb.Booster(model_file=model_path)

    print(f"✅ Model loaded successfully")
    print(f"  Features: {model.num_feature()}")
    print(f"  Trees: {model.num_trees()}")
    print(f"  Classes: {num_classes}")

    # Define input type for ONNX conversion
    # FloatTensorType([batch_size, num_features])
    # Use 'None' for dynamic batch size
    initial_types = [
        ('input', FloatTensorType([None, num_features]))
    ]

    print(f"\n🔄 Converting to ONNX...")
    print(f"  Input shape: (batch_size, {num_features})")
    print(f"  Output shape: (batch_size, {num_classes})")

    # Convert to ONNX
    try:
        onnx_model = onnxmltools.convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=13  # ONNX opset version
        )

        # Save ONNX model
        onnxmltools.utils.save_model(onnx_model, output_path)

        print(f"✅ ONNX conversion completed: {output_path}")

    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        sys.exit(1)

    # Get model size
    model_size_kb = Path(output_path).stat().st_size / 1024
    print(f"\n📊 ONNX Model Size: {model_size_kb:.2f} KB")

    return output_path


def validate_onnx_output(
    onnx_path: str,
    lgbm_path: str,
    num_features: int = 18,
    num_samples: int = 100
) -> bool:
    """
    Validate ONNX model output matches LightGBM model

    Args:
        onnx_path: Path to ONNX model
        lgbm_path: Path to LightGBM model
        num_features: Number of input features
        num_samples: Number of test samples

    Returns:
        True if outputs match within tolerance
    """
    print(f"\n🔍 Validating ONNX output against LightGBM...")

    # Load models
    lgbm_model = lgb.Booster(model_file=lgbm_path)
    ort_session = ort.InferenceSession(onnx_path)

    # Generate random test data
    np.random.seed(42)
    test_data = np.random.randn(num_samples, num_features).astype(np.float32)

    # LightGBM inference
    lgbm_output = lgbm_model.predict(test_data)  # Shape: (num_samples, num_classes)

    # ONNX Runtime inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_data}
    ort_outputs = ort_session.run(None, ort_inputs)

    # ONNX may return labels and probabilities
    # Get probability outputs
    if len(ort_outputs) == 2:
        # (labels, probabilities)
        onnx_output = ort_outputs[1]  # Probabilities
    else:
        onnx_output = ort_outputs[0]

    # Compare outputs
    if lgbm_output.shape != onnx_output.shape:
        print(f"⚠️  Shape mismatch: LightGBM {lgbm_output.shape} vs ONNX {onnx_output.shape}")

        # Handle shape differences
        if lgbm_output.ndim == 2 and onnx_output.ndim == 3:
            # ONNX may add batch dimension
            onnx_output = onnx_output.squeeze()

    max_diff = np.max(np.abs(lgbm_output - onnx_output))
    mean_diff = np.mean(np.abs(lgbm_output - onnx_output))
    relative_diff = mean_diff / (np.mean(np.abs(lgbm_output)) + 1e-7)

    print(f"  Max absolute difference:  {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Relative difference:      {relative_diff:.6f} ({relative_diff*100:.4f}%)")

    # Check predictions match
    lgbm_preds = lgbm_output.argmax(axis=1)
    onnx_preds = onnx_output.argmax(axis=1)
    accuracy = (lgbm_preds == onnx_preds).mean()

    print(f"  Prediction accuracy:      {accuracy:.6f} ({accuracy*100:.2f}%)")

    tolerance = 1e-3
    if max_diff < tolerance and accuracy > 0.99:
        print(f"✅ Validation PASSED (max_diff < {tolerance}, accuracy > 99%)")
        return True
    else:
        print(f"⚠️  Validation WARNING: Check output differences")
        return False


def benchmark_onnx_latency(
    onnx_path: str,
    num_features: int = 18,
    num_iterations: int = 1000,
    warmup: int = 100
) -> dict:
    """
    Benchmark ONNX model inference latency

    Args:
        onnx_path: Path to ONNX model
        num_features: Number of input features
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with latency metrics
    """
    print(f"\n⏱️  Benchmarking ONNX latency ({num_iterations} iterations)...")

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)

    # Create sample input
    sample_input = np.random.randn(1, num_features).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input}

    # Warmup
    print(f"  Warmup: {warmup} iterations...")
    for _ in range(warmup):
        _ = ort_session.run(None, ort_inputs)

    # Benchmark
    import time
    latencies = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

    latencies = np.array(latencies)

    metrics = {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }

    print(f"\n📊 ONNX Inference Latency:")
    print(f"  Mean:   {metrics['mean']:.4f} ms")
    print(f"  Median: {metrics['median']:.4f} ms")
    print(f"  P95:    {metrics['p95']:.4f} ms")
    print(f"  P99:    {metrics['p99']:.4f} ms")
    print(f"  Min:    {metrics['min']:.4f} ms")
    print(f"  Max:    {metrics['max']:.4f} ms")

    # Compare with original LightGBM latency (0.064ms P95)
    original_p95 = 0.064
    overhead = metrics['p95'] - original_p95

    print(f"\n  ONNX Overhead vs LightGBM:")
    print(f"    LightGBM P95: {original_p95:.4f} ms")
    print(f"    ONNX P95:     {metrics['p95']:.4f} ms")
    print(f"    Overhead:     {overhead:.4f} ms ({overhead/original_p95*100:.1f}%)")

    # Quality gate: P95 < 5ms (target for mobile)
    target_latency = 5.0
    if metrics['p95'] < target_latency:
        print(f"  ✅ PASS: P95 latency {metrics['p95']:.4f}ms < {target_latency}ms")
    else:
        print(f"  ⚠️  WARNING: P95 latency {metrics['p95']:.4f}ms >= {target_latency}ms")

    return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert LightGBM model to ONNX format"
    )
    parser.add_argument(
        '--input',
        default='../models/lightgbm_behavior.txt',
        help='Path to LightGBM model (.txt)'
    )
    parser.add_argument(
        '--output',
        default='../models/lightgbm_behavior.onnx',
        help='Path to save ONNX model (.onnx)'
    )
    parser.add_argument(
        '--num-features',
        type=int,
        default=18,
        help='Number of input features (default: 18)'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=3,
        help='Number of output classes (default: 3)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate ONNX output against LightGBM'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark ONNX inference latency'
    )

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input model not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to ONNX
    onnx_path = convert_lightgbm_to_onnx(
        model_path=str(input_path),
        output_path=str(output_path),
        num_features=args.num_features,
        num_classes=args.num_classes
    )

    # Validate if requested
    if args.validate:
        is_valid = validate_onnx_output(
            onnx_path=onnx_path,
            lgbm_path=str(input_path),
            num_features=args.num_features
        )

        if not is_valid:
            print("\n⚠️  Validation warnings detected. Review output differences.")

    # Benchmark if requested
    if args.benchmark:
        metrics = benchmark_onnx_latency(
            onnx_path=onnx_path,
            num_features=args.num_features
        )

    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)
    print(f"\nONNX model saved to: {onnx_path}")
    print(f"\nNext step:")
    print(f"  python convert_onnx_to_tflite.py --input {onnx_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
