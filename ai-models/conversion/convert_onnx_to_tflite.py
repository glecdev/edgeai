"""
ONNX to TFLite Converter
Convert ONNX model to TensorFlow Lite format for Android deployment

Phase 1 PoC: ONNX → TensorFlow → TFLite (web-compatible, no GPU required)
"""

import argparse
import sys
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf


def convert_onnx_to_tensorflow(
    onnx_path: str,
    tf_output_dir: str
) -> str:
    """
    Convert ONNX model to TensorFlow SavedModel format

    Note: This function wraps the onnx2tf command-line tool
    as it provides better compatibility than programmatic APIs

    Args:
        onnx_path: Path to ONNX model (.onnx)
        tf_output_dir: Directory to save TensorFlow model

    Returns:
        Path to TensorFlow SavedModel directory
    """
    import sys
    import subprocess
    import shutil

    print("=" * 80)
    print("ONNX TO TENSORFLOW CONVERTER")
    print("=" * 80)

    print(f"\n📥 Input ONNX model: {onnx_path}")
    print(f"📁 Output TensorFlow directory: {tf_output_dir}")

    # Check onnx2tf is installed
    try:
        import onnx2tf
        print(f"✅ onnx2tf version: {onnx2tf.__version__}")
    except ImportError:
        print("❌ onnx2tf not installed. Install with: pip install onnx2tf")
        sys.exit(1)

    # Create output directory
    tf_output_path = Path(tf_output_dir)
    if tf_output_path.exists():
        print(f"⚠️  Output directory exists, removing: {tf_output_dir}")
        shutil.rmtree(tf_output_dir)

    tf_output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🔄 Converting ONNX → TensorFlow...")
    print(f"  This may take 1-3 minutes...")

    # Use onnx2tf via python -m for best compatibility
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tf_output_dir,
        "-osd",  # Output SavedModel
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)

        if result.returncode == 0:
            print(f"✅ TensorFlow conversion completed: {tf_output_dir}")
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"❌ onnx2tf conversion failed:")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Python executable not found: {e}")
        print("Install onnx2tf with: pip install onnx2tf")
        sys.exit(1)

    return tf_output_dir


def convert_tensorflow_to_tflite(
    tf_model_dir: str,
    tflite_output_path: str,
    quantize: str = 'none',
    representative_dataset: np.ndarray = None
) -> str:
    """
    Convert TensorFlow SavedModel to TFLite format

    Args:
        tf_model_dir: Path to TensorFlow SavedModel directory
        tflite_output_path: Path to save TFLite model (.tflite)
        quantize: Quantization mode ('none', 'float16', 'int8')
        representative_dataset: Calibration data for INT8 quantization

    Returns:
        Path to TFLite model
    """
    print("\n" + "=" * 80)
    print("TENSORFLOW TO TFLITE CONVERTER")
    print("=" * 80)

    print(f"\n📥 Input TensorFlow model: {tf_model_dir}")
    print(f"📁 Output TFLite model: {tflite_output_path}")
    print(f"🔧 Quantization: {quantize}")

    # Load TensorFlow model
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
        print(f"✅ TensorFlow model loaded")
    except Exception as e:
        print(f"❌ Failed to load TensorFlow model: {e}")
        sys.exit(1)

    # Configure quantization
    if quantize == 'float16':
        print(f"\n🔧 Applying FP16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantize == 'int8':
        print(f"\n🔧 Applying INT8 quantization...")

        if representative_dataset is None:
            print("⚠️  No representative dataset provided, using random data")
            # Generate random calibration data
            def representative_dataset_gen():
                for _ in range(100):
                    # Assuming input shape (1, 18)
                    yield [np.random.randn(1, 18).astype(np.float32)]

            converter.representative_dataset = representative_dataset_gen
        else:
            def representative_dataset_gen():
                for sample in representative_dataset:
                    yield [sample.reshape(1, -1).astype(np.float32)]

            converter.representative_dataset = representative_dataset_gen

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    else:
        print(f"\n🔧 No quantization applied (FP32)")

    # Convert to TFLite
    print(f"\n🔄 Converting to TFLite...")

    try:
        tflite_model = converter.convert()
        print(f"✅ TFLite conversion completed")
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        sys.exit(1)

    # Save TFLite model
    output_path = Path(tflite_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Get model size
    model_size_kb = len(tflite_model) / 1024
    print(f"\n📊 TFLite Model Size: {model_size_kb:.2f} KB")

    # Quality gate: < 500KB (acceptable overhead from 22KB LightGBM)
    size_limit_kb = 500
    if model_size_kb < size_limit_kb:
        print(f"✅ PASS: Model size {model_size_kb:.2f}KB < {size_limit_kb}KB")
    else:
        print(f"⚠️  WARNING: Model size {model_size_kb:.2f}KB >= {size_limit_kb}KB")

    return str(output_path)


def benchmark_tflite_latency(
    tflite_path: str,
    num_features: int = 18,
    num_iterations: int = 1000,
    warmup: int = 100
) -> dict:
    """
    Benchmark TFLite model inference latency

    Args:
        tflite_path: Path to TFLite model
        num_features: Number of input features
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with latency metrics
    """
    print("\n" + "=" * 80)
    print(f"TFLITE LATENCY BENCHMARK ({num_iterations} iterations)")
    print("=" * 80)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\n📊 Model Info:")
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input dtype:  {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")

    # Create sample input
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.uint8:
        # INT8 quantized
        sample_input = np.random.randint(0, 256, (1, num_features), dtype=np.uint8)
    else:
        # FP32
        sample_input = np.random.randn(1, num_features).astype(np.float32)

    # Warmup
    print(f"\n🔥 Warmup: {warmup} iterations...")
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()

    # Benchmark
    print(f"⏱️  Benchmarking...")
    import time
    latencies = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        interpreter.invoke()
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

    print(f"\n📊 TFLite Inference Latency:")
    print(f"  Mean:   {metrics['mean']:.4f} ms")
    print(f"  Median: {metrics['median']:.4f} ms")
    print(f"  P95:    {metrics['p95']:.4f} ms")
    print(f"  P99:    {metrics['p99']:.4f} ms")
    print(f"  Min:    {metrics['min']:.4f} ms")
    print(f"  Max:    {metrics['max']:.4f} ms")

    # Compare with targets
    original_p95 = 0.064  # LightGBM original
    target_p95 = 5.0      # Mobile target
    overhead = metrics['p95'] - original_p95

    print(f"\n  Latency Comparison:")
    print(f"    LightGBM P95: {original_p95:.4f} ms")
    print(f"    TFLite P95:   {metrics['p95']:.4f} ms")
    print(f"    Overhead:     {overhead:.4f} ms ({overhead/original_p95*100:.1f}%)")
    print(f"    Target:       {target_p95:.4f} ms")

    # Quality gate
    if metrics['p95'] < target_p95:
        print(f"  ✅ PASS: P95 latency {metrics['p95']:.4f}ms < {target_p95}ms (target)")
    else:
        print(f"  ❌ FAIL: P95 latency {metrics['p95']:.4f}ms >= {target_p95}ms (target)")

    return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to TensorFlow Lite format"
    )
    parser.add_argument(
        '--input',
        default='../models/lightgbm_behavior.onnx',
        help='Path to ONNX model (.onnx)'
    )
    parser.add_argument(
        '--output',
        default='../models/lightgbm_behavior.tflite',
        help='Path to save TFLite model (.tflite)'
    )
    parser.add_argument(
        '--tf-dir',
        default='../models/lightgbm_tf',
        help='Temporary directory for TensorFlow model'
    )
    parser.add_argument(
        '--quantize',
        choices=['none', 'float16', 'int8'],
        default='none',
        help='Quantization mode (default: none)'
    )
    parser.add_argument(
        '--calibration-data',
        default=None,
        help='Path to calibration data for INT8 quantization (CSV)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark TFLite inference latency'
    )
    parser.add_argument(
        '--keep-tf',
        action='store_true',
        help='Keep intermediate TensorFlow model'
    )

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input ONNX model not found: {input_path}")
        sys.exit(1)

    # Load calibration data if provided
    representative_dataset = None
    if args.calibration_data and args.quantize == 'int8':
        print(f"📥 Loading calibration data: {args.calibration_data}")
        import pandas as pd
        calib_df = pd.read_csv(args.calibration_data)
        # Assume first 18 columns are features
        representative_dataset = calib_df.iloc[:1000, :18].values.astype(np.float32)
        print(f"✅ Loaded {len(representative_dataset)} calibration samples")

    # Step 1: ONNX → TensorFlow
    tf_model_dir = convert_onnx_to_tensorflow(
        onnx_path=str(input_path),
        tf_output_dir=args.tf_dir
    )

    # Step 2: TensorFlow → TFLite
    tflite_path = convert_tensorflow_to_tflite(
        tf_model_dir=tf_model_dir,
        tflite_output_path=args.output,
        quantize=args.quantize,
        representative_dataset=representative_dataset
    )

    # Benchmark if requested
    if args.benchmark:
        metrics = benchmark_tflite_latency(
            tflite_path=tflite_path
        )

    # Clean up intermediate TensorFlow model
    if not args.keep_tf:
        print(f"\n🗑️  Removing intermediate TensorFlow model: {tf_model_dir}")
        shutil.rmtree(tf_model_dir)

    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)
    print(f"\nTFLite model saved to: {tflite_path}")
    print(f"\nNext step:")
    print(f"  python validate_tflite_model.py --tflite {tflite_path} --original ../models/lightgbm_behavior.txt")

    sys.exit(0)


if __name__ == "__main__":
    main()
