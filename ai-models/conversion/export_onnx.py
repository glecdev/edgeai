"""
GLEC DTG Edge AI - ONNX Export
Export PyTorch models to ONNX format for SNPE/TFLite conversion
"""

import os
import yaml
import argparse
import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "training"))
from train_tcn import TCN
from train_lstm_ae import LSTM_Autoencoder

from typing import Dict, Tuple


def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor,
                   output_path: str, config: Dict) -> str:
    """
    Export PyTorch model to ONNX format

    Args:
        model: PyTorch model
        dummy_input: Example input tensor
        output_path: Path to save ONNX file
        config: Configuration dictionary

    Returns:
        Path to exported ONNX file
    """
    print(f"Exporting model to ONNX: {output_path}")

    model.eval()

    # Dynamic axes for batch size
    dynamic_axes = config['onnx']['dynamic_axes']

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=config['onnx']['opset_version'],
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    print(f"✅ ONNX export completed: {output_path}")

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verification passed")

    return output_path


def simplify_onnx(onnx_path: str) -> str:
    """
    Simplify ONNX model for better compatibility

    Args:
        onnx_path: Path to ONNX file

    Returns:
        Path to simplified ONNX file
    """
    print(f"Simplifying ONNX model: {onnx_path}")

    # Load model
    onnx_model = onnx.load(onnx_path)

    # Simplify
    simplified_model, check = simplify(onnx_model)

    if not check:
        print("⚠️ Warning: Simplified model may not be valid")
    else:
        print("✅ ONNX simplification successful")

    # Save simplified model
    simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
    onnx.save(simplified_model, simplified_path)

    return simplified_path


def validate_onnx(onnx_path: str, pytorch_model: torch.nn.Module,
                  dummy_input: torch.Tensor, tolerance: float = 1e-3) -> bool:
    """
    Validate ONNX model output matches PyTorch model

    Args:
        onnx_path: Path to ONNX file
        pytorch_model: Original PyTorch model
        dummy_input: Example input tensor
        tolerance: Numerical tolerance for comparison

    Returns:
        True if outputs match within tolerance
    """
    print("Validating ONNX model against PyTorch...")

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < tolerance:
        print(f"✅ Validation passed (max_diff < {tolerance})")
        return True
    else:
        print(f"❌ Validation failed (max_diff >= {tolerance})")
        return False


def benchmark_onnx(onnx_path: str, dummy_input: torch.Tensor,
                   num_runs: int = 100) -> Dict:
    """
    Benchmark ONNX model performance

    Args:
        onnx_path: Path to ONNX file
        dummy_input: Example input tensor
        num_runs: Number of inference runs

    Returns:
        Dictionary with benchmark metrics
    """
    print(f"\nBenchmarking ONNX model: {num_runs} runs")

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, ort_inputs)

    # Benchmark
    import time
    latencies = []

    for _ in range(num_runs):
        start = time.time()
        _ = ort_session.run(None, ort_inputs)
        latencies.append((time.time() - start) * 1000)  # ms

    latencies = np.array(latencies)

    metrics = {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
    }

    print(f"Mean latency: {metrics['mean_latency_ms']:.2f} ± {metrics['std_latency_ms']:.2f} ms")
    print(f"P50: {metrics['p50_latency_ms']:.2f} ms")
    print(f"P95: {metrics['p95_latency_ms']:.2f} ms")
    print(f"P99: {metrics['p99_latency_ms']:.2f} ms")

    # Model size
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    metrics['model_size_mb'] = model_size_mb
    print(f"Model size: {model_size_mb:.2f} MB")

    return metrics


def export_tcn(config: Dict) -> str:
    """
    Export TCN model to ONNX

    Args:
        config: Configuration dictionary

    Returns:
        Path to exported ONNX file
    """
    print("=== Exporting TCN Model to ONNX ===\n")

    # Load model
    checkpoint = torch.load("models/tcn_fuel_best.pth", map_location='cpu')

    model = TCN(
        input_dim=config['tcn']['input_dim'],
        output_dim=config['tcn']['output_dim'],
        num_channels=config['tcn']['num_channels'],
        kernel_size=config['tcn']['kernel_size'],
        dropout=config['tcn']['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input (batch_size=1, sequence_length=60, input_dim=10)
    dummy_input = torch.randn(1, config['dataset']['window_size'],
                              config['tcn']['input_dim'])

    # Export to ONNX
    output_path = "models/tcn_fuel.onnx"
    os.makedirs("models", exist_ok=True)

    onnx_path = export_to_onnx(model, dummy_input, output_path, config)

    # Simplify ONNX
    if config['onnx']['optimization']:
        simplified_path = simplify_onnx(onnx_path)
    else:
        simplified_path = onnx_path

    # Validate
    is_valid = validate_onnx(simplified_path, model, dummy_input)

    if not is_valid:
        print("⚠️ Warning: ONNX validation failed. Check model compatibility.")

    # Benchmark
    metrics = benchmark_onnx(simplified_path, dummy_input)

    # Check targets
    targets = config['tcn']['targets']
    print("\n=== Performance Targets ===")
    print(f"Size: {metrics['model_size_mb']:.2f} MB (target: < {targets['size_mb']} MB) "
          f"{'✅' if metrics['model_size_mb'] < targets['size_mb'] else '❌'}")
    print(f"Latency: {metrics['mean_latency_ms']:.2f} ms (target: < {targets['latency_ms']} ms) "
          f"{'✅' if metrics['mean_latency_ms'] < targets['latency_ms'] else '❌'}")

    print(f"\n✅ TCN ONNX export completed: {simplified_path}")

    return simplified_path


def export_lstm_ae(config: Dict) -> str:
    """
    Export LSTM-Autoencoder model to ONNX

    Args:
        config: Configuration dictionary

    Returns:
        Path to exported ONNX file
    """
    print("=== Exporting LSTM-AE Model to ONNX ===\n")

    # Load model
    checkpoint = torch.load("models/lstm_ae_best.pth", map_location='cpu')

    model = LSTM_Autoencoder(
        input_dim=config['lstm_ae']['input_dim'],
        hidden_dim=config['lstm_ae']['hidden_dim'],
        num_layers=config['lstm_ae']['num_layers'],
        latent_dim=config['lstm_ae']['latent_dim'],
        dropout=config['lstm_ae']['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, config['dataset']['window_size'],
                              config['lstm_ae']['input_dim'])

    # Export to ONNX
    output_path = "models/lstm_ae_anomaly.onnx"

    onnx_path = export_to_onnx(model, dummy_input, output_path, config)

    # Simplify ONNX
    if config['onnx']['optimization']:
        simplified_path = simplify_onnx(onnx_path)
    else:
        simplified_path = onnx_path

    # Validate
    is_valid = validate_onnx(simplified_path, model, dummy_input)

    if not is_valid:
        print("⚠️ Warning: ONNX validation failed")

    # Benchmark
    metrics = benchmark_onnx(simplified_path, dummy_input)

    # Check targets
    targets = config['lstm_ae']['targets']
    print("\n=== Performance Targets ===")
    print(f"Size: {metrics['model_size_mb']:.2f} MB (target: < {targets['size_mb']} MB) "
          f"{'✅' if metrics['model_size_mb'] < targets['size_mb'] else '❌'}")
    print(f"Latency: {metrics['mean_latency_ms']:.2f} ms (target: < {targets['latency_ms']} ms) "
          f"{'✅' if metrics['mean_latency_ms'] < targets['latency_ms'] else '❌'}")

    print(f"\n✅ LSTM-AE ONNX export completed: {simplified_path}")

    return simplified_path


def generate_snpe_conversion_script(onnx_path: str, output_dlc: str) -> str:
    """
    Generate shell script for SNPE DLC conversion

    Note: SNPE SDK must be installed locally

    Args:
        onnx_path: Path to ONNX file
        output_dlc: Output DLC file path

    Returns:
        Path to conversion script
    """
    script_path = "models/convert_to_snpe.sh"

    script_content = f"""#!/bin/bash
# SNPE DLC Conversion Script
# Requires: Qualcomm SNPE SDK installed

set -e

ONNX_PATH="{onnx_path}"
DLC_PATH="{output_dlc}"
DLC_QUANTIZED="${{DLC_PATH%.dlc}}_int8.dlc"

echo "=== Converting ONNX to SNPE DLC ==="

# Step 1: Convert ONNX to DLC
echo "Step 1: ONNX → DLC"
snpe-onnx-to-dlc \\
    --input_network $ONNX_PATH \\
    --output_path $DLC_PATH

echo "✅ DLC created: $DLC_PATH"

# Step 2: Quantize DLC to INT8
echo "Step 2: Quantizing to INT8 for DSP/HTP"

# Create input list for calibration (TODO: provide calibration data)
echo "input_data.raw" > input_list.txt

snpe-dlc-quantize \\
    --input_dlc $DLC_PATH \\
    --input_list input_list.txt \\
    --output_dlc $DLC_QUANTIZED \\
    --use_enhanced_quantizer

echo "✅ Quantized DLC created: $DLC_QUANTIZED"

# Step 3: Get model info
echo "Step 3: DLC Model Info"
snpe-dlc-info -i $DLC_QUANTIZED

# Step 4: Benchmark on device (requires connected device)
echo "Step 4: Benchmarking on DSP (optional)"
# snpe-net-run --container $DLC_QUANTIZED --use_dsp

echo "✅ SNPE conversion completed!"
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"\n✅ SNPE conversion script generated: {script_path}")
    print(f"   Run locally with SNPE SDK: ./{script_path}")

    return script_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Export models to ONNX')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='tcn',
                        choices=['tcn', 'lstm_ae', 'all'],
                        help='Model to export')
    parser.add_argument('--generate-snpe-script', action='store_true',
                        help='Generate SNPE conversion script')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Export models
    exported_models = []

    if args.model == 'tcn' or args.model == 'all':
        onnx_path = export_tcn(config)
        exported_models.append(('tcn', onnx_path))

    if args.model == 'lstm_ae' or args.model == 'all':
        onnx_path = export_lstm_ae(config)
        exported_models.append(('lstm_ae', onnx_path))

    # Generate SNPE conversion scripts
    if args.generate_snpe_script:
        print("\n=== Generating SNPE Conversion Scripts ===")
        for model_name, onnx_path in exported_models:
            dlc_path = onnx_path.replace('.onnx', '.dlc')
            generate_snpe_conversion_script(onnx_path, dlc_path)

    print("\n✅ All export tasks completed!")
    print("\nNext steps:")
    print("1. Transfer ONNX models to local machine with SNPE SDK")
    print("2. Run SNPE conversion scripts")
    print("3. Deploy .dlc files to Android DTG app assets/")


if __name__ == "__main__":
    main()
