"""
GLEC DTG Edge AI - Model Quantization
Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_qat, prepare_qat, convert
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "training"))
from train_tcn import TCN, VehicleDataset
from train_lstm_ae import LSTM_Autoencoder, AnomalyDataset

from typing import Dict


def post_training_quantization(model: nn.Module, method: str = 'dynamic') -> nn.Module:
    """
    Post-Training Quantization (PTQ)

    Methods:
    - dynamic: Dynamic quantization (weights only)
    - static: Static quantization (weights + activations)

    Args:
        model: PyTorch model to quantize
        method: Quantization method

    Returns:
        Quantized model
    """
    print(f"Applying {method} quantization...")

    if method == 'dynamic':
        # Dynamic quantization (best for LSTMs/RNNs)
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.Conv1d},
            dtype=torch.qint8
        )
    elif method == 'static':
        # Static quantization (requires calibration)
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        quantized_model = torch.quantization.prepare(model)
        # TODO: Run calibration data through model
        quantized_model = torch.quantization.convert(quantized_model)
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    return quantized_model


def quantization_aware_training(model: nn.Module, train_loader: DataLoader,
                                 val_loader: DataLoader, config: Dict) -> nn.Module:
    """
    Quantization-Aware Training (QAT)

    Train model with fake quantization nodes to simulate INT8 inference

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary

    Returns:
        QAT-trained model ready for quantization
    """
    print("Starting Quantization-Aware Training...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Configure QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_qat = prepare_qat(model.train())

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model_qat.parameters(),
        lr=config['quantization']['qat']['learning_rate']
    )

    num_epochs = config['quantization']['qat']['num_epochs']

    # QAT training loop
    for epoch in range(num_epochs):
        model_qat.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model_qat(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"QAT Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

        # Freeze batch norm after certain epochs
        if epoch > 3 and config['quantization']['qat']['freeze_bn']:
            model_qat.apply(torch.quantization.disable_observer)

    # Convert to quantized model
    model_qat.eval()
    quantized_model = convert(model_qat)

    return quantized_model


def calibrate_model(model: nn.Module, calibration_loader: DataLoader,
                    device: str = 'cpu') -> None:
    """
    Calibrate model for static quantization

    Args:
        model: Prepared model with observers
        calibration_loader: Calibration data loader
        device: Device to run on
    """
    print("Calibrating model...")
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            data = data.to(device)
            _ = model(data)

            if batch_idx % 10 == 0:
                print(f"Calibrated {batch_idx}/{len(calibration_loader)} batches")

    print("Calibration completed")


def compare_models(original_model: nn.Module, quantized_model: nn.Module,
                   test_loader: DataLoader, device: str = 'cpu') -> Dict:
    """
    Compare original and quantized model performance

    Returns:
        Dictionary with comparison metrics
    """
    print("\n=== Model Comparison ===")

    # Model sizes
    def get_model_size(model):
        torch.save(model.state_dict(), "/tmp/temp_model.pth")
        size_mb = os.path.getsize("/tmp/temp_model.pth") / (1024 * 1024)
        os.remove("/tmp/temp_model.pth")
        return size_mb

    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)

    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")

    # Inference latency
    import time

    def measure_latency(model, data, num_runs=100):
        model.eval()
        latencies = []

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(data)

            # Measure
            for _ in range(num_runs):
                start = time.time()
                _ = model(data)
                latencies.append((time.time() - start) * 1000)  # ms

        return np.mean(latencies), np.std(latencies)

    # Get sample batch
    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data.to(device)

    original_latency, original_std = measure_latency(original_model, sample_data)
    quantized_latency, quantized_std = measure_latency(quantized_model, sample_data)

    print(f"\nOriginal latency: {original_latency:.2f} ± {original_std:.2f} ms")
    print(f"Quantized latency: {quantized_latency:.2f} ± {quantized_std:.2f} ms")
    print(f"Speedup: {original_latency/quantized_latency:.2f}x")

    # Accuracy comparison
    criterion = nn.MSELoss()

    def evaluate_model(model, loader):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(loader)

    original_loss = evaluate_model(original_model, test_loader)
    quantized_loss = evaluate_model(quantized_model, test_loader)

    print(f"\nOriginal loss: {original_loss:.4f}")
    print(f"Quantized loss: {quantized_loss:.4f}")
    print(f"Accuracy degradation: {((quantized_loss - original_loss) / original_loss * 100):.2f}%")

    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': original_size / quantized_size,
        'original_latency_ms': original_latency,
        'quantized_latency_ms': quantized_latency,
        'speedup': original_latency / quantized_latency,
        'original_loss': original_loss,
        'quantized_loss': quantized_loss,
    }


def quantize_tcn(config: Dict) -> None:
    """Quantize TCN model"""
    print("=== Quantizing TCN Model ===")

    # Load original model
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

    # Quantize
    method = config['quantization']['method']
    if method == 'ptq':
        quantized_model = post_training_quantization(model, method='dynamic')
    elif method == 'qat':
        # Load datasets for QAT
        train_dataset = VehicleDataset(
            config['dataset']['train_path'],
            window_size=config['dataset']['window_size']
        )
        val_dataset = VehicleDataset(
            config['dataset']['val_path'],
            window_size=config['dataset']['window_size']
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        quantized_model = quantization_aware_training(
            model, train_loader, val_loader, config
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # Save quantized model
    output_path = f"models/tcn_fuel_{config['quantization']['dtype']}.pth"
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Quantized model saved to: {output_path}")

    # Compare models
    test_dataset = VehicleDataset(
        config['dataset']['test_path'],
        window_size=config['dataset']['window_size']
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    metrics = compare_models(model, quantized_model, test_loader)

    print("\n✅ TCN quantization completed!")
    return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Quantize AI models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='tcn',
                        choices=['tcn', 'lstm_ae', 'all'],
                        help='Model to quantize')
    parser.add_argument('--method', type=str, default=None,
                        choices=['ptq', 'qat'],
                        help='Quantization method (overrides config)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override method if specified
    if args.method:
        config['quantization']['method'] = args.method

    # Quantize models
    if args.model == 'tcn' or args.model == 'all':
        quantize_tcn(config)

    if args.model == 'lstm_ae' or args.model == 'all':
        print("\n=== LSTM-AE quantization ===")
        print("TODO: Implement LSTM-AE quantization")

    print("\n✅ All quantization tasks completed!")


if __name__ == "__main__":
    main()
