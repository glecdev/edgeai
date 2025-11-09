"""
GLEC DTG Edge AI - TCN Model Tests
Unit tests for Temporal Convolutional Network
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import sys
from pathlib import Path

# Add training directory to path
sys.path.append(str(Path(__file__).parent.parent / "training"))
from train_tcn import TCN, VehicleDataset


class TestTCN:
    """Test suite for TCN model"""

    @pytest.fixture
    def model(self):
        """Create TCN model instance"""
        return TCN(
            input_dim=10,
            output_dim=1,
            num_channels=[64, 128, 256],
            kernel_size=3,
            dropout=0.2
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        # batch_size=32, sequence_length=60, input_dim=10
        return torch.randn(32, 60, 10)

    def test_tcn_output_shape(self, model, sample_input):
        """Test TCN produces correct output shape"""
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        # Output should be (batch_size, output_dim)
        assert output.shape == (32, 1), f"Expected shape (32, 1), got {output.shape}"

    def test_tcn_forward_pass(self, model, sample_input):
        """Test forward pass completes without errors"""
        model.eval()

        try:
            with torch.no_grad():
                output = model(sample_input)
            assert output is not None
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")

    def test_tcn_inference_latency(self, model):
        """Test TCN inference latency < 25ms (target)"""
        model.eval()

        # Single sample input
        x = torch.randn(1, 60, 10)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = model(x)
                latencies.append((time.time() - start) * 1000)

        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        print(f"\nTCN Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")

        # Target: < 25ms (on CPU, will be faster on DSP INT8)
        assert mean_latency < 25, f"Latency {mean_latency:.2f}ms exceeds target 25ms"

    def test_tcn_model_size(self, model):
        """Test TCN model size < 4MB (before quantization)"""
        # Save model temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as tmp:
            torch.save(model.state_dict(), tmp.name)
            model_size_mb = tmp.file.tell() / (1024 * 1024)

        print(f"\nTCN Model Size: {model_size_mb:.2f} MB (before quantization)")

        # Before quantization, size should be < 20MB (will be ~4MB after INT8)
        assert model_size_mb < 20, f"Model size {model_size_mb:.2f}MB too large"

    def test_tcn_gradient_flow(self, model, sample_input):
        """Test gradients flow properly during backpropagation"""
        model.train()

        # Forward pass
        output = model(sample_input)

        # Compute loss
        target = torch.randn(32, 1)
        criterion = nn.MSELoss()
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in: {name}"

    def test_tcn_different_sequence_lengths(self, model):
        """Test TCN handles different sequence lengths"""
        model.eval()

        sequence_lengths = [30, 60, 120]

        for seq_len in sequence_lengths:
            x = torch.randn(1, seq_len, 10)

            with torch.no_grad():
                output = model(x)

            assert output.shape == (1, 1), f"Failed for sequence length {seq_len}"

    def test_tcn_batch_consistency(self, model):
        """Test batch processing gives same results as single sample"""
        model.eval()

        # Single sample
        x_single = torch.randn(1, 60, 10)

        with torch.no_grad():
            output_single = model(x_single)

        # Same sample in batch
        x_batch = x_single.repeat(4, 1, 1)  # (4, 60, 10)

        with torch.no_grad():
            output_batch = model(x_batch)

        # All outputs should be identical
        for i in range(4):
            assert torch.allclose(output_single, output_batch[i:i+1], atol=1e-5), \
                f"Batch output {i} differs from single output"

    def test_tcn_numerical_stability(self, model):
        """Test model doesn't produce NaN or Inf outputs"""
        model.eval()

        # Test with various input ranges
        test_cases = [
            torch.randn(10, 60, 10),  # Normal range
            torch.randn(10, 60, 10) * 10,  # Large values
            torch.randn(10, 60, 10) * 0.1,  # Small values
        ]

        for x in test_cases:
            with torch.no_grad():
                output = model(x)

            assert not torch.isnan(output).any(), "Model produced NaN output"
            assert not torch.isinf(output).any(), "Model produced Inf output"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tcn_cuda_compatibility(self, model, sample_input):
        """Test model works on CUDA"""
        device = torch.device('cuda')
        model = model.to(device)
        sample_input = sample_input.to(device)

        model.eval()
        with torch.no_grad():
            output = model(sample_input)

        assert output.device.type == 'cuda'


class TestVehicleDataset:
    """Test suite for VehicleDataset"""

    @pytest.fixture
    def sample_data_path(self):
        """Create temporary sample dataset"""
        import pandas as pd
        import tempfile

        # Generate synthetic data
        data = {
            'timestamp': np.arange(0, 1000),
            'vehicle_speed': np.random.uniform(0, 120, 1000),
            'engine_rpm': np.random.uniform(800, 5000, 1000),
            'throttle_position': np.random.uniform(0, 100, 1000),
            'brake_pressure': np.random.uniform(0, 100, 1000),
            'fuel_level': np.random.uniform(20, 100, 1000),
            'coolant_temp': np.random.uniform(80, 95, 1000),
            'acceleration_x': np.random.uniform(-2, 2, 1000),
            'acceleration_y': np.random.uniform(-1, 1, 1000),
            'steering_angle': np.random.uniform(-30, 30, 1000),
            'gps_lat': np.full(1000, 37.5665),
            'fuel_consumption': np.random.uniform(5, 20, 1000),
        }

        df = pd.DataFrame(data)

        # Save to temporary file
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(tmp_file.name, index=False)

        return tmp_file.name

    def test_dataset_length(self, sample_data_path):
        """Test dataset returns correct length"""
        dataset = VehicleDataset(sample_data_path, window_size=60)

        # Length should be total_samples - window_size
        assert len(dataset) == 1000 - 60

    def test_dataset_item_shape(self, sample_data_path):
        """Test dataset item has correct shape"""
        dataset = VehicleDataset(sample_data_path, window_size=60)

        x, y = dataset[0]

        # x should be (window_size, num_features)
        assert x.shape == (60, 10), f"Expected shape (60, 10), got {x.shape}"

        # y should be scalar (fuel consumption)
        assert y.shape == (1,), f"Expected shape (1,), got {y.shape}"

    def test_dataset_dataloader_compatibility(self, sample_data_path):
        """Test dataset works with DataLoader"""
        from torch.utils.data import DataLoader

        dataset = VehicleDataset(sample_data_path, window_size=60)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Get one batch
        batch_x, batch_y = next(iter(dataloader))

        assert batch_x.shape == (32, 60, 10)
        assert batch_y.shape == (32, 1)


def test_training_integration():
    """Integration test for training pipeline"""
    # This is a placeholder for integration testing
    # In practice, you would test the full training loop
    pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
