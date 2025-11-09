"""
GLEC DTG Edge AI - LSTM-AE Model Tests
Unit tests for LSTM-Autoencoder anomaly detection
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "training"))
from train_lstm_ae import LSTM_Autoencoder, AnomalyDataset


class TestLSTM_AE:
    """Test suite for LSTM-Autoencoder model"""

    @pytest.fixture
    def model(self):
        """Create LSTM-AE model instance"""
        return LSTM_Autoencoder(
            input_dim=10,
            hidden_dim=128,
            num_layers=2,
            latent_dim=32,
            dropout=0.2
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        # batch_size=16, sequence_length=60, input_dim=10
        return torch.randn(16, 60, 10)

    def test_lstm_ae_output_shape(self, model, sample_input):
        """Test LSTM-AE produces correct output shape"""
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        # Output should have same shape as input (reconstruction)
        assert output.shape == sample_input.shape, \
            f"Expected shape {sample_input.shape}, got {output.shape}"

    def test_lstm_ae_encoder(self, model, sample_input):
        """Test encoder produces correct latent representation"""
        model.eval()

        with torch.no_grad():
            latent = model.encode(sample_input)

        # Latent should be (batch_size, latent_dim)
        assert latent.shape == (16, 32), f"Expected shape (16, 32), got {latent.shape}"

    def test_lstm_ae_decoder(self, model):
        """Test decoder produces correct output shape"""
        model.eval()

        # Create latent representation
        latent = torch.randn(16, 32)

        with torch.no_grad():
            output = model.decode(latent, sequence_length=60)

        # Output should be (batch_size, sequence_length, input_dim)
        assert output.shape == (16, 60, 10), f"Expected shape (16, 60, 10), got {output.shape}"

    def test_lstm_ae_reconstruction_error(self, model, sample_input):
        """Test reconstruction error calculation"""
        model.eval()

        with torch.no_grad():
            errors = model.get_reconstruction_error(sample_input)

        # Errors should be (batch_size,)
        assert errors.shape == (16,), f"Expected shape (16,), got {errors.shape}"

        # Errors should be non-negative
        assert torch.all(errors >= 0), "Reconstruction errors must be non-negative"

    def test_lstm_ae_inference_latency(self, model):
        """Test LSTM-AE inference latency < 35ms (target)"""
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

        print(f"\nLSTM-AE Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")

        # Target: < 35ms
        assert mean_latency < 35, f"Latency {mean_latency:.2f}ms exceeds target 35ms"

    def test_lstm_ae_model_size(self, model):
        """Test LSTM-AE model size < 3MB (before quantization)"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as tmp:
            torch.save(model.state_dict(), tmp.name)
            model_size_mb = tmp.file.tell() / (1024 * 1024)

        print(f"\nLSTM-AE Model Size: {model_size_mb:.2f} MB (before quantization)")

        # Before quantization, size should be < 15MB (will be ~3MB after INT8)
        assert model_size_mb < 15, f"Model size {model_size_mb:.2f}MB too large"

    def test_lstm_ae_anomaly_detection(self, model):
        """Test anomaly detection capability"""
        model.eval()

        # Normal data (low variance)
        normal_data = torch.randn(10, 60, 10) * 0.5

        # Anomalous data (high variance)
        anomalous_data = torch.randn(10, 60, 10) * 5.0

        with torch.no_grad():
            normal_errors = model.get_reconstruction_error(normal_data)
            anomalous_errors = model.get_reconstruction_error(anomalous_data)

        # Anomalous data should have higher reconstruction error
        assert anomalous_errors.mean() > normal_errors.mean(), \
            "Anomalous data should have higher reconstruction error"

        print(f"\nNormal error: {normal_errors.mean():.4f}")
        print(f"Anomalous error: {anomalous_errors.mean():.4f}")

    def test_lstm_ae_gradient_flow(self, model, sample_input):
        """Test gradients flow properly during backpropagation"""
        model.train()

        # Forward pass
        output = model(sample_input)

        # Compute loss (reconstruction loss)
        criterion = nn.MSELoss()
        loss = criterion(output, sample_input)

        # Backward pass
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in: {name}"

    def test_lstm_ae_numerical_stability(self, model):
        """Test model doesn't produce NaN or Inf outputs"""
        model.eval()

        # Test with various input ranges
        test_cases = [
            torch.randn(5, 60, 10),  # Normal range
            torch.randn(5, 60, 10) * 10,  # Large values
            torch.randn(5, 60, 10) * 0.1,  # Small values
        ]

        for x in test_cases:
            with torch.no_grad():
                output = model(x)

            assert not torch.isnan(output).any(), "Model produced NaN output"
            assert not torch.isinf(output).any(), "Model produced Inf output"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_lstm_ae_cuda_compatibility(self, model, sample_input):
        """Test model works on CUDA"""
        device = torch.device('cuda')
        model = model.to(device)
        sample_input = sample_input.to(device)

        model.eval()
        with torch.no_grad():
            output = model(sample_input)

        assert output.device.type == 'cuda'


class TestAnomalyDataset:
    """Test suite for AnomalyDataset"""

    @pytest.fixture
    def sample_data_path(self):
        """Create temporary sample dataset"""
        import pandas as pd
        import tempfile

        # Generate synthetic data with labels
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
            'label': np.random.choice(['normal', 'anomaly'], 1000)
        }

        df = pd.DataFrame(data)

        # Save to temporary file
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(tmp_file.name, index=False)

        return tmp_file.name

    def test_dataset_length(self, sample_data_path):
        """Test dataset returns correct length"""
        dataset = AnomalyDataset(sample_data_path, window_size=60)

        # Length should be total_samples - window_size
        assert len(dataset) == 1000 - 60

    def test_dataset_item_shape(self, sample_data_path):
        """Test dataset item has correct shape"""
        dataset = AnomalyDataset(sample_data_path, window_size=60)

        x, y = dataset[0]

        # x should be (window_size, num_features)
        assert x.shape == (60, 10), f"Expected shape (60, 10), got {x.shape}"

        # y should be scalar (binary label)
        assert y.shape == (1,), f"Expected shape (1,), got {y.shape}"

    def test_dataset_dataloader_compatibility(self, sample_data_path):
        """Test dataset works with DataLoader"""
        from torch.utils.data import DataLoader

        dataset = AnomalyDataset(sample_data_path, window_size=60)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Get one batch
        batch_x, batch_y = next(iter(dataloader))

        assert batch_x.shape == (16, 60, 10)
        assert batch_y.shape == (16, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
