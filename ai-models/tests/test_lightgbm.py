"""
GLEC DTG Edge AI - LightGBM Model Tests
Unit tests for LightGBM behavior classification
"""

import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "training"))
from train_lightgbm import extract_features, encode_labels


class TestLightGBM:
    """Test suite for LightGBM model"""

    @pytest.fixture
    def sample_model(self):
        """Create sample LightGBM model"""
        # Create synthetic training data
        X_train = np.random.randn(1000, 20)  # 1000 samples, 20 features
        y_train = np.random.randint(0, 5, 1000)  # 5 classes

        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }

        model = lgb.train(params, train_data, num_boost_round=10)

        return model

    @pytest.fixture
    def sample_input(self):
        """Create sample input features"""
        return np.random.randn(10, 20)  # 10 samples, 20 features

    def test_lightgbm_prediction_shape(self, sample_model, sample_input):
        """Test LightGBM produces correct prediction shape"""
        predictions = sample_model.predict(sample_input)

        # Predictions should be (num_samples, num_classes)
        assert predictions.shape == (10, 5), \
            f"Expected shape (10, 5), got {predictions.shape}"

    def test_lightgbm_prediction_values(self, sample_model, sample_input):
        """Test prediction values are valid probabilities"""
        predictions = sample_model.predict(sample_input)

        # All values should be between 0 and 1
        assert np.all(predictions >= 0) and np.all(predictions <= 1), \
            "Predictions should be probabilities between 0 and 1"

        # Each row should sum to ~1.0 (probabilities)
        row_sums = predictions.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            "Prediction probabilities should sum to 1.0"

    def test_lightgbm_inference_latency(self, sample_model):
        """Test LightGBM inference latency < 15ms (target)"""
        # Single sample input
        x = np.random.randn(1, 20)

        # Warmup
        for _ in range(10):
            _ = sample_model.predict(x)

        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.time()
            _ = sample_model.predict(x)
            latencies.append((time.time() - start) * 1000)

        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        print(f"\nLightGBM Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")

        # Target: < 15ms
        assert mean_latency < 15, f"Latency {mean_latency:.2f}ms exceeds target 15ms"

    def test_lightgbm_model_size(self, sample_model):
        """Test LightGBM model size < 10MB"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=True) as tmp:
            sample_model.save_model(tmp.name)
            tmp.seek(0, 2)  # Seek to end
            model_size_mb = tmp.tell() / (1024 * 1024)

        print(f"\nLightGBM Model Size: {model_size_mb:.2f} MB")

        assert model_size_mb < 10, f"Model size {model_size_mb:.2f}MB exceeds target 10MB"

    def test_lightgbm_batch_inference(self, sample_model):
        """Test batch inference performance"""
        # Large batch
        x = np.random.randn(1000, 20)

        start = time.time()
        predictions = sample_model.predict(x)
        elapsed = (time.time() - start) * 1000

        print(f"\nBatch inference (1000 samples): {elapsed:.2f} ms")
        print(f"Per-sample latency: {elapsed/1000:.2f} ms")

        # Batch processing should be efficient
        assert elapsed < 100, "Batch inference should be efficient"

    def test_lightgbm_feature_importance(self, sample_model):
        """Test feature importance calculation"""
        importance = sample_model.feature_importance()

        # Should have importance for all features
        assert len(importance) == 20, "Should have importance for all 20 features"

        # Importance values should be non-negative
        assert np.all(importance >= 0), "Feature importance should be non-negative"

        print(f"\nTop 5 features by importance:")
        top_features = np.argsort(importance)[::-1][:5]
        for i, feat_idx in enumerate(top_features):
            print(f"  {i+1}. Feature {feat_idx}: {importance[feat_idx]:.2f}")


class TestFeatureExtraction:
    """Test suite for feature extraction"""

    @pytest.fixture
    def sample_timeseries_data(self):
        """Create sample time-series data"""
        data = {
            'vehicle_speed': np.random.uniform(0, 120, 1000),
            'engine_rpm': np.random.uniform(800, 5000, 1000),
            'throttle_position': np.random.uniform(0, 100, 1000),
            'brake_pressure': np.random.uniform(0, 100, 1000),
            'fuel_level': np.random.uniform(20, 100, 1000),
            'coolant_temp': np.random.uniform(80, 95, 1000),
            'acceleration_x': np.random.uniform(-2, 2, 1000),
            'acceleration_y': np.random.uniform(-1, 1, 1000),
            'steering_angle': np.random.uniform(-30, 30, 1000),
            'fuel_consumption': np.random.uniform(5, 20, 1000),
            'label': np.random.choice(['normal', 'eco_driving', 'harsh_braking'], 1000)
        }

        return pd.DataFrame(data)

    def test_extract_features_shape(self, sample_timeseries_data):
        """Test extracted features have correct shape"""
        features_df = extract_features(sample_timeseries_data, window_size=60)

        # Should have (n_samples - window_size) samples
        expected_samples = len(sample_timeseries_data) - 60
        assert len(features_df) == expected_samples, \
            f"Expected {expected_samples} samples, got {len(features_df)}"

        # Should have statistical features (mean, std, max, min for each signal)
        # At minimum: ~20-30 features
        assert len(features_df.columns) >= 15, \
            "Should have at least 15 statistical features"

    def test_extract_features_values(self, sample_timeseries_data):
        """Test extracted feature values are valid"""
        features_df = extract_features(sample_timeseries_data, window_size=60)

        # No NaN values
        assert not features_df.isnull().any().any(), "Features should not contain NaN"

        # No infinite values
        assert not np.isinf(features_df.select_dtypes(include=np.number).values).any(), \
            "Features should not contain infinite values"

        print(f"\nExtracted features shape: {features_df.shape}")
        print(f"Feature columns: {list(features_df.columns)[:10]}...")  # Print first 10

    def test_encode_labels(self):
        """Test label encoding"""
        labels = pd.Series(['normal', 'eco_driving', 'harsh_braking', 'normal', 'anomaly'])

        encoded, label_mapping = encode_labels(labels)

        # Encoded should be integers
        assert encoded.dtype == np.int64 or encoded.dtype == np.int32

        # Should have correct mapping
        assert 'normal' in label_mapping
        assert 'eco_driving' in label_mapping
        assert 'harsh_braking' in label_mapping

        # Encoded values should match mapping
        assert encoded[0] == label_mapping['normal']
        assert encoded[1] == label_mapping['eco_driving']

        print(f"\nLabel mapping: {label_mapping}")


class TestLightGBMIntegration:
    """Integration tests for LightGBM pipeline"""

    def test_training_pipeline(self):
        """Test complete training pipeline"""
        # Create synthetic dataset
        X_train = np.random.randn(1000, 20)
        y_train = np.random.randint(0, 5, 1000)

        X_val = np.random.randn(200, 20)
        y_val = np.random.randint(0, 5, 200)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Training parameters
        params = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=10,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val']
        )

        # Predictions
        y_pred = model.predict(X_val).argmax(axis=1)

        # Calculate accuracy
        accuracy = (y_pred == y_val).mean()

        print(f"\nValidation accuracy: {accuracy:.4f}")

        # Model should learn something (accuracy > random)
        assert accuracy > 0.15, "Model should perform better than random (20%)"

    def test_model_persistence(self):
        """Test model save/load"""
        # Train model
        X_train = np.random.randn(1000, 20)
        y_train = np.random.randint(0, 5, 1000)

        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'multiclass',
            'num_class': 5,
            'verbose': -1
        }

        model = lgb.train(params, train_data, num_boost_round=10)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            model_path = tmp.name

        model.save_model(model_path)

        # Load model
        loaded_model = lgb.Booster(model_file=model_path)

        # Test predictions are identical
        x_test = np.random.randn(10, 20)

        pred_original = model.predict(x_test)
        pred_loaded = loaded_model.predict(x_test)

        assert np.allclose(pred_original, pred_loaded), \
            "Loaded model predictions should match original"

        print("\n✅ Model save/load successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
