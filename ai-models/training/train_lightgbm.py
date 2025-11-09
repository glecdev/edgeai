"""
GLEC DTG Edge AI - LightGBM Training
Driving behavior classification and carbon emission estimation
"""

import os
import yaml
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.lightgbm

from typing import Dict, Tuple


def extract_features(df: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
    """
    Extract statistical features from time-series windows

    Args:
        df: Raw vehicle data
        window_size: Size of rolling window

    Returns:
        Feature dataframe with statistical aggregations
    """
    features_list = []

    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size]

        # Calculate statistical features
        features = {}

        # Speed features
        features['speed_mean'] = window['vehicle_speed'].mean()
        features['speed_std'] = window['vehicle_speed'].std()
        features['speed_max'] = window['vehicle_speed'].max()
        features['speed_min'] = window['vehicle_speed'].min()

        # RPM features
        features['rpm_mean'] = window['engine_rpm'].mean()
        features['rpm_std'] = window['engine_rpm'].std()

        # Throttle features
        features['throttle_mean'] = window['throttle_position'].mean()
        features['throttle_std'] = window['throttle_position'].std()
        features['throttle_max'] = window['throttle_position'].max()

        # Brake features
        features['brake_mean'] = window['brake_pressure'].mean()
        features['brake_std'] = window['brake_pressure'].std()
        features['brake_max'] = window['brake_pressure'].max()

        # Acceleration features
        features['accel_x_mean'] = window['acceleration_x'].mean()
        features['accel_x_std'] = window['acceleration_x'].std()
        features['accel_x_max'] = abs(window['acceleration_x']).max()
        features['accel_y_mean'] = window['acceleration_y'].mean()
        features['accel_y_std'] = window['acceleration_y'].std()

        # Fuel consumption (target)
        features['fuel_consumption'] = window['fuel_consumption'].mean() if 'fuel_consumption' in window else 0

        # Label (if available)
        if 'label' in window.columns:
            # Take most common label in window
            features['label'] = window['label'].mode()[0]

        features_list.append(features)

    return pd.DataFrame(features_list)


def encode_labels(labels: pd.Series) -> Tuple[np.ndarray, Dict]:
    """
    Encode string labels to integers

    Labels:
    - 0: normal
    - 1: eco_driving
    - 2: harsh_braking
    - 3: harsh_acceleration
    - 4: anomaly
    """
    label_mapping = {
        'normal': 0,
        'eco_driving': 1,
        'harsh_braking': 2,
        'harsh_acceleration': 3,
        'anomaly': 4
    }

    encoded = labels.map(label_mapping)
    return encoded.values, label_mapping


def train_lightgbm(config: Dict) -> None:
    """
    Main training function

    Args:
        config: Configuration dictionary from config.yaml
    """
    print("Starting LightGBM training...")

    # MLflow setup
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name="lightgbm_behavior_classification"):
        # Log configuration
        mlflow.log_params(config['lightgbm']['params'])

        # Load and process data
        print("Loading and processing datasets...")
        train_df = pd.read_csv(config['dataset']['train_path'])
        val_df = pd.read_csv(config['dataset']['val_path'])

        # Extract features
        print("Extracting features...")
        train_features = extract_features(train_df, config['dataset']['window_size'])
        val_features = extract_features(val_df, config['dataset']['window_size'])

        # Separate features and labels
        if 'label' in train_features.columns:
            X_train = train_features.drop('label', axis=1)
            y_train, label_mapping = encode_labels(train_features['label'])
            X_val = val_features.drop('label', axis=1)
            y_val, _ = encode_labels(val_features['label'])

            mlflow.log_params(label_mapping)
        else:
            # If no labels, create synthetic labels (for skeleton code)
            print("Warning: No labels found. Creating synthetic labels for demonstration.")
            X_train = train_features
            y_train = np.random.randint(0, config['lightgbm']['num_classes'], len(X_train))
            X_val = val_features
            y_val = np.random.randint(0, config['lightgbm']['num_classes'], len(X_val))

        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Training parameters
        params = config['lightgbm']['params'].copy()
        params['num_class'] = config['lightgbm']['num_classes']

        # Train model
        print("Training model...")
        start_time = time.time()

        callbacks = [
            lgb.log_evaluation(period=10),
            lgb.early_stopping(config['lightgbm']['training']['early_stopping_rounds'])
        ]

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f}s")

        # Evaluate model
        print("Evaluating model...")
        y_train_pred = model.predict(X_train).argmax(axis=1)
        y_val_pred = model.predict(X_val).argmax(axis=1)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print(f"Val F1 Score: {val_f1:.4f}")

        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1_score", val_f1)
        mlflow.log_metric("training_time_seconds", training_time)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred))

        # Feature importance
        print("\nTop 10 Feature Importance:")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(10))

        # Save feature importance
        importance_path = "models/lightgbm_feature_importance.csv"
        os.makedirs("models", exist_ok=True)
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Save model
        model_path = "models/lightgbm_model.txt"
        model.save_model(model_path)
        mlflow.log_artifact(model_path)

        # Log model to MLflow
        mlflow.lightgbm.log_model(model, "lightgbm_model")

        # Model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\nModel size: {model_size_mb:.2f} MB")
        mlflow.log_metric("model_size_mb", model_size_mb)

        # Check if targets are met
        targets = config['lightgbm']['targets']
        print("\n=== Performance Targets ===")
        print(f"Size: {model_size_mb:.2f} MB (target: < {targets['size_mb']} MB) "
              f"{'✅' if model_size_mb < targets['size_mb'] else '❌'}")
        print(f"Accuracy: {val_accuracy:.4f} (target: > {targets['accuracy']}) "
              f"{'✅' if val_accuracy > targets['accuracy'] else '❌'}")

        print("\nTraining completed successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train LightGBM model for behavior classification')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train model
    train_lightgbm(config)


if __name__ == "__main__":
    main()
