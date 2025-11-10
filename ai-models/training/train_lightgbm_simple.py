"""
GLEC DTG Edge AI - LightGBM Training (Simplified, No MLflow)
Driving behavior classification - Web environment compatible

This is a simplified version for CPU-only training without MLflow dependency.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def extract_features(df: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
    """
    Extract statistical features from time-series windows

    Args:
        df: Raw vehicle data with columns: vehicle_speed, engine_rpm, etc.
        window_size: Size of rolling window (default 60 seconds)

    Returns:
        DataFrame with extracted statistical features
    """
    print(f"Extracting features with window size={window_size}...")
    features_list = []

    for i in range(len(df) - window_size):
        if i % 5000 == 0:
            print(f"  Processing window {i:,}/{len(df) - window_size:,}")

        window = df.iloc[i:i+window_size]

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

        # Fuel consumption
        features['fuel_consumption'] = window['fuel_consumption'].mean()

        # Label (most common in window)
        if 'label' in window.columns:
            features['label'] = window['label'].mode()[0]

        features_list.append(features)

    return pd.DataFrame(features_list)


def encode_labels(labels: pd.Series):
    """
    Encode string labels to integers

    Returns:
        encoded: numpy array of integer labels
        label_mapping: dict mapping labels to integers
    """
    label_mapping = {
        'normal': 0,
        'eco_driving': 1,
        'aggressive': 2,
    }

    encoded = labels.map(label_mapping)

    # Handle any unmapped labels
    if encoded.isnull().any():
        print(f"Warning: Found unmapped labels: {labels[encoded.isnull()].unique()}")
        # Fill with 'normal' as default
        encoded = encoded.fillna(0)

    return encoded.values.astype(int), label_mapping


def train_model(train_path: str, val_path: str, window_size: int = 60,
                num_boost_round: int = 100, early_stopping_rounds: int = 10):
    """
    Train LightGBM model

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        window_size: Feature extraction window size
        num_boost_round: Number of boosting iterations
        early_stopping_rounds: Early stopping patience
    """
    print("=" * 80)
    print("GLEC DTG Edge AI - LightGBM Training")
    print("=" * 80)

    # Load data
    print(f"\n📥 Loading datasets...")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"\n📊 Dataset info:")
    print(f"  Train: {train_df.shape[0]:,} samples")
    print(f"  Val:   {val_df.shape[0]:,} samples")
    print(f"\n  Label distribution (train):")
    print(train_df['label'].value_counts())

    # Extract features
    print(f"\n🔧 Extracting features...")
    start_time = time.time()

    train_features = extract_features(train_df, window_size)
    val_features = extract_features(val_df, window_size)

    elapsed = time.time() - start_time
    print(f"✅ Feature extraction complete ({elapsed:.1f}s)")
    print(f"  Train features: {train_features.shape}")
    print(f"  Val features:   {val_features.shape}")

    # Separate features and labels
    X_train = train_features.drop('label', axis=1)
    y_train, label_mapping = encode_labels(train_features['label'])

    X_val = val_features.drop('label', axis=1)
    y_val, _ = encode_labels(val_features['label'])

    print(f"\n🏷️  Label mapping: {label_mapping}")
    print(f"  Class distribution:")
    for label, idx in label_mapping.items():
        count = (y_train == idx).sum()
        pct = count / len(y_train) * 100
        print(f"    {label:15s}: {count:6,} ({pct:5.1f}%)")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Training parameters
    params = {
        'objective': 'multiclass',
        'num_class': len(label_mapping),
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1
    }

    print(f"\n🚀 Training LightGBM model...")
    print(f"  Parameters: {params}")

    # Train
    start_time = time.time()

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=10)
        ]
    )

    train_time = time.time() - start_time
    print(f"\n✅ Training complete ({train_time:.1f}s)")

    # Evaluate
    print(f"\n📈 Evaluation on validation set:")

    y_pred_proba = model.predict(X_val)
    y_pred = y_pred_proba.argmax(axis=1)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    print(f"\n  Classification Report:")
    # Get unique classes in predictions and ground truth
    unique_classes = np.unique(np.concatenate([y_val, y_pred]))
    target_names_subset = [list(label_mapping.keys())[i] for i in unique_classes]

    print(classification_report(y_val, y_pred,
                                labels=unique_classes,
                                target_names=target_names_subset,
                                digits=4))

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)

    # Feature importance
    print(f"\n🔍 Top 10 Feature Importances:")
    feature_names = X_train.columns
    importances = model.feature_importance(importance_type='gain')

    # Sort by importance
    indices = np.argsort(importances)[::-1][:10]

    for i, idx in enumerate(indices):
        print(f"  {i+1:2d}. {feature_names[idx]:20s}: {importances[idx]:10.1f}")

    # Save model
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "lightgbm_behavior.txt"
    model.save_model(str(model_path))

    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\n💾 Model saved: {model_path}")
    print(f"  Size: {model_size_mb:.2f} MB")

    # Performance summary
    print(f"\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"✅ Accuracy:       {accuracy:.4f} (target: >0.90)")
    print(f"✅ F1-Score:       {f1:.4f}")
    print(f"✅ Model Size:     {model_size_mb:.2f} MB (target: <10 MB)")
    print(f"✅ Training Time:  {train_time:.1f}s")
    print(f"✅ Best Iteration: {model.best_iteration}")
    print("=" * 80)

    # Quality gate check
    if accuracy >= 0.90:
        print("🎉 PASS: Accuracy target met (>90%)")
    else:
        print(f"⚠️  WARNING: Accuracy {accuracy:.4f} < 0.90 target")

    if model_size_mb < 10:
        print("🎉 PASS: Model size target met (<10 MB)")
    else:
        print(f"⚠️  WARNING: Model size {model_size_mb:.2f} MB > 10 MB target")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM for driving behavior classification"
    )
    parser.add_argument(
        '--train',
        default='../../datasets/train.csv',
        help='Path to training CSV'
    )
    parser.add_argument(
        '--val',
        default='../../datasets/val.csv',
        help='Path to validation CSV'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=60,
        help='Feature extraction window size (default: 60)'
    )
    parser.add_argument(
        '--num-boost-round',
        type=int,
        default=100,
        help='Number of boosting iterations (default: 100)'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Early stopping rounds (default: 10)'
    )

    args = parser.parse_args()

    model = train_model(
        train_path=args.train,
        val_path=args.val,
        window_size=args.window_size,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping
    )

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
