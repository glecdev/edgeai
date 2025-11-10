"""
LightGBM Model Test Set Evaluation
Verify model performance on unseen test data to check for overfitting
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add training module to path
sys.path.insert(0, str(Path(__file__).parent))
from train_lightgbm_simple import extract_features, encode_labels


def evaluate_model(model_path: str, test_data_path: str, window_size: int = 60):
    """
    Evaluate trained LightGBM model on test set

    Args:
        model_path: Path to trained model (.txt)
        test_data_path: Path to test CSV
        window_size: Feature extraction window size

    Returns:
        dict: Evaluation metrics
    """
    print("=" * 80)
    print("LIGHTGBM TEST SET EVALUATION")
    print("=" * 80)

    # Load model
    print(f"\n📥 Loading model: {model_path}")
    model = lgb.Booster(model_file=model_path)
    print("✅ Model loaded successfully")

    # Load test data
    print(f"\n📥 Loading test data: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    print(f"  Test samples: {test_df.shape[0]:,}")
    print(f"\n  Label distribution:")
    print(test_df['label'].value_counts())

    # Extract features
    print(f"\n🔧 Extracting features (window={window_size})...")
    test_features = extract_features(test_df, window_size)
    print(f"✅ Feature extraction complete")
    print(f"  Test features: {test_features.shape}")

    # Separate features and labels
    X_test = test_features.drop('label', axis=1)
    y_test, label_mapping = encode_labels(test_features['label'])

    print(f"\n🏷️  Label mapping: {label_mapping}")

    # Predict
    print(f"\n🔮 Running inference on test set...")
    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.argmax(axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n📊 TEST SET RESULTS:")
    print("=" * 80)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # Classification report
    print(f"\n  Classification Report:")
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names_subset = [list(label_mapping.keys())[i] for i in unique_classes]

    print(classification_report(
        y_test, y_pred,
        labels=unique_classes,
        target_names=target_names_subset,
        digits=4
    ))

    # Confusion matrix
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Per-class analysis
    print(f"\n  Per-Class Performance:")
    for i, label in enumerate(label_mapping.keys()):
        if i in unique_classes:
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == i).sum() / mask.sum()
                print(f"    {label:15s}: {class_acc:.4f} ({mask.sum():,} samples)")

    # Quality gate checks
    print(f"\n" + "=" * 80)
    print("QUALITY GATE CHECKS")
    print("=" * 80)

    target_accuracy = 0.90
    if accuracy >= target_accuracy:
        print(f"✅ PASS: Accuracy {accuracy:.4f} >= {target_accuracy:.2f}")
    else:
        print(f"❌ FAIL: Accuracy {accuracy:.4f} < {target_accuracy:.2f}")

    if f1 >= 0.85:
        print(f"✅ PASS: F1-Score {f1:.4f} >= 0.85")
    else:
        print(f"⚠️  WARNING: F1-Score {f1:.4f} < 0.85")

    # Check for overfitting (compare with validation performance)
    # Validation performance was 96.92%
    val_accuracy = 0.9692
    accuracy_drop = val_accuracy - accuracy

    print(f"\n  Overfitting Analysis:")
    print(f"    Validation Accuracy: {val_accuracy:.4f}")
    print(f"    Test Accuracy:       {accuracy:.4f}")
    print(f"    Accuracy Drop:       {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)")

    if accuracy_drop < 0.05:  # Less than 5% drop
        print(f"  ✅ PASS: Minimal overfitting (drop < 5%)")
    elif accuracy_drop < 0.10:  # 5-10% drop
        print(f"  ⚠️  WARNING: Moderate overfitting (drop 5-10%)")
    else:
        print(f"  ❌ FAIL: Significant overfitting (drop > 10%)")

    print("=" * 80)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'ground_truth': y_test,
        'label_mapping': label_mapping
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM model on test set"
    )
    parser.add_argument(
        '--model',
        default='../models/lightgbm_behavior.txt',
        help='Path to trained model'
    )
    parser.add_argument(
        '--test',
        default='../../datasets/test.csv',
        help='Path to test CSV'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=60,
        help='Feature extraction window size'
    )

    args = parser.parse_args()

    # Check files exist
    model_path = Path(args.model)
    test_path = Path(args.test)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        sys.exit(1)

    # Evaluate
    results = evaluate_model(
        model_path=str(model_path),
        test_data_path=str(test_path),
        window_size=args.window_size
    )

    print("\n✅ Evaluation complete!")

    # Return exit code based on quality gates
    if results['accuracy'] >= 0.90:
        sys.exit(0)
    else:
        print("\n⚠️  Model did not meet quality gate (accuracy < 90%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
