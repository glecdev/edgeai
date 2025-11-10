"""
TFLite Model Validator
Validate TFLite model accuracy against original LightGBM model

Ensures conversion pipeline (LightGBM → ONNX → TFLite) maintains accuracy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Import feature extraction from training script
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_lightgbm_simple import extract_features, encode_labels


def load_tflite_model(tflite_path: str):
    """Load TFLite model and return interpreter"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"📊 TFLite Model Info:")
    print(f"  Input:  {input_details[0]['shape']} ({input_details[0]['dtype'].__name__})")
    print(f"  Output: {output_details[0]['shape']} ({output_details[0]['dtype'].__name__})")

    return interpreter, input_details, output_details


def tflite_predict(interpreter, input_details, output_details, features: np.ndarray) -> np.ndarray:
    """
    Run TFLite inference

    Args:
        interpreter: TFLite interpreter
        input_details: Input tensor details
        output_details: Output tensor details
        features: Input features (N, 18)

    Returns:
        Predictions (N, num_classes) or (N,) depending on model output
    """
    num_samples = features.shape[0]
    input_dtype = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']

    predictions = []

    # Run inference sample by sample (TFLite typically expects batch_size=1)
    for i in range(num_samples):
        sample = features[i:i+1].astype(np.float32)

        # Handle quantized input
        if input_dtype == np.uint8:
            # Quantize input (simplified linear quantization)
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            sample_quantized = (sample / input_scale + input_zero_point).astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], sample_quantized)
        else:
            interpreter.set_tensor(input_details[0]['index'], sample)

        # Run inference
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])

        # Handle quantized output
        if output_dtype == np.uint8:
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        predictions.append(output)

    predictions = np.vstack(predictions)

    return predictions


def validate_accuracy(
    tflite_path: str,
    lgbm_path: str,
    test_data_path: str,
    window_size: int = 60
) -> dict:
    """
    Validate TFLite model accuracy against original LightGBM

    Args:
        tflite_path: Path to TFLite model
        lgbm_path: Path to original LightGBM model
        test_data_path: Path to test data CSV
        window_size: Feature extraction window size

    Returns:
        Dictionary with validation metrics
    """
    print("=" * 80)
    print("TFLITE MODEL ACCURACY VALIDATION")
    print("=" * 80)

    # Load models
    print(f"\n📥 Loading models...")
    print(f"  TFLite: {tflite_path}")
    print(f"  LightGBM: {lgbm_path}")

    tflite_interpreter, input_details, output_details = load_tflite_model(tflite_path)
    lgbm_model = lgb.Booster(model_file=lgbm_path)

    print(f"✅ Models loaded")

    # Load test data
    print(f"\n📥 Loading test data: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    print(f"  Test samples: {len(test_df):,}")

    # Extract features
    print(f"\n🔧 Extracting features (window={window_size})...")
    test_features = extract_features(test_df, window_size)

    X_test = test_features.drop('label', axis=1).values
    y_test, label_mapping = encode_labels(test_features['label'])

    print(f"✅ Feature extraction complete")
    print(f"  Features shape: {X_test.shape}")
    print(f"  Labels: {label_mapping}")

    # LightGBM predictions
    print(f"\n🔮 Running LightGBM inference...")
    lgbm_pred_proba = lgbm_model.predict(X_test)
    lgbm_pred = lgbm_pred_proba.argmax(axis=1)
    lgbm_accuracy = accuracy_score(y_test, lgbm_pred)
    lgbm_f1 = f1_score(y_test, lgbm_pred, average='weighted')

    print(f"  LightGBM Accuracy: {lgbm_accuracy:.6f} ({lgbm_accuracy*100:.2f}%)")
    print(f"  LightGBM F1-Score: {lgbm_f1:.6f}")

    # TFLite predictions
    print(f"\n🔮 Running TFLite inference...")
    tflite_pred_proba = tflite_predict(
        tflite_interpreter,
        input_details,
        output_details,
        X_test
    )

    # Handle different output shapes
    if tflite_pred_proba.ndim == 2 and tflite_pred_proba.shape[1] > 1:
        # Probability outputs
        tflite_pred = tflite_pred_proba.argmax(axis=1)
    elif tflite_pred_proba.ndim == 2 and tflite_pred_proba.shape[1] == 1:
        # Single output (class index)
        tflite_pred = tflite_pred_proba.squeeze().astype(int)
    else:
        # 1D output
        tflite_pred = tflite_pred_proba.astype(int)

    tflite_accuracy = accuracy_score(y_test, tflite_pred)
    tflite_f1 = f1_score(y_test, tflite_pred, average='weighted')

    print(f"  TFLite Accuracy:   {tflite_accuracy:.6f} ({tflite_accuracy*100:.2f}%)")
    print(f"  TFLite F1-Score:   {tflite_f1:.6f}")

    # Compare models
    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON")
    print("=" * 80)

    accuracy_diff = abs(lgbm_accuracy - tflite_accuracy)
    f1_diff = abs(lgbm_f1 - tflite_f1)

    print(f"\n  Model Comparison:")
    print(f"    LightGBM Accuracy: {lgbm_accuracy:.6f}")
    print(f"    TFLite Accuracy:   {tflite_accuracy:.6f}")
    print(f"    Absolute Diff:     {accuracy_diff:.6f} ({accuracy_diff*100:.2f}%)")
    print(f"")
    print(f"    LightGBM F1:       {lgbm_f1:.6f}")
    print(f"    TFLite F1:         {tflite_f1:.6f}")
    print(f"    Absolute Diff:     {f1_diff:.6f} ({f1_diff*100:.2f}%)")

    # Prediction agreement
    agreement = (lgbm_pred == tflite_pred).mean()
    print(f"\n  Prediction Agreement: {agreement:.6f} ({agreement*100:.2f}%)")

    # Quality Gates
    print("\n" + "=" * 80)
    print("QUALITY GATE CHECKS")
    print("=" * 80)

    # Gate 1: TFLite accuracy should be > 98% (allowing 1% degradation from 99.54%)
    min_accuracy = 0.98
    if tflite_accuracy >= min_accuracy:
        print(f"✅ PASS: TFLite accuracy {tflite_accuracy:.4f} >= {min_accuracy:.2f}")
    else:
        print(f"❌ FAIL: TFLite accuracy {tflite_accuracy:.4f} < {min_accuracy:.2f}")

    # Gate 2: Accuracy difference should be < 1%
    max_accuracy_diff = 0.01
    if accuracy_diff < max_accuracy_diff:
        print(f"✅ PASS: Accuracy diff {accuracy_diff:.4f} < {max_accuracy_diff:.2f}")
    else:
        print(f"❌ FAIL: Accuracy diff {accuracy_diff:.4f} >= {max_accuracy_diff:.2f}")

    # Gate 3: Prediction agreement should be > 99%
    min_agreement = 0.99
    if agreement >= min_agreement:
        print(f"✅ PASS: Prediction agreement {agreement:.4f} >= {min_agreement:.2f}")
    else:
        print(f"⚠️  WARNING: Prediction agreement {agreement:.4f} < {min_agreement:.2f}")

    # Detailed classification report
    print("\n" + "=" * 80)
    print("TFLITE CLASSIFICATION REPORT")
    print("=" * 80)

    unique_classes = np.unique(np.concatenate([y_test, tflite_pred]))
    target_names = [list(label_mapping.keys())[i] for i in unique_classes]

    print(classification_report(
        y_test, tflite_pred,
        labels=unique_classes,
        target_names=target_names,
        digits=4
    ))

    # Confusion matrix
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, tflite_pred)
    print(cm)

    # Return metrics
    metrics = {
        'lgbm_accuracy': lgbm_accuracy,
        'lgbm_f1': lgbm_f1,
        'tflite_accuracy': tflite_accuracy,
        'tflite_f1': tflite_f1,
        'accuracy_diff': accuracy_diff,
        'f1_diff': f1_diff,
        'prediction_agreement': agreement,
        'passed_gates': (
            tflite_accuracy >= min_accuracy and
            accuracy_diff < max_accuracy_diff and
            agreement >= min_agreement
        )
    }

    return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate TFLite model accuracy against original LightGBM"
    )
    parser.add_argument(
        '--tflite',
        default='../models/lightgbm_behavior.tflite',
        help='Path to TFLite model (.tflite)'
    )
    parser.add_argument(
        '--original',
        default='../models/lightgbm_behavior.txt',
        help='Path to original LightGBM model (.txt)'
    )
    parser.add_argument(
        '--test-data',
        default='../../datasets/test.csv',
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=60,
        help='Feature extraction window size (default: 60)'
    )

    args = parser.parse_args()

    # Check files exist
    tflite_path = Path(args.tflite)
    lgbm_path = Path(args.original)
    test_path = Path(args.test_data)

    if not tflite_path.exists():
        print(f"❌ TFLite model not found: {tflite_path}")
        sys.exit(1)

    if not lgbm_path.exists():
        print(f"❌ LightGBM model not found: {lgbm_path}")
        sys.exit(1)

    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        sys.exit(1)

    # Validate
    metrics = validate_accuracy(
        tflite_path=str(tflite_path),
        lgbm_path=str(lgbm_path),
        test_data_path=str(test_path),
        window_size=args.window_size
    )

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if metrics['passed_gates']:
        print("✅ ALL QUALITY GATES PASSED")
        print("\nTFLite model is ready for Android deployment!")
        print(f"  Accuracy: {metrics['tflite_accuracy']:.4f} ({metrics['tflite_accuracy']*100:.2f}%)")
        print(f"  F1-Score: {metrics['tflite_f1']:.4f}")
        print(f"  Accuracy preserved: {100-metrics['accuracy_diff']*100:.2f}%")
        exit_code = 0
    else:
        print("❌ QUALITY GATES FAILED")
        print("\nReview conversion pipeline or consider alternative approaches:")
        print("  1. Try different quantization (FP16 instead of INT8)")
        print("  2. Use native JNI wrapper for maximum accuracy")
        print("  3. Check ONNX conversion parameters")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
