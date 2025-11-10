"""
GLEC DTG Edge AI - IBM Granite TTM-r2 Setup
Download and validate IBM's Tiny Time Mixer model for fuel prediction

Model: ibm-granite/granite-timeseries-ttm-r2
Paper: NeurIPS 2024 - Tiny Time Mixers
License: Apache 2.0
"""

import os
import sys
from pathlib import Path
import argparse
import json

try:
    import torch
    from transformers import AutoModel, AutoConfig
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install: pip install torch transformers numpy")
    sys.exit(1)


class TTMr2Setup:
    """Setup and validate IBM TTM-r2 model"""

    MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r2"
    MODEL_DIR = Path(__file__).parent.parent / "models" / "ttm-r2"

    def __init__(self):
        self.model = None
        self.config = None

    def download_model(self, force: bool = False):
        """
        Download TTM-r2 model from Hugging Face

        Args:
            force: Re-download even if exists
        """
        print(f"📥 Downloading IBM TTM-r2 from Hugging Face: {self.MODEL_NAME}")

        try:
            # Create models directory
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

            # Download config first (lightweight check)
            print("   → Fetching model configuration...")
            self.config = AutoConfig.from_pretrained(
                self.MODEL_NAME,
                cache_dir=str(self.MODEL_DIR),
                force_download=force
            )

            # Download full model
            print("   → Downloading model weights...")
            self.model = AutoModel.from_pretrained(
                self.MODEL_NAME,
                cache_dir=str(self.MODEL_DIR),
                force_download=force
            )

            print(f"✅ Model downloaded to: {self.MODEL_DIR}")
            return True

        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\nTroubleshooting:")
            print("  1. Check internet connection")
            print("  2. Verify Hugging Face access: https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2")
            print("  3. Install transformers: pip install transformers")
            return False

    def validate_model(self):
        """Validate downloaded model with dummy data"""
        print("\n🧪 Validating model with test input...")

        try:
            self.model.eval()

            # Test input: (batch=1, lookback=60, features=10)
            # Simulates 60 seconds of vehicle data with 10 features
            test_input = torch.randn(1, 60, 10)

            print(f"   → Input shape: {test_input.shape}")
            print(f"   → Running inference (zero-shot)...")

            with torch.no_grad():
                # TTM-r2 zero-shot forecasting
                output = self.model(
                    past_values=test_input,
                    freq_token=0,  # 1Hz sampling
                )

            print(f"   → Output shape: {output.prediction_outputs.shape}")
            print(f"✅ Model validation successful!")

            return True

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

    def get_model_info(self):
        """Print model specifications"""
        print("\n📊 Model Specifications:")
        print("=" * 60)

        if self.config:
            print(f"Model Name:        {self.MODEL_NAME}")
            print(f"Architecture:      Tiny Time Mixer (NeurIPS 2024)")
            print(f"License:           Apache 2.0")
            print(f"Parameters:        ~{self.count_parameters() / 1e6:.1f}M")
            print(f"Config:")
            for key, value in self.config.to_dict().items():
                if key in ['model_type', 'd_model', 'num_layers', 'prediction_length']:
                    print(f"  {key:20s}: {value}")

        print("=" * 60)

        # Performance targets
        print("\n🎯 Edge AI Performance Targets:")
        print(f"  Model Size:    < 20MB (INT8 quantized)")
        print(f"  Latency (CPU): < 15ms (P95)")
        print(f"  Accuracy:      > 85% (R² score)")
        print(f"  Power:         < 2W average")

    def count_parameters(self):
        """Count model parameters"""
        if self.model:
            return sum(p.numel() for p in self.model.parameters())
        return 0

    def save_config(self):
        """Save model configuration for integration"""
        config_path = self.MODEL_DIR / "ttm_r2_config.json"

        config_data = {
            "model_name": self.MODEL_NAME,
            "model_type": "ttm-r2",
            "parameters": self.count_parameters(),
            "input_shape": [1, 60, 10],  # batch, lookback, features
            "output_shape": "auto",
            "framework": "pytorch",
            "license": "Apache 2.0",
            "optimizations": {
                "quantization": "INT8",
                "target_size_mb": 20,
                "target_latency_ms": 15
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"\n💾 Configuration saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup IBM Granite TTM-r2 model for GLEC DTG Edge AI"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if model exists'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip model validation after download'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GLEC DTG Edge AI - IBM TTM-r2 Setup")
    print("=" * 60)

    setup = TTMr2Setup()

    # Step 1: Download
    if not setup.download_model(force=args.force):
        sys.exit(1)

    # Step 2: Get info
    setup.get_model_info()

    # Step 3: Validate
    if not args.skip_validation:
        if not setup.validate_model():
            sys.exit(1)

    # Step 4: Save config
    setup.save_config()

    print("\n" + "=" * 60)
    print("✅ TTM-r2 setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Fine-tune with our data: python train_ttm_r2.py --data ../../datasets/train.csv")
    print("  2. Compare with TCN: python compare_models.py --models ttm-r2,tcn")
    print("  3. Quantize for edge: python quantize_ttm_r2.py --method int8")
    print("\nDocumentation: docs/EDGE_AI_MODELS_COMPREHENSIVE_ANALYSIS.md")


if __name__ == "__main__":
    main()
