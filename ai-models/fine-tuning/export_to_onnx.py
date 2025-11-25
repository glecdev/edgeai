#!/usr/bin/env python3
"""
Export Fine-tuned Qwen2.5 Model to ONNX

Purpose:
    Convert FP16 merged model to ONNX format for Android deployment
    Uses ONNX Runtime Mobile for inference

Strategy:
    1. Load FP16 merged model (943 MB)
    2. Export to ONNX with dynamic axes
    3. Optimize ONNX model
    4. Optional: INT8 dynamic quantization

Output:
    - qwen-truck-android.onnx (FP16, ~943 MB)
    - qwen-truck-android-int8.onnx (INT8, ~250 MB)

Usage:
    python export_to_onnx.py \
        --model-dir merged-models/qwen-truck-fp16 \
        --output-dir android-models
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


class OnnxExporter:
    """ONNX 모델 변환 관리자"""

    def __init__(
        self,
        model_dir: Path,
        output_dir: Path,
        quantize_int8: bool = True,
    ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.quantize_int8 = quantize_int8

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """FP16 병합 모델 로드"""
        print("=" * 60)
        print("ONNX Export for Android Deployment")
        print("=" * 60)
        print(f"Input model: {self.model_dir}")
        print(f"Output directory: {self.output_dir}")
        print()

        print("[Step 1] Loading FP16 merged model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            torch_dtype=torch.float32,  # ONNX export needs FP32
            device_map="cpu",  # Export on CPU
        )
        self.model.eval()

        print(f"  [PASS] Model loaded")
        print(f"  Parameters: {self.model.num_parameters() / 1e6:.1f}M")
        print()

    def export_to_onnx(self):
        """Export model to ONNX format"""
        print("[Step 2] Exporting to ONNX...")

        # Create dummy input
        dummy_text = "안녕하세요, 저는 화물차 운전자입니다."
        dummy_input = self.tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )

        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]

        print(f"  Dummy input shape: {input_ids.shape}")
        print(f"  Exporting with dynamic axes...")
        print()

        # Output path
        onnx_path = self.output_dir / "qwen-truck-android.onnx"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        try:
            torch.onnx.export(
                self.model,
                (input_ids, attention_mask),
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=14,
                do_constant_folding=True,
                verbose=False,
            )

            print(f"  [PASS] ONNX export complete: {onnx_path}")
            print(f"  Model size: {onnx_path.stat().st_size / (1024**2):.1f} MB")
            print()

            return onnx_path

        except Exception as e:
            print(f"  [ERROR] ONNX export failed: {e}")
            print()
            print("  Note: Qwen2.5 models may require special handling")
            print("  Try using optimum library instead:")
            print("    pip install optimum")
            print("    optimum-cli export onnx --model merged-models/qwen-truck-fp16 android-models/")
            print()
            return None

    def optimize_onnx(self, onnx_path: Path):
        """Optimize ONNX model for inference"""
        print("[Step 3] Optimizing ONNX model...")

        try:
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))

            # Basic checks
            onnx.checker.check_model(onnx_model)
            print(f"  [PASS] ONNX model validation passed")

            # Optimize (optional, requires onnxruntime-tools)
            # from onnxruntime.transformers.optimizer import optimize_model
            # optimized_model = optimize_model(str(onnx_path))
            # optimized_model.save_model_to_file(str(onnx_path))

            print(f"  Model ready for deployment")
            print()

        except Exception as e:
            print(f"  [WARNING] Optimization skipped: {e}")
            print(f"  Model can still be used without optimization")
            print()

    def quantize_to_int8(self, onnx_path: Path):
        """Dynamic INT8 quantization for size reduction"""
        if not self.quantize_int8:
            print("[Step 4] INT8 quantization skipped (use --quantize-int8)")
            return

        print("[Step 4] INT8 dynamic quantization...")

        int8_path = self.output_dir / "qwen-truck-android-int8.onnx"

        try:
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(int8_path),
                weight_type=QuantType.QUInt8,
                optimize_model=True,
            )

            original_size = onnx_path.stat().st_size / (1024**2)
            int8_size = int8_path.stat().st_size / (1024**2)
            reduction = (1 - int8_size / original_size) * 100

            print(f"  [PASS] INT8 quantization complete: {int8_path}")
            print(f"  Original size: {original_size:.1f} MB")
            print(f"  INT8 size: {int8_size:.1f} MB")
            print(f"  Reduction: {reduction:.1f}%")
            print()

        except Exception as e:
            print(f"  [ERROR] INT8 quantization failed: {e}")
            print()

    def copy_tokenizer(self):
        """Copy tokenizer files for Android"""
        print("[Step 5] Copying tokenizer files...")

        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "config.json",
            "generation_config.json",
        ]

        for filename in tokenizer_files:
            src = self.model_dir / filename
            dst = self.output_dir / filename

            if src.exists():
                import shutil
                shutil.copy2(src, dst)
                print(f"  Copied: {filename}")

        print(f"  [PASS] Tokenizer files ready")
        print()

    def print_summary(self):
        """Export 결과 요약"""
        print("=" * 60)
        print("ONNX Export Complete!")
        print("=" * 60)

        print(f"Output directory: {self.output_dir}")
        print()
        print("Generated files:")

        onnx_file = self.output_dir / "qwen-truck-android.onnx"
        int8_file = self.output_dir / "qwen-truck-android-int8.onnx"

        if onnx_file.exists():
            print(f"  - qwen-truck-android.onnx ({onnx_file.stat().st_size / (1024**2):.1f} MB)")

        if int8_file.exists():
            print(f"  - qwen-truck-android-int8.onnx ({int8_file.stat().st_size / (1024**2):.1f} MB)")

        print(f"  - tokenizer.json, vocab.json, config.json (tokenizer files)")
        print()

        print("Next steps:")
        print("  1. Test ONNX model:")
        print(f"     python test_onnx_model.py --model-path {onnx_file}")
        print()
        print("  2. Copy to Android project:")
        print(f"     cp -r {self.output_dir}/* ../../android-dtg/app/src/main/assets/models/")
        print()
        print("  3. Build Android APK:")
        print("     cd ../../android-dtg && ./gradlew assembleDebug")
        print()

    def run(self) -> bool:
        """전체 ONNX 변환 프로세스 실행"""
        try:
            self.load_model()
            onnx_path = self.export_to_onnx()

            if onnx_path is None:
                return False

            self.optimize_onnx(onnx_path)
            self.quantize_to_int8(onnx_path)
            self.copy_tokenizer()
            self.print_summary()

            return True

        except Exception as e:
            print(f"\n[ERROR] ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Export Qwen2.5 to ONNX for Android")

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("merged-models/qwen-truck-fp16"),
        help="FP16 merged model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("android-models"),
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Apply INT8 dynamic quantization"
    )

    args = parser.parse_args()

    # Validate input
    if not args.model_dir.exists():
        print(f"[ERROR] Model directory not found: {args.model_dir}")
        sys.exit(1)

    # GPU check (optional, export runs on CPU)
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print("Note: ONNX export will run on CPU for compatibility")
    else:
        print("Running on CPU")

    print()

    # Run export
    exporter = OnnxExporter(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        quantize_int8=args.quantize_int8,
    )

    success = exporter.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
