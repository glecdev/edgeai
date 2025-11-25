#!/usr/bin/env python3
"""
병합된 모델 INT4 양자화 스크립트

Purpose:
    LoRA 병합 후 FP16 모델 (~980MB)을 INT4로 양자화하여
    Android 배포용으로 크기 축소 (~300MB)

Quantization Method:
    - MLC-LLM INT4 quantization
    - Compatible with Android SNPE/MLC runtime

Performance:
    - Size: 980MB → 300MB (69% reduction)
    - Speed: ~2s inference (Snapdragon QCM2290)
    - Accuracy: <5% degradation

Usage:
    python quantize_merged_model.py \\
        --model-dir ./merged-models/qwen-truck-fp16 \\
        --output-dir ./quantized-models/qwen-truck-int4 \\
        --quantization int4

References:
    - PHASE3K_LLM_INTEGRATION.md - INT4 quantization
    - LLM_SETUP_GUIDE.md - MLC-LLM quantization
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


class MLCQuantizer:
    """MLC-LLM INT4 양자화 관리자"""

    def __init__(
        self,
        model_dir: Path,
        output_dir: Path,
        quantization: str = "q4f16_1",
    ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.quantization = quantization

    def check_mlc_llm_installed(self) -> bool:
        """MLC-LLM 설치 확인"""
        print("[Check] MLC-LLM 설치 확인...")

        try:
            result = subprocess.run(
                ["mlc_llm", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"  [PASS] MLC-LLM 설치됨: {version}")
                return True
            else:
                print(f"  [FAIL] MLC-LLM 미설치")
                print(f"         설치: pip install mlc-llm==0.1.0")
                return False

        except FileNotFoundError:
            print(f"  [FAIL] MLC-LLM 미설치")
            print(f"         설치: pip install mlc-llm==0.1.0")
            return False

    def quantize_model(self):
        """모델 INT4 양자화 실행"""
        print("=" * 60)
        print("MLC-LLM INT4 Quantization")
        print("=" * 60)
        print(f"입력 모델: {self.model_dir}")
        print(f"출력 경로: {self.output_dir}")
        print(f"양자화 모드: {self.quantization} (INT4)")
        print()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # MLC-LLM quantization command
        cmd = [
            "mlc_llm",
            "convert_weight",
            str(self.model_dir),
            "--quantization",
            self.quantization,
            "-o",
            str(self.output_dir),
        ]

        print("명령어:")
        print(f"  {' '.join(cmd)}")
        print()
        print("양자화 시작 (5-10분 소요)...")
        print()

        # Run quantization
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            print(result.stdout)
            print()
            print("[PASS] 양자화 완료")
            return True

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 양자화 실패:")
            print(e.stdout)
            return False

    def copy_tokenizer(self):
        """토크나이저 복사 (양자화 안 됨)"""
        print("[Step] 토크나이저 복사...")

        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ]

        for filename in tokenizer_files:
            src = self.model_dir / filename
            dst = self.output_dir / filename

            if src.exists():
                shutil.copy2(src, dst)
                print(f"  복사: {filename}")

        print(f"  [PASS] 토크나이저 복사 완료")
        print()

    def create_config(self):
        """양자화 모델 설정 파일 생성"""
        print("[Step] 설정 파일 생성...")

        config = {
            "model_type": "qwen2",
            "quantization": self.quantization,
            "max_seq_len": 512,
            "vocab_size": 151936,  # Qwen2.5
            "context_window_size": 512,
            "prefill_chunk_size": 512,
            "tensor_parallel_shards": 1,
        }

        config_path = self.output_dir / "mlc-chat-config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"  생성: mlc-chat-config.json")
        print(f"  [PASS] 설정 파일 생성 완료")
        print()

    def print_summary(self):
        """양자화 결과 요약"""
        print("=" * 60)
        print("양자화 완료!")
        print("=" * 60)

        # Calculate sizes
        model_files = list(self.output_dir.glob("*.bin")) + list(
            self.output_dir.glob("*.safetensors")
        )
        if model_files:
            total_size = sum(f.stat().st_size for f in model_files) / (1024**2)
            print(f"모델 크기: {total_size:.1f} MB (INT4)")
        else:
            print(f"모델 크기: 확인 불가 (파일 없음)")

        print(f"출력 경로: {self.output_dir}")
        print()
        print("Android 배포 준비:")
        print("  1. 모델 파일을 Android assets/로 복사")
        print("  2. Qwen25InferenceEngine.kt의 MODEL_PATH 업데이트")
        print("  3. APK 빌드 및 테스트")
        print()

    def run(self) -> bool:
        """전체 양자화 프로세스 실행"""
        # Check MLC-LLM
        if not self.check_mlc_llm_installed():
            return False

        print()

        # Quantize
        if not self.quantize_model():
            return False

        # Copy tokenizer
        self.copy_tokenizer()

        # Create config
        self.create_config()

        # Summary
        self.print_summary()

        return True


def main():
    parser = argparse.ArgumentParser(description="병합된 모델 INT4 양자화")

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="병합된 FP16 모델 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="양자화된 모델 출력 디렉토리",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4f16_1",
        choices=["q4f16_1", "q4f32_1", "q8f16_1"],
        help="양자화 모드 (기본: q4f16_1 = INT4)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.model_dir.exists():
        print(f"[ERROR] 모델 디렉토리 없음: {args.model_dir}")
        sys.exit(1)

    # Run quantization
    quantizer = MLCQuantizer(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        quantization=args.quantization,
    )

    success = quantizer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
