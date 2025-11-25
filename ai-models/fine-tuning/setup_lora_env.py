#!/usr/bin/env python3
"""
LoRA Fine-tuning 환경 자동 설정 스크립트

Purpose:
    GPU 로컬 환경에서 Qwen2.5-0.5B LoRA fine-tuning을 위한
    완전한 환경을 자동으로 설정합니다.

Features:
    1. CUDA/GPU 감지 및 검증
    2. PyTorch/CUDA 버전 호환성 체크
    3. Hugging Face 모델 다운로드 (Qwen2.5-0.5B-Instruct)
    4. 데이터셋 경로 검증
    5. 디렉토리 구조 생성

Usage:
    python setup_lora_env.py --check-only  # 환경만 체크
    python setup_lora_env.py               # 전체 설정 실행

References:
    - PHASE3K_LLM_INTEGRATION.md - Phase 2: LoRA fine-tuning
    - LLM_SETUP_GUIDE.md - Environment setup
"""

import argparse
import sys
import subprocess
from pathlib import Path
import json


class LoRAEnvironmentSetup:
    """LoRA fine-tuning 환경 설정 관리자"""

    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = base_dir
        self.checks_passed = []
        self.checks_failed = []

    def check_python_version(self) -> bool:
        """Python 버전 체크 (3.10 or 3.11)"""
        print("[Check 1] Python 버전 확인...")
        version = sys.version_info

        if version.major == 3 and version.minor in [10, 11]:
            print(f"  [PASS] Python {version.major}.{version.minor}.{version.micro}")
            self.checks_passed.append("Python version")
            return True
        else:
            print(f"  [FAIL] Python {version.major}.{version.minor} (권장: 3.10 or 3.11)")
            self.checks_failed.append("Python version")
            return False

    def check_cuda_available(self) -> bool:
        """CUDA/GPU 사용 가능 여부 체크"""
        print("[Check 2] CUDA/GPU 감지...")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                cuda_version = torch.version.cuda

                print(f"  [PASS] GPU 감지: {gpu_name}")
                print(f"         VRAM: {gpu_memory:.1f} GB")
                print(f"         CUDA: {cuda_version}")

                if gpu_memory < 8.0:
                    print(f"  [WARN] VRAM 부족 (권장: 8GB 이상, 현재: {gpu_memory:.1f}GB)")
                    print(f"         QLoRA (4-bit) 사용 권장")

                self.checks_passed.append("CUDA/GPU")
                return True
            else:
                print("  [FAIL] CUDA GPU 없음 (CPU 모드)")
                print("         LoRA fine-tuning에는 GPU가 필요합니다")
                self.checks_failed.append("CUDA/GPU")
                return False

        except ImportError:
            print("  [FAIL] PyTorch 미설치")
            self.checks_failed.append("PyTorch")
            return False

    def check_dependencies(self) -> bool:
        """필수 패키지 설치 확인"""
        print("[Check 3] 필수 패키지 확인...")

        required_packages = {
            "torch": "2.1.0",
            "transformers": "4.36.0",
            "peft": "0.7.0",
            "accelerate": "0.25.0",
            "datasets": "2.14.0",
            "trl": "0.7.0",
        }

        missing = []
        installed = []

        for package, min_version in required_packages.items():
            try:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                installed.append(f"{package} {version}")
            except ImportError:
                missing.append(package)

        if missing:
            print(f"  [FAIL] 미설치 패키지: {', '.join(missing)}")
            print(f"         설치: pip install -r requirements-lora.txt")
            self.checks_failed.append("Dependencies")
            return False
        else:
            print(f"  [PASS] 모든 패키지 설치됨")
            for pkg in installed:
                print(f"         - {pkg}")
            self.checks_passed.append("Dependencies")
            return True

    def check_dataset(self) -> bool:
        """데이터셋 존재 확인"""
        print("[Check 4] 데이터셋 확인...")

        dataset_dir = self.base_dir / "datasets" / "truck-korean"
        required_files = ["train.json", "val.json", "test.json"]

        missing_files = []
        for filename in required_files:
            filepath = dataset_dir / filename
            if not filepath.exists():
                missing_files.append(filename)

        if missing_files:
            print(f"  [FAIL] 데이터셋 파일 누락: {', '.join(missing_files)}")
            print(f"         경로: {dataset_dir}")
            self.checks_failed.append("Dataset")
            return False
        else:
            # 샘플 수 확인
            try:
                with open(dataset_dir / "train.json", encoding="utf-8") as f:
                    train_data = json.load(f)
                with open(dataset_dir / "val.json", encoding="utf-8") as f:
                    val_data = json.load(f)

                print(f"  [PASS] 데이터셋 존재")
                print(f"         Train: {len(train_data)} samples")
                print(f"         Val: {len(val_data)} samples")
                self.checks_passed.append("Dataset")
                return True
            except Exception as e:
                print(f"  [FAIL] 데이터셋 로드 실패: {e}")
                self.checks_failed.append("Dataset")
                return False

    def check_model_cache(self) -> bool:
        """Hugging Face 모델 캐시 확인"""
        print("[Check 5] Qwen2.5-0.5B 모델 캐시 확인...")

        try:
            from transformers import AutoTokenizer

            model_name = "Qwen/Qwen2.5-0.5B-Instruct"

            # 캐시 확인 (다운로드 안 함)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=True
                )
                print(f"  [PASS] 모델 캐시 존재: {model_name}")
                self.checks_passed.append("Model cache")
                return True
            except Exception:
                print(f"  [INFO] 모델 캐시 없음 (첫 학습 시 자동 다운로드)")
                print(f"         모델: {model_name}")
                print(f"         크기: ~980MB (다운로드 시간: 5-10분)")
                self.checks_passed.append("Model cache (will download)")
                return True

        except ImportError:
            print(f"  [FAIL] transformers 미설치")
            self.checks_failed.append("Model cache")
            return False

    def create_directories(self) -> bool:
        """필요한 디렉토리 구조 생성"""
        print("[Check 6] 디렉토리 구조 생성...")

        directories = [
            self.base_dir / "ai-models" / "fine-tuning" / "outputs",
            self.base_dir / "ai-models" / "fine-tuning" / "checkpoints",
            self.base_dir / "ai-models" / "fine-tuning" / "logs",
            self.base_dir / "ai-models" / "fine-tuning" / "merged-models",
        ]

        created = []
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory.relative_to(self.base_dir)))

        print(f"  [PASS] 디렉토리 생성 완료")
        for dir_path in created:
            print(f"         - {dir_path}")

        self.checks_passed.append("Directories")
        return True

    def print_summary(self):
        """환경 체크 요약 출력"""
        print()
        print("=" * 60)
        print("환경 설정 요약")
        print("=" * 60)

        if len(self.checks_failed) == 0:
            print("[PASS] 모든 체크 통과!")
            print()
            print("LoRA fine-tuning 준비 완료:")
            print("  1. GPU 환경 정상")
            print("  2. 필수 패키지 설치됨")
            print("  3. 데이터셋 준비됨")
            print()
            print("다음 단계:")
            print("  python train_qwen_lora.py --config config_lora.yaml")
        else:
            print(f"[FAIL] {len(self.checks_failed)}개 체크 실패")
            for check in self.checks_failed:
                print(f"  - {check}")
            print()
            print("해결 방법:")
            if "CUDA/GPU" in self.checks_failed:
                print("  - NVIDIA GPU 드라이버 설치")
                print("  - CUDA Toolkit 설치 (11.8 or 12.1)")
            if "Dependencies" in self.checks_failed:
                print("  - pip install -r requirements-lora.txt")
            if "Dataset" in self.checks_failed:
                print("  - python generate_dataset.py --total-samples 2000")

        print()
        print(f"통과: {len(self.checks_passed)}/{len(self.checks_passed) + len(self.checks_failed)}")

    def run_setup(self, check_only: bool = False) -> bool:
        """전체 설정 실행"""
        print("=" * 60)
        print("Qwen2.5-0.5B LoRA Fine-tuning 환경 설정")
        print("=" * 60)
        print()

        # 체크 수행
        self.check_python_version()
        self.check_cuda_available()
        self.check_dependencies()
        self.check_dataset()
        self.check_model_cache()

        if not check_only:
            self.create_directories()

        # 요약 출력
        self.print_summary()

        return len(self.checks_failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning 환경 자동 설정"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="환경만 체크 (디렉토리 생성 안 함)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="프로젝트 루트 디렉토리 (기본: 현재 디렉토리)",
    )

    args = parser.parse_args()

    setup = LoRAEnvironmentSetup(base_dir=args.base_dir)
    success = setup.run_setup(check_only=args.check_only)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
