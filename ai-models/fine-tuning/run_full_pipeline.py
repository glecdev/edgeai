#!/usr/bin/env python3
"""
LoRA Fine-tuning 전체 파이프라인 자동 실행 스크립트

Purpose:
    환경 검증 → 학습 → 평가 → 병합 → 양자화를 한 번에 실행

Usage:
    # 전체 파이프라인 실행
    python run_full_pipeline.py

    # 특정 단계만 실행
    python run_full_pipeline.py --steps setup train evaluate

    # Dry-run (실제 실행 안 함)
    python run_full_pipeline.py --dry-run

Estimated Time: 1.5-2.5 hours (RTX 4060 8GB)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import json


class LoRAPipeline:
    """LoRA fine-tuning 파이프라인 자동화"""

    def __init__(
        self,
        base_dir: Path = Path("."),
        output_dir: Path = Path("./outputs/qwen-lora-truck"),
        dry_run: bool = False,
    ):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.dry_run = dry_run

        self.steps_completed = []
        self.steps_failed = []

    def run_command(self, cmd: List[str], description: str) -> bool:
        """명령어 실행 (dry-run 지원)"""
        print(f"\n[{description}]")
        print(f"명령어: {' '.join(cmd)}")

        if self.dry_run:
            print("  [DRY-RUN] 실행 생략")
            return True

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] 명령 실패:")
            print(e.stdout)
            return False

    def step_setup(self) -> bool:
        """Step 1: 환경 검증"""
        print("\n" + "=" * 60)
        print("Step 1: 환경 검증")
        print("=" * 60)

        success = self.run_command(
            ["python", "setup_lora_env.py"], "Environment setup check"
        )

        if success:
            self.steps_completed.append("setup")
        else:
            self.steps_failed.append("setup")

        return success

    def step_train(self) -> bool:
        """Step 2: LoRA 학습"""
        print("\n" + "=" * 60)
        print("Step 2: LoRA Fine-tuning")
        print("=" * 60)
        print("예상 소요 시간: 1-2시간 (RTX 4060 8GB)")
        print()

        if not self.dry_run:
            user_input = input("학습을 시작하시겠습니까? (y/n): ")
            if user_input.lower() != "y":
                print("  [SKIP] 사용자가 학습 건너뜀")
                return True

        success = self.run_command(
            ["python", "train_qwen_lora.py"], "LoRA training"
        )

        if success:
            self.steps_completed.append("train")
        else:
            self.steps_failed.append("train")

        return success

    def step_evaluate(self) -> bool:
        """Step 3: 모델 평가"""
        print("\n" + "=" * 60)
        print("Step 3: 모델 평가")
        print("=" * 60)

        lora_dir = self.output_dir / "final"

        if not lora_dir.exists() and not self.dry_run:
            print(f"  [FAIL] LoRA 모델 없음: {lora_dir}")
            self.steps_failed.append("evaluate")
            return False

        success = self.run_command(
            [
                "python",
                "evaluate_lora.py",
                "--lora-dir",
                str(lora_dir),
                "--baseline",
            ],
            "Model evaluation",
        )

        if success:
            self.steps_completed.append("evaluate")

            # 평가 결과 읽기
            if not self.dry_run:
                results_path = lora_dir / "evaluation_results.json"
                if results_path.exists():
                    with open(results_path) as f:
                        results = json.load(f)

                    print("\n평가 결과:")
                    print(f"  Perplexity: {results['perplexity']:.2f}")
                    print(f"  BLEU Score: {results['bleu_score']:.2f}")
                    print(f"  Relevance: {results['relevance']:.1f}%")
                    print(f"  Latency: {results['avg_latency']:.3f}s")
        else:
            self.steps_failed.append("evaluate")

        return success

    def step_merge(self) -> bool:
        """Step 4: LoRA 어댑터 병합"""
        print("\n" + "=" * 60)
        print("Step 4: LoRA 어댑터 병합")
        print("=" * 60)

        lora_dir = self.output_dir / "final"
        merged_dir = self.base_dir / "merged-models" / "qwen-truck-fp16"

        success = self.run_command(
            [
                "python",
                "merge_lora.py",
                "--lora-dir",
                str(lora_dir),
                "--output-dir",
                str(merged_dir),
            ],
            "LoRA adapter merging",
        )

        if success:
            self.steps_completed.append("merge")
        else:
            self.steps_failed.append("merge")

        return success

    def step_quantize(self) -> bool:
        """Step 5: INT4 양자화"""
        print("\n" + "=" * 60)
        print("Step 5: INT4 양자화")
        print("=" * 60)

        merged_dir = self.base_dir / "merged-models" / "qwen-truck-fp16"
        quantized_dir = self.base_dir / "quantized-models" / "qwen-truck-int4"

        success = self.run_command(
            [
                "python",
                "quantize_merged_model.py",
                "--model-dir",
                str(merged_dir),
                "--output-dir",
                str(quantized_dir),
            ],
            "INT4 quantization",
        )

        if success:
            self.steps_completed.append("quantize")
        else:
            self.steps_failed.append("quantize")

        return success

    def run_pipeline(self, steps: List[str] = None) -> bool:
        """전체 파이프라인 실행"""

        if steps is None:
            steps = ["setup", "train", "evaluate", "merge", "quantize"]

        print("=" * 60)
        print("LoRA Fine-tuning 자동 파이프라인")
        print("=" * 60)
        print(f"실행 단계: {', '.join(steps)}")
        print(f"Dry-run: {self.dry_run}")
        print()

        step_functions = {
            "setup": self.step_setup,
            "train": self.step_train,
            "evaluate": self.step_evaluate,
            "merge": self.step_merge,
            "quantize": self.step_quantize,
        }

        for step in steps:
            if step not in step_functions:
                print(f"[WARN] 알 수 없는 단계: {step}")
                continue

            success = step_functions[step]()

            if not success and not self.dry_run:
                print(f"\n[ERROR] {step} 단계 실패, 파이프라인 중단")
                break

        # 요약 출력
        self.print_summary()

        return len(self.steps_failed) == 0

    def print_summary(self):
        """파이프라인 결과 요약"""
        print("\n" + "=" * 60)
        print("파이프라인 실행 결과")
        print("=" * 60)

        if self.steps_completed:
            print(f"✅ 완료된 단계 ({len(self.steps_completed)}):")
            for step in self.steps_completed:
                print(f"  - {step}")

        if self.steps_failed:
            print(f"\n❌ 실패한 단계 ({len(self.steps_failed)}):")
            for step in self.steps_failed:
                print(f"  - {step}")

        print()

        if len(self.steps_failed) == 0:
            print("🎉 모든 단계 성공!")
            print()
            print("Android 배포 준비:")
            print("  1. quantized-models/qwen-truck-int4/ 복사")
            print("  2. android-dtg/app/src/main/assets/models/로 이동")
            print("  3. APK 빌드 및 테스트")
        else:
            print("⚠️  일부 단계 실패, 로그 확인 필요")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning 전체 파이프라인 자동 실행"
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["setup", "train", "evaluate", "merge", "quantize"],
        help="실행할 단계 (기본: 전체)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/qwen-lora-truck"),
        help="학습 출력 디렉토리",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run 모드 (실제 실행 안 함, 명령어만 출력)",
    )

    args = parser.parse_args()

    pipeline = LoRAPipeline(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    success = pipeline.run_pipeline(steps=args.steps)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
