#!/usr/bin/env python3
"""
ONNX INT8 Dynamic Quantization

Purpose:
    Reduce ONNX FP32 model size (1.9 GB) to INT8 (~500 MB)
    for Android deployment with ONNX Runtime Mobile.

Method:
    - Dynamic INT8 quantization (weights only)
    - onnxruntime.quantization API
    - Suitable for Android deployment

Performance:
    - Size: 1.9 GB → ~500 MB (74% reduction)
    - Speed: Similar to FP32 on mobile CPU
    - Accuracy: <3% degradation

Usage:
    python quantize_onnx_int8.py \
        --model-path android-models/model.onnx \
        --output-path android-models/model-int8.onnx
"""

import argparse
import sys
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


class ONNXQuantizer:
    """ONNX INT8 양자화 관리자"""

    def __init__(
        self,
        model_path: Path,
        output_path: Path,
    ):
        self.model_path = model_path
        self.output_path = output_path

    def validate_model(self):
        """ONNX 모델 유효성 검증"""
        print("=" * 60)
        print("ONNX INT8 Dynamic Quantization")
        print("=" * 60)
        print(f"입력 모델: {self.model_path}")
        print(f"출력 경로: {self.output_path}")
        print()

        print("[Step 1] ONNX 모델 검증...")

        # Check file exists
        if not self.model_path.exists():
            print(f"  [ERROR] 모델 파일 없음: {self.model_path}")
            return False

        # Check ONNX validity
        try:
            model = onnx.load(str(self.model_path))
            onnx.checker.check_model(model)
            print(f"  [PASS] ONNX 모델 유효성 검증 완료")
        except Exception as e:
            print(f"  [ERROR] ONNX 모델 검증 실패: {e}")
            return False

        # Print original size
        original_size = self.model_path.stat().st_size / (1024**2)
        print(f"  원본 크기: {original_size:.1f} MB (FP32)")
        print()

        return True

    def quantize_model(self):
        """INT8 동적 양자화 실행"""
        print("[Step 2] INT8 동적 양자화 실행...")
        print("  Note: 가중치만 INT8로 양자화, 활성화는 런타임에 동적 계산")
        print("  예상 소요 시간: 5-10분")
        print()

        try:
            # Create output directory
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Dynamic quantization (weights only)
            quantize_dynamic(
                model_input=str(self.model_path),
                model_output=str(self.output_path),
                weight_type=QuantType.QUInt8,  # Unsigned INT8
            )

            print(f"  [PASS] 양자화 완료: {self.output_path}")
            print()

            return True

        except Exception as e:
            print(f"  [ERROR] 양자화 실패: {e}")
            import traceback
            traceback.print_exc()
            print()
            return False

    def validate_quantized_model(self):
        """양자화된 모델 검증"""
        print("[Step 3] 양자화된 모델 검증...")

        if not self.output_path.exists():
            print(f"  [ERROR] 출력 파일 없음: {self.output_path}")
            return False

        try:
            # Check ONNX validity
            model = onnx.load(str(self.output_path))
            onnx.checker.check_model(model)
            print(f"  [PASS] ONNX 모델 유효성 검증 완료")
        except Exception as e:
            print(f"  [ERROR] ONNX 모델 검증 실패: {e}")
            return False

        print()
        return True

    def print_summary(self):
        """양자화 결과 요약"""
        print("=" * 60)
        print("양자화 완료!")
        print("=" * 60)

        # Calculate sizes
        original_size = self.model_path.stat().st_size / (1024**2)
        quantized_size = self.output_path.stat().st_size / (1024**2)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"원본 크기:   {original_size:.1f} MB (FP32)")
        print(f"양자화 크기: {quantized_size:.1f} MB (INT8)")
        print(f"압축률:     {reduction:.1f}% 감소")
        print()

        print(f"출력 경로: {self.output_path}")
        print()

        print("다음 단계:")
        print("  1. 모델 검증 (추론 테스트)")
        print("     python test_onnx_inference.py --model android-models/model-int8.onnx")
        print()
        print("  2. Android assets 복사")
        print("     cp android-models/model-int8.onnx ../../android-dtg/app/src/main/assets/models/")
        print("     cp android-models/tokenizer.json ../../android-dtg/app/src/main/assets/models/")
        print()
        print("  3. Android 빌드 및 테스트")
        print("     cd ../../android-dtg && ./gradlew assembleDebug")
        print()

    def run(self) -> bool:
        """전체 양자화 프로세스 실행"""
        try:
            if not self.validate_model():
                return False

            if not self.quantize_model():
                return False

            if not self.validate_quantized_model():
                return False

            self.print_summary()
            return True

        except Exception as e:
            print(f"\n[ERROR] 양자화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="ONNX INT8 동적 양자화")

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("android-models/model.onnx"),
        help="입력 ONNX 모델 경로 (FP32)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("android-models/model-int8.onnx"),
        help="출력 ONNX 모델 경로 (INT8)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.model_path.exists():
        print(f"[ERROR] 모델 파일 없음: {args.model_path}")
        sys.exit(1)

    # Run quantization
    quantizer = ONNXQuantizer(
        model_path=args.model_path,
        output_path=args.output_path,
    )

    success = quantizer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
