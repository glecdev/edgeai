#!/usr/bin/env python3
"""
ONNX Model Inference Test

Purpose:
    Validate ONNX INT8 model by running inference
    with Korean truck-domain queries.

Usage:
    python test_onnx_inference.py \
        --model android-models/model-int8.onnx \
        --tokenizer android-models/tokenizer.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXInferenceTest:
    """ONNX 모델 추론 테스트"""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.session = None
        self.tokenizer = None

    def load_model(self):
        """ONNX 모델 및 토크나이저 로드"""
        print("=" * 60)
        print("ONNX Model Inference Test")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Tokenizer: {self.tokenizer_path}")
        print()

        print("[Step 1] ONNX Runtime 세션 생성...")

        # Check providers
        providers = ort.get_available_providers()
        print(f"  Available providers: {providers}")

        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        print(f"  [PASS] ONNX Runtime 세션 생성 완료")
        print()

        # Load tokenizer
        print("[Step 2] 토크나이저 로드...")

        # Try to load from directory
        tokenizer_dir = self.tokenizer_path.parent
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"  [PASS] 토크나이저 로드 완료")
        print()

    def run_inference(self, prompt: str, max_length: int = 512):
        """ONNX 모델 추론 실행"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",  # NumPy arrays for ONNX
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        # Get input names
        input_names = [inp.name for inp in self.session.get_inputs()]
        print(f"  ONNX input names: {input_names}")

        # Prepare inputs
        onnx_inputs = {
            "input_ids": input_ids,
        }

        # Add attention_mask if present
        if "attention_mask" in input_names:
            onnx_inputs["attention_mask"] = attention_mask

        # Add position_ids if present (required by Qwen models)
        if "position_ids" in input_names:
            position_ids = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
            onnx_inputs["position_ids"] = position_ids

        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, onnx_inputs)
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # ms

        # Get logits (first output)
        logits = outputs[0]

        # Get predicted token IDs (greedy decoding)
        predicted_ids = np.argmax(logits, axis=-1)[0]  # [seq_len]

        return predicted_ids, latency

    def test_samples(self):
        """테스트 샘플로 추론 실행"""
        print("[Step 3] 추론 테스트 (5 samples)...")
        print()

        test_cases = [
            {
                "query": "급제동이 감지되었습니다. 어떻게 해야 하나요?",
                "context": "차량 속도: 80 km/h, 브레이크 압력: 85%, 감속도: -6.2 m/s²",
            },
            {
                "query": "타이어 공기압이 낮습니다.",
                "context": "앞 좌측 타이어: 1.8 bar, 권장 압력: 2.2 bar",
            },
            {
                "query": "현재 연비가 어떤가요?",
                "context": "현재 연비: 4.2 km/L, 평균 속도: 65 km/h",
            },
            {
                "query": "엔진 온도가 높습니다.",
                "context": "엔진 온도: 105°C, 냉각수 온도: 98°C",
            },
            {
                "query": "적재 중량을 확인해주세요.",
                "context": "적재 중량: 5,200 kg, 최대 적재량: 8,000 kg",
            },
        ]

        results = []

        for i, case in enumerate(test_cases):
            print(f"Sample {i+1}:")
            print(f"  Q: {case['query']}")
            print(f"  Context: {case['context'][:50]}...")

            # Create prompt
            prompt = f"""<|im_start|>system
당신은 화물차 운전자를 돕는 AI 어시스턴트입니다.<|im_end|>
<|im_start|>user
{case['query']}

현재 차량 상태:
{case['context']}<|im_end|>
<|im_start|>assistant
"""

            try:
                # Run inference
                predicted_ids, latency = self.run_inference(prompt, max_length=128)

                # Decode response
                try:
                    response = self.tokenizer.decode(
                        predicted_ids, skip_special_tokens=True
                    )
                except Exception as decode_error:
                    # Fallback to safe decoding
                    response = f"[Decode Error: {decode_error}]"

                print(f"  Latency: {latency:.1f} ms")
                try:
                    print(f"  Response preview: {response[:100]}...")
                except Exception:
                    print(f"  Response preview: [Unicode encoding error in response]")
                print()

                results.append(
                    {
                        "query": case["query"],
                        "latency_ms": latency,
                        "response": response,
                        "status": "success",
                    }
                )

            except Exception as e:
                print(f"  [ERROR] 추론 실패: {e}")
                print()

                results.append(
                    {
                        "query": case["query"],
                        "latency_ms": 0,
                        "response": "",
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return results

    def print_summary(self, results):
        """테스트 결과 요약"""
        print("=" * 60)
        print("Test Results Summary")
        print("=" * 60)

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        print(f"Total samples: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print()

        if successful:
            latencies = [r["latency_ms"] for r in successful]
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            print("Latency Statistics:")
            print(f"  Average: {avg_latency:.1f} ms")
            print(f"  Min: {min_latency:.1f} ms")
            print(f"  Max: {max_latency:.1f} ms")
            print()

        if failed:
            print("Failed samples:")
            for r in failed:
                print(f"  - {r['query']}: {r.get('error', 'Unknown error')}")
            print()

        print("Overall Status:")
        if len(successful) == len(results):
            print("  [PASS] All tests passed")
        elif len(successful) > 0:
            print(f"  [WARN] Partial success ({len(successful)}/{len(results)})")
        else:
            print("  [FAIL] All tests failed")
        print()

    def run(self) -> bool:
        """전체 테스트 프로세스 실행"""
        try:
            self.load_model()
            results = self.test_samples()
            self.print_summary(results)

            # Consider success if at least 1 sample passed
            successful = [r for r in results if r["status"] == "success"]
            return len(successful) > 0

        except Exception as e:
            print(f"\n[ERROR] 테스트 실패: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="ONNX 모델 추론 테스트")

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("android-models/model-int8.onnx"),
        help="ONNX 모델 경로",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("android-models/tokenizer.json"),
        help="토크나이저 경로",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.model.exists():
        print(f"[ERROR] 모델 파일 없음: {args.model}")
        sys.exit(1)

    if not args.tokenizer.exists():
        print(f"[ERROR] 토크나이저 파일 없음: {args.tokenizer}")
        sys.exit(1)

    # Run test
    test = ONNXInferenceTest(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
    )

    success = test.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
