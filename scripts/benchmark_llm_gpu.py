#!/usr/bin/env python3
"""
LLM GPU Benchmark Script
Phase 3-K: Spec-Driven Development - LLM Performance Validation

Benchmarks Qwen2.5-0.5B GGUF model on GPU for truck domain Korean responses.
"""

import os
import sys
import io
import time
import json
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Model path
MODEL_PATH = Path("d:/edgeai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")

# Test prompts for truck domain (Korean)
TRUCK_DOMAIN_PROMPTS = [
    {
        "category": "vehicle_status",
        "query": "차량 상태를 알려줘",
        "context": "현재 차량 정보: 적재 중량 5200kg, 타이어 압력 220kPa, 엔진 온도 88°C, 연비 6.8km/L"
    },
    {
        "category": "cargo_info",
        "query": "적재 정보 확인해줘",
        "context": "적재 중량: 5200kg, 최대 적재량: 8000kg, 적재율: 65%"
    },
    {
        "category": "fuel_efficiency",
        "query": "연비가 어때?",
        "context": "현재 연비: 6.8km/L, 목표 연비: 7.5km/L, 금일 주행거리: 245km"
    },
    {
        "category": "tire_pressure",
        "query": "타이어 압력 상태는?",
        "context": "전륜 좌: 220kPa, 전륜 우: 218kPa, 후륜 좌: 225kPa, 후륜 우: 222kPa, 정상 범위: 200-250kPa"
    },
    {
        "category": "safety_warning",
        "query": "안전 경고가 있어?",
        "context": "경고 없음. 모든 센서 정상. DPF 상태: 양호, 브레이크: 정상"
    },
    {
        "category": "driving_behavior",
        "query": "내 운전 습관은 어때?",
        "context": "오늘 운전 스타일: ECO_DRIVING (92% 신뢰도), 급가속 0회, 급감속 1회, 공회전 15분"
    },
    {
        "category": "engine_temp",
        "query": "엔진 온도 괜찮아?",
        "context": "엔진 온도: 88°C, 정상 범위: 80-105°C, 냉각수 온도: 85°C"
    },
    {
        "category": "dpf_status",
        "query": "DPF 상태 알려줘",
        "context": "DPF 충전율: 45%, 재생 필요: 아니오, 마지막 재생: 3일 전"
    },
]


def build_prompt(query: str, context: str) -> str:
    """Build a prompt with truck context for the LLM."""
    return f"""당신은 화물차 운전자를 돕는 AI 어시스턴트입니다.
한국어로 간결하고 친절하게 답변하세요. 100자 이내로 답변하세요.

{context}

질문: {query}
답변:"""


def benchmark_cpu():
    """Benchmark LLM on CPU."""
    from llama_cpp import Llama

    print("\n" + "=" * 60)
    print("CPU BENCHMARK")
    print("=" * 60)

    # Load model (CPU)
    print(f"\nLoading model: {MODEL_PATH}")
    start_load = time.time()

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_threads=8,
        n_gpu_layers=0,  # CPU only
        verbose=False
    )

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s (CPU)")

    results = []

    for i, prompt_data in enumerate(TRUCK_DOMAIN_PROMPTS):
        prompt = build_prompt(prompt_data["query"], prompt_data["context"])

        print(f"\n[{i+1}/{len(TRUCK_DOMAIN_PROMPTS)}] {prompt_data['category']}: {prompt_data['query']}")

        start_time = time.time()

        output = llm(
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            stop=["질문:", "\n\n"],
            echo=False
        )

        latency = (time.time() - start_time) * 1000
        response = output["choices"][0]["text"].strip()
        tokens_generated = output["usage"]["completion_tokens"]
        tokens_per_sec = tokens_generated / (latency / 1000) if latency > 0 else 0

        print(f"  Response: {response}")
        print(f"  Latency: {latency:.0f}ms, Tokens: {tokens_generated}, Speed: {tokens_per_sec:.1f} tok/s")

        results.append({
            "category": prompt_data["category"],
            "query": prompt_data["query"],
            "response": response,
            "latency_ms": latency,
            "tokens": tokens_generated,
            "tokens_per_sec": tokens_per_sec,
            "device": "CPU"
        })

    del llm
    return results


def benchmark_gpu():
    """Benchmark LLM on GPU with CUDA."""
    from llama_cpp import Llama

    print("\n" + "=" * 60)
    print("GPU BENCHMARK (CUDA)")
    print("=" * 60)

    # Load model (GPU)
    print(f"\nLoading model: {MODEL_PATH}")
    start_load = time.time()

    try:
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=512,
            n_gpu_layers=35,  # Offload all layers to GPU
            verbose=False
        )
    except Exception as e:
        print(f"GPU loading failed: {e}")
        print("Falling back to CPU...")
        return None

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s (GPU)")

    results = []

    for i, prompt_data in enumerate(TRUCK_DOMAIN_PROMPTS):
        prompt = build_prompt(prompt_data["query"], prompt_data["context"])

        print(f"\n[{i+1}/{len(TRUCK_DOMAIN_PROMPTS)}] {prompt_data['category']}: {prompt_data['query']}")

        start_time = time.time()

        output = llm(
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            stop=["질문:", "\n\n"],
            echo=False
        )

        latency = (time.time() - start_time) * 1000
        response = output["choices"][0]["text"].strip()
        tokens_generated = output["usage"]["completion_tokens"]
        tokens_per_sec = tokens_generated / (latency / 1000) if latency > 0 else 0

        print(f"  Response: {response}")
        print(f"  Latency: {latency:.0f}ms, Tokens: {tokens_generated}, Speed: {tokens_per_sec:.1f} tok/s")

        results.append({
            "category": prompt_data["category"],
            "query": prompt_data["query"],
            "response": response,
            "latency_ms": latency,
            "tokens": tokens_generated,
            "tokens_per_sec": tokens_per_sec,
            "device": "GPU"
        })

    del llm
    return results


def analyze_results(cpu_results, gpu_results):
    """Analyze and compare benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if cpu_results:
        cpu_avg_latency = sum(r["latency_ms"] for r in cpu_results) / len(cpu_results)
        cpu_avg_tokens = sum(r["tokens_per_sec"] for r in cpu_results) / len(cpu_results)
        print(f"\nCPU Performance:")
        print(f"  Avg Latency: {cpu_avg_latency:.0f}ms")
        print(f"  Avg Speed: {cpu_avg_tokens:.1f} tokens/sec")

    if gpu_results:
        gpu_avg_latency = sum(r["latency_ms"] for r in gpu_results) / len(gpu_results)
        gpu_avg_tokens = sum(r["tokens_per_sec"] for r in gpu_results) / len(gpu_results)
        print(f"\nGPU Performance:")
        print(f"  Avg Latency: {gpu_avg_latency:.0f}ms")
        print(f"  Avg Speed: {gpu_avg_tokens:.1f} tokens/sec")

        if cpu_results:
            speedup = cpu_avg_latency / gpu_avg_latency
            print(f"\nGPU Speedup: {speedup:.1f}x faster than CPU")

    # Quality check: Korean responses
    print("\n" + "-" * 40)
    print("KOREAN RESPONSE QUALITY CHECK")
    print("-" * 40)

    results_to_check = gpu_results if gpu_results else cpu_results

    for r in results_to_check:
        # Check for Korean characters
        has_korean = any('\uAC00' <= c <= '\uD7A3' for c in r["response"])
        # Check for polite endings
        polite_endings = ["입니다", "습니다", "세요", "해요", "니다", "어요"]
        is_polite = any(ending in r["response"] for ending in polite_endings)
        # Check length
        is_concise = len(r["response"]) < 100

        status = "✅" if (has_korean and is_concise) else "⚠️"
        print(f"{status} [{r['category']}] {r['response'][:50]}...")

    return {
        "cpu": cpu_results,
        "gpu": gpu_results,
        "timestamp": datetime.now().isoformat(),
        "model": str(MODEL_PATH),
        "summary": {
            "cpu_avg_latency_ms": sum(r["latency_ms"] for r in cpu_results) / len(cpu_results) if cpu_results else None,
            "gpu_avg_latency_ms": sum(r["latency_ms"] for r in gpu_results) / len(gpu_results) if gpu_results else None,
        }
    }


def main():
    print("=" * 60)
    print("LLM GPU BENCHMARK - Qwen2.5-0.5B")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return 1

    print(f"\nModel: {MODEL_PATH}")
    print(f"Size: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    # Run CPU benchmark
    print("\n[1/2] Running CPU benchmark...")
    cpu_results = benchmark_cpu()

    # Run GPU benchmark
    print("\n[2/2] Running GPU benchmark...")
    gpu_results = benchmark_gpu()

    # Analyze results
    report = analyze_results(cpu_results, gpu_results)

    # Save report
    report_path = Path("d:/edgeai/edgeai-repo/specs/validation_reports/llm_benchmark_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Report saved: {report_path}")

    # Final verdict
    print("\n" + "=" * 60)
    print("SPEC VALIDATION")
    print("=" * 60)

    target_latency = 3000  # 3 seconds
    results = gpu_results if gpu_results else cpu_results
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    if avg_latency < target_latency:
        print(f"✅ PASS: Avg latency {avg_latency:.0f}ms < {target_latency}ms target")
    else:
        print(f"❌ FAIL: Avg latency {avg_latency:.0f}ms > {target_latency}ms target")

    return 0


if __name__ == "__main__":
    sys.exit(main())
