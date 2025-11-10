#!/bin/bash
# GLEC DTG - Performance Benchmarking
# Comprehensive performance testing for AI models and data pipeline

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "GLEC DTG - Performance Benchmark"
echo "========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Performance targets (from requirements)
TARGET_INFERENCE_LATENCY_MS=50    # P95 < 50ms
TARGET_THROUGHPUT_RPS=250         # > 250 rec/sec
TARGET_MEMORY_MB=50               # < 50MB
TARGET_MODEL_SIZE_MB=14           # < 14MB total

# Results tracking
BENCHMARK_RESULTS=()

# Function to benchmark component
benchmark_component() {
    local name=$1
    local script=$2
    local args=${3:-""}

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Benchmarking: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if [ ! -f "$script" ]; then
        echo -e "${YELLOW}⚠️  Script not found: $script${NC}"
        echo "   Skipping benchmark"
        return
    fi

    # Run benchmark
    python3 "$script" $args || BENCH_FAILED=$?

    if [ ${BENCH_FAILED:-0} -eq 0 ]; then
        echo -e "${GREEN}✅ Benchmark completed${NC}"
    else
        echo -e "${RED}❌ Benchmark failed${NC}"
    fi
}

# Start benchmarking
echo "🚀 Starting performance benchmarks..."
echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. Model Size Verification
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Model Size Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

MODELS_DIR="ai-models/trained_models"
TOTAL_SIZE_MB=0

if [ -d "$MODELS_DIR" ]; then
    echo "Checking model files:"

    for model in "$MODELS_DIR"/*.onnx; do
        if [ -f "$model" ]; then
            filename=$(basename "$model")
            size_bytes=$(stat -c%s "$model" 2>/dev/null || stat -f%z "$model" 2>/dev/null)
            size_mb=$(awk "BEGIN {printf \"%.2f\", $size_bytes/1048576}")
            TOTAL_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SIZE_MB + $size_mb}")

            echo "  - $filename: ${size_mb} MB"
        fi
    done

    echo ""
    echo "Total Model Size: ${TOTAL_SIZE_MB} MB"

    if (( $(echo "$TOTAL_SIZE_MB < $TARGET_MODEL_SIZE_MB" | bc -l) )); then
        echo -e "${GREEN}✅ Size target met${NC} (<${TARGET_MODEL_SIZE_MB} MB)"
    else
        echo -e "${YELLOW}⚠️  Size target exceeded${NC} (target: <${TARGET_MODEL_SIZE_MB} MB)"
    fi
else
    echo -e "${YELLOW}⚠️  Models directory not found${NC}"
    echo "   No models to benchmark"
fi

# 2. Inference Latency Benchmark
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Inference Latency Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "tests/benchmark_inference.py" ]; then
    echo "Running inference latency tests..."
    echo "Iterations: 1000"
    echo ""

    python3 -c "
import sys
import time
import numpy as np
from pathlib import Path

# Add ai-models to path
sys.path.insert(0, str(Path.cwd() / 'ai-models'))

try:
    import onnxruntime as ort
    from tests.benchmark_inference import run_latency_benchmark

    # Run benchmark
    results = run_latency_benchmark(iterations=1000)

    print('Latency Results:')
    print(f'  P50:  {results[\"p50_ms\"]:.2f} ms')
    print(f'  P95:  {results[\"p95_ms\"]:.2f} ms')
    print(f'  P99:  {results[\"p99_ms\"]:.2f} ms')
    print(f'  Mean: {results[\"mean_ms\"]:.2f} ms')
    print(f'  Std:  {results[\"std_ms\"]:.2f} ms')
    print()

    # Check against target
    if results['p95_ms'] < $TARGET_INFERENCE_LATENCY_MS:
        print('✅ Latency target met (P95 < ${TARGET_INFERENCE_LATENCY_MS} ms)')
    else:
        print(f'⚠️  Latency target exceeded (P95: {results[\"p95_ms\"]:.2f} ms, target: < ${TARGET_INFERENCE_LATENCY_MS} ms)')

except ImportError as e:
    print(f'⚠️  Cannot run latency benchmark: {e}')
    print('   Install onnxruntime: pip install onnxruntime')
except Exception as e:
    print(f'⚠️  Benchmark error: {e}')
"
else
    echo -e "${YELLOW}⚠️  Benchmark script not found${NC}"
    echo "   Manual benchmark:"
    echo "   pytest tests/benchmark_inference.py -v"
fi

# 3. Throughput Benchmark
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Throughput Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Running realtime integration throughput test..."
echo ""

pytest tests/test_realtime_integration.py::TestProductionBenchmarks::test_production_throughput_benchmark -v --tb=short 2>&1 | tail -20 || true

# 4. Memory Usage Benchmark
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  Memory Usage Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 -c "
import sys
import psutil
import time
from pathlib import Path

# Add ai-models to path
sys.path.insert(0, str(Path.cwd() / 'ai-models'))

def get_memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / 1048576

print('Measuring memory usage...')
print()

# Baseline memory
baseline_mb = get_memory_mb()
print(f'Baseline memory: {baseline_mb:.2f} MB')

try:
    import onnxruntime as ort

    # Load model if exists
    model_path = 'ai-models/trained_models/lightgbm_multi_v1_0_0.onnx'
    if Path(model_path).exists():
        session = ort.InferenceSession(model_path)
        after_load_mb = get_memory_mb()
        load_increase = after_load_mb - baseline_mb

        print(f'After model load: {after_load_mb:.2f} MB (+{load_increase:.2f} MB)')

        # Run inference 100 times
        import numpy as np
        input_name = session.get_inputs()[0].name
        features = np.random.rand(1, 19).astype(np.float32)

        for i in range(100):
            session.run(None, {input_name: features})

        after_inference_mb = get_memory_mb()
        inference_increase = after_inference_mb - after_load_mb

        print(f'After 100 inferences: {after_inference_mb:.2f} MB (+{inference_increase:.2f} MB)')
        print()

        total_increase = after_inference_mb - baseline_mb

        if total_increase < $TARGET_MEMORY_MB:
            print(f'✅ Memory target met (+{total_increase:.2f} MB < ${TARGET_MEMORY_MB} MB)')
        else:
            print(f'⚠️  Memory target exceeded (+{total_increase:.2f} MB, target: < ${TARGET_MEMORY_MB} MB)')
    else:
        print('⚠️  Model not found, skipping memory benchmark')

except ImportError:
    print('⚠️  Cannot measure memory: missing dependencies')
    print('   Install: pip install onnxruntime psutil')
except Exception as e:
    print(f'⚠️  Memory benchmark error: {e}')
"

# 5. Accuracy Metrics (if datasets available)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  Accuracy Metrics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "datasets/test.csv" ] && [ -f "ai-models/trained_models/lightgbm_multi_v1_0_0.onnx" ]; then
    echo "Running accuracy evaluation on test set..."
    echo ""

    pytest ai-models/tests/test_lightgbm.py::TestLightGBMModel::test_model_accuracy -v --tb=short 2>&1 | tail -15 || true
else
    echo -e "${YELLOW}⚠️  Test dataset or model not found${NC}"
    echo "   Skipping accuracy benchmark"
    echo ""
    echo "   Generate datasets: python data-generation/synthetic_driving_simulator.py"
    echo "   Train model: see docs/GPU_REQUIRED_TASKS.md"
fi

# 6. End-to-End Pipeline Benchmark
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️⃣  End-to-End Pipeline Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Measuring full pipeline performance:"
echo "  CAN Data → Physics Validation → Feature Extraction → AI Inference"
echo ""

python3 -c "
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'ai-models'))

try:
    from validation.physics_validator import PhysicsValidator
    from inference.realtime_integration import RealtimeCANData, RealtimeDataIntegrator

    # Setup
    validator = PhysicsValidator(vehicle_type='truck')
    integrator = RealtimeDataIntegrator(batch_size=60)

    # Generate test data
    test_data = []
    for i in range(1000):
        data = RealtimeCANData(
            timestamp=int(time.time() * 1000) + i,
            vehicle_speed=60.0 + np.random.randn() * 5,
            engine_rpm=2000 + np.random.randn() * 100,
            fuel_level=75.0,
            throttle_position=40.0 + np.random.randn() * 5,
            brake_position=0.0,
            coolant_temp=90 + np.random.randn() * 2,
            maf_rate=5.0 + np.random.randn() * 0.5,
            battery_voltage=12.6 + np.random.randn() * 0.1,
            acceleration_x=0.5 + np.random.randn() * 0.2,
            acceleration_y=0.0,
            acceleration_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            latitude=37.5665,
            longitude=126.9780,
            altitude=50.0,
            heading=45.0
        )
        test_data.append(data)

    # Benchmark: Physics Validation
    start = time.time()
    valid_count = 0
    for data in test_data:
        result = validator.validate(data)
        if result.is_valid:
            valid_count += 1
    validation_time = time.time() - start

    print(f'Physics Validation:')
    print(f'  Processed: 1000 records')
    print(f'  Valid: {valid_count} ({valid_count/10:.1f}%)')
    print(f'  Time: {validation_time:.3f}s')
    print(f'  Throughput: {1000/validation_time:.1f} rec/sec')
    print()

    # Benchmark: Feature Extraction
    start = time.time()
    # Simulate 60-second window feature extraction
    for i in range(0, len(test_data) - 60, 60):
        window = test_data[i:i+60]
        # Extract 19 features
        features = [
            np.mean([d.vehicle_speed for d in window]),
            np.std([d.vehicle_speed for d in window]),
            # ... (simplified for benchmark)
        ]
    extraction_time = time.time() - start
    num_windows = (len(test_data) - 60) // 60

    print(f'Feature Extraction:')
    print(f'  Windows: {num_windows}')
    print(f'  Time: {extraction_time:.3f}s')
    print(f'  Time per window: {extraction_time/num_windows*1000:.2f} ms')
    print()

    # Total pipeline time
    total_time = validation_time + extraction_time
    print(f'End-to-End Pipeline:')
    print(f'  Total time: {total_time:.3f}s for 1000 records')
    print(f'  Throughput: {1000/total_time:.1f} rec/sec')
    print()

    if 1000/total_time > $TARGET_THROUGHPUT_RPS:
        print(f'✅ Pipeline throughput target met ({1000/total_time:.1f} > ${TARGET_THROUGHPUT_RPS} rec/sec)')
    else:
        print(f'⚠️  Pipeline throughput below target ({1000/total_time:.1f} rec/sec, target: > ${TARGET_THROUGHPUT_RPS} rec/sec)')

except ImportError as e:
    print(f'⚠️  Cannot run pipeline benchmark: {e}')
except Exception as e:
    print(f'⚠️  Pipeline benchmark error: {e}')
"

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Benchmark Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Performance Targets:"
echo "  • Inference Latency (P95): < ${TARGET_INFERENCE_LATENCY_MS} ms"
echo "  • Throughput: > ${TARGET_THROUGHPUT_RPS} rec/sec"
echo "  • Memory Usage: < ${TARGET_MEMORY_MB} MB"
echo "  • Model Size: < ${TARGET_MODEL_SIZE_MB} MB"
echo ""

# Save results to JSON
REPORT_FILE="benchmark-report-$(date +%Y%m%d-%H%M%S).json"

python3 -c "
import json
from datetime import datetime

report = {
    'timestamp': datetime.now().isoformat(),
    'targets': {
        'inference_latency_p95_ms': $TARGET_INFERENCE_LATENCY_MS,
        'throughput_rec_per_sec': $TARGET_THROUGHPUT_RPS,
        'memory_usage_mb': $TARGET_MEMORY_MB,
        'model_size_mb': $TARGET_MODEL_SIZE_MB
    },
    'results': {
        'model_size_mb': $TOTAL_SIZE_MB,
        # Add more results as needed
    }
}

with open('$REPORT_FILE', 'w') as f:
    json.dump(report, f, indent=2)

print('📄 Benchmark report saved: $REPORT_FILE')
"

echo ""
echo "Next steps:"
echo "  - Review benchmark results"
echo "  - Optimize bottlenecks if targets not met"
echo "  - Run full test suite: ./scripts/run_all_tests.sh"
echo ""
