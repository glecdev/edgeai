# CLAUDE.md

**This file provides guidance to Claude Code when working in this repository.**

---

## 🚨 IMMEDIATE DIRECTIVES - Read This First!

**Always check current phase status before starting any task:**
1. See `docs/GPU_REQUIRED_TASKS.md` for GPU-dependent tasks (Phase 2 local)
2. See `docs/PHASE3_TESTING.md` for testing requirements (Phase 3)
3. Current branch: `claude/artifact-701ca010-011CUxNEi8V3zxgnuGp9E8Ss`

### Quick Decision Tree

```
┌─ Is task GPU-dependent? (CARLA, model training, quantization)
│  YES → Document in GPU_REQUIRED_TASKS.md, defer to local environment
│  NO  → Continue below
│
├─ Can task run in web environment?
│  NO (requires: Android build, STM32 compile, hardware) → Document and defer
│  YES → Continue with TDD workflow below
│
└─ Implementing new feature or fixing bug?
   → Follow Red-Green-Refactor cycle (see below)
```

### ⚡ ROOT CAUSE RESOLUTION - Production-Grade Quality Imperative

**CRITICAL**: 근본 문제를 해결하라. 단순화나 우회로 상용화급 품질을 포기하지 마라.

**철칙 (DO NOT VIOLATE)**:

1. **절대 단순화하지 마라 (NEVER Simplify)**
   - ❌ 의존성 설치 실패 → 간단한 버전 만들기 (BAD)
   - ✅ 의존성 설치 실패 → 근본 원인 해결 (`--ignore-installed`, 가상환경 수정 등)
   - ❌ 테스트 실패 → 테스트 비활성화 (BAD)
   - ✅ 테스트 실패 → 코드 또는 테스트 수정으로 해결

2. **상용화급 품질 기준**
   - 모든 의존성 제대로 설치 (MLflow, transformers, 모든 production 도구)
   - 원본 코드 사용 (simplified 버전 금지)
   - 전체 테스트 suite 통과
   - Production 설정 유지 (config, logging, monitoring)

3. **의존성 문제 해결 방법**
   ```bash
   # ❌ 나쁜 예: MLflow 제거하고 단순 버전 만들기
   # ✅ 좋은 예: 근본 원인 해결
   pip install --ignore-installed blinker mlflow  # 시스템 패키지 충돌 우회
   pip install --upgrade --force-reinstall mlflow  # 강제 재설치
   python3 -m venv venv && source venv/bin/activate  # 가상환경 사용
   ```

4. **품질 체크리스트 (Every Commit)**
   - [ ] 모든 production 의존성 설치됨?
   - [ ] 원본 코드 사용 (simplified/workaround 없음)?
   - [ ] 전체 기능 동작 (일부 비활성화 없음)?
   - [ ] 테스트 통과 (skip 없음)?

**예시: MLflow 설치 실패 해결**
```python
# ❌ 잘못된 접근 (단순화)
# - train_lightgbm.py에서 MLflow import 제거
# - train_lightgbm_simple.py 생성 (기능 축소)
# → 상용화급 품질 저하!

# ✅ 올바른 접근 (근본 해결)
# 1. 오류 분석: blinker 1.7.0 시스템 패키지 충돌
# 2. 해결: pip install --ignore-installed blinker mlflow
# 3. 검증: import mlflow 성공
# → Production-grade 유지!
```

**실패 사례 (Learn from This)**:
- Issue: MLflow 설치 시 blinker 충돌
- 잘못된 해결: `train_lightgbm_simple.py` 생성 (MLflow 제거)
- 문제: Production tracking, experiment management 상실
- 올바른 해결: `--ignore-installed` 플래그로 근본 해결

**Remember**: "재귀적으로 천천히 상용화급으로 진행" = 근본 문제 해결, 품질 타협 금지

---

### Environment Constraints (Web-Based Development)

**Available**:
✅ Python scripting (data generation, tests, utilities)
✅ Kotlin/Java source code (Android apps)
✅ C/C++ source code (STM32 firmware)
✅ Documentation and configuration files
✅ Git operations

**NOT Available** (Defer to local):
❌ GPU operations (CARLA, model training)
❌ Android builds (`./gradlew assembleDebug`)
❌ STM32 firmware compilation (`make`)
❌ Hardware testing (CAN bus, BLE, sensors)
❌ SNPE/TFLite model conversion

---

## ⚡ Quick Start (Local Environment Only)

**Note**: These setup steps are for local development environments with full hardware access.

### Initial Setup
```bash
# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Android setup (requires Android Studio + NDK 26.1.10909125)
cd android-dtg
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Prerequisites
- Python 3.9 or 3.10
- Android Studio Hedgehog | 2023.1.1+
- Android NDK 26.1.10909125
- Qualcomm SNPE SDK (device deployment only)
- STM32CubeIDE (firmware development only)

For detailed architecture and integration information, see [README.md](README.md).

---

## 🔴🟢🔵 Red-Green-Refactor: The Core Development Cycle

**You are an expert practitioner of Test-Driven Development (TDD) and the Red-Green-Refactor cycle.**

### The Three States

1. **🔴 RED**: Write a failing test first
   - Test describes desired behavior
   - Test fails because feature doesn't exist yet
   - Compile errors are also "red"

2. **🟢 GREEN**: Write minimal code to make test pass
   - Only enough to pass the test
   - Don't worry about elegance yet
   - Shortcuts are okay if they work

3. **🔵 REFACTOR**: Improve code structure without changing behavior
   - Extract functions
   - Rename for clarity
   - Remove duplication
   - **SEPARATE COMMIT** from behavioral changes

### TDD Workflow (Step-by-Step)

When given a task like "Add harsh braking detection":

**Step 1 - RED**: Write the test first
```python
# tests/test_anomaly_detection.py
def test_harsh_braking_detection():
    """Test that harsh braking is detected when deceleration < -4 m/s²"""
    can_data = CANData(
        acceleration_x=-5.0,  # Harsh deceleration
        brake_position=80.0,
        vehicle_speed=60.0,
        # ... other fields
    )

    assert can_data.isHarshBraking() == True  # FAILS - method doesn't exist
```

Run test: `pytest tests/test_anomaly_detection.py -v`
**Expected**: ❌ Test fails (AttributeError: isHarshBraking)

**Step 2 - GREEN**: Write minimal code to pass
```kotlin
// CANData.kt
fun isHarshBraking(): Boolean {
    return accelerationX < -4.0f && brakePosition > 50.0f
}
```

Run test: `pytest tests/test_anomaly_detection.py -v`
**Expected**: ✅ Test passes

**Step 3 - REFACTOR** (if needed, separate commit):
```kotlin
// Extract threshold constants
companion object {
    private const val HARSH_BRAKING_THRESHOLD = -4.0f
    private const val BRAKE_ACTIVATION_THRESHOLD = 50.0f
}

fun isHarshBraking(): Boolean {
    return accelerationX < HARSH_BRAKING_THRESHOLD &&
           brakePosition > BRAKE_ACTIVATION_THRESHOLD
}
```

**Step 4 - COMMIT** (see Commit Discipline below)

---

## 📝 Commit Discipline: Tidy First

**NEVER mix structural and behavioral changes in a single commit.**

### Two Types of Changes

**STRUCTURAL CHANGES** (code reorganization, no behavior change):
- Extracting functions or classes
- Renaming variables/functions/files
- Moving code between files
- Formatting, comments, documentation
- Removing dead code
- Commit prefix: `refactor:`, `docs:`, `style:`

**BEHAVIORAL CHANGES** (functionality change):
- Adding new features
- Fixing bugs
- Changing logic or algorithms
- Modifying data structures
- Commit prefix: `feat:`, `fix:`, `perf:`

### Tidy First Principle

When you notice code needs both:
1. **First commit** (structural): Clean up the code
2. **Second commit** (behavioral): Add the new feature

Example:
```bash
# BAD (mixed changes)
git commit -m "refactor CAN parser and add anomaly detection"

# GOOD (separated)
git commit -m "refactor: Extract CAN message parsing to dedicated class"
git commit -m "feat: Add harsh braking anomaly detection

- Detect deceleration < -4 m/s²
- Requires brake position > 50%
- Add test coverage for edge cases"
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`

**Examples**:
```
feat(android-dtg): Add real-time anomaly detection

- Implement harsh braking detection (accel < -4 m/s²)
- Implement harsh acceleration detection (accel > 3 m/s²)
- Add UI alerts for detected anomalies
- Test coverage: 95%

Closes #42
```

```
refactor(can-parser): Extract OBD-II parsing logic

- Move PID parsing to separate class
- Add type safety with sealed classes
- Improve readability with extension functions
- No behavior changes

Part of #38
```

---

## 🐛 Defect Handling Protocol

When fixing bugs, follow this specific sequence:

**Step 1**: Write API-level failing test
```python
def test_fuel_calculation_doesnt_crash_on_zero_speed():
    """High-level test showing user impact"""
    result = calculate_fuel_efficiency(speed=0.0, maf=5.0)
    assert result == 0.0  # Should return 0, not crash
```

**Step 2**: Write minimal reproduction test
```python
def test_division_by_zero_in_fuel_formula():
    """Specific bug reproduction for debugging"""
    speed = 0.0
    maf = 5.0
    # This line causes ZeroDivisionError
    fuel_per_km = (maf / 14.7 * 3600 / 750) / speed * 100
```

**Step 3**: Fix to make both tests pass
```python
def calculate_fuel_efficiency(speed: float, maf: float) -> float:
    if speed < 1.0 or maf < 0.1:  # Guard against division by zero
        return 0.0
    return (maf / 14.7 * 3600 / 750) / speed * 100
```

**Step 4**: Commit with context
```
fix: Prevent division by zero in fuel efficiency calculation

- Add guard clause for zero/near-zero speed
- Return 0.0 instead of crashing
- Add regression tests (API-level and unit-level)

Fixes #127
```

---

## 🔄 Complete Development Workflow

Execute these steps in order for every task:

### 1. PLAN
- Read current phase requirements (`GPU_REQUIRED_TASKS.md` or `PHASE3_TESTING.md`)
- Create todo list: `TodoWrite` tool
- Identify if task is web-compatible or requires local environment
- Design approach, identify affected files

### 2. RED - Write Failing Test
```bash
# Python
pytest tests/test_new_feature.py -v  # Should fail

# Android (Kotlin)
cd android-dtg
./gradlew testDebugUnitTest  # Should fail

# STM32 (if applicable)
cd stm32-firmware
make test  # Should fail
```

### 3. GREEN - Implement Minimal Code
- Write only enough code to pass the test
- Don't optimize yet
- Run test again: `pytest tests/test_new_feature.py -v`  # Should pass

### 4. VERIFY - Run Full Test Suite
```bash
# Ensure no regressions
pytest tests/ -v --cov

# Check coverage target (>80%)
pytest tests/ --cov-report=html
open htmlcov/index.html
```

### 5. REFACTOR - Improve Structure (Optional, Separate Commit)
If code needs improvement:
```bash
# Commit behavioral change first
git add tests/ src/
git commit -m "feat: Add harsh braking detection"

# Then refactor (separate commit)
git add src/
git commit -m "refactor: Extract braking threshold constants"
```

### 6. DOCUMENT
- Update relevant docs if API changed
- Add docstrings to new functions
- Update CLAUDE.md if workflow changed

### 7. COMMIT & PUSH
```bash
# Semantic commit message
git add -A
git commit -m "feat(scope): Description

- Bullet points of changes
- Performance impacts
- Breaking changes if any

Closes #issue"

# Push to feature branch
git push -u origin claude/artifact-701ca010-011CUxNEi8V3zxgnuGp9E8Ss
```

### 8. REPEAT
Mark current todo as complete, move to next task

---

## 🎯 Quality Gates

Before committing, verify:

**Test Quality Gates**:
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Coverage >80% for new code
- [ ] No regressions in existing tests
- [ ] Performance targets met (if applicable)

**Code Quality Gates**:
- [ ] No mixed structural/behavioral changes
- [ ] Clear, descriptive commit message
- [ ] Documentation updated if needed
- [ ] No security vulnerabilities introduced

**Review Questions**:
- [ ] Is this the minimal change to achieve the goal?
- [ ] Are structural and behavioral changes separated?
- [ ] Does the test clearly describe the expected behavior?
- [ ] Will another developer understand this in 6 months?

---

## 📚 Project Context: GLEC DTG Edge AI SDK

**An edge AI system for commercial vehicle telematics running on STM32 MCU + Qualcomm Snapdragon Android hardware.**

### Core Requirements
- **Model Size**: < 100MB total
- **Inference Latency**: < 50ms (P95)
- **Power Consumption**: < 2W average
- **Accuracy**: > 85% for behavior classification
- **Data Collection**: 1Hz from CAN bus
- **AI Inference**: Every 60 seconds

### Hardware Platform
- **STM32 MCU**: CAN bus interface, sensor management, real-time operations (<1ms response)
- **Qualcomm Snapdragon**: Android OS, AI inference (DSP/HTP acceleration), application layer
- **Communication**: UART 921600 baud (STM32 ↔ Snapdragon), CAN bus (vehicle), BLE (driver app)

### Repository Structure

```
edgeai/
├── ai-models/              # Edge AI model development
│   ├── training/           # PyTorch/TensorFlow training scripts
│   ├── optimization/       # Quantization (PTQ/QAT), pruning
│   ├── conversion/         # ONNX → TFLite/SNPE DLC
│   └── tests/              # Model unit tests (TCN, LSTM-AE, LightGBM)
│
├── android-dtg/            # DTG device Android application
│   ├── app/src/main/java/  # Kotlin/Java (Services, UI, BLE)
│   ├── app/src/main/cpp/   # JNI bridge for UART/SNPE
│   └── app/src/test/       # Unit tests
│
├── android-driver/         # Driver smartphone application
│   ├── app/src/main/java/  # BLE client, Voice AI, External APIs
│   └── app/src/test/       # Unit tests
│
├── stm32-firmware/         # STM32 CAN bridge firmware
│   ├── Core/               # HAL drivers, main loop, CAN/UART
│   └── Tests/              # Hardware-in-loop tests
│
├── fleet-integration/      # Fleet AI platform connectivity
│   └── mqtt-client/        # MQTT over TLS, offline queuing
│
├── data-generation/        # Synthetic data generation
│   ├── carla-scenarios/    # CARLA simulator integration (GPU)
│   └── augmentation/       # Time-series augmentation
│
├── tests/                  # Integration tests
│   ├── e2e_test.py         # End-to-end data flow validation
│   ├── data_validator.py   # Dataset quality checks
│   ├── benchmark_inference.py  # Performance benchmarking
│   └── test_can_parser.py  # CAN protocol unit tests
│
└── docs/                   # Documentation
    ├── GPU_REQUIRED_TASKS.md      # Phase 2 local tasks (GPU)
    ├── PHASE3_TESTING.md          # Testing strategy
    └── RECURSIVE_WORKFLOW.md      # Detailed workflow guide
```

### AI Model Stack
1. **TCN**: Fuel consumption prediction (2-4MB, 15-25ms, 85-90% accuracy)
2. **LSTM-Autoencoder**: Anomaly detection (2-3MB, 25-35ms, F1 0.85-0.92)
3. **LightGBM**: Behavior classification (5-10MB, 5-15ms, 90-95% accuracy)

**Total**: ~12MB models, 30ms parallel inference

### Production Integration (Phase 3-A) ✅

**Integrated from**: [glec-dtg-ai-production](https://github.com/glecdev/glec-dtg-ai-production)

**Key Modules Added**:
1. **Realtime Data Pipeline** - 47x performance improvement (238s → 5s), 254.7 rec/sec
2. **Physics-Based Validation** - Newton's laws, 6 anomaly types, sensor correlation
3. **J1939 CAN Protocol** - 12 PGNs for commercial vehicles (Engine, TPMS, Weight, DPF)
4. **3D Dashboard** - Three.js truck rendering with WebView bridge
5. **AI Model Manager** - Semantic versioning, hot-swapping, multi-runtime (SNPE/TFLite)
6. **Truck Voice Commands** - 12 Korean truck-specific voice commands

**Impact**: 50-60% development time reduction through production code reuse

See [docs/INTEGRATION_ANALYSIS.md](docs/INTEGRATION_ANALYSIS.md) and [README.md](README.md) for details.

---

## 🛠️ Common Workflows

### Adding a New Feature

```bash
# 1. Create test (RED)
cat > tests/test_speeding_detection.py << 'EOF'
def test_speeding_detection_over_100kmh():
    can_data = CANData(vehicle_speed=110.0, ...)
    assert can_data.isSpeeding() == True
EOF

pytest tests/test_speeding_detection.py -v  # FAILS

# 2. Implement (GREEN)
# Edit: android-dtg/app/src/main/java/com/glec/dtg/models/CANData.kt
# Add: fun isSpeeding() = vehicleSpeed > 100.0f

pytest tests/test_speeding_detection.py -v  # PASSES

# 3. Run full suite (VERIFY)
pytest tests/ -v --cov

# 4. Commit (behavioral)
git add tests/ android-dtg/
git commit -m "feat: Add speeding detection (>100 km/h)"

# 5. Refactor if needed (structural, separate commit)
# Extract threshold constant
git commit -m "refactor: Extract speeding threshold constant"

# 6. Push
git push
```

### Fixing a Bug

```bash
# 1. Write failing tests (both levels)
cat > tests/test_bugfix_fuel_zero_speed.py << 'EOF'
def test_fuel_calc_zero_speed_api():
    # API-level: user impact
    assert calculate_fuel(speed=0, maf=5.0) == 0.0

def test_fuel_calc_zero_speed_unit():
    # Unit-level: specific bug
    # Previously raised ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        result = fuel_flow / 0  # The bug
EOF

pytest tests/test_bugfix_fuel_zero_speed.py -v  # FAILS

# 2. Fix the bug
# Add guard: if speed < 1.0: return 0.0

pytest tests/test_bugfix_fuel_zero_speed.py -v  # PASSES

# 3. Verify no regressions
pytest tests/ -v

# 4. Commit with issue reference
git commit -m "fix: Prevent ZeroDivisionError in fuel calculation

- Add guard clause for zero/near-zero speed
- Return 0.0 instead of crashing
- Add API-level and unit-level regression tests

Fixes #127"

# 5. Push
git push
```

### Refactoring Existing Code

```bash
# 1. Ensure tests exist and pass
pytest tests/test_can_parser.py -v  # All green

# 2. Refactor (structural only, no behavior change)
# Example: Extract method, rename variables, improve structure

# 3. Verify tests still pass (same behavior)
pytest tests/test_can_parser.py -v  # Still all green

# 4. Commit (structural only)
git commit -m "refactor: Extract OBD-II PID parsing to separate class

- Improve code organization
- Better type safety with sealed classes
- No behavior changes
- All tests still pass"

# 5. Push
git push
```

---

## 🔧 Technical Quick Reference

### Test Execution

```bash
# Python AI Models & Data Generation (Web-Compatible)
pytest tests/test_synthetic_simulator.py -v      # 14 tests - synthetic data generator
pytest ai-models/tests/test_tcn.py -v            # TCN fuel prediction model
pytest ai-models/tests/test_lstm_ae.py -v        # LSTM-AE anomaly detection
pytest ai-models/tests/test_lightgbm.py -v       # LightGBM behavior classification

# Production Integration Tests (Phase 3-A)
python -m unittest tests.test_can_parser           # 18 tests - CAN protocol
python -m unittest tests.test_realtime_integration # 8 tests - realtime pipeline
python -m unittest tests.test_physics_validation   # 20+ tests - physics checks

# Coverage Report
pytest tests/ -v --cov=ai-models --cov=fleet-integration --cov-report=html

# Android DTG (Requires Android SDK)
cd android-dtg
./gradlew testDebugUnitTest
./gradlew connectedAndroidTest  # Requires device/emulator

# Integration Tests (Requires Hardware)
python tests/e2e_test.py --duration 300
python tests/benchmark_inference.py --model tcn --iterations 1000
```

### Data Generation (CPU-Only, No GPU Required)

```bash
# Generate 35,000 synthetic training samples (2h, CPU-only)
cd data-generation
python synthetic_driving_simulator.py --output-dir ../datasets --samples 35000

# Output:
#   datasets/train.csv (28,000 samples, 80%)
#   datasets/val.csv (3,500 samples, 10%)
#   datasets/test.csv (3,501 samples, 10%)

# Validate dataset quality
python tests/data_validator.py --input datasets/train.csv
```

### Performance Targets

| Metric | Target | Test |
|--------|--------|------|
| AI Inference (P95) | <50ms | `benchmark_inference.py` |
| Model Size | <14MB total | `pytest tests/test_*.py -k size` |
| Test Coverage | >80% | `pytest --cov-report=html` |
| CAN→Android Latency | <100ms | `e2e_test.py` |
| Power Consumption | <2W avg | Snapdragon Profiler |

### CAN Message Protocol

**UART Packet**: 83 bytes total
```
[START(0xAA)] [TIMESTAMP(8)] [VEHICLE_DATA(72)] [CRC16(2)] [END(0x55)]
```

**Key OBD-II PIDs**:
- `0x0C`: Engine RPM = ((A×256)+B)/4
- `0x0D`: Vehicle Speed = A km/h
- `0x11`: Throttle = A×100/255 %
- `0x2F`: Fuel Level = A×100/255 %
- `0x05`: Coolant Temp = A-40 °C

### Build Commands (Local Environment Only)

```bash
# ❌ NOT available in web environment
# These require local setup:

# Android builds
cd android-dtg && ./gradlew assembleDebug

# STM32 compilation
cd stm32-firmware && make -j$(nproc)

# Model training (requires GPU)
python ai-models/training/train_tcn.py --epochs 100

# CARLA data generation (requires GPU)
python data-generation/carla-scenarios/generate_driving_data.py
```

---

## 🎓 Design Patterns & Best Practices

### Thread Safety (Android)
```kotlin
// Use @Synchronized for shared state
@Synchronized
fun updateCANData(data: CANData) {
    canDataBuffer.offer(data)
}

// Coroutines with proper scope
serviceScope.launch(Dispatchers.IO) {
    // Background work
}
```

### Memory Management (JNI)
```cpp
// Always release JNI references
env->DeleteLocalRef(byteArray);
env->ReleaseByteArrayElements(array, elements, 0);
```

### CAN Message Validation
```python
def validate_can_data(data: CANData) -> bool:
    """Validate CAN data ranges before processing"""
    return (
        0 <= data.vehicle_speed <= 255 and
        0 <= data.engine_rpm <= 16383 and
        0 <= data.throttle_position <= 100 and
        -40 <= data.coolant_temp <= 215
    )
```

---

## 🔒 Security Checklist

Before committing code that handles:

**Sensitive Data**:
- [ ] No hardcoded API keys (use Android Keystore)
- [ ] TLS certificate pinning for MQTT
- [ ] Input validation for all external data

**CAN Bus**:
- [ ] CRC validation for all messages
- [ ] Message authentication for critical commands
- [ ] Rate limiting for CAN transmissions

**OTA Updates**:
- [ ] Signature verification before applying
- [ ] Rollback mechanism on failure
- [ ] Secure channel (HTTPS/MQTT-TLS)

---

## 📖 Additional Resources

**Phase-Specific Guides**:
- `docs/GPU_REQUIRED_TASKS.md` - Phase 2 local tasks (CARLA, training, quantization)
- `docs/PHASE3_TESTING.md` - Comprehensive testing strategy
- `docs/RECURSIVE_WORKFLOW.md` - Detailed workflow methodology
- `docs/PHASE2_IMPLEMENTATION.md` - Phase 2 completion summary

**External Documentation**:
- SNPE: https://developer.qualcomm.com/docs/snpe/
- CARLA: https://carla.readthedocs.io/
- Vosk: https://alphacephei.com/vosk/
- Eclipse Paho: https://github.com/eclipse-paho/paho.mqtt.android

**Custom Skills** (Workflow Automation):
- `./.claude/skills/run-tests/` - Full test suite execution
- `./.claude/skills/code-review/` - Automated quality checks
- `./.claude/skills/optimize-performance/` - Performance benchmarking

---

## ✅ Summary Checklist

Every time you work on this codebase:

1. [ ] Check phase requirements (GPU_REQUIRED_TASKS.md / PHASE3_TESTING.md)
2. [ ] Verify task is web-compatible (no GPU/build/hardware required)
3. [ ] Write test FIRST (🔴 RED)
4. [ ] Implement minimal code (🟢 GREEN)
5. [ ] Verify no regressions (full test suite)
6. [ ] Refactor if needed (🔵 REFACTOR, separate commit)
7. [ ] Never mix structural and behavioral changes
8. [ ] Use semantic commit messages
9. [ ] Push to feature branch
10. [ ] Update todos, repeat

**Remember**: Test-driven, incremental, separated concerns, always improving.
