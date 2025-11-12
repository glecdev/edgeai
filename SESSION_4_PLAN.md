# Session 4: Anomaly Injection Implementation

**Date**: 2025-11-12
**Focus**: Anomaly Injector 구현 (NumPy 기반, PyTorch 설치 전)
**Status**: Planning

---

## 🎯 Session Goals

CLAUDE.MD 원칙 준수:
- ✅ **ROOT CAUSE RESOLUTION**: 물리 기반 anomaly (단순 노이즈 아님)
- ✅ **TDD Red-Green-Refactor**: RED → GREEN → REFACTOR
- ✅ **Production-Grade**: 8개 anomaly types 완전 구현

**Session 3 Complete** ✅:
- Synthetic Vehicle Simulator (NumPy-based, 8/8 tests passing)
- TCN Architecture Document (350 lines)
- LSTM-AE Architecture Document (450 lines)
- **NEW**: Anomaly Injection Design Document (220 lines)

---

## 📋 Phase-Based Plan

### Phase 1: Anomaly Injector Core Implementation 🔄
**Status**: 🔴 RED → 🟢 GREEN → 🔵 REFACTOR

**Target**: NumPy 기반 anomaly injection (PyTorch 불필요)

#### 1.1 Unit Tests 작성 (TDD RED) ⏳
**File**: `edgeai-repo/ai-models/tests/test_anomaly_injector.py` (~300 lines)

**Test Categories**:
1. **TestAnomalyConfig** (2 tests)
   - AnomalyConfig 생성 검증
   - Parameter validation (duration > 0, probability 0-1)

2. **TestAnomalyInjector** (8 tests, 1 per anomaly type)
   - `test_overheating_injection`
   - `test_overrevving_injection`
   - `test_harsh_braking_injection`
   - `test_aggressive_acceleration_injection`
   - `test_erratic_driving_injection`
   - `test_fuel_leak_injection`
   - `test_excessive_idling_injection`
   - `test_gps_jump_injection`

3. **TestTemporalPhases** (3 tests)
   - `test_onset_phase_progression` (0% → 100%)
   - `test_sustain_phase_stability` (maintains peak)
   - `test_recovery_phase_return` (100% → 0%)

4. **TestMultiFeatureCorrelations** (3 tests)
   - `test_overheating_rpm_correlation` (temp ↑ → RPM ↑)
   - `test_harsh_braking_acceleration_correlation` (brake ↑ → accel ↓)
   - `test_fuel_leak_consumption_independence` (fuel ↓ without engine load)

**Expected Results**:
- 16 tests total
- All FAILING initially (🔴 RED phase)
- Test execution time: < 1.0 second (NumPy fast)

#### 1.2 AnomalyInjector Implementation (TDD GREEN) ⏳
**File**: `edgeai-repo/ai-models/utils/anomaly_injector.py` (~350 lines)

**Classes**:
```python
@dataclass
class AnomalyConfig:
    """Configuration for a specific anomaly type"""
    anomaly_type: str
    trigger_probability: float
    onset_duration: float      # seconds
    sustain_duration: float    # seconds
    recovery_duration: float   # seconds
    parameters: dict           # Anomaly-specific parameters

class AnomalyInjector:
    """
    Injects realistic anomalies into synthetic vehicle data

    3-phase temporal model:
    - Onset: Gradual development (15-120 seconds)
    - Sustain: Peak intensity (20-600 seconds)
    - Recovery: Return to normal (3-60 seconds)
    """

    def __init__(self, config: AnomalyConfig, total_steps: int):
        """Initialize with anomaly configuration"""

    def get_phase(self, step: int) -> Tuple[str, float]:
        """Returns: phase ('onset'|'sustain'|'recovery'), progress (0.0-1.0)"""

    def apply_overheating(self, step: int, state: VehicleState) -> VehicleState:
        """Apply overheating anomaly modifications"""

    def apply_overrevving(self, step: int, state: VehicleState) -> VehicleState:
        """Apply over-revving anomaly modifications"""

    # ... 6 more apply_* methods for remaining anomaly types

    def apply(self, step: int, state: VehicleState) -> VehicleState:
        """Main entry point: apply anomaly to vehicle state"""
```

**Implementation Details**:
- 8 `apply_*` methods (one per anomaly type)
- NumPy-based temporal interpolation (linear, sigmoid, exponential)
- Multi-feature correlations (e.g., temp ↑ → RPM ↑ 15%)
- Physical constraints enforcement (no impossible states)

**Expected Test Results**:
- 16/16 tests passing (🟢 GREEN phase)
- Code coverage: > 95% (all anomaly types tested)

#### 1.3 Refactoring & Optimization (TDD REFACTOR) ⏳
- Extract common temporal progression logic
- Add helper functions for interpolation
- Improve code readability (comments, docstrings)
- Performance optimization (vectorized NumPy operations)

---

### Phase 2: Synthetic Simulator Integration ⏳
**Status**: Pending Phase 1 completion

#### 2.1 Extend synthetic_simulator.py ⏳
**File**: `edgeai-repo/ai-models/utils/synthetic_simulator.py` (204 → ~280 lines)

**Changes**:
```python
def simulate_pattern(
    self, pattern: str, duration_minutes: float,
    start_fuel_level: float = 80.0,
    inject_anomaly: bool = False,
    anomaly_type: Optional[str] = None,  # NEW
    anomaly_start_time: Optional[float] = None  # NEW (default: random)
) -> List[VehicleState]:
    """
    Simulate a driving pattern with optional anomaly injection

    Args:
        anomaly_type: Specific anomaly to inject (if None, random selection)
            Options: "overheating", "overrevving", "harsh_braking",
                     "aggressive_accel", "erratic_driving", "fuel_leak",
                     "excessive_idling", "gps_jump"
        anomaly_start_time: When to start anomaly (minutes, default: random)
    """
    # ... existing code ...

    # NEW: Initialize anomaly injector
    anomaly_injector = None
    if inject_anomaly:
        if anomaly_type is None:
            # Random selection with realistic probabilities
            anomaly_type = np.random.choice(
                ['overheating', 'overrevving', 'harsh_braking',
                 'aggressive_accel', 'erratic_driving', 'fuel_leak',
                 'excessive_idling', 'gps_jump'],
                p=[0.02, 0.015, 0.03, 0.025, 0.02, 0.01, 0.01, 0.015]
            )

        # Create anomaly config based on type
        config = self._create_anomaly_config(anomaly_type, total_steps)
        anomaly_injector = AnomalyInjector(config, total_steps)

    # Main simulation loop
    for step in range(total_steps):
        # ... existing physics simulation ...

        # NEW: Apply anomaly if active
        if anomaly_injector is not None:
            state = anomaly_injector.apply(step, state)

        states.append(state)

    return states
```

**New Methods**:
```python
def _create_anomaly_config(self, anomaly_type: str, total_steps: int) -> AnomalyConfig:
    """Factory method for creating anomaly configurations"""
    # Returns AnomalyConfig with type-specific parameters
```

#### 2.2 Update generate_dataset() Function ⏳
**File**: `edgeai-repo/ai-models/utils/synthetic_simulator.py`

**Changes**:
```python
def generate_dataset(
    num_samples: int,
    duration_minutes: float = 5.0,
    patterns: Optional[List[str]] = None,
    sampling_rate_hz: float = 1.0,
    noise_std: float = 0.05,
    anomaly_ratio: float = 0.0,  # NEW: fraction with anomalies
    anomaly_types: Optional[List[str]] = None  # NEW: specific types only
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic vehicle dataset with optional anomalies

    Args:
        anomaly_ratio: Fraction of samples with anomalies (0.0-1.0)
            - 0.0: All normal (for LSTM-AE training)
            - 0.1: 10% anomalies (for validation)
            - 1.0: All anomalies (for testing)
        anomaly_types: Specific anomaly types to inject (if None, use all 8)

    Returns:
        X: (num_samples, 300, 10) - sensor data
        y_fuel: (num_samples,) - fuel consumption %
        y_anomaly: (num_samples,) - anomaly labels (0=normal, 1-8=anomaly type)
    """
```

---

### Phase 3: Integration Tests ⏳
**Status**: Pending Phase 2 completion

#### 3.1 Integration Test Suite ⏳
**File**: `edgeai-repo/ai-models/tests/test_integration_anomaly.py` (~200 lines)

**Test Categories**:
1. **TestDatasetGeneration** (3 tests)
   - `test_normal_dataset_no_anomalies` (anomaly_ratio=0.0)
   - `test_mixed_dataset_10pct_anomalies` (anomaly_ratio=0.1)
   - `test_full_anomaly_dataset` (anomaly_ratio=1.0)

2. **TestAnomalyDistribution** (2 tests)
   - `test_anomaly_type_distribution` (all 8 types present)
   - `test_anomaly_timing_randomness` (not all at same timestep)

3. **TestLSTMAETrainingData** (2 tests)
   - `test_training_data_purity` (verify no anomalies in training set)
   - `test_validation_data_balance` (normal + anomaly mix)

**Expected Results**:
- 7 integration tests passing
- Dataset shapes validated: (N, 300, 10)
- Anomaly labels validated: 0-8 range

#### 3.2 End-to-End Validation ⏳
**Validation Script**: `edgeai-repo/ai-models/scripts/validate_anomaly_data.py`

**Checks**:
1. Generate 1000-sample dataset with 10% anomalies
2. Verify physical constraints (no impossible states)
3. Check multi-feature correlations (e.g., overheating → RPM ↑)
4. Validate temporal profiles (onset → sustain → recovery)
5. Visualize sample anomalies (matplotlib plots)

---

### Phase 4: Documentation & Updates ⏳
**Status**: Pending Phase 3 completion

#### 4.1 Update IMPLEMENTATION_STATUS.md ⏳
**Add Section**: Session 4 - Anomaly Injection Implementation

```markdown
### 6. Anomaly Injection System (Session 4) ✅

#### 6.1 Anomaly Injector Core
**File**: anomaly_injector.py - 350 lines
**Tests**: 16/16 PASSING
**Coverage**: > 95%

**Features**:
- 8 anomaly types with physics-based signatures
- 3-phase temporal model (onset, sustain, recovery)
- Multi-feature correlations
- NumPy-based (no PyTorch dependency)

#### 6.2 Synthetic Simulator Extensions
**File**: synthetic_simulator.py - 280 lines (204 → 280)
**New Parameters**: inject_anomaly, anomaly_type, anomaly_start_time
**Integration**: Seamless anomaly injection in simulation loop

#### 6.3 Dataset Generation
**Function**: generate_dataset() - Extended
**New Parameters**: anomaly_ratio, anomaly_types
**Use Cases**:
- Training: anomaly_ratio=0.0 (normal data only)
- Validation: anomaly_ratio=0.1 (10% anomalies)
- Testing: anomaly_ratio=1.0 (full anomaly dataset)
```

#### 4.2 Update LOCAL_AND_GPU_TASKS.md ⏳
**Add to "Environment: Local (No GPU)" Section**:
```markdown
### ✅ COMPLETED (Session 4)
- [x] Anomaly Injector Implementation (anomaly_injector.py)
- [x] Synthetic Simulator Integration (extended simulate_pattern)
- [x] Dataset Generation with Anomalies (generate_dataset extended)
- [x] Unit Tests (16/16 passing)
- [x] Integration Tests (7/7 passing)
```

#### 4.3 Create SESSION_4_SUMMARY.md ⏳
**Content**: Similar structure to SESSION_3_SUMMARY.md
- Objectives achieved
- Deliverables (files created/modified)
- Test results
- CLAUDE.MD compliance verification
- Next steps (Session 5 preview)

---

## 🚧 Current Constraints

**Environment Limitations** (CLAUDE.MD 참조):
- ❌ **No GPU**: 모델 학습 불가
- ❌ **No PyTorch**: import torch 시도하면 실패
- ❌ **No Android Build**: Gradle 빌드 로컬 환경 필요

**Workarounds** (CLAUDE.MD 준수 - 단순화 아님):
1. **Anomaly Injector**: NumPy만 사용 (물리 모델 유지)
2. **Temporal Models**: NumPy interpolation (scipy.interpolate 대신)
3. **Tests**: NumPy 기반 단위 테스트

---

## 📊 Progress Tracking

### TODO List (Session 4)
- [ ] 🔴 RED: test_anomaly_injector.py (16 tests)
- [ ] 🟢 GREEN: anomaly_injector.py 구현 (350 lines)
- [ ] 🔵 REFACTOR: Code optimization
- [ ] Extend synthetic_simulator.py (280 lines)
- [ ] Update generate_dataset() function
- [ ] 🔴 RED: test_integration_anomaly.py (7 tests)
- [ ] 🟢 GREEN: Integration tests passing
- [ ] Documentation updates (3 files)
- [ ] SESSION_4_SUMMARY.md

### Metrics (Target)
- **Tests Written**: 23 tests (16 unit + 7 integration)
- **Tests Passing**: 23/23 (100%) ✅
- **Test Execution Time**: < 2.0s (NumPy performance)
- **Code Coverage**: > 95% (all anomaly types tested)
- **Lines of Code**: ~600 lines (350 anomaly_injector + 80 simulator extension + 170 test code)

---

## 🎯 Next Session Preview (Session 5)

**After Session 4 Completion**:
1. GPU 환경 준비 가이드 작성
2. PyTorch + CUDA 설치 체크리스트
3. Miniconda 환경 구성 스크립트
4. 모델 구현 준비 (tcn.py, lstm_ae.py 스켈레톤)

**OR (if GPU available)**:
1. PyTorch 설치
2. TCN 모델 구현 (tcn.py, ~200 lines)
3. LSTM-AE 모델 구현 (lstm_ae.py, ~250 lines)
4. Training script 초안 (train_tcn.py, train_lstm_ae.py)

---

## 📁 Files to Create/Modify

### New Files (Session 4)
1. `edgeai-repo/ai-models/utils/anomaly_injector.py` - 350 lines
2. `edgeai-repo/ai-models/tests/test_anomaly_injector.py` - 300 lines
3. `edgeai-repo/ai-models/tests/test_integration_anomaly.py` - 200 lines
4. `edgeai-repo/ai-models/scripts/validate_anomaly_data.py` - 150 lines
5. `SESSION_4_SUMMARY.md` - ~300 lines

### Modified Files (Session 4)
1. `edgeai-repo/ai-models/utils/synthetic_simulator.py` (204 → 280 lines)
2. `IMPLEMENTATION_STATUS.md` (add Session 4 section)
3. `LOCAL_AND_GPU_TASKS.md` (mark anomaly injection tasks as complete)

### Total Lines (Session 4)
- **Code**: ~430 lines (350 anomaly_injector + 80 simulator extension)
- **Tests**: ~500 lines (300 unit + 200 integration)
- **Scripts**: 150 lines (validation)
- **Documentation**: ~300 lines (summary)
- **Total**: ~1,380 lines of production-grade code + tests + docs

---

## 💡 Key Design Principles (CLAUDE.MD)

### 1. ROOT CAUSE RESOLUTION ✅
**No Simplification**:
- ❌ Random noise injection (BAD)
- ✅ Physics-based anomalies (GOOD)
  - Overheating: coolant temp ↑ → RPM stress ↑ → driver throttle ↓
  - Harsh braking: brake ↑ → deceleration ↑ → speed ↓ (correlated)

### 2. TDD Red-Green-Refactor ✅
**Strict Cycle**:
1. 🔴 RED: Write tests first (16 unit + 7 integration = 23 tests)
2. 🟢 GREEN: Implement until tests pass (anomaly_injector.py)
3. 🔵 REFACTOR: Optimize code while keeping tests green

### 3. Production-Grade Quality ✅
**Standards**:
- SAE J1939 compliance (commercial vehicle anomalies)
- Realistic temporal profiles (not instantaneous)
- Multi-feature correlations (not independent features)
- Type hints + docstrings + comments
- > 95% test coverage

### 4. Environment-Aware Implementation ✅
**Not Simplification**:
- NumPy used because PyTorch unavailable (environment constraint)
- Full physics model retained (no shortcuts)
- All 8 anomaly types implemented (no reduction in scope)

---

## ✅ Session 4 Success Criteria

### Must Achieve:
- [ ] AnomalyInjector class fully implemented (350 lines)
- [ ] All 8 anomaly types working (overheating, overrevving, harsh braking, aggressive accel, erratic driving, fuel leak, excessive idling, GPS jump)
- [ ] 3-phase temporal model working (onset, sustain, recovery)
- [ ] Multi-feature correlations validated (physics-based)
- [ ] 23/23 tests passing (16 unit + 7 integration)
- [ ] > 95% code coverage
- [ ] Synthetic simulator integration complete
- [ ] generate_dataset() supports anomaly_ratio parameter
- [ ] Documentation updated (3 files)

### Quality Gates:
- [ ] No simplification (CLAUDE.MD compliance)
- [ ] TDD cycle followed (RED → GREEN → REFACTOR)
- [ ] Production-grade code (type hints, docstrings, comments)
- [ ] Physical constraints enforced (no impossible states)
- [ ] Test execution time < 2.0s (performance)

---

**Session 4 Status**: 📋 **PLANNED** (Ready to start)

**Dependencies**:
- ✅ Session 3 complete (Synthetic Simulator + Architecture Docs)
- ✅ Anomaly Injection Design Document complete (220 lines)
- ✅ Python venv with pytest, numpy, pandas
- ❌ PyTorch NOT required (NumPy-based implementation)

**Estimated Duration**: 4-6 hours
- Phase 1 (RED → GREEN → REFACTOR): 2-3 hours
- Phase 2 (Simulator Integration): 1 hour
- Phase 3 (Integration Tests): 1 hour
- Phase 4 (Documentation): 1 hour

**Ready to begin when instructed!** 🚀
