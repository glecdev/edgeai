# GLEC DTG Edge AI SDK - Recursive Improvement Workflow
## 세계 최고 수준의 Claude Code 개발 워크플로우

---

## 🎯 핵심 철학

### 1. 재귀적 개선 (Recursive Improvement)
```
Plan → Implement → Test → Review → Improve → Document → Commit
  ↓                                                          ↓
  ←←←←←←←←←←←← Learn & Iterate ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

**원칙**:
- 모든 작업은 **Plan → Execute → Validate → Improve** 사이클로 진행
- 각 사이클에서 얻은 **학습을 다음 사이클에 적용**
- **실패는 학습의 기회** - 실패 패턴을 문서화하고 회피 전략 수립

### 2. 기술 지식 노하우 증강 (Knowledge Augmentation)
```
Code → Memory MCP → Pattern Library → Best Practices → Reuse
```

**전략**:
- **Memory MCP**: 설계 결정, 실험 결과, 최적 설정 저장
- **Pattern Library**: 재사용 가능한 코드 패턴 추출
- **Best Practices**: 프로젝트 특화 베스트 프랙티스 정립
- **Continuous Learning**: 각 작업에서 배운 내용을 체계화

### 3. 컨텍스트 선명함 유지 (Context Clarity)
```
Clear Structure + Consistent Naming + State Management + Documentation
```

**방법**:
- **명확한 폴더 구조**: 역할별 디렉토리 분리
- **일관된 네이밍**: 컨벤션 준수 (snake_case, camelCase, PascalCase)
- **상태 관리**: Todo, Git, Memory MCP로 진행 상황 추적
- **문서-코드 동기화**: 코드 변경 시 문서 자동 업데이트

---

## 🔄 7-Phase Recursive Workflow

### Phase 1️⃣: PLAN (계획)

**목표**: 작업을 명확히 이해하고 최적의 접근 방식 설계

**활동**:
1. **Task 분석**
   ```bash
   # What: 무엇을 만들 것인가?
   # Why: 왜 필요한가?
   # How: 어떻게 구현할 것인가?
   # Dependencies: 선행 작업은?
   # Risks: 예상 위험은?
   ```

2. **아키텍처 설계**
   - 컴포넌트 다이어그램 작성
   - 데이터 플로우 정의
   - API 인터페이스 설계

3. **Memory MCP에 저장**
   ```json
   {
     "entity": "design_decision",
     "name": "TCN_architecture",
     "observations": [
       "1D Conv with dilation for temporal modeling",
       "Residual connections for gradient flow",
       "3 layers, 64 filters each"
     ]
   }
   ```

4. **Todo 생성**
   - 구체적이고 측정 가능한 작업으로 분해
   - 각 작업에 예상 시간 및 우선순위 부여

**출력물**:
- Architecture diagram (draw.io, PlantUML)
- API specification (OpenAPI)
- Todo list (TodoWrite)
- Design decisions (Memory MCP)

**Quality Gate**:
- [ ] 모든 요구사항이 설계에 반영되었는가?
- [ ] 아키텍처가 확장 가능한가?
- [ ] 의존성이 명확한가?

---

### Phase 2️⃣: IMPLEMENT (구현)

**목표**: 고품질 코드 작성 with TDD

**활동**:
1. **Test-First Development**
   ```python
   # 1. 테스트 먼저 작성 (Red)
   def test_tcn_output_shape():
       model = TCN(input_dim=10, output_dim=1)
       x = torch.randn(32, 60, 10)  # batch, seq, features
       y = model(x)
       assert y.shape == (32, 1)

   # 2. 최소 구현 (Green)
   class TCN(nn.Module):
       # ... implementation

   # 3. 리팩토링 (Refactor)
   ```

2. **Skill 활용**
   ```bash
   # 반복 작업 자동화
   ./.claude/skills/train-model/run.sh tcn --epochs 100
   ```

3. **실시간 검증**
   ```bash
   # Watch mode로 테스트 자동 실행
   pytest-watch tests/
   ```

4. **점진적 커밋**
   ```bash
   # 작은 단위로 자주 커밋
   git add -p  # Interactive staging
   git commit -m "feat(tcn): Add dilated convolution layer"
   ```

**출력물**:
- Production code
- Unit tests (>80% coverage)
- Integration tests
- Git commits (semantic)

**Quality Gate**:
- [ ] 모든 테스트가 통과하는가?
- [ ] 코드 커버리지 >80%인가?
- [ ] Linter 에러가 없는가?
- [ ] 코드가 가독성이 있는가?

---

### Phase 3️⃣: TEST (테스트)

**목표**: 다층적 테스트로 품질 보장

**활동**:
1. **Unit Test** (개별 함수/클래스)
   ```python
   # ai-models/tests/test_tcn.py
   def test_tcn_forward_pass():
       """TCN forward pass produces correct shape"""
       model = TCN(input_dim=10, output_dim=1, num_layers=3)
       x = torch.randn(32, 60, 10)
       y = model(x)
       assert y.shape == (32, 1)
       assert not torch.isnan(y).any()
   ```

2. **Integration Test** (컴포넌트 간 통신)
   ```python
   def test_can_to_uart_communication():
       """End-to-end: CAN → STM32 → UART → Android"""
       # STM32 시뮬레이터
       stm32_sim = STM32Simulator()

       # CAN 메시지 전송
       can_msg = CANMessage(id=0x123, data=[0x10, 0x20])
       stm32_sim.send_can(can_msg)

       # UART 수신 확인
       uart_data = stm32_sim.read_uart()
       assert uart_data.startswith(b'\xAA')  # START marker
   ```

3. **Performance Benchmark**
   ```python
   def test_tcn_inference_latency():
       """TCN inference completes within 25ms"""
       model = TCN(...)
       x = torch.randn(1, 60, 10)

       import time
       start = time.time()
       with torch.no_grad():
           y = model(x)
       latency = (time.time() - start) * 1000

       assert latency < 25  # Target: <25ms
   ```

4. **Coverage Analysis**
   ```bash
   pytest --cov=ai_models --cov-report=html
   open htmlcov/index.html
   ```

**출력물**:
- Test results
- Coverage report (>80%)
- Performance benchmarks
- Test documentation

**Quality Gate**:
- [ ] 모든 테스트 통과?
- [ ] Coverage >80%?
- [ ] 성능 목표 달성? (<25ms, <2W, >85%)
- [ ] Edge case 처리?

---

### Phase 4️⃣: REVIEW (검토)

**목표**: 코드 품질, 아키텍처, 보안 검증

**활동**:
1. **자동 코드 리뷰** (Skill 사용)
   ```bash
   ./.claude/skills/code-review/run.sh --target ai-models/training/
   ```

2. **정적 분석**
   ```bash
   # Python
   pylint ai_models/
   mypy ai_models/
   bandit -r ai_models/  # Security

   # Android
   ./gradlew lint
   ./gradlew detekt
   ```

3. **아키텍처 일관성 확인**
   - 설계 문서와 코드 일치 여부
   - SOLID 원칙 준수
   - DRY (Don't Repeat Yourself) 위반 검사

4. **보안 검토**
   ```bash
   # Dependency vulnerability scan
   pip-audit

   # Android
   ./gradlew dependencyCheckAnalyze
   ```

5. **문서 동기화**
   - 코드 주석 완성도
   - API 문서 최신성
   - README 정확성

**출력물**:
- Code review report
- Security audit report
- Architecture compliance report
- Documentation gaps list

**Quality Gate**:
- [ ] Linter 에러 0개?
- [ ] 보안 취약점 없음?
- [ ] 아키텍처 일관성 유지?
- [ ] 문서가 코드와 동기화?

---

### Phase 5️⃣: IMPROVE (개선)

**목표**: 코드 품질 향상 및 최적화

**활동**:
1. **성능 프로파일링**
   ```bash
   # AI 모델
   python -m cProfile -o profile.stats train_tcn.py
   snakeviz profile.stats

   # Android
   # Snapdragon Profiler 사용
   ```

2. **코드 리팩토링**
   ```python
   # Before (복잡도 높음)
   def process_data(data):
       if data is not None:
           if len(data) > 0:
               if data[0] > 0:
                   return data[0] * 2
       return None

   # After (복잡도 낮음)
   def process_data(data):
       if not data or data[0] <= 0:
           return None
       return data[0] * 2
   ```

3. **기술 부채 해결**
   - TODO 주석 처리
   - FIXME 해결
   - HACK 제거 및 정상화

4. **재사용 패턴 추출**
   ```python
   # 공통 패턴을 유틸리티로 추출
   # utils/model_utils.py
   def load_and_quantize_model(path, quantization='int8'):
       """Reusable pattern for model loading"""
       model = torch.load(path)
       if quantization == 'int8':
           model = quantize_int8(model)
       return model
   ```

5. **Memory MCP에 학습 저장**
   ```json
   {
     "entity": "optimization_result",
     "name": "tcn_quantization",
     "observations": [
       "INT8 quantization: 4x size reduction",
       "Accuracy loss: only 1.2%",
       "Inference speed: 3x faster",
       "Best config: PTQ with 500 calibration samples"
     ]
   }
   ```

**출력물**:
- Refactored code
- Performance report (before/after)
- Pattern library updates
- Memory MCP entries

**Quality Gate**:
- [ ] 성능 개선 측정 가능?
- [ ] 복잡도 감소?
- [ ] 재사용성 증가?
- [ ] 기술 부채 감소?

---

### Phase 6️⃣: DOCUMENT (문서화)

**목표**: 지식 체계화 및 공유

**활동**:
1. **코드 주석**
   ```python
   def train_tcn(config: TrainConfig) -> ModelMetrics:
       """Train Temporal Convolutional Network for fuel prediction.

       Args:
           config: Training configuration containing:
               - epochs: Number of training epochs (default: 100)
               - batch_size: Batch size (default: 64)
               - learning_rate: Learning rate (default: 0.001)

       Returns:
           ModelMetrics containing:
               - train_accuracy: Training set accuracy
               - val_accuracy: Validation set accuracy
               - model_size_mb: Model size in MB
               - inference_time_ms: Average inference time

       Raises:
           ValueError: If config is invalid
           RuntimeError: If training fails

       Example:
           >>> config = TrainConfig(epochs=100)
           >>> metrics = train_tcn(config)
           >>> print(f"Accuracy: {metrics.val_accuracy}%")
       """
   ```

2. **API 문서 생성**
   ```bash
   # Python
   pdoc --html ai_models -o docs/api

   # Android
   ./gradlew dokkaHtml
   ```

3. **아키텍처 다이어그램**
   ```plantuml
   @startuml
   component "CAN Bus" as CAN
   component "STM32" as STM32
   component "Android App" as Android
   component "AI Engine" as AI

   CAN --> STM32 : CAN messages (1Hz)
   STM32 --> Android : UART (921600 baud)
   Android --> AI : Inference (60s window)
   AI --> Android : Predictions
   @enduml
   ```

4. **Changelog 업데이트**
   ```markdown
   ## [1.2.0] - 2025-01-09

   ### Added
   - TCN model with INT8 quantization
   - LSTM-Autoencoder for anomaly detection
   - LightGBM for behavior classification

   ### Changed
   - Improved inference speed by 3x
   - Reduced model size by 75%

   ### Fixed
   - Memory leak in JNI bridge
   - CAN message parsing edge cases
   ```

5. **CLAUDE.md 업데이트**
   ```bash
   ./.claude/skills/update-docs/run.sh --target CLAUDE.md
   ```

**출력물**:
- Code comments (docstrings)
- API documentation
- Architecture diagrams
- Changelog
- Updated CLAUDE.md

**Quality Gate**:
- [ ] 모든 public API 문서화?
- [ ] 다이어그램이 최신 아키텍처 반영?
- [ ] Changelog 업데이트?
- [ ] CLAUDE.md 동기화?

---

### Phase 7️⃣: COMMIT (커밋)

**목표**: 버전 관리 및 배포 준비

**활동**:
1. **Semantic Commit**
   ```bash
   # Conventional Commits 사용
   git commit -m "feat(tcn): Add INT8 quantization support

   - Implement post-training quantization (PTQ)
   - Add calibration dataset support (500 samples)
   - Achieve 4x size reduction with 1.2% accuracy loss

   Performance:
   - Model size: 12MB → 3MB
   - Inference: 60ms → 20ms
   - Accuracy: 89.7% → 88.5%

   BREAKING CHANGE: Requires SNPE SDK 2.35.0+

   Closes #42"
   ```

2. **Git Tag (버전)**
   ```bash
   git tag -a v1.2.0 -m "Release v1.2.0: TCN quantization"
   git push origin v1.2.0
   ```

3. **Changelog 생성**
   ```bash
   # Conventional Commits → Changelog
   npx conventional-changelog -p angular -i CHANGELOG.md -s
   ```

4. **CI/CD 트리거**
   ```bash
   git push origin main
   # → GitHub Actions 자동 실행
   # → Tests, Build, Deploy
   ```

**출력물**:
- Git commits (semantic)
- Git tags (version)
- Updated CHANGELOG.md
- CI/CD pipeline execution

**Quality Gate**:
- [ ] Commit message가 컨벤션 준수?
- [ ] 모든 테스트 통과?
- [ ] CI/CD 파이프라인 성공?
- [ ] 버전 태그 생성?

---

## 🔁 재귀적 학습 루프 (Recursive Learning Loop)

### 사이클 1: 기본 구현
```
Plan → Implement → Test → Review → Improve → Document → Commit
Output: Working prototype (70% quality)
Learning: Basic architecture, pain points identified
```

### 사이클 2: 개선 구현
```
Plan (refined) → Implement (optimized) → Test (comprehensive) → ...
Output: Production-ready (85% quality)
Learning: Performance bottlenecks, optimization techniques
```

### 사이클 3: 최적화 구현
```
Plan (data-driven) → Implement (best practices) → ...
Output: Optimized solution (95% quality)
Learning: Edge cases, best configurations, reusable patterns
```

### 학습 저장 (Memory MCP)
```json
{
  "cycle": 3,
  "improvements": [
    "Batch size 64 → 128 improved throughput by 40%",
    "Learning rate 0.001 → 0.0005 stabilized training",
    "Data augmentation increased accuracy by 3%"
  ],
  "patterns": [
    "Always use learning rate scheduling",
    "Monitor validation loss for early stopping",
    "Use mixed precision for 2x speedup"
  ]
}
```

---

## 📊 품질 메트릭 (Quality Metrics)

### 코드 품질
- **Coverage**: >80% (목표: 90%)
- **Complexity**: Cyclomatic complexity <10
- **Duplication**: <3%
- **Maintainability Index**: >20

### 성능
- **AI Inference**: <50ms (목표: <30ms)
- **Power**: <2W
- **Model Size**: <100MB (목표: <20MB)
- **Accuracy**: >85% (목표: >90%)

### 프로세스
- **Cycle Time**: Plan → Commit <1 day (작은 작업)
- **Lead Time**: 요청 → 배포 <1 week
- **Deployment Frequency**: 주 1회 이상
- **Change Failure Rate**: <5%

---

## 🛠 도구 통합 (Tool Integration)

### Memory MCP
```bash
# 설계 결정 저장
curl -X POST http://localhost:3000/entities \
  -d '{"name": "tcn_architecture", "entityType": "design_decision", "observations": ["..."]}'

# 실험 결과 조회
curl http://localhost:3000/entities?entityType=experiment_result
```

### MLflow
```python
with mlflow.start_run(run_name="tcn_v1.2.0"):
    mlflow.log_param("quantization", "int8")
    mlflow.log_metric("accuracy", 88.5)
    mlflow.pytorch.log_model(model, "model")
```

### Git
```bash
# Semantic versioning
git tag -a v1.2.0 -m "TCN quantization release"

# Conventional commits
git commit -m "feat(tcn): Add quantization support"
```

### DVC
```bash
# Data versioning
dvc add data/training_set.csv
dvc push

# Reproduce experiment
dvc repro
```

---

## 🎯 성공 사례 템플릿

### Task: TCN 모델 INT8 양자화

**Phase 1: Plan**
- Target: 모델 크기 75% 감소, 정확도 손실 <2%
- Approach: Post-Training Quantization (PTQ)
- Risks: 정확도 저하, 지원되지 않는 연산자

**Phase 2: Implement**
```python
# tests/test_quantization.py
def test_quantized_model_accuracy():
    original = load_model("tcn_fp32.pth")
    quantized = quantize_int8(original)

    acc_original = evaluate(original, test_set)
    acc_quantized = evaluate(quantized, test_set)

    accuracy_loss = acc_original - acc_quantized
    assert accuracy_loss < 2.0  # Target: <2%
```

**Phase 3: Test**
- Unit: ✅ Quantization reduces size by 75%
- Integration: ✅ SNPE DLC conversion successful
- Performance: ✅ Inference 20ms (target <25ms)

**Phase 4: Review**
- Code quality: ✅ Pylint score 9.5/10
- Security: ✅ No vulnerabilities
- Architecture: ✅ Consistent with design

**Phase 5: Improve**
- Optimization: Calibration samples 500 → 1000 (accuracy +0.5%)
- Refactor: Extract `quantize_int8()` to utils

**Phase 6: Document**
```python
def quantize_int8(model: nn.Module, calibration_data: Dataset) -> nn.Module:
    """Apply INT8 post-training quantization.

    Reduces model size by ~75% with <2% accuracy loss.

    Args:
        model: PyTorch model to quantize
        calibration_data: Representative dataset (500-1000 samples)

    Returns:
        Quantized model ready for SNPE conversion
    """
```

**Phase 7: Commit**
```bash
git commit -m "feat(tcn): Add INT8 quantization support

- Implement PTQ with 1000 calibration samples
- Achieve 75% size reduction (12MB → 3MB)
- Accuracy loss: only 1.2% (89.7% → 88.5%)
- Inference speed: 3x faster (60ms → 20ms)

Closes #42"
```

**Learning** (Memory MCP):
```json
{
  "entity": "best_practice",
  "name": "quantization_workflow",
  "observations": [
    "1000 calibration samples optimal (vs 500)",
    "Always test on real device (SNPE)",
    "Monitor outlier activations during calibration",
    "PTQ sufficient for TCN (QAT not needed)"
  ]
}
```

---

## 🚀 Quick Start

### 1. 새 작업 시작
```bash
# 1. Todo 생성
echo "Implement TCN quantization" | claude-code

# 2. Memory MCP에서 관련 지식 조회
curl http://localhost:3000/entities?entityType=best_practice&name=quantization

# 3. Plan 작성
vi docs/plans/tcn_quantization.md

# 4. Implement with TDD
pytest-watch tests/test_quantization.py
```

### 2. 사이클 실행
```bash
# Implement → Test → Review
./.claude/skills/train-model/run.sh tcn --quantize int8
./.claude/skills/run-tests/run.sh ai
./.claude/skills/code-review/run.sh --target ai-models/

# Improve → Document → Commit
./.claude/skills/optimize-performance/run.sh --model tcn
./.claude/skills/update-docs/run.sh
git add -A && git commit
```

### 3. 학습 저장
```bash
# Memory MCP에 결과 저장
echo '{
  "entity": "experiment_result",
  "name": "tcn_quantization_v1",
  "observations": ["..."]
}' | http POST localhost:3000/entities
```

---

## 📈 지속적 개선 (Continuous Improvement)

### 주간 회고 (Weekly Retrospective)
```markdown
## Week 1 Retrospective

### What went well?
- TCN quantization achieved better than expected results
- Test coverage increased to 85%
- No production incidents

### What could be improved?
- Documentation lagged behind code changes
- Some edge cases not covered in tests
- Build time increased to 15 minutes

### Action items:
- [ ] Set up automatic documentation generation
- [ ] Add property-based testing for edge cases
- [ ] Optimize Docker build caching
```

### 월간 메트릭 리뷰 (Monthly Metrics Review)
```
Code Quality Trend:
  Coverage: 75% → 80% → 85% ↗️
  Complexity: 8.5 → 7.2 → 6.8 ↗️
  Tech Debt: 45 → 38 → 32 hours ↗️

Performance Trend:
  Inference: 60ms → 30ms → 20ms ↗️
  Model Size: 48MB → 12MB → 3MB ↗️
  Accuracy: 85.3% → 88.5% → 89.7% ↗️
```

### Pattern Library 성장
```
Iteration 1: 5 patterns documented
Iteration 2: 12 patterns documented (+140%)
Iteration 3: 18 patterns documented (+50%)
→ Knowledge compounding!
```

---

## 🎓 핵심 원칙 (Core Principles)

1. **Small, Frequent Iterations**
   - 큰 작업을 작은 단위로 분해
   - 매일 가시적 진전
   - 빠른 피드백 루프

2. **Test Everything**
   - 테스트 없는 코드 = 기술 부채
   - Coverage >80% 필수
   - Performance 테스트 포함

3. **Document as You Go**
   - 코드 작성 중 즉시 문서화
   - 나중으로 미루지 않기
   - 문서 = 미래의 나를 위한 투자

4. **Learn from Failures**
   - 실패를 Memory MCP에 기록
   - 회피 전략 수립
   - 팀과 공유

5. **Automate Repetition**
   - 같은 작업 2번 하면 Skill 만들기
   - CI/CD로 수동 작업 제거
   - 사람은 창의적 작업에 집중

---

이 워크플로우를 따르면:
- ✅ **60-70% 빠른 개발 속도**
- ✅ **95%+ 코드 품질**
- ✅ **지속 가능한 기술 성장**
- ✅ **명확한 컨텍스트 유지**
- ✅ **재사용 가능한 지식 축적**

🎯 **목표: 세계 최고 수준의 Edge AI SDK 개발!**
