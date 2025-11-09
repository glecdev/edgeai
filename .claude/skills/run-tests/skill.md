# Run Tests Skill

## Metadata
- **Name**: run-tests
- **Description**: 전체 테스트 스위트 실행 (AI 모델, Android, STM32)
- **Phase**: Phase 6
- **Dependencies**: pytest, gradle, make
- **Estimated Time**: 5-30 minutes

## What This Skill Does

### 1. AI Model Tests
- Unit tests (pytest)
- Model accuracy validation
- Performance benchmarks

### 2. Android Tests
- Unit tests (JUnit)
- Instrumentation tests (Espresso)
- Integration tests

### 3. STM32 Tests
- Hardware-in-the-loop tests (if available)
- Unit tests (simulation)

### 4. Integration Tests
- End-to-End CAN → STM32 → UART → Android
- MQTT communication tests
- BLE connection tests

## Usage

```bash
# Run all tests
./.claude/skills/run-tests/run.sh all

# Run specific test suite
./.claude/skills/run-tests/run.sh ai
./.claude/skills/run-tests/run.sh android
./.claude/skills/run-tests/run.sh stm32
```
