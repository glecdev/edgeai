# Phase 3: Testing & Validation

**Status**: 📋 Ready for execution (after Phase 2 local tasks)
**Prerequisites**: Phase 2 GPU tasks completed, models trained, hardware available

---

## 📊 Phase 3 Overview

Phase 3 focuses on comprehensive testing, validation, and quality assurance:
- Unit testing (target: >80% coverage)
- Integration testing
- Performance benchmarking
- Hardware-in-the-loop testing
- Field trials

**Estimated Duration**: 2-3 weeks
**Team Size**: 2-3 engineers (1 SW, 1 QA, 1 Field test)

---

## 🧪 1. Unit Testing

### 1.1 AI Models Testing

**Files**: `ai-models/tests/test_*.py`

```bash
cd ai-models
pytest tests/ -v --cov=training --cov=optimization --cov-report=html

# Expected coverage: >80%
```

**Test Coverage**:
- ✅ TCN model (already completed)
- ✅ LSTM-AE model (already completed)
- ✅ LightGBM model (already completed)
- ⏸️ SNPE inference wrapper
- ⏸️ ONNX Runtime integration
- ⏸️ Quantized model accuracy

### 1.2 Android DTG App Testing

**Files**: `android-dtg/app/src/test/` and `android-dtg/app/src/androidTest/`

```bash
cd android-dtg

# Unit tests
./gradlew testDebugUnitTest

# Instrumentation tests (requires device/emulator)
./gradlew connectedAndroidTest

# Coverage report
./gradlew jacocoTestReport
```

**Test Cases**:
- `CANData` model validation
- `CANMessageParser` OBD-II/J1939 parsing
- `DTGForegroundService` lifecycle
- UART protocol encoding/decoding
- MQTT client connection & publishing
- AI inference scheduling
- Safety score calculation
- Anomaly detection logic

**Target Coverage**: >75%

### 1.3 Android Driver App Testing

```bash
cd android-driver

./gradlew testDebugUnitTest
./gradlew connectedAndroidTest
```

**Test Cases**:
- BLE connection management
- Voice assistant wake word detection
- Voice command intent parsing
- External API integration (weather, traffic)
- UI state management
- Data parsing from BLE

**Target Coverage**: >70%

### 1.4 STM32 Firmware Testing

**Challenges**: Embedded testing requires hardware or simulator

**Approaches**:
1. **Unit Testing** (Host PC):
   ```bash
   cd stm32-firmware
   mkdir build && cd build
   cmake -DUNIT_TEST=ON ..
   make
   ./run_tests
   ```

2. **Hardware-in-the-Loop** (HIL):
   - Connect STM32 to CAN simulator
   - Use CANdevStudio or PCAN-View
   - Verify CAN message parsing
   - Check UART output

**Test Cases**:
- CAN message reception & filtering
- OBD-II PID parsing
- UART packet construction
- CRC-16 calculation
- Timer-based 1Hz data collection
- IMU data reading
- GPS NMEA parsing

---

## 🔗 2. Integration Testing

### 2.1 End-to-End Data Flow

**Test**: CAN Bus → STM32 → UART → Android → MQTT → Fleet Platform

**Tool**: `tests/e2e_test.py`

```bash
python tests/e2e_test.py --duration 300 --verbose

# Expected results:
# - Packets received: ~300/300 (1Hz rate)
# - Valid packets: >99%
# - MQTT messages: >5 (every 60s)
```

**Validation**:
- ✅ 1Hz data collection rate maintained
- ✅ Packet CRC validation pass rate >99%
- ✅ UART communication stable
- ✅ AI inference runs every 60 seconds
- ✅ MQTT messages delivered successfully

### 2.2 BLE Communication

**Test**: Android DTG → BLE → Android Driver

**Setup**:
1. Install both apps on separate devices
2. Enable BLE on DTG device
3. Connect from Driver app
4. Verify data streaming

**Validation**:
- BLE connection stable (>5 minutes)
- Vehicle data updates in real-time
- AI results displayed correctly
- MTU negotiation successful (517 bytes)
- Latency < 100ms

### 2.3 MQTT Integration

**Test**: Android DTG → MQTT Broker → Fleet Platform

**Setup**:
```bash
# Start local MQTT broker for testing
docker run -d -p 1883:1883 -p 8883:8883 eclipse-mosquitto

# Subscribe to telemetry
mosquitto_sub -h localhost -t "fleet/vehicles/+/telemetry" -v
```

**Validation**:
- TLS connection established
- Telemetry published every 60 seconds
- Offline messages queued and delivered
- Gzip compression working
- QoS 1 delivery confirmed

### 2.4 Voice Assistant

**Test**: Wake word → STT → Intent → Action → TTS

**Test Cases**:
1. Wake word detection accuracy
2. Korean STT accuracy
3. Intent parsing success rate
4. TTS responsiveness

**Procedure**:
```
Test 1: "헤이 드라이버" → "배차 수락"
Expected: Dispatch accepted, confirmation spoken

Test 2: "헤이 드라이버" → "긴급 상황"
Expected: Emergency alert sent, confirmation spoken

Test 3: "헤이 드라이버" → "안전 점수"
Expected: Safety score announced via TTS
```

**Target**: >90% intent recognition accuracy

---

## ⚡ 3. Performance Benchmarking

### 3.1 AI Inference Performance

**Tool**: `tests/benchmark_inference.py`

```bash
# Benchmark TCN
python tests/benchmark_inference.py --model tcn --iterations 1000

# Benchmark LSTM-AE
python tests/benchmark_inference.py --model lstm_ae --iterations 1000

# Benchmark LightGBM
python tests/benchmark_inference.py --model lightgbm --iterations 1000
```

**Targets**:
| Model | Latency (P95) | Model Size | Device |
|-------|---------------|------------|--------|
| TCN | <25ms | <2MB | DSP INT8 |
| LSTM-AE | <35ms | <2MB | DSP INT8 |
| LightGBM | <15ms | <10MB | CPU |
| **Total** | **<50ms** | **<14MB** | **Parallel** |

### 3.2 Power Consumption

**Tool**: Qualcomm Snapdragon Profiler

**Test Setup**:
1. Install Snapdragon Profiler on host PC
2. Connect Android device via USB
3. Run DTG service for 30 minutes
4. Measure power consumption

**Targets**:
- Idle: <500mW
- Data collection (1Hz): <1W
- AI inference (every 60s): <3W peak, <2W average
- **Overall average**: <2W

**Measurement**:
```bash
# Via adb
adb shell dumpsys batterystats --reset
# Run app for 30 minutes
adb shell dumpsys batterystats

# Or use Snapdragon Profiler GUI
```

### 3.3 Memory Usage

**Monitoring**:
```bash
# Android memory profiler
adb shell dumpsys meminfo com.glec.dtg

# STM32 memory analysis
arm-none-eabi-size stm32-firmware.elf
```

**Targets**:
- Android app RAM: <200MB
- STM32 RAM usage: <64KB (of 128KB available)
- STM32 Flash usage: <512KB (of 1MB available)

### 3.4 Network Performance

**MQTT Latency Test**:
```bash
# Publish test message
mosquitto_pub -h mqtt.glec.ai -p 8883 -t "test" -m "ping" --cafile ca.crt

# Measure round-trip time
# Expected: <500ms over LTE
```

**Throughput Test**:
- Telemetry size: ~500 bytes (uncompressed)
- Telemetry size: ~200 bytes (gzip compressed)
- Frequency: Every 60 seconds
- **Bandwidth**: <1 KB/minute (<10 KB/hour)

---

## 🔧 4. Hardware-in-the-Loop Testing

### 4.1 CAN Bus Simulation

**Tools**:
- CANdevStudio
- PCAN-View
- Vector CANoe (if available)

**Test Scenarios**:

#### Scenario 1: Normal Driving
```
Speed: 60-80 km/h
RPM: 2000-2500
Throttle: 30-50%
Duration: 5 minutes
Expected: Normal behavior classification
```

#### Scenario 2: Harsh Braking
```
Speed: 80 → 40 km/h in 2 seconds
Brake: 100%
Acceleration: -5 m/s²
Expected: Harsh braking anomaly detected
```

#### Scenario 3: Aggressive Driving
```
Speed: 0 → 100 km/h in 8 seconds
Throttle: 90-100%
RPM: 4000-5500
Acceleration: >3 m/s²
Expected: Aggressive behavior classification
```

#### Scenario 4: Eco Driving
```
Speed: 50-60 km/h (constant)
Throttle: 20-30%
RPM: 1500-2000
Smooth acceleration/deceleration
Expected: Eco driving behavior classification
```

### 4.2 IMU & GPS Integration

**Test**: Physical sensors connected to STM32

**IMU Test**:
- Stationary: accel_z ≈ 9.81 m/s², others ≈ 0
- Rotation: gyro values change accordingly
- Harsh braking: accel_x < -4 m/s²
- Harsh acceleration: accel_x > 3 m/s²

**GPS Test**:
- Valid NMEA sentences received
- Fix quality: 1 or 2 (GPS fix)
- Satellite count: >4
- Location accuracy: <10 meters

### 4.3 UART Communication Stress Test

**Test**: High-frequency data transmission

```bash
# Send continuous data for 1 hour
python tests/uart_stress_test.py --duration 3600 --rate 1

# Check for packet loss, CRC errors
```

**Validation**:
- Packet loss rate: <0.1%
- CRC errors: <0.01%
- Buffer overflows: 0
- UART errors: 0

---

## 🚗 5. Field Trials

### 5.1 Test Environment

**Vehicle**: Commercial truck or van
**Route**: Mixed (highway + city + traffic jam)
**Duration**: 2-4 hours per trial
**Trials**: 3 minimum

### 5.2 Data Collection

**Metrics**:
- Total distance: >100 km per trial
- Data samples collected: >7,200 (2 hours × 3600 s/h × 1Hz)
- AI inferences run: >120 (2 hours × 60 inferences/h)
- Anomalies detected: Log all
- System crashes: 0 (target)

### 5.3 Validation Criteria

**Functional**:
- ✅ Service runs without crashes for entire duration
- ✅ Data collection maintains 1Hz rate (>95%)
- ✅ AI inference runs every 60 seconds (>95%)
- ✅ MQTT connection maintained (or offline queue works)
- ✅ BLE connection stable if Driver app connected

**Performance**:
- ✅ Inference latency < 50ms (P95)
- ✅ Power consumption < 2W average
- ✅ No device overheating
- ✅ Battery drain < 10% per hour

**Accuracy**:
- ✅ Behavior classification accuracy >85% (manual validation)
- ✅ Anomaly detection false positive rate <5%
- ✅ Safety score correlation with driving style

### 5.4 Edge Cases

**Test**:
1. **Network Loss**: Drive through tunnel, verify offline queueing
2. **Device Reboot**: Reboot device, verify auto-start works
3. **Low Battery**: Run with <20% battery, verify low-power mode
4. **Sensor Fault**: Disconnect GPS, verify graceful degradation
5. **CAN Bus Noise**: Inject random CAN messages, verify filtering

---

## 📋 6. Test Reporting

### 6.1 Test Summary Report

**Template**:
```
GLEC DTG Edge AI - Test Summary Report
Test Date: YYYY-MM-DD
Tester: [Name]
Environment: [Lab / Field / HIL]

Test Results:
- Total Tests: XXX
- Passed: XXX
- Failed: XXX
- Skipped: XXX
- Pass Rate: XX%

Critical Issues: [List]
Known Limitations: [List]
Recommendations: [List]
```

### 6.2 Performance Report

**Tool**: `tests/generate_report.py`

```bash
python tests/generate_report.py \
    --benchmark-results results/benchmark_*.json \
    --e2e-results results/e2e_*.log \
    --field-test-data results/field_*.csv \
    --output reports/performance_report.pdf
```

**Sections**:
- Executive Summary
- Test Environment
- Functional Test Results
- Performance Benchmarks
- Accuracy Metrics
- Issues & Resolutions
- Conclusions & Recommendations

### 6.3 Coverage Report

```bash
# AI models
pytest --cov=ai-models --cov-report=html
# Open htmlcov/index.html

# Android DTG
./gradlew jacocoTestReport
# Open app/build/reports/jacoco/html/index.html
```

**Target**: >80% line coverage for critical components

---

## ✅ 7. Acceptance Criteria

### 7.1 Functional Requirements

| Requirement | Target | Status |
|-------------|--------|--------|
| Data collection rate | 1Hz ±5% | ⏸️ |
| AI inference interval | 60s ±5s | ⏸️ |
| CAN message parsing | >99% success | ⏸️ |
| UART communication | >99.9% reliability | ⏸️ |
| MQTT delivery | >99% (with retry) | ⏸️ |
| BLE stability | >95% uptime | ⏸️ |
| Voice recognition | >90% accuracy | ⏸️ |

### 7.2 Performance Requirements

| Requirement | Target | Status |
|-------------|--------|--------|
| AI inference latency | <50ms (P95) | ⏸️ |
| Model size | <14MB total | ⏸️ |
| Power consumption | <2W average | ⏸️ |
| Memory usage | <200MB RAM | ⏸️ |
| System uptime | >99.5% | ⏸️ |

### 7.3 Accuracy Requirements

| Requirement | Target | Status |
|-------------|--------|--------|
| Behavior classification | >85% accuracy | ⏸️ |
| Anomaly detection F1 | >0.85 | ⏸️ |
| Fuel prediction MAPE | <15% | ⏸️ |
| False positive rate | <5% | ⏸️ |

---

## 🔄 8. Continuous Integration

### 8.1 CI Pipeline

**GitHub Actions**: `.github/workflows/test.yml`

```yaml
name: Phase 3 Testing

on: [push, pull_request]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/ -v --cov
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  android-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up JDK 11
        uses: actions/setup-java@v3
        with:
          java-version: '11'
      - name: Run unit tests
        run: |
          cd android-dtg
          ./gradlew testDebugUnitTest
```

### 8.2 Automated Testing Schedule

- **On every commit**: Unit tests
- **Nightly**: Integration tests, benchmarks
- **Weekly**: Full test suite, field trial simulation
- **Pre-release**: Complete Phase 3 validation

---

## 📝 9. Known Issues & Limitations

**Current Limitations** (to be addressed in Phase 3):
1. SNPE inference wrapper not tested (requires SNPE SDK)
2. Android builds not generated (requires Android SDK)
3. STM32 firmware not compiled (requires ARM toolchain)
4. Field trial data not collected (requires vehicle)
5. Power consumption not measured (requires profiler)

**Workarounds**:
- Use mock data for testing
- Simulate CAN bus with CANdevStudio
- Test on emulator where possible
- Defer hardware-specific tests to later stage

---

## 🎯 10. Phase 3 Success Criteria

Phase 3 is considered complete when:

1. ✅ All unit tests passing (>80% coverage)
2. ✅ Integration tests passing (E2E, BLE, MQTT)
3. ✅ Performance benchmarks meet targets
4. ✅ Field trial completed successfully (3+ trials)
5. ✅ Critical bugs resolved (P0/P1)
6. ✅ Test reports generated
7. ✅ Acceptance criteria met
8. ✅ Documentation updated

**Estimated Completion**: 2-3 weeks after Phase 2 local tasks

---

**Last Updated**: 2025-01-09
**Status**: Ready for execution
**Next Phase**: Phase 4 - Deployment & Production
