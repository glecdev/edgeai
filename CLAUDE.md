# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLEC DTG Edge AI SDK - An edge AI system for commercial vehicle telematics running on STM32 MCU + Qualcomm Snapdragon Android hardware. The system performs real-time vehicle data collection via CAN bus and provides AI-powered driving efficiency, safety, and carbon emission analysis.

### Hardware Platform
- **STM32**: CAN bus interface, sensor management, real-time operations (<1ms response)
- **Qualcomm Snapdragon**: Android OS, AI inference (DSP/HTP acceleration), application layer
- **Communication**: UART 921600 baud (STM32 ↔ Snapdragon), CAN bus (vehicle), BLE (driver app)

### Core Requirements
- **Model Size**: < 100MB
- **Inference Latency**: < 50ms
- **Power Consumption**: < 2W
- **Accuracy**: > 85%
- **Data Collection**: 1Hz from CAN bus
- **AI Inference**: Every 60 seconds

## Repository Structure

```
edgeai/
├── ai-models/              # Edge AI model development
│   ├── training/           # Model training scripts (PyTorch/TensorFlow)
│   ├── optimization/       # Quantization, pruning, QAT scripts
│   ├── conversion/         # ONNX → TFLite/SNPE DLC conversion
│   └── simulation/         # CARLA simulator integration
├── stm32-firmware/         # STM32 CAN bridge firmware
│   ├── Core/               # HAL drivers, main loop
│   ├── Drivers/            # CAN, UART, sensor interfaces
│   └── Middlewares/        # Protocol buffers, CRC validation
├── android-dtg/            # DTG device Android application
│   ├── app/src/main/
│   │   ├── java/           # Kotlin/Java application code
│   │   ├── cpp/            # JNI bridge for UART/SNPE
│   │   └── assets/         # AI models (.dlc, .tflite)
│   └── snpe-integration/   # SNPE runtime wrapper
├── android-driver/         # Driver smartphone application
│   ├── app/src/main/
│   │   ├── java/           # BLE, external APIs, UI
│   │   └── res/            # Layouts, resources
│   └── voice-module/       # Vosk STT, TTS, Porcupine
├── fleet-integration/      # Fleet AI platform connectivity
│   ├── mqtt-client/        # MQTT over TLS implementation
│   └── protocol/           # JSON schemas, message formats
├── data-generation/        # Synthetic data generation
│   ├── carla-scenarios/    # CARLA simulator scripts
│   ├── can-simulation/     # CANdevStudio configs
│   └── augmentation/       # Time-series augmentation (tsaug)
└── docs/                   # Architecture, API documentation
```

---

## 🔄 Recursive Improvement Workflow System

### Core Philosophy: Plan → Implement → Test → Review → Improve → Document → Commit

This project follows a **world-class recursive improvement workflow** designed for continuous quality enhancement and knowledge accumulation.

**Full Documentation**: See [`docs/RECURSIVE_WORKFLOW.md`](docs/RECURSIVE_WORKFLOW.md) for complete details.

### Quick Workflow Guide

#### 7-Phase Development Cycle

```
1️⃣ PLAN      → Design architecture, create todos, Memory MCP
2️⃣ IMPLEMENT → TDD, Skills automation, incremental commits
3️⃣ TEST      → Unit (>80%), Integration, Performance benchmarks
4️⃣ REVIEW    → Code quality, security, architecture (use code-review skill)
5️⃣ IMPROVE   → Optimize performance, refactor, extract patterns
6️⃣ DOCUMENT  → API docs, diagrams, CLAUDE.md sync
7️⃣ COMMIT    → Semantic commits, tags, changelog, CI/CD
```

#### Custom Skills for Workflow Automation

| Skill | Purpose | Usage |
|-------|---------|-------|
| **code-review** | Phase 4: Automated quality checks | `./.claude/skills/code-review/run.sh --target ai-models/` |
| **optimize-performance** | Phase 5: Performance benchmarking | `./.claude/skills/optimize-performance/run.sh --model tcn` |
| **update-docs** | Phase 6: Documentation sync | `./.claude/skills/update-docs/run.sh --all` |
| **run-tests** | Phase 3: Full test suite | `./.claude/skills/run-tests/run.sh all` |

See [`.claude/skills/README.md`](.claude/skills/README.md) for all 9 available skills.

### Recursive Learning Loop

Each development cycle improves upon the previous:

```
Cycle 1: Basic Implementation (70% quality)
   ↓ Learn architecture patterns, identify pain points
Cycle 2: Refined Implementation (85% quality)
   ↓ Learn performance bottlenecks, optimization techniques
Cycle 3: Optimized Implementation (95% quality)
   ↓ Learn edge cases, best configurations, reusable patterns
```

**Knowledge Storage**: Use Memory MCP to save:
- Design decisions
- Experiment results (MLflow)
- Best configurations
- Solution patterns
- Known issues

### Quality Gates

Every phase has quality gates to ensure excellence:

**Phase 3 (Test)**:
- [ ] All tests passing
- [ ] Coverage >80%
- [ ] Performance targets met (<50ms, <2W, >85%)

**Phase 4 (Review)**:
- [ ] Pylint score >8.0
- [ ] No critical security issues
- [ ] Architecture consistency

**Phase 7 (Commit)**:
- [ ] Semantic commit message
- [ ] CI/CD pipeline success
- [ ] Documentation updated

### Example Workflow Execution

```bash
# Phase 1: PLAN
# - Analyze task: "Add TCN quantization"
# - Design approach: Post-Training Quantization (PTQ)
# - Create todos and save design to Memory MCP

# Phase 2: IMPLEMENT (TDD)
# Write test first
cat > tests/test_quantization.py << 'EOF'
def test_quantized_model_size():
    model = quantize_int8(load_model("tcn.pth"))
    size_mb = get_model_size(model)
    assert size_mb < 5  # Target: <5MB
EOF

# Implement feature
./.claude/skills/train-model/run.sh tcn --quantize int8

# Phase 3: TEST
./.claude/skills/run-tests/run.sh ai

# Phase 4: REVIEW
./.claude/skills/code-review/run.sh --target ai-models/

# Phase 5: IMPROVE
./.claude/skills/optimize-performance/run.sh --model tcn

# Phase 6: DOCUMENT
./.claude/skills/update-docs/run.sh --all

# Phase 7: COMMIT
git add -A
git commit -m "feat(tcn): Add INT8 quantization

- Reduce model size by 75% (12MB → 3MB)
- Accuracy loss only 1.2%
- Inference 3x faster (60ms → 20ms)

Closes #42"
```

### Integration with Tools

**Memory MCP** (Knowledge Base):
```bash
# Save design decision
curl -X POST http://localhost:3000/entities \
  -d '{"name": "tcn_quantization", "observations": ["PTQ with 1000 samples optimal"]}'
```

**MLflow** (Experiment Tracking):
```python
with mlflow.start_run(run_name="tcn_v1.2.0"):
    mlflow.log_param("quantization", "int8")
    mlflow.log_metric("accuracy", 88.5)
```

**DVC** (Data Versioning):
```bash
dvc add data/training_set.csv
dvc push
```

### Continuous Improvement Metrics

Track progress over time:

| Metric | Week 1 | Week 4 | Week 8 | Trend |
|--------|--------|--------|--------|-------|
| Coverage | 75% | 82% | 89% | ↗️ |
| Complexity | 8.5 | 7.2 | 6.1 | ↗️ |
| Inference Speed | 60ms | 35ms | 20ms | ↗️ |
| Model Size | 48MB | 12MB | 3MB | ↗️ |

---

## AI Model Architecture

### Model Stack
1. **TCN (Temporal Convolutional Network)**: Fuel consumption prediction, speed pattern analysis
   - Size: 2-4MB (INT8 quantized)
   - Latency: 15-25ms
   - Accuracy: 85-90%

2. **LSTM-Autoencoder**: Anomaly detection for dangerous driving, CAN intrusion, sensor faults
   - Size: 2-3MB (INT8 quantized)
   - Latency: 25-35ms
   - F1-Score: 0.85-0.92

3. **LightGBM**: Carbon emission estimation, driving behavior classification
   - Size: 5-10MB
   - Latency: 5-15ms
   - Accuracy: 90-95%

**Total**: ~12MB models, 30ms parallel inference (60ms sequential)

### Model Development Workflow

```bash
# 1. Training (PyTorch)
cd ai-models/training
python train_tcn.py --dataset carla_synthetic --epochs 100 --batch-size 64

# 2. Quantization (Post-Training or QAT)
python quantize_model.py --model tcn_fuel.pth --method ptq --calibration-samples 500

# 3. Export to ONNX
python export_onnx.py --model tcn_fuel_quantized.pth --output tcn_fuel.onnx

# 4. Convert to SNPE DLC (Qualcomm)
snpe-onnx-to-dlc --input_network tcn_fuel.onnx --output_path tcn_fuel.dlc

# 5. Quantize DLC for DSP/HTP
snpe-dlc-quantize --input_dlc tcn_fuel.dlc \
                  --input_list calibration_data.txt \
                  --output_dlc tcn_fuel_int8.dlc

# 6. Benchmark on device
snpe-net-run --container tcn_fuel_int8.dlc --use_dsp

# 7. Alternative: ONNX Runtime with QNN EP
python run_onnx_qnn.py --model tcn_fuel.onnx --backend qnn
```

### Synthetic Data Generation

```bash
# CARLA simulator (requires GPU: RTX 2070+, 32GB RAM)
cd data-generation/carla-scenarios
python generate_driving_data.py --episodes 10000 --weather random --traffic dense

# Time-series augmentation
python augment_timeseries.py --input raw_data.csv --methods jitter,scaling,timewarping --output augmented_data.csv

# CAN message simulation
candevstudio --config can_simulation.xml
```

## STM32 Firmware Development

### Build & Flash

```bash
cd stm32-firmware

# Build with STM32CubeIDE or command-line
make clean && make -j$(nproc)

# Flash via ST-Link
st-flash write build/dtg_firmware.bin 0x8000000

# Monitor UART output (debugging)
screen /dev/ttyUSB0 921600
# or
minicom -D /dev/ttyUSB0 -b 921600
```

### CAN Message Protocol

**Message Format** (STM32 → Snapdragon via UART):
```
[START(0xAA)] [ID_H] [ID_L] [DLC] [DATA(8 bytes)] [CRC16(2)] [END(0x55)]
Total: 15 bytes per CAN frame
```

**Key OBD-II PIDs**:
- `0x0C`: Engine RPM = ((A*256)+B)/4
- `0x0D`: Vehicle Speed = A (km/h)
- `0x2F`: Fuel Level = A*100/255 (%)
- `0x11`: Throttle Position = A*100/255 (%)
- `0x05`: Coolant Temp = A-40 (°C)

**J1939 PGNs** (for commercial vehicles):
- `61444 (F004)`: Engine speed, torque
- `65265 (FEF1)`: Vehicle speed
- `65262 (FEEE)`: Fuel level

### Hardware Interfaces

- **CAN Bus**: `can0` interface, 500kbps bitrate, MCP2551 transceiver
- **UART**: 921600 baud, 8N1, TX/RX to Snapdragon GPIO
- **Sensors**: I2C (IMU, GPS), SPI (external flash), ADC (analog sensors)

## Android DTG Application

### Build & Install

```bash
cd android-dtg

# Debug build
./gradlew assembleDebug

# Install to device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# View logs (filter by tag)
adb logcat -s DTGService:V AIInference:V CANReceiver:V

# Release build with ProGuard
./gradlew assembleRelease

# System app installation (requires root or OEM signature)
adb root
adb remount
adb push app-release.apk /system/priv-app/DTG/
adb reboot
```

### Key Components

1. **BootReceiver**: Auto-start on `ACTION_BOOT_COMPLETED`
2. **DTGForegroundService**: Persistent background service (START_STICKY)
3. **CANReceiverJNI**: Native UART reader via JNI
4. **SNPEInferenceEngine**: AI model inference wrapper
5. **MQTTClientService**: Fleet AI platform communication

### JNI Bridge Architecture

```
Android Java/Kotlin
    ↓ (JNI)
Native C++ Layer
    ├── UART Reader (select/poll, 1Hz data collection)
    ├── CAN Message Parser (DBC file decoding)
    └── SNPE Runtime (AI inference on DSP/HTP)
```

**Build native code**:
```bash
cd android-dtg/app/src/main/cpp
mkdir build && cd build
cmake .. -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29
make -j$(nproc)
```

### AI Inference Integration

```kotlin
// Initialize SNPE runtime (once)
val snpeEngine = SNPEEngine.getInstance(context)
snpeEngine.loadModel("tcn_fuel_int8.dlc", SNPERuntime.DSP)

// 1-minute inference scheduler
val inferenceJob = GlobalScope.launch(Dispatchers.IO) {
    while (isServiceRunning) {
        val canData = collectLast60Seconds() // 60 samples at 1Hz

        // Parallel inference
        val fuelPrediction = async { snpeEngine.infer("tcn", canData) }
        val anomalyScore = async { snpeEngine.infer("lstm_ae", canData) }
        val behaviorClass = async { snpeEngine.infer("lightgbm", extractFeatures(canData)) }

        val results = awaitAll(fuelPrediction, anomalyScore, behaviorClass)
        processInferenceResults(results)

        delay(60_000) // 1 minute
    }
}
```

### Power Optimization

- **DVFS**: Use `PowerManager.THERMAL_STATUS_*` APIs
- **Doze Mode**: Request battery optimization exemption
- **Wake Locks**: PARTIAL_WAKE_LOCK only during inference
- **Target Power**: < 2W via DSP INT8 inference

## Driver Smartphone Application

### Build & Install

```bash
cd android-driver
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### BLE Communication

**GATT Profile** (DTG as peripheral):
```
Service UUID: 0000FFF0-0000-1000-8000-00805F9B34FB
├── Characteristic (Read/Notify): Vehicle Data
│   UUID: 0000FFF1-...
├── Characteristic (Write): Commands
│   UUID: 0000FFF2-...
└── Characteristic (Read): AI Results
    UUID: 0000FFF3-...
```

**Optimization**:
- MTU: 517 bytes (`gatt.requestMtu(517)`)
- Connection interval: 7.5-15ms (high-speed mode)
- Write Without Response for telemetry

### External Data Integration

```kotlin
// Weather API (Korea Meteorological Administration)
interface WeatherApiService {
    @GET("getUltraSrtFcst")
    suspend fun getWeather(
        @Query("serviceKey") key: String,
        @Query("base_date") date: String,
        @Query("nx") nx: Int,
        @Query("ny") ny: Int
    ): WeatherResponse
}

// Traffic API (Korea Transport Database)
interface TrafficApiService {
    @GET("api/trafficInfo")
    suspend fun getTrafficInfo(
        @Query("key") key: String,
        @Query("type") type: String = "json"
    ): TrafficResponse
}
```

### Voice Interface

**Architecture**:
```
[Wake Word] "헤이 드라이버" (Porcupine)
    ↓
[Response] "네, 말씀하세요" (Google TTS)
    ↓
[STT Active] (Vosk Korean model, 82MB)
    ↓
[Intent Parsing]
    • "배차 수락" → ACCEPT_DISPATCH
    • "배차 거부" → REJECT_DISPATCH
    • "긴급 상황" → EMERGENCY_ALERT
    ↓
[Action Execution]
    ↓
[Confirmation] "배차를 수락했습니다" (Google TTS)
```

**Setup**:
```bash
# Download Vosk Korean model
cd android-driver/app/src/main/assets
wget https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip
unzip vosk-model-small-ko-0.22.zip
mv vosk-model-small-ko-0.22 vosk-model-ko
```

**Porcupine Custom Wake Word**:
1. Visit https://console.picovoice.ai/
2. Create wake word: "헤이 드라이버"
3. Download `.ppn` file
4. Add to `android-driver/app/src/main/assets/wake_word.ppn`

## Fleet AI Platform Integration

### MQTT Configuration

```kotlin
// Eclipse Paho MQTT client
val options = MqttConnectOptions().apply {
    isAutomaticReconnect = true
    isCleanSession = false
    connectionTimeout = 30
    keepAliveInterval = 60
    socketFactory = getSSLSocketFactory(caCert) // TLS 1.2/1.3
}

val client = MqttAndroidClient(context, "ssl://mqtt.glec.ai:8883", clientId)
client.connect(options)
```

### Message Protocol

**Telemetry (Publish to `fleet/vehicles/{vehicle_id}/telemetry`)**:
```json
{
  "vehicle_id": "GLEC-DTG-001",
  "timestamp": 1699564800000,
  "location": {"lat": 37.5665, "lon": 126.9780, "speed": 80.5, "heading": 45.2},
  "diagnostics": {"engine_rpm": 2500, "fuel_level": 75.3, "battery_voltage": 12.6},
  "ai_results": {
    "fuel_efficiency": 12.5,
    "safety_score": 85,
    "carbon_emission": 120.3,
    "anomalies": ["harsh_braking"]
  }
}
```

**Commands (Subscribe to `fleet/vehicles/{vehicle_id}/commands`)**:
```json
{
  "command": "ASSIGN_DISPATCH",
  "dispatch_id": "D123456",
  "destination": {"lat": 37.5012, "lon": 127.0396},
  "cargo_weight": 5000,
  "deadline": 1699568400000
}
```

### Offline Queuing

```kotlin
// SQLite-based message buffer
class MqttMessageBuffer(context: Context) {
    fun enqueue(topic: String, payload: ByteArray, qos: Int) {
        db.insert("mqtt_queue", null, ContentValues().apply {
            put("topic", topic)
            put("payload", Base64.encodeToString(payload, Base64.DEFAULT))
            put("qos", qos)
            put("timestamp", System.currentTimeMillis())
        })
    }

    fun dequeueAll(): List<QueuedMessage> {
        // Return pending messages when connection restored
    }
}
```

**QoS Levels**:
- QoS 0: Telemetry (occasional loss acceptable)
- QoS 1: Vehicle data, AI results (recommended)
- QoS 2: Critical commands, safety alerts

### Compression

```kotlin
// Gzip compression (60-80% reduction for JSON)
fun compressPayload(json: String): ByteArray {
    val bos = ByteArrayOutputStream()
    GZIPOutputStream(bos).use { it.write(json.toByteArray(Charsets.UTF_8)) }
    return bos.toByteArray()
}
```

## Testing & Validation

### Unit Tests

```bash
# AI models (accuracy, latency)
cd ai-models
pytest tests/ -v --cov=training --cov=optimization

# Android DTG app
cd android-dtg
./gradlew testDebugUnitTest

# STM32 firmware (requires hardware-in-the-loop)
cd stm32-firmware
make test
```

### Integration Tests

```bash
# CAN bus simulation
candump can0 &
cangen can0 -I 0x123 -L 8 -D r -g 1000  # Generate random CAN frames

# End-to-end vehicle data flow
python tests/e2e_test.py --duration 300  # 5 minutes

# AI inference benchmark
cd android-dtg
./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.glec.dtg.InferenceLatencyTest
```

### Performance Validation

**Targets**:
- CAN → STM32 → UART → Android: < 100ms total latency
- AI inference: < 50ms (parallel), < 2W power
- MQTT round-trip: < 500ms (LTE)
- Voice command: < 2s total (wake word → action → confirmation)

**Profiling**:
```bash
# Snapdragon Profiler (power, CPU, GPU, DSP usage)
# Download from: https://developer.qualcomm.com/software/snapdragon-profiler

# Android Profiler (memory, CPU, network)
# Built into Android Studio
```

## CI/CD Pipeline

### GitHub Actions

```bash
# Trigger on push
.github/workflows/android-build.yml  # Build both Android apps
.github/workflows/stm32-build.yml    # Cross-compile STM32 firmware
.github/workflows/model-validation.yml  # Validate model accuracy/size
```

### Deployment

```bash
# OTA update package (Android)
cd android-dtg
./gradlew assembleRelease
python scripts/generate_ota_package.py \
    --input app/build/outputs/apk/release/app-release.apk \
    --output ota_update_v1.2.0.zip

# Upload to Fleet AI platform
curl -X POST https://api.glec.ai/ota/upload \
    -H "Authorization: Bearer $API_TOKEN" \
    -F "file=@ota_update_v1.2.0.zip" \
    -F "version=1.2.0" \
    -F "target_devices=dtg_snapdragon_865"
```

## Development Environment Setup

### Prerequisites

```bash
# Android Studio Hedgehog | 2023.1.1+
# Download from: https://developer.android.com/studio

# Android NDK 26.1.10909125
# Install via SDK Manager → SDK Tools → NDK (side by side)

# Python 3.9 or 3.10
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Qualcomm SNPE SDK
# Download from: https://softwarecenter.qualcomm.com (requires account)
# Extract to: ~/snpe-sdk/

# STM32CubeIDE (for firmware)
# Download from: https://www.st.com/en/development-tools/stm32cubeide.html

# ARM GCC Toolchain (cross-compilation)
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

### Environment Variables

```bash
export SNPE_ROOT=~/snpe-sdk
export PATH=$SNPE_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/aarch64-android:$LD_LIBRARY_PATH
export ANDROID_HOME=~/Android/Sdk
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/26.1.10909125
```

## Critical Design Patterns

### Thread Safety (Android Services)
- Use `@Synchronized` for shared CAN data buffers
- Executor services for background tasks: `Executors.newScheduledThreadPool()`
- Coroutines with proper scope: `serviceScope.launch(Dispatchers.IO)`

### Memory Management (JNI)
- Always release native resources: `DeleteLocalRef()`, `ReleaseByteArrayElements()`
- Use `weak_ptr` for circular references in C++
- Monitor native heap: `adb shell dumpsys meminfo <package>`

### CAN Message Parsing
- Use DBC file parsers: `cantools` (Python) or `dbcppp` (C++)
- Validate CRC checksums before processing
- Handle big-endian/little-endian conversions

### Model Versioning
- Semantic versioning: `tcn_fuel_v1.2.3_int8.dlc`
- DVC for dataset tracking: `dvc add data/training_set.csv`
- MLflow for experiment tracking: `mlflow.log_model(model, "tcn_fuel")`

## Security Considerations

- **TLS Certificates**: Pin CA certificate for MQTT (prevent MITM)
- **API Keys**: Store in Android Keystore, never hardcode
- **CAN Bus**: Implement message authentication (HMAC) for safety-critical commands
- **OTA Updates**: Verify signature before applying (Android Verified Boot)

## Known Hardware Limitations

- **Snapdragon 865 DSP**: Does not support all ONNX ops (e.g., Squeeze-Excitation, Swish activation)
  - **Workaround**: Use RELU6, avoid unsupported layers, test with `snpe-dlc-info`
- **STM32 UART Buffer**: Limited to 256 bytes
  - **Workaround**: Implement circular buffer, send in chunks
- **Android Doze Mode**: May kill service on some OEMs (Xiaomi, Huawei)
  - **Workaround**: Request battery optimization exemption, use system app installation

## Support Resources

- SNPE Documentation: https://developer.qualcomm.com/docs/snpe/
- CARLA Documentation: https://carla.readthedocs.io/
- Vosk API: https://alphacephei.com/vosk/
- Eclipse Paho MQTT: https://github.com/eclipse-paho/paho.mqtt.android
- STM32 CAN Examples: https://github.com/timsonater/stm32-CAN-bus-example-HAL-API
