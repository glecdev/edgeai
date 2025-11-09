# Android DTG Application

## Overview

The DTG (Digital Tachograph) Android application runs on Qualcomm Snapdragon hardware to perform:
- Real-time vehicle data collection via UART from STM32
- AI inference using SNPE (Snapdragon Neural Processing Engine)
- Fleet AI platform communication via MQTT
- BLE communication with driver smartphone app

## System Requirements

- **Android Version**: 10.0+ (API 29+)
- **Hardware**: Qualcomm Snapdragon 845/865+ with DSP/HTP support
- **Permissions**: FOREGROUND_SERVICE, BOOT_COMPLETED, BLUETOOTH, INTERNET
- **Installation**: System app (requires OEM signature or root)

## Architecture

```
DTGForegroundService (START_STICKY)
    ├── CANReceiverJNI (Native C++ via JNI)
    │   └── UART Reader (921600 baud, 1Hz polling)
    ├── SNPEInferenceEngine (DSP INT8 inference)
    │   ├── TCN Model (fuel prediction)
    │   ├── LSTM-AE Model (anomaly detection)
    │   └── LightGBM Model (behavior classification)
    ├── MQTTClientService (TLS connection)
    │   └── Message Buffer (offline queuing)
    └── BLEPeripheralService
        └── GATT Server (vehicle data broadcast)
```

## Directory Structure

```
android-dtg/
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/glec/dtg/
│       │   ├── MainActivity.kt
│       │   ├── DTGForegroundService.kt
│       │   ├── BootReceiver.kt
│       │   └── snpe/SNPEEngine.kt
│       ├── cpp/
│       │   ├── uart_reader.cpp
│       │   └── CMakeLists.txt
│       └── assets/
│           ├── tcn_fuel_int8.dlc
│           ├── lstm_ae_int8.dlc
│           └── lightgbm.txt
├── build.gradle.kts
└── settings.gradle.kts
```

## Build & Install

```bash
# Debug build
./.claude/skills/android-build/run.sh dtg

# Release build with install
./.claude/skills/android-build/run.sh dtg --release --install

# View logs
adb logcat -s DTGService:V AIInference:V
```

## Performance Targets

- **AI Inference**: < 50ms (parallel execution)
- **Power Consumption**: < 2W (DSP INT8 mode)
- **Memory Usage**: < 500MB peak
- **Data Collection**: 1Hz (60 samples per inference cycle)

## Next Steps

1. Implement Kotlin application code
2. Write JNI bridge for UART communication
3. Integrate SNPE runtime
4. Configure Gradle build system
5. Create unit tests and instrumentation tests
