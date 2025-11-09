# Android Driver Application

## Overview

The Driver smartphone application provides:
- BLE connection to DTG device for real-time vehicle data
- Voice command interface (Korean STT/TTS)
- External data integration (weather, traffic APIs)
- Dispatch management and navigation
- Driver safety alerts

## System Requirements

- **Android Version**: 8.0+ (API 26+)
- **Permissions**: BLUETOOTH, LOCATION, RECORD_AUDIO, INTERNET
- **Hardware**: Standard Android smartphone with BLE 4.2+

## Architecture

```
MainActivity
    ├── BLECentralService
    │   ├── GATT Client (connects to DTG)
    │   └── MTU Negotiation (517 bytes)
    ├── VoiceCommandModule
    │   ├── Porcupine (Wake word: "헤이 드라이버")
    │   ├── Vosk STT (Korean, 82MB)
    │   └── Google TTS (Korean)
    ├── ExternalDataService
    │   ├── WeatherAPI (기상청)
    │   └── TrafficAPI (교통DB)
    └── DispatchManagementService
        └── Fleet AI Platform API
```

## Voice Commands

| Command | Action | Response |
|---------|--------|----------|
| "배차 수락" | Accept dispatch | "배차를 수락했습니다" |
| "배차 거부" | Reject dispatch | "배차를 거부했습니다" |
| "긴급 상황" | Emergency alert | "긴급 상황이 신고되었습니다" |
| "연비 확인" | Check fuel efficiency | "현재 연비는 12.5km/L입니다" |

## BLE GATT Profile

```
Service UUID: 0000FFF0-0000-1000-8000-00805F9B34FB
├── Vehicle Data (Read/Notify): 0000FFF1-...
├── Commands (Write): 0000FFF2-...
└── AI Results (Read): 0000FFF3-...
```

## Directory Structure

```
android-driver/
├── app/
│   └── src/main/
│       ├── java/com/glec/driver/
│       │   ├── MainActivity.kt
│       │   ├── BLECentralService.kt
│       │   ├── VoiceCommandModule.kt
│       │   └── DispatchManagementService.kt
│       ├── res/
│       │   ├── layout/
│       │   ├── values/
│       │   └── drawable/
│       └── assets/
│           ├── vosk-model-ko/
│           └── wake_word.ppn
└── build.gradle.kts
```

## Build & Install

```bash
# Debug build
./.claude/skills/android-build/run.sh driver

# Install to device
./.claude/skills/android-build/run.sh driver --install --log
```

## Next Steps

1. Implement BLE central connection logic
2. Integrate Vosk STT and Porcupine wake word
3. Create UI layouts and navigation
4. Implement external API integrations
5. Write unit and UI tests
