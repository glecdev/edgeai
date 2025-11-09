# Android Build Skill

## Metadata
- **Name**: android-build
- **Description**: DTG 및 운전자 Android 앱 빌드 및 설치 자동화
- **Phase**: Phase 4
- **Dependencies**: Android SDK, Gradle, ADB
- **Estimated Time**: 3-10 minutes

## What This Skill Does

### 1. Build Android Apps
- **DTG 앱**: 차량 탑재 디바이스 앱 (Foreground Service, AI Inference)
- **운전자 앱**: 스마트폰 앱 (BLE, 음성 인터페이스)
- Debug/Release 빌드 지원
- JNI 네이티브 라이브러리 포함

### 2. Install to Device
- ADB를 통한 자동 설치
- 여러 디바이스 지원
- 이전 버전 자동 제거

### 3. Log Monitoring
- Logcat 실시간 모니터링
- 태그 필터링 (DTGService, AIInference, BLE)
- 크래시 감지 및 리포트

## Usage

```bash
# DTG 앱 빌드 (Debug)
./.claude/skills/android-build/run.sh dtg

# 운전자 앱 빌드 (Debug)
./.claude/skills/android-build/run.sh driver

# Release 빌드
./.claude/skills/android-build/run.sh dtg --release

# 빌드 + 설치 + 로그
./.claude/skills/android-build/run.sh dtg --install --log

# 모든 앱 빌드
./.claude/skills/android-build/run.sh all
```

## Files Created
- `android-dtg/app/build/outputs/apk/debug/app-debug.apk`
- `android-driver/app/build/outputs/apk/debug/app-debug.apk`
