# GLEC DTG Edge AI SDK - Custom Skills

## 📋 개요

이 디렉토리는 GLEC DTG Edge AI SDK 프로젝트의 반복 작업을 자동화하는 Custom Skills를 포함합니다.

## 🎯 사용 가능한 Skills

| Skill | 설명 | Phase | 사용 예시 |
|-------|------|-------|----------|
| **setup-dev-env** | Python, Docker, Git 개발 환경 자동 설정 | Phase 1 | `.claude/skills/setup-dev-env/run.sh` |
| **train-model** | MLflow + DVC 통합 AI 모델 학습 | Phase 2 | `.claude/skills/train-model/run.sh tcn` |
| **build-stm32** | STM32 펌웨어 빌드 및 플래시 | Phase 3 | `.claude/skills/build-stm32/run.sh flash` |
| **android-build** | Android 앱 빌드 및 설치 | Phase 4 | `.claude/skills/android-build/run.sh dtg --install` |
| **deploy-ota** | OTA 업데이트 패키지 생성/배포 | Phase 7 | `.claude/skills/deploy-ota/run.sh deploy --version 1.2.0` |
| **run-tests** | 전체 테스트 스위트 실행 | Phase 6 | `.claude/skills/run-tests/run.sh all` |

---

## 🚀 빠른 시작

### 1. 개발 환경 설정
```bash
# Python 가상환경, Docker, Git, DVC, MLflow 자동 설정
./.claude/skills/setup-dev-env/run.sh

# 가상환경 활성화
source venv/bin/activate

# MLflow 서버 시작
mlflow server --host 0.0.0.0 --port 5000
```

### 2. AI 모델 학습
```bash
# TCN 모델 학습 (100 epochs)
./.claude/skills/train-model/run.sh tcn --epochs 100

# LSTM-AE 모델 학습
./.claude/skills/train-model/run.sh lstm_ae

# 모든 모델 순차 학습
./.claude/skills/train-model/run.sh all

# MLflow UI에서 결과 확인
# http://localhost:5000
```

### 3. STM32 펌웨어 빌드 & 플래시
```bash
# 빌드만
./.claude/skills/build-stm32/run.sh build

# 빌드 + 플래시
./.claude/skills/build-stm32/run.sh flash

# 빌드 + 플래시 + 시리얼 모니터
./.claude/skills/build-stm32/run.sh flash --monitor
```

### 4. Android 앱 빌드
```bash
# DTG 앱 빌드 (Debug)
./.claude/skills/android-build/run.sh dtg

# 운전자 앱 빌드 + 설치
./.claude/skills/android-build/run.sh driver --install

# Release 빌드
./.claude/skills/android-build/run.sh dtg --release

# 빌드 + 설치 + 로그 모니터링
./.claude/skills/android-build/run.sh dtg --install --log
```

### 5. 테스트 실행
```bash
# 전체 테스트
./.claude/skills/run-tests/run.sh all

# AI 모델 테스트만
./.claude/skills/run-tests/run.sh ai

# Android 테스트만
./.claude/skills/run-tests/run.sh android
```

### 6. OTA 배포
```bash
# OTA 패키지 생성
./.claude/skills/deploy-ota/run.sh create --version 1.2.0

# Canary 배포 (10% 디바이스)
./.claude/skills/deploy-ota/run.sh upload --version 1.2.0 --target canary

# 전체 배포
./.claude/skills/deploy-ota/run.sh deploy --version 1.2.0 --target all
```

---

## 🤖 Claude Code에서 사용하기

### 방법 1: 직접 요청
```
Please run the setup-dev-env skill to configure my development environment.
```

### 방법 2: 작업 설명
```
I need to train the TCN model with 100 epochs. Can you use the train-model skill?
```

### 방법 3: 워크플로우
```
Can you:
1. Set up the development environment
2. Train all AI models
3. Run tests to verify everything works
```

---

## 📁 Skill 구조

각 Skill은 다음 구조를 가집니다:

```
.claude/skills/
├── setup-dev-env/
│   ├── skill.md         # Skill 설명 및 문서
│   └── run.sh           # 실행 스크립트
├── train-model/
│   ├── skill.md
│   └── run.sh
├── build-stm32/
│   ├── skill.md
│   └── run.sh
├── android-build/
│   ├── skill.md
│   └── run.sh
├── deploy-ota/
│   ├── skill.md
│   └── run.sh
└── run-tests/
    ├── skill.md
    └── run.sh
```

---

## 🔧 Skill 상세 설명

### 1. setup-dev-env

**목적**: 개발 환경 자동 설정

**수행 작업**:
- Python 3.9/3.10 가상환경 생성
- AI 라이브러리 설치 (PyTorch, TensorFlow, ONNX)
- Docker 및 Docker Compose 확인
- Git 초기화 및 .gitignore 생성
- DVC 초기화 (데이터 버전 관리)
- MLflow 디렉토리 구성

**소요 시간**: 5-10분

**의존성**:
- Python 3.9 또는 3.10
- pip
- Docker (선택적)

### 2. train-model

**목적**: AI 모델 학습 자동화

**수행 작업**:
- TCN (연료 소비 예측) 학습
- LSTM-AE (이상 탐지) 학습
- LightGBM (운전 행동 분류) 학습
- MLflow로 실험 추적
- DVC로 모델 버전 관리
- ONNX 내보내기

**소요 시간**: 30분 - 3시간 (모델에 따라)

**의존성**:
- Python 가상환경 (setup-dev-env 실행 필요)
- MLflow 서버 실행 중
- 학습 데이터 준비

### 3. build-stm32

**목적**: STM32 펌웨어 빌드 및 플래시

**수행 작업**:
- ARM GCC 툴체인으로 컴파일
- 바이너리 파일 생성 (.bin, .hex, .elf)
- ST-Link를 통한 자동 플래시
- 시리얼 모니터 시작 (선택적)

**소요 시간**: 2-5분

**의존성**:
- arm-none-eabi-gcc
- st-flash (ST-Link 드라이버)
- STM32 보드 + ST-Link 연결

### 4. android-build

**목적**: Android 앱 빌드 및 설치

**수행 작업**:
- Gradle 빌드 (Debug/Release)
- JNI 네이티브 라이브러리 컴파일
- ADB를 통한 자동 설치
- Logcat 모니터링 (선택적)

**소요 시간**: 3-10분

**의존성**:
- Android SDK
- Gradle
- ADB (설치 시)

### 5. deploy-ota

**목적**: OTA 업데이트 배포

**수행 작업**:
- Release APK 서명
- OTA 패키지 생성 (.zip)
- Metadata 생성 (버전, 체크섬)
- Fleet AI 플랫폼 업로드

**소요 시간**: 5-10분

**의존성**:
- Android Release APK
- curl
- zip

### 6. run-tests

**목적**: 전체 테스트 스위트 실행

**수행 작업**:
- AI 모델 단위 테스트 (pytest)
- Android 단위 테스트 (JUnit)
- Android Instrumentation 테스트 (Espresso)
- STM32 테스트 (시뮬레이션)
- 통합 테스트

**소요 시간**: 5-30분

**의존성**:
- pytest (AI 테스트)
- Gradle (Android 테스트)
- make (STM32 테스트)

---

## 🎯 개발 워크플로우 예시

### 전체 프로젝트 처음 시작
```bash
# 1. 개발 환경 설정
./.claude/skills/setup-dev-env/run.sh

# 2. 가상환경 활성화
source venv/bin/activate

# 3. AI 모델 학습
./.claude/skills/train-model/run.sh all

# 4. STM32 펌웨어 빌드
./.claude/skills/build-stm32/run.sh flash

# 5. Android 앱 빌드
./.claude/skills/android-build/run.sh dtg --install

# 6. 테스트 실행
./.claude/skills/run-tests/run.sh all
```

### 일상 개발 (코드 수정 후)
```bash
# Android 앱만 재빌드
./.claude/skills/android-build/run.sh dtg --install --log

# 또는 STM32 펌웨어만
./.claude/skills/build-stm32/run.sh flash --monitor
```

### 프로덕션 배포
```bash
# 1. Release 빌드
./.claude/skills/android-build/run.sh dtg --release

# 2. 테스트
./.claude/skills/run-tests/run.sh all

# 3. OTA 배포
./.claude/skills/deploy-ota/run.sh deploy --version 1.2.0 --target canary

# 4. 모니터링 후 전체 배포
./.claude/skills/deploy-ota/run.sh deploy --version 1.2.0 --target all
```

---

## 🛠 문제 해결

### Skill 실행 권한 오류
```bash
# 모든 스크립트에 실행 권한 부여
chmod +x .claude/skills/*/run.sh
```

### Python 가상환경 활성화 안됨
```bash
# 수동 활성화
source venv/bin/activate

# 또는 setup-dev-env 재실행
./.claude/skills/setup-dev-env/run.sh
```

### MLflow 서버 연결 실패
```bash
# MLflow 서버 시작
mlflow server --host 0.0.0.0 --port 5000

# 다른 터미널에서 학습 실행
```

### ST-Link 인식 안됨
```bash
# ST-Link 연결 확인
st-info --probe

# USB 권한 설정 (Linux)
sudo usermod -aG dialout $USER
```

---

## 📊 예상 생산성 향상

| 작업 | 수동 소요 시간 | Skill 사용 시간 | 절감 |
|-----|-------------|--------------|------|
| 개발 환경 설정 | 30-60분 | 5-10분 | 75% ↓ |
| AI 모델 학습 시작 | 10-15분 | 1분 | 90% ↓ |
| STM32 빌드 & 플래시 | 5-10분 | 2-5분 | 50% ↓ |
| Android 빌드 & 설치 | 5-10분 | 3-5분 | 50% ↓ |
| 전체 테스트 실행 | 15-30분 | 5-10분 | 60% ↓ |
| OTA 배포 | 20-30분 | 5-10분 | 70% ↓ |

**총 예상 절감**: **반복 작업 60-70% 단축**

---

## 📖 추가 문서

- [MCP 설정 가이드](../docs/MCP_SETUP_GUIDE.md) - Claude Desktop MCP 서버 설정
- [Skills & MCP 분석](../docs/SKILLS_MCP_ANALYSIS.md) - 도구 활용 분석
- [CLAUDE.md](../CLAUDE.md) - 프로젝트 전체 가이드

---

## 🔄 업데이트 이력

- **2025-01-09**: 초기 6개 Skills 생성
  - setup-dev-env
  - train-model
  - build-stm32
  - android-build
  - deploy-ota
  - run-tests

---

## 💡 팁

1. **Skills 체인**: Skills를 조합하여 워크플로우 자동화
2. **Claude Code 활용**: 자연어로 Skills 실행 요청
3. **로그 확인**: 각 Skill은 상세한 진행 상황 출력
4. **문제 발생 시**: skill.md 파일에서 Troubleshooting 섹션 참조

---

## 🎉 결론

Custom Skills를 사용하면 GLEC DTG Edge AI SDK 개발 과정에서 **반복 작업을 60-70% 단축**할 수 있으며, Claude Code와 함께 사용 시 더욱 효율적인 개발이 가능합니다.

Happy Coding! 🚀
