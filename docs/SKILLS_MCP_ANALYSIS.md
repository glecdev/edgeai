# Claude Code Skills & MCP 분석 보고서
## GLEC DTG Edge AI SDK 프로젝트

### 📊 프로젝트 개요
- **총 작업**: 141개 세부 todo
- **개발 기간**: 27주 (6-7개월)
- **7개 Phase**: 환경 설정 → AI 모델 → 임베디드 → Android → Fleet 연동 → 테스트 → 배포

---

## 1. 현재 사용 가능한 도구 분석

### 1.1 Built-in Tools (항상 사용 가능)
| 도구 | 용도 | 주요 Phase |
|-----|------|----------|
| **Read** | 파일 읽기, 이미지/PDF 읽기 | All Phases |
| **Write** | 새 파일 생성 | Phase 1-7 |
| **Edit** | 기존 파일 수정 | Phase 2-7 |
| **Glob** | 파일 패턴 검색 | All Phases |
| **Grep** | 코드 검색 (ripgrep) | Phase 2-7 |
| **Bash** | 명령 실행, 빌드, 테스트 | All Phases |
| **WebSearch** | 최신 기술 정보 검색 | Phase 1-2 |
| **WebFetch** | 문서/API 참조 | Phase 1-2 |
| **Task** | 서브에이전트 실행 (복잡한 작업) | All Phases |
| **TodoWrite** | 작업 추적 및 관리 | All Phases |

### 1.2 Available Skills
| Skill | 설명 | 필요 Phase |
|-------|------|----------|
| **session-start-hook** | 프로젝트 시작 시 자동 설정 (테스트, linter 등) | Phase 1, 7 |

### 1.3 Available MCP Servers
| MCP | 설명 | 필요 Phase |
|-----|------|----------|
| **mcp__codesign__sign_file** | 파일 서명 (Android APK 서명 등) | Phase 7 |

---

## 2. Phase별 필요한 Skills/MCP 분석

### Phase 1: 환경 설정 및 기초 (10개 작업)

#### ✅ 현재 도구로 충분
- **Bash**: Python venv, pip install, Docker setup, git init
- **Write**: requirements.txt, Dockerfile, .gitignore, build.gradle
- **Read/Edit**: 설정 파일 수정

#### 🔧 권장 추가 MCP
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/edgeai"]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "/home/user/edgeai"]
    }
  }
}
```

**이유**:
- `filesystem` MCP: 대량 파일 생성, 디렉토리 트리 구조화에 유리
- `git` MCP: Git 작업 간소화 (커밋, 브랜치 관리)

#### 💡 유용한 Custom Skill
```bash
# .claude/skills/setup-python-env.sh
#!/bin/bash
# Python 가상환경 자동 설정 및 의존성 설치
```

---

### Phase 2: AI 모델 개발 (23개 작업)

#### ✅ 현재 도구로 충분
- **Bash**: pip install pytorch, CARLA 실행, 모델 학습
- **Write**: train_tcn.py, quantize_model.py, export_onnx.py
- **Task (Explore)**: 오픈소스 모델 코드 검색 및 분석
- **WebSearch**: 최신 양자화 기법, SNPE 최적화 방법

#### 🔧 권장 추가 MCP
```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

**이유**:
- `fetch` MCP: 데이터셋 다운로드 (nuScenes, Waymo, Kaggle API)
- `memory` MCP: 학습 하이퍼파라미터, 모델 아키텍처 결정 컨텍스트 유지

#### 💡 유용한 Custom Skill
```bash
# .claude/skills/train-model.sh
#!/bin/bash
# MLflow 실험 추적과 함께 모델 학습 자동화
# DVC로 데이터 버전 관리

python train_tcn.py --config config.yaml
mlflow log-model model tcn_fuel
dvc add data/training_set.csv
```

#### 📦 필요한 Python 패키지 (requirements.txt)
```
torch==2.1.0
tensorflow==2.14.0
onnx==1.15.0
onnx2tf==1.17.5
lightgbm==4.1.0
tsaug==0.2.1
scikit-learn==1.3.2
mlflow==2.9.0
dvc==3.35.0
carla==0.9.15  # CARLA Python API
```

---

### Phase 3: 임베디드 시스템 통합 (18개 작업)

#### ✅ 현재 도구로 충분
- **Bash**: STM32 빌드 (make), st-flash 명령
- **Write**: STM32 HAL 코드 (.c/.h), CMakeLists.txt (JNI)
- **Edit**: 기존 펌웨어 수정

#### 🔧 권장 추가 MCP
```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "YOUR_API_KEY"
      }
    }
  }
}
```

**이유**:
- `brave-search` MCP: STM32 HAL 예제, JNI 메모리 관리 베스트 프랙티스 검색

#### 💡 유용한 Custom Skill
```bash
# .claude/skills/build-stm32.sh
#!/bin/bash
# STM32 펌웨어 빌드 및 플래시 자동화
cd stm32-firmware
make clean && make -j$(nproc)
st-flash write build/dtg_firmware.bin 0x8000000
```

#### 🎯 STM32CubeIDE 프로젝트 생성 방법
Claude Code는 GUI 도구를 직접 실행할 수 없으므로:
1. **사용자가 STM32CubeMX로 프로젝트 생성** (.ioc 파일)
2. **Claude Code가 HAL 코드 작성** (Src/, Inc/ 디렉토리)
3. **Claude Code가 Makefile 생성/수정**

---

### Phase 4: Android 애플리케이션 (35개 작업)

#### ✅ 현재 도구로 충분
- **Bash**: ./gradlew assembleDebug, adb install, adb logcat
- **Write**: MainActivity.kt, DTGForegroundService.kt, build.gradle
- **Edit**: AndroidManifest.xml, strings.xml
- **Task (general-purpose)**: 복잡한 Android 컴포넌트 구현

#### 🔧 권장 추가 MCP
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_TOKEN"
      }
    }
  }
}
```

**이유**:
- `github` MCP: 오픈소스 Android 예제 검색 (BLE, SNPE, Vosk 통합 코드)

#### 💡 유용한 Custom Skill
```bash
# .claude/skills/android-build-install.sh
#!/bin/bash
# Android 앱 빌드 및 디바이스 설치 자동화
APP=$1  # "dtg" or "driver"

cd android-${APP}
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat -s ${APP}Service:V AIInference:V
```

#### 📱 Android Studio 프로젝트 생성
**권장 방법**:
1. **Claude Code가 Gradle 기반 템플릿 생성**:
```bash
mkdir -p android-dtg
cd android-dtg
gradle init --type kotlin-application
```

2. **Claude Code가 Android 프로젝트 구조로 변환**:
```
android-dtg/
├── app/
│   ├── build.gradle.kts
│   ├── src/main/
│   │   ├── AndroidManifest.xml
│   │   ├── java/com/glec/dtg/
│   │   ├── cpp/
│   │   └── res/
```

---

### Phase 5: Fleet AI 플랫폼 연동 (18개 작업)

#### ✅ 현재 도구로 충분
- **Write**: MqttClient.kt, MqttMessageBuffer.kt, JSON schemas
- **Bash**: MQTT 연결 테스트 (mosquitto_pub/sub)
- **Edit**: Retrofit API 인터페이스

#### 🔧 권장 추가 MCP
```json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/home/user/edgeai/test.db"]
    }
  }
}
```

**이유**:
- `sqlite` MCP: 오프라인 큐 데이터베이스 스키마 설계 및 테스트 쿼리

#### 💡 유용한 Custom Skill
```bash
# .claude/skills/test-mqtt.sh
#!/bin/bash
# MQTT 메시지 발행/구독 테스트
mosquitto_pub -h mqtt.glec.ai -p 8883 \
  --cafile ca.crt \
  -t "fleet/vehicles/GLEC-DTG-001/telemetry" \
  -m '{"vehicle_id":"GLEC-DTG-001","speed":80.5}'
```

---

### Phase 6: 테스트 및 최적화 (17개 작업)

#### ✅ 현재 도구로 충분
- **Bash**: pytest, ./gradlew test, valgrind, adb shell dumpsys
- **Task (general-purpose)**: 복잡한 테스트 시나리오 작성
- **Grep**: 메모리 누수, 성능 병목 검색

#### 🔧 권장 추가 MCP
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

**이유**:
- `sequential-thinking` MCP: 복잡한 디버깅 시나리오 분석 (메모리 누수, 전력 프로파일링)

#### 💡 유용한 Custom Skill
```bash
# .claude/skills/run-tests.sh
#!/bin/bash
# 전체 테스트 스위트 실행
set -e

# AI 모델 테스트
cd ai-models
pytest tests/ -v --cov=training

# Android 단위 테스트
cd ../android-dtg
./gradlew testDebugUnitTest

# STM32 하드웨어 테스트 (시뮬레이션)
cd ../stm32-firmware
make test

echo "✅ All tests passed"
```

---

### Phase 7: 배포 준비 (20개 작업)

#### ✅ 현재 도구로 충분
- **Write**: .github/workflows/*.yml, Dockerfile, README.md
- **Bash**: git tag, GitHub Actions 트리거
- **mcp__codesign__sign_file**: APK 서명

#### 🔧 권장 추가 MCP
```json
{
  "mcpServers": {
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    }
  }
}
```

**이유**:
- `puppeteer` MCP: API 문서 자동 생성, 스크린샷 캡처 (사용자 매뉴얼용)

#### 💡 유용한 Custom Skill
```bash
# .claude/skills/create-release.sh
#!/bin/bash
# 프로덕션 릴리스 자동화
VERSION=$1  # e.g., "v1.0.0"

# Android 릴리스 빌드
cd android-dtg
./gradlew assembleRelease

# Git 태그 생성
git tag -a $VERSION -m "Release $VERSION"
git push origin $VERSION

# GitHub Release 생성
gh release create $VERSION \
  app/build/outputs/apk/release/app-release.apk \
  --title "GLEC DTG $VERSION" \
  --notes "See CHANGELOG.md"
```

---

## 3. 종합 추천 MCP 서버 설정

### 3.1 필수 MCP (우선순위 높음)
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/edgeai"]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "/home/user/edgeai"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### 3.2 권장 MCP (생산성 향상)
```json
{
  "mcpServers": {
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/home/user/edgeai/test.db"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### 3.3 선택적 MCP (특정 작업용)
```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    }
  }
}
```

---

## 4. Custom Skills 구현 계획

### 4.1 환경 설정 Skill
```bash
# .claude/skills/setup-dev-env/skill.md
---
name: setup-dev-env
description: GLEC DTG 개발 환경 자동 설정
---

## 실행 순서
1. Python 가상환경 생성 및 의존성 설치
2. Docker 이미지 빌드
3. Git hooks 설정 (pre-commit)
4. DVC 초기화
5. MLflow 서버 시작
```

### 4.2 빌드 & 테스트 Skill
```bash
# .claude/skills/build-all/skill.md
---
name: build-all
description: 전체 컴포넌트 빌드 (AI 모델, STM32, Android)
---

## 빌드 대상
- AI 모델: ONNX → SNPE DLC 변환
- STM32 펌웨어: make -j$(nproc)
- Android DTG 앱: ./gradlew assembleDebug
- Android 운전자 앱: ./gradlew assembleDebug
```

### 4.3 배포 Skill
```bash
# .claude/skills/deploy-ota/skill.md
---
name: deploy-ota
description: OTA 업데이트 패키지 생성 및 업로드
---

## 실행 순서
1. 릴리스 빌드 생성
2. 서명 (mcp__codesign__sign_file)
3. OTA 패키지 생성
4. Fleet AI 플랫폼 업로드
5. 배포 검증
```

---

## 5. Phase별 도구 사용 매트릭스

| Phase | Built-in Tools | 필수 MCP | 권장 MCP | Custom Skills |
|-------|---------------|---------|---------|--------------|
| **1. 환경 설정** | Bash, Write, Read | filesystem, git | memory | setup-dev-env |
| **2. AI 모델** | Bash, Write, Task | memory | fetch, brave-search | train-model |
| **3. 임베디드** | Bash, Write, Edit | - | brave-search | build-stm32 |
| **4. Android 앱** | Bash, Write, Edit, Task | - | github | android-build |
| **5. Fleet 연동** | Write, Bash | - | sqlite | test-mqtt |
| **6. 테스트** | Bash, Grep, Task | - | sequential-thinking | run-tests |
| **7. 배포** | Write, Bash | git | github, puppeteer | deploy-ota |

---

## 6. 구현 우선순위

### 즉시 설정 (Phase 1 시작 전)
1. ✅ `filesystem` MCP 설치
2. ✅ `git` MCP 설치
3. ✅ `memory` MCP 설치
4. ✅ `setup-dev-env` Skill 생성

### Phase 진행 중 추가
- Phase 2 시작 시: `fetch` MCP
- Phase 4 시작 시: `github` MCP
- Phase 5 시작 시: `sqlite` MCP
- Phase 7 시작 시: `puppeteer` MCP (문서화용)

---

## 7. MCP 설치 방법

### 7.1 Claude Desktop 설정 파일 위치
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### 7.2 설정 예시
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/edgeai"]
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "/home/user/edgeai"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### 7.3 적용 방법
1. Claude Desktop 종료
2. 설정 파일 수정
3. Claude Desktop 재시작
4. 새 대화에서 MCP 도구 확인: "list available tools"

---

## 8. 결론 및 권장사항

### ✅ 현재 Built-in 도구로 가능한 작업 (약 85%)
- 파일 생성/수정: Read, Write, Edit
- 빌드/테스트: Bash
- 코드 검색: Grep, Glob
- 복잡한 작업: Task (서브에이전트)

### 🔧 MCP 추가 시 생산성 향상 (15% 효율 증가)
- **filesystem**: 대량 파일 작업 간소화
- **git**: Git 작업 자동화
- **memory**: 컨텍스트 유지 (학습 하이퍼파라미터, 설계 결정)
- **fetch**: 외부 데이터/API 호출
- **github**: 오픈소스 예제 검색

### 💡 Custom Skills 구현 시 반복 작업 자동화
- 환경 설정, 빌드, 테스트, 배포 파이프라인

### 📊 예상 개발 속도 향상
- **Built-in 도구만**: 27주 (기준)
- **+ 필수 MCP 3개**: 24주 (11% 단축)
- **+ 권장 MCP 6개**: 22주 (18% 단축)
- **+ Custom Skills 5개**: 20주 (26% 단축)

**최종 권장**: 필수 MCP 3개 + Custom Skills 5개 조합으로 **약 4-5주 단축** 가능
