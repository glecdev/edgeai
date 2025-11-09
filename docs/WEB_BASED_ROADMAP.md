# 웹 기반 Claude Code 개발 로드맵

## 🌐 환경 특성

### 웹 기반 Claude Code (GitHub 연동)
- ✅ **가능한 작업**: 코드 작성, 구조 설계, 문서화, Git 작업
- ❌ **불가능한 작업**: 하드웨어 연결, GPU 필요 작업, 로컬 앱 실행
- 🔄 **하이브리드 전략**: 웹에서 코드 작성 → 로컬에서 실행/테스트

### 제약사항
| 작업 | 웹 가능 | 로컬 필요 | 해결책 |
|-----|---------|----------|--------|
| 프로젝트 구조 생성 | ✅ | - | 즉시 시작 |
| Python 코드 작성 | ✅ | - | 즉시 시작 |
| Android 앱 템플릿 | ✅ | - | 즉시 시작 |
| STM32 코드 작성 | ✅ | - | 즉시 시작 |
| CARLA 시뮬레이션 | ❌ | ✅ | 스크립트만 작성 |
| AI 모델 학습 | ❌ | ✅ | 스크립트만 작성 |
| Android 빌드 | ❌ | ✅ | Gradle 설정만 |
| STM32 빌드 | ❌ | ✅ | Makefile만 작성 |

---

## 🎯 우선순위 로드맵 (웹 기반)

### Priority 1: 프로젝트 구조 생성 ⭐⭐⭐⭐⭐
**목표**: 전체 디렉토리 및 기본 파일 생성

```bash
# 실행 예상 시간: 2-3분
edgeai/
├── ai-models/
│   ├── training/
│   │   ├── train_tcn.py
│   │   ├── train_lstm_ae.py
│   │   ├── train_lightgbm.py
│   │   ├── quantize_model.py
│   │   ├── export_onnx.py
│   │   └── config.yaml
│   ├── optimization/
│   │   ├── ptq.py
│   │   ├── qat.py
│   │   └── pruning.py
│   ├── conversion/
│   │   ├── onnx_to_tflite.py
│   │   └── onnx_to_snpe.sh
│   ├── simulation/
│   │   └── carla_data_generation.py
│   └── tests/
│       ├── test_tcn.py
│       ├── test_lstm_ae.py
│       └── test_lightgbm.py
├── android-dtg/
│   ├── app/
│   │   ├── build.gradle.kts
│   │   └── src/main/
│   │       ├── AndroidManifest.xml
│   │       ├── java/com/glec/dtg/
│   │       │   ├── MainActivity.kt
│   │       │   ├── DTGForegroundService.kt
│   │       │   ├── BootReceiver.kt
│   │       │   └── snpe/SNPEEngine.kt
│   │       └── cpp/
│   │           ├── uart_reader.cpp
│   │           └── CMakeLists.txt
│   ├── build.gradle.kts
│   └── settings.gradle.kts
├── android-driver/
│   └── (similar structure)
├── stm32-firmware/
│   ├── Core/
│   │   ├── Src/
│   │   │   ├── main.c
│   │   │   ├── can.c
│   │   │   └── uart.c
│   │   └── Inc/
│   │       ├── main.h
│   │       ├── can.h
│   │       └── uart.h
│   ├── Drivers/
│   └── Makefile
├── fleet-integration/
│   ├── mqtt-client/
│   │   └── mqtt_client.py
│   └── protocol/
│       └── schemas.json
├── data-generation/
│   └── carla-scenarios/
├── .github/
│   └── workflows/
│       ├── python-tests.yml
│       ├── android-build.yml
│       └── stm32-build.yml
└── requirements.txt
```

**가치**: 전체 프로젝트 골격 완성 → 팀원이 즉시 작업 시작 가능

---

### Priority 2: Python 환경 설정 파일 ⭐⭐⭐⭐⭐
**목표**: requirements.txt, setup.py, config.yaml 작성

```python
# requirements.txt
torch==2.1.0
tensorflow==2.14.0
onnx==1.15.0
onnx2tf==1.17.5
lightgbm==4.1.0
scikit-learn==1.3.2
mlflow==2.9.0
dvc==3.35.0
pytest==7.4.3
```

**가치**: 로컬 개발자가 즉시 환경 설정 가능

---

### Priority 3: AI 모델 스켈레톤 코드 ⭐⭐⭐⭐
**목표**: TCN, LSTM-AE, LightGBM 기본 구조 작성

```python
# ai-models/training/train_tcn.py
import torch
import torch.nn as nn
import mlflow

class TCN(nn.Module):
    """Temporal Convolutional Network for fuel prediction"""
    def __init__(self, input_dim=10, output_dim=1, num_layers=3):
        super(TCN, self).__init__()
        # TODO: Implement architecture
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass

def train_tcn(config):
    """Train TCN model with MLflow tracking"""
    with mlflow.start_run():
        # TODO: Implement training loop
        pass

if __name__ == "__main__":
    train_tcn(config)
```

**가치**:
- 아키텍처 검증
- 테스트 작성 가능
- 문서 자동 생성 가능

---

### Priority 4: Android 앱 템플릿 ⭐⭐⭐⭐
**목표**: Gradle 프로젝트 구조 및 핵심 클래스 생성

```kotlin
// android-dtg/app/src/main/java/com/glec/dtg/MainActivity.kt
package com.glec.dtg

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // TODO: Initialize UI
    }
}

// DTGForegroundService.kt
class DTGForegroundService : Service() {
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, createNotification())
        startInferenceScheduler()
        return START_STICKY
    }

    private fun startInferenceScheduler() {
        // TODO: 1-minute periodic scheduler
    }
}

// BootReceiver.kt
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            // Start DTGForegroundService
        }
    }
}
```

**가치**:
- Android Studio에서 즉시 빌드 가능
- 아키텍처 검증
- JNI 연동 준비

---

### Priority 5: STM32 펌웨어 스켈레톤 ⭐⭐⭐
**목표**: HAL 기반 CAN/UART 코드 작성

```c
// stm32-firmware/Core/Src/main.c
#include "main.h"
#include "can.h"
#include "uart.h"

CAN_HandleTypeDef hcan1;
UART_HandleTypeDef huart2;

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_CAN1_Init();
    MX_UART2_Init();

    // Start CAN and UART
    HAL_CAN_Start(&hcan1);

    while (1) {
        // TODO: CAN → UART bridge
    }
}

// can.c
void MX_CAN1_Init(void) {
    hcan1.Instance = CAN1;
    hcan1.Init.Prescaler = 4;
    hcan1.Init.Mode = CAN_MODE_NORMAL;
    hcan1.Init.SyncJumpWidth = CAN_SJW_1TQ;
    hcan1.Init.TimeSeg1 = CAN_BS1_13TQ;
    hcan1.Init.TimeSeg2 = CAN_BS2_2TQ;
    // TODO: Configure filters
}
```

**가치**:
- STM32CubeIDE 프로젝트 베이스
- 로컬에서 즉시 컴파일 가능

---

### Priority 6: 테스트 템플릿 ⭐⭐⭐
**목표**: pytest, JUnit 테스트 스켈레톤

```python
# ai-models/tests/test_tcn.py
import pytest
import torch
from training.train_tcn import TCN

def test_tcn_output_shape():
    """TCN produces correct output shape"""
    model = TCN(input_dim=10, output_dim=1)
    x = torch.randn(32, 60, 10)  # batch, seq, features
    y = model(x)
    assert y.shape == (32, 1)

def test_tcn_inference_latency():
    """TCN inference < 25ms"""
    import time
    model = TCN()
    x = torch.randn(1, 60, 10)

    start = time.time()
    with torch.no_grad():
        y = model(x)
    latency_ms = (time.time() - start) * 1000

    assert latency_ms < 25  # Target
```

**가치**:
- TDD 가능
- CI/CD 준비
- 품질 보장

---

### Priority 7: GitHub Actions CI/CD ⭐⭐⭐
**목표**: 자동 빌드 및 테스트

```yaml
# .github/workflows/python-tests.yml
name: Python AI Models Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest ai-models/tests/ -v --cov=ai-models
```

**가치**:
- 자동 테스트
- 코드 품질 보장
- 팀 협업 효율

---

## 📊 웹 vs 로컬 작업 분담

### 웹 기반 Claude Code에서 할 작업 (70%)
✅ 프로젝트 구조 생성
✅ 모든 코드 스켈레톤 작성
✅ 테스트 템플릿 작성
✅ 문서 작성
✅ GitHub Actions 설정
✅ 코드 리뷰 및 개선
✅ 아키텍처 설계

### 로컬 환경에서 할 작업 (30%)
🏠 CARLA 시뮬레이션 실행
🏠 AI 모델 실제 학습
🏠 Android 앱 빌드/설치
🏠 STM32 펌웨어 빌드/플래시
🏠 하드웨어 연결 테스트

---

## 🚀 즉시 실행 계획

### Step 1: 프로젝트 구조 생성 (5분)
```bash
# 모든 디렉토리 및 README 생성
mkdir -p ai-models/{training,optimization,conversion,simulation,tests}
mkdir -p android-dtg/app/src/main/{java/com/glec/dtg,cpp,res,assets}
mkdir -p android-driver/app/src/main/{java/com/glec/driver,res,assets}
mkdir -p stm32-firmware/{Core/{Src,Inc},Drivers}
mkdir -p fleet-integration/{mqtt-client,protocol}
mkdir -p data-generation/carla-scenarios
mkdir -p .github/workflows
```

### Step 2: 핵심 파일 작성 (10분)
- requirements.txt
- AI 모델 스켈레톤 (TCN, LSTM-AE, LightGBM)
- Android Gradle 설정
- STM32 Makefile

### Step 3: 테스트 작성 (5분)
- pytest 템플릿
- GitHub Actions 워크플로우

### Step 4: 커밋 및 푸시 (2분)
```bash
git add -A
git commit -m "feat: Initialize complete project structure

- Create all directory structures
- Add Python AI model skeletons
- Add Android app templates
- Add STM32 firmware skeleton
- Configure GitHub Actions CI/CD

Project is now ready for local development"
git push
```

---

## 🎯 예상 결과

완료 후:
- ✅ **완전한 프로젝트 구조** - 모든 디렉토리 및 파일
- ✅ **즉시 빌드 가능** - requirements.txt, Gradle, Makefile
- ✅ **테스트 준비 완료** - pytest, GitHub Actions
- ✅ **팀 협업 가능** - 명확한 구조, 문서, CI/CD

로컬 개발자가 할 일:
1. `git clone`
2. `pip install -r requirements.txt` (또는 `.claude/skills/setup-dev-env/run.sh`)
3. 즉시 개발 시작!

---

## 💡 핵심 전략

**웹 기반의 강점 활용**:
- 🌐 언제 어디서나 접근 가능
- 🤝 GitHub로 즉시 협업
- 📝 코드 작성 및 리뷰에 집중
- 🔄 로컬 팀원과 완벽한 분업

**제약 극복**:
- 코드만 작성, 실행은 로컬에서
- 스크립트와 설정만 준비
- CI/CD로 자동 검증

**결과**:
웹 환경에서도 프로젝트의 **70-80%를 완성** 가능!
