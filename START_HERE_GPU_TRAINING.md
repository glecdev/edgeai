# 🚀 시작하기: GPU 모델 학습 실행

**GLEC DTG EdgeAI - 로컬 GPU 환경에서 실제 모델 학습**

---

## ⚡ 빠른 시작 (Windows)

### 원클릭 자동화 실행

```cmd
REM 1. Miniconda가 설치되어 있는지 확인
conda --version

REM 2. NVIDIA GPU 확인
nvidia-smi

REM 3. 자동화 스크립트 실행 (모든 과정 자동화)
quick_start_gpu.bat
```

**이것만 실행하면 됩니다!** 스크립트가 자동으로:
1. Conda 환경 생성 (dtg-ai)
2. PyTorch + CUDA 11.8 설치
3. 의존성 설치 (requirements.txt)
4. 데이터셋 생성 (테스트 또는 production)
5. TCN + LSTM-AE 모델 학습 (4-8시간)
6. ONNX 변환 및 Android 통합

---

## 📋 수동 실행 (단계별)

자동화 스크립트가 작동하지 않거나 세밀한 제어가 필요한 경우:

### Step 1: 환경 구축 (30분-1시간)

```bash
# 1.1 Conda 환경 생성
conda create -n dtg-ai python=3.10 -y
conda activate dtg-ai

# 1.2 PyTorch + CUDA 설치
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 1.3 의존성 설치
cd edgeai-repo
pip install -r requirements.txt

# 1.4 검증
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -m pytest tests/ -v --tb=no -q --ignore=tests\e2e_test.py --ignore=tests\benchmark_inference.py
# 예상: 159/159 tests passing
```

---

### Step 2: 데이터 생성 (5-30분)

**옵션 A: 테스트 데이터 (빠른 검증, 5분)**
```bash
cd edgeai-repo
python ai-models/scripts/generate_test_dataset.py
# 결과: train.csv (800 samples), val.csv (150), test.csv (150)
```

**옵션 B: Production 데이터 (전체 학습, 30분)**
```bash
cd edgeai-repo
python ai-models/scripts/generate_production_dataset.py
# 결과: train.csv (8,000 samples), val.csv (1,500), test.csv (1,500)
```

**검증**:
```bash
python -c "import pandas as pd; df = pd.read_csv('datasets/train.csv'); print(f'Shape: {df.shape}'); print(f'Anomaly ratio: {(df.label != \"normal\").mean()*100:.1f}%')"
# 예상: Shape: (48000, 15), Anomaly ratio: 0.0%
```

---

### Step 3: TCN 모델 학습 (2-4시간)

```bash
cd edgeai-repo/ai-models/training
conda activate dtg-ai

# TCN 학습 시작
python train_tcn.py --config ../config.yaml --epochs 100 --batch-size 64

# 예상 출력:
# Training on device: cuda
# Model parameters: 412928
# Epoch 1/100 | Train Loss: 0.3245 | Val Loss: 0.2891 | R² Score: 0.6234 | Time: 45s
# ...
# Early stopping at epoch 42
# Training completed! Best validation loss: 0.1234
# Model saved: models/tcn_fuel_best.pth
```

**모니터링** (다른 터미널에서):
```bash
watch -n 1 nvidia-smi  # Linux
# 또는 Windows: nvidia-smi (수동으로 반복 실행)

# 예상: GPU-Util 90-95%, Memory 8GB/12GB
```

**OOM 에러 발생 시**:
```bash
# Batch size 줄이기
python train_tcn.py --config ../config.yaml --epochs 100 --batch-size 32
```

---

### Step 4: LSTM-AE 모델 학습 (2-4시간)

```bash
cd edgeai-repo/ai-models/training
conda activate dtg-ai

# LSTM-AE 학습 시작
python train_lstm_ae.py --config ../config.yaml --epochs 100 --batch-size 64

# 예상 출력:
# Training on device: cuda
# Model parameters: 156672
# Calculating anomaly threshold... (from training data)
# Anomaly threshold: 0.0234
# Epoch 1/100 | Train Loss: 0.0456 | Val Loss: 0.0389 | F1: 0.6234 | Time: 52s
# ...
# Early stopping at epoch 38
# Training completed! Best F1 score: 0.8734
# Model saved: models/lstm_ae_best.pth
```

**중요**: LSTM-AE는 정상 데이터만 학습합니다!
- Training data: 0% anomaly (unsupervised learning)
- Validation data: 10% anomaly (threshold calibration)

---

### Step 5: ONNX 변환 (10-20분)

```bash
cd edgeai-repo/ai-models/conversion
conda activate dtg-ai

# TCN ONNX 변환
python -c "
import torch
import sys
sys.path.append('..')
from training.train_tcn import TCN

device = torch.device('cuda')
model = TCN(input_dim=11, output_dim=1, num_channels=[64, 128, 256]).to(device)
checkpoint = torch.load('../training/models/tcn_fuel_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 60, 11).to(device)
torch.onnx.export(
    model, dummy_input, '../models/tcn_fuel_prediction.onnx',
    export_params=True, opset_version=13, do_constant_folding=True,
    input_names=['input'], output_names=['output']
)
print('✅ TCN ONNX export complete')
"

# LSTM-AE ONNX 변환
python -c "
import torch
import sys
sys.path.append('..')
from training.train_lstm_ae import LSTM_Autoencoder

device = torch.device('cuda')
model = LSTM_Autoencoder(input_dim=11, hidden_dim=128, num_layers=2, latent_dim=32).to(device)
checkpoint = torch.load('../training/models/lstm_ae_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 60, 11).to(device)
torch.onnx.export(
    model, dummy_input, '../models/lstm_ae_anomaly_detection.onnx',
    export_params=True, opset_version=13, do_constant_folding=True
)
print('✅ LSTM-AE ONNX export complete')
"

# 모델 크기 확인
dir ..\models\*.onnx  # Windows
# 또는: ls -lh ../models/*.onnx  # Linux

# 예상:
# tcn_fuel_prediction.onnx           3.2 MB
# lstm_ae_anomaly_detection.onnx     2.1 MB
# Total: 5.3 MB (target: <14MB) ✅
```

---

### Step 6: Android 통합 (30분-1시간)

```bash
# 6.1 ONNX 모델을 Android assets로 복사
cd edgeai-repo

# Windows:
xcopy /Y ai-models\models\*.onnx android-dtg\app\src\main\assets\models\

# Linux:
cp ai-models/models/*.onnx android-dtg/app/src/main/assets/models/

# 6.2 Android APK 빌드
cd android-dtg

# Windows:
gradlew.bat assembleDebug

# Linux:
./gradlew assembleDebug

# 예상 시간: 5-10분
# 결과: app\build\outputs\apk\debug\app-debug.apk

# 6.3 디바이스에 설치
adb devices  # 디바이스 연결 확인
adb install -r app\build\outputs\apk\debug\app-debug.apk

# 6.4 앱 실행 및 로그 확인
adb shell am start -n com.glec.dtg/.MainActivity
adb logcat -s DTG:* EdgeAI:* ONNX:*

# 예상 로그:
# EdgeAI: TCN inference: 23.4ms ✅
# EdgeAI: LSTM-AE inference: 31.2ms ✅
# EdgeAI: LightGBM inference: 0.064ms ✅
# EdgeAI: Total inference: 54.7ms (target: <50ms)
```

---

## ✅ 성공 확인

### 모델 성능 목표

| 모델 | 크기 목표 | 지연 목표 | 정확도 목표 | 검증 |
|------|----------|----------|------------|------|
| **TCN** | <4MB | <25ms | >85% R² | `tcn_fuel_best.pth` 내 `r2_score` |
| **LSTM-AE** | <3MB | <35ms | >85% F1 | `lstm_ae_best.pth` 내 `f1_score` |
| **LightGBM** | <10MB | <15ms | >90% Acc | 이미 99.54% ✅ |
| **Total** | <14MB | <50ms | - | 합계 확인 |

### 검증 명령어

```bash
cd edgeai-repo/ai-models/training

# TCN 정확도 확인
python -c "
import torch
checkpoint = torch.load('models/tcn_fuel_best.pth')
print(f'TCN R² Score: {checkpoint[\"r2_score\"]:.4f} (target: >0.85)')
print(f'Status: {\"✅ PASS\" if checkpoint[\"r2_score\"] > 0.85 else \"❌ FAIL\"}')"

# LSTM-AE 정확도 확인
python -c "
import torch
checkpoint = torch.load('models/lstm_ae_best.pth')
print(f'LSTM-AE F1 Score: {checkpoint[\"f1_score\"]:.4f} (target: >0.85)')
print(f'Status: {\"✅ PASS\" if checkpoint[\"f1_score\"] > 0.85 else \"❌ FAIL\"}')"

# 모델 크기 확인
dir ..\models\*.onnx  # Windows
# 또는: ls -lh ../models/*.onnx && du -sh ../models/*.onnx  # Linux
```

---

## 🎉 완료!

**축하합니다!** 다음을 완료했습니다:

1. ✅ GPU 환경 구축 (PyTorch + CUDA)
2. ✅ 학습 데이터셋 생성 (물리 기반 anomaly injection)
3. ✅ TCN 모델 학습 (연료 소비 예측)
4. ✅ LSTM-AE 모델 학습 (이상 탐지)
5. ✅ ONNX 변환 (edge deployment)
6. ✅ Android 통합 (실 디바이스 테스트)

**총 소요 시간**: 6-10시간
- 환경 구축: 1시간
- 데이터 생성: 30분
- 모델 학습: 4-8시간
- ONNX 변환 + Android: 1시간

---

## 📊 다음 단계

### 성능 최적화 (선택)

**INT8 Quantization** (모델 크기 50-75% 감소):
```bash
cd edgeai-repo/ai-models/optimization

python quantize_models.py --model tcn --input ../models/tcn_fuel_prediction.onnx --output ../models/tcn_fuel_prediction_int8.onnx

python quantize_models.py --model lstm_ae --input ../models/lstm_ae_anomaly_detection.onnx --output ../models/lstm_ae_anomaly_detection_int8.onnx
```

---

### 실제 차량 테스트

1. STM32 CAN bus 연결
2. 실제 차량 데이터로 추론
3. Edge 환경 성능 검증
4. Fleet AI 플랫폼 연동

---

### 하이퍼파라미터 튜닝 (더 높은 정확도)

```bash
# Grid search (예시)
for lr in 0.001 0.0005 0.0001; do
    python train_tcn.py --config ../config.yaml --epochs 200 --learning-rate $lr
done

# 또는 Optuna/Ray Tune 사용
python hyperparameter_search.py --model tcn --trials 50
```

---

## 🚨 문제 해결

### 문제: CUDA out of memory

**해결**:
```bash
# Batch size 줄이기
python train_tcn.py --batch-size 32  # 64 → 32
python train_lstm_ae.py --batch-size 16  # 64 → 16
```

---

### 문제: 학습이 너무 느림 (epoch당 5분 이상)

**해결**:
```python
# train_tcn.py 파일 수정
# Line ~320: DataLoader(..., num_workers=0) → num_workers=4
```

---

### 문제: 정확도가 목표에 미달

**해결**:
```bash
# 더 많은 epoch
python train_tcn.py --epochs 200

# 더 많은 데이터 (20,000 samples)
python ai-models/scripts/generate_production_dataset.py
# (스크립트 내 num_samples 수정)

# Learning rate 조정
# config.yaml: learning_rate: 0.0005  # 0.001 → 0.0005
```

---

### 문제: Android APK 빌드 실패

**해결**:
```bash
# Gradle 캐시 정리
gradlew.bat clean
del /s /q .gradle
del /s /q build

# 다시 빌드 (상세 로그)
gradlew.bat assembleDebug --stacktrace --info
```

---

## 📞 추가 리소스

**상세 가이드**:
1. **LOCAL_GPU_EXECUTION_GUIDE.md** - 완전한 단계별 가이드 (700+ lines)
2. **PRE_EXECUTION_VALIDATION_CHECKLIST.md** - 32가지 환경 검증 (1,220 lines)
3. **GPU_TRAINING_EXECUTION_GUIDE.md** - Phase-별 실행 가이드
4. **CURSOR.md** - Cursor AI를 위한 가이드

**자동화 스크립트**:
- `quick_start_gpu.bat` - Windows 원클릭 실행
- `ai-models/scripts/generate_test_dataset.py` - 테스트 데이터 생성
- `ai-models/scripts/generate_production_dataset.py` - Production 데이터 생성

**문의**:
- GitHub Issues: https://github.com/glecdev/edgeai/issues

---

## 🎓 핵심 교훈

### LSTM-AE Unsupervised Learning

**Critical**: LSTM-Autoencoder는 **정상 데이터만** 학습합니다!

```
Train:      0% anomaly   → 모델이 정상 패턴 학습
Validation: 5-15% anomaly → Threshold 보정 (95th percentile)
Test:       5-15% anomaly → 성능 평가
```

**이유**: Autoencoder는 입력을 재구성하도록 학습합니다. 만약 anomaly 데이터로 학습하면, 모델은 anomaly도 잘 재구성하게 되어 anomaly detection이 불가능합니다.

---

### Physics-Based Anomalies

**8가지 anomaly types** (Session 4에서 구현):
1. Overheating (과열)
2. Overrevving (과속회전)
3. Harsh Braking (급제동)
4. Aggressive Acceleration (급가속)
5. Erratic Driving (불규칙 운전)
6. Fuel Leak (연료 누출)
7. Excessive Idling (과도한 공회전)
8. GPS Jump (GPS 오류)

**특징**:
- 3-phase temporal model (onset → sustain → recovery)
- Multi-feature correlations (acceleration ↔ braking, RPM ↔ throttle)
- Realistic commercial vehicle dynamics

---

### Hardware Estimates

| GPU | VRAM | 예상 학습 시간 |
|-----|------|---------------|
| **RTX 4090** | 24GB | 3-4 hours ⚡ |
| **RTX 3080** | 10GB | 4-6 hours ✅ |
| **RTX 3060** | 12GB | 6-8 hours ✅ |
| **GTX 1660** | 6GB | 10-12 hours ⚠️ |
| **CPU Only** | N/A | 4-6 days ❌ |

---

**준비됐나요? 시작하세요! 🚀**

```bash
# Windows: 자동화 스크립트 실행
quick_start_gpu.bat

# 또는 수동: Step 1부터 시작
conda create -n dtg-ai python=3.10 -y
conda activate dtg-ai
...
```

**좋은 학습 되세요! 💪**
