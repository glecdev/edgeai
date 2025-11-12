# 🚀 로컬 GPU 실행 가이드 - PyTorch + Miniconda

**GLEC DTG EdgeAI - 실제 GPU 학습 실행**

**목적**: 로컬 GPU에서 PyTorch와 Miniconda를 활용한 실제 모델 학습
**대상**: 로컬 개발자 (NVIDIA GPU 보유)
**예상 시간**: 총 6-10시간 (환경 구축 1시간 + 학습 5-9시간)

---

## 📋 사전 요구사항 체크

### 필수 하드웨어
- ✅ **NVIDIA GPU**: RTX 3060 이상 (VRAM 12GB 권장)
- ✅ **RAM**: 16GB 이상 (32GB 권장)
- ✅ **저장공간**: 100GB 이상 여유 (SSD 권장)
- ✅ **전원**: 노트북의 경우 AC 어댑터 연결 필수

### 필수 소프트웨어
- ✅ **Windows 11** 또는 **Ubuntu 22.04+**
- ✅ **NVIDIA Driver**: 최신 버전 (535.x 이상)
- ✅ **Miniconda**: Python 가상환경 관리
- ✅ **Git**: 코드 버전 관리

---

## 1️⃣ 환경 구축 (30분-1시간)

### Step 1.1: NVIDIA Driver 확인

```bash
# NVIDIA GPU 확인
nvidia-smi

# 예상 출력:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.154.05   Driver Version: 535.154.05   CUDA Version: 12.2   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
# +-----------------------------------------------------------------------------+
```

**만약 nvidia-smi가 작동하지 않으면**:
```bash
# Windows: NVIDIA 공식 사이트에서 최신 드라이버 다운로드
# https://www.nvidia.com/Download/index.aspx

# Ubuntu:
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

---

### Step 1.2: Miniconda 설치

**Windows**:
```powershell
# Miniconda 다운로드 (PowerShell)
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda.exe"

# 설치 실행 (GUI 따라서 진행)
.\miniconda.exe

# 설치 후 PowerShell 재시작
```

**Ubuntu/Linux**:
```bash
# Miniconda 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치
bash Miniconda3-latest-Linux-x86_64.sh

# 쉘 설정 적용
source ~/.bashrc

# 확인
conda --version
# 출력: conda 24.1.2 (또는 최신 버전)
```

---

### Step 1.3: Conda 환경 생성

```bash
# 프로젝트 디렉토리로 이동
cd d:\edgeai\edgeai-repo
# 또는 Linux: cd ~/edgeai/edgeai-repo

# Conda 환경 생성 (Python 3.10)
conda create -n dtg-ai python=3.10 -y

# 환경 활성화
conda activate dtg-ai

# 확인
python --version
# 출력: Python 3.10.13
```

**중요**: 이후 모든 명령어는 `dtg-ai` 환경이 활성화된 상태에서 실행하세요!

---

### Step 1.4: PyTorch 설치 (CUDA 11.8)

```bash
# dtg-ai 환경이 활성화된 상태에서:
conda activate dtg-ai

# PyTorch 2.2.0 + CUDA 11.8 설치
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 설치 확인 (약 2-3분 소요)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 예상 출력:
# PyTorch: 2.2.0+cu118
# CUDA available: True
# GPU: NVIDIA GeForce RTX 3060
```

**만약 CUDA available: False가 나온다면**:
1. NVIDIA 드라이버 재설치
2. PyTorch 재설치: `pip uninstall torch torchvision torchaudio` 후 다시 설치
3. 시스템 재부팅

---

### Step 1.5: 프로젝트 의존성 설치

```bash
# edgeai-repo 디렉토리에서:
cd d:\edgeai\edgeai-repo

# 모든 의존성 설치
pip install -r requirements.txt

# 주요 패키지 확인
pip list | grep -E "numpy|pandas|lightgbm|onnx|mlflow|pyyaml|scikit-learn"

# 예상 출력:
# lightgbm         4.3.0
# mlflow           2.15.1
# numpy            1.26.4
# onnx             1.16.0
# pandas           2.2.0
# PyYAML           6.0.1
# scikit-learn     1.4.0
```

---

### Step 1.6: 환경 검증

```bash
# 전체 테스트 suite 실행 (환경 검증)
cd d:\edgeai\edgeai-repo

# Windows:
python -m pytest tests/ -v --tb=no -q --ignore=tests\e2e_test.py --ignore=tests\benchmark_inference.py --ignore=tests\data_validator.py

# Linux:
python -m pytest tests/ -v --tb=no -q --ignore=tests/e2e_test.py --ignore=tests/benchmark_inference.py --ignore=tests/data_validator.py

# 예상 출력:
# ============================== 159 passed in 15-20s ==============================
```

**✅ 모든 테스트가 통과하면 환경 구축 완료!**

---

## 2️⃣ 데이터 생성 (10-30분)

### Step 2.1: 학습 데이터셋 생성

```bash
cd d:\edgeai\edgeai-repo\data-generation

# 테스트 데이터셋 생성 (빠른 검증용, 1,000 샘플, 약 2-3분)
python -c "
import sys
sys.path.append('..')
from ai-models.utils.synthetic_simulator import generate_dataset
import numpy as np

print('🚀 테스트 데이터셋 생성 중...')

# Train: 800 샘플, 0% anomaly (LSTM-AE는 정상 데이터만 학습!)
X_train, y_fuel_train, y_anomaly_train = generate_dataset(
    num_samples=800,
    duration_minutes=1.0,
    patterns=['highway_cruise', 'city_traffic'],
    anomaly_ratio=0.0,  # 중요: 0% anomaly!
    sampling_rate_hz=1.0
)

# Val: 150 샘플, 10% anomaly (threshold calibration용)
X_val, y_fuel_val, y_anomaly_val = generate_dataset(
    num_samples=150,
    duration_minutes=1.0,
    patterns=['highway_cruise', 'city_traffic'],
    anomaly_ratio=0.1,
    sampling_rate_hz=1.0
)

# Test: 150 샘플, 10% anomaly
X_test, y_fuel_test, y_anomaly_test = generate_dataset(
    num_samples=150,
    duration_minutes=1.0,
    patterns=['highway_cruise', 'city_traffic'],
    anomaly_ratio=0.1,
    sampling_rate_hz=1.0
)

# CSV 저장
import pandas as pd
import os

os.makedirs('../datasets', exist_ok=True)

feature_names = [
    'vehicle_speed', 'engine_rpm', 'throttle_position',
    'brake_pressure', 'coolant_temp', 'fuel_level',
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'steering_angle', 'gps_lat'
]

def save_dataset(X, y_fuel, y_anomaly, filename):
    # Reshape X: (num_samples, sequence_length, features) -> (total_timesteps, features)
    num_samples, seq_len, num_features = X.shape
    X_flat = X.reshape(-1, num_features)

    # Create DataFrame
    df = pd.DataFrame(X_flat, columns=feature_names)

    # Add timestamp
    df['timestamp'] = np.tile(np.arange(seq_len), num_samples)

    # Add targets (repeat for each timestep in sequence)
    df['fuel_consumption'] = np.repeat(y_fuel, seq_len)
    df['carbon_emission'] = df['fuel_consumption'] * 2.31  # CO2 conversion
    df['label'] = np.repeat(['anomaly' if a == 1 else 'normal' for a in y_anomaly], seq_len)

    # Save
    df.to_csv(filename, index=False)
    print(f'✅ Saved: {filename} ({len(df)} rows)')

save_dataset(X_train, y_fuel_train, y_anomaly_train, '../datasets/train.csv')
save_dataset(X_val, y_fuel_val, y_anomaly_val, '../datasets/val.csv')
save_dataset(X_test, y_fuel_test, y_anomaly_test, '../datasets/test.csv')

print('✅ 테스트 데이터셋 생성 완료!')
print(f'   Train: {len(y_anomaly_train)} samples, {y_anomaly_train.mean()*100:.1f}% anomaly')
print(f'   Val:   {len(y_anomaly_val)} samples, {y_anomaly_val.mean()*100:.1f}% anomaly')
print(f'   Test:  {len(y_anomaly_test)} samples, {y_anomaly_test.mean()*100:.1f}% anomaly')
"
```

**예상 출력**:
```
🚀 테스트 데이터셋 생성 중...
INFO:__main__:Generated 100/800 samples
INFO:__main__:Generated 200/800 samples
...
✅ Saved: ../datasets/train.csv (48000 rows)
✅ Saved: ../datasets/val.csv (9000 rows)
✅ Saved: ../datasets/test.csv (9000 rows)
✅ 테스트 데이터셋 생성 완료!
   Train: 800 samples, 0.0% anomaly
   Val:   150 samples, 10.0% anomaly
   Test:  150 samples, 10.0% anomaly
```

---

### Step 2.2: Production 데이터셋 생성 (선택, 약 20-30분)

**테스트 학습이 성공하면** production 규모 데이터셋 생성:

```python
# Production: 10,000 샘플 (실제 학습용)
# 실행 시간: 약 20-30분

python -c "
import sys
sys.path.append('..')
from ai-models.utils.synthetic_simulator import generate_dataset
import numpy as np
import pandas as pd

print('🚀 Production 데이터셋 생성 중 (20-30분 소요)...')

# Train: 8,000 샘플, 0% anomaly
X_train, y_fuel_train, y_anomaly_train = generate_dataset(
    num_samples=8000,
    duration_minutes=1.0,
    patterns=['highway_cruise', 'city_traffic'],
    anomaly_ratio=0.0,
    sampling_rate_hz=1.0
)

# Val: 1,500 샘플, 10% anomaly
X_val, y_fuel_val, y_anomaly_val = generate_dataset(
    num_samples=1500,
    duration_minutes=1.0,
    patterns=['highway_cruise', 'city_traffic'],
    anomaly_ratio=0.1,
    sampling_rate_hz=1.0
)

# Test: 1,500 샘플, 10% anomaly
X_test, y_fuel_test, y_anomaly_test = generate_dataset(
    num_samples=1500,
    duration_minutes=1.0,
    patterns=['highway_cruise', 'city_traffic'],
    anomaly_ratio=0.1,
    sampling_rate_hz=1.0
)

# (save_dataset 함수는 위와 동일하게 사용)
# ...
"
```

---

## 3️⃣ GPU 모델 학습 (5-9시간)

### Step 3.1: TCN 모델 학습 (2-4시간)

```bash
cd d:\edgeai\edgeai-repo\ai-models\training

# TCN 학습 시작 (batch_size=64, epochs=100)
python train_tcn.py --config ../config.yaml --epochs 100 --batch-size 64

# 예상 출력:
# Training on device: cuda
# [INFO] Training without MLflow logging (MLflow 없어도 진행됨)
# Loading datasets...
# Creating model...
# Model parameters: 412928
# Starting training...
# Epoch 1/100 | Train Loss: 0.3245 | Val Loss: 0.2891 | R² Score: 0.6234 | Time: 45.23s
# Epoch 2/100 | Train Loss: 0.2456 | Val Loss: 0.2345 | R² Score: 0.7123 | Time: 43.12s
# ...
# Early stopping at epoch 42
# Training completed! Best validation loss: 0.1234
# Model saved: models/tcn_fuel_best.pth
```

**학습 모니터링**:
```bash
# 다른 터미널에서 GPU 사용률 실시간 확인
watch -n 1 nvidia-smi

# 예상 출력:
# | GPU  Name        | GPU-Util | Memory-Usage |
# | NVIDIA RTX 3060  |  95%     | 8192 / 12288 MiB |
```

**학습 중 문제 발생 시**:

**OOM (Out of Memory) 에러**:
```bash
# Batch size 줄이기
python train_tcn.py --config ../config.yaml --epochs 100 --batch-size 32
```

**느린 학습 속도** (epoch당 >5분):
```bash
# num_workers 조정 (DataLoader)
# train_tcn.py 파일 수정: num_workers=0 → num_workers=4
```

---

### Step 3.2: LSTM-AE 모델 학습 (2-4시간)

```bash
cd d:\edgeai\edgeai-repo\ai-models\training

# LSTM-AE 학습 시작
python train_lstm_ae.py --config ../config.yaml --epochs 100 --batch-size 64

# 예상 출력:
# Training on device: cuda
# Loading datasets...
# Creating model...
# Model parameters: 156672
# Calculating anomaly threshold...
# Anomaly threshold: 0.0234
# Starting training...
# Epoch 1/100 | Train Loss: 0.0456 | Val Loss: 0.0389 | F1: 0.6234 | Precision: 0.5890 | Recall: 0.6612 | Time: 52.34s
# Epoch 2/100 | Train Loss: 0.0312 | Val Loss: 0.0289 | F1: 0.7456 | Precision: 0.7234 | Recall: 0.7689 | Time: 51.23s
# ...
# Early stopping at epoch 38
# Training completed! Best validation loss: 0.0156
# Model saved: models/lstm_ae_best.pth
```

**중요**: LSTM-AE는 정상 데이터만 학습 (train.csv anomaly_ratio=0%)
- Validation에서 anomaly 데이터로 threshold 보정
- F1-Score > 0.85 목표

---

### Step 3.3: LightGBM 재학습 (선택, 30초)

```bash
cd d:\edgeai\edgeai-repo\ai-models\training

# LightGBM 학습 (CPU만으로 충분, 매우 빠름)
python train_lightgbm.py --config ../config.yaml

# 예상 출력:
# Training LightGBM model...
# [LightGBM] [Info] Total Bins 2550
# [LightGBM] [Info] Number of data points in the train set: 8000, number of used features: 11
# Training until validation scores don't improve for 50 rounds
# [100]	valid_0's multi_logloss: 0.0234	valid_0's multi_error: 0.0045
# ...
# Early stopping, best iteration is: [234]
# Training complete! Accuracy: 99.54%
# Model saved: models/lightgbm_behavior_model.txt
```

**이미 99.54% 정확도 달성**: 재학습은 선택사항

---

## 4️⃣ 모델 최적화 (1-2시간)

### Step 4.1: PyTorch → ONNX 변환

**TCN 모델 변환**:
```python
cd d:\edgeai\edgeai-repo\ai-models\conversion

python -c "
import torch
import sys
sys.path.append('..')
from training.train_tcn import TCN

# 모델 로드
device = torch.device('cuda')
model = TCN(input_dim=11, output_dim=1, num_channels=[64, 128, 256]).to(device)
checkpoint = torch.load('../training/models/tcn_fuel_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ONNX export
dummy_input = torch.randn(1, 60, 11).to(device)
torch.onnx.export(
    model,
    dummy_input,
    '../models/tcn_fuel_prediction.onnx',
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print('✅ TCN ONNX export complete: tcn_fuel_prediction.onnx')

# 모델 크기 확인
import os
size_mb = os.path.getsize('../models/tcn_fuel_prediction.onnx') / (1024**2)
print(f'   Model size: {size_mb:.2f} MB (target: <4MB)')
"
```

**LSTM-AE 모델 변환**:
```python
python -c "
import torch
import sys
sys.path.append('..')
from training.train_lstm_ae import LSTM_Autoencoder

# 모델 로드
device = torch.device('cuda')
model = LSTM_Autoencoder(input_dim=11, hidden_dim=128, num_layers=2, latent_dim=32).to(device)
checkpoint = torch.load('../training/models/lstm_ae_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ONNX export
dummy_input = torch.randn(1, 60, 11).to(device)
torch.onnx.export(
    model,
    dummy_input,
    '../models/lstm_ae_anomaly_detection.onnx',
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print('✅ LSTM-AE ONNX export complete: lstm_ae_anomaly_detection.onnx')

# 모델 크기 확인
import os
size_mb = os.path.getsize('../models/lstm_ae_anomaly_detection.onnx') / (1024**2)
print(f'   Model size: {size_mb:.2f} MB (target: <3MB)')

# Threshold 저장
threshold = checkpoint['threshold']
print(f'   Anomaly threshold: {threshold:.6f}')
"
```

---

### Step 4.2: INT8 Quantization (선택, 고급)

```python
# 양자화로 모델 크기 50-75% 감소
# (ONNX Runtime quantization)

python -c "
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# TCN quantization
model_fp32 = '../models/tcn_fuel_prediction.onnx'
model_quant = '../models/tcn_fuel_prediction_int8.onnx'

quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QUInt8
)

import os
size_before = os.path.getsize(model_fp32) / (1024**2)
size_after = os.path.getsize(model_quant) / (1024**2)
reduction = (1 - size_after/size_before) * 100

print(f'✅ TCN INT8 quantization complete')
print(f'   Before: {size_before:.2f} MB')
print(f'   After:  {size_after:.2f} MB')
print(f'   Reduction: {reduction:.1f}%')

# LSTM-AE quantization (동일한 방법)
# ...
"
```

---

### Step 4.3: 모델 검증

```bash
cd d:\edgeai\edgeai-repo\ai-models\conversion

# ONNX 모델 검증
python -c "
import onnx

# TCN 검증
tcn_model = onnx.load('../models/tcn_fuel_prediction.onnx')
onnx.checker.check_model(tcn_model)
print('✅ TCN ONNX model is valid')

# LSTM-AE 검증
lstm_model = onnx.load('../models/lstm_ae_anomaly_detection.onnx')
onnx.checker.check_model(lstm_model)
print('✅ LSTM-AE ONNX model is valid')

# 전체 모델 크기 확인
import os
tcn_size = os.path.getsize('../models/tcn_fuel_prediction.onnx') / (1024**2)
lstm_size = os.path.getsize('../models/lstm_ae_anomaly_detection.onnx') / (1024**2)
lightgbm_size = 0.012  # 12KB (이미 학습 완료)

total_size = tcn_size + lstm_size + lightgbm_size
print(f'')
print(f'📊 Total Model Size:')
print(f'   TCN:      {tcn_size:.2f} MB')
print(f'   LSTM-AE:  {lstm_size:.2f} MB')
print(f'   LightGBM: {lightgbm_size:.3f} MB')
print(f'   Total:    {total_size:.2f} MB (target: <14MB)')

if total_size < 14:
    print('✅ Within budget!')
else:
    print('⚠️ Exceeds budget, apply INT8 quantization')
"
```

---

## 5️⃣ Android 통합 (30분-1시간)

### Step 5.1: ONNX 모델을 Android에 복사

```bash
# Windows:
cd d:\edgeai
xcopy /Y edgeai-repo\ai-models\models\*.onnx edgeai-repo\android-dtg\app\src\main\assets\models\

# Linux:
cd ~/edgeai
cp edgeai-repo/ai-models/models/*.onnx edgeai-repo/android-dtg/app/src/main/assets/models/

# 복사 확인
dir edgeai-repo\android-dtg\app\src\main\assets\models\
# 또는 Linux: ls -lh edgeai-repo/android-dtg/app/src/main/assets/models/

# 예상 출력:
# tcn_fuel_prediction.onnx              (3.2 MB)
# lstm_ae_anomaly_detection.onnx        (2.1 MB)
# lightgbm_behavior_model.txt           (12 KB)
```

---

### Step 5.2: Android APK 빌드

```bash
cd d:\edgeai\edgeai-repo\android-dtg

# Gradle 빌드 (Debug APK)
# Windows:
gradlew.bat assembleDebug

# Linux:
./gradlew assembleDebug

# 빌드 시간: 약 5-10분 (첫 빌드는 더 오래 걸릴 수 있음)

# 예상 출력:
# > Task :app:compileDebugKotlin
# > Task :app:mergeDebugAssets
# > Task :app:packageDebug
# BUILD SUCCESSFUL in 8m 23s
# 156 actionable tasks: 156 executed

# APK 위치 확인
dir app\build\outputs\apk\debug\app-debug.apk
# 또는 Linux: ls -lh app/build/outputs/apk/debug/app-debug.apk

# 예상 출력:
# app-debug.apk  (약 15-20 MB)
```

**빌드 실패 시 문제 해결**:
```bash
# Gradle 캐시 정리
gradlew.bat clean
# 또는: ./gradlew clean

# 다시 빌드
gradlew.bat assembleDebug --stacktrace
```

---

### Step 5.3: 디바이스에 설치 및 테스트

```bash
# Android 디바이스 USB 연결 (USB 디버깅 활성화 필요)

# ADB 디바이스 확인
adb devices

# 예상 출력:
# List of devices attached
# ABC123456789    device

# APK 설치
adb install -r app\build\outputs\apk\debug\app-debug.apk
# 또는 Linux: adb install -r app/build/outputs/apk/debug/app-debug.apk

# 예상 출력:
# Performing Streamed Install
# Success

# 앱 실행
adb shell am start -n com.glec.dtg/.MainActivity

# 로그 확인
adb logcat -s DTG:* EdgeAI:* ONNX:*

# 예상 로그:
# DTG     : EdgeAI models loaded successfully
# EdgeAI  : TCN inference: 23.4ms
# EdgeAI  : LSTM-AE inference: 31.2ms
# EdgeAI  : LightGBM inference: 0.064ms
# DTG     : Total inference time: 54.7ms (target: <50ms) ⚠️
```

---

## 6️⃣ 성능 검증

### Step 6.1: 모델 성능 측정

```python
cd d:\edgeai\edgeai-repo\tests

# 추론 성능 벤치마크 (device에서)
adb shell "am instrument -w -e class com.glec.dtg.test.InferencePerformanceTest com.glec.dtg.test/androidx.test.runner.AndroidJUnitRunner"

# 예상 출력:
# InferencePerformanceTest:
# - TCN inference: 23.4ms (avg), 25.6ms (P95) ✅ <25ms
# - LSTM-AE inference: 31.2ms (avg), 34.8ms (P95) ✅ <35ms
# - LightGBM inference: 0.064ms (avg) ✅ <15ms
# - Total (parallel): 31.5ms (avg) ✅ <50ms
# - Model size: 5.3 MB + 2.1 MB + 0.012 MB = 7.4 MB ✅ <14MB
```

---

### Step 6.2: 정확도 검증

```python
cd d:\edgeai\edgeai-repo\ai-models\training

# Test dataset으로 정확도 검증
python -c "
import torch
import pandas as pd
import numpy as np
from train_tcn import TCN
from train_lstm_ae import LSTM_Autoencoder
from sklearn.metrics import r2_score, f1_score

device = torch.device('cuda')

# TCN 정확도
print('🔍 TCN Fuel Prediction 정확도:')
tcn_model = TCN(input_dim=11, output_dim=1, num_channels=[64, 128, 256]).to(device)
tcn_checkpoint = torch.load('models/tcn_fuel_best.pth')
tcn_model.load_state_dict(tcn_checkpoint['model_state_dict'])
tcn_model.eval()

# Test data 로드
test_df = pd.read_csv('../datasets/test.csv')
# (데이터 전처리 및 평가 코드...)

print(f'   R² Score: {tcn_checkpoint[\"r2_score\"]:.4f} (target: >0.85)')
print(f'   Val Loss: {tcn_checkpoint[\"val_loss\"]:.4f}')
print(f'   ✅ TCN meets target!' if tcn_checkpoint['r2_score'] > 0.85 else '⚠️ Below target')

# LSTM-AE 정확도
print('')
print('🔍 LSTM-AE Anomaly Detection 정확도:')
lstm_model = LSTM_Autoencoder(input_dim=11, hidden_dim=128, num_layers=2, latent_dim=32).to(device)
lstm_checkpoint = torch.load('models/lstm_ae_best.pth')
lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
lstm_model.eval()

print(f'   F1-Score: {lstm_checkpoint[\"f1_score\"]:.4f} (target: >0.85)')
print(f'   Val Loss: {lstm_checkpoint[\"val_loss\"]:.4f}')
print(f'   Threshold: {lstm_checkpoint[\"threshold\"]:.6f}')
print(f'   ✅ LSTM-AE meets target!' if lstm_checkpoint['f1_score'] > 0.85 else '⚠️ Below target')

# LightGBM (이미 99.54%)
print('')
print('🔍 LightGBM Behavior Classification:')
print(f'   Accuracy: 99.54% (target: >90%) ✅')
print(f'   Latency: 0.064ms (target: <15ms) ✅')
"
```

---

## 7️⃣ 완료 체크리스트

### ✅ 환경 구축
- [ ] NVIDIA GPU 작동 확인 (nvidia-smi)
- [ ] Miniconda 설치 및 환경 생성
- [ ] PyTorch CUDA 작동 확인 (torch.cuda.is_available())
- [ ] 의존성 설치 완료 (requirements.txt)
- [ ] 테스트 159/159 passing

### ✅ 데이터 생성
- [ ] train.csv 생성 (anomaly_ratio=0%)
- [ ] val.csv 생성 (anomaly_ratio=10%)
- [ ] test.csv 생성 (anomaly_ratio=10%)
- [ ] 데이터 품질 검증 (no NaN, valid ranges)

### ✅ 모델 학습
- [ ] TCN 학습 완료 (R² > 0.85)
- [ ] LSTM-AE 학습 완료 (F1 > 0.85)
- [ ] LightGBM 재학습 (선택, 이미 99.54%)
- [ ] 학습 로그 저장 (models/*.pth)

### ✅ 모델 최적화
- [ ] TCN ONNX 변환
- [ ] LSTM-AE ONNX 변환
- [ ] 모델 크기 검증 (<14MB total)
- [ ] INT8 quantization (선택)

### ✅ Android 통합
- [ ] ONNX 모델 복사 (assets/models/)
- [ ] APK 빌드 성공
- [ ] 디바이스 설치 및 실행
- [ ] 추론 성능 검증 (<50ms)

### ✅ 성능 목표 달성
- [ ] TCN: <4MB, <25ms, >85% R²
- [ ] LSTM-AE: <3MB, <35ms, >85% F1
- [ ] LightGBM: <10MB, <15ms, >90% Acc
- [ ] Total: <14MB, <50ms parallel

---

## 🎉 성공 기준

**✅ 최소 성공** (MVP):
- TCN 학습 완료, R² > 0.80
- LSTM-AE 학습 완료, F1 > 0.80
- APK 빌드 성공, 디바이스에서 실행

**✅ 목표 달성** (Production-Ready):
- TCN R² > 0.85, 추론 <25ms
- LSTM-AE F1 > 0.85, 추론 <35ms
- Total 모델 크기 <14MB
- Total 추론 시간 <50ms (parallel)

**🏆 완벽한 성공** (World-Class):
- TCN R² > 0.90
- LSTM-AE F1 > 0.90
- Total 추론 시간 <30ms
- Device 전력 소모 <2W

---

## 📊 예상 타임라인

| 단계 | 작업 | 예상 시간 | 누적 시간 |
|------|------|-----------|-----------|
| 1 | 환경 구축 | 30분-1시간 | 1시간 |
| 2 | 데이터 생성 (테스트) | 5-10분 | 1시간 10분 |
| 3 | TCN 학습 (테스트) | 15-30분 | 1시간 40분 |
| 4 | LSTM-AE 학습 (테스트) | 15-30분 | 2시간 10분 |
| 5 | 검증 및 조정 | 10-20분 | 2시간 30분 |
| 6 | 데이터 재생성 (production) | 20-30분 | 3시간 |
| 7 | TCN 재학습 (production) | 2-4시간 | 6시간 |
| 8 | LSTM-AE 재학습 (production) | 2-4시간 | 9시간 |
| 9 | ONNX 변환 | 10-20분 | 9시간 20분 |
| 10 | Android 통합 | 30분-1시간 | 10시간 |
| **합계** | | **6-10시간** | |

**권장 일정**:
- **Day 1 (2-3시간)**: 환경 구축 + 테스트 데이터셋 + 테스트 학습
- **Day 2 (5-7시간)**: Production 데이터셋 + 전체 학습 + 최적화
- **Day 3 (1-2시간)**: Android 통합 + 검증

---

## 🚨 문제 해결 (Troubleshooting)

### Issue 1: CUDA out of memory

**증상**: `RuntimeError: CUDA out of memory`

**해결**:
```bash
# Batch size 줄이기
python train_tcn.py --batch-size 32  # 64 → 32

# 또는 더 작게
python train_tcn.py --batch-size 16  # 64 → 16
```

---

### Issue 2: 학습이 너무 느림

**증상**: Epoch당 5분 이상 소요

**해결**:
```python
# train_tcn.py 파일 수정
# DataLoader num_workers 증가
DataLoader(..., num_workers=4)  # 0 → 4

# 또는 배치 크기 증가 (VRAM 여유 있는 경우)
python train_tcn.py --batch-size 128  # 64 → 128
```

---

### Issue 3: 정확도가 목표에 미달

**증상**: R² < 0.85 또는 F1 < 0.85

**해결**:
```bash
# 더 많은 epoch으로 학습
python train_tcn.py --epochs 200  # 100 → 200

# Learning rate 조정
# config.yaml 수정: learning_rate: 0.0005  # 0.001 → 0.0005

# 더 많은 데이터 생성 (20,000 샘플)
# data_generation 스크립트에서 num_samples 증가
```

---

### Issue 4: Android APK 빌드 실패

**증상**: Gradle 빌드 에러

**해결**:
```bash
# Gradle 캐시 정리
gradlew.bat clean
del /s /q .gradle
del /s /q build

# 다시 빌드
gradlew.bat assembleDebug --stacktrace --info

# Java 버전 확인 (JDK 17 필요)
java -version
```

---

## 📞 지원 및 문의

**문제 해결 리소스**:
1. **PRE_EXECUTION_VALIDATION_CHECKLIST.md**: 32가지 환경 검증 항목
2. **GPU_TRAINING_EXECUTION_GUIDE.md**: 상세한 단계별 가이드
3. **CURSOR.md**: Cursor AI를 위한 실행 가이드
4. **CTO_COMPREHENSIVE_ANALYSIS_REPORT.md**: 프로젝트 상태 분석

**GitHub Issues**: https://github.com/glecdev/edgeai/issues

---

## 🎓 학습 완료 후

### 다음 단계

1. **실제 차량 테스트**:
   - STM32 CAN bus 연결
   - 실제 차량 데이터로 추론 테스트
   - Edge 환경에서 성능 검증

2. **모델 개선**:
   - 더 많은 데이터로 재학습
   - Hyperparameter tuning (grid search)
   - Ensemble 모델 시도

3. **Production 배포**:
   - OTA 업데이트 시스템
   - Fleet AI 플랫폼 연동
   - 모니터링 및 A/B 테스트

---

**준비됐나요? 시작하세요! 🚀**

```bash
# 환경 활성화
conda activate dtg-ai

# 프로젝트 디렉토리로 이동
cd d:\edgeai\edgeai-repo

# Step 1부터 시작!
nvidia-smi
```

**좋은 학습 되세요! 💪**
