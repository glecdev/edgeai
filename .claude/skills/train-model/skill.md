# Train AI Model Skill

## Metadata
- **Name**: train-model
- **Description**: MLflow + DVC 통합 AI 모델 학습 자동화
- **Phase**: Phase 2
- **Dependencies**: Python 가상환경, MLflow, DVC, PyTorch/TensorFlow
- **Estimated Time**: 30 minutes - 3 hours (depending on model)

## What This Skill Does

### 1. Model Training
- TCN (Temporal Convolutional Network) - 연료 소비 예측
- LSTM-Autoencoder - 이상 탐지 (위험 운전, CAN 침입)
- LightGBM - 운전 행동 분류, 탄소 배출 추정

### 2. Experiment Tracking
- MLflow로 하이퍼파라미터 자동 기록
- 학습 메트릭 실시간 추적 (loss, accuracy, F1-score)
- 모델 아티팩트 자동 저장

### 3. Data Versioning
- DVC로 학습 데이터 버전 관리
- 데이터셋 변경사항 추적
- 재현 가능한 실험 보장

### 4. Model Evaluation
- Validation set 성능 평가
- Test set 최종 검증
- 성능 목표 달성 확인 (>85% accuracy)

### 5. Model Export
- PyTorch → ONNX 변환
- 모델 저장 (MLflow Model Registry)
- 다음 단계(양자화)를 위한 준비

## Usage

### From Command Line
```bash
# TCN 모델 학습
./.claude/skills/train-model/run.sh tcn --epochs 100 --batch-size 64

# LSTM-AE 모델 학습
./.claude/skills/train-model/run.sh lstm_ae --epochs 50 --threshold 0.95

# LightGBM 모델 학습
./.claude/skills/train-model/run.sh lightgbm --n-estimators 1000

# 모든 모델 순차 학습
./.claude/skills/train-model/run.sh all
```

### From Claude Code
```
Please run the train-model skill to train the TCN model with 100 epochs.
```

## Configuration

### config.yaml 예시
```yaml
# ai-models/training/config.yaml
tcn:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  window_size: 60  # 60 seconds
  target: fuel_consumption

lstm_ae:
  epochs: 50
  batch_size: 32
  latent_dim: 32
  threshold_percentile: 95

lightgbm:
  n_estimators: 1000
  max_depth: 7
  learning_rate: 0.05
  num_leaves: 31
```

## Expected Output
```
🚀 Starting AI Model Training...

Model: TCN (Fuel Consumption Prediction)
Dataset: data/carla_synthetic/train.csv (10,000 episodes)

Epoch 1/100
  Train Loss: 0.4523 | Val Loss: 0.4012 | Accuracy: 72.3%
Epoch 10/100
  Train Loss: 0.2145 | Val Loss: 0.2534 | Accuracy: 84.1%
...
Epoch 100/100
  Train Loss: 0.0823 | Val Loss: 0.1123 | Accuracy: 89.7% ✅

📊 Final Metrics:
  • Train Accuracy: 91.2%
  • Val Accuracy: 89.7%
  • Test Accuracy: 88.5% (Target: >85% ✅)
  • Model Size: 3.2 MB (Target: <4 MB ✅)

💾 Model Saved:
  • MLflow Run ID: a7f3b2c1d4e5
  • ONNX Export: models/tcn_fuel_v1.0.0.onnx
  • DVC Tracked: data/models/tcn_fuel.pth.dvc

🔗 MLflow UI: http://localhost:5000/#/experiments/1/runs/a7f3b2c1d4e5
```

## Performance Targets

| Model | Size Target | Latency Target | Accuracy Target | Status |
|-------|-------------|----------------|-----------------|--------|
| TCN | < 4 MB | 15-25ms | > 85% | 🎯 |
| LSTM-AE | < 3 MB | 25-35ms | F1 > 0.85 | 🎯 |
| LightGBM | < 10 MB | 5-15ms | > 90% | 🎯 |

## Files Created
- `ai-models/training/models/tcn_fuel_v1.0.0.pth` - PyTorch 모델
- `ai-models/training/models/tcn_fuel_v1.0.0.onnx` - ONNX 모델
- `mlruns/` - MLflow 실험 결과
- `data/models/*.dvc` - DVC 추적 파일
- `training_report.md` - 학습 리포트

## Troubleshooting

### Out of Memory (OOM)
```bash
# 배치 크기 줄이기
./.claude/skills/train-model/run.sh tcn --batch-size 32

# GPU 메모리 정리
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 학습 정확도 목표 미달
```yaml
# config.yaml 조정
learning_rate: 0.0005  # 더 작게
epochs: 150  # 더 많이
```

### MLflow 연결 오류
```bash
# MLflow 서버 시작 확인
mlflow server --host 0.0.0.0 --port 5000

# 다른 터미널에서 학습 실행
```

### DVC 추적 오류
```bash
# DVC 재초기화
dvc init --force
dvc add data/training_set.csv
```

## Integration with Next Steps

학습 완료 후 자동으로:
1. **Phase 2 (모델 최적화)**: 양자화 및 프루닝 준비
2. **Phase 2 (모델 변환)**: ONNX → SNPE DLC 변환
3. **Phase 6 (테스트)**: 모델 성능 벤치마크
