# GPU Required Tasks (Deferred to Local Environment)

> **⚠️ Important**: These tasks require GPU hardware and must be executed in a local environment with appropriate hardware setup.

---

## 📋 Task List

### 1. Data Generation
**Status**: ✅ **Backup Ready** (Synthetic simulator available)
**Priority**: High
**Estimated Time**: 2-10 hours

**Option A: CARLA Simulator** (GPU Required, 8-10 hours)
- GPU: NVIDIA RTX 2070 or better
- RAM: 32GB minimum
- CARLA Simulator 0.9.15+ installed

```bash
# Start CARLA server
cd /path/to/CARLA
./CarlaUE4.sh

# Generate 10,000 episodes
cd data-generation/carla-scenarios
python generate_driving_data.py \
    --episodes 10000 \
    --duration 300 \
    --weather random \
    --traffic dense \
    --output ../datasets/carla_full.csv
```

**Option B: Synthetic Simulator** ✅ (CPU Only, 2 hours)
- ✅ **Available now** - Physics-based synthetic data generator
- No GPU required, runs on CPU
- Generates realistic driving data with 5 behavior types

```bash
# Generate 35,000 samples (train/val/test split)
cd data-generation
python synthetic_driving_simulator.py \
    --output-dir ../datasets \
    --samples 35000

# Output:
#   datasets/train.csv (28,000 samples)
#   datasets/val.csv (3,500 samples)
#   datasets/test.csv (3,500 samples)
```

**Data Distribution**:
- Eco driving: 30% (부드러운 가감속)
- Normal driving: 55% (일반 주행)
- Aggressive driving: 15% (급가속/급감속)

**Output**:
- Training data with labels: `eco_driving`, `normal`, `aggressive`, `highway`, `urban`
- Features: 15 columns (speed, RPM, throttle, brake, fuel, IMU, GPS, etc.)

---

### 2. Data Augmentation
**Status**: Pending
**Priority**: High
**Estimated Time**: 1-2 hours

**Requirements**:
- Python environment with tsaug
- Input: Generated CARLA data

**Commands**:
```bash
cd data-generation/augmentation
python augment_timeseries.py \
    --input ../datasets/carla_full.csv \
    --output ../datasets/carla_augmented.csv \
    --methods jitter scaling timewarp magwarp \
    --factor 3 \
    --split

# Expected output:
#   datasets/train.csv (70% × 3 augmentation = ~105,000 samples)
#   datasets/val.csv (15% × 3 = ~22,500 samples)
#   datasets/test.csv (15% × 1 = ~7,500 samples - no augmentation)
```

---

### 3. AI Model Training
**Status**: Pending
**Priority**: High
**Estimated Time**: 6-12 hours total

**Requirements**:
- GPU: NVIDIA RTX 3080/3090 or better (for faster training)
- CUDA 11.8+ and cuDNN 8.6+
- PyTorch 2.0+ with CUDA support
- MLflow server running

#### 3.1 TCN (Temporal Convolutional Network)
**Time**: 2-4 hours
**Target**: Fuel consumption prediction, speed pattern analysis

```bash
cd ai-models/training

# Start MLflow server (in separate terminal)
mlflow server --host 0.0.0.0 --port 5000 &

# Train TCN
python train_tcn.py \
    --config ../config.yaml \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --early-stopping 10

# Expected output:
#   models/tcn_fuel_best.pth (~4-5MB)
#   Accuracy: >85%
#   MLflow experiment logged
```

#### 3.2 LSTM-Autoencoder
**Time**: 2-4 hours
**Target**: Anomaly detection (dangerous driving, CAN intrusion, sensor faults)

```bash
python train_lstm_ae.py \
    --config ../config.yaml \
    --epochs 100 \
    --batch-size 64 \
    --latent-dim 16 \
    --early-stopping 10

# Expected output:
#   models/lstm_ae_best.pth (~3-4MB)
#   F1-Score: >0.85
#   Reconstruction error threshold: auto-calculated
```

#### 3.3 LightGBM
**Time**: 30 minutes - 1 hour
**Target**: Carbon emission estimation, driving behavior classification

```bash
python train_lightgbm.py \
    --config ../config.yaml \
    --num-leaves 31 \
    --learning-rate 0.05 \
    --n-estimators 500

# Expected output:
#   models/lightgbm_behavior.txt (~5-10MB)
#   Accuracy: >90%
```

#### 3.4 IBM Granite TTM-r2 (Tiny Time Mixer) ⭐ NEW
**Time**: 30 minutes - 1 hour (setup + zero-shot test)
**Target**: Fuel consumption prediction with pre-trained foundation model
**Status**: ✅ **Scripts Ready** (setup_ttm_r2.py, test_ttm_integration.py)

```bash
# Step 1: Download and validate model from Hugging Face
python setup_ttm_r2.py

# Expected output:
#   📥 Downloading ibm-granite/granite-timeseries-ttm-r2...
#   ✅ Model downloaded to: ai-models/models/ttm-r2/
#   📊 Model: ~1-10M parameters (4-40MB FP32)
#   🧪 Zero-shot validation successful
#   💾 Config saved: ai-models/models/ttm-r2/ttm_r2_config.json

# Step 2: Run integration tests
cd ../tests
python test_ttm_integration.py -v

# Expected: 8/8 tests passing
#   ✅ Model parameters < 20M (edge-friendly)
#   ✅ Input shape (1, 60, 10) validated
#   ✅ Zero-shot inference working
#   ⏱️ Latency benchmark (target <50ms after INT8)

# Step 3: Few-shot fine-tuning (optional, for accuracy boost)
cd ../training
python train_ttm_r2.py \
    --data ../../datasets/train.csv \
    --pretrained ibm-granite/granite-timeseries-ttm-r2 \
    --epochs 10 \
    --learning-rate 0.0001 \
    --few-shot 1000  # Use only 1000 samples

# Expected output:
#   models/ttm_r2_finetuned.pth (~10-15MB)
#   R² Score: >0.85 (zero-shot) → >0.90 (fine-tuned)
```

**Why TTM-r2?**
- ✅ **Pre-trained on time series** (NeurIPS 2024)
- ✅ **Zero-shot capable** (works without training)
- ✅ **Few-shot efficient** (1000 samples vs 28,000)
- ✅ **Edge-optimized** (1-10M params, laptop-runnable)
- ✅ **Apache 2.0 License**

**Comparison with TCN**:
| Metric | Custom TCN | IBM TTM-r2 |
|--------|------------|------------|
| Training Time | 2-4 hours | 30 min (few-shot) |
| Data Required | 28,000 samples | 1,000 samples |
| Model Size | 2-4MB (INT8) | 2-5MB (INT8) |
| Accuracy | 85-90% | 85-95% |
| Latency | 15-25ms | 5-15ms |

**Next Steps**:
1. Run `setup_ttm_r2.py` to download model
2. Compare zero-shot vs TCN baseline
3. Optional: Fine-tune with 1000 samples
4. Quantize to INT8 (see Section 4)

---

### 4. Model Quantization
**Status**: Pending
**Priority**: High
**Estimated Time**: 2-3 hours

**Requirements**:
- Trained PyTorch models
- Calibration dataset (1,000-5,000 samples)
- PyTorch quantization tools

#### 4.1 Post-Training Quantization (PTQ)
```bash
cd ai-models/optimization

# Quantize TCN
python quantize_model.py \
    --model ../training/models/tcn_fuel_best.pth \
    --method ptq \
    --calibration-samples 1000 \
    --output ../models/tcn_fuel_int8.pth

# Quantize LSTM-AE
python quantize_model.py \
    --model ../training/models/lstm_ae_best.pth \
    --method ptq \
    --calibration-samples 1000 \
    --output ../models/lstm_ae_int8.pth

# Expected size reduction: 4-5MB → 1-2MB per model
```

#### 4.2 Quantization-Aware Training (QAT) - Optional
```bash
# QAT for better accuracy after quantization
python quantize_model.py \
    --model ../training/models/tcn_fuel_best.pth \
    --method qat \
    --epochs 20 \
    --output ../models/tcn_fuel_qat_int8.pth
```

---

### 5. ONNX Export
**Status**: Pending
**Priority**: Medium
**Estimated Time**: 1 hour

**Requirements**:
- ONNX 1.14+
- onnxruntime 1.15+

```bash
cd ai-models/conversion

# Export all models to ONNX
python export_onnx.py \
    --model all \
    --generate-snpe-script

# Expected output:
#   models/tcn_fuel_int8.onnx
#   models/lstm_ae_int8.onnx
#   models/lightgbm_behavior.onnx (via sklearn2onnx)
#   models/convert_to_snpe.sh
```

---

## 🔄 Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Generation (8-10 hours)                             │
│    CARLA → carla_full.csv (~50,000 samples)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 2. Data Augmentation (1-2 hours)                            │
│    Augment → train.csv, val.csv, test.csv (~135,000 total)  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 3. Model Training (6-12 hours)                              │
│    TCN (2-4h) → tcn_fuel_best.pth                           │
│    LSTM-AE (2-4h) → lstm_ae_best.pth                        │
│    LightGBM (0.5-1h) → lightgbm_behavior.txt                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 4. Quantization (2-3 hours)                                 │
│    PTQ/QAT → INT8 models (~50% size reduction)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 5. ONNX Export (1 hour)                                     │
│    Export → .onnx files (ready for SNPE conversion)         │
└─────────────────────────────────────────────────────────────┘
```

**Total Estimated Time**: 18-28 hours (mostly automated, can run overnight)

---

## 📝 Performance Targets

After completion, models should meet these targets:

| Model | Size (INT8) | Latency | Accuracy/F1 |
|-------|-------------|---------|-------------|
| TCN | <2MB | <25ms | >85% |
| LSTM-AE | <2MB | <35ms | F1>0.85 |
| LightGBM | <10MB | <15ms | >90% |
| **Total** | **<14MB** | **<50ms** | **>85%** |

---

## 🔗 Dependencies

### Software
- CARLA Simulator 0.9.13+: https://github.com/carla-simulator/carla/releases
- PyTorch 2.0+ with CUDA: https://pytorch.org/get-started/locally/
- MLflow 2.5+: `pip install mlflow`
- ONNX 1.14+: `pip install onnx onnxruntime`

### Hardware
- **Minimum**: NVIDIA RTX 2070, 32GB RAM, 500GB SSD
- **Recommended**: NVIDIA RTX 3090, 64GB RAM, 1TB NVMe SSD
- **Optimal**: NVIDIA RTX 4090, 128GB RAM, 2TB NVMe SSD

---

## ✅ Checklist

Before starting GPU tasks, ensure:

- [ ] CARLA Simulator installed and tested
- [ ] CUDA and cuDNN properly configured
- [ ] Python environment with all dependencies installed
- [ ] MLflow server accessible
- [ ] Sufficient disk space (>500GB free)
- [ ] GPU driver updated (NVIDIA 525+ for RTX 30/40 series)
- [ ] Data generation scripts tested with small sample
- [ ] Training scripts tested with toy dataset

---

## 📞 Support

If issues arise during GPU tasks:
1. Check CARLA logs: `~/.config/Epic/CarlaUE4/Saved/Logs/`
2. Monitor GPU usage: `nvidia-smi -l 1`
3. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Verify MLflow UI: http://localhost:5000

---

**Last Updated**: 2025-01-09
**Status**: All GPU tasks deferred to local environment
**Next Review**: After web-based tasks completed
