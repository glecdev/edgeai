# AI Models - GLEC DTG Edge AI SDK

## Overview

This directory contains all AI model development components for the GLEC DTG Edge AI system.

## Model Architecture

### 1. TCN (Temporal Convolutional Network)
- **Purpose**: Fuel consumption prediction, speed pattern analysis
- **Target Size**: 2-4MB (INT8 quantized)
- **Target Latency**: 15-25ms
- **Target Accuracy**: 85-90%

### 2. LSTM-Autoencoder
- **Purpose**: Anomaly detection (dangerous driving, CAN intrusion, sensor faults)
- **Target Size**: 2-3MB (INT8 quantized)
- **Target Latency**: 25-35ms
- **Target F1-Score**: 0.85-0.92

### 3. LightGBM
- **Purpose**: Carbon emission estimation, driving behavior classification
- **Target Size**: 5-10MB
- **Target Latency**: 5-15ms
- **Target Accuracy**: 90-95%

## Directory Structure

```
ai-models/
├── training/           # Model training scripts
├── optimization/       # Quantization, pruning, QAT
├── conversion/         # ONNX → TFLite/SNPE conversion
├── simulation/         # CARLA data generation
└── tests/             # Model validation tests
```

## Development Workflow

```bash
# 1. Train model
python training/train_tcn.py --epochs 100 --batch-size 64

# 2. Quantize
python optimization/quantize_model.py --model tcn_fuel.pth --method ptq

# 3. Export to ONNX
python conversion/export_onnx.py --model tcn_fuel_quantized.pth

# 4. Convert to SNPE DLC (on local machine with SNPE SDK)
snpe-onnx-to-dlc --input_network tcn_fuel.onnx --output_path tcn_fuel.dlc

# 5. Run tests
pytest tests/ -v --cov=training
```

## Requirements

See `/requirements.txt` in project root.

## Next Steps

1. Implement model architectures in `training/`
2. Set up MLflow experiment tracking
3. Configure DVC for data versioning
4. Create training pipelines with `.claude/skills/train-model/run.sh`
