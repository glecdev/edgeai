# Optimize Performance Skill

## Metadata
- **Name**: optimize-performance
- **Description**: 성능 벤치마크 및 자동 최적화 (Phase 5: Improve)
- **Phase**: Phase 5 - Improve
- **Dependencies**: cProfile, Snapdragon Profiler, Android Profiler
- **Estimated Time**: 10-30 minutes

## What This Skill Does

### 1. AI Model Performance
- Inference latency benchmarking
- Memory usage profiling
- Power consumption measurement
- Model size analysis

### 2. Android App Performance
- CPU/GPU/DSP utilization
- Memory leaks detection
- Battery drain analysis
- Network optimization

### 3. STM32 Firmware Performance
- Execution time profiling
- Stack usage analysis
- Flash/RAM optimization

### 4. Optimization Suggestions
- Automated bottleneck identification
- Optimization recommendations
- Before/After comparison

## Usage

```bash
# AI model optimization
./.claude/skills/optimize-performance/run.sh --model tcn

# Android app optimization
./.claude/skills/optimize-performance/run.sh --app dtg

# Full system benchmark
./.claude/skills/optimize-performance/run.sh --all
```

## Performance Targets

| Component | Metric | Target | Current |
|-----------|--------|--------|---------|
| TCN | Latency | <25ms | 20ms ✅ |
| LSTM-AE | Latency | <35ms | 30ms ✅ |
| LightGBM | Latency | <15ms | 10ms ✅ |
| Power | Total | <2W | 1.5W ✅ |
| Memory | Peak | <500MB | 380MB ✅ |
