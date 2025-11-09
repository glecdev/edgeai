# Data Generation - CARLA Simulation

## Overview

This module generates synthetic driving data for AI model training using:
- CARLA Simulator (0.9.13+) for realistic driving scenarios
- Time-series data augmentation (tsaug)
- CAN message simulation (CANdevStudio)

## CARLA Simulation

### System Requirements

- **GPU**: NVIDIA RTX 2070+ (8GB+ VRAM)
- **RAM**: 32GB+
- **OS**: Ubuntu 20.04+ or Windows 10+
- **CARLA Version**: 0.9.13 or later

### Scenarios

```
carla-scenarios/
├── highway_driving.py      # Highway scenarios (60-120 km/h)
├── urban_driving.py        # City scenarios (30-60 km/h)
├── aggressive_driving.py   # Harsh acceleration/braking
├── eco_driving.py          # Fuel-efficient driving patterns
└── anomaly_scenarios.py    # Dangerous driving, sudden stops
```

## Data Collection

### Vehicle Telemetry (1Hz)

```python
{
    "timestamp": 1699564800.0,
    "vehicle_speed": 80.5,        # km/h
    "engine_rpm": 2500,            # rpm
    "throttle_position": 45.3,     # %
    "brake_pressure": 0.0,         # %
    "fuel_level": 75.0,            # %
    "coolant_temp": 85.0,          # °C
    "acceleration_x": 0.5,         # m/s²
    "acceleration_y": 0.1,         # m/s²
    "steering_angle": -5.2,        # degrees
    "gps_lat": 37.5665,
    "gps_lon": 126.9780
}
```

### Synthetic Data Generation

```bash
# Generate 10,000 episodes
cd data-generation/carla-scenarios
python generate_driving_data.py \
    --episodes 10000 \
    --weather random \
    --traffic dense \
    --output ../datasets/carla_synthetic.csv

# Time-series augmentation
python augment_timeseries.py \
    --input ../datasets/carla_synthetic.csv \
    --methods jitter,scaling,timewarping \
    --output ../datasets/carla_augmented.csv
```

## CAN Message Simulation

### CANdevStudio Configuration

```bash
# Simulate CAN messages without physical hardware
candevstudio --config can_simulation.xml

# Or use command-line tools
cangen can0 -I 0x7DF -L 8 -D r -g 100  # OBD-II requests
candump can0  # Monitor CAN traffic
```

## Directory Structure

```
data-generation/
├── carla-scenarios/
│   ├── generate_driving_data.py
│   ├── highway_driving.py
│   ├── urban_driving.py
│   └── anomaly_scenarios.py
├── can-simulation/
│   ├── can_simulation.xml
│   └── dbc_files/
│       ├── obd2.dbc
│       └── j1939.dbc
├── augmentation/
│   └── augment_timeseries.py
└── datasets/
    ├── carla_synthetic.csv
    └── carla_augmented.csv
```

## Data Format

### Training Dataset CSV

```csv
timestamp,vehicle_speed,engine_rpm,throttle_position,brake_pressure,fuel_consumption,label
1699564800.0,80.5,2500,45.3,0.0,12.5,normal
1699564801.0,82.1,2550,48.1,0.0,12.8,normal
1699564802.0,65.3,2200,10.5,65.0,8.2,harsh_braking
```

### Labels

- **normal**: Regular driving
- **eco_driving**: Fuel-efficient patterns
- **harsh_braking**: Sudden stops
- **harsh_acceleration**: Aggressive acceleration
- **anomaly**: Dangerous driving, sensor faults

## Usage Workflow

```bash
# 1. Start CARLA server (requires local machine with GPU)
./CarlaUE4.sh

# 2. Generate synthetic data
python carla-scenarios/generate_driving_data.py --episodes 10000

# 3. Augment data
python augmentation/augment_timeseries.py --input datasets/carla_synthetic.csv

# 4. Split train/val/test
python split_dataset.py --input datasets/carla_augmented.csv --ratios 0.7 0.15 0.15

# 5. Train models (see ai-models/training/)
cd ../ai-models/training
python train_tcn.py --dataset ../../data-generation/datasets/train.csv
```

## Next Steps

1. Install CARLA simulator (local machine)
2. Implement scenario scripts
3. Set up data collection pipelines
4. Create augmentation scripts
5. Integrate with DVC for data versioning
