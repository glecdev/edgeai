"""
Production Dataset Generator (Real Korean Traffic Safety Board Structure)

Generates synthetic vehicle data matching Korean Traffic Safety Board format:
- 27 features (초단위 데이터 구조 준수)
- 8 anomaly types (급가속, 급감속, 급회전 등)
- Seoul GPS range (126.88-126.90, 37.51-37.58)
- Commercial vehicle patterns (택시, 버스, 화물차)

Usage:
    python generate_production_dataset_real_structure.py --num-samples 35000
"""
import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time


# ============================================================================
# Constants
# ============================================================================

SEOUL_GPS_RANGE = {
    'lon_min': 126.88,
    'lon_max': 126.90,
    'lat_min': 37.51,
    'lat_max': 37.58,
}

VEHICLE_PATTERNS = {
    'city_traffic': {
        'avg_speed': 25,
        'max_speed': 60,
        'stop_ratio': 0.3,
        'harsh_accel_prob': 0.05,
        'harsh_brake_prob': 0.08,
        'sharp_turn_prob': 0.10,
    },
    'highway_cruise': {
        'avg_speed': 90,
        'max_speed': 120,
        'stop_ratio': 0.02,
        'harsh_accel_prob': 0.02,
        'harsh_brake_prob': 0.03,
        'sharp_turn_prob': 0.02,
    },
    'taxi_operation': {
        'avg_speed': 30,
        'max_speed': 70,
        'stop_ratio': 0.25,
        'harsh_accel_prob': 0.10,
        'harsh_brake_prob': 0.12,
        'sharp_turn_prob': 0.15,
    },
    'bus_operation': {
        'avg_speed': 20,
        'max_speed': 60,
        'stop_ratio': 0.40,
        'harsh_accel_prob': 0.03,
        'harsh_brake_prob': 0.05,
        'sharp_turn_prob': 0.08,
    },
}

ANOMALY_TYPES = {
    'harsh_acceleration_20': {
        'speed_increase': 20,  # km/h in 1 second
        'throttle': 90,
        'rpm_increase': 1500,
    },
    'harsh_acceleration_40': {
        'speed_increase': 40,
        'throttle': 100,
        'rpm_increase': 2500,
    },
    'harsh_acceleration_60': {
        'speed_increase': 60,
        'throttle': 100,
        'rpm_increase': 3000,
    },
    'harsh_braking': {
        'speed_decrease': -20,
        'brake_pressure': 80,
    },
    'sharp_left_turn': {
        'heading_change': 90,
        'lateral_g': 0.5,
        'speed_reduction': 0.7,
    },
    'sharp_right_turn': {
        'heading_change': -90,
        'lateral_g': 0.5,
        'speed_reduction': 0.7,
    },
    'sudden_stop': {
        'speed_decrease': -30,
        'brake_pressure': 100,
        'time': 2,
    },
    'sharp_lane_change': {
        'lateral_movement': 3.5,
        'lateral_g': 0.4,
        'time': 2,
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def detect_harsh_acceleration(speed_prev: float, speed_cur: float, dt: float = 1.0) -> Tuple[int, int, int]:
    """
    Detect harsh acceleration based on speed change

    NOTE: Cumulative counting (40+ includes 20+, 60+ includes 20+ and 40+)
    This matches Korean Traffic Safety Board data format where:
    - 급가속건수20KM이상: ALL accelerations >= 20 km/h
    - 급가속건수40KM이상: ALL accelerations >= 40 km/h (subset of 20+)
    - 급가속건수60KM이상: ALL accelerations >= 60 km/h (subset of 40+ and 20+)

    Args:
        speed_prev: Previous speed (km/h)
        speed_cur: Current speed (km/h)
        dt: Time interval (seconds)

    Returns:
        (count_20km, count_40km, count_60km)
    """
    delta_v = speed_cur - speed_prev

    # Cumulative thresholds (40+ is also 20+, 60+ is also 40+ and 20+)
    count_20 = 1 if delta_v >= 20 else 0
    count_40 = 1 if delta_v >= 40 else 0
    count_60 = 1 if delta_v >= 60 else 0

    return count_20, count_40, count_60


def simulate_gps_movement(
    lat: float,
    lon: float,
    speed: float,
    heading: float,
    dt: float = 1.0
) -> Tuple[float, float]:
    """
    Simulate GPS position movement

    Args:
        lat: Current latitude (37.51-37.58)
        lon: Current longitude (126.88-126.90)
        speed: Speed (km/h)
        heading: Direction (0-360 degrees)
        dt: Time interval (seconds)

    Returns:
        (new_lat, new_lon)
    """
    # Distance = speed × time
    distance = (speed / 3.6) * dt  # meters

    # Haversine formula
    R = 6371000  # Earth radius in meters
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    heading_rad = np.radians(heading)

    # New position
    new_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(distance / R) +
        np.cos(lat_rad) * np.sin(distance / R) * np.cos(heading_rad)
    )
    new_lon_rad = lon_rad + np.arctan2(
        np.sin(heading_rad) * np.sin(distance / R) * np.cos(lat_rad),
        np.cos(distance / R) - np.sin(lat_rad) * np.sin(new_lat_rad)
    )

    new_lat = np.degrees(new_lat_rad)
    new_lon = np.degrees(new_lon_rad)

    # Clip to Seoul range
    new_lat = np.clip(new_lat, SEOUL_GPS_RANGE['lat_min'], SEOUL_GPS_RANGE['lat_max'])
    new_lon = np.clip(new_lon, SEOUL_GPS_RANGE['lon_min'], SEOUL_GPS_RANGE['lon_max'])

    return new_lat, new_lon


# ============================================================================
# Vehicle Simulator
# ============================================================================

class VehicleSimulator:
    """Vehicle simulator with physics-based motion and anomaly injection"""

    def __init__(self, pattern: str = 'city_traffic'):
        self.pattern = pattern
        self.config = VEHICLE_PATTERNS[pattern]

        # Initial state
        self.lat = np.random.uniform(SEOUL_GPS_RANGE['lat_min'], SEOUL_GPS_RANGE['lat_max'])
        self.lon = np.random.uniform(SEOUL_GPS_RANGE['lon_min'], SEOUL_GPS_RANGE['lon_max'])
        self.speed = 0.0
        self.heading = np.random.uniform(0, 360)
        self.rpm = 800
        self.throttle = 0
        self.brake = 0

        # Counters
        self.timestep = 0

    def step(self, inject_anomaly: Optional[str] = None) -> Dict:
        """
        Simulate one timestep (1 second)

        Args:
            inject_anomaly: Anomaly type to inject (or None for normal)

        Returns:
            dict with 27 features
        """
        dt = 1.0  # seconds
        speed_prev = self.speed

        # Normal driving behavior
        if inject_anomaly is None:
            # Random acceleration/deceleration (smooth, non-harsh)
            if np.random.random() < self.config['stop_ratio']:
                # Stop or slow down (gradually, < 20 km/h change)
                self.speed = max(0, self.speed - np.random.uniform(2, 10))
                self.throttle = 0
                self.brake = np.random.uniform(20, 60)
            else:
                # Accelerate or cruise
                target_speed = np.random.uniform(
                    self.config['avg_speed'] * 0.7,
                    self.config['avg_speed'] * 1.3
                )
                target_speed = min(target_speed, self.config['max_speed'])

                if self.speed < target_speed:
                    # Gradual acceleration (< 20 km/h change)
                    self.speed += np.random.uniform(2, 15)
                    self.throttle = np.random.uniform(30, 70)
                    self.brake = 0
                else:
                    self.speed -= np.random.uniform(1, 5)
                    self.throttle = np.random.uniform(10, 30)
                    self.brake = np.random.uniform(10, 30)

                self.speed = np.clip(self.speed, 0, self.config['max_speed'])

            # Random heading change (turns)
            if np.random.random() < 0.1:
                self.heading += np.random.uniform(-20, 20)
                self.heading = self.heading % 360

        # Anomaly injection
        else:
            anomaly = ANOMALY_TYPES[inject_anomaly]

            if 'harsh_acceleration' in inject_anomaly:
                self.speed += anomaly['speed_increase']
                self.throttle = anomaly['throttle']
                self.brake = 0
                self.rpm += anomaly['rpm_increase']

            elif inject_anomaly == 'harsh_braking':
                self.speed += anomaly['speed_decrease']
                self.throttle = 0
                self.brake = anomaly['brake_pressure']

            elif 'turn' in inject_anomaly:
                self.heading += anomaly['heading_change']
                self.heading = self.heading % 360
                self.speed *= anomaly['speed_reduction']

            elif inject_anomaly == 'sudden_stop':
                self.speed += anomaly['speed_decrease']
                self.throttle = 0
                self.brake = anomaly['brake_pressure']

        # Clip speed
        self.speed = np.clip(self.speed, 0, 120)

        # Update RPM based on speed (gear ratio approximation)
        if self.speed < 10:
            self.rpm = 800 + self.speed * 50
        elif self.speed < 40:
            self.rpm = 1200 + (self.speed - 10) * 40
        elif self.speed < 80:
            self.rpm = 2400 + (self.speed - 40) * 20
        else:
            self.rpm = 3200 + (self.speed - 80) * 10

        self.rpm = np.clip(self.rpm, 800, 4000)

        # Update GPS position
        self.lat, self.lon = simulate_gps_movement(
            self.lat, self.lon, self.speed, self.heading, dt
        )

        # Detect harsh acceleration
        count_20, count_40, count_60 = detect_harsh_acceleration(speed_prev, self.speed, dt)

        # Detect harsh braking
        harsh_brake_count = 1 if (self.speed - speed_prev) < -20 else 0

        # Detect sharp turn
        sharp_left_turn = 1 if inject_anomaly == 'sharp_left_turn' else 0
        sharp_right_turn = 1 if inject_anomaly == 'sharp_right_turn' else 0

        # Calculate fuel consumption (simplified)
        if self.speed > 0:
            fuel_per_second = (self.rpm / 1000) * 0.05 + (self.throttle / 100) * 0.1
        else:
            fuel_per_second = 0.01  # Idle fuel consumption

        # Calculate carbon emission (fuel × 2.31 kg CO2/L)
        carbon = fuel_per_second * 2.31

        # Build state dict (27 features)
        state = {
            'trip_id': 0,  # Will be set by generate_trip()
            'timestamp': 0,  # Will be set by generate_trip()
            'time_seconds': self.timestep,
            'gps_x': self.lon,
            'gps_y': self.lat,
            'vehicle_speed': self.speed,
            '급가속건수20KM이상': count_20,
            '급가속시간20KM이상': count_20,  # Simplified: count = time in seconds
            '급가속건수40KM이상': count_40,
            '급가속시간40KM이상': count_40,
            '급가속건수60KM이상': count_60,
            '급가속시간60KM이상': count_60,
            '급출발건수': 1 if count_20 > 0 and speed_prev == 0 else 0,
            '급출발시간': 1 if count_20 > 0 and speed_prev == 0 else 0,
            '급감속건수': harsh_brake_count,
            '급정지건수': 1 if harsh_brake_count > 0 and self.speed == 0 else 0,
            '급좌회전건수': sharp_left_turn,
            '우회전건수': sharp_right_turn,
            '급정거건수': 1 if inject_anomaly == 'sudden_stop' else 0,
            '급유턴운전건수': 1 if inject_anomaly == 'sharp_lane_change' else 0,
            '급차로변경건수': 1 if inject_anomaly == 'sharp_lane_change' else 0,
            'engine_rpm': self.rpm,
            'throttle_position': self.throttle,
            'brake_pressure': self.brake,
            'fuel_consumption': fuel_per_second,
            'carbon_emission': carbon,
            'label': 'anomaly' if inject_anomaly else 'normal',
        }

        self.timestep += 1
        return state


# ============================================================================
# Trip Generation
# ============================================================================

def generate_trip(
    trip_id: int,
    pattern: str = 'city_traffic',
    duration_seconds: int = 60,
    anomaly_ratio: float = 0.0
) -> pd.DataFrame:
    """
    Generate one trip (time-series data)

    Args:
        trip_id: Trip ID
        pattern: Driving pattern (city_traffic, highway_cruise, taxi_operation, bus_operation)
        duration_seconds: Trip duration in seconds (default 60)
        anomaly_ratio: Ratio of anomalous timesteps (0.0-1.0)

    Returns:
        DataFrame with 27 features × duration_seconds rows
    """
    simulator = VehicleSimulator(pattern=pattern)

    # Decide which timesteps will have anomalies
    num_anomalies = int(duration_seconds * anomaly_ratio)
    anomaly_timesteps = np.random.choice(
        duration_seconds,
        size=num_anomalies,
        replace=False
    ) if num_anomalies > 0 else []

    # Generate timesteps
    states = []
    base_timestamp = int(time.time())

    for t in range(duration_seconds):
        # Inject anomaly?
        if t in anomaly_timesteps:
            anomaly_type = np.random.choice(list(ANOMALY_TYPES.keys()))
        else:
            anomaly_type = None

        state = simulator.step(inject_anomaly=anomaly_type)
        state['trip_id'] = trip_id
        state['timestamp'] = base_timestamp + t
        states.append(state)

    df = pd.DataFrame(states)
    return df


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_dataset(
    num_samples: int,
    duration_seconds: int = 60,
    anomaly_ratio: float = 0.0,
    patterns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate full dataset

    Args:
        num_samples: Number of trips
        duration_seconds: Duration per trip (seconds)
        anomaly_ratio: Ratio of anomalous trips
        patterns: List of patterns to use (random if None)

    Returns:
        DataFrame with num_samples × duration_seconds rows
    """
    if patterns is None:
        patterns = ['city_traffic', 'highway_cruise', 'taxi_operation', 'bus_operation']

    all_trips = []

    print(f'Generating {num_samples} trips...')
    start_time = time.time()

    for i in range(num_samples):
        pattern = np.random.choice(patterns)
        df_trip = generate_trip(
            trip_id=i,
            pattern=pattern,
            duration_seconds=duration_seconds,
            anomaly_ratio=anomaly_ratio
        )
        all_trips.append(df_trip)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (num_samples - i - 1)
            print(f'  Progress: {i+1}/{num_samples} ({(i+1)/num_samples*100:.1f}%), '
                  f'Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s')

    df_all = pd.concat(all_trips, ignore_index=True)

    elapsed_total = time.time() - start_time
    print(f'Generation complete! Total time: {elapsed_total:.1f}s')
    print(f'Dataset shape: {df_all.shape}')
    print(f'Anomaly ratio: {(df_all["label"] == "anomaly").mean()*100:.2f}%')

    return df_all


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate production dataset (real structure)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Total number of samples (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='../datasets',
                        help='Output directory (default: ../datasets)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration per trip in seconds (default: 60)')

    args = parser.parse_args()

    print('=' * 80)
    print('GLEC DTG EdgeAI - Production Dataset Generator (Real Structure)')
    print('=' * 80)
    print()
    print(f'Parameters:')
    print(f'  Total samples: {args.num_samples:,}')
    print(f'  Output directory: {args.output_dir}')
    print(f'  Duration per trip: {args.duration}s')
    print()

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate splits
    num_train = int(args.num_samples * 0.80)
    num_val = int(args.num_samples * 0.10)
    num_test = args.num_samples - num_train - num_val

    print(f'Splits:')
    print(f'  Train: {num_train:,} samples (80%, 0% anomaly)')
    print(f'  Val:   {num_val:,} samples (10%, 10% anomaly)')
    print(f'  Test:  {num_test:,} samples (10%, 10% anomaly)')
    print()

    # Generate train
    print('[1/3] Generating training data...')
    df_train = generate_dataset(
        num_samples=num_train,
        duration_seconds=args.duration,
        anomaly_ratio=0.0  # LSTM-AE: unsupervised learning
    )
    train_path = os.path.join(args.output_dir, 'train.csv')
    df_train.to_csv(train_path, index=False)
    train_size_mb = os.path.getsize(train_path) / (1024**2)
    print(f'   Saved: {train_path} ({train_size_mb:.2f} MB)')
    print()

    # Generate val
    print('[2/3] Generating validation data...')
    df_val = generate_dataset(
        num_samples=num_val,
        duration_seconds=args.duration,
        anomaly_ratio=0.1
    )
    val_path = os.path.join(args.output_dir, 'val.csv')
    df_val.to_csv(val_path, index=False)
    val_size_mb = os.path.getsize(val_path) / (1024**2)
    print(f'   Saved: {val_path} ({val_size_mb:.2f} MB)')
    print()

    # Generate test
    print('[3/3] Generating test data...')
    df_test = generate_dataset(
        num_samples=num_test,
        duration_seconds=args.duration,
        anomaly_ratio=0.1
    )
    test_path = os.path.join(args.output_dir, 'test.csv')
    df_test.to_csv(test_path, index=False)
    test_size_mb = os.path.getsize(test_path) / (1024**2)
    print(f'   Saved: {test_path} ({test_size_mb:.2f} MB)')
    print()

    # Summary
    total_size_mb = train_size_mb + val_size_mb + test_size_mb
    print('=' * 80)
    print('Dataset Generation Complete!')
    print('=' * 80)
    print(f'Total samples: {args.num_samples:,}')
    print(f'Total size: {total_size_mb:.2f} MB')
    print(f'Features: 27 (Korean Traffic Safety Board format)')
    print(f'Anomaly types: 8 (급가속, 급감속, 급회전 등)')
    print()
    print('Next steps:')
    print(f'  1. Verify data: python -c "import pandas as pd; print(pd.read_csv(\'{train_path}\').info())"')
    print(f'  2. Train TCN: cd ai-models/training && python train_simple.py')
    print(f'  3. Train LSTM-AE: cd ai-models/training && python train_simple.py')
    print()


if __name__ == '__main__':
    main()
