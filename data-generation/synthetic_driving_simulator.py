"""
GLEC DTG Edge AI - Synthetic Driving Data Generator (CARLA Backup)

합성 주행 데이터 생성기 (CARLA 시뮬레이터 백업용)
물리 법칙 기반으로 현실적인 차량 데이터 생성
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import json
from pathlib import Path


class DrivingBehavior(Enum):
    """운전 행동 유형"""
    ECO = "eco_driving"           # 경제 운전 (부드러운 가감속)
    NORMAL = "normal"             # 일반 운전
    AGGRESSIVE = "aggressive"     # 공격적 운전 (급가속/급감속)
    HIGHWAY = "highway"           # 고속도로 (일정 속도 유지)
    URBAN = "urban"               # 도심 (빈번한 정차)


@dataclass
class VehiclePhysics:
    """차량 물리 파라미터"""
    mass: float = 1500.0          # kg (승용차 기본값)
    max_acceleration: float = 4.0  # m/s² (0-100km/h in ~7s)
    max_braking: float = -8.0     # m/s² (급제동)
    drag_coefficient: float = 0.3  # 공기 저항 계수
    rolling_resistance: float = 0.015  # 구름 저항
    engine_efficiency: float = 0.25    # 엔진 효율 (25%)
    fuel_density: float = 0.75    # kg/L (휘발유)


@dataclass
class DrivingScenario:
    """주행 시나리오 설정"""
    name: str
    behavior: DrivingBehavior
    duration_seconds: int
    target_speed_range: Tuple[float, float]  # km/h
    acceleration_limit: float  # m/s²
    braking_limit: float  # m/s²
    throttle_pattern: str  # smooth, aggressive, varying
    brake_pattern: str  # gentle, aggressive
    speed_variation: float  # 속도 변화 정도 (0-1)


class SyntheticDrivingSimulator:
    """
    합성 주행 데이터 생성기

    특징:
    - 물리 법칙 기반 시뮬레이션 (뉴턴 운동 법칙)
    - 현실적인 엔진/센서 특성
    - 다양한 운전 행동 패턴
    - CARLA 없이 독립 실행 가능
    """

    def __init__(self, vehicle: VehiclePhysics = None, sampling_rate: int = 1):
        """
        Args:
            vehicle: 차량 물리 파라미터 (None이면 기본값)
            sampling_rate: 샘플링 주파수 (Hz)
        """
        self.vehicle = vehicle or VehiclePhysics()
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate  # 시간 간격

        # 시뮬레이션 상태
        self.state = {
            'speed': 0.0,  # km/h
            'acceleration': 0.0,  # m/s²
            'position': 0.0,  # m
            'throttle': 0.0,  # 0-100%
            'brake': 0.0,  # 0-100%
            'rpm': 800.0,  # idle RPM
            'fuel_level': 100.0,  # %
            'fuel_consumed': 0.0,  # L
            'coolant_temp': 85.0,  # °C
            'steering_angle': 0.0,  # degrees
        }

    def generate_scenario(self, scenario: DrivingScenario, add_noise: bool = True) -> pd.DataFrame:
        """
        주행 시나리오에 따른 데이터 생성

        Args:
            scenario: 시나리오 설정
            add_noise: 센서 노이즈 추가 여부

        Returns:
            시간별 차량 데이터 DataFrame
        """
        num_samples = scenario.duration_seconds * self.sampling_rate
        data_points = []

        # 시나리오에 따른 목표 속도 프로파일 생성
        target_speeds = self._generate_speed_profile(scenario, num_samples)

        for i in range(num_samples):
            timestamp = i / self.sampling_rate
            target_speed = target_speeds[i]

            # 물리 기반 시뮬레이션 스텝
            self._simulation_step(
                target_speed=target_speed,
                behavior=scenario.behavior,
                accel_limit=scenario.acceleration_limit,
                brake_limit=scenario.brake_limit
            )

            # 센서 데이터 생성
            sensor_data = self._generate_sensor_data(add_noise=add_noise)
            sensor_data['timestamp'] = timestamp
            sensor_data['label'] = scenario.behavior.value

            data_points.append(sensor_data)

        return pd.DataFrame(data_points)

    def _generate_speed_profile(self, scenario: DrivingScenario, num_samples: int) -> np.ndarray:
        """목표 속도 프로파일 생성"""
        min_speed, max_speed = scenario.target_speed_range

        if scenario.behavior == DrivingBehavior.ECO:
            # 경제 운전: 부드러운 가속, 일정 속도 유지
            profile = np.ones(num_samples) * ((min_speed + max_speed) / 2)
            # 부드러운 변화
            profile += np.sin(np.linspace(0, 4*np.pi, num_samples)) * 10 * scenario.speed_variation

        elif scenario.behavior == DrivingBehavior.AGGRESSIVE:
            # 공격적 운전: 급가속/급감속 반복
            profile = np.random.uniform(min_speed, max_speed, num_samples)
            # 급격한 변화
            for i in range(0, num_samples, 100):
                if i + 50 < num_samples:
                    profile[i:i+50] = np.linspace(profile[i], max_speed, 50)
                if i + 100 < num_samples:
                    profile[i+50:i+100] = np.linspace(max_speed, min_speed, 50)

        elif scenario.behavior == DrivingBehavior.HIGHWAY:
            # 고속도로: 일정 속도 유지 (100-120 km/h)
            profile = np.ones(num_samples) * ((min_speed + max_speed) / 2)
            # 약간의 변화 (차선 변경, 추월 등)
            profile += np.random.normal(0, 5, num_samples) * scenario.speed_variation

        elif scenario.behavior == DrivingBehavior.URBAN:
            # 도심: 빈번한 정차 및 재출발
            profile = np.zeros(num_samples)
            for i in range(0, num_samples, 200):
                # 가속 (0 → 50 km/h)
                if i + 100 < num_samples:
                    profile[i:i+100] = np.linspace(0, 50, 100)
                # 일정 속도
                if i + 150 < num_samples:
                    profile[i+100:i+150] = 50
                # 감속 (50 → 0 km/h)
                if i + 200 < num_samples:
                    profile[i+150:i+200] = np.linspace(50, 0, 50)

        else:  # NORMAL
            # 일반 운전: 다양한 속도 변화
            profile = np.random.uniform(min_speed, max_speed, num_samples)
            # 스무딩 (급격한 변화 제거)
            profile = np.convolve(profile, np.ones(20)/20, mode='same')

        return np.clip(profile, 0, max_speed)

    def _simulation_step(self, target_speed: float, behavior: DrivingBehavior,
                         accel_limit: float, brake_limit: float):
        """물리 기반 시뮬레이션 한 스텝 실행"""
        current_speed_ms = self.state['speed'] / 3.6  # km/h → m/s
        target_speed_ms = target_speed / 3.6

        # 속도 오차
        speed_error = target_speed_ms - current_speed_ms

        # 제어 입력 계산 (PID 제어)
        if speed_error > 0.5:  # 가속 필요
            self.state['throttle'] = min(100.0, speed_error * 20)
            self.state['brake'] = 0.0
            desired_accel = min(accel_limit, speed_error * 2)
        elif speed_error < -0.5:  # 감속 필요
            self.state['throttle'] = 0.0
            self.state['brake'] = min(100.0, abs(speed_error) * 20)
            desired_accel = max(brake_limit, speed_error * 2)
        else:  # 일정 속도 유지
            self.state['throttle'] = 20.0
            self.state['brake'] = 0.0
            desired_accel = 0.0

        # 물리 법칙: F = ma, 공기저항, 구름저항
        drag_force = 0.5 * self.vehicle.drag_coefficient * 1.225 * (current_speed_ms ** 2) * 2.5
        rolling_force = self.vehicle.rolling_resistance * self.vehicle.mass * 9.81

        # 실제 가속도 (저항 고려)
        net_force = self.vehicle.mass * desired_accel - drag_force - rolling_force
        actual_accel = net_force / self.vehicle.mass

        # 상태 업데이트
        self.state['acceleration'] = actual_accel
        new_speed_ms = current_speed_ms + actual_accel * self.dt
        new_speed_ms = max(0.0, new_speed_ms)  # 음수 속도 방지

        self.state['speed'] = new_speed_ms * 3.6  # m/s → km/h
        self.state['position'] += new_speed_ms * self.dt

        # RPM 계산 (기어비 고려)
        # 가정: 5단 기어, 최종 기어비 3.5
        if self.state['speed'] < 1.0:
            self.state['rpm'] = 800.0  # idle
        else:
            # RPM = (속도 × 기어비 × 60) / (바퀴 둘레 × 기어단)
            gear = self._get_gear(self.state['speed'])
            gear_ratios = [3.5, 2.0, 1.3, 1.0, 0.8]
            self.state['rpm'] = (new_speed_ms * gear_ratios[gear-1] * 60) / (2.0 * np.pi * 0.3)
            self.state['rpm'] = np.clip(self.state['rpm'], 800, 6500)

        # 연료 소비 계산
        # 연료 소비 = (RPM × 스로틀) / 효율
        fuel_rate = (self.state['rpm'] / 1000.0) * (self.state['throttle'] / 100.0) * 0.001
        self.state['fuel_consumed'] += fuel_rate * self.dt
        self.state['fuel_level'] = max(0.0, 100.0 - (self.state['fuel_consumed'] / 50.0) * 100)

        # 냉각수 온도 (엔진 부하에 따라 변화)
        load = (self.state['rpm'] / 6500.0) * (self.state['throttle'] / 100.0)
        target_temp = 85.0 + load * 15.0  # 85-100°C
        temp_diff = target_temp - self.state['coolant_temp']
        self.state['coolant_temp'] += temp_diff * 0.01  # 서서히 변화

        # 조향각 (간단한 모델)
        self.state['steering_angle'] += np.random.normal(0, 2)
        self.state['steering_angle'] = np.clip(self.state['steering_angle'], -45, 45)

    def _get_gear(self, speed_kmh: float) -> int:
        """속도에 따른 기어 선택"""
        if speed_kmh < 20:
            return 1
        elif speed_kmh < 40:
            return 2
        elif speed_kmh < 60:
            return 3
        elif speed_kmh < 80:
            return 4
        else:
            return 5

    def _generate_sensor_data(self, add_noise: bool = True) -> Dict:
        """센서 데이터 생성 (OBD-II + IMU)"""
        data = {
            # OBD-II 데이터
            'vehicle_speed': self.state['speed'],  # km/h
            'engine_rpm': self.state['rpm'],
            'throttle_position': self.state['throttle'],
            'brake_pressure': self.state['brake'],
            'fuel_level': self.state['fuel_level'],
            'coolant_temp': self.state['coolant_temp'],

            # IMU 데이터
            'acceleration_x': self.state['acceleration'],  # 종방향
            'acceleration_y': np.random.normal(0, 0.1),  # 횡방향 (코너링)
            'acceleration_z': np.random.normal(0, 0.05),  # 수직 (범프)

            # 조향각
            'steering_angle': self.state['steering_angle'],

            # GPS (더미 데이터)
            'gps_lat': 37.5665 + np.random.normal(0, 0.0001),
            'gps_lon': 126.9780 + np.random.normal(0, 0.0001),

            # 연료 소비량 (계산값)
            'fuel_consumption': self._calculate_fuel_consumption(),
        }

        # 센서 노이즈 추가
        if add_noise:
            noise_levels = {
                'vehicle_speed': 0.5,  # ±0.5 km/h
                'engine_rpm': 50.0,  # ±50 RPM
                'throttle_position': 1.0,
                'brake_pressure': 2.0,
                'fuel_level': 0.5,
                'coolant_temp': 0.2,
                'acceleration_x': 0.1,
                'acceleration_y': 0.05,
                'acceleration_z': 0.02,
                'steering_angle': 0.5,
            }

            for key, noise_std in noise_levels.items():
                if key in data:
                    data[key] += np.random.normal(0, noise_std)

        return data

    def _calculate_fuel_consumption(self) -> float:
        """연료 소비량 계산 (L/100km)"""
        if self.state['speed'] < 1.0:
            return 0.0  # 정차 중

        # 연료 소비 = (MAF × 시간) / (거리 × 연료 밀도)
        # 간단한 모델: RPM과 스로틀에 비례
        maf = (self.state['rpm'] / 6500.0) * (self.state['throttle'] / 100.0) * 10.0  # g/s
        fuel_rate = maf / 14.7  # 공연비 14.7:1 (이론적)
        fuel_per_second = fuel_rate / 1000.0  # L/s

        distance_per_second = self.state['speed'] / 3600.0  # km/s
        fuel_per_100km = (fuel_per_second / distance_per_second) * 100.0 if distance_per_second > 0 else 0.0

        return np.clip(fuel_per_100km, 0, 30)  # 0-30 L/100km


# 사전 정의된 시나리오
PREDEFINED_SCENARIOS = {
    'eco_driving': DrivingScenario(
        name='Eco Driving',
        behavior=DrivingBehavior.ECO,
        duration_seconds=3600,  # 1시간
        target_speed_range=(40, 80),
        acceleration_limit=2.0,
        braking_limit=-2.5,
        throttle_pattern='smooth',
        brake_pattern='gentle',
        speed_variation=0.3
    ),
    'aggressive_driving': DrivingScenario(
        name='Aggressive Driving',
        behavior=DrivingBehavior.AGGRESSIVE,
        duration_seconds=1800,  # 30분
        target_speed_range=(50, 140),
        acceleration_limit=4.0,
        braking_limit=-6.0,
        throttle_pattern='aggressive',
        brake_pattern='aggressive',
        speed_variation=0.8
    ),
    'normal_driving': DrivingScenario(
        name='Normal Driving',
        behavior=DrivingBehavior.NORMAL,
        duration_seconds=7200,  # 2시간
        target_speed_range=(30, 100),
        acceleration_limit=3.0,
        braking_limit=-4.0,
        throttle_pattern='varying',
        brake_pattern='gentle',
        speed_variation=0.5
    ),
    'highway_cruising': DrivingScenario(
        name='Highway Cruising',
        behavior=DrivingBehavior.HIGHWAY,
        duration_seconds=3600,
        target_speed_range=(100, 120),
        acceleration_limit=2.0,
        braking_limit=-3.0,
        throttle_pattern='smooth',
        brake_pattern='gentle',
        speed_variation=0.2
    ),
    'urban_stop_go': DrivingScenario(
        name='Urban Stop-and-Go',
        behavior=DrivingBehavior.URBAN,
        duration_seconds=1800,
        target_speed_range=(0, 50),
        acceleration_limit=2.5,
        braking_limit=-4.0,
        throttle_pattern='varying',
        brake_pattern='aggressive',
        speed_variation=1.0
    ),
}


def generate_training_dataset(output_dir: str = 'datasets', total_samples: int = 35000):
    """
    학습용 데이터셋 생성

    Args:
        output_dir: 저장 디렉토리
        total_samples: 총 샘플 수 (기본 35,000)

    분포:
        - Eco: 30% (10,500)
        - Normal: 55% (19,250)
        - Aggressive: 15% (5,250)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulator = SyntheticDrivingSimulator(sampling_rate=1)  # 1 Hz

    datasets = []

    # Eco driving (10,500 samples = 10,500 seconds)
    print("Generating Eco driving data...")
    for i in range(3):  # 3,500초씩 3번
        scenario = PREDEFINED_SCENARIOS['eco_driving']
        scenario.duration_seconds = 3500
        df = simulator.generate_scenario(scenario)
        datasets.append(df)
        print(f"  Batch {i+1}/3: {len(df)} samples")

    # Normal driving (19,250 samples)
    print("Generating Normal driving data...")
    for i in range(3):  # 6,417초씩 3번
        scenario = PREDEFINED_SCENARIOS['normal_driving']
        scenario.duration_seconds = 6417
        df = simulator.generate_scenario(scenario)
        datasets.append(df)
        print(f"  Batch {i+1}/3: {len(df)} samples")

    # Aggressive driving (5,250 samples)
    print("Generating Aggressive driving data...")
    for i in range(3):  # 1,750초씩 3번
        scenario = PREDEFINED_SCENARIOS['aggressive_driving']
        scenario.duration_seconds = 1750
        df = simulator.generate_scenario(scenario)
        datasets.append(df)
        print(f"  Batch {i+1}/3: {len(df)} samples")

    # 합치기
    full_dataset = pd.concat(datasets, ignore_index=True)

    # 셔플
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # 저장
    train_size = int(len(full_dataset) * 0.8)
    val_size = int(len(full_dataset) * 0.1)

    train_df = full_dataset[:train_size]
    val_df = full_dataset[train_size:train_size+val_size]
    test_df = full_dataset[train_size+val_size:]

    train_path = output_path / 'train.csv'
    val_path = output_path / 'val.csv'
    test_path = output_path / 'test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n✅ Dataset generation complete!")
    print(f"  Train: {len(train_df)} samples ({train_path})")
    print(f"  Val:   {len(val_df)} samples ({val_path})")
    print(f"  Test:  {len(test_df)} samples ({test_path})")

    # 통계 정보
    print(f"\nLabel distribution:")
    print(full_dataset['label'].value_counts())

    return train_df, val_df, test_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic driving data')
    parser.add_argument('--output-dir', type=str, default='datasets',
                        help='Output directory for datasets')
    parser.add_argument('--samples', type=int, default=35000,
                        help='Total number of samples to generate')

    args = parser.parse_args()

    # 데이터 생성
    generate_training_dataset(args.output_dir, args.samples)
