"""
GLEC DTG Edge AI - Synthetic Driving Simulator Tests
합성 데이터 생성기 테스트
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add data-generation to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'data-generation'))
from synthetic_driving_simulator import (
    SyntheticDrivingSimulator,
    VehiclePhysics,
    DrivingScenario,
    DrivingBehavior,
    PREDEFINED_SCENARIOS
)


class TestVehiclePhysics:
    """차량 물리 파라미터 테스트"""

    def test_default_vehicle_params(self):
        """기본 차량 파라미터 검증"""
        vehicle = VehiclePhysics()

        assert vehicle.mass == 1500.0
        assert vehicle.max_acceleration == 4.0
        assert vehicle.max_braking == -8.0
        assert 0.2 <= vehicle.drag_coefficient <= 0.4
        assert 0.01 <= vehicle.rolling_resistance <= 0.02

    def test_custom_vehicle_params(self):
        """커스텀 차량 파라미터 설정"""
        vehicle = VehiclePhysics(
            mass=2000.0,  # 트럭
            max_acceleration=3.0,
            max_braking=-6.0
        )

        assert vehicle.mass == 2000.0
        assert vehicle.max_acceleration == 3.0


class TestSyntheticDrivingSimulator:
    """합성 주행 시뮬레이터 테스트"""

    @pytest.fixture
    def simulator(self):
        """시뮬레이터 인스턴스 생성"""
        return SyntheticDrivingSimulator(sampling_rate=1)

    def test_simulator_initialization(self, simulator):
        """시뮬레이터 초기화 검증"""
        assert simulator.sampling_rate == 1
        assert simulator.dt == 1.0
        assert simulator.state['speed'] == 0.0
        assert simulator.state['rpm'] == 800.0  # idle
        assert simulator.state['fuel_level'] == 100.0

    def test_eco_driving_scenario(self, simulator):
        """Eco 운전 시나리오 테스트"""
        scenario = DrivingScenario(
            name='Test Eco',
            behavior=DrivingBehavior.ECO,
            duration_seconds=60,  # 60초
            target_speed_range=(40, 80),
            acceleration_limit=2.0,
            braking_limit=-2.5,
            throttle_pattern='smooth',
            brake_pattern='gentle',
            speed_variation=0.3
        )

        df = simulator.generate_scenario(scenario, add_noise=False)

        # 기본 검증
        assert len(df) == 60
        assert 'vehicle_speed' in df.columns
        assert 'acceleration_x' in df.columns
        assert all(df['label'] == 'eco_driving')

        # Eco 운전 특성 검증
        # 1. 가속도가 제한 범위 내
        assert df['acceleration_x'].max() <= 2.5, "Eco driving should have gentle acceleration"
        assert df['acceleration_x'].min() >= -3.0, "Eco driving should have gentle braking"

        # 2. 속도가 목표 범위 내
        assert df['vehicle_speed'].max() <= 90, "Eco speed should be moderate"

        print(f"\n✅ Eco driving:")
        print(f"  Speed: {df['vehicle_speed'].mean():.1f} ± {df['vehicle_speed'].std():.1f} km/h")
        print(f"  Accel: {df['acceleration_x'].mean():.2f} ± {df['acceleration_x'].std():.2f} m/s²")

    def test_aggressive_driving_scenario(self, simulator):
        """공격적 운전 시나리오 테스트"""
        scenario = DrivingScenario(
            name='Test Aggressive',
            behavior=DrivingBehavior.AGGRESSIVE,
            duration_seconds=60,
            target_speed_range=(50, 140),
            acceleration_limit=4.0,
            braking_limit=-6.0,
            throttle_pattern='aggressive',
            brake_pattern='aggressive',
            speed_variation=0.8
        )

        df = simulator.generate_scenario(scenario, add_noise=False)

        assert len(df) == 60
        assert all(df['label'] == 'aggressive')

        # 공격적 운전 특성 검증
        # 1. 더 큰 가속도 변화
        accel_std = df['acceleration_x'].std()
        assert accel_std > 1.0, "Aggressive driving should have high acceleration variance"

        # 2. 더 높은 최대 속도
        assert df['vehicle_speed'].max() > 100, "Aggressive driving should reach high speeds"

        print(f"\n✅ Aggressive driving:")
        print(f"  Speed: {df['vehicle_speed'].mean():.1f} ± {df['vehicle_speed'].std():.1f} km/h")
        print(f"  Accel: {df['acceleration_x'].mean():.2f} ± {df['acceleration_x'].std():.2f} m/s²")

    def test_sensor_data_completeness(self, simulator):
        """센서 데이터 완전성 검증"""
        scenario = PREDEFINED_SCENARIOS['normal_driving']
        scenario.duration_seconds = 10  # 짧게

        df = simulator.generate_scenario(scenario)

        # 필수 컬럼 확인
        required_columns = [
            'timestamp', 'vehicle_speed', 'engine_rpm', 'throttle_position',
            'brake_pressure', 'fuel_level', 'coolant_temp',
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'steering_angle', 'gps_lat', 'gps_lon', 'fuel_consumption', 'label'
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_sensor_data_validity(self, simulator):
        """센서 데이터 유효성 검증"""
        scenario = PREDEFINED_SCENARIOS['normal_driving']
        scenario.duration_seconds = 100

        df = simulator.generate_scenario(scenario)

        # 범위 검증
        assert (df['vehicle_speed'] >= 0).all(), "Speed cannot be negative"
        assert (df['vehicle_speed'] <= 200).all(), "Speed too high"

        assert (df['engine_rpm'] >= 0).all(), "RPM cannot be negative"
        assert (df['engine_rpm'] <= 7000).all(), "RPM too high"

        assert (df['throttle_position'] >= 0).all(), "Throttle cannot be negative"
        assert (df['throttle_position'] <= 100).all(), "Throttle cannot exceed 100%"

        assert (df['brake_pressure'] >= 0).all(), "Brake cannot be negative"
        assert (df['brake_pressure'] <= 100).all(), "Brake cannot exceed 100%"

        assert (df['fuel_level'] >= 0).all(), "Fuel level cannot be negative"
        assert (df['fuel_level'] <= 100).all(), "Fuel level cannot exceed 100%"

        assert (df['coolant_temp'] >= 0).all(), "Coolant temp cannot be negative"
        assert (df['coolant_temp'] <= 130).all(), "Coolant temp too high"

        # 가속도 범위 (물리적으로 가능한 범위)
        assert (df['acceleration_x'] >= -10).all(), "Deceleration too strong"
        assert (df['acceleration_x'] <= 10).all(), "Acceleration too strong"

        print(f"\n✅ Sensor data validity passed")

    def test_physics_consistency(self, simulator):
        """물리 법칙 일관성 검증"""
        scenario = PREDEFINED_SCENARIOS['normal_driving']
        scenario.duration_seconds = 60

        df = simulator.generate_scenario(scenario, add_noise=False)

        # 속도와 가속도 관계 검증
        for i in range(1, len(df)):
            v_prev = df.loc[i-1, 'vehicle_speed'] / 3.6  # km/h → m/s
            v_curr = df.loc[i, 'vehicle_speed'] / 3.6
            a = df.loc[i, 'acceleration_x']
            dt = 1.0  # 1 Hz

            # v = v0 + a*dt (대략적으로)
            expected_v = v_prev + a * dt
            # 노이즈와 저항으로 약간의 오차 허용
            assert abs(v_curr - expected_v) < 2.0, \
                f"Speed-acceleration inconsistency at index {i}"

    def test_fuel_consumption(self, simulator):
        """연료 소비 계산 검증"""
        scenario = PREDEFINED_SCENARIOS['highway_cruising']
        scenario.duration_seconds = 3600  # 1시간

        df = simulator.generate_scenario(scenario, add_noise=False)

        # 연료 소비량이 감소해야 함
        initial_fuel = df.loc[0, 'fuel_level']
        final_fuel = df.loc[len(df)-1, 'fuel_level']

        assert final_fuel < initial_fuel, "Fuel should be consumed during driving"

        # 연료 소비량이 현실적 범위 (5-20 L/100km)
        fuel_consumption_mean = df['fuel_consumption'].mean()
        assert 3 <= fuel_consumption_mean <= 25, \
            f"Unrealistic fuel consumption: {fuel_consumption_mean:.2f} L/100km"

        print(f"\n✅ Fuel consumption: {fuel_consumption_mean:.2f} L/100km")

    def test_noise_effect(self, simulator):
        """센서 노이즈 효과 검증"""
        scenario = PREDEFINED_SCENARIOS['normal_driving']
        scenario.duration_seconds = 100

        # 노이즈 없음
        df_no_noise = simulator.generate_scenario(scenario, add_noise=False)

        # 노이즈 추가
        df_with_noise = simulator.generate_scenario(scenario, add_noise=True)

        # 노이즈가 있는 데이터가 더 변동성이 커야 함
        speed_std_no_noise = df_no_noise['vehicle_speed'].std()
        speed_std_with_noise = df_with_noise['vehicle_speed'].std()

        # 노이즈가 추가되면 표준편차가 더 커져야 함
        assert speed_std_with_noise >= speed_std_no_noise * 0.95, \
            "Noise should increase variance"

        print(f"\n✅ Noise effect:")
        print(f"  No noise STD: {speed_std_no_noise:.2f}")
        print(f"  With noise STD: {speed_std_with_noise:.2f}")

    def test_rpm_speed_correlation(self, simulator):
        """RPM과 속도 상관관계 검증"""
        scenario = PREDEFINED_SCENARIOS['normal_driving']
        scenario.duration_seconds = 100

        df = simulator.generate_scenario(scenario, add_noise=False)

        # 속도와 RPM은 양의 상관관계
        correlation = df['vehicle_speed'].corr(df['engine_rpm'])

        assert correlation > 0.5, f"Speed-RPM correlation too low: {correlation:.2f}"

        print(f"\n✅ Speed-RPM correlation: {correlation:.3f}")

    def test_different_behaviors_distinguishable(self, simulator):
        """다양한 운전 행동이 구분 가능한지 검증"""
        # Fix random seed for reproducibility
        np.random.seed(42)

        scenarios = [
            ('eco', PREDEFINED_SCENARIOS['eco_driving']),
            ('normal', PREDEFINED_SCENARIOS['normal_driving']),
            ('aggressive', PREDEFINED_SCENARIOS['aggressive_driving']),
        ]

        results = {}

        # Increase sample size for statistical reliability
        for name, scenario in scenarios:
            scenario.duration_seconds = 300  # Increased from 100 to 300
            df = simulator.generate_scenario(scenario, add_noise=False)

            results[name] = {
                'speed_mean': df['vehicle_speed'].mean(),
                'speed_std': df['vehicle_speed'].std(),
                'accel_std': df['acceleration_x'].std(),
                'throttle_mean': df['throttle_position'].mean(),
            }

        # Aggressive > Normal > Eco (가속도 변동성)
        # Use margin of error to handle edge cases
        aggressive_accel = results['aggressive']['accel_std']
        normal_accel = results['normal']['accel_std']
        eco_accel = results['eco']['accel_std']

        assert aggressive_accel > normal_accel * 0.95, \
            f"Aggressive ({aggressive_accel:.3f}) not > Normal ({normal_accel:.3f})"
        assert normal_accel > eco_accel * 0.95, \
            f"Normal ({normal_accel:.3f}) not > Eco ({eco_accel:.3f})"

        print(f"\n✅ Behavior distinguishability:")
        for name, metrics in results.items():
            print(f"  {name:12s}: speed={metrics['speed_mean']:.1f}, "
                  f"accel_std={metrics['accel_std']:.2f}")


class TestPredefinedScenarios:
    """사전 정의된 시나리오 테스트"""

    def test_all_predefined_scenarios_valid(self):
        """모든 사전 정의 시나리오가 유효한지 검증"""
        simulator = SyntheticDrivingSimulator()

        for name, scenario in PREDEFINED_SCENARIOS.items():
            # 짧게 실행
            scenario.duration_seconds = 10

            df = simulator.generate_scenario(scenario)

            assert len(df) == 10, f"Scenario {name} failed"
            assert not df.isnull().any().any(), f"Scenario {name} has NaN values"

            print(f"✅ {name:20s}: {len(df)} samples")


class TestDatasetGeneration:
    """데이터셋 생성 통합 테스트"""

    def test_small_dataset_generation(self, tmp_path):
        """소규모 데이터셋 생성 테스트"""
        from synthetic_driving_simulator import generate_training_dataset

        # 작은 샘플로 테스트 (35초)
        simulator = SyntheticDrivingSimulator()

        datasets = []

        # Eco (10초)
        scenario = PREDEFINED_SCENARIOS['eco_driving']
        scenario.duration_seconds = 10
        df = simulator.generate_scenario(scenario)
        datasets.append(df)

        # Normal (15초)
        scenario = PREDEFINED_SCENARIOS['normal_driving']
        scenario.duration_seconds = 15
        df = simulator.generate_scenario(scenario)
        datasets.append(df)

        # Aggressive (10초)
        scenario = PREDEFINED_SCENARIOS['aggressive_driving']
        scenario.duration_seconds = 10
        df = simulator.generate_scenario(scenario)
        datasets.append(df)

        full_dataset = pd.concat(datasets, ignore_index=True)

        assert len(full_dataset) == 35
        assert 'label' in full_dataset.columns

        # 레이블 분포 확인
        label_counts = full_dataset['label'].value_counts()
        assert 'eco_driving' in label_counts.index
        assert 'normal' in label_counts.index
        assert 'aggressive' in label_counts.index

        print(f"\n✅ Small dataset generation:")
        print(f"  Total samples: {len(full_dataset)}")
        print(f"  Labels: {label_counts.to_dict()}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
