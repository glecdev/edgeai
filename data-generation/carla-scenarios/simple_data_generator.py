"""
GLEC DTG - Simple Synthetic Data Generator
Generate vehicle telemetry data WITHOUT external dependencies (no CARLA needed)

100% self-implemented, no external simulators required.
Uses physics-based models to generate realistic driving data.
"""

import argparse
import csv
import random
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class VehicleState:
    """Vehicle state at a given timestep"""
    timestamp: float
    position_x: float
    position_y: float
    velocity: float  # m/s
    acceleration: float  # m/s²
    steering_angle: float  # degrees
    throttle: float  # 0-1
    brake: float  # 0-1
    fuel_level: float  # percentage


class SimpleDrivingSimulator:
    """
    Physics-based driving simulator

    No external dependencies - pure Python implementation
    Simulates realistic vehicle dynamics and driving scenarios
    """

    def __init__(self, dt=1.0):
        """
        Args:
            dt: Time step in seconds (default 1.0 for 1Hz sampling)
        """
        self.dt = dt

        # Vehicle parameters
        self.mass = 1500  # kg
        self.max_acceleration = 3.0  # m/s²
        self.max_deceleration = 8.0  # m/s²
        self.max_speed = 180 / 3.6  # 180 km/h in m/s
        self.drag_coefficient = 0.3

        # Engine parameters
        self.idle_rpm = 800
        self.max_rpm = 6000

        # Fuel parameters
        self.fuel_capacity = 50  # liters
        self.base_fuel_consumption = 0.05  # L/s at idle

    def calculate_engine_rpm(self, velocity_ms, throttle):
        """Calculate engine RPM from velocity and throttle"""
        if velocity_ms < 0.1:
            return self.idle_rpm

        # Simple gear ratio simulation
        speed_rpm = (velocity_ms / (self.max_speed / 3.6)) * (self.max_rpm - self.idle_rpm)
        throttle_rpm = throttle * 1000

        rpm = self.idle_rpm + speed_rpm + throttle_rpm
        return min(rpm, self.max_rpm)

    def calculate_fuel_consumption(self, velocity_ms, throttle, brake):
        """Calculate instantaneous fuel consumption (L/s)"""
        # Base consumption
        consumption = self.base_fuel_consumption

        # Speed factor
        speed_kmh = velocity_ms * 3.6
        if speed_kmh > 0:
            speed_factor = 1.0 + (speed_kmh / 100) * 0.5
            consumption *= speed_factor

        # Throttle factor
        throttle_factor = 1.0 + throttle * 2.0
        consumption *= throttle_factor

        # Braking wastes fuel (engine braking)
        if brake > 0:
            consumption *= (1.0 + brake * 0.3)

        return consumption

    def calculate_acceleration(self, current_velocity, throttle, brake):
        """Calculate acceleration based on throttle and brake"""
        if brake > 0:
            # Braking
            return -self.max_deceleration * brake
        elif throttle > 0:
            # Accelerating
            # Reduce acceleration at high speeds (drag)
            speed_factor = max(0.1, 1.0 - (current_velocity / self.max_speed))
            return self.max_acceleration * throttle * speed_factor
        else:
            # Coasting (drag deceleration)
            drag = -0.01 * (current_velocity ** 2) * self.drag_coefficient
            return drag

    def simulate_driver_behavior(self, scenario, current_velocity, time_step):
        """
        Simulate different driving scenarios

        Returns: (throttle, brake, steering)
        """
        if scenario == 'highway_normal':
            # Maintain 100 km/h
            target_speed = 100 / 3.6
            speed_diff = target_speed - current_velocity

            if speed_diff > 0.5:
                throttle = min(0.6, speed_diff * 0.2)
                brake = 0.0
            elif speed_diff < -0.5:
                throttle = 0.0
                brake = min(0.3, abs(speed_diff) * 0.1)
            else:
                throttle = 0.3
                brake = 0.0

            # Slight steering variations
            steering = math.sin(time_step * 0.1) * 5

        elif scenario == 'city_normal':
            # Vary speed 30-60 km/h
            target_speed = (45 + 15 * math.sin(time_step * 0.05)) / 3.6
            speed_diff = target_speed - current_velocity

            if speed_diff > 0.5:
                throttle = min(0.5, speed_diff * 0.3)
                brake = 0.0
            else:
                throttle = 0.2
                brake = 0.1 if speed_diff < -1.0 else 0.0

            steering = math.sin(time_step * 0.2) * 10

        elif scenario == 'eco_driving':
            # Smooth acceleration, maintain 80 km/h
            target_speed = 80 / 3.6
            speed_diff = target_speed - current_velocity

            throttle = min(0.4, max(0, speed_diff * 0.1))
            brake = 0.0
            steering = math.sin(time_step * 0.05) * 3

        elif scenario == 'aggressive':
            # High throttle, hard braking
            if random.random() < 0.1:
                # Sudden braking
                throttle = 0.0
                brake = random.uniform(0.7, 1.0)
            else:
                # High throttle
                throttle = random.uniform(0.7, 1.0)
                brake = 0.0

            steering = random.uniform(-20, 20)

        elif scenario == 'traffic_jam':
            # Stop-and-go
            cycle = time_step % 30
            if cycle < 10:
                # Accelerating
                throttle = 0.3
                brake = 0.0
            elif cycle < 20:
                # Constant speed
                throttle = 0.1
                brake = 0.0
            else:
                # Braking to stop
                throttle = 0.0
                brake = 0.5

            steering = 0

        else:  # random
            throttle = random.uniform(0, 0.7)
            brake = random.uniform(0, 0.3) if throttle < 0.2 else 0.0
            steering = random.uniform(-15, 15)

        return throttle, brake, steering

    def classify_behavior(self, velocity_ms, acceleration, throttle, brake):
        """Classify driving behavior"""
        speed_kmh = velocity_ms * 3.6

        # Harsh braking
        if brake > 0.7 or acceleration < -4.0:
            return 'harsh_braking'

        # Harsh acceleration
        if throttle > 0.8 and acceleration > 3.0:
            return 'harsh_acceleration'

        # Eco driving
        if 40 < speed_kmh < 90 and abs(acceleration) < 1.0 and throttle < 0.5:
            return 'eco_driving'

        # Anomaly
        if speed_kmh > 140 or abs(acceleration) > 5.0:
            return 'anomaly'

        return 'normal'

    def generate_episode(self, scenario='highway_normal', duration=300):
        """
        Generate one driving episode

        Args:
            scenario: Driving scenario
            duration: Duration in seconds

        Returns:
            List of data samples
        """
        samples = []

        # Initialize vehicle state
        state = VehicleState(
            timestamp=0.0,
            position_x=0.0,
            position_y=0.0,
            velocity=0.0,
            acceleration=0.0,
            steering_angle=0.0,
            throttle=0.0,
            brake=0.0,
            fuel_level=random.uniform(30, 100)
        )

        for step in range(duration):
            # Simulate driver behavior
            throttle, brake, steering = self.simulate_driver_behavior(
                scenario, state.velocity, step
            )

            # Calculate acceleration
            acceleration = self.calculate_acceleration(state.velocity, throttle, brake)

            # Update velocity
            new_velocity = state.velocity + acceleration * self.dt
            new_velocity = max(0, min(new_velocity, self.max_speed))  # Clamp

            # Update position
            avg_velocity = (state.velocity + new_velocity) / 2
            state.position_x += avg_velocity * self.dt * math.cos(math.radians(state.steering_angle))
            state.position_y += avg_velocity * self.dt * math.sin(math.radians(state.steering_angle))

            # Calculate fuel consumption
            fuel_consumed = self.calculate_fuel_consumption(new_velocity, throttle, brake) * self.dt
            state.fuel_level -= (fuel_consumed / self.fuel_capacity) * 100
            state.fuel_level = max(0, state.fuel_level)

            # Calculate engine RPM
            engine_rpm = self.calculate_engine_rpm(new_velocity, throttle)

            # Classify behavior
            label = self.classify_behavior(new_velocity, acceleration, throttle, brake)

            # Create sample
            sample = {
                'timestamp': step,
                'vehicle_speed': new_velocity * 3.6,  # Convert to km/h
                'engine_rpm': engine_rpm,
                'throttle_position': throttle * 100,
                'brake_pressure': brake * 100,
                'fuel_level': state.fuel_level,
                'coolant_temp': random.uniform(85, 95),  # Simulated
                'acceleration_x': acceleration,
                'acceleration_y': random.uniform(-0.5, 0.5),  # Lateral accel
                'steering_angle': steering,
                'gps_lat': state.position_y / 100000,  # Simulated GPS
                'gps_lon': state.position_x / 100000,
                'fuel_consumption': fuel_consumed * 3600 / (new_velocity * 3.6 / 100) if new_velocity > 0 else 0,  # L/100km
                'label': label
            }

            samples.append(sample)

            # Update state
            state.velocity = new_velocity
            state.acceleration = acceleration
            state.steering_angle = steering
            state.throttle = throttle
            state.brake = brake
            state.timestamp = step

        return samples


def generate_mixed_scenarios(num_episodes=10, output_file='../datasets/synthetic_data.csv'):
    """Generate data from multiple scenarios"""
    simulator = SimpleDrivingSimulator(dt=1.0)

    scenarios = [
        ('highway_normal', 300),
        ('city_normal', 300),
        ('eco_driving', 300),
        ('aggressive', 200),
        ('traffic_jam', 200),
    ]

    all_samples = []

    print("🎬 Generating synthetic driving data...")
    print(f"Total episodes: {num_episodes}")
    print(f"Scenarios: {[s[0] for s in scenarios]}\n")

    for episode in range(num_episodes):
        # Randomly select scenario
        scenario, duration = random.choice(scenarios)

        print(f"Episode {episode+1}/{num_episodes}: {scenario} ({duration}s)")

        samples = simulator.generate_episode(scenario, duration)
        all_samples.extend(samples)

    print(f"\n✅ Generated {len(all_samples)} samples")

    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = all_samples[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)

    print(f"💾 Saved to: {output_path}")

    # Statistics
    print("\n📊 Dataset Statistics:")
    import pandas as pd
    df = pd.DataFrame(all_samples)

    print(f"  Total samples: {len(df)}")
    print(f"  Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")

    print(f"\n  Speed range: {df['vehicle_speed'].min():.1f} - {df['vehicle_speed'].max():.1f} km/h")
    print(f"  Mean speed: {df['vehicle_speed'].mean():.1f} km/h")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic vehicle data (no CARLA required)'
    )
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to generate')
    parser.add_argument('--output', default='../datasets/synthetic_data.csv',
                        help='Output CSV file')

    args = parser.parse_args()

    print("=" * 60)
    print("GLEC DTG - Simple Synthetic Data Generator")
    print("100% Self-Implemented - No External Dependencies")
    print("=" * 60)
    print()

    generate_mixed_scenarios(args.episodes, args.output)

    print("\n✅ Data generation completed!")
    print("This data can be used for model training without CARLA.")


if __name__ == "__main__":
    main()
