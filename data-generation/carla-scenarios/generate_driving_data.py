"""
GLEC DTG - CARLA Driving Data Generation
Generate synthetic vehicle telemetry data for AI model training

Requirements:
- CARLA Simulator 0.9.13+
- Python 3.9+
- GPU: NVIDIA RTX 2070+ (8GB+ VRAM)
- RAM: 32GB+
"""

import argparse
import glob
import os
import sys
import time
import random
import csv
from datetime import datetime
from pathlib import Path

try:
    # Add CARLA Python API to path
    carla_path = os.environ.get('CARLA_ROOT', '/opt/carla')
    sys.path.append(glob.glob(f'{carla_path}/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("CARLA Python API not found. Please set CARLA_ROOT environment variable.")
    print("Example: export CARLA_ROOT=/path/to/carla")
    sys.exit(1)

import carla
import numpy as np


class VehicleDataGenerator:
    """
    Generate synthetic vehicle telemetry data using CARLA simulator

    Data format (1Hz sampling):
    - timestamp, vehicle_speed, engine_rpm, throttle_position, brake_pressure
    - fuel_level, coolant_temp, acceleration_x, acceleration_y, steering_angle
    - gps_lat, gps_lon, fuel_consumption, label
    """

    def __init__(self, host='127.0.0.1', port=2000, town='Town03'):
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensor_data = {}

        self.host = host
        self.port = port
        self.town = town

        # Simulated vehicle state
        self.fuel_level = 100.0  # Start at 100%
        self.coolant_temp = 85.0  # Normal operating temp
        self.engine_rpm_base = 800.0  # Idle RPM

    def connect(self):
        """Connect to CARLA server"""
        print(f"Connecting to CARLA server at {self.host}:{self.port}...")

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        # Load town
        print(f"Loading town: {self.town}")
        self.world = self.client.load_world(self.town)

        # Set synchronous mode for deterministic behavior
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0  # 1Hz simulation
        self.world.apply_settings(settings)

        print("✅ Connected to CARLA")

    def spawn_vehicle(self, vehicle_type='vehicle.tesla.model3'):
        """Spawn vehicle at random spawn point"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_type)[0]

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Enable autopilot
        self.vehicle.set_autopilot(True)

        print(f"✅ Spawned vehicle: {vehicle_type}")

    def set_weather(self, weather_preset='ClearNoon'):
        """Set weather conditions"""
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
        }

        if weather_preset == 'random':
            weather = random.choice(list(weather_presets.values()))
        else:
            weather = weather_presets.get(weather_preset, carla.WeatherParameters.ClearNoon)

        self.world.set_weather(weather)
        print(f"✅ Weather set: {weather_preset}")

    def set_traffic_density(self, num_vehicles=50, num_pedestrians=20):
        """Spawn traffic vehicles and pedestrians"""
        print(f"Spawning traffic: {num_vehicles} vehicles, {num_pedestrians} pedestrians")

        # Spawn vehicles
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')

        for i in range(min(num_vehicles, len(spawn_points))):
            try:
                bp = random.choice(vehicle_bps)
                vehicle = self.world.spawn_actor(bp, spawn_points[i])
                vehicle.set_autopilot(True)
            except Exception as e:
                continue

        print(f"✅ Traffic spawned")

    def calculate_engine_rpm(self, speed_kmh, throttle):
        """Calculate simulated engine RPM based on speed and throttle"""
        # Simple RPM simulation
        # Idle: 800 RPM
        # Max: 6000 RPM

        if speed_kmh < 1.0:
            return self.engine_rpm_base

        # RPM increases with speed and throttle
        speed_rpm = (speed_kmh / 120.0) * 5000  # Max speed → 5000 RPM
        throttle_rpm = throttle * 1000  # Throttle adds up to 1000 RPM

        rpm = self.engine_rpm_base + speed_rpm + throttle_rpm
        return min(rpm, 6000.0)

    def calculate_fuel_consumption(self, speed_kmh, throttle, brake):
        """Calculate simulated fuel consumption (L/100km)"""
        # Base consumption
        base_consumption = 5.0  # L/100km at optimal speed

        # Speed factor (consumption increases at very high/low speeds)
        if speed_kmh < 30:
            speed_factor = 1.5
        elif speed_kmh > 100:
            speed_factor = 1.3 + (speed_kmh - 100) * 0.01
        else:
            speed_factor = 1.0

        # Throttle factor
        throttle_factor = 1.0 + throttle * 0.5

        # Braking wastes fuel
        brake_factor = 1.0 + brake * 0.3

        consumption = base_consumption * speed_factor * throttle_factor * brake_factor

        # Update fuel level (decrease by small amount)
        self.fuel_level -= consumption * 0.001  # Simulated decrease
        self.fuel_level = max(self.fuel_level, 0.0)

        return consumption

    def classify_driving_behavior(self, speed_kmh, acceleration, throttle, brake):
        """Classify driving behavior for labels"""
        # Normal driving
        if abs(acceleration) < 2.0 and throttle < 0.7 and brake < 0.3:
            return 'normal'

        # Eco driving (smooth, moderate speed)
        if 40 < speed_kmh < 80 and abs(acceleration) < 1.0 and throttle < 0.5:
            return 'eco_driving'

        # Harsh braking
        if brake > 0.7 or acceleration < -4.0:
            return 'harsh_braking'

        # Harsh acceleration
        if throttle > 0.8 and acceleration > 3.0:
            return 'harsh_acceleration'

        # Anomaly (extreme values)
        if speed_kmh > 130 or abs(acceleration) > 5.0:
            return 'anomaly'

        return 'normal'

    def collect_data_sample(self):
        """Collect one data sample from vehicle"""
        if not self.vehicle:
            return None

        # Get vehicle transform and velocity
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        control = self.vehicle.get_control()
        acceleration = self.vehicle.get_acceleration()

        # Calculate speed in km/h
        speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        speed_kmh = speed_ms * 3.6

        # Calculate acceleration in m/s²
        accel_x = acceleration.x
        accel_y = acceleration.y

        # Get control inputs
        throttle = control.throttle  # 0-1
        brake = control.brake  # 0-1
        steering = control.steer  # -1 to 1

        # Calculate engine RPM
        engine_rpm = self.calculate_engine_rpm(speed_kmh, throttle)

        # Calculate fuel consumption
        fuel_consumption = self.calculate_fuel_consumption(speed_kmh, throttle, brake)

        # Classify behavior
        label = self.classify_driving_behavior(speed_kmh, accel_x, throttle, brake)

        # Create data sample
        sample = {
            'timestamp': time.time(),
            'vehicle_speed': speed_kmh,
            'engine_rpm': engine_rpm,
            'throttle_position': throttle * 100,  # Convert to percentage
            'brake_pressure': brake * 100,  # Convert to percentage
            'fuel_level': self.fuel_level,
            'coolant_temp': self.coolant_temp,
            'acceleration_x': accel_x,
            'acceleration_y': accel_y,
            'steering_angle': steering * 30,  # Convert to degrees (-30 to 30)
            'gps_lat': transform.location.x / 100000,  # Simulated GPS (CARLA coords)
            'gps_lon': transform.location.y / 100000,
            'fuel_consumption': fuel_consumption,
            'label': label
        }

        return sample

    def generate_episode(self, duration_seconds=300, output_file=None):
        """Generate one driving episode"""
        print(f"\n🎬 Starting episode (duration: {duration_seconds}s)")

        samples = []
        start_time = time.time()

        # Reset fuel level
        self.fuel_level = random.uniform(30, 100)

        try:
            for step in range(duration_seconds):
                # Tick simulation
                self.world.tick()

                # Collect data
                sample = self.collect_data_sample()
                if sample:
                    samples.append(sample)

                # Print progress
                if step % 60 == 0:
                    print(f"  Progress: {step}/{duration_seconds}s ({len(samples)} samples)")

        except KeyboardInterrupt:
            print("\n⚠️  Episode interrupted by user")

        elapsed = time.time() - start_time
        print(f"✅ Episode completed: {len(samples)} samples in {elapsed:.1f}s")

        # Save to CSV
        if output_file and samples:
            self.save_to_csv(samples, output_file)

        return samples

    def save_to_csv(self, samples, filename):
        """Save samples to CSV file"""
        fieldnames = [
            'timestamp', 'vehicle_speed', 'engine_rpm', 'throttle_position',
            'brake_pressure', 'fuel_level', 'coolant_temp', 'acceleration_x',
            'acceleration_y', 'steering_angle', 'gps_lat', 'gps_lon',
            'fuel_consumption', 'label'
        ]

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        print(f"💾 Saved {len(samples)} samples to: {filename}")

    def cleanup(self):
        """Cleanup CARLA resources"""
        if self.vehicle:
            self.vehicle.destroy()

        # Restore async mode
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        print("🧹 Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description='Generate driving data with CARLA')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--town', default='Town03', help='CARLA town')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--duration', type=int, default=300, help='Episode duration (seconds)')
    parser.add_argument('--weather', default='random', help='Weather preset')
    parser.add_argument('--traffic', default='dense', choices=['none', 'light', 'dense'],
                        help='Traffic density')
    parser.add_argument('--output', default='../datasets/carla_synthetic.csv',
                        help='Output CSV file')

    args = parser.parse_args()

    # Traffic density mapping
    traffic_map = {
        'none': (0, 0),
        'light': (20, 10),
        'dense': (50, 30)
    }
    num_vehicles, num_pedestrians = traffic_map[args.traffic]

    generator = VehicleDataGenerator(args.host, args.port, args.town)

    try:
        # Connect to CARLA
        generator.connect()

        # Set weather
        generator.set_weather(args.weather)

        # Spawn traffic
        generator.set_traffic_density(num_vehicles, num_pedestrians)

        # Spawn ego vehicle
        generator.spawn_vehicle()

        # Generate episodes
        all_samples = []

        for episode in range(args.episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode+1}/{args.episodes}")
            print(f"{'='*60}")

            samples = generator.generate_episode(
                duration_seconds=args.duration,
                output_file=None
            )

            all_samples.extend(samples)

            # Re-randomize weather for next episode
            if episode < args.episodes - 1:
                generator.set_weather('random')

        # Save all samples to single file
        generator.save_to_csv(all_samples, args.output)

        print(f"\n{'='*60}")
        print(f"✅ Data generation completed!")
        print(f"Total samples: {len(all_samples)}")
        print(f"Output file: {args.output}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()
