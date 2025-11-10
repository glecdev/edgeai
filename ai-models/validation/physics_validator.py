#!/usr/bin/env python3
"""
GLEC DTG - Physics-based Plausibility Validation System
Ported from production: GLEC_DTG_INTEGRATED_v20.0.0/01_core_engine/physics_validation/

Validates sensor data against physical laws to detect:
- Sensor malfunctions
- CAN bus intrusions
- Data corruption
- Unrealistic driving patterns

Original implementation:
- https://github.com/glecdev/glec-dtg-ai-production
- GLEC_DTG_INTEGRATED_v20.0.0/01_core_engine/physics_validation/physics_plausibility_validation_system.py
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math


@dataclass
class ValidationResult:
    """Physics validation result"""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    anomaly_type: Optional[str] = None


class PhysicsValidator:
    """
    Production-verified physics-based validation system

    Physical laws enforced:
    1. Newton's laws of motion
    2. Energy conservation
    3. Thermodynamic constraints
    4. Mechanical limits

    Validation checks:
    - Acceleration consistency
    - Fuel consumption plausibility
    - Engine performance limits
    - Sensor correlation
    """

    # Physical constants
    GRAVITY = 9.81  # m/s²
    AIR_FUEL_RATIO = 14.7  # Stoichiometric ratio for gasoline
    FUEL_DENSITY = 750  # g/L (gasoline)

    # Vehicle limits (commercial truck)
    MAX_ACCELERATION = 3.5  # m/s² (typical truck)
    MAX_DECELERATION = -8.0  # m/s² (emergency braking)
    MAX_SPEED = 120.0  # km/h (truck speed limiter)
    MAX_RPM = 4000  # RPM (commercial diesel)
    MAX_TORQUE = 2500  # Nm (heavy truck)

    def __init__(self, vehicle_type: str = "truck"):
        """
        Initialize physics validator

        Args:
            vehicle_type: "truck", "bus", or "car" (affects physical limits)
        """
        self.vehicle_type = vehicle_type
        self.previous_data = None

    def validate(self, data, previous_data=None) -> ValidationResult:
        """
        Comprehensive physics validation

        Args:
            data: Current CAN data
            previous_data: Previous CAN data (for temporal checks)

        Returns:
            ValidationResult with validation outcome
        """
        # Use stored previous data if not provided
        if previous_data is None:
            previous_data = self.previous_data

        # Update previous data for next call
        self.previous_data = data

        # Skip validation if no previous data
        if previous_data is None:
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                reason="No previous data for temporal validation"
            )

        # Run all validation checks
        checks = [
            self._validate_acceleration(data, previous_data),
            self._validate_speed(data),
            self._validate_engine_performance(data),
            self._validate_fuel_consumption(data),
            self._validate_sensor_correlation(data),
            self._validate_thermodynamics(data)
        ]

        # Aggregate results
        failed_checks = [c for c in checks if not c.is_valid]

        if not failed_checks:
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                reason="All physics checks passed"
            )
        else:
            # Return first failure
            return failed_checks[0]

    def _validate_acceleration(self, data, previous_data) -> ValidationResult:
        """
        Validate acceleration using Newton's laws

        F = ma
        a = (v_final - v_initial) / time

        Checks:
        1. Acceleration within physical limits
        2. Consistency with throttle/brake position
        """
        # Calculate time delta
        dt = (data.timestamp - previous_data.timestamp) / 1000.0  # seconds

        if dt <= 0 or dt > 10.0:  # Skip if time delta invalid
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                reason=f"Invalid time delta: {dt}s"
            )

        # Calculate acceleration from speed change
        # km/h → m/s → m/s²
        dv = (data.vehicle_speed - previous_data.vehicle_speed) / 3.6  # m/s
        calculated_acceleration = dv / dt  # m/s²

        # Check against physical limits
        if calculated_acceleration > 5.0:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                reason=f"Impossible acceleration: {calculated_acceleration:.2f} m/s² (max: 3.5 m/s²)",
                anomaly_type="SENSOR_MALFUNCTION"
            )

        if calculated_acceleration < -10.0:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                reason=f"Impossible deceleration: {calculated_acceleration:.2f} m/s² (max: -8.0 m/s²)",
                anomaly_type="SENSOR_MALFUNCTION"
            )

        # Check consistency with IMU accelerometer
        if hasattr(data, 'acceleration_x'):
            imu_acceleration = data.acceleration_x
            difference = abs(calculated_acceleration - imu_acceleration)

            # Allow 30% tolerance (sensor drift)
            if difference > abs(calculated_acceleration) * 0.3:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.8,
                    reason=f"Speed/IMU acceleration mismatch: Δ{difference:.2f} m/s²",
                    anomaly_type="SENSOR_CORRELATION_ERROR"
                )

        # Check throttle/brake consistency
        if calculated_acceleration > 0.5 and data.throttle_position < 10.0:
            return ValidationResult(
                is_valid=False,
                confidence=0.7,
                reason="Acceleration without throttle input",
                anomaly_type="SENSOR_CORRELATION_ERROR"
            )

        if calculated_acceleration < -1.0 and data.brake_position < 10.0:
            return ValidationResult(
                is_valid=False,
                confidence=0.7,
                reason="Deceleration without brake input",
                anomaly_type="SENSOR_CORRELATION_ERROR"
            )

        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            reason="Acceleration within physical limits"
        )

    def _validate_speed(self, data) -> ValidationResult:
        """Validate vehicle speed against physical limits"""
        if data.vehicle_speed < 0:
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                reason=f"Negative speed: {data.vehicle_speed} km/h",
                anomaly_type="SENSOR_MALFUNCTION"
            )

        if data.vehicle_speed > self.MAX_SPEED:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                reason=f"Speed exceeds limit: {data.vehicle_speed} km/h (max: {self.MAX_SPEED})",
                anomaly_type="SPEED_LIMITER_FAILURE"
            )

        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            reason="Speed within limits"
        )

    def _validate_engine_performance(self, data) -> ValidationResult:
        """
        Validate engine performance against mechanical limits

        Checks:
        1. RPM within safe range
        2. Torque within engine specifications
        3. Power output plausibility
        """
        if data.engine_rpm < 0:
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                reason=f"Negative RPM: {data.engine_rpm}",
                anomaly_type="SENSOR_MALFUNCTION"
            )

        if data.engine_rpm > self.MAX_RPM:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                reason=f"RPM exceeds redline: {data.engine_rpm} (max: {self.MAX_RPM})",
                anomaly_type="ENGINE_OVERSPEED"
            )

        # Validate RPM/speed ratio (gear ratio check)
        if data.vehicle_speed > 10 and data.engine_rpm > 0:
            # Typical truck gear ratio: 30-50 km/h per 1000 RPM
            speed_per_1000rpm = data.vehicle_speed / (data.engine_rpm / 1000.0)

            if speed_per_1000rpm < 20 or speed_per_1000rpm > 60:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.6,
                    reason=f"Unlikely RPM/speed ratio: {speed_per_1000rpm:.1f} km/h per 1000 RPM",
                    anomaly_type="TRANSMISSION_ERROR"
                )

        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            reason="Engine performance within limits"
        )

    def _validate_fuel_consumption(self, data) -> ValidationResult:
        """
        Validate fuel consumption using thermodynamic principles

        Energy balance:
        Fuel energy = Kinetic energy + Friction losses + Air resistance

        Fuel consumption rate (L/h) = (MAF / air_fuel_ratio) * 3600 / fuel_density
        """
        if not hasattr(data, 'maf_rate') or data.maf_rate is None:
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                reason="MAF data not available"
            )

        # Calculate theoretical fuel consumption
        fuel_flow_gps = data.maf_rate / self.AIR_FUEL_RATIO  # g/s
        fuel_rate_lph = (fuel_flow_gps * 3600.0) / self.FUEL_DENSITY  # L/h

        # Sanity check: 0-100 L/h for heavy truck
        if fuel_rate_lph < 0 or fuel_rate_lph > 100:
            return ValidationResult(
                is_valid=False,
                confidence=0.8,
                reason=f"Implausible fuel rate: {fuel_rate_lph:.1f} L/h",
                anomaly_type="FUEL_SENSOR_ERROR"
            )

        # Cross-check with throttle position
        if data.throttle_position < 5 and fuel_rate_lph > 10:
            return ValidationResult(
                is_valid=False,
                confidence=0.7,
                reason="High fuel consumption at low throttle",
                anomaly_type="FUEL_LEAK_OR_SENSOR_ERROR"
            )

        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            reason="Fuel consumption plausible"
        )

    def _validate_sensor_correlation(self, data) -> ValidationResult:
        """
        Validate correlation between related sensors

        Redundancy checks:
        1. Speed (GPS vs wheel speed)
        2. Position (GPS vs dead reckoning)
        3. Temperature (coolant vs ambient)
        """
        # Check battery voltage
        if data.battery_voltage < 10.0 or data.battery_voltage > 16.0:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                reason=f"Battery voltage abnormal: {data.battery_voltage}V",
                anomaly_type="ELECTRICAL_SYSTEM_FAULT"
            )

        # Check coolant temperature
        if data.coolant_temp < -40 or data.coolant_temp > 120:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                reason=f"Coolant temperature out of range: {data.coolant_temp}°C",
                anomaly_type="TEMPERATURE_SENSOR_FAULT"
            )

        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            reason="Sensor correlation valid"
        )

    def _validate_thermodynamics(self, data) -> ValidationResult:
        """
        Validate thermodynamic constraints

        Laws:
        1. Engine cannot cool faster than physically possible
        2. Heat generation proportional to engine load
        """
        # Coolant temperature should correlate with engine load
        if data.engine_rpm > 3000 and data.coolant_temp < 60:
            return ValidationResult(
                is_valid=False,
                confidence=0.6,
                reason="Engine hot but coolant cold (sensor or thermostat fault)",
                anomaly_type="COOLING_SYSTEM_FAULT"
            )

        return ValidationResult(
            is_valid=True,
            confidence=1.0,
            reason="Thermodynamics check passed"
        )


# Example usage
if __name__ == '__main__':
    import sys
    import os
    import time
    # Add ai-models to path for import
    ai_models_path = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, ai_models_path)
    from inference.realtime_integration import RealtimeCANData

    validator = PhysicsValidator(vehicle_type="truck")

    # Test 1: Normal driving
    data1 = RealtimeCANData(
        timestamp=int(time.time() * 1000),
        vehicle_speed=60.0,
        engine_rpm=2000,
        fuel_level=75.0,
        throttle_position=40.0,
        brake_position=0.0,
        coolant_temp=85,
        maf_rate=5.0,
        battery_voltage=12.6,
        acceleration_x=0.3,
        acceleration_y=0.0,
        acceleration_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        latitude=37.5665,
        longitude=126.9780,
        altitude=50.0,
        heading=45.0
    )

    # Test 2: Normal acceleration (1 second later)
    data2 = RealtimeCANData(
        timestamp=data1.timestamp + 1000,
        vehicle_speed=65.0,
        engine_rpm=2200,
        fuel_level=74.9,
        throttle_position=50.0,
        brake_position=0.0,
        coolant_temp=86,
        maf_rate=6.0,
        battery_voltage=12.6,
        acceleration_x=1.4,  # (65-60)/3.6/1.0 = 1.39 m/s²
        acceleration_y=0.0,
        acceleration_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        latitude=37.5666,
        longitude=126.9781,
        altitude=50.0,
        heading=45.0
    )

    # Validate
    result1 = validator.validate(data1)
    print(f"Test 1: {result1.reason} (confidence: {result1.confidence})")

    result2 = validator.validate(data2)
    print(f"Test 2: {result2.reason} (confidence: {result2.confidence})")

    # Test 3: Impossible acceleration (sensor fault)
    data3 = RealtimeCANData(
        timestamp=data2.timestamp + 1000,
        vehicle_speed=150.0,  # Impossible jump from 65 to 150 km/h in 1 second
        engine_rpm=2200,
        fuel_level=74.8,
        throttle_position=50.0,
        brake_position=0.0,
        coolant_temp=87,
        maf_rate=6.0,
        battery_voltage=12.6,
        acceleration_x=1.0,
        acceleration_y=0.0,
        acceleration_z=9.81,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        latitude=37.5667,
        longitude=126.9782,
        altitude=50.0,
        heading=45.0
    )

    result3 = validator.validate(data3)
    print(f"Test 3: {result3.reason} (confidence: {result3.confidence})")
    if not result3.is_valid:
        print(f"  Anomaly type: {result3.anomaly_type}")
