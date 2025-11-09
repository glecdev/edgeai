#!/usr/bin/env python3
"""
GLEC DTG Edge AI - End-to-End Integration Test

Tests the complete data flow:
CAN Bus → STM32 (UART) → Android (AI Inference) → MQTT → Fleet Platform

Usage:
    python e2e_test.py --duration 300 --verbose
"""

import argparse
import time
import serial
import struct
import sys
from typing import Optional, Dict, Any
import paho.mqtt.client as mqtt


class E2ETest:
    """End-to-end integration test"""

    def __init__(self, uart_port: str = '/dev/ttyUSB0', mqtt_broker: str = 'mqtt.glec.ai'):
        self.uart_port = uart_port
        self.mqtt_broker = mqtt_broker

        self.uart_connection: Optional[serial.Serial] = None
        self.mqtt_client: Optional[mqtt.Client] = None

        self.packets_received = 0
        self.packets_valid = 0
        self.packets_invalid = 0
        self.mqtt_messages_received = 0

        self.start_time = 0
        self.latest_inference_result: Optional[Dict[str, Any]] = None

    def setup(self):
        """Setup UART and MQTT connections"""
        print("Setting up E2E test...")

        # Open UART connection
        try:
            self.uart_connection = serial.Serial(
                port=self.uart_port,
                baudrate=921600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            print(f"✓ UART connected: {self.uart_port}")
        except serial.SerialException as e:
            print(f"✗ Failed to open UART: {e}")
            sys.exit(1)

        # Connect to MQTT broker
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message

            # Connect with timeout
            self.mqtt_client.connect(self.mqtt_broker, 8883, 60)
            self.mqtt_client.loop_start()

            # Wait for connection
            time.sleep(2)
            print(f"✓ MQTT connected: {self.mqtt_broker}")
        except Exception as e:
            print(f"✗ Failed to connect to MQTT: {e}")
            print("  (Continuing without MQTT monitoring)")
            self.mqtt_client = None

    def run(self, duration: int, verbose: bool = False):
        """Run E2E test for specified duration"""
        print(f"\nRunning E2E test for {duration} seconds...\n")

        self.start_time = time.time()
        end_time = self.start_time + duration

        last_summary_time = self.start_time

        while time.time() < end_time:
            # Read UART data packet
            packet = self._read_uart_packet(verbose)

            if packet:
                self.packets_received += 1

                # Validate packet
                if self._validate_packet(packet):
                    self.packets_valid += 1

                    if verbose:
                        self._print_packet_summary(packet)
                else:
                    self.packets_invalid += 1

            # Print summary every 10 seconds
            if time.time() - last_summary_time >= 10:
                self._print_progress()
                last_summary_time = time.time()

        print("\n" + "="*60)
        self._print_final_summary(duration)

    def _read_uart_packet(self, verbose: bool) -> Optional[bytes]:
        """Read one UART packet (83 bytes)"""
        PACKET_SIZE = 83
        START_BYTE = 0xAA
        END_BYTE = 0x55

        # Look for start byte
        while True:
            byte = self.uart_connection.read(1)
            if not byte:
                return None

            if byte[0] == START_BYTE:
                break

        # Read rest of packet
        packet_data = byte + self.uart_connection.read(PACKET_SIZE - 1)

        if len(packet_data) != PACKET_SIZE:
            if verbose:
                print(f"✗ Incomplete packet: {len(packet_data)} bytes")
            return None

        # Check end byte
        if packet_data[-1] != END_BYTE:
            if verbose:
                print(f"✗ Invalid end byte: 0x{packet_data[-1]:02X}")
            return None

        return packet_data

    def _validate_packet(self, packet: bytes) -> bool:
        """Validate packet CRC and structure"""
        # Extract CRC
        crc_received = struct.unpack('<H', packet[81:83])[0]

        # Calculate CRC
        crc_calculated = self._calculate_crc16(packet[1:81])

        if crc_calculated != crc_received:
            return False

        # Validate data ranges
        offset = 9  # After header and timestamp

        # Vehicle speed (0-255 km/h)
        vehicle_speed = struct.unpack('<f', packet[offset:offset+4])[0]
        if not (0 <= vehicle_speed <= 255):
            return False
        offset += 4

        # Engine RPM (0-16383)
        engine_rpm = struct.unpack('<I', packet[offset:offset+4])[0]
        if not (0 <= engine_rpm <= 16383):
            return False

        # Additional validations could be added here

        return True

    def _calculate_crc16(self, data: bytes) -> int:
        """Calculate CRC-16 (CCITT)"""
        crc = 0xFFFF

        for byte in data:
            crc ^= byte << 8

            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1

        return crc & 0xFFFF

    def _print_packet_summary(self, packet: bytes):
        """Print packet summary"""
        offset = 9

        # Extract key values
        vehicle_speed = struct.unpack('<f', packet[offset:offset+4])[0]
        offset += 4
        engine_rpm = struct.unpack('<I', packet[offset:offset+4])[0]
        offset += 4
        throttle = struct.unpack('<f', packet[offset:offset+4])[0]
        offset += 8  # Skip brake
        fuel_level = struct.unpack('<f', packet[offset:offset+4])[0]

        print(f"  Speed: {vehicle_speed:6.1f} km/h | "
              f"RPM: {engine_rpm:5d} | "
              f"Throttle: {throttle:5.1f}% | "
              f"Fuel: {fuel_level:5.1f}%")

    def _print_progress(self):
        """Print progress summary"""
        elapsed = time.time() - self.start_time
        rate = self.packets_received / elapsed if elapsed > 0 else 0

        print(f"[{elapsed:6.1f}s] Packets: {self.packets_received} "
              f"(Valid: {self.packets_valid}, Invalid: {self.packets_invalid}) "
              f"Rate: {rate:.2f} Hz | "
              f"MQTT: {self.mqtt_messages_received} msgs")

    def _print_final_summary(self, duration: int):
        """Print final test summary"""
        success_rate = (self.packets_valid / self.packets_received * 100) if self.packets_received > 0 else 0
        expected_packets = duration  # 1Hz rate

        print("E2E Test Summary")
        print("="*60)
        print(f"Duration:          {duration} seconds")
        print(f"Packets Received:  {self.packets_received} / {expected_packets} expected")
        print(f"  Valid:           {self.packets_valid} ({success_rate:.1f}%)")
        print(f"  Invalid:         {self.packets_invalid}")
        print(f"MQTT Messages:     {self.mqtt_messages_received}")

        # Check if we got expected 1Hz rate
        if self.packets_received >= expected_packets * 0.95:
            print("\n✓ PASS: Data collection rate within tolerance (>95%)")
        else:
            print(f"\n✗ FAIL: Data collection rate too low ({self.packets_received}/{expected_packets})")

        if success_rate >= 99.0:
            print("✓ PASS: Packet validation rate excellent (>99%)")
        elif success_rate >= 95.0:
            print("⚠ WARNING: Packet validation rate acceptable (>95%)")
        else:
            print("✗ FAIL: Too many invalid packets")

        if self.mqtt_messages_received > 0:
            print(f"✓ PASS: MQTT communication working ({self.mqtt_messages_received} messages)")
        else:
            print("✗ FAIL: No MQTT messages received")

        print("="*60)

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            # Subscribe to telemetry topic
            client.subscribe("fleet/vehicles/+/telemetry")
            print("✓ MQTT subscribed to telemetry topic")
        else:
            print(f"✗ MQTT connection failed: {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        self.mqtt_messages_received += 1

        # Could parse and validate AI inference results here
        # For now just count messages

    def cleanup(self):
        """Cleanup connections"""
        print("\nCleaning up...")

        if self.uart_connection:
            self.uart_connection.close()
            print("✓ UART closed")

        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("✓ MQTT disconnected")


def main():
    parser = argparse.ArgumentParser(description='GLEC DTG E2E Integration Test')
    parser.add_argument('--duration', type=int, default=60,
                        help='Test duration in seconds (default: 60)')
    parser.add_argument('--uart', type=str, default='/dev/ttyUSB0',
                        help='UART port (default: /dev/ttyUSB0)')
    parser.add_argument('--mqtt', type=str, default='mqtt.glec.ai',
                        help='MQTT broker address (default: mqtt.glec.ai)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Create and run test
    test = E2ETest(uart_port=args.uart, mqtt_broker=args.mqtt)

    try:
        test.setup()
        test.run(duration=args.duration, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        test.cleanup()


if __name__ == '__main__':
    main()
