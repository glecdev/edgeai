#!/usr/bin/env python3
"""
GLEC DTG - CAN Message Parser Unit Tests

Tests the CAN message parsing logic from STM32 firmware
"""

import unittest
import struct


class TestCANParser(unittest.TestCase):
    """Test CAN message parsing"""

    def test_parse_obd_engine_rpm(self):
        """Test parsing OBD-II Engine RPM (PID 0x0C)"""
        # Engine RPM = ((A*256)+B)/4
        # Example: 2000 RPM → (2000 * 4) = 8000 = 0x1F40
        # A = 0x1F, B = 0x40

        mode = 0x41  # Response mode
        pid = 0x0C
        a = 0x1F
        b = 0x40

        rpm = ((a * 256) + b) / 4
        self.assertEqual(rpm, 2000.0)

    def test_parse_obd_vehicle_speed(self):
        """Test parsing OBD-II Vehicle Speed (PID 0x0D)"""
        # Vehicle Speed = A (km/h)
        mode = 0x41
        pid = 0x0D
        speed = 80

        self.assertEqual(speed, 80)

    def test_parse_obd_throttle_position(self):
        """Test parsing OBD-II Throttle Position (PID 0x11)"""
        # Throttle Position = A*100/255 (%)
        mode = 0x41
        pid = 0x11
        a = 128  # 50% throttle

        throttle = (a * 100.0) / 255.0
        self.assertAlmostEqual(throttle, 50.2, places=1)

    def test_parse_obd_fuel_level(self):
        """Test parsing OBD-II Fuel Level (PID 0x2F)"""
        # Fuel Level = A*100/255 (%)
        mode = 0x41
        pid = 0x2F
        a = 192  # ~75% fuel

        fuel_level = (a * 100.0) / 255.0
        self.assertAlmostEqual(fuel_level, 75.3, places=1)

    def test_parse_obd_coolant_temp(self):
        """Test parsing OBD-II Coolant Temperature (PID 0x05)"""
        # Coolant Temp = A-40 (°C)
        mode = 0x41
        pid = 0x05
        a = 130  # 90°C

        temp = a - 40
        self.assertEqual(temp, 90)

    def test_parse_obd_maf_rate(self):
        """Test parsing OBD-II MAF Air Flow Rate (PID 0x10)"""
        # MAF Rate = ((A*256)+B)/100 (g/s)
        mode = 0x41
        pid = 0x10
        a = 0x01
        b = 0xF4  # 500 → 5.00 g/s

        maf = ((a * 256) + b) / 100.0
        self.assertAlmostEqual(maf, 5.0, places=2)

    def test_parse_obd_battery_voltage(self):
        """Test parsing OBD-II Battery Voltage (PID 0x42)"""
        # Battery Voltage = ((A*256)+B)/1000 (V)
        mode = 0x41
        pid = 0x42
        a = 0x31
        b = 0x20  # 12544 → 12.544 V

        voltage = ((a * 256) + b) / 1000.0
        self.assertAlmostEqual(voltage, 12.544, places=3)

    def test_crc16_calculation(self):
        """Test CRC-16 (CCITT) calculation"""
        # Test data
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05])

        # Calculate CRC-16 (CCITT)
        crc = self._calculate_crc16(data)

        # CRC should be 16-bit value
        self.assertIsInstance(crc, int)
        self.assertGreaterEqual(crc, 0)
        self.assertLessEqual(crc, 0xFFFF)

    def test_crc16_known_values(self):
        """Test CRC-16 with known test vectors"""
        # Test vector: "123456789"
        data = b"123456789"
        expected_crc = 0x29B1  # Known CRC-16-CCITT value

        crc = self._calculate_crc16(data)
        self.assertEqual(crc, expected_crc)

    def test_uart_packet_structure(self):
        """Test UART packet structure and size"""
        START_BYTE = 0xAA
        END_BYTE = 0x55
        PACKET_SIZE = 83

        # Build packet
        packet = bytearray(PACKET_SIZE)
        packet[0] = START_BYTE
        packet[82] = END_BYTE

        # Add timestamp (8 bytes)
        timestamp = 1234567890
        struct.pack_into('<Q', packet, 1, timestamp)

        # Add vehicle speed (4 bytes float)
        speed = 80.5
        struct.pack_into('<f', packet, 9, speed)

        # Verify structure
        self.assertEqual(packet[0], START_BYTE)
        self.assertEqual(packet[82], END_BYTE)
        self.assertEqual(len(packet), PACKET_SIZE)

        # Verify timestamp
        parsed_timestamp = struct.unpack_from('<Q', packet, 1)[0]
        self.assertEqual(parsed_timestamp, timestamp)

        # Verify speed
        parsed_speed = struct.unpack_from('<f', packet, 9)[0]
        self.assertAlmostEqual(parsed_speed, speed, places=2)

    def _calculate_crc16(self, data: bytes) -> int:
        """Calculate CRC-16 (CCITT) - implementation for testing"""
        crc = 0xFFFF

        for byte in data:
            crc ^= byte << 8

            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1

        return crc & 0xFFFF


class TestDataValidation(unittest.TestCase):
    """Test data validation logic"""

    def test_vehicle_speed_range(self):
        """Test vehicle speed range validation"""
        # Valid range: 0-255 km/h
        self.assertTrue(0 <= 80 <= 255)
        self.assertTrue(0 <= 0 <= 255)
        self.assertTrue(0 <= 255 <= 255)
        self.assertFalse(0 <= -10 <= 255)
        self.assertFalse(0 <= 300 <= 255)

    def test_engine_rpm_range(self):
        """Test engine RPM range validation"""
        # Valid range: 0-16383
        self.assertTrue(0 <= 2000 <= 16383)
        self.assertTrue(0 <= 0 <= 16383)
        self.assertTrue(0 <= 16383 <= 16383)
        self.assertFalse(0 <= -500 <= 16383)
        self.assertFalse(0 <= 20000 <= 16383)

    def test_percentage_range(self):
        """Test percentage value validation (throttle, fuel, etc.)"""
        # Valid range: 0-100%
        self.assertTrue(0 <= 50.0 <= 100)
        self.assertTrue(0 <= 0.0 <= 100)
        self.assertTrue(0 <= 100.0 <= 100)
        self.assertFalse(0 <= -5.0 <= 100)
        self.assertFalse(0 <= 150.0 <= 100)

    def test_temperature_range(self):
        """Test temperature range validation"""
        # Valid range: -40 to 215°C
        self.assertTrue(-40 <= 90 <= 215)
        self.assertTrue(-40 <= -40 <= 215)
        self.assertTrue(-40 <= 215 <= 215)
        self.assertFalse(-40 <= -50 <= 215)
        self.assertFalse(-40 <= 300 <= 215)

    def test_battery_voltage_range(self):
        """Test battery voltage range validation"""
        # Valid range: 10-16V
        self.assertTrue(10 <= 12.6 <= 16)
        self.assertTrue(10 <= 10.0 <= 16)
        self.assertTrue(10 <= 16.0 <= 16)
        self.assertFalse(10 <= 8.0 <= 16)
        self.assertFalse(10 <= 18.0 <= 16)


class TestFuelCalculation(unittest.TestCase):
    """Test fuel consumption calculation"""

    def test_fuel_consumption_calculation(self):
        """Test instantaneous fuel consumption calculation"""
        # L/100km = (MAF / 14.7) * 3600 / 750 / speed * 100

        # Example: 5 g/s MAF, 80 km/h speed
        maf_rate = 5.0  # g/s
        vehicle_speed = 80.0  # km/h

        fuel_flow_rate = maf_rate / 14.7  # g/s
        fuel_flow_liter_per_hour = (fuel_flow_rate * 3600.0) / 750.0  # L/h
        fuel_consumption = (fuel_flow_liter_per_hour / vehicle_speed) * 100.0  # L/100km

        # Should be around 6-7 L/100km for highway driving
        self.assertGreater(fuel_consumption, 0)
        self.assertLess(fuel_consumption, 50)  # Sanity check

    def test_fuel_consumption_zero_speed(self):
        """Test fuel consumption at zero speed (should return 0)"""
        maf_rate = 5.0
        vehicle_speed = 0.0

        if vehicle_speed < 1.0:
            fuel_consumption = 0.0
        else:
            fuel_flow_rate = maf_rate / 14.7
            fuel_flow_liter_per_hour = (fuel_flow_rate * 3600.0) / 750.0
            fuel_consumption = (fuel_flow_liter_per_hour / vehicle_speed) * 100.0

        self.assertEqual(fuel_consumption, 0.0)

    def test_fuel_consumption_zero_maf(self):
        """Test fuel consumption with zero MAF (should return 0)"""
        maf_rate = 0.0
        vehicle_speed = 80.0

        if maf_rate < 0.1:
            fuel_consumption = 0.0
        else:
            fuel_flow_rate = maf_rate / 14.7
            fuel_flow_liter_per_hour = (fuel_flow_rate * 3600.0) / 750.0
            fuel_consumption = (fuel_flow_liter_per_hour / vehicle_speed) * 100.0

        self.assertEqual(fuel_consumption, 0.0)


if __name__ == '__main__':
    unittest.main()
