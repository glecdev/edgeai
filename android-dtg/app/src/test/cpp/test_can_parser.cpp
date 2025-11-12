/**
 * GLEC DTG - CAN Parser Unit Tests
 *
 * Production-grade tests for J1939 CAN parsing
 * Following TDD Red-Green-Refactor methodology
 */

#include <gtest/gtest.h>
#include "../../main/cpp/can_parser.cpp"

// ========================================
// J1939 Engine Speed/Torque Tests (PGN 61444)
// ========================================

TEST(J1939ParserTest, EngineSpeed_ValidData) {
    // Arrange: Valid engine data
    // RPM = 1500.0 (raw = 1500 / 0.125 = 12000 = 0x2EE0)
    // Torque = 50% (raw = 50 + 125 = 175 = 0xAF)
    uint8_t data[] = {0x00, 0x00, 0xAF, 0xE0, 0x2E, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939EngineData result = parseJ1939EngineSpeed(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.engineRPM, 1500.0f);
    EXPECT_FLOAT_EQ(result.engineTorquePercent, 50.0f);
}

TEST(J1939ParserTest, EngineSpeed_Idle) {
    // Arrange: Engine at idle (800 RPM)
    // RPM = 800.0 (raw = 800 / 0.125 = 6400 = 0x1900)
    // Torque = 0% (raw = 0 + 125 = 125 = 0x7D)
    uint8_t data[] = {0x00, 0x00, 0x7D, 0x00, 0x19, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939EngineData result = parseJ1939EngineSpeed(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.engineRPM, 800.0f);
    EXPECT_FLOAT_EQ(result.engineTorquePercent, 0.0f);
}

TEST(J1939ParserTest, EngineSpeed_HighRPM) {
    // Arrange: High RPM (4000 RPM)
    // RPM = 4000.0 (raw = 4000 / 0.125 = 32000 = 0x7D00)
    // Torque = 80% (raw = 80 + 125 = 205 = 0xCD)
    uint8_t data[] = {0x00, 0x00, 0xCD, 0x00, 0x7D, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939EngineData result = parseJ1939EngineSpeed(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.engineRPM, 4000.0f);
    EXPECT_FLOAT_EQ(result.engineTorquePercent, 80.0f);
}

TEST(J1939ParserTest, EngineSpeed_NotAvailable) {
    // Arrange: Engine data not available (0xFFFF)
    uint8_t data[] = {0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939EngineData result = parseJ1939EngineSpeed(data, dlc);

    // Assert
    EXPECT_FALSE(result.valid);
    EXPECT_FLOAT_EQ(result.engineRPM, 0.0f);
}

TEST(J1939ParserTest, EngineSpeed_InvalidDLC) {
    // Arrange: Insufficient data length
    uint8_t data[] = {0x00, 0x00, 0xAF};
    uint8_t dlc = 3;

    // Act
    J1939EngineData result = parseJ1939EngineSpeed(data, dlc);

    // Assert
    EXPECT_FALSE(result.valid);
}

TEST(J1939ParserTest, EngineSpeed_NegativeTorque) {
    // Arrange: Negative torque (engine braking)
    // RPM = 2000.0 (raw = 2000 / 0.125 = 16000 = 0x3E80)
    // Torque = -20% (raw = -20 + 125 = 105 = 0x69)
    uint8_t data[] = {0x00, 0x00, 0x69, 0x80, 0x3E, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939EngineData result = parseJ1939EngineSpeed(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.engineRPM, 2000.0f);
    EXPECT_FLOAT_EQ(result.engineTorquePercent, -20.0f);
}

// ========================================
// J1939 Vehicle Speed Tests (PGN 65265)
// ========================================

TEST(J1939ParserTest, VehicleSpeed_Highway) {
    // Arrange: Highway speed (100 km/h)
    // Speed = 100.0 (raw = 100 * 256 = 25600 = 0x6400)
    uint8_t data[] = {0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    float speed = parseJ1939VehicleSpeed(data, dlc);

    // Assert
    EXPECT_FLOAT_EQ(speed, 100.0f);
}

TEST(J1939ParserTest, VehicleSpeed_City) {
    // Arrange: City speed (50 km/h)
    // Speed = 50.0 (raw = 50 * 256 = 12800 = 0x3200)
    uint8_t data[] = {0x00, 0x00, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    float speed = parseJ1939VehicleSpeed(data, dlc);

    // Assert
    EXPECT_FLOAT_EQ(speed, 50.0f);
}

TEST(J1939ParserTest, VehicleSpeed_Stationary) {
    // Arrange: Vehicle stationary (0 km/h)
    uint8_t data[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    float speed = parseJ1939VehicleSpeed(data, dlc);

    // Assert
    EXPECT_FLOAT_EQ(speed, 0.0f);
}

TEST(J1939ParserTest, VehicleSpeed_NotAvailable) {
    // Arrange: Speed not available (0xFFFF)
    uint8_t data[] = {0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    float speed = parseJ1939VehicleSpeed(data, dlc);

    // Assert
    EXPECT_FLOAT_EQ(speed, 0.0f);
}

TEST(J1939ParserTest, VehicleSpeed_InvalidDLC) {
    // Arrange: Insufficient data
    uint8_t data[] = {0x00, 0x00};
    uint8_t dlc = 2;

    // Act
    float speed = parseJ1939VehicleSpeed(data, dlc);

    // Assert
    EXPECT_FLOAT_EQ(speed, 0.0f);
}

TEST(J1939ParserTest, VehicleSpeed_FractionalValue) {
    // Arrange: Fractional speed (50.5 km/h)
    // Speed = 50.5 (raw = 50.5 * 256 = 12928 = 0x3280)
    uint8_t data[] = {0x00, 0x80, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    float speed = parseJ1939VehicleSpeed(data, dlc);

    // Assert
    EXPECT_NEAR(speed, 50.5f, 0.01f);
}

// ========================================
// J1939 Fuel Consumption Tests (PGN 65262)
// ========================================

TEST(J1939ParserTest, FuelConsumption_Highway) {
    // Arrange: Highway cruising
    // Fuel rate = 15.0 L/h (raw = 15.0 / 0.05 = 300 = 0x012C)
    // Economy = 10.0 km/L (raw = 10.0 * 512 = 5120 = 0x1400)
    uint8_t data[] = {0x2C, 0x01, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939FuelData result = parseJ1939FuelConsumption(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.fuelRateLph, 15.0f);
    EXPECT_FLOAT_EQ(result.fuelEconomyKmpl, 10.0f);
}

TEST(J1939ParserTest, FuelConsumption_Idle) {
    // Arrange: Engine idling
    // Fuel rate = 1.0 L/h (raw = 1.0 / 0.05 = 20 = 0x0014)
    // Economy = 0.0 km/L (vehicle stationary)
    uint8_t data[] = {0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939FuelData result = parseJ1939FuelConsumption(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.fuelRateLph, 1.0f);
    EXPECT_FLOAT_EQ(result.fuelEconomyKmpl, 0.0f);
}

TEST(J1939ParserTest, FuelConsumption_HeavyLoad) {
    // Arrange: Heavy load (uphill, acceleration)
    // Fuel rate = 50.0 L/h (raw = 50.0 / 0.05 = 1000 = 0x03E8)
    // Economy = 5.0 km/L (raw = 5.0 * 512 = 2560 = 0x0A00)
    uint8_t data[] = {0xE8, 0x03, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939FuelData result = parseJ1939FuelConsumption(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.fuelRateLph, 50.0f);
    EXPECT_FLOAT_EQ(result.fuelEconomyKmpl, 5.0f);
}

TEST(J1939ParserTest, FuelConsumption_NotAvailable) {
    // Arrange: Fuel data not available (0xFFFF)
    uint8_t data[] = {0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939FuelData result = parseJ1939FuelConsumption(data, dlc);

    // Assert
    EXPECT_FALSE(result.valid);
}

TEST(J1939ParserTest, FuelConsumption_InvalidDLC) {
    // Arrange: Insufficient data
    uint8_t data[] = {0x2C, 0x01};
    uint8_t dlc = 2;

    // Act
    J1939FuelData result = parseJ1939FuelConsumption(data, dlc);

    // Assert
    EXPECT_FALSE(result.valid);
}

TEST(J1939ParserTest, FuelConsumption_PartialData) {
    // Arrange: Only fuel rate available, economy not available
    // Fuel rate = 20.0 L/h (raw = 20.0 / 0.05 = 400 = 0x0190)
    // Economy = not available (0xFFFF)
    uint8_t data[] = {0x90, 0x01, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939FuelData result = parseJ1939FuelConsumption(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.fuelRateLph, 20.0f);
    EXPECT_FLOAT_EQ(result.fuelEconomyKmpl, 0.0f);
}

// ========================================
// OBD-II Parser Tests (Baseline)
// ========================================

TEST(OBD2ParserTest, EngineRPM) {
    // Arrange: 2000 RPM = ((A * 256) + B) / 4
    // => A = 31, B = 64: ((31 * 256) + 64) / 4 = 2000
    uint8_t data[] = {31, 64};
    uint8_t dlc = 2;

    // Act
    float rpm = parseOBD2Response(OBD2_ENGINE_RPM, data, dlc);

    // Assert
    EXPECT_FLOAT_EQ(rpm, 2000.0f);
}

TEST(OBD2ParserTest, VehicleSpeed) {
    // Arrange: 80 km/h
    uint8_t data[] = {80};
    uint8_t dlc = 1;

    // Act
    float speed = parseOBD2Response(OBD2_VEHICLE_SPEED, data, dlc);

    // Assert
    EXPECT_FLOAT_EQ(speed, 80.0f);
}

TEST(OBD2ParserTest, ThrottlePosition) {
    // Arrange: 50% throttle = 50 * 255 / 100 = 127.5 ≈ 128
    uint8_t data[] = {128};
    uint8_t dlc = 1;

    // Act
    float throttle = parseOBD2Response(OBD2_THROTTLE_POSITION, data, dlc);

    // Assert
    EXPECT_NEAR(throttle, 50.2f, 0.5f);  // Allow small rounding error
}

TEST(OBD2ParserTest, CoolantTemperature) {
    // Arrange: 90°C = A - 40 => A = 130
    uint8_t data[] = {130};
    uint8_t dlc = 1;

    // Act
    float temp = parseOBD2Response(OBD2_COOLANT_TEMP, data, dlc);

    // Assert
    EXPECT_FLOAT_EQ(temp, 90.0f);
}

TEST(OBD2ParserTest, FuelLevel) {
    // Arrange: 75% = 75 * 255 / 100 = 191.25 ≈ 191
    uint8_t data[] = {191};
    uint8_t dlc = 1;

    // Act
    float fuelLevel = parseOBD2Response(OBD2_FUEL_LEVEL, data, dlc);

    // Assert
    EXPECT_NEAR(fuelLevel, 74.9f, 0.5f);  // Allow small rounding error
}

// ========================================
// Edge Case Tests
// ========================================

TEST(J1939ParserTest, EdgeCase_MaxEngineSpeed) {
    // Arrange: Maximum RPM (8031.875 rpm)
    // raw = 0xFFFE (0xFFFF is reserved for "not available")
    uint8_t data[] = {0x00, 0x00, 0x7D, 0xFE, 0xFF, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939EngineData result = parseJ1939EngineSpeed(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.engineRPM, 8031.75f);
}

TEST(J1939ParserTest, EdgeCase_MaxVehicleSpeed) {
    // Arrange: Maximum speed (250.996 km/h)
    // raw = 0xFFFE
    uint8_t data[] = {0x00, 0xFE, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    float speed = parseJ1939VehicleSpeed(data, dlc);

    // Assert
    EXPECT_NEAR(speed, 255.99f, 0.01f);
}

TEST(J1939ParserTest, EdgeCase_MaxFuelRate) {
    // Arrange: Maximum fuel rate (3212.75 L/h)
    // raw = 0xFFFE
    uint8_t data[] = {0xFE, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t dlc = 8;

    // Act
    J1939FuelData result = parseJ1939FuelConsumption(data, dlc);

    // Assert
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.fuelRateLph, 3212.70f);
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
