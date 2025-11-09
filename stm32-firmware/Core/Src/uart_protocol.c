/**
 * GLEC DTG - STM32 UART Protocol
 * Handles UART communication with Snapdragon (921600 baud)
 *
 * Protocol format:
 * [START(0xAA)] [TIMESTAMP(8)] [VEHICLE_DATA(72)] [CRC16(2)] [END(0x55)]
 * Total: 83 bytes per packet
 *
 * Packet structure:
 * - Header: 1 byte (0xAA)
 * - Timestamp: 8 bytes (uint64_t milliseconds)
 * - Vehicle Speed: 4 bytes (float)
 * - Engine RPM: 4 bytes (uint32_t)
 * - Throttle Position: 4 bytes (float)
 * - Brake Position: 4 bytes (float)
 * - Fuel Level: 4 bytes (float)
 * - Coolant Temp: 4 bytes (float)
 * - Engine Load: 4 bytes (float)
 * - Intake Air Temp: 4 bytes (float)
 * - MAF Rate: 4 bytes (float)
 * - Battery Voltage: 4 bytes (float)
 * - Acceleration X/Y/Z: 12 bytes (3× float)
 * - Gyro X/Y/Z: 12 bytes (3× float)
 * - GPS Latitude: 8 bytes (double)
 * - GPS Longitude: 8 bytes (double)
 * - GPS Altitude: 4 bytes (float)
 * - GPS Speed: 4 bytes (float)
 * - GPS Heading: 4 bytes (float)
 * - CRC16: 2 bytes
 * - Footer: 1 byte (0x55)
 */

#include "uart_protocol.h"
#include "main.h"
#include <string.h>

/* External UART handle */
extern UART_HandleTypeDef huart1;

/* Protocol constants */
#define UART_START_BYTE     0xAA
#define UART_END_BYTE       0x55
#define UART_PACKET_SIZE    83

/* TX buffer */
static uint8_t txBuffer[UART_PACKET_SIZE];

/* Statistics */
static uint32_t packetsSent = 0;
static uint32_t packetsFailed = 0;

/**
 * Calculate CRC-16 (CCITT)
 * Polynomial: 0x1021
 */
static uint16_t UART_CalculateCRC16(uint8_t *data, uint16_t length)
{
    uint16_t crc = 0xFFFF;

    for (uint16_t i = 0; i < length; i++)
    {
        crc ^= (uint16_t)data[i] << 8;

        for (uint8_t j = 0; j < 8; j++)
        {
            if (crc & 0x8000)
            {
                crc = (crc << 1) ^ 0x1021;
            }
            else
            {
                crc <<= 1;
            }
        }
    }

    return crc;
}

/**
 * Write float to buffer (little-endian)
 */
static void UART_WriteFloat(uint8_t *buffer, uint16_t offset, float value)
{
    uint32_t intValue;
    memcpy(&intValue, &value, sizeof(float));

    buffer[offset + 0] = (intValue >> 0) & 0xFF;
    buffer[offset + 1] = (intValue >> 8) & 0xFF;
    buffer[offset + 2] = (intValue >> 16) & 0xFF;
    buffer[offset + 3] = (intValue >> 24) & 0xFF;
}

/**
 * Write uint32 to buffer (little-endian)
 */
static void UART_WriteUint32(uint8_t *buffer, uint16_t offset, uint32_t value)
{
    buffer[offset + 0] = (value >> 0) & 0xFF;
    buffer[offset + 1] = (value >> 8) & 0xFF;
    buffer[offset + 2] = (value >> 16) & 0xFF;
    buffer[offset + 3] = (value >> 24) & 0xFF;
}

/**
 * Write uint64 to buffer (little-endian)
 */
static void UART_WriteUint64(uint8_t *buffer, uint16_t offset, uint64_t value)
{
    buffer[offset + 0] = (value >> 0) & 0xFF;
    buffer[offset + 1] = (value >> 8) & 0xFF;
    buffer[offset + 2] = (value >> 16) & 0xFF;
    buffer[offset + 3] = (value >> 24) & 0xFF;
    buffer[offset + 4] = (value >> 32) & 0xFF;
    buffer[offset + 5] = (value >> 40) & 0xFF;
    buffer[offset + 6] = (value >> 48) & 0xFF;
    buffer[offset + 7] = (value >> 56) & 0xFF;
}

/**
 * Write double to buffer (little-endian)
 */
static void UART_WriteDouble(uint8_t *buffer, uint16_t offset, double value)
{
    uint64_t intValue;
    memcpy(&intValue, &value, sizeof(double));

    buffer[offset + 0] = (intValue >> 0) & 0xFF;
    buffer[offset + 1] = (intValue >> 8) & 0xFF;
    buffer[offset + 2] = (intValue >> 16) & 0xFF;
    buffer[offset + 3] = (intValue >> 24) & 0xFF;
    buffer[offset + 4] = (intValue >> 32) & 0xFF;
    buffer[offset + 5] = (intValue >> 40) & 0xFF;
    buffer[offset + 6] = (intValue >> 48) & 0xFF;
    buffer[offset + 7] = (intValue >> 56) & 0xFF;
}

/**
 * Send vehicle data packet via UART
 */
HAL_StatusTypeDef UART_SendVehicleData(VehicleData *data)
{
    uint16_t offset = 0;

    /* Header */
    txBuffer[offset++] = UART_START_BYTE;

    /* Timestamp */
    UART_WriteUint64(txBuffer, offset, data->timestamp);
    offset += 8;

    /* Vehicle speed */
    UART_WriteFloat(txBuffer, offset, data->vehicleSpeed);
    offset += 4;

    /* Engine RPM */
    UART_WriteUint32(txBuffer, offset, data->engineRPM);
    offset += 4;

    /* Throttle position */
    UART_WriteFloat(txBuffer, offset, data->throttlePosition);
    offset += 4;

    /* Brake position */
    UART_WriteFloat(txBuffer, offset, data->brakePosition);
    offset += 4;

    /* Fuel level */
    UART_WriteFloat(txBuffer, offset, data->fuelLevel);
    offset += 4;

    /* Coolant temperature */
    UART_WriteFloat(txBuffer, offset, data->coolantTemp);
    offset += 4;

    /* Engine load */
    UART_WriteFloat(txBuffer, offset, data->engineLoad);
    offset += 4;

    /* Intake air temperature */
    UART_WriteFloat(txBuffer, offset, data->intakeAirTemp);
    offset += 4;

    /* MAF rate */
    UART_WriteFloat(txBuffer, offset, data->mafRate);
    offset += 4;

    /* Battery voltage */
    UART_WriteFloat(txBuffer, offset, data->batteryVoltage);
    offset += 4;

    /* Acceleration X/Y/Z */
    UART_WriteFloat(txBuffer, offset, data->accelerationX);
    offset += 4;
    UART_WriteFloat(txBuffer, offset, data->accelerationY);
    offset += 4;
    UART_WriteFloat(txBuffer, offset, data->accelerationZ);
    offset += 4;

    /* Gyro X/Y/Z */
    UART_WriteFloat(txBuffer, offset, data->gyroX);
    offset += 4;
    UART_WriteFloat(txBuffer, offset, data->gyroY);
    offset += 4;
    UART_WriteFloat(txBuffer, offset, data->gyroZ);
    offset += 4;

    /* GPS latitude */
    UART_WriteDouble(txBuffer, offset, data->gpsLatitude);
    offset += 8;

    /* GPS longitude */
    UART_WriteDouble(txBuffer, offset, data->gpsLongitude);
    offset += 8;

    /* GPS altitude */
    UART_WriteFloat(txBuffer, offset, data->gpsAltitude);
    offset += 4;

    /* GPS speed */
    UART_WriteFloat(txBuffer, offset, data->gpsSpeed);
    offset += 4;

    /* GPS heading */
    UART_WriteFloat(txBuffer, offset, data->gpsHeading);
    offset += 4;

    /* Calculate CRC (over all data except header, CRC, and footer) */
    uint16_t crc = UART_CalculateCRC16(txBuffer + 1, offset - 1);
    txBuffer[offset++] = (crc >> 8) & 0xFF;  // CRC high byte
    txBuffer[offset++] = crc & 0xFF;         // CRC low byte

    /* Footer */
    txBuffer[offset++] = UART_END_BYTE;

    /* Send packet */
    HAL_StatusTypeDef status = HAL_UART_Transmit(&huart1, txBuffer, UART_PACKET_SIZE, 100);

    if (status == HAL_OK)
    {
        packetsSent++;
    }
    else
    {
        packetsFailed++;
    }

    return status;
}

/**
 * Get UART statistics
 */
void UART_GetStatistics(UARTStatistics *stats)
{
    stats->packetsSent = packetsSent;
    stats->packetsFailed = packetsFailed;
}

/**
 * Reset UART statistics
 */
void UART_ResetStatistics(void)
{
    packetsSent = 0;
    packetsFailed = 0;
}
