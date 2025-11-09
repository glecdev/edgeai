/**
 * GLEC DTG - STM32 UART Protocol Header
 */

#ifndef __UART_PROTOCOL_H
#define __UART_PROTOCOL_H

#include "stm32f4xx_hal.h"

/* Vehicle Data structure */
typedef struct {
    uint64_t timestamp;
    float vehicleSpeed;
    uint32_t engineRPM;
    float throttlePosition;
    float brakePosition;
    float fuelLevel;
    float coolantTemp;
    float engineLoad;
    float intakeAirTemp;
    float mafRate;
    float batteryVoltage;
    float accelerationX;
    float accelerationY;
    float accelerationZ;
    float gyroX;
    float gyroY;
    float gyroZ;
    double gpsLatitude;
    double gpsLongitude;
    float gpsAltitude;
    float gpsSpeed;
    float gpsHeading;
} VehicleData;

/* UART Statistics */
typedef struct {
    uint32_t packetsSent;
    uint32_t packetsFailed;
} UARTStatistics;

/* Function prototypes */
HAL_StatusTypeDef UART_SendVehicleData(VehicleData *data);
void UART_GetStatistics(UARTStatistics *stats);
void UART_ResetStatistics(void);

#endif /* __UART_PROTOCOL_H */
