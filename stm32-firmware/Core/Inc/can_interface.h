/**
 * GLEC DTG - STM32 CAN Interface Header
 */

#ifndef __CAN_INTERFACE_H
#define __CAN_INTERFACE_H

#include "stm32f4xx_hal.h"

/* CAN Message structure */
typedef struct {
    uint32_t canId;
    uint8_t dlc;
    uint8_t data[8];
    uint32_t timestamp;
} CANMessage;

/* OBD-II Data structure */
typedef struct {
    uint16_t engineRPM;
    uint8_t vehicleSpeed;
    float throttlePosition;
    float fuelLevel;
    int8_t coolantTemp;
    float engineLoad;
    int8_t intakeAirTemp;
    float mafRate;
    float batteryVoltage;
    uint16_t validFlags;
} OBDData;

/* OBD-II valid flags */
#define OBD_FLAG_RPM            (1 << 0)
#define OBD_FLAG_SPEED          (1 << 1)
#define OBD_FLAG_THROTTLE       (1 << 2)
#define OBD_FLAG_FUEL           (1 << 3)
#define OBD_FLAG_COOLANT        (1 << 4)
#define OBD_FLAG_LOAD           (1 << 5)
#define OBD_FLAG_INTAKE_TEMP    (1 << 6)
#define OBD_FLAG_MAF            (1 << 7)
#define OBD_FLAG_BATTERY        (1 << 8)

/* CAN Statistics */
typedef struct {
    uint32_t messagesReceived;
    uint32_t messagesDropped;
    uint32_t crcErrors;
    uint16_t queueUsage;
} CANStatistics;

/* Function prototypes */
HAL_StatusTypeDef CAN_Init(void);
HAL_StatusTypeDef CAN_SendOBDRequest(uint8_t mode, uint8_t pid);
void CAN_RequestAllPIDs(void);
uint8_t CAN_DequeueMessage(CANMessage *msg);
uint8_t CAN_ParseOBDResponse(CANMessage *msg, OBDData *obdData);
void CAN_GetStatistics(CANStatistics *stats);
void CAN_ResetStatistics(void);

#endif /* __CAN_INTERFACE_H */
