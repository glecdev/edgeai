/**
 * GLEC DTG - STM32 Application Header
 */

#ifndef __DTG_APP_H
#define __DTG_APP_H

#include "stm32f4xx_hal.h"
#include "can_interface.h"
#include "uart_protocol.h"

/* IMU Data structure */
typedef struct {
    float accelX;      // m/s²
    float accelY;      // m/s²
    float accelZ;      // m/s²
    float gyroX;       // °/s
    float gyroY;       // °/s
    float gyroZ;       // °/s
} IMUData;

/* GPS Data structure */
typedef struct {
    double latitude;
    double longitude;
    float altitude;
    float speed;
    float heading;
    uint8_t fixQuality;
    uint8_t satelliteCount;
} GPSData;

/* Function prototypes */
void DTG_Init(void);
void DTG_Loop(void);

/* Helper functions */
HAL_StatusTypeDef IMU_Init(void);
void IMU_Read(IMUData *data);

HAL_StatusTypeDef GPS_Init(void);
void GPS_Read(GPSData *data);

#endif /* __DTG_APP_H */
