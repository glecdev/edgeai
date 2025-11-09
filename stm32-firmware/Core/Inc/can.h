/**
  ******************************************************************************
  * @file           : can.h
  * @brief          : Header for can.c file - GLEC DTG
  ******************************************************************************
  */

#ifndef __CAN_H
#define __CAN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Exported variables --------------------------------------------------------*/
extern CAN_HandleTypeDef hcan1;

/* Exported functions --------------------------------------------------------*/
void MX_CAN1_Init(void);
void CAN_ParseOBD2Response(uint8_t pid, uint8_t *data, float *value);

#ifdef __cplusplus
}
#endif

#endif /* __CAN_H */
