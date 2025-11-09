/**
  ******************************************************************************
  * @file           : uart.h
  * @brief          : Header for uart.c file - GLEC DTG
  ******************************************************************************
  */

#ifndef __UART_H
#define __UART_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Exported variables --------------------------------------------------------*/
extern UART_HandleTypeDef huart2;

/* Exported functions --------------------------------------------------------*/
void MX_UART2_Init(void);
uint16_t UART_CalculateCRC16(uint8_t *data, uint16_t length);
void UART_TransmitCANMessage(uint16_t canId, uint8_t dlc, uint8_t *data);

#ifdef __cplusplus
}
#endif

#endif /* __UART_H */
