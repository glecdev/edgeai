/**
  ******************************************************************************
  * @file    uart.c
  * @brief   UART HAL module driver - GLEC DTG
  ******************************************************************************
  * UART Configuration:
  * - Baud Rate: 921600
  * - Data Bits: 8
  * - Stop Bits: 1
  * - Parity: None
  * - Mode: TX/RX
  *
  * Frame Format: [START(0xAA)] [ID_H] [ID_L] [DLC] [DATA(8)] [CRC16(2)] [END(0x55)]
  * Total: 15 bytes per CAN frame
  ******************************************************************************
  */

#include "uart.h"

UART_HandleTypeDef huart2;

#define FRAME_START 0xAA
#define FRAME_END   0x55
#define FRAME_SIZE  15

/**
  * @brief UART2 Initialization Function
  * @param None
  * @retval None
  */
void MX_UART2_Init(void)
{
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 921600;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;

    if (HAL_UART_Init(&huart2) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief Calculate CRC16 for frame validation
  * @param data Pointer to data array
  * @param length Length of data
  * @retval CRC16 value
  */
uint16_t UART_CalculateCRC16(uint8_t *data, uint16_t length)
{
    uint16_t crc = 0xFFFF;

    for (uint16_t i = 0; i < length; i++)
    {
        crc ^= data[i];
        for (uint8_t j = 0; j < 8; j++)
        {
            if (crc & 0x0001)
            {
                crc >>= 1;
                crc ^= 0xA001;
            }
            else
            {
                crc >>= 1;
            }
        }
    }

    return crc;
}

/**
  * @brief Transmit CAN message to UART
  * @param canId CAN message ID
  * @param dlc Data length code (0-8)
  * @param data Pointer to data array
  * @retval None
  */
void UART_TransmitCANMessage(uint16_t canId, uint8_t dlc, uint8_t *data)
{
    uint8_t frame[FRAME_SIZE];

    /* Build frame */
    frame[0] = FRAME_START;
    frame[1] = (canId >> 8) & 0xFF;  /* ID High */
    frame[2] = canId & 0xFF;          /* ID Low */
    frame[3] = dlc;

    /* Copy data (pad with 0 if DLC < 8) */
    for (uint8_t i = 0; i < 8; i++)
    {
        frame[4 + i] = (i < dlc) ? data[i] : 0;
    }

    /* Calculate CRC */
    uint16_t crc = UART_CalculateCRC16(frame + 1, 11);
    frame[12] = (crc >> 8) & 0xFF;
    frame[13] = crc & 0xFF;

    frame[14] = FRAME_END;

    /* Transmit frame via UART */
    HAL_UART_Transmit(&huart2, frame, FRAME_SIZE, 10);
}
