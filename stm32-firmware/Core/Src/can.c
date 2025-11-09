/**
  ******************************************************************************
  * @file    can.c
  * @brief   CAN HAL module driver - GLEC DTG
  ******************************************************************************
  * CAN Bus Configuration:
  * - Bitrate: 500 kbps
  * - Transceiver: MCP2551
  * - Filters: Accept all OBD-II and J1939 messages
  ******************************************************************************
  */

#include "can.h"
#include "uart.h"

CAN_HandleTypeDef hcan1;

/**
  * @brief CAN1 Initialization Function
  * @param None
  * @retval None
  */
void MX_CAN1_Init(void)
{
    hcan1.Instance = CAN1;
    hcan1.Init.Prescaler = 4;
    hcan1.Init.Mode = CAN_MODE_NORMAL;
    hcan1.Init.SyncJumpWidth = CAN_SJW_1TQ;
    hcan1.Init.TimeSeg1 = CAN_BS1_13TQ;
    hcan1.Init.TimeSeg2 = CAN_BS2_2TQ;
    hcan1.Init.TimeTriggeredMode = DISABLE;
    hcan1.Init.AutoBusOff = ENABLE;
    hcan1.Init.AutoWakeUp = DISABLE;
    hcan1.Init.AutoRetransmission = ENABLE;
    hcan1.Init.ReceiveFifoLocked = DISABLE;
    hcan1.Init.TransmitFifoPriority = DISABLE;

    if (HAL_CAN_Init(&hcan1) != HAL_OK)
    {
        Error_Handler();
    }

    /* Configure CAN filter to accept all messages */
    CAN_FilterTypeDef filter;
    filter.FilterBank = 0;
    filter.FilterMode = CAN_FILTERMODE_IDMASK;
    filter.FilterScale = CAN_FILTERSCALE_32BIT;
    filter.FilterIdHigh = 0x0000;
    filter.FilterIdLow = 0x0000;
    filter.FilterMaskIdHigh = 0x0000;
    filter.FilterMaskIdLow = 0x0000;
    filter.FilterFIFOAssignment = CAN_RX_FIFO0;
    filter.FilterActivation = ENABLE;
    filter.SlaveStartFilterBank = 14;

    if (HAL_CAN_ConfigFilter(&hcan1, &filter) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief CAN RX FIFO 0 message pending callback
  * @param hcan pointer to a CAN_HandleTypeDef structure
  * @retval None
  */
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
    CAN_RxHeaderTypeDef rxHeader;
    uint8_t rxData[8];

    /* Get RX message */
    if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &rxHeader, rxData) == HAL_OK)
    {
        /* Transmit CAN message to UART */
        UART_TransmitCANMessage(rxHeader.StdId, rxHeader.DLC, rxData);
    }
}

/**
  * @brief Parse OBD-II PID response
  * @param pid OBD-II PID
  * @param data Pointer to data array
  * @param value Pointer to output value
  * @retval None
  */
void CAN_ParseOBD2Response(uint8_t pid, uint8_t *data, float *value)
{
    switch (pid)
    {
        case 0x0C:  /* Engine RPM */
            *value = ((data[0] * 256.0f) + data[1]) / 4.0f;
            break;

        case 0x0D:  /* Vehicle Speed */
            *value = (float)data[0];
            break;

        case 0x2F:  /* Fuel Level */
            *value = (data[0] * 100.0f) / 255.0f;
            break;

        case 0x11:  /* Throttle Position */
            *value = (data[0] * 100.0f) / 255.0f;
            break;

        case 0x05:  /* Coolant Temperature */
            *value = (float)(data[0] - 40);
            break;

        default:
            *value = 0.0f;
            break;
    }
}
