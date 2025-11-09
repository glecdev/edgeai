/**
 * GLEC DTG - STM32 CAN Interface
 * Handles CAN bus communication and message filtering
 *
 * Hardware: STM32F4 with MCP2551 CAN transceiver
 * CAN Bitrate: 500 kbps
 * Filter: OBD-II PIDs (0x7E8, 0x7DF) and J1939 PGNs
 */

#include "can_interface.h"
#include "main.h"
#include <string.h>

/* External CAN handle */
extern CAN_HandleTypeDef hcan1;

/* CAN RX buffer */
static CAN_RxHeaderTypeDef rxHeader;
static uint8_t rxData[8];

/* Message queue (circular buffer) */
#define CAN_QUEUE_SIZE 64
static CANMessage canQueue[CAN_QUEUE_SIZE];
static volatile uint16_t queueHead = 0;
static volatile uint16_t queueTail = 0;

/* Statistics */
static uint32_t messagesReceived = 0;
static uint32_t messagesDropped = 0;
static uint32_t crcErrors = 0;

/**
 * Initialize CAN interface
 */
HAL_StatusTypeDef CAN_Init(void)
{
    /* Configure CAN filter for OBD-II and J1939 */
    CAN_FilterTypeDef canFilter;

    canFilter.FilterIdHigh = 0x7E8 << 5;      // OBD-II response ID
    canFilter.FilterIdLow = 0x7DF << 5;       // OBD-II broadcast ID
    canFilter.FilterMaskIdHigh = 0x7F0 << 5;  // Mask for OBD-II range
    canFilter.FilterMaskIdLow = 0x7F0 << 5;
    canFilter.FilterFIFOAssignment = CAN_RX_FIFO0;
    canFilter.FilterBank = 0;
    canFilter.FilterMode = CAN_FILTERMODE_IDMASK;
    canFilter.FilterScale = CAN_FILTERSCALE_32BIT;
    canFilter.FilterActivation = ENABLE;
    canFilter.SlaveStartFilterBank = 14;

    if (HAL_CAN_ConfigFilter(&hcan1, &canFilter) != HAL_OK)
    {
        return HAL_ERROR;
    }

    /* Configure second filter for J1939 PGNs */
    canFilter.FilterIdHigh = 0x0CF00000 >> 16;  // J1939 PGN base
    canFilter.FilterIdLow = 0x0CF00000 & 0xFFFF;
    canFilter.FilterMaskIdHigh = 0x0FFFF000 >> 16;
    canFilter.FilterMaskIdLow = 0x0FFFF000 & 0xFFFF;
    canFilter.FilterBank = 1;

    if (HAL_CAN_ConfigFilter(&hcan1, &canFilter) != HAL_OK)
    {
        return HAL_ERROR;
    }

    /* Activate CAN RX interrupt */
    if (HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING) != HAL_OK)
    {
        return HAL_ERROR;
    }

    /* Start CAN */
    if (HAL_CAN_Start(&hcan1) != HAL_OK)
    {
        return HAL_ERROR;
    }

    return HAL_OK;
}

/**
 * Send OBD-II PID request
 */
HAL_StatusTypeDef CAN_SendOBDRequest(uint8_t mode, uint8_t pid)
{
    CAN_TxHeaderTypeDef txHeader;
    uint8_t txData[8] = {0};
    uint32_t txMailbox;

    /* Configure TX header */
    txHeader.StdId = 0x7DF;  // OBD-II broadcast ID
    txHeader.IDE = CAN_ID_STD;
    txHeader.RTR = CAN_RTR_DATA;
    txHeader.DLC = 8;
    txHeader.TransmitGlobalTime = DISABLE;

    /* Build OBD-II request */
    txData[0] = 0x02;  // Number of additional bytes
    txData[1] = mode;  // Mode (e.g., 0x01 for current data)
    txData[2] = pid;   // PID
    txData[3] = 0xAA;  // Padding
    txData[4] = 0xAA;
    txData[5] = 0xAA;
    txData[6] = 0xAA;
    txData[7] = 0xAA;

    /* Send message */
    if (HAL_CAN_AddTxMessage(&hcan1, &txHeader, txData, &txMailbox) != HAL_OK)
    {
        return HAL_ERROR;
    }

    return HAL_OK;
}

/**
 * Request all required OBD-II PIDs
 */
void CAN_RequestAllPIDs(void)
{
    /* Request essential PIDs */
    CAN_SendOBDRequest(0x01, 0x0C);  // Engine RPM
    HAL_Delay(10);

    CAN_SendOBDRequest(0x01, 0x0D);  // Vehicle Speed
    HAL_Delay(10);

    CAN_SendOBDRequest(0x01, 0x11);  // Throttle Position
    HAL_Delay(10);

    CAN_SendOBDRequest(0x01, 0x2F);  // Fuel Level
    HAL_Delay(10);

    CAN_SendOBDRequest(0x01, 0x05);  // Coolant Temperature
    HAL_Delay(10);

    CAN_SendOBDRequest(0x01, 0x04);  // Engine Load
    HAL_Delay(10);

    CAN_SendOBDRequest(0x01, 0x0F);  // Intake Air Temperature
    HAL_Delay(10);

    CAN_SendOBDRequest(0x01, 0x10);  // MAF Air Flow Rate
    HAL_Delay(10);

    CAN_SendOBDRequest(0x01, 0x42);  // Control Module Voltage
    HAL_Delay(10);
}

/**
 * Check if queue is full
 */
static inline uint8_t CAN_IsQueueFull(void)
{
    return ((queueHead + 1) % CAN_QUEUE_SIZE) == queueTail;
}

/**
 * Check if queue is empty
 */
static inline uint8_t CAN_IsQueueEmpty(void)
{
    return queueHead == queueTail;
}

/**
 * Add message to queue
 */
static void CAN_EnqueueMessage(uint32_t canId, uint8_t dlc, uint8_t *data)
{
    if (CAN_IsQueueFull())
    {
        messagesDropped++;
        return;
    }

    CANMessage *msg = &canQueue[queueHead];
    msg->canId = canId;
    msg->dlc = dlc;
    memcpy(msg->data, data, 8);
    msg->timestamp = HAL_GetTick();

    queueHead = (queueHead + 1) % CAN_QUEUE_SIZE;
    messagesReceived++;
}

/**
 * Get message from queue
 */
uint8_t CAN_DequeueMessage(CANMessage *msg)
{
    if (CAN_IsQueueEmpty())
    {
        return 0;
    }

    *msg = canQueue[queueTail];
    queueTail = (queueTail + 1) % CAN_QUEUE_SIZE;

    return 1;
}

/**
 * CAN RX callback (called by HAL on message reception)
 */
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
    if (hcan != &hcan1)
    {
        return;
    }

    /* Receive message */
    if (HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &rxHeader, rxData) == HAL_OK)
    {
        uint32_t canId;

        /* Extract CAN ID */
        if (rxHeader.IDE == CAN_ID_STD)
        {
            canId = rxHeader.StdId;
        }
        else
        {
            canId = rxHeader.ExtId;
        }

        /* Add to queue */
        CAN_EnqueueMessage(canId, rxHeader.DLC, rxData);
    }
}

/**
 * Parse OBD-II PID response
 */
uint8_t CAN_ParseOBDResponse(CANMessage *msg, OBDData *obdData)
{
    /* Check if OBD-II response */
    if (msg->canId != 0x7E8 || msg->dlc < 3)
    {
        return 0;
    }

    uint8_t mode = msg->data[1];
    uint8_t pid = msg->data[2];

    /* Check if response mode (request mode + 0x40) */
    if (mode != 0x41)
    {
        return 0;
    }

    switch (pid)
    {
        case 0x0C:  // Engine RPM
            obdData->engineRPM = ((msg->data[3] << 8) | msg->data[4]) / 4;
            obdData->validFlags |= OBD_FLAG_RPM;
            break;

        case 0x0D:  // Vehicle Speed
            obdData->vehicleSpeed = msg->data[3];
            obdData->validFlags |= OBD_FLAG_SPEED;
            break;

        case 0x11:  // Throttle Position
            obdData->throttlePosition = (msg->data[3] * 100.0f) / 255.0f;
            obdData->validFlags |= OBD_FLAG_THROTTLE;
            break;

        case 0x2F:  // Fuel Level
            obdData->fuelLevel = (msg->data[3] * 100.0f) / 255.0f;
            obdData->validFlags |= OBD_FLAG_FUEL;
            break;

        case 0x05:  // Coolant Temperature
            obdData->coolantTemp = msg->data[3] - 40;
            obdData->validFlags |= OBD_FLAG_COOLANT;
            break;

        case 0x04:  // Engine Load
            obdData->engineLoad = (msg->data[3] * 100.0f) / 255.0f;
            obdData->validFlags |= OBD_FLAG_LOAD;
            break;

        case 0x0F:  // Intake Air Temperature
            obdData->intakeAirTemp = msg->data[3] - 40;
            obdData->validFlags |= OBD_FLAG_INTAKE_TEMP;
            break;

        case 0x10:  // MAF Air Flow Rate
            obdData->mafRate = ((msg->data[3] << 8) | msg->data[4]) / 100.0f;
            obdData->validFlags |= OBD_FLAG_MAF;
            break;

        case 0x42:  // Control Module Voltage
            obdData->batteryVoltage = ((msg->data[3] << 8) | msg->data[4]) / 1000.0f;
            obdData->validFlags |= OBD_FLAG_BATTERY;
            break;

        default:
            return 0;
    }

    return 1;
}

/**
 * Get CAN statistics
 */
void CAN_GetStatistics(CANStatistics *stats)
{
    stats->messagesReceived = messagesReceived;
    stats->messagesDropped = messagesDropped;
    stats->crcErrors = crcErrors;
    stats->queueUsage = (queueHead >= queueTail) ?
                        (queueHead - queueTail) :
                        (CAN_QUEUE_SIZE - queueTail + queueHead);
}

/**
 * Reset statistics
 */
void CAN_ResetStatistics(void)
{
    messagesReceived = 0;
    messagesDropped = 0;
    crcErrors = 0;
}
