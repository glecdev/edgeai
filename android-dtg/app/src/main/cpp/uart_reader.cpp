/**
 * GLEC DTG - UART Reader (JNI)
 * Reads CAN data from STM32 via UART (921600 baud)
 *
 * Frame Format: [START(0xAA)] [ID_H] [ID_L] [DLC] [DATA(8)] [CRC16(2)] [END(0x55)]
 * Total: 15 bytes per CAN frame
 */

#include <jni.h>
#include <string>
#include <android/log.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#define LOG_TAG "UART_Reader"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// UART configuration
#define UART_DEVICE "/dev/ttyHS0"  // Snapdragon UART (adjust for your hardware)
#define BAUD_RATE B921600

// CAN frame protocol
#define FRAME_START 0xAA
#define FRAME_END 0x55
#define FRAME_SIZE 15

static int uart_fd = -1;

/**
 * Open UART device
 */
bool openUART() {
    if (uart_fd >= 0) {
        LOGI("UART already open");
        return true;
    }

    uart_fd = open(UART_DEVICE, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (uart_fd < 0) {
        LOGE("Failed to open UART device: %s", UART_DEVICE);
        return false;
    }

    // Configure UART
    struct termios options;
    tcgetattr(uart_fd, &options);

    // Set baud rate
    cfsetispeed(&options, BAUD_RATE);
    cfsetospeed(&options, BAUD_RATE);

    // 8N1 mode
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;

    // No hardware flow control
    options.c_cflag &= ~CRTSCTS;

    // Enable receiver
    options.c_cflag |= (CLOCAL | CREAD);

    // Raw input mode
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    // Raw output mode
    options.c_oflag &= ~OPOST;

    tcsetattr(uart_fd, TCSANOW, &options);

    LOGI("UART opened successfully: %s @ 921600 baud", UART_DEVICE);
    return true;
}

/**
 * Close UART device
 */
void closeUART() {
    if (uart_fd >= 0) {
        close(uart_fd);
        uart_fd = -1;
        LOGI("UART closed");
    }
}

/**
 * Calculate CRC16 for frame validation
 */
uint16_t calculateCRC16(const uint8_t* data, int length) {
    uint16_t crc = 0xFFFF;

    for (int i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc >>= 1;
                crc ^= 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }

    return crc;
}

/**
 * Read CAN frame from UART
 */
bool readCANFrame(uint8_t* frame) {
    if (uart_fd < 0) {
        if (!openUART()) {
            return false;
        }
    }

    uint8_t buffer[FRAME_SIZE];
    int bytesRead = 0;

    // Read until we find frame start
    while (bytesRead == 0) {
        uint8_t byte;
        int n = read(uart_fd, &byte, 1);

        if (n > 0 && byte == FRAME_START) {
            buffer[0] = byte;
            bytesRead = 1;
            break;
        }

        usleep(100);  // Small delay to avoid busy-waiting
    }

    // Read rest of frame
    while (bytesRead < FRAME_SIZE) {
        int n = read(uart_fd, buffer + bytesRead, FRAME_SIZE - bytesRead);
        if (n > 0) {
            bytesRead += n;
        } else {
            usleep(100);
        }
    }

    // Validate frame
    if (buffer[FRAME_SIZE - 1] != FRAME_END) {
        LOGE("Invalid frame end: 0x%02X", buffer[FRAME_SIZE - 1]);
        return false;
    }

    // Validate CRC
    uint16_t receivedCRC = (buffer[12] << 8) | buffer[13];
    uint16_t calculatedCRC = calculateCRC16(buffer + 1, 11);

    if (receivedCRC != calculatedCRC) {
        LOGE("CRC mismatch: received=0x%04X, calculated=0x%04X", receivedCRC, calculatedCRC);
        return false;
    }

    // Copy valid frame
    memcpy(frame, buffer, FRAME_SIZE);
    return true;
}

/**
 * JNI: Read CAN data from UART
 */
extern "C" JNIEXPORT jobject JNICALL
Java_com_glec_dtg_service_DTGForegroundService_readCANDataFromUART(
        JNIEnv* env,
        jobject /* this */) {

    uint8_t frame[FRAME_SIZE];

    if (!readCANFrame(frame)) {
        // Return null on failure
        return nullptr;
    }

    // Parse CAN frame
    uint16_t canID = (frame[1] << 8) | frame[2];
    uint8_t dlc = frame[3];
    uint8_t* data = frame + 4;

    // TODO: Parse CAN data based on CAN ID (OBD-II PIDs or J1939 PGNs)
    // For now, return synthetic data

    // Find CANData class
    jclass canDataClass = env->FindClass("com/glec/dtg/service/CANData");
    if (canDataClass == nullptr) {
        LOGE("Failed to find CANData class");
        return nullptr;
    }

    // Get constructor
    jmethodID constructor = env->GetMethodID(canDataClass, "<init>",
                                             "(JFFFFFFFFFFDD)V");
    if (constructor == nullptr) {
        LOGE("Failed to find CANData constructor");
        return nullptr;
    }

    // Create CANData object (synthetic data for skeleton)
    jobject canData = env->NewObject(canDataClass, constructor,
                                      (jlong) (System.currentTimeMillis()),
                                      (jfloat) 80.5,   // vehicle speed
                                      (jfloat) 2500.0, // engine RPM
                                      (jfloat) 45.0,   // throttle
                                      (jfloat) 0.0,    // brake
                                      (jfloat) 75.0,   // fuel level
                                      (jfloat) 85.0,   // coolant temp
                                      (jfloat) 0.5,    // accel X
                                      (jfloat) 0.1,    // accel Y
                                      (jfloat) -5.2,   // steering angle
                                      (jdouble) 37.5665, // GPS lat
                                      (jdouble) 126.9780 // GPS lon
    );

    LOGI("CAN frame read: ID=0x%04X, DLC=%d", canID, dlc);

    return canData;
}

/**
 * JNI: Preprocess CAN data for AI inference
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_glec_dtg_service_DTGForegroundService_preprocessCANData(
        JNIEnv* env,
        jobject /* this */,
        jobject canDataList) {

    // TODO: Implement preprocessing
    // 1. Extract 60 samples from list
    // 2. Normalize features
    // 3. Convert to float array (60 x 10 = 600 values)

    // For skeleton, return placeholder
    jfloatArray result = env->NewFloatArray(600);
    float* data = new float[600];

    for (int i = 0; i < 600; i++) {
        data[i] = 0.0f;  // Placeholder
    }

    env->SetFloatArrayRegion(result, 0, 600, data);
    delete[] data;

    return result;
}
