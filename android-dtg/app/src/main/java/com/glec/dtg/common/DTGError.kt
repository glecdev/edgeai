package com.glec.dtg.common

/**
 * GLEC DTG - Production-Grade Error Hierarchy
 *
 * Comprehensive error types for all DTG subsystems with recovery strategies.
 *
 * Error Categories:
 * - UARTError: Hardware communication failures
 * - InferenceError: AI model execution failures
 * - NetworkError: MQTT/HTTP communication failures
 * - StorageError: Database/file I/O failures
 * - BLEError: Bluetooth connectivity failures
 * - ValidationError: Data validation failures
 *
 * Each error includes:
 * - Descriptive message
 * - Optional cause (Throwable)
 * - Error code (for logging/tracking)
 * - Severity level
 * - Suggested recovery action
 */
sealed class DTGError {

    abstract val message: String
    abstract val cause: Throwable?
    abstract val code: String
    abstract val severity: ErrorSeverity

    // ========================================
    // UART Communication Errors
    // ========================================

    sealed class UARTError : DTGError() {
        override val severity: ErrorSeverity = ErrorSeverity.HIGH

        /**
         * Failed to open UART device
         * Recovery: Retry after 3 seconds, max 5 times
         */
        data class DeviceNotFound(
            val devicePath: String,
            override val cause: Throwable? = null
        ) : UARTError() {
            override val message = "UART device not found: $devicePath"
            override val code = "UART_001"
        }

        /**
         * Failed to configure UART (baud rate, parity, etc.)
         * Recovery: Reset device and retry
         */
        data class ConfigurationFailed(
            val reason: String,
            override val cause: Throwable? = null
        ) : UARTError() {
            override val message = "UART configuration failed: $reason"
            override val code = "UART_002"
        }

        /**
         * Timeout while reading CAN frame
         * Recovery: Continue (non-fatal), log warning
         */
        data class ReadTimeout(
            val timeoutMs: Long,
            override val cause: Throwable? = null
        ) : UARTError() {
            override val message = "UART read timeout after ${timeoutMs}ms"
            override val code = "UART_003"
            override val severity = ErrorSeverity.MEDIUM
        }

        /**
         * Invalid CAN frame (CRC mismatch, invalid start/end bytes)
         * Recovery: Discard frame, continue reading
         */
        data class InvalidFrame(
            val reason: String,
            override val cause: Throwable? = null
        ) : UARTError() {
            override val message = "Invalid CAN frame: $reason"
            override val code = "UART_004"
            override val severity = ErrorSeverity.LOW
        }

        /**
         * I/O error during read/write
         * Recovery: Close and reopen device
         */
        data class IOError(
            val operation: String,
            override val cause: Throwable
        ) : UARTError() {
            override val message = "UART I/O error during $operation: ${cause.message}"
            override val code = "UART_005"
        }
    }

    // ========================================
    // AI Inference Errors
    // ========================================

    sealed class InferenceError : DTGError() {
        override val severity: ErrorSeverity = ErrorSeverity.MEDIUM

        /**
         * Failed to load ONNX model from assets
         * Recovery: Fatal - cannot continue
         */
        data class ModelLoadFailed(
            val modelPath: String,
            override val cause: Throwable
        ) : InferenceError() {
            override val message = "Failed to load model: $modelPath"
            override val code = "INF_001"
            override val severity = ErrorSeverity.CRITICAL
        }

        /**
         * Invalid input features (wrong size, NaN, Inf)
         * Recovery: Skip inference, use previous result
         */
        data class InvalidInput(
            val reason: String,
            val expectedSize: Int,
            val actualSize: Int
        ) : InferenceError() {
            override val message = "Invalid input: $reason (expected $expectedSize, got $actualSize)"
            override val code = "INF_002"
            override val cause: Throwable? = null
        }

        /**
         * ONNX Runtime execution failed
         * Recovery: Retry once, then fallback to default classification
         */
        data class ExecutionFailed(
            val modelName: String,
            override val cause: Throwable
        ) : InferenceError() {
            override val message = "Model execution failed: $modelName - ${cause.message}"
            override val code = "INF_003"
        }

        /**
         * Output tensor has unexpected shape or type
         * Recovery: Use default classification
         */
        data class InvalidOutput(
            val reason: String
        ) : InferenceError() {
            override val message = "Invalid model output: $reason"
            override val code = "INF_004"
            override val cause: Throwable? = null
        }

        /**
         * Inference latency exceeded threshold (performance degradation)
         * Recovery: Log warning, continue (non-fatal)
         */
        data class PerformanceDegraded(
            val latencyMs: Double,
            val thresholdMs: Double
        ) : InferenceError() {
            override val message = "Inference slow: ${latencyMs}ms (threshold: ${thresholdMs}ms)"
            override val code = "INF_005"
            override val severity = ErrorSeverity.LOW
            override val cause: Throwable? = null
        }
    }

    // ========================================
    // Network Errors (MQTT, HTTP)
    // ========================================

    sealed class NetworkError : DTGError() {
        override val severity: ErrorSeverity = ErrorSeverity.MEDIUM

        /**
         * Failed to connect to MQTT broker
         * Recovery: Exponential backoff retry (1s, 2s, 4s, 8s, 16s)
         */
        data class ConnectionFailed(
            val broker: String,
            override val cause: Throwable
        ) : NetworkError() {
            override val message = "Failed to connect to $broker: ${cause.message}"
            override val code = "NET_001"
        }

        /**
         * Connection lost (unexpected disconnect)
         * Recovery: Auto-reconnect with exponential backoff
         */
        data class ConnectionLost(
            val reason: String
        ) : NetworkError() {
            override val message = "Connection lost: $reason"
            override val code = "NET_002"
            override val cause: Throwable? = null
        }

        /**
         * Failed to publish message
         * Recovery: Add to offline queue
         */
        data class PublishFailed(
            val topic: String,
            override val cause: Throwable
        ) : NetworkError() {
            override val message = "Failed to publish to $topic: ${cause.message}"
            override val code = "NET_003"
        }

        /**
         * HTTP request failed
         * Recovery: Retry with exponential backoff
         */
        data class HTTPError(
            val url: String,
            val statusCode: Int,
            val responseBody: String?
        ) : NetworkError() {
            override val message = "HTTP $statusCode from $url: $responseBody"
            override val code = "NET_004"
            override val cause: Throwable? = null
        }

        /**
         * Network timeout
         * Recovery: Retry once, then fail
         */
        data class Timeout(
            val operation: String,
            val timeoutMs: Long
        ) : NetworkError() {
            override val message = "$operation timeout after ${timeoutMs}ms"
            override val code = "NET_005"
            override val cause: Throwable? = null
        }

        /**
         * TLS/SSL certificate validation failed
         * Recovery: Fatal - do not retry (security issue)
         */
        data class TLSError(
            val reason: String,
            override val cause: Throwable
        ) : NetworkError() {
            override val message = "TLS error: $reason"
            override val code = "NET_006"
            override val severity = ErrorSeverity.CRITICAL
        }
    }

    // ========================================
    // Storage Errors (Database, Files)
    // ========================================

    sealed class StorageError : DTGError() {
        override val severity: ErrorSeverity = ErrorSeverity.MEDIUM

        /**
         * Database operation failed
         * Recovery: Retry once, then skip
         */
        data class DatabaseError(
            val operation: String,
            override val cause: Throwable
        ) : StorageError() {
            override val message = "Database error during $operation: ${cause.message}"
            override val code = "STO_001"
        }

        /**
         * Insufficient storage space
         * Recovery: Delete old data, notify user
         */
        data class InsufficientSpace(
            val requiredBytes: Long,
            val availableBytes: Long
        ) : StorageError() {
            override val message = "Insufficient space: need ${requiredBytes}B, have ${availableBytes}B"
            override val code = "STO_002"
            override val severity = ErrorSeverity.HIGH
            override val cause: Throwable? = null
        }

        /**
         * File I/O error
         * Recovery: Retry once, then skip
         */
        data class FileError(
            val filePath: String,
            override val cause: Throwable
        ) : StorageError() {
            override val message = "File error: $filePath - ${cause.message}"
            override val code = "STO_003"
        }

        /**
         * Encryption/decryption failed
         * Recovery: Fatal if decryption, skip if encryption
         */
        data class CryptoError(
            val operation: String,
            override val cause: Throwable
        ) : StorageError() {
            override val message = "Crypto error during $operation: ${cause.message}"
            override val code = "STO_004"
            override val severity = ErrorSeverity.HIGH
        }
    }

    // ========================================
    // BLE Errors
    // ========================================

    sealed class BLEError : DTGError() {
        override val severity: ErrorSeverity = ErrorSeverity.MEDIUM

        /**
         * BLE not supported on device
         * Recovery: Fatal - cannot continue
         */
        object NotSupported : BLEError() {
            override val message = "BLE not supported on this device"
            override val code = "BLE_001"
            override val severity = ErrorSeverity.CRITICAL
            override val cause: Throwable? = null
        }

        /**
         * BLE not enabled (user disabled)
         * Recovery: Prompt user to enable
         */
        object NotEnabled : BLEError() {
            override val message = "BLE is not enabled"
            override val code = "BLE_002"
            override val cause: Throwable? = null
        }

        /**
         * Device not found during scan
         * Recovery: Continue scanning
         */
        data class DeviceNotFound(
            val deviceName: String
        ) : BLEError() {
            override val message = "BLE device not found: $deviceName"
            override val code = "BLE_003"
            override val cause: Throwable? = null
        }

        /**
         * Connection to device failed
         * Recovery: Retry 3 times
         */
        data class ConnectionFailed(
            val deviceAddress: String,
            override val cause: Throwable?
        ) : BLEError() {
            override val message = "BLE connection failed: $deviceAddress"
            override val code = "BLE_004"
        }

        /**
         * Service discovery failed
         * Recovery: Reconnect and retry
         */
        data class ServiceDiscoveryFailed(
            override val cause: Throwable
        ) : BLEError() {
            override val message = "BLE service discovery failed: ${cause.message}"
            override val code = "BLE_005"
        }
    }

    // ========================================
    // Validation Errors
    // ========================================

    sealed class ValidationError : DTGError() {
        override val severity: ErrorSeverity = ErrorSeverity.LOW

        /**
         * CAN data validation failed (out of range, invalid format)
         * Recovery: Discard data, continue
         */
        data class InvalidCANData(
            val field: String,
            val value: Any?,
            val reason: String
        ) : ValidationError() {
            override val message = "Invalid CAN data: $field=$value ($reason)"
            override val code = "VAL_001"
            override val cause: Throwable? = null
        }

        /**
         * Configuration validation failed
         * Recovery: Use default configuration
         */
        data class InvalidConfiguration(
            val key: String,
            val value: Any?,
            val reason: String
        ) : ValidationError() {
            override val message = "Invalid config: $key=$value ($reason)"
            override val code = "VAL_002"
            override val cause: Throwable? = null
        }

        /**
         * GPS coordinates validation failed
         * Recovery: Skip location update
         */
        data class InvalidGPS(
            val latitude: Double,
            val longitude: Double,
            val reason: String
        ) : ValidationError() {
            override val message = "Invalid GPS: ($latitude, $longitude) - $reason"
            override val code = "VAL_003"
            override val cause: Throwable? = null
        }
    }

    // ========================================
    // Error Severity Levels
    // ========================================

    enum class ErrorSeverity {
        /**
         * Low severity - can be ignored or logged
         * Examples: Invalid single frame, performance degradation
         */
        LOW,

        /**
         * Medium severity - should be logged and handled
         * Examples: Network timeout, database error
         */
        MEDIUM,

        /**
         * High severity - requires immediate attention
         * Examples: UART device lost, insufficient storage
         */
        HIGH,

        /**
         * Critical severity - fatal, cannot recover
         * Examples: Model load failed, TLS error, BLE not supported
         */
        CRITICAL
    }

    // ========================================
    // Error Logging & Formatting
    // ========================================

    /**
     * Format error for logging
     */
    fun toLogString(): String {
        return "[$code] [$severity] $message${cause?.let { " | Cause: ${it.message}" } ?: ""}"
    }

    /**
     * Format error for user display (without technical details)
     */
    fun toUserMessage(): String {
        return when (this) {
            is UARTError.DeviceNotFound -> "차량과 연결할 수 없습니다. 케이블을 확인해주세요."
            is UARTError.ReadTimeout -> "차량 데이터 수신이 지연되고 있습니다."
            is InferenceError.ModelLoadFailed -> "AI 모델을 불러올 수 없습니다. 앱을 재설치해주세요."
            is NetworkError.ConnectionFailed -> "서버에 연결할 수 없습니다. 네트워크를 확인해주세요."
            is NetworkError.TLSError -> "보안 연결에 실패했습니다."
            is StorageError.InsufficientSpace -> "저장 공간이 부족합니다. (${requiredBytes / 1024}KB 필요)"
            is BLEError.NotSupported -> "이 기기는 블루투스를 지원하지 않습니다."
            is BLEError.NotEnabled -> "블루투스를 켜주세요."
            else -> "오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        }
    }
}
