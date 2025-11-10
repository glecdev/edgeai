package com.glec.dtg.mqtt

/**
 * GLEC DTG - MQTT Connection Callback
 *
 * Callback interface for MQTT connection state changes.
 *
 * Usage:
 * ```kotlin
 * mqttManager.setConnectionCallback(object : ConnectionCallback {
 *     override fun onConnected() {
 *         Log.i(TAG, "MQTT connected")
 *     }
 *
 *     override fun onConnectionLost(cause: Throwable?) {
 *         Log.w(TAG, "Connection lost", cause)
 *     }
 *
 *     override fun onReconnecting() {
 *         Log.i(TAG, "Reconnecting...")
 *     }
 * })
 * ```
 */
interface ConnectionCallback {
    /**
     * Called when successfully connected to MQTT broker
     */
    fun onConnected()

    /**
     * Called when connection is lost
     *
     * @param cause Exception that caused connection loss (null if graceful disconnect)
     */
    fun onConnectionLost(cause: Throwable?)

    /**
     * Called when attempting to reconnect
     */
    fun onReconnecting()
}

/**
 * Connection state enum
 */
enum class ConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    RECONNECTING,
    ERROR
}

/**
 * MQTT message data class for offline queue
 */
data class QueuedMessage(
    val id: Long = 0,
    val topic: String,
    val payload: String,
    val qos: Int,
    val timestamp: Long = System.currentTimeMillis(),
    val ttl: Long,
    val retryCount: Int = 0
) {
    /**
     * Check if message is expired based on TTL
     */
    fun isExpired(): Boolean {
        return System.currentTimeMillis() > ttl
    }

    /**
     * Check if message should be retried
     */
    fun canRetry(maxRetries: Int = 3): Boolean {
        return retryCount < maxRetries
    }

    /**
     * Create new message with incremented retry count
     */
    fun withRetry(): QueuedMessage {
        return copy(retryCount = retryCount + 1)
    }
}

/**
 * MQTT metrics data class
 */
data class MQTTMetrics(
    val isConnected: Boolean = false,
    val messagesSent: Long = 0,
    val messagesFailed: Long = 0,
    val messagesQueued: Int = 0,
    val reconnectCount: Int = 0,
    val lastConnectTime: Long = 0,
    val lastDisconnectTime: Long = 0,
    val averageLatencyMs: Double = 0.0
) {
    /**
     * Get connection uptime in seconds
     */
    fun getUptimeSeconds(): Long {
        if (!isConnected || lastConnectTime == 0L) return 0
        return (System.currentTimeMillis() - lastConnectTime) / 1000
    }

    /**
     * Get success rate
     */
    fun getSuccessRate(): Double {
        val total = messagesSent + messagesFailed
        if (total == 0L) return 0.0
        return (messagesSent.toDouble() / total.toDouble()) * 100.0
    }
}
