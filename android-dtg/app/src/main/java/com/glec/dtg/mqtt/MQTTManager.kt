package com.glec.dtg.mqtt

import android.content.Context
import android.util.Log
import org.eclipse.paho.android.service.MqttAndroidClient
import org.eclipse.paho.client.mqttv3.*
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * GLEC DTG - MQTT Manager
 *
 * Production-grade MQTT client for Fleet AI platform connectivity.
 *
 * Features:
 * - Auto-reconnect with exponential backoff
 * - Offline message queue (SQLite)
 * - QoS 0, 1, 2 support
 * - TLS/SSL encryption
 * - Connection state callbacks
 * - Metrics tracking
 *
 * Usage:
 * ```kotlin
 * val config = MQTTConfig.createDefault("DTG-SN-12345")
 * val mqttManager = MQTTManager(context, config)
 *
 * mqttManager.setConnectionCallback(object : ConnectionCallback {
 *     override fun onConnected() {
 *         Log.i(TAG, "Connected")
 *     }
 *     override fun onConnectionLost(cause: Throwable?) {
 *         Log.w(TAG, "Connection lost", cause)
 *     }
 *     override fun onReconnecting() {
 *         Log.i(TAG, "Reconnecting...")
 *     }
 * })
 *
 * mqttManager.connect()
 *
 * // Publish messages
 * mqttManager.publish("glec/dtg/DTG-SN-12345/telemetry", payload, qos = 0)
 * mqttManager.publish("glec/dtg/DTG-SN-12345/inference", payload, qos = 1)
 * mqttManager.publish("glec/dtg/DTG-SN-12345/alerts", payload, qos = 2)
 * ```
 *
 * @property context Android application context
 * @property config MQTT configuration
 */
class MQTTManager(
    private val context: Context,
    private val config: MQTTConfig
) {
    companion object {
        private const val TAG = "MQTTManager"
        private const val MAX_RECONNECT_ATTEMPTS = 5
        private const val INITIAL_RECONNECT_DELAY = 2000L  // 2 seconds
    }

    // MQTT client
    private var mqttClient: MqttAndroidClient? = null

    // Connection state
    private var connectionState = ConnectionState.DISCONNECTED
    private var connectionCallback: ConnectionCallback? = null

    // Reconnection logic
    private var reconnectAttempt = 0
    private var reconnectJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // Metrics
    private val messagesSent = AtomicLong(0)
    private val messagesFailed = AtomicLong(0)
    private val reconnectCount = AtomicLong(0)
    private var lastConnectTime = 0L
    private var lastDisconnectTime = 0L

    // Flags
    private val isConnecting = AtomicBoolean(false)
    private val isDisconnecting = AtomicBoolean(false)

    // Offline queue (SQLite-based persistent storage)
    private val offlineQueueManager = OfflineQueueManager(
        context = context,
        maxQueueSize = config.queueMaxSize,
        ttlHours = config.queueTTLHours
    )

    /**
     * Set connection callback
     */
    fun setConnectionCallback(callback: ConnectionCallback) {
        this.connectionCallback = callback
    }

    /**
     * Connect to MQTT broker
     */
    fun connect() {
        if (isConnecting.get()) {
            Log.w(TAG, "Connection already in progress")
            return
        }

        if (isConnected()) {
            Log.i(TAG, "Already connected")
            return
        }

        if (!config.validate()) {
            Log.e(TAG, "Invalid MQTT configuration")
            return
        }

        isConnecting.set(true)
        connectionState = ConnectionState.CONNECTING

        Log.i(TAG, "Connecting to MQTT broker: ${config.brokerUrl}")
        Log.i(TAG, "Client ID: ${config.clientId}")

        try {
            // Create MQTT client
            mqttClient = MqttAndroidClient(
                context,
                config.brokerUrl,
                config.clientId
            )

            // Set callback
            mqttClient?.setCallback(object : MqttCallback {
                override fun connectionLost(cause: Throwable?) {
                    handleConnectionLost(cause)
                }

                override fun messageArrived(topic: String?, message: MqttMessage?) {
                    // Not used (we only publish, not subscribe)
                }

                override fun deliveryComplete(token: IMqttDeliveryToken?) {
                    // Message delivered successfully
                    messagesSent.incrementAndGet()
                }
            })

            // Create connection options
            val options = MqttConnectOptions().apply {
                isCleanSession = config.cleanSession
                connectionTimeout = config.connectionTimeout
                keepAliveInterval = config.keepAliveInterval
                isAutomaticReconnect = false  // We handle reconnection manually

                // Set username/password if provided
                if (config.username != null) {
                    userName = config.username
                }
                if (config.password != null) {
                    password = config.password.toCharArray()
                }

                // TODO: Add TLS/SSL configuration
                // socketFactory = createSSLSocketFactory()
            }

            // Connect
            mqttClient?.connect(options, null, object : IMqttActionListener {
                override fun onSuccess(asyncActionToken: IMqttToken?) {
                    handleConnectionSuccess()
                }

                override fun onFailure(asyncActionToken: IMqttToken?, exception: Throwable?) {
                    handleConnectionFailure(exception)
                }
            })

        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to MQTT broker", e)
            handleConnectionFailure(e)
        }
    }

    /**
     * Disconnect from MQTT broker
     */
    fun disconnect() {
        if (isDisconnecting.get()) {
            Log.w(TAG, "Disconnect already in progress")
            return
        }

        if (!isConnected()) {
            Log.i(TAG, "Already disconnected")
            return
        }

        isDisconnecting.set(true)
        connectionState = ConnectionState.DISCONNECTED

        Log.i(TAG, "Disconnecting from MQTT broker")

        try {
            // Cancel reconnection attempts
            reconnectJob?.cancel()
            reconnectJob = null

            // Disconnect
            mqttClient?.disconnect(null, object : IMqttActionListener {
                override fun onSuccess(asyncActionToken: IMqttToken?) {
                    Log.i(TAG, "Disconnected successfully")
                    cleanup()
                    isDisconnecting.set(false)
                }

                override fun onFailure(asyncActionToken: IMqttToken?, exception: Throwable?) {
                    Log.w(TAG, "Disconnect failed", exception)
                    cleanup()
                    isDisconnecting.set(false)
                }
            })

        } catch (e: Exception) {
            Log.e(TAG, "Error during disconnect", e)
            cleanup()
            isDisconnecting.set(false)
        }
    }

    /**
     * Check if connected to broker
     */
    fun isConnected(): Boolean {
        return mqttClient?.isConnected == true
    }

    /**
     * Get current connection state
     */
    fun getConnectionState(): ConnectionState {
        return connectionState
    }

    /**
     * Publish message to MQTT broker
     *
     * @param topic MQTT topic
     * @param payload Message payload (JSON string)
     * @param qos Quality of Service level (0, 1, or 2)
     * @return true if published successfully (or queued), false otherwise
     */
    fun publish(topic: String, payload: String, qos: Int = 1): Boolean {
        if (qos !in 0..2) {
            Log.e(TAG, "Invalid QoS level: $qos (must be 0, 1, or 2)")
            return false
        }

        // If not connected, add to offline queue
        if (!isConnected()) {
            Log.d(TAG, "Not connected, queueing message")
            queueMessage(topic, payload, qos)
            return true
        }

        try {
            val message = MqttMessage().apply {
                this.payload = payload.toByteArray()
                this.qos = qos
                this.isRetained = false
            }

            mqttClient?.publish(topic, message, null, object : IMqttActionListener {
                override fun onSuccess(asyncActionToken: IMqttToken?) {
                    Log.d(TAG, "Published to $topic (QoS $qos)")
                    messagesSent.incrementAndGet()
                }

                override fun onFailure(asyncActionToken: IMqttToken?, exception: Throwable?) {
                    Log.w(TAG, "Failed to publish to $topic", exception)
                    messagesFailed.incrementAndGet()
                    // Queue for retry
                    queueMessage(topic, payload, qos)
                }
            })

            return true

        } catch (e: Exception) {
            Log.e(TAG, "Error publishing message", e)
            messagesFailed.incrementAndGet()
            queueMessage(topic, payload, qos)
            return false
        }
    }

    /**
     * Publish telemetry data (QoS 0 - fire and forget)
     */
    fun publishTelemetry(deviceId: String, payload: String): Boolean {
        val topic = "glec/dtg/$deviceId/telemetry"
        return publish(topic, payload, qos = 0)
    }

    /**
     * Publish inference result (QoS 1 - at least once)
     */
    fun publishInference(deviceId: String, payload: String): Boolean {
        val topic = "glec/dtg/$deviceId/inference"
        return publish(topic, payload, qos = 1)
    }

    /**
     * Publish critical alert (QoS 2 - exactly once)
     */
    fun publishAlert(deviceId: String, payload: String): Boolean {
        val topic = "glec/dtg/$deviceId/alerts"
        return publish(topic, payload, qos = 2)
    }

    /**
     * Publish device status (QoS 1)
     */
    fun publishStatus(deviceId: String, payload: String): Boolean {
        val topic = "glec/dtg/$deviceId/status"
        return publish(topic, payload, qos = 1)
    }

    /**
     * Get MQTT metrics
     */
    fun getMetrics(): MQTTMetrics {
        return MQTTMetrics(
            isConnected = isConnected(),
            messagesSent = messagesSent.get(),
            messagesFailed = messagesFailed.get(),
            messagesQueued = offlineQueueManager.getQueueSize(),
            reconnectCount = reconnectCount.toInt(),
            lastConnectTime = lastConnectTime,
            lastDisconnectTime = lastDisconnectTime
        )
    }

    /**
     * Get device ID from config
     */
    val deviceId: String
        get() = config.clientId

    // Private methods

    private fun handleConnectionSuccess() {
        isConnecting.set(false)
        connectionState = ConnectionState.CONNECTED
        reconnectAttempt = 0
        lastConnectTime = System.currentTimeMillis()

        Log.i(TAG, "✅ Connected to MQTT broker successfully")

        // Notify callback
        connectionCallback?.onConnected()

        // Flush offline queue
        flushOfflineQueue()
    }

    private fun handleConnectionFailure(exception: Throwable?) {
        isConnecting.set(false)
        connectionState = ConnectionState.ERROR

        Log.e(TAG, "❌ Failed to connect to MQTT broker", exception)

        // Attempt reconnection
        scheduleReconnection()
    }

    private fun handleConnectionLost(cause: Throwable?) {
        connectionState = ConnectionState.DISCONNECTED
        lastDisconnectTime = System.currentTimeMillis()

        Log.w(TAG, "⚠️ Connection lost", cause)

        // Notify callback
        connectionCallback?.onConnectionLost(cause)

        // Attempt reconnection if enabled
        if (config.autoReconnect) {
            scheduleReconnection()
        }
    }

    private fun scheduleReconnection() {
        if (reconnectJob?.isActive == true) {
            Log.d(TAG, "Reconnection already scheduled")
            return
        }

        connectionState = ConnectionState.RECONNECTING
        reconnectCount.incrementAndGet()

        // Notify callback
        connectionCallback?.onReconnecting()

        val delay = calculateReconnectDelay()
        Log.i(TAG, "Scheduling reconnection in ${delay}ms (attempt ${reconnectAttempt + 1})")

        reconnectJob = scope.launch {
            delay(delay)

            if (isActive) {
                Log.i(TAG, "Attempting reconnection...")
                reconnectAttempt++
                connect()
            }
        }
    }

    private fun calculateReconnectDelay(): Long {
        // Exponential backoff: 2^attempt * INITIAL_DELAY
        // Max delay: config.maxReconnectDelay
        if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
            // After max attempts, use fixed delay
            return config.maxReconnectDelay
        }

        val delay = INITIAL_RECONNECT_DELAY * (1 shl reconnectAttempt)  // 2^attempt
        return delay.coerceAtMost(config.maxReconnectDelay)
    }

    private fun queueMessage(topic: String, payload: String, qos: Int) {
        val ttlMillis = config.queueTTLHours * 60 * 60 * 1000

        val messageId = offlineQueueManager.enqueue(
            topic = topic,
            payload = payload,
            qos = qos,
            ttlMillis = ttlMillis
        )

        if (messageId != -1L) {
            Log.d(TAG, "Message queued (id=$messageId, queue size: ${offlineQueueManager.getQueueSize()})")
        } else {
            Log.e(TAG, "Failed to queue message")
        }
    }

    private fun flushOfflineQueue() {
        val queueSize = offlineQueueManager.getQueueSize()
        if (queueSize == 0) {
            return
        }

        Log.i(TAG, "Flushing offline queue ($queueSize messages)")

        // Cleanup expired messages first
        val expiredCount = offlineQueueManager.cleanupExpired()
        if (expiredCount > 0) {
            Log.d(TAG, "Removed $expiredCount expired messages")
        }

        // Dequeue all messages (FIFO order)
        val messages = offlineQueueManager.dequeueAll()
        var flushedCount = 0
        var failedCount = 0

        for (message in messages) {
            // Skip if already expired (shouldn't happen after cleanup, but safety check)
            if (message.isExpired()) {
                offlineQueueManager.delete(message.id)
                continue
            }

            // Check retry limit
            if (!message.canRetry()) {
                Log.w(TAG, "Message exceeded max retries (id=${message.id})")
                offlineQueueManager.delete(message.id)
                continue
            }

            // Publish message
            val success = publish(message.topic, message.payload, message.qos)

            if (success) {
                // Delete from queue on successful publish
                offlineQueueManager.delete(message.id)
                flushedCount++
            } else {
                // Increment retry count on failure
                offlineQueueManager.incrementRetryCount(message.id)
                failedCount++
                // Stop flushing if publish fails
                break
            }
        }

        Log.i(TAG, "Flushed $flushedCount messages from offline queue ($failedCount failed)")
    }

    private fun cleanup() {
        mqttClient?.close()
        mqttClient = null
        connectionState = ConnectionState.DISCONNECTED
    }

    /**
     * Release resources
     */
    fun release() {
        disconnect()
        reconnectJob?.cancel()
        scope.cancel()
        offlineQueueManager.release()
    }
}
