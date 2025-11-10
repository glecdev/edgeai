package com.glec.dtg.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.glec.dtg.R
import com.glec.dtg.models.AIInferenceResult
import com.glec.dtg.models.CANData
import com.glec.dtg.models.DrivingBehavior
import com.glec.dtg.utils.CANMessageParser
import com.glec.dtg.inference.EdgeAIInferenceService
import com.glec.dtg.mqtt.MQTTManager
import com.glec.dtg.mqtt.MQTTConfig
import com.glec.dtg.mqtt.ConnectionCallback
import kotlinx.coroutines.*
import org.json.JSONObject
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.min

/**
 * GLEC DTG - Foreground Service
 * Main service for DTG operations:
 * 1. Collect CAN data from UART (1Hz)
 * 2. Buffer 60 seconds of data
 * 3. Run AI inference every 60 seconds
 * 4. Send results to Fleet AI platform via MQTT
 * 5. Broadcast results to Driver app via BLE
 */
class DTGForegroundService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private var canReceiverJob: Job? = null
    private var inferenceJob: Job? = null
    private var statusJob: Job? = null

    private var wakeLock: PowerManager.WakeLock? = null

    private lateinit var canReceiver: CANReceiver
    private lateinit var inferenceService: EdgeAIInferenceService
    private lateinit var mqttManager: MQTTManager

    private var isRunning = false
    private var dataCollectionStartTime = 0L
    private var totalSamplesCollected = 0L
    private var totalInferencesRun = 0L

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "DTG Foreground Service created")

        // Acquire wake lock to prevent service from sleeping
        wakeLock = (getSystemService(Context.POWER_SERVICE) as PowerManager).run {
            newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "DTG::WakeLock").apply {
                acquire()
            }
        }

        // Initialize components
        canReceiver = CANReceiver(this)
        inferenceService = EdgeAIInferenceService(this)

        // Initialize MQTT Manager
        val mqttConfig = MQTTConfig.createDefault("DTG-SN-12345")  // TODO: Load device ID from config
        mqttManager = MQTTManager(this, mqttConfig)

        // Set MQTT connection callback
        mqttManager.setConnectionCallback(object : ConnectionCallback {
            override fun onConnected() {
                Log.i(TAG, "✅ MQTT connected to broker")
            }

            override fun onConnectionLost(cause: Throwable?) {
                Log.w(TAG, "⚠️ MQTT connection lost", cause)
            }

            override fun onReconnecting() {
                Log.i(TAG, "🔄 MQTT reconnecting...")
            }
        })

        Log.i(TAG, "Components initialized")
        Log.i(TAG, "  EdgeAIInferenceService: LightGBM behavior classification ready")
        Log.i(TAG, "  MQTTManager: Fleet platform connectivity ready")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "onStartCommand: action=${intent?.action}")

        when (intent?.action) {
            ACTION_START_SERVICE -> startService()
            ACTION_STOP_SERVICE -> stopService()
            ACTION_RESTART_SERVICE -> restartService()
        }

        // START_STICKY: restart service if killed by system
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? {
        // Not a bound service
        return null
    }

    override fun onDestroy() {
        Log.i(TAG, "DTG Foreground Service destroyed")

        stopService()

        wakeLock?.let {
            if (it.isHeld) {
                it.release()
            }
        }

        serviceScope.cancel()

        super.onDestroy()
    }

    /**
     * Start DTG service
     */
    private fun startService() {
        if (isRunning) {
            Log.w(TAG, "Service already running")
            return
        }

        Log.i(TAG, "Starting DTG service...")

        // Create foreground notification
        val notification = createNotification("DTG service is running")
        startForeground(NOTIFICATION_ID, notification)

        // Connect to services
        serviceScope.launch {
            // EdgeAIInferenceService is already initialized
            Log.i(TAG, "LightGBM ONNX model ready (12.62KB, 0.0119ms P95 latency)")

            // Connect to MQTT broker
            mqttManager.connect()
            Log.i(TAG, "MQTT connection initiated (async)")

            // Start CAN data collection
            startCANDataCollection()

            // Start AI inference scheduler
            startInferenceScheduler()

            // Start device status publisher
            startStatusPublisher()

            isRunning = true
            dataCollectionStartTime = System.currentTimeMillis()

            updateNotification("DTG service active - Collecting data")

            Log.i(TAG, "DTG service started successfully")
        }
    }

    /**
     * Stop DTG service
     */
    private fun stopService() {
        if (!isRunning) {
            Log.w(TAG, "Service not running")
            return
        }

        Log.i(TAG, "Stopping DTG service...")

        isRunning = false

        // Stop jobs
        canReceiverJob?.cancel()
        inferenceJob?.cancel()
        statusJob?.cancel()

        // Disconnect MQTT
        mqttManager.disconnect()

        // Close CAN receiver
        canReceiver.close()

        // Reset inference service (clears 60-second window)
        inferenceService.reset()

        updateNotification("DTG service stopped")

        Log.i(TAG, "DTG service stopped")

        stopForeground(true)
        stopSelf()
    }

    /**
     * Restart DTG service
     */
    private fun restartService() {
        Log.i(TAG, "Restarting DTG service...")
        stopService()
        Thread.sleep(1000)  // Wait 1 second
        startService()
    }

    /**
     * Start CAN data collection (1Hz)
     */
    private fun startCANDataCollection() {
        canReceiverJob = serviceScope.launch(Dispatchers.IO) {
            Log.i(TAG, "Starting CAN data collection (1Hz)...")

            try {
                canReceiver.open()

                while (isActive && isRunning) {
                    val startTime = System.currentTimeMillis()

                    // Read CAN data from UART
                    val canData = canReceiver.readCANData()

                    if (canData != null && canData.isValid()) {
                        // Add to EdgeAIInferenceService (manages 60-second window internally)
                        inferenceService.addSample(canData)
                        totalSamplesCollected++

                        // Publish telemetry to MQTT (1Hz, QoS 0)
                        publishTelemetry(canData)

                        // Detect immediate anomalies
                        detectImmediateAnomalies(canData)

                        Log.d(TAG, "CAN data collected: speed=${canData.vehicleSpeed} km/h, " +
                                "rpm=${canData.engineRPM}, window=${inferenceService.getSampleCount()}/60")
                    } else {
                        Log.w(TAG, "Invalid CAN data received")
                    }

                    // Sleep to maintain 1Hz rate
                    val elapsed = System.currentTimeMillis() - startTime
                    val sleepTime = 1000 - elapsed
                    if (sleepTime > 0) {
                        delay(sleepTime)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in CAN data collection", e)
                updateNotification("Error: CAN data collection failed")
            }
        }
    }

    /**
     * Start AI inference scheduler (every 60 seconds)
     */
    private fun startInferenceScheduler() {
        inferenceJob = serviceScope.launch(Dispatchers.Default) {
            Log.i(TAG, "Starting AI inference scheduler (60s interval)...")

            // Wait for initial buffer to fill
            delay(60000)  // 60 seconds

            while (isActive && isRunning) {
                val startTime = System.currentTimeMillis()

                try {
                    // Check if 60-second window is ready
                    if (inferenceService.isReady()) {
                        Log.i(TAG, "Running AI inference (window ready: ${inferenceService.getSampleCount()}/60)")
                        updateNotification("Running AI inference...")

                        // Run LightGBM inference with confidence scores
                        val inferenceResult = inferenceService.runInferenceWithConfidence()

                        if (inferenceResult != null) {
                            // Create AIInferenceResult for legacy compatibility
                            val result = AIInferenceResult(
                                timestamp = inferenceResult.timestamp,
                                fuelEfficiencyPrediction = 0.0f,  // TODO: TCN model
                                anomalyScore = 0.0f,  // TODO: LSTM-AE model
                                behaviorClass = inferenceResult.behavior,
                                safetyScore = calculateSafetyScore(inferenceResult),
                                carbonEmission = 0.0f,  // TODO: Calculate from fuel
                                anomalies = emptyList(),  // TODO: Detect from anomaly score
                                inferenceLatency = inferenceResult.latencyMs
                            )

                            // Send to MQTT
                            publishToMQTT(result, inferenceResult)

                            // Broadcast via BLE
                            broadcastInferenceResult(result)

                            totalInferencesRun++

                            Log.i(TAG, "Inference completed: " +
                                    "behavior=${result.behaviorClass} (confidence=${inferenceResult.confidence}), " +
                                    "safety=${result.safetyScore}, " +
                                    "latency=${result.inferenceLatency}ms")

                            updateNotification("DTG active - Safety: ${result.safetyScore}/100")
                        } else {
                            Log.w(TAG, "Inference returned null")
                        }
                    } else {
                        Log.d(TAG, "Window not ready: ${inferenceService.getSampleCount()}/60 samples")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in AI inference", e)
                    updateNotification("Error: AI inference failed")
                }

                // Sleep until next inference (60 seconds from start)
                val elapsed = System.currentTimeMillis() - startTime
                val sleepTime = 60000 - elapsed
                if (sleepTime > 0) {
                    delay(sleepTime)
                }
            }
        }
    }

    /**
     * Calculate safety score from inference result (0-100)
     */
    private fun calculateSafetyScore(
        inferenceResult: com.glec.dtg.inference.InferenceResult
    ): Int {
        var score = 100

        // Deduct based on driving behavior classification
        score -= when (inferenceResult.behavior) {
            DrivingBehavior.ECO_DRIVING -> 0    // Perfect
            DrivingBehavior.NORMAL -> 5         // Good
            DrivingBehavior.AGGRESSIVE -> 25    // Dangerous
            DrivingBehavior.HARSH_BRAKING -> 15
            DrivingBehavior.HARSH_ACCELERATION -> 15
            DrivingBehavior.SPEEDING -> 20
            DrivingBehavior.ANOMALY -> 30
        }

        // Bonus for high confidence eco driving
        if (inferenceResult.behavior == DrivingBehavior.ECO_DRIVING &&
            inferenceResult.confidence > 0.9f) {
            score += 5  // Bonus for very consistent eco driving
        }

        // Penalty for low confidence predictions (uncertain behavior)
        if (inferenceResult.confidence < 0.7f) {
            score -= 10  // Deduct for inconsistent driving patterns
        }

        return score.coerceIn(0, 100)
    }

    /**
     * Publish inference result to MQTT broker
     */
    private fun publishToMQTT(
        result: AIInferenceResult,
        inferenceResult: com.glec.dtg.inference.InferenceResult
    ) {
        try {
            // Create JSON payload
            val payload = JSONObject().apply {
                put("timestamp", result.timestamp)
                put("device_id", mqttManager.deviceId)
                put("behavior", result.behaviorClass.name)
                put("confidence", inferenceResult.confidence)
                put("safety_score", result.safetyScore)
                put("latency_ms", result.inferenceLatency)

                // TODO: Add fuel efficiency and anomaly data when available
                // put("fuel_efficiency", result.fuelEfficiencyPrediction)
                // put("anomaly_score", result.anomalyScore)
            }.toString()

            // Publish with QoS 1 (at least once delivery)
            val success = mqttManager.publishInference(mqttManager.deviceId, payload)

            if (success) {
                Log.d(TAG, "✅ Published inference result to MQTT")
            } else {
                Log.w(TAG, "⚠️ Failed to publish to MQTT (queued for retry)")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error publishing to MQTT", e)
        }
    }

    /**
     * Publish real-time telemetry data to MQTT (1Hz, QoS 0)
     */
    private fun publishTelemetry(canData: CANData) {
        try {
            // Create JSON payload
            val payload = JSONObject().apply {
                put("timestamp", System.currentTimeMillis())
                put("device_id", mqttManager.deviceId)
                put("vehicle_speed", canData.vehicleSpeed)
                put("engine_rpm", canData.engineRPM)
                put("throttle_position", canData.throttlePosition)
                put("fuel_level", canData.fuelLevel)
                put("coolant_temp", canData.coolantTemp)
                put("brake_position", canData.brakePosition)
                put("acceleration_x", canData.accelerationX)
                put("acceleration_y", canData.accelerationY)
                put("acceleration_z", canData.accelerationZ)
                put("steering_angle", canData.steeringAngle)

                // GPS data (if available)
                put("gps", JSONObject().apply {
                    put("lat", 0.0)  // TODO: Get from GPS
                    put("lon", 0.0)
                    put("speed", canData.vehicleSpeed)
                })
            }.toString()

            // Publish with QoS 0 (fire and forget - high frequency data)
            val success = mqttManager.publishTelemetry(mqttManager.deviceId, payload)

            if (!success) {
                Log.v(TAG, "Telemetry queued for retry")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error publishing telemetry", e)
        }
    }

    /**
     * Publish device status to MQTT (5min, QoS 1)
     */
    private fun publishStatus() {
        try {
            val uptime = System.currentTimeMillis() - dataCollectionStartTime

            // Create JSON payload
            val payload = JSONObject().apply {
                put("timestamp", System.currentTimeMillis())
                put("device_id", mqttManager.deviceId)
                put("status", "ONLINE")
                put("uptime_ms", uptime)
                put("samples_collected", totalSamplesCollected)
                put("inferences_run", totalInferencesRun)
                put("mqtt_metrics", JSONObject().apply {
                    val metrics = mqttManager.getMetrics()
                    put("connected", metrics.isConnected)
                    put("messages_sent", metrics.messagesSent)
                    put("messages_failed", metrics.messagesFailed)
                    put("messages_queued", metrics.messagesQueued)
                    put("reconnect_count", metrics.reconnectCount)
                })
                put("inference_ready", inferenceService.isReady())
                put("window_size", inferenceService.getSampleCount())
            }.toString()

            // Publish with QoS 1 (at least once delivery)
            val success = mqttManager.publishStatus(mqttManager.deviceId, payload)

            if (success) {
                Log.i(TAG, "✅ Published device status to MQTT")
            } else {
                Log.w(TAG, "⚠️ Failed to publish status (queued for retry)")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error publishing status", e)
        }
    }

    /**
     * Publish safety alert to MQTT (QoS 2 - critical)
     */
    private fun publishAlert(
        alertType: String,
        severity: String,
        message: String,
        canData: CANData
    ) {
        try {
            // Create JSON payload
            val payload = JSONObject().apply {
                put("timestamp", System.currentTimeMillis())
                put("device_id", mqttManager.deviceId)
                put("alert_type", alertType)
                put("severity", severity)
                put("message", message)
                put("vehicle_data", JSONObject().apply {
                    put("speed", canData.vehicleSpeed)
                    put("rpm", canData.engineRPM)
                    put("throttle", canData.throttlePosition)
                    put("brake", canData.brakePosition)
                    put("coolant_temp", canData.coolantTemp)
                    put("fuel_level", canData.fuelLevel)
                })
            }.toString()

            // Publish with QoS 2 (exactly once delivery - critical alerts)
            val success = mqttManager.publishAlert(mqttManager.deviceId, payload)

            if (success) {
                Log.w(TAG, "🚨 Published alert to MQTT: $alertType ($severity)")
            } else {
                Log.e(TAG, "❌ Failed to publish alert (queued for retry)")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error publishing alert", e)
        }
    }

    /**
     * Start device status publisher (every 5 minutes)
     */
    private fun startStatusPublisher() {
        statusJob = serviceScope.launch(Dispatchers.Default) {
            Log.i(TAG, "Starting device status publisher (5min interval)...")

            // Delay initial status by 10 seconds
            delay(10000)

            while (isActive && isRunning) {
                val startTime = System.currentTimeMillis()

                try {
                    publishStatus()
                } catch (e: Exception) {
                    Log.e(TAG, "Error in status publisher", e)
                }

                // Sleep until next status (5 minutes from start)
                val elapsed = System.currentTimeMillis() - startTime
                val sleepTime = 5 * 60 * 1000 - elapsed  // 5 minutes
                if (sleepTime > 0) {
                    delay(sleepTime)
                }
            }
        }
    }

    /**
     * Detect immediate anomalies (real-time)
     */
    private fun detectImmediateAnomalies(canData: CANData) {
        if (canData.isHarshBraking()) {
            Log.w(TAG, "⚠️ Harsh braking detected!")
            publishAlert(
                alertType = "HARSH_BRAKING",
                severity = "WARNING",
                message = "Harsh braking detected: deceleration < -4 m/s²",
                canData = canData
            )
        }

        if (canData.isHarshAcceleration()) {
            Log.w(TAG, "⚠️ Harsh acceleration detected!")
            publishAlert(
                alertType = "HARSH_ACCELERATION",
                severity = "WARNING",
                message = "Harsh acceleration detected: acceleration > 3 m/s²",
                canData = canData
            )
        }

        if (canData.coolantTemp > 105) {
            Log.e(TAG, "🔥 Engine overheating! Temp: ${canData.coolantTemp}°C")
            publishAlert(
                alertType = "ENGINE_OVERHEATING",
                severity = "CRITICAL",
                message = "Engine overheating: ${canData.coolantTemp}°C (threshold: 105°C)",
                canData = canData
            )
        }

        if (canData.fuelLevel < 10.0f) {
            Log.w(TAG, "⚠️ Low fuel: ${canData.fuelLevel}%")
            publishAlert(
                alertType = "LOW_FUEL",
                severity = "INFO",
                message = "Low fuel level: ${canData.fuelLevel}%",
                canData = canData
            )
        }
    }

    /**
     * Broadcast inference result via BLE
     */
    private fun broadcastInferenceResult(result: AIInferenceResult) {
        // TODO: Implement BLE broadcast to Driver app
        Log.d(TAG, "Broadcasting result via BLE: safety=${result.safetyScore}")
    }

    /**
     * Create foreground notification
     */
    private fun createNotification(message: String): Notification {
        createNotificationChannel()

        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            packageManager.getLaunchIntentForPackage(packageName),
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(message)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    /**
     * Update notification message
     */
    private fun updateNotification(message: String) {
        val notification = createNotification(message)
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    /**
     * Create notification channel (Android O+)
     */
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_description)
            }

            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }

    companion object {
        private const val TAG = "DTGForegroundService"

        const val ACTION_START_SERVICE = "com.glec.dtg.ACTION_START_SERVICE"
        const val ACTION_STOP_SERVICE = "com.glec.dtg.ACTION_STOP_SERVICE"
        const val ACTION_RESTART_SERVICE = "com.glec.dtg.ACTION_RESTART_SERVICE"

        private const val CHANNEL_ID = "DTG_SERVICE_CHANNEL"
        private const val NOTIFICATION_ID = 1001
    }
}

/**
 * Placeholder classes (to be implemented)
 */
private class CANReceiver(private val context: Context) {
    fun open() {
        // TODO: Open UART connection
    }

    fun close() {
        // TODO: Close UART connection
    }

    fun readCANData(): CANData? {
        // TODO: Read from UART and parse CAN data
        return null
    }
}
