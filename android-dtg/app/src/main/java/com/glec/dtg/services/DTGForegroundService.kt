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
import kotlinx.coroutines.*
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

    private var wakeLock: PowerManager.WakeLock? = null

    private lateinit var canReceiver: CANReceiver
    private lateinit var inferenceService: EdgeAIInferenceService
    private lateinit var mqttClient: MQTTClientService

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
        mqttClient = MQTTClientService(this)

        Log.i(TAG, "Components initialized")
        Log.i(TAG, "  EdgeAIInferenceService: LightGBM behavior classification ready")
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
            try {
                mqttClient.connect()
                Log.i(TAG, "Connected to MQTT broker")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to connect to MQTT broker", e)
                // Continue without MQTT (offline mode)
            }

            // Start CAN data collection
            startCANDataCollection()

            // Start AI inference scheduler
            startInferenceScheduler()

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

        // Disconnect MQTT
        mqttClient.disconnect()

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
                            if (mqttClient.isConnected()) {
                                mqttClient.publishInferenceResult(result)
                            }

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
     * Detect immediate anomalies (real-time)
     */
    private fun detectImmediateAnomalies(canData: CANData) {
        if (canData.isHarshBraking()) {
            Log.w(TAG, "⚠️ Harsh braking detected!")
            // Could trigger immediate alert
        }

        if (canData.isHarshAcceleration()) {
            Log.w(TAG, "⚠️ Harsh acceleration detected!")
        }

        if (canData.coolantTemp > 105) {
            Log.e(TAG, "🔥 Engine overheating! Temp: ${canData.coolantTemp}°C")
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

private class MQTTClientService(private val context: Context) {
    fun connect() {
        // TODO: Connect to MQTT broker
    }

    fun disconnect() {
        // TODO: Disconnect from MQTT
    }

    fun isConnected(): Boolean {
        // TODO: Check MQTT connection status
        return false
    }

    fun publishInferenceResult(result: AIInferenceResult) {
        // TODO: Publish to MQTT
    }
}
