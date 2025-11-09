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
import com.glec.dtg.utils.CANMessageParser
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

    private val canDataBuffer = ConcurrentLinkedQueue<CANData>()
    private val maxBufferSize = 60  // 60 seconds at 1Hz

    private var wakeLock: PowerManager.WakeLock? = null

    private lateinit var canReceiver: CANReceiver
    private lateinit var inferenceEngine: SNPEInferenceEngine
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
        inferenceEngine = SNPEInferenceEngine(this)
        mqttClient = MQTTClientService(this)

        Log.i(TAG, "Components initialized")
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

        // Initialize AI models
        serviceScope.launch {
            try {
                inferenceEngine.loadModels()
                Log.i(TAG, "AI models loaded successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load AI models", e)
                updateNotification("Error: Failed to load AI models")
                return@launch
            }

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

        // Clear buffer
        canDataBuffer.clear()

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
                        // Add to buffer
                        canDataBuffer.offer(canData)
                        totalSamplesCollected++

                        // Maintain buffer size (60 seconds)
                        while (canDataBuffer.size > maxBufferSize) {
                            canDataBuffer.poll()
                        }

                        // Detect immediate anomalies
                        detectImmediateAnomalies(canData)

                        Log.d(TAG, "CAN data collected: speed=${canData.vehicleSpeed} km/h, " +
                                "rpm=${canData.engineRPM}, buffer=${canDataBuffer.size}")
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
                    // Get last 60 seconds of data
                    val dataWindow = canDataBuffer.toList()

                    if (dataWindow.size >= 30) {  // Minimum 30 samples
                        Log.i(TAG, "Running AI inference on ${dataWindow.size} samples...")
                        updateNotification("Running AI inference...")

                        // Run inference
                        val result = runInference(dataWindow)

                        // Send to MQTT
                        if (mqttClient.isConnected()) {
                            mqttClient.publishInferenceResult(result)
                        }

                        // Broadcast via BLE
                        broadcastInferenceResult(result)

                        totalInferencesRun++

                        Log.i(TAG, "Inference completed: latency=${result.inferenceLatency}ms, " +
                                "behavior=${result.behaviorClass}, safety=${result.safetyScore}")

                        updateNotification("DTG active - Safety: ${result.safetyScore}/100")
                    } else {
                        Log.w(TAG, "Insufficient data for inference: ${dataWindow.size} samples")
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
     * Run AI inference on data window
     */
    private suspend fun runInference(dataWindow: List<CANData>): AIInferenceResult {
        return withContext(Dispatchers.Default) {
            val inferenceStartTime = System.currentTimeMillis()

            // Parallel inference using async
            val fuelPredictionDeferred = async {
                inferenceEngine.runTCN(dataWindow)
            }

            val anomalyScoreDeferred = async {
                inferenceEngine.runLSTMAE(dataWindow)
            }

            val behaviorClassDeferred = async {
                inferenceEngine.runLightGBM(dataWindow)
            }

            // Await all results
            val fuelPrediction = fuelPredictionDeferred.await()
            val anomalyScore = anomalyScoreDeferred.await()
            val behaviorClass = behaviorClassDeferred.await()

            // Calculate safety score (0-100)
            val safetyScore = calculateSafetyScore(dataWindow, anomalyScore, behaviorClass)

            // Calculate carbon emission (g CO₂/km)
            val carbonEmission = fuelPrediction * 2.31f  // 1L gasoline ≈ 2.31 kg CO₂

            // Detect anomalies
            val anomalies = detectAnomalies(dataWindow, anomalyScore)

            val inferenceLatency = System.currentTimeMillis() - inferenceStartTime

            AIInferenceResult(
                timestamp = System.currentTimeMillis(),
                fuelEfficiencyPrediction = fuelPrediction,
                anomalyScore = anomalyScore,
                behaviorClass = behaviorClass,
                safetyScore = safetyScore,
                carbonEmission = carbonEmission,
                anomalies = anomalies,
                inferenceLatency = inferenceLatency
            )
        }
    }

    /**
     * Calculate safety score (0-100)
     */
    private fun calculateSafetyScore(
        dataWindow: List<CANData>,
        anomalyScore: Float,
        behaviorClass: com.glec.dtg.models.DrivingBehavior
    ): Int {
        var score = 100

        // Deduct for anomaly score
        score -= (anomalyScore * 30).toInt()

        // Deduct for harsh events
        val harshBrakingCount = dataWindow.count { it.isHarshBraking() }
        val harshAccelCount = dataWindow.count { it.isHarshAcceleration() }
        score -= min(harshBrakingCount * 5, 20)
        score -= min(harshAccelCount * 5, 20)

        // Deduct for speeding
        val speedingCount = dataWindow.count { it.vehicleSpeed > 100 }
        score -= min(speedingCount * 2, 15)

        // Deduct for behavior class
        score -= when (behaviorClass) {
            com.glec.dtg.models.DrivingBehavior.ECO_DRIVING -> 0
            com.glec.dtg.models.DrivingBehavior.NORMAL -> 5
            com.glec.dtg.models.DrivingBehavior.HARSH_BRAKING -> 15
            com.glec.dtg.models.DrivingBehavior.HARSH_ACCELERATION -> 15
            com.glec.dtg.models.DrivingBehavior.SPEEDING -> 20
            com.glec.dtg.models.DrivingBehavior.AGGRESSIVE -> 25
            com.glec.dtg.models.DrivingBehavior.ANOMALY -> 30
        }

        return score.coerceIn(0, 100)
    }

    /**
     * Detect anomalies from inference results
     */
    private fun detectAnomalies(
        dataWindow: List<CANData>,
        anomalyScore: Float
    ): List<com.glec.dtg.models.AnomalyType> {
        val anomalies = mutableListOf<com.glec.dtg.models.AnomalyType>()

        // High anomaly score
        if (anomalyScore > 0.7f) {
            anomalies.add(com.glec.dtg.models.AnomalyType.ABNORMAL_PATTERN)
        }

        // Check latest data point for specific anomalies
        val latest = dataWindow.lastOrNull() ?: return anomalies

        if (latest.isHarshBraking()) {
            anomalies.add(com.glec.dtg.models.AnomalyType.HARSH_BRAKING)
        }

        if (latest.isHarshAcceleration()) {
            anomalies.add(com.glec.dtg.models.AnomalyType.HARSH_ACCELERATION)
        }

        if (latest.vehicleSpeed > 100) {
            anomalies.add(com.glec.dtg.models.AnomalyType.SPEEDING)
        }

        if (latest.coolantTemp > 105) {
            anomalies.add(com.glec.dtg.models.AnomalyType.ENGINE_OVERHEATING)
        }

        if (latest.batteryVoltage < 11.5f) {
            anomalies.add(com.glec.dtg.models.AnomalyType.LOW_BATTERY)
        }

        return anomalies
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

private class SNPEInferenceEngine(private val context: Context) {
    suspend fun loadModels() {
        // TODO: Load SNPE models
    }

    suspend fun runTCN(dataWindow: List<CANData>): Float {
        // TODO: Run TCN inference
        return 12.5f  // Dummy fuel efficiency
    }

    suspend fun runLSTMAE(dataWindow: List<CANData>): Float {
        // TODO: Run LSTM-AE inference
        return 0.3f  // Dummy anomaly score
    }

    suspend fun runLightGBM(dataWindow: List<CANData>): com.glec.dtg.models.DrivingBehavior {
        // TODO: Run LightGBM inference
        return com.glec.dtg.models.DrivingBehavior.NORMAL
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
