package com.glec.dtg

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.glec.dtg.service.DTGForegroundService
import com.google.android.material.button.MaterialButton
import com.google.android.material.textview.MaterialTextView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import timber.log.Timber
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * GLEC DTG Main Activity
 * Displays status and allows starting/stopping DTG service
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusTextView: MaterialTextView
    private lateinit var startButton: MaterialButton
    private lateinit var stopButton: MaterialButton
    private lateinit var voiceQueryButton: MaterialButton

    // Real-time CAN data views
    private lateinit var tvVehicleSpeed: MaterialTextView
    private lateinit var tvEngineRPM: MaterialTextView
    private lateinit var tvFuelLevel: MaterialTextView
    private lateinit var tvCoolantTemp: MaterialTextView

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private val REQUIRED_PERMISSIONS = buildList {
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            add(Manifest.permission.ACCESS_COARSE_LOCATION)
            add(Manifest.permission.BLUETOOTH_CONNECT)
            add(Manifest.permission.BLUETOOTH_ADVERTISE)

            // Android 13+ notification permission (critical for foreground service)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }.toTypedArray()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Timber.i("MainActivity onCreate")

        initViews()
        setupClickListeners()
        checkPermissions()
        requestBatteryOptimizationExemption()
    }

    override fun onResume() {
        super.onResume()
        // Update UI when returning to activity
        updateUI()
    }

    private fun initViews() {
        statusTextView = findViewById(R.id.statusTextView)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        voiceQueryButton = findViewById(R.id.voiceQueryButton)

        // Real-time CAN data views
        tvVehicleSpeed = findViewById(R.id.tvVehicleSpeed)
        tvEngineRPM = findViewById(R.id.tvEngineRPM)
        tvFuelLevel = findViewById(R.id.tvFuelLevel)
        tvCoolantTemp = findViewById(R.id.tvCoolantTemp)

        updateUI()
        startRealtimeDataUpdates()
    }

    private fun setupClickListeners() {
        startButton.setOnClickListener {
            startDTGService()
        }

        stopButton.setOnClickListener {
            stopDTGService()
        }

        voiceQueryButton.setOnClickListener {
            testVoiceQuery()
        }
    }

    private fun startDTGService() {
        Timber.i("Starting DTG Foreground Service")

        val intent = Intent(this, DTGForegroundService::class.java)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }

        updateUI()
    }

    private fun stopDTGService() {
        Timber.i("Stopping DTG Foreground Service")

        val intent = Intent(this, DTGForegroundService::class.java)
        stopService(intent)

        updateUI()
    }

    private fun updateUI() {
        val isRunning = isServiceRunning(DTGForegroundService::class.java)

        statusTextView.text = if (isRunning) {
            "DTG Service Status: Running ✓"
        } else {
            "DTG Service Status: Stopped"
        }

        startButton.isEnabled = !isRunning
        stopButton.isEnabled = isRunning
    }

    private fun isServiceRunning(serviceClass: Class<*>): Boolean {
        val manager = getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
        @Suppress("DEPRECATION")
        for (service in manager.getRunningServices(Integer.MAX_VALUE)) {
            if (serviceClass.name == service.service.className) {
                return true
            }
        }
        return false
    }

    private fun checkPermissions() {
        val permissionsToRequest = mutableListOf<String>()

        for (permission in REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission)
                != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(permission)
            }
        }

        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissionsToRequest.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun requestBatteryOptimizationExemption() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val powerManager = getSystemService(POWER_SERVICE) as PowerManager
            val packageName = packageName

            if (!powerManager.isIgnoringBatteryOptimizations(packageName)) {
                Timber.w("App is not exempted from battery optimization")

                val intent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                startActivity(intent)
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSION_REQUEST_CODE) {
            val deniedPermissions = permissions.filterIndexed { index, _ ->
                grantResults[index] != PackageManager.PERMISSION_GRANTED
            }

            if (deniedPermissions.isNotEmpty()) {
                Timber.w("Denied permissions: $deniedPermissions")
                statusTextView.text = "Please grant all permissions"
            } else {
                Timber.i("All permissions granted")
                statusTextView.text = "Permissions granted"
            }
        }
    }

    private fun startRealtimeDataUpdates() {
        // Update CAN data every second
        val handler = android.os.Handler(android.os.Looper.getMainLooper())
        val updateTask = object : Runnable {
            override fun run() {
                updateRealtimeData()
                handler.postDelayed(this, 1000) // Update every 1 second
            }
        }
        handler.post(updateTask)
    }

    private fun updateRealtimeData() {
        val isRunning = isServiceRunning(DTGForegroundService::class.java)

        if (isRunning) {
            // Simulate real-time CAN data (in production, get from service)
            // This is mock data that changes over time for visualization
            val speed = (40 + (Math.random() * 60)).toInt() // 40-100 km/h
            val rpm = (1200 + (Math.random() * 1500)).toInt() // 1200-2700 rpm
            val fuel = (60 + (Math.random() * 30)).toInt() // 60-90%
            val temp = (80 + (Math.random() * 15)).toInt() // 80-95°C

            tvVehicleSpeed.text = speed.toString()
            tvEngineRPM.text = rpm.toString()
            tvFuelLevel.text = fuel.toString()
            tvCoolantTemp.text = temp.toString()
        } else {
            // Service stopped - show "--"
            tvVehicleSpeed.text = "--"
            tvEngineRPM.text = "--"
            tvFuelLevel.text = "--"
            tvCoolantTemp.text = "--"
        }
    }

    /**
     * Test voice query functionality (production-grade)
     *
     * Features:
     * - Try-catch error handling
     * - User-friendly error messages
     * - Timber logging for debugging
     * - Button state management (disable during processing)
     * - MaterialAlertDialog for results
     */
    private fun testVoiceQuery() {
        Timber.i("Voice query button clicked")

        // Disable button during processing
        voiceQueryButton.isEnabled = false
        voiceQueryButton.text = "Processing..."

        CoroutineScope(Dispatchers.Main).launch {
            try {
                // Run voice test on IO thread
                val result = withContext(Dispatchers.IO) {
                    SimpleVoiceTest.testVoice(this@MainActivity)
                }

                // Show results in dialog
                MaterialAlertDialogBuilder(this@MainActivity)
                    .setTitle("Voice Query Test Result")
                    .setMessage(result)
                    .setPositiveButton("OK", null)
                    .show()

                Timber.i("Voice query test succeeded")

            } catch (e: Exception) {
                Timber.e(e, "Voice query test failed")

                // Show error dialog
                MaterialAlertDialogBuilder(this@MainActivity)
                    .setTitle("Voice Query Error")
                    .setMessage("Error: ${e.message}\n\nPlease check logs for details.")
                    .setPositiveButton("OK", null)
                    .show()

            } finally {
                // Re-enable button
                voiceQueryButton.isEnabled = true
                voiceQueryButton.text = "Ask Voice Question (Test)"
            }
        }
    }
}
