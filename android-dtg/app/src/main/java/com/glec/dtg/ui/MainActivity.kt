package com.glec.dtg.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.glec.dtg.R
import com.glec.dtg.databinding.ActivityMainBinding
import com.glec.dtg.services.DTGForegroundService

/**
 * GLEC DTG - Main Activity
 * Displays service status and vehicle telemetry
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: MainViewModel

    private val requiredPermissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.POST_NOTIFICATIONS
        )
    } else {
        arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN
        )
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            startDTGService()
        } else {
            Toast.makeText(this, R.string.permission_denied, Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel = ViewModelProvider(this)[MainViewModel::class.java]

        setupViews()
        observeViewModel()
    }

    private fun setupViews() {
        // Service control buttons
        binding.buttonStartService.setOnClickListener {
            if (checkPermissions()) {
                startDTGService()
            } else {
                requestPermissions()
            }
        }

        binding.buttonStopService.setOnClickListener {
            stopDTGService()
        }

        binding.buttonRestartService.setOnClickListener {
            restartDTGService()
        }
    }

    private fun observeViewModel() {
        // Service status
        viewModel.serviceStatus.observe(this) { status ->
            binding.textServiceStatus.text = when (status) {
                ServiceStatus.RUNNING -> getString(R.string.status_running)
                ServiceStatus.STOPPED -> getString(R.string.status_stopped)
                ServiceStatus.ERROR -> getString(R.string.status_error)
                else -> getString(R.string.status_ready)
            }

            binding.buttonStartService.isEnabled = status != ServiceStatus.RUNNING
            binding.buttonStopService.isEnabled = status == ServiceStatus.RUNNING
            binding.buttonRestartService.isEnabled = status == ServiceStatus.RUNNING
        }

        // Latest CAN data
        viewModel.latestCANData.observe(this) { canData ->
            canData?.let {
                binding.textVehicleSpeed.text = getString(R.string.speed_value, it.vehicleSpeed)
                binding.textEngineRPM.text = getString(R.string.rpm_value, it.engineRPM)
                binding.textThrottlePosition.text = getString(R.string.throttle_value, it.throttlePosition)
                binding.textFuelLevel.text = getString(R.string.fuel_level_value, it.fuelLevel)
                binding.textCoolantTemp.text = getString(R.string.coolant_temp_value, it.coolantTemp)
                binding.textBatteryVoltage.text = getString(R.string.battery_voltage_value, it.batteryVoltage)
            }
        }

        // Latest AI inference result
        viewModel.latestInferenceResult.observe(this) { result ->
            result?.let {
                binding.textFuelEfficiency.text = getString(R.string.fuel_efficiency_value, it.fuelEfficiencyPrediction)
                binding.textSafetyScore.text = getString(R.string.safety_score_value, it.safetyScore)
                binding.textCarbonEmission.text = getString(R.string.carbon_emission_value, it.carbonEmission)
                binding.textBehaviorClass.text = it.behaviorClass.label
                binding.textInferenceLatency.text = getString(R.string.inference_latency_value, it.inferenceLatency)

                // Display anomalies
                if (it.anomalies.isNotEmpty()) {
                    val anomalyText = it.anomalies.joinToString("\n") { anomaly ->
                        "⚠️ ${anomaly.description}"
                    }
                    binding.textAnomalies.text = anomalyText
                } else {
                    binding.textAnomalies.text = getString(R.string.no_anomalies)
                }

                // Set safety score color
                val safetyColor = when {
                    it.safetyScore >= 80 -> ContextCompat.getColor(this, R.color.status_success)
                    it.safetyScore >= 60 -> ContextCompat.getColor(this, R.color.status_warning)
                    else -> ContextCompat.getColor(this, R.color.status_error)
                }
                binding.textSafetyScore.setTextColor(safetyColor)
            }
        }

        // Statistics
        viewModel.statistics.observe(this) { stats ->
            stats?.let {
                binding.textDataCollected.text = getString(R.string.data_collected_value, it.samplesCollected)
                binding.textInferencesRun.text = getString(R.string.inferences_run_value, it.inferencesRun)
                binding.textUptime.text = getString(R.string.uptime_value, it.uptimeMinutes)
                binding.textMQTTStatus.text = if (it.mqttConnected) {
                    getString(R.string.mqtt_connected)
                } else {
                    getString(R.string.mqtt_disconnected)
                }
            }
        }

        // Error messages
        viewModel.errorMessage.observe(this) { error ->
            error?.let {
                Toast.makeText(this, it, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun checkPermissions(): Boolean {
        return requiredPermissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestPermissions() {
        permissionLauncher.launch(requiredPermissions)
    }

    private fun startDTGService() {
        val intent = Intent(this, DTGForegroundService::class.java).apply {
            action = DTGForegroundService.ACTION_START_SERVICE
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }

        viewModel.updateServiceStatus(ServiceStatus.RUNNING)
        Toast.makeText(this, R.string.button_start_service, Toast.LENGTH_SHORT).show()
    }

    private fun stopDTGService() {
        val intent = Intent(this, DTGForegroundService::class.java).apply {
            action = DTGForegroundService.ACTION_STOP_SERVICE
        }
        startService(intent)

        viewModel.updateServiceStatus(ServiceStatus.STOPPED)
        Toast.makeText(this, R.string.button_stop_service, Toast.LENGTH_SHORT).show()
    }

    private fun restartDTGService() {
        val intent = Intent(this, DTGForegroundService::class.java).apply {
            action = DTGForegroundService.ACTION_RESTART_SERVICE
        }
        startService(intent)

        Toast.makeText(this, R.string.button_restart_service, Toast.LENGTH_SHORT).show()
    }
}

/**
 * Service status enum
 */
enum class ServiceStatus {
    READY,
    RUNNING,
    STOPPED,
    ERROR
}

/**
 * Service statistics
 */
data class ServiceStatistics(
    val samplesCollected: Long,
    val inferencesRun: Long,
    val uptimeMinutes: Long,
    val mqttConnected: Boolean
)
