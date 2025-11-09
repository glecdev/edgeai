package com.glec.driver.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.glec.driver.R
import com.glec.driver.ble.BLEManager
import com.glec.driver.databinding.ActivityMainBinding
import com.glec.driver.voice.VoiceAssistant

/**
 * GLEC Driver - Main Activity
 * Driver smartphone application for fleet management
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: MainViewModel

    private lateinit var bleManager: BLEManager
    private lateinit var voiceAssistant: VoiceAssistant

    private val requiredPermissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.RECORD_AUDIO
        )
    } else {
        arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.RECORD_AUDIO
        )
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            initializeBLE()
            initializeVoice()
        } else {
            Toast.makeText(this, "Permissions required for app functionality", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel = ViewModelProvider(this)[MainViewModel::class.java]

        if (checkPermissions()) {
            initializeBLE()
            initializeVoice()
        } else {
            requestPermissions()
        }

        setupViews()
        observeViewModel()
    }

    private fun setupViews() {
        // BLE connection button
        binding.buttonConnectBLE.setOnClickListener {
            if (viewModel.bleConnectionState.value == BLEConnectionState.DISCONNECTED) {
                connectToDTG()
            } else {
                disconnectFromDTG()
            }
        }

        // Voice command button
        binding.buttonVoiceCommand.setOnClickListener {
            startVoiceCommand()
        }

        // External data refresh
        binding.buttonRefreshData.setOnClickListener {
            refreshExternalData()
        }

        // Dispatch actions
        binding.buttonAcceptDispatch.setOnClickListener {
            viewModel.acceptDispatch()
        }

        binding.buttonRejectDispatch.setOnClickListener {
            viewModel.rejectDispatch()
        }
    }

    private fun observeViewModel() {
        // BLE connection state
        viewModel.bleConnectionState.observe(this) { state ->
            when (state) {
                BLEConnectionState.DISCONNECTED -> {
                    binding.textBLEStatus.text = "Disconnected"
                    binding.textBLEStatus.setTextColor(ContextCompat.getColor(this, R.color.status_error))
                    binding.buttonConnectBLE.text = "Connect to DTG"
                }
                BLEConnectionState.SCANNING -> {
                    binding.textBLEStatus.text = "Scanning..."
                    binding.textBLEStatus.setTextColor(ContextCompat.getColor(this, R.color.status_warning))
                }
                BLEConnectionState.CONNECTING -> {
                    binding.textBLEStatus.text = "Connecting..."
                    binding.textBLEStatus.setTextColor(ContextCompat.getColor(this, R.color.status_warning))
                }
                BLEConnectionState.CONNECTED -> {
                    binding.textBLEStatus.text = "Connected"
                    binding.textBLEStatus.setTextColor(ContextCompat.getColor(this, R.color.status_success))
                    binding.buttonConnectBLE.text = "Disconnect"
                }
            }
        }

        // Vehicle data
        viewModel.vehicleData.observe(this) { data ->
            data?.let {
                binding.textVehicleSpeed.text = getString(R.string.speed_format, it.speed)
                binding.textFuelLevel.text = getString(R.string.fuel_format, it.fuelLevel)
                binding.textEngineTemp.text = getString(R.string.temp_format, it.engineTemp)
            }
        }

        // AI results
        viewModel.aiResults.observe(this) { results ->
            results?.let {
                binding.textFuelEfficiency.text = getString(R.string.fuel_efficiency_format, it.fuelEfficiency)
                binding.textSafetyScore.text = getString(R.string.safety_score_format, it.safetyScore)
                binding.textCarbonEmission.text = getString(R.string.carbon_format, it.carbonEmission)
                binding.textDrivingBehavior.text = it.behaviorClass

                // Update safety score color
                val safetyColor = when {
                    it.safetyScore >= 80 -> ContextCompat.getColor(this, R.color.status_success)
                    it.safetyScore >= 60 -> ContextCompat.getColor(this, R.color.status_warning)
                    else -> ContextCompat.getColor(this, R.color.status_error)
                }
                binding.textSafetyScore.setTextColor(safetyColor)
            }
        }

        // Weather data
        viewModel.weatherData.observe(this) { weather ->
            weather?.let {
                binding.textTemperature.text = getString(R.string.temperature_format, it.temperature)
                binding.textHumidity.text = getString(R.string.humidity_format, it.humidity)
                binding.textSkyCondition.text = it.skyCondition
            }
        }

        // Traffic data
        viewModel.trafficData.observe(this) { traffic ->
            traffic?.let {
                binding.textTrafficLevel.text = it.congestionLevel
                binding.textAverageSpeed.text = getString(R.string.speed_format, it.averageSpeed)

                // Update traffic color
                val trafficColor = when (it.congestionLevel) {
                    "원활" -> ContextCompat.getColor(this, R.color.status_success)
                    "서행" -> ContextCompat.getColor(this, R.color.status_warning)
                    else -> ContextCompat.getColor(this, R.color.status_error)
                }
                binding.textTrafficLevel.setTextColor(trafficColor)
            }
        }

        // Voice command status
        viewModel.voiceCommandActive.observe(this) { active ->
            binding.buttonVoiceCommand.isEnabled = !active
            if (active) {
                binding.textVoiceStatus.text = "Listening..."
            } else {
                binding.textVoiceStatus.text = "Ready"
            }
        }

        // Toast messages
        viewModel.toastMessage.observe(this) { message ->
            message?.let {
                Toast.makeText(this, it, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun initializeBLE() {
        bleManager = BLEManager(this).apply {
            connectionCallback = object : BLEManager.ConnectionCallback {
                override fun onConnected() {
                    viewModel.updateBLEState(BLEConnectionState.CONNECTED)
                    viewModel.showToast("Connected to DTG device")
                }

                override fun onDisconnected() {
                    viewModel.updateBLEState(BLEConnectionState.DISCONNECTED)
                    viewModel.showToast("Disconnected from DTG device")
                }

                override fun onConnectionFailed(reason: String) {
                    viewModel.updateBLEState(BLEConnectionState.DISCONNECTED)
                    viewModel.showToast("Connection failed: $reason")
                }
            }

            dataCallback = object : BLEManager.DataCallback {
                override fun onVehicleDataReceived(data: ByteArray) {
                    viewModel.processVehicleData(data)
                }

                override fun onAIResultsReceived(data: ByteArray) {
                    viewModel.processAIResults(data)
                }
            }
        }
    }

    private fun initializeVoice() {
        voiceAssistant = VoiceAssistant(this).apply {
            initialize()

            commandCallback = object : VoiceAssistant.CommandCallback {
                override fun onCommandReceived(intent: VoiceAssistant.VoiceIntent) {
                    viewModel.handleVoiceCommand(intent)
                }
            }

            startListeningForWakeWord()
        }
    }

    private fun connectToDTG() {
        viewModel.updateBLEState(BLEConnectionState.SCANNING)
        bleManager.startScan()
    }

    private fun disconnectFromDTG() {
        bleManager.disconnect()
        viewModel.updateBLEState(BLEConnectionState.DISCONNECTED)
    }

    private fun startVoiceCommand() {
        viewModel.setVoiceCommandActive(true)
        // Voice assistant is already listening for wake word
        viewModel.showToast("Say: 헤이 드라이버")
    }

    private fun refreshExternalData() {
        // TODO: Get current location and refresh external data
        val latitude = 37.5665  // Seoul (placeholder)
        val longitude = 126.9780

        viewModel.refreshExternalData(latitude, longitude)
        viewModel.showToast("Refreshing data...")
    }

    private fun checkPermissions(): Boolean {
        return requiredPermissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestPermissions() {
        permissionLauncher.launch(requiredPermissions)
    }

    override fun onDestroy() {
        super.onDestroy()

        if (::bleManager.isInitialized) {
            bleManager.disconnect()
        }

        if (::voiceAssistant.isInitialized) {
            voiceAssistant.shutdown()
        }
    }
}

/**
 * BLE connection state
 */
enum class BLEConnectionState {
    DISCONNECTED,
    SCANNING,
    CONNECTING,
    CONNECTED
}
