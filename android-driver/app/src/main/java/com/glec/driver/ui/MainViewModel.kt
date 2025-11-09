package com.glec.driver.ui

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.glec.driver.api.ExternalDataService
import com.glec.driver.api.TrafficData
import com.glec.driver.api.WeatherData
import com.glec.driver.voice.VoiceAssistant
import kotlinx.coroutines.launch

/**
 * GLEC Driver - Main ViewModel
 * Manages driver app UI state and business logic
 */
class MainViewModel : ViewModel() {

    private val externalDataService = ExternalDataService()

    private val _bleConnectionState = MutableLiveData<BLEConnectionState>(BLEConnectionState.DISCONNECTED)
    val bleConnectionState: LiveData<BLEConnectionState> = _bleConnectionState

    private val _vehicleData = MutableLiveData<VehicleDataUI?>()
    val vehicleData: LiveData<VehicleDataUI?> = _vehicleData

    private val _aiResults = MutableLiveData<AIResultsUI?>()
    val aiResults: LiveData<AIResultsUI?> = _aiResults

    private val _weatherData = MutableLiveData<WeatherData?>()
    val weatherData: LiveData<WeatherData?> = _weatherData

    private val _trafficData = MutableLiveData<TrafficData?>()
    val trafficData: LiveData<TrafficData?> = _trafficData

    private val _voiceCommandActive = MutableLiveData<Boolean>(false)
    val voiceCommandActive: LiveData<Boolean> = _voiceCommandActive

    private val _toastMessage = MutableLiveData<String?>()
    val toastMessage: LiveData<String?> = _toastMessage

    /**
     * Update BLE connection state
     */
    fun updateBLEState(state: BLEConnectionState) {
        _bleConnectionState.postValue(state)
    }

    /**
     * Process vehicle data received via BLE
     */
    fun processVehicleData(data: ByteArray) {
        // Parse vehicle data from BLE packet
        // Format: [speed(4)][fuel_level(4)][engine_temp(4)]...

        try {
            val buffer = java.nio.ByteBuffer.wrap(data).order(java.nio.ByteOrder.LITTLE_ENDIAN)

            val speed = buffer.float
            val fuelLevel = buffer.float
            val engineTemp = buffer.float

            _vehicleData.postValue(
                VehicleDataUI(
                    speed = speed,
                    fuelLevel = fuelLevel,
                    engineTemp = engineTemp
                )
            )
        } catch (e: Exception) {
            showToast("Failed to parse vehicle data")
        }
    }

    /**
     * Process AI results received via BLE
     */
    fun processAIResults(data: ByteArray) {
        // Parse AI results from BLE packet
        // Format: [fuel_efficiency(4)][safety_score(4)][carbon_emission(4)][behavior_class_len(4)][behavior_class(...)]

        try {
            val buffer = java.nio.ByteBuffer.wrap(data).order(java.nio.ByteOrder.LITTLE_ENDIAN)

            val fuelEfficiency = buffer.float
            val safetyScore = buffer.int
            val carbonEmission = buffer.float

            val behaviorClassLen = buffer.int
            val behaviorClassBytes = ByteArray(behaviorClassLen)
            buffer.get(behaviorClassBytes)
            val behaviorClass = String(behaviorClassBytes)

            _aiResults.postValue(
                AIResultsUI(
                    fuelEfficiency = fuelEfficiency,
                    safetyScore = safetyScore,
                    carbonEmission = carbonEmission,
                    behaviorClass = behaviorClass
                )
            )
        } catch (e: Exception) {
            showToast("Failed to parse AI results")
        }
    }

    /**
     * Refresh external data (weather, traffic)
     */
    fun refreshExternalData(latitude: Double, longitude: Double) {
        viewModelScope.launch {
            // Fetch weather
            val weather = externalDataService.fetchWeather(latitude, longitude)
            _weatherData.postValue(weather)

            // Fetch traffic
            val traffic = externalDataService.fetchTraffic(latitude, longitude)
            _trafficData.postValue(traffic)
        }
    }

    /**
     * Handle voice command
     */
    fun handleVoiceCommand(intent: VoiceAssistant.VoiceIntent) {
        setVoiceCommandActive(false)

        when (intent) {
            VoiceAssistant.VoiceIntent.ACCEPT_DISPATCH -> {
                acceptDispatch()
            }
            VoiceAssistant.VoiceIntent.REJECT_DISPATCH -> {
                rejectDispatch()
            }
            VoiceAssistant.VoiceIntent.EMERGENCY_ALERT -> {
                sendEmergencyAlert()
            }
            VoiceAssistant.VoiceIntent.SHOW_LOCATION -> {
                showToast("Showing current location")
            }
            VoiceAssistant.VoiceIntent.SHOW_FUEL_INFO -> {
                val fuel = _vehicleData.value?.fuelLevel
                if (fuel != null) {
                    showToast("Fuel level: ${fuel}%")
                } else {
                    showToast("Fuel data not available")
                }
            }
            VoiceAssistant.VoiceIntent.SHOW_SAFETY_SCORE -> {
                val safety = _aiResults.value?.safetyScore
                if (safety != null) {
                    showToast("Safety score: $safety/100")
                } else {
                    showToast("Safety data not available")
                }
            }
            VoiceAssistant.VoiceIntent.START_NAVIGATION -> {
                showToast("Starting navigation")
            }
            VoiceAssistant.VoiceIntent.ARRIVE_DESTINATION -> {
                showToast("Marked as arrived")
            }
        }
    }

    /**
     * Accept dispatch
     */
    fun acceptDispatch() {
        // TODO: Send accept command to fleet platform
        showToast("Dispatch accepted")
    }

    /**
     * Reject dispatch
     */
    fun rejectDispatch() {
        // TODO: Send reject command to fleet platform
        showToast("Dispatch rejected")
    }

    /**
     * Send emergency alert
     */
    private fun sendEmergencyAlert() {
        // TODO: Send emergency alert to fleet platform
        showToast("Emergency alert sent")
    }

    /**
     * Set voice command active state
     */
    fun setVoiceCommandActive(active: Boolean) {
        _voiceCommandActive.postValue(active)
    }

    /**
     * Show toast message
     */
    fun showToast(message: String) {
        _toastMessage.postValue(message)
    }
}

/**
 * Vehicle data UI model
 */
data class VehicleDataUI(
    val speed: Float,
    val fuelLevel: Float,
    val engineTemp: Float
)

/**
 * AI results UI model
 */
data class AIResultsUI(
    val fuelEfficiency: Float,
    val safetyScore: Int,
    val carbonEmission: Float,
    val behaviorClass: String
)
