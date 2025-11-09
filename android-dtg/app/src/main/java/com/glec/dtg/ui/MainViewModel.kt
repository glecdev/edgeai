package com.glec.dtg.ui

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.glec.dtg.models.AIInferenceResult
import com.glec.dtg.models.CANData
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * GLEC DTG - Main ViewModel
 * Manages UI state and business logic
 */
class MainViewModel : ViewModel() {

    private val _serviceStatus = MutableLiveData<ServiceStatus>(ServiceStatus.READY)
    val serviceStatus: LiveData<ServiceStatus> = _serviceStatus

    private val _latestCANData = MutableLiveData<CANData?>()
    val latestCANData: LiveData<CANData?> = _latestCANData

    private val _latestInferenceResult = MutableLiveData<AIInferenceResult?>()
    val latestInferenceResult: LiveData<AIInferenceResult?> = _latestInferenceResult

    private val _statistics = MutableLiveData<ServiceStatistics>()
    val statistics: LiveData<ServiceStatistics> = _statistics

    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage

    init {
        // Initialize statistics
        _statistics.value = ServiceStatistics(
            samplesCollected = 0,
            inferencesRun = 0,
            uptimeMinutes = 0,
            mqttConnected = false
        )

        // Start periodic updates
        startPeriodicUpdates()
    }

    /**
     * Update service status
     */
    fun updateServiceStatus(status: ServiceStatus) {
        _serviceStatus.postValue(status)
    }

    /**
     * Update latest CAN data
     */
    fun updateCANData(canData: CANData) {
        _latestCANData.postValue(canData)

        // Update statistics
        val currentStats = _statistics.value ?: return
        _statistics.postValue(
            currentStats.copy(
                samplesCollected = currentStats.samplesCollected + 1
            )
        )
    }

    /**
     * Update latest inference result
     */
    fun updateInferenceResult(result: AIInferenceResult) {
        _latestInferenceResult.postValue(result)

        // Update statistics
        val currentStats = _statistics.value ?: return
        _statistics.postValue(
            currentStats.copy(
                inferencesRun = currentStats.inferencesRun + 1
            )
        )
    }

    /**
     * Update MQTT connection status
     */
    fun updateMQTTStatus(connected: Boolean) {
        val currentStats = _statistics.value ?: return
        _statistics.postValue(
            currentStats.copy(
                mqttConnected = connected
            )
        )
    }

    /**
     * Show error message
     */
    fun showError(message: String) {
        _errorMessage.postValue(message)
    }

    /**
     * Clear error message
     */
    fun clearError() {
        _errorMessage.postValue(null)
    }

    /**
     * Start periodic updates (uptime, etc.)
     */
    private fun startPeriodicUpdates() {
        viewModelScope.launch {
            var uptime = 0L

            while (true) {
                delay(60000)  // Update every minute

                if (_serviceStatus.value == ServiceStatus.RUNNING) {
                    uptime++

                    val currentStats = _statistics.value ?: continue
                    _statistics.postValue(
                        currentStats.copy(
                            uptimeMinutes = uptime
                        )
                    )
                } else {
                    uptime = 0
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        // Cleanup if needed
    }
}
