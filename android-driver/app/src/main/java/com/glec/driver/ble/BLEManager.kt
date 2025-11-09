package com.glec.driver.ble

import android.annotation.SuppressLint
import android.bluetooth.*
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.*

/**
 * GLEC Driver - BLE Manager
 * Manages Bluetooth Low Energy connection to DTG device
 *
 * GATT Service: 0000FFF0-0000-1000-8000-00805F9B34FB
 * - Characteristic (Read/Notify): Vehicle Data (0000FFF1)
 * - Characteristic (Write): Commands (0000FFF2)
 * - Characteristic (Read): AI Results (0000FFF3)
 */
@SuppressLint("MissingPermission")
class BLEManager(private val context: Context) {

    private val bluetoothAdapter: BluetoothAdapter? by lazy {
        val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothManager.adapter
    }

    private var bluetoothGatt: BluetoothGatt? = null
    private var connectionState = ConnectionState.DISCONNECTED

    private val handler = Handler(Looper.getMainLooper())
    private val scanTimeout = 30000L  // 30 seconds

    private var vehicleDataCharacteristic: BluetoothGattCharacteristic? = null
    private var commandCharacteristic: BluetoothGattCharacteristic? = null
    private var aiResultsCharacteristic: BluetoothGattCharacteristic? = null

    var connectionCallback: ConnectionCallback? = null
    var dataCallback: DataCallback? = null

    /**
     * Start scanning for DTG devices
     */
    fun startScan() {
        if (bluetoothAdapter == null || !bluetoothAdapter!!.isEnabled) {
            Log.e(TAG, "Bluetooth adapter not available or not enabled")
            connectionCallback?.onConnectionFailed("Bluetooth not enabled")
            return
        }

        Log.i(TAG, "Starting BLE scan for DTG devices...")

        val scanner = bluetoothAdapter?.bluetoothLeScanner
        if (scanner == null) {
            Log.e(TAG, "BLE scanner not available")
            connectionCallback?.onConnectionFailed("BLE scanner not available")
            return
        }

        // Scan filters for DTG device
        val filters = listOf(
            ScanFilter.Builder()
                .setServiceUuid(android.os.ParcelUuid(DTG_SERVICE_UUID))
                .build()
        )

        // Scan settings for balanced mode
        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_BALANCED)
            .setCallbackType(ScanSettings.CALLBACK_TYPE_ALL_MATCHES)
            .setMatchMode(ScanSettings.MATCH_MODE_AGGRESSIVE)
            .setNumOfMatches(ScanSettings.MATCH_NUM_ONE_ADVERTISEMENT)
            .build()

        scanner.startScan(filters, settings, scanCallback)

        // Stop scan after timeout
        handler.postDelayed({
            stopScan()
            if (connectionState == ConnectionState.DISCONNECTED) {
                Log.w(TAG, "Scan timeout, no DTG device found")
                connectionCallback?.onConnectionFailed("No DTG device found")
            }
        }, scanTimeout)

        connectionState = ConnectionState.SCANNING
    }

    /**
     * Stop scanning
     */
    fun stopScan() {
        bluetoothAdapter?.bluetoothLeScanner?.stopScan(scanCallback)
        Log.i(TAG, "BLE scan stopped")
    }

    /**
     * Connect to DTG device
     */
    fun connect(device: BluetoothDevice) {
        Log.i(TAG, "Connecting to DTG device: ${device.address}")

        bluetoothGatt = device.connectGatt(
            context,
            false,  // autoConnect = false for faster connection
            gattCallback,
            BluetoothDevice.TRANSPORT_LE
        )

        connectionState = ConnectionState.CONNECTING
    }

    /**
     * Disconnect from DTG device
     */
    fun disconnect() {
        Log.i(TAG, "Disconnecting from DTG device")

        bluetoothGatt?.let { gatt ->
            gatt.disconnect()
            gatt.close()
        }

        bluetoothGatt = null
        vehicleDataCharacteristic = null
        commandCharacteristic = null
        aiResultsCharacteristic = null

        connectionState = ConnectionState.DISCONNECTED

        connectionCallback?.onDisconnected()
    }

    /**
     * Send command to DTG device
     */
    fun sendCommand(command: DTGCommand) {
        if (connectionState != ConnectionState.CONNECTED) {
            Log.w(TAG, "Cannot send command: not connected")
            return
        }

        val characteristic = commandCharacteristic
        if (characteristic == null) {
            Log.e(TAG, "Command characteristic not available")
            return
        }

        val data = command.toByteArray()
        characteristic.value = data
        characteristic.writeType = BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT

        val success = bluetoothGatt?.writeCharacteristic(characteristic) ?: false
        if (success) {
            Log.d(TAG, "Command sent: $command")
        } else {
            Log.e(TAG, "Failed to send command: $command")
        }
    }

    /**
     * Request MTU size increase for better performance
     */
    private fun requestMtuSize() {
        val success = bluetoothGatt?.requestMtu(517) ?: false  // Max MTU: 517 bytes
        if (success) {
            Log.d(TAG, "MTU size increase requested")
        } else {
            Log.w(TAG, "Failed to request MTU size increase")
        }
    }

    /**
     * Enable notifications for vehicle data and AI results
     */
    private fun enableNotifications() {
        // Enable vehicle data notifications
        vehicleDataCharacteristic?.let { characteristic ->
            val success = bluetoothGatt?.setCharacteristicNotification(characteristic, true) ?: false
            if (success) {
                val descriptor = characteristic.getDescriptor(CCCD_UUID)
                descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                bluetoothGatt?.writeDescriptor(descriptor)
                Log.d(TAG, "Vehicle data notifications enabled")
            }
        }

        // Enable AI results notifications
        aiResultsCharacteristic?.let { characteristic ->
            val success = bluetoothGatt?.setCharacteristicNotification(characteristic, true) ?: false
            if (success) {
                val descriptor = characteristic.getDescriptor(CCCD_UUID)
                descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                bluetoothGatt?.writeDescriptor(descriptor)
                Log.d(TAG, "AI results notifications enabled")
            }
        }
    }

    /**
     * Scan callback
     */
    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val device = result.device
            Log.i(TAG, "DTG device found: ${device.address}, RSSI: ${result.rssi}")

            stopScan()
            connect(device)
        }

        override fun onScanFailed(errorCode: Int) {
            Log.e(TAG, "BLE scan failed: error code $errorCode")
            connectionCallback?.onConnectionFailed("Scan failed: error code $errorCode")
        }
    }

    /**
     * GATT callback
     */
    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    Log.i(TAG, "Connected to GATT server")
                    connectionState = ConnectionState.CONNECTED

                    // Request MTU increase
                    requestMtuSize()

                    // Discover services
                    handler.postDelayed({
                        gatt.discoverServices()
                    }, 600)  // Small delay before service discovery
                }

                BluetoothProfile.STATE_DISCONNECTED -> {
                    Log.i(TAG, "Disconnected from GATT server")
                    connectionState = ConnectionState.DISCONNECTED
                    connectionCallback?.onDisconnected()
                }
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.i(TAG, "Services discovered")

                val service = gatt.getService(DTG_SERVICE_UUID)
                if (service != null) {
                    vehicleDataCharacteristic = service.getCharacteristic(VEHICLE_DATA_UUID)
                    commandCharacteristic = service.getCharacteristic(COMMAND_UUID)
                    aiResultsCharacteristic = service.getCharacteristic(AI_RESULTS_UUID)

                    // Enable notifications
                    enableNotifications()

                    connectionCallback?.onConnected()
                    Log.i(TAG, "DTG service ready")
                } else {
                    Log.e(TAG, "DTG service not found")
                    disconnect()
                    connectionCallback?.onConnectionFailed("DTG service not found")
                }
            } else {
                Log.e(TAG, "Service discovery failed: status $status")
                disconnect()
                connectionCallback?.onConnectionFailed("Service discovery failed")
            }
        }

        override fun onMtuChanged(gatt: BluetoothGatt, mtu: Int, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.i(TAG, "MTU size changed to $mtu bytes")
            } else {
                Log.w(TAG, "MTU change failed: status $status")
            }
        }

        override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
            when (characteristic.uuid) {
                VEHICLE_DATA_UUID -> {
                    val data = characteristic.value
                    Log.d(TAG, "Vehicle data received: ${data.size} bytes")
                    dataCallback?.onVehicleDataReceived(data)
                }

                AI_RESULTS_UUID -> {
                    val data = characteristic.value
                    Log.d(TAG, "AI results received: ${data.size} bytes")
                    dataCallback?.onAIResultsReceived(data)
                }
            }
        }

        override fun onCharacteristicWrite(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d(TAG, "Characteristic write successful")
            } else {
                Log.e(TAG, "Characteristic write failed: status $status")
            }
        }
    }

    /**
     * Connection state
     */
    enum class ConnectionState {
        DISCONNECTED,
        SCANNING,
        CONNECTING,
        CONNECTED
    }

    /**
     * Connection callback interface
     */
    interface ConnectionCallback {
        fun onConnected()
        fun onDisconnected()
        fun onConnectionFailed(reason: String)
    }

    /**
     * Data callback interface
     */
    interface DataCallback {
        fun onVehicleDataReceived(data: ByteArray)
        fun onAIResultsReceived(data: ByteArray)
    }

    companion object {
        private const val TAG = "BLEManager"

        // GATT UUIDs
        private val DTG_SERVICE_UUID = UUID.fromString("0000FFF0-0000-1000-8000-00805F9B34FB")
        private val VEHICLE_DATA_UUID = UUID.fromString("0000FFF1-0000-1000-8000-00805F9B34FB")
        private val COMMAND_UUID = UUID.fromString("0000FFF2-0000-1000-8000-00805F9B34FB")
        private val AI_RESULTS_UUID = UUID.fromString("0000FFF3-0000-1000-8000-00805F9B34FB")
        private val CCCD_UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
    }
}

/**
 * DTG Commands
 */
sealed class DTGCommand {
    object RequestVehicleData : DTGCommand()
    object RequestAIResults : DTGCommand()
    data class SetInferenceInterval(val intervalSeconds: Int) : DTGCommand()
    object ResetStatistics : DTGCommand()

    fun toByteArray(): ByteArray {
        return when (this) {
            is RequestVehicleData -> byteArrayOf(0x01)
            is RequestAIResults -> byteArrayOf(0x02)
            is SetInferenceInterval -> byteArrayOf(0x03, intervalSeconds.toByte())
            is ResetStatistics -> byteArrayOf(0x04)
        }
    }
}
