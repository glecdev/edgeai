package com.glec.driver

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothProfile
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.textview.MaterialTextView
import timber.log.Timber

/**
 * GLEC Driver Application
 * Connects to DTG device via BLE and provides:
 * - Real-time vehicle data display
 * - Voice command interface
 * - Dispatch management
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusTextView: MaterialTextView
    private lateinit var connectButton: MaterialButton
    private lateinit var voiceButton: MaterialButton

    private var bluetoothGatt: BluetoothGatt? = null

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.RECORD_AUDIO
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Timber.plant(Timber.DebugTree())
        Timber.i("Driver App onCreate")

        initViews()
        checkPermissions()
    }

    private fun initViews() {
        statusTextView = findViewById(R.id.statusTextView)
        connectButton = findViewById(R.id.connectButton)
        voiceButton = findViewById(R.id.voiceButton)

        connectButton.setOnClickListener {
            connectToDTGDevice()
        }

        voiceButton.setOnClickListener {
            startVoiceCommand()
        }

        statusTextView.text = "Ready to connect"
    }

    private fun checkPermissions() {
        val permissionsToRequest = REQUIRED_PERMISSIONS.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissionsToRequest.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun connectToDTGDevice() {
        Timber.i("Connecting to DTG device via BLE")
        statusTextView.text = "Searching for DTG device..."

        // TODO: Implement BLE scanning and connection
        // 1. Scan for DTG device (service UUID: 0000FFF0-...)
        // 2. Connect to GATT server
        // 3. Discover services and characteristics
        // 4. Subscribe to vehicle data notifications
    }

    private fun startVoiceCommand() {
        Timber.i("Starting voice command")

        // TODO: Implement voice interface
        // 1. Listen for wake word "헤이 드라이버" (Porcupine)
        // 2. Activate STT (Vosk Korean)
        // 3. Parse intent
        // 4. Execute action
        // 5. Provide TTS feedback
    }

    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt?, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    Timber.i("Connected to DTG device")
                    runOnUiThread {
                        statusTextView.text = "Connected to DTG"
                    }
                    gatt?.discoverServices()
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    Timber.i("Disconnected from DTG device")
                    runOnUiThread {
                        statusTextView.text = "Disconnected"
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        bluetoothGatt?.close()
        super.onDestroy()
    }
}
