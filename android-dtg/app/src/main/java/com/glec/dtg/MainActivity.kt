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
import timber.log.Timber

/**
 * GLEC DTG Main Activity
 * Displays status and allows starting/stopping DTG service
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusTextView: MaterialTextView
    private lateinit var startButton: MaterialButton
    private lateinit var stopButton: MaterialButton

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.BLUETOOTH_ADVERTISE,
        )
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

    private fun initViews() {
        statusTextView = findViewById(R.id.statusTextView)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)

        updateUI()
    }

    private fun setupClickListeners() {
        startButton.setOnClickListener {
            startDTGService()
        }

        stopButton.setOnClickListener {
            stopDTGService()
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
        // TODO: Check if service is actually running
        statusTextView.text = "DTG Service Status: Ready"
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
}
