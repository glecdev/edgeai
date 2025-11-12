# EdgeAI SDK Architecture Specification

**Document Version:** 1.0
**Last Updated:** 2025-01-10
**Status:** Architecture Design Phase
**Target:** GLEC DTG Edge AI SDK for Commercial Vehicle Telematics

---

## Table of Contents

1. [Overview & Requirements](#1-overview--requirements)
2. [SDK Module Structure](#2-sdk-module-structure)
3. [Core Components](#3-core-components)
4. [Automatic Sensor Detection](#4-automatic-sensor-detection)
5. [Multi-Sensor Support](#5-multi-sensor-support)
6. [Auto Data Collection & Analysis](#6-auto-data-collection--analysis)
7. [Driver UI Requirements](#7-driver-ui-requirements)
8. [Launcher App Integration](#8-launcher-app-integration)
9. [Data Flow Architecture](#9-data-flow-architecture)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [API Reference](#11-api-reference)
12. [Security & Privacy](#12-security--privacy)

---

## 1. Overview & Requirements

### 1.1 System Purpose

The GLEC DTG EdgeAI SDK transforms the DTG device into a **comprehensive multi-sensor hub** for commercial vehicles, automatically detecting and integrating data from multiple sensor types:

- **CAN Bus Data** (via STM32 MCU)
- **Parking Sensors** (주차 파킹 센서)
- **Dashcams** (블랙박스)
- **Refrigeration Temperature Sensors** (냉장냉온 온도측정 센서)
- **Load Weight Sensors** (적재무게 측정 센서)
- **Wheel/Tire Sensors** (휠 타이어 센서)

### 1.2 Key Requirements

#### Functional Requirements
- **FR-1**: SDK must be packaged as Android AAR library
- **FR-2**: Automatic detection of sensors via USB and Bluetooth
- **FR-3**: Automatic data collection start upon sensor connection
- **FR-4**: AI analysis automatically starts when data is collected
- **FR-5**: Driver must see which devices are connected in real-time
- **FR-6**: Support simultaneous operation of multiple sensor types
- **FR-7**: Launcher app auto-starts on device boot
- **FR-8**: Background service runs continuously (Foreground Service)

#### Non-Functional Requirements
- **NFR-1**: Sensor detection latency < 2 seconds
- **NFR-2**: USB reconnection handling (plug/unplug events)
- **NFR-3**: BLE reconnection with exponential backoff
- **NFR-4**: Memory footprint < 150MB with all sensors active
- **NFR-5**: CPU usage < 15% average (excluding AI inference)
- **NFR-6**: Battery drain < 3W average with all sensors active

### 1.3 Architecture Philosophy

**"Zero Configuration, Full Automation"**

1. **Plug & Play**: Physical sensors auto-detected via USB OTG
2. **Scan & Connect**: BLE sensors auto-discovered and paired
3. **Collect & Analyze**: Data collection → AI analysis → Results transmission (fully automated)
4. **Visibility First**: Driver always knows what's connected and working

---

## 2. SDK Module Structure

### 2.1 Module Organization

```
edgeai-sdk/
├── build.gradle.kts                  # AAR packaging configuration
├── proguard-rules.pro                # Code obfuscation rules
├── src/main/
│   ├── AndroidManifest.xml           # SDK permissions & services
│   ├── java/com/glec/edgeai/
│   │   ├── EdgeAIManager.kt          # Public SDK entry point ⭐
│   │   ├── config/
│   │   │   ├── SDKConfig.kt          # Configuration data class
│   │   │   └── SensorConfig.kt       # Per-sensor configuration
│   │   ├── sensor/
│   │   │   ├── SensorAutoDetector.kt # USB/BLE auto-detection ⭐
│   │   │   ├── MultiSensorManager.kt # Multi-sensor orchestration ⭐
│   │   │   ├── usb/
│   │   │   │   ├── USBSensorDriver.kt
│   │   │   │   └── STM32Driver.kt
│   │   │   └── ble/
│   │   │       ├── BLESensorScanner.kt
│   │   │       └── BLESensorDriver.kt
│   │   ├── collection/
│   │   │   ├── AutoDataCollector.kt  # Sensor-triggered collection ⭐
│   │   │   └── DataAggregator.kt
│   │   ├── inference/
│   │   │   ├── EdgeAIInferenceService.kt  # Existing multi-model AI
│   │   │   └── ModelRegistry.kt
│   │   ├── service/
│   │   │   └── DTGForegroundService.kt    # Refactored background service
│   │   └── ui/
│   │       ├── SensorStatusListener.kt    # Driver UI callbacks
│   │       └── models/
│   │           └── SensorStatus.kt        # UI data models
│   └── res/
│       └── values/
│           └── attrs.xml                  # SDK attributes
└── src/test/
    └── java/com/glec/edgeai/
        ├── SensorAutoDetectorTest.kt
        └── MultiSensorManagerTest.kt
```

### 2.2 AAR Build Configuration

**edgeai-sdk/build.gradle.kts**:

```kotlin
plugins {
    id("com.android.library")
    kotlin("android")
    id("maven-publish")
}

android {
    namespace = "com.glec.edgeai"
    compileSdk = 34

    defaultConfig {
        minSdk = 26  // Android 8.0+ (USB Host API Level 26+)
        targetSdk = 34

        consumerProguardFiles("consumer-rules.pro")
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // Core Android
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // USB Host API
    implementation("androidx.core:core:1.12.0")  // UsbManager support

    // BLE
    implementation("no.nordicsemi.android:ble:2.6.1")  // Nordic BLE library

    // AI Inference (existing)
    implementation("ai.onnxruntime:onnxruntime-android:1.16.3")

    // Logging
    implementation("com.jakewharton.timber:timber:5.0.1")

    // Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.mockito.kotlin:mockito-kotlin:5.1.0")
}

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "com.glec"
            artifactId = "edgeai-sdk"
            version = "1.0.0"

            afterEvaluate {
                from(components["release"])
            }
        }
    }
}
```

### 2.3 SDK Permissions

**edgeai-sdk/src/main/AndroidManifest.xml**:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <!-- USB Host API -->
    <uses-feature android:name="android.hardware.usb.host" android:required="true" />

    <!-- Bluetooth Low Energy -->
    <uses-permission android:name="android.permission.BLUETOOTH" />
    <uses-permission android:name="android.permission.BLUETOOTH_ADMIN" />
    <uses-permission android:name="android.permission.BLUETOOTH_SCAN" />
    <uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />

    <!-- Background Service -->
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_CONNECTED_DEVICE" />
    <uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />

    <!-- Network (for MQTT) -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

    <!-- Wake Lock -->
    <uses-permission android:name="android.permission.WAKE_LOCK" />

    <application>
        <!-- DTG Foreground Service -->
        <service
            android:name=".service.DTGForegroundService"
            android:enabled="true"
            android:exported="false"
            android:foregroundServiceType="connectedDevice">
        </service>

        <!-- USB Device Attachment Intent Filter -->
        <receiver android:name=".sensor.usb.USBDeviceReceiver"
            android:exported="true">
            <intent-filter>
                <action android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED" />
                <action android:name="android.hardware.usb.action.USB_DEVICE_DETACHED" />
            </intent-filter>
            <meta-data
                android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED"
                android:resource="@xml/device_filter" />
        </receiver>
    </application>

</manifest>
```

---

## 3. Core Components

### 3.1 EdgeAIManager (SDK Entry Point)

**Purpose**: Public-facing SDK API for app integration

**edgeai-sdk/src/main/java/com/glec/edgeai/EdgeAIManager.kt**:

```kotlin
package com.glec.edgeai

import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import com.glec.edgeai.config.SDKConfig
import com.glec.edgeai.sensor.MultiSensorManager
import com.glec.edgeai.service.DTGForegroundService
import com.glec.edgeai.ui.SensorStatusListener
import kotlinx.coroutines.flow.StateFlow
import timber.log.Timber

/**
 * EdgeAI SDK Main Entry Point
 *
 * Usage in Launcher App:
 * ```
 * EdgeAIManager.initialize(
 *     context = applicationContext,
 *     config = SDKConfig(
 *         autoStart = true,
 *         autoSensorDetection = true,
 *         autoDataCollection = true
 *     )
 * )
 * EdgeAIManager.startService()
 * EdgeAIManager.registerSensorStatusListener(listener)
 * ```
 */
object EdgeAIManager {

    private var isInitialized = false
    private lateinit var appContext: Context
    private lateinit var config: SDKConfig
    private var sensorManager: MultiSensorManager? = null

    /**
     * Initialize SDK with configuration
     *
     * @param context Application context
     * @param config SDK configuration
     * @throws IllegalStateException if already initialized
     */
    fun initialize(context: Context, config: SDKConfig = SDKConfig()) {
        if (isInitialized) {
            Timber.w("EdgeAI SDK already initialized")
            return
        }

        appContext = context.applicationContext
        this.config = config

        // Initialize Timber logging
        if (config.enableLogging) {
            Timber.plant(Timber.DebugTree())
        }

        // Initialize sensor manager
        sensorManager = MultiSensorManager(appContext)

        isInitialized = true
        Timber.i("EdgeAI SDK initialized with config: $config")
    }

    /**
     * Start DTG Foreground Service
     *
     * Starts background service that:
     * - Auto-detects sensors (if enabled)
     * - Auto-collects data (if enabled)
     * - Runs AI inference
     * - Transmits results via MQTT/BLE
     */
    fun startService() {
        checkInitialized()

        val intent = Intent(appContext, DTGForegroundService::class.java).apply {
            putExtra("config", config)
        }

        ContextCompat.startForegroundService(appContext, intent)
        Timber.i("DTG Foreground Service started")
    }

    /**
     * Stop DTG Foreground Service
     */
    fun stopService() {
        checkInitialized()

        val intent = Intent(appContext, DTGForegroundService::class.java)
        appContext.stopService(intent)
        Timber.i("DTG Foreground Service stopped")
    }

    /**
     * Register listener for sensor status updates
     *
     * Listener receives:
     * - Sensor connected events
     * - Sensor disconnected events
     * - Data collection status
     * - AI inference results
     *
     * @param listener Callback interface
     */
    fun registerSensorStatusListener(listener: SensorStatusListener) {
        checkInitialized()
        sensorManager?.registerListener(listener)
        Timber.d("Sensor status listener registered")
    }

    /**
     * Unregister sensor status listener
     */
    fun unregisterSensorStatusListener(listener: SensorStatusListener) {
        checkInitialized()
        sensorManager?.unregisterListener(listener)
        Timber.d("Sensor status listener unregistered")
    }

    /**
     * Get current connected sensors (StateFlow for UI observation)
     */
    fun getConnectedSensors(): StateFlow<List<SensorStatus>> {
        checkInitialized()
        return sensorManager!!.connectedSensorsFlow
    }

    /**
     * Manually trigger sensor scan (USB + BLE)
     */
    fun scanForSensors() {
        checkInitialized()
        sensorManager?.startScan()
        Timber.i("Manual sensor scan triggered")
    }

    /**
     * Get SDK version
     */
    fun getVersion(): String = BuildConfig.VERSION_NAME

    private fun checkInitialized() {
        check(isInitialized) { "EdgeAI SDK not initialized. Call initialize() first." }
    }
}
```

### 3.2 SDKConfig (Configuration)

**edgeai-sdk/src/main/java/com/glec/edgeai/config/SDKConfig.kt**:

```kotlin
package com.glec.edgeai.config

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/**
 * EdgeAI SDK Configuration
 */
@Parcelize
data class SDKConfig(
    /**
     * Auto-start service on boot (requires RECEIVE_BOOT_COMPLETED)
     */
    val autoStart: Boolean = true,

    /**
     * Auto-detect sensors via USB and Bluetooth
     */
    val autoSensorDetection: Boolean = true,

    /**
     * Auto-start data collection when sensor connects
     */
    val autoDataCollection: Boolean = true,

    /**
     * AI inference interval (seconds)
     * Default: 60s (every minute)
     */
    val inferenceIntervalSeconds: Int = 60,

    /**
     * CAN data collection frequency (Hz)
     * Default: 1Hz (every second)
     */
    val canDataFrequencyHz: Int = 1,

    /**
     * Enable debug logging
     */
    val enableLogging: Boolean = BuildConfig.DEBUG,

    /**
     * MQTT configuration
     */
    val mqttEnabled: Boolean = true,
    val mqttBrokerUrl: String = "ssl://mqtt.glec.kr:8883",
    val mqttTopic: String = "dtg/data",

    /**
     * BLE configuration
     */
    val bleEnabled: Boolean = true,
    val bleAdvertisingEnabled: Boolean = true,

    /**
     * Sensor-specific configurations
     */
    val sensorConfigs: Map<SensorType, SensorConfig> = emptyMap()
) : Parcelable

/**
 * Per-sensor configuration
 */
@Parcelize
data class SensorConfig(
    val enabled: Boolean = true,
    val samplingRateHz: Int = 1,
    val autoReconnect: Boolean = true,
    val reconnectIntervalMs: Long = 5000,
    val maxReconnectAttempts: Int = 10
) : Parcelable
```

---

## 4. Automatic Sensor Detection

### 4.1 SensorAutoDetector (USB + BLE)

**Purpose**: Automatically detect and connect to sensors via USB OTG and Bluetooth

**edgeai-sdk/src/main/java/com/glec/edgeai/sensor/SensorAutoDetector.kt**:

```kotlin
package com.glec.edgeai.sensor

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import com.glec.edgeai.sensor.ble.BLESensorScanner
import com.glec.edgeai.sensor.usb.STM32Driver
import com.glec.edgeai.sensor.usb.USBSensorDriver
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import timber.log.Timber

/**
 * Automatic Sensor Detection via USB and Bluetooth
 *
 * Responsibilities:
 * 1. USB OTG device enumeration
 * 2. BLE device scanning
 * 3. Device identification (VID/PID, BLE UUID)
 * 4. Automatic connection establishment
 * 5. Reconnection on disconnect
 */
class SensorAutoDetector(
    private val context: Context,
    private val sensorManager: MultiSensorManager
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // USB Manager
    private val usbManager: UsbManager by lazy {
        context.getSystemService(Context.USB_SERVICE) as UsbManager
    }

    // BLE Scanner
    private val bleScanner: BLESensorScanner by lazy {
        BLESensorScanner(context)
    }

    // Detection state
    private val _isScanning = MutableStateFlow(false)
    val isScanning: StateFlow<Boolean> = _isScanning

    private val ACTION_USB_PERMISSION = "com.glec.edgeai.USB_PERMISSION"

    // USB permission receiver
    private val usbPermissionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (ACTION_USB_PERMISSION == intent.action) {
                synchronized(this) {
                    val device: UsbDevice? = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                        device?.let { connectUSBDevice(it) }
                    } else {
                        Timber.w("USB permission denied for device: ${device?.deviceName}")
                    }
                }
            }
        }
    }

    // USB attach/detach receiver
    private val usbDeviceReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                UsbManager.ACTION_USB_DEVICE_ATTACHED -> {
                    val device: UsbDevice? = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    device?.let { onUSBDeviceAttached(it) }
                }
                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    val device: UsbDevice? = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    device?.let { onUSBDeviceDetached(it) }
                }
            }
        }
    }

    /**
     * Start automatic sensor detection
     */
    fun startAutoDetection() {
        Timber.i("Starting automatic sensor detection (USB + BLE)")
        _isScanning.value = true

        // Register USB receivers
        val permissionFilter = IntentFilter(ACTION_USB_PERMISSION)
        context.registerReceiver(usbPermissionReceiver, permissionFilter, Context.RECEIVER_NOT_EXPORTED)

        val usbFilter = IntentFilter().apply {
            addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }
        context.registerReceiver(usbDeviceReceiver, usbFilter, Context.RECEIVER_NOT_EXPORTED)

        // Scan for already-connected USB devices
        detectExistingUSBDevices()

        // Start BLE scanning
        startBLEScanning()
    }

    /**
     * Stop automatic sensor detection
     */
    fun stopAutoDetection() {
        Timber.i("Stopping automatic sensor detection")
        _isScanning.value = false

        context.unregisterReceiver(usbPermissionReceiver)
        context.unregisterReceiver(usbDeviceReceiver)

        bleScanner.stopScanning()
        scope.cancel()
    }

    // ========== USB Detection ==========

    /**
     * Detect USB devices already connected
     */
    private fun detectExistingUSBDevices() {
        val deviceList = usbManager.deviceList
        Timber.d("Scanning for USB devices: ${deviceList.size} devices found")

        deviceList.values.forEach { device ->
            if (isRecognizedUSBDevice(device)) {
                requestUSBPermission(device)
            }
        }
    }

    /**
     * Handle USB device attached event
     */
    private fun onUSBDeviceAttached(device: UsbDevice) {
        Timber.i("USB device attached: ${device.deviceName} (VID: 0x${device.vendorId.toString(16)}, PID: 0x${device.productId.toString(16)})")

        if (isRecognizedUSBDevice(device)) {
            requestUSBPermission(device)
        }
    }

    /**
     * Handle USB device detached event
     */
    private fun onUSBDeviceDetached(device: UsbDevice) {
        Timber.i("USB device detached: ${device.deviceName}")

        // Notify sensor manager
        val sensorType = identifyUSBSensorType(device)
        sensorManager.onSensorDisconnected(device.deviceName, sensorType)
    }

    /**
     * Check if USB device is a recognized sensor
     */
    private fun isRecognizedUSBDevice(device: UsbDevice): Boolean {
        return when {
            isSTM32Device(device) -> true
            isParkingSensor(device) -> true
            isWeightSensor(device) -> true
            isDashcam(device) -> true
            else -> false
        }
    }

    /**
     * Identify sensor type from USB device
     */
    private fun identifyUSBSensorType(device: UsbDevice): SensorType {
        return when {
            isSTM32Device(device) -> SensorType.CAN_BUS
            isParkingSensor(device) -> SensorType.PARKING_SENSOR
            isWeightSensor(device) -> SensorType.LOAD_WEIGHT
            isDashcam(device) -> SensorType.DASHCAM
            else -> SensorType.UNKNOWN
        }
    }

    /**
     * Check if device is STM32 DTG (CAN bus interface)
     */
    private fun isSTM32Device(device: UsbDevice): Boolean {
        // STM32 Virtual COM Port: VID 0x0483, PID 0x5740
        return device.vendorId == 0x0483 && device.productId == 0x5740
    }

    /**
     * Check if device is parking sensor
     */
    private fun isParkingSensor(device: UsbDevice): Boolean {
        // Example VID/PID for parking sensors
        // TODO: Update with actual parking sensor VID/PID
        return device.vendorId == 0x1234 && device.productId == 0x5678
    }

    /**
     * Check if device is weight sensor
     */
    private fun isWeightSensor(device: UsbDevice): Boolean {
        // Example VID/PID for load weight sensors
        // TODO: Update with actual weight sensor VID/PID
        return device.vendorId == 0xABCD && device.productId == 0xEF01
    }

    /**
     * Check if device is dashcam
     */
    private fun isDashcam(device: UsbDevice): Boolean {
        // Example VID/PID for dashcams (e.g., BlackVue, Thinkware)
        // TODO: Update with actual dashcam VID/PID
        return device.vendorId == 0x2222 && device.productId == 0x3333
    }

    /**
     * Request USB permission from user
     */
    private fun requestUSBPermission(device: UsbDevice) {
        if (usbManager.hasPermission(device)) {
            connectUSBDevice(device)
        } else {
            val permissionIntent = PendingIntent.getBroadcast(
                context,
                0,
                Intent(ACTION_USB_PERMISSION),
                PendingIntent.FLAG_IMMUTABLE
            )
            usbManager.requestPermission(device, permissionIntent)
            Timber.d("USB permission requested for: ${device.deviceName}")
        }
    }

    /**
     * Connect to USB device
     */
    private fun connectUSBDevice(device: UsbDevice) {
        scope.launch {
            try {
                val driver = createUSBDriver(device)
                driver.connect()

                val sensorType = identifyUSBSensorType(device)
                sensorManager.onSensorConnected(
                    sensorId = device.deviceName,
                    sensorType = sensorType,
                    connectionType = ConnectionType.USB,
                    driver = driver
                )

                Timber.i("USB device connected successfully: ${device.deviceName} ($sensorType)")
            } catch (e: Exception) {
                Timber.e(e, "Failed to connect USB device: ${device.deviceName}")
            }
        }
    }

    /**
     * Create appropriate USB driver for device
     */
    private fun createUSBDriver(device: UsbDevice): USBSensorDriver {
        return when {
            isSTM32Device(device) -> STM32Driver(context, device)
            // Add other USB sensor drivers here
            else -> throw IllegalArgumentException("Unsupported USB device: ${device.deviceName}")
        }
    }

    // ========== BLE Detection ==========

    /**
     * Start BLE sensor scanning
     */
    private fun startBLEScanning() {
        bleScanner.startScanning(
            onDeviceFound = { bleDevice, rssi ->
                onBLEDeviceFound(bleDevice, rssi)
            },
            onScanFailed = { errorCode ->
                Timber.e("BLE scan failed with error code: $errorCode")
            }
        )
    }

    /**
     * Handle BLE device found event
     */
    private fun onBLEDeviceFound(device: android.bluetooth.BluetoothDevice, rssi: Int) {
        val sensorType = identifyBLESensorType(device)

        if (sensorType != SensorType.UNKNOWN) {
            Timber.i("BLE sensor found: ${device.name} (${device.address}), RSSI: $rssi, Type: $sensorType")

            scope.launch {
                try {
                    val driver = bleScanner.connect(device)

                    sensorManager.onSensorConnected(
                        sensorId = device.address,
                        sensorType = sensorType,
                        connectionType = ConnectionType.BLUETOOTH,
                        driver = driver
                    )

                    Timber.i("BLE sensor connected successfully: ${device.name} ($sensorType)")
                } catch (e: Exception) {
                    Timber.e(e, "Failed to connect BLE device: ${device.name}")
                }
            }
        }
    }

    /**
     * Identify sensor type from BLE device
     */
    private fun identifyBLESensorType(device: android.bluetooth.BluetoothDevice): SensorType {
        val name = device.name ?: return SensorType.UNKNOWN

        return when {
            name.contains("TEMP", ignoreCase = true) ||
            name.contains("냉장", ignoreCase = true) -> SensorType.REFRIGERATION_TEMP

            name.contains("TIRE", ignoreCase = true) ||
            name.contains("TPMS", ignoreCase = true) ||
            name.contains("타이어", ignoreCase = true) -> SensorType.TIRE_SENSOR

            name.contains("DRIVER", ignoreCase = true) ||
            name.contains("운전자", ignoreCase = true) -> SensorType.DRIVER_APP

            else -> SensorType.UNKNOWN
        }
    }
}
```

---

## 5. Multi-Sensor Support

### 5.1 Sensor Type Definitions

**edgeai-sdk/src/main/java/com/glec/edgeai/sensor/SensorType.kt**:

```kotlin
package com.glec.edgeai.sensor

/**
 * Supported sensor types for commercial vehicle DTG
 */
enum class SensorType(
    val displayName: String,
    val icon: String,
    val requiresUSB: Boolean = false,
    val requiresBLE: Boolean = false
) {
    /**
     * CAN Bus interface via STM32 MCU (UART 921600 baud)
     * Data: RPM, speed, throttle, brake, fuel, coolant temp, etc.
     */
    CAN_BUS(
        displayName = "CAN 버스",
        icon = "🚛",
        requiresUSB = true
    ),

    /**
     * Parking sensors (ultrasonic or radar)
     * Data: Distance to obstacles (front/rear/side)
     */
    PARKING_SENSOR(
        displayName = "주차 센서",
        icon = "📡",
        requiresUSB = true
    ),

    /**
     * Dashcam / Blackbox (video recording device)
     * Data: Video frames, GPS, G-sensor, event markers
     */
    DASHCAM(
        displayName = "블랙박스",
        icon = "📹",
        requiresUSB = true
    ),

    /**
     * Refrigeration/Freezer temperature sensor
     * Data: Internal temperature, humidity, door status
     */
    REFRIGERATION_TEMP(
        displayName = "냉장 온도 센서",
        icon = "🌡️",
        requiresBLE = true
    ),

    /**
     * Load weight sensor (axle weight measurement)
     * Data: Front/rear axle weight, total weight, overload alerts
     */
    LOAD_WEIGHT(
        displayName = "적재 무게 센서",
        icon = "⚖️",
        requiresUSB = true
    ),

    /**
     * Tire pressure monitoring system (TPMS)
     * Data: Tire pressure, temperature, leak detection
     */
    TIRE_SENSOR(
        displayName = "타이어 센서",
        icon = "🛞",
        requiresBLE = true
    ),

    /**
     * Driver smartphone app (BLE connection)
     * Data: Voice commands, manual trip inputs, external APIs
     */
    DRIVER_APP(
        displayName = "운전자 앱",
        icon = "📱",
        requiresBLE = true
    ),

    /**
     * Unknown/Unrecognized sensor
     */
    UNKNOWN(
        displayName = "알 수 없음",
        icon = "❓"
    );

    /**
     * Check if sensor supports given connection type
     */
    fun supportsConnection(connectionType: ConnectionType): Boolean {
        return when (connectionType) {
            ConnectionType.USB -> requiresUSB
            ConnectionType.BLUETOOTH -> requiresBLE
            ConnectionType.UART -> this == CAN_BUS  // Only STM32 uses UART
        }
    }
}

/**
 * Connection type for sensors
 */
enum class ConnectionType {
    USB,        // USB OTG
    BLUETOOTH,  // BLE
    UART        // Direct UART (internal STM32)
}
```

### 5.2 MultiSensorManager (Orchestration)

**edgeai-sdk/src/main/java/com/glec/edgeai/sensor/MultiSensorManager.kt**:

```kotlin
package com.glec.edgeai.sensor

import android.content.Context
import com.glec.edgeai.collection.AutoDataCollector
import com.glec.edgeai.ui.SensorStatusListener
import com.glec.edgeai.ui.models.SensorStatus
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import timber.log.Timber
import java.util.concurrent.ConcurrentHashMap

/**
 * Multi-Sensor Manager
 *
 * Responsibilities:
 * 1. Track all connected sensors
 * 2. Manage sensor lifecycle (connect/disconnect)
 * 3. Coordinate data collection from multiple sensors
 * 4. Notify UI listeners of sensor status changes
 */
class MultiSensorManager(private val context: Context) {

    // Connected sensors map (thread-safe)
    private val connectedSensors = ConcurrentHashMap<String, Sensor>()

    // Sensor status flow (for UI observation)
    private val _connectedSensorsFlow = MutableStateFlow<List<SensorStatus>>(emptyList())
    val connectedSensorsFlow: StateFlow<List<SensorStatus>> = _connectedSensorsFlow

    // Registered UI listeners
    private val statusListeners = mutableSetOf<SensorStatusListener>()

    // Auto data collector
    private val dataCollector = AutoDataCollector(context)

    /**
     * Handle sensor connected event
     *
     * Called by SensorAutoDetector when new sensor is detected and connected
     */
    fun onSensorConnected(
        sensorId: String,
        sensorType: SensorType,
        connectionType: ConnectionType,
        driver: Any  // USBSensorDriver or BLESensorDriver
    ) {
        val sensor = Sensor(
            id = sensorId,
            type = sensorType,
            connectionType = connectionType,
            driver = driver,
            connectedAt = System.currentTimeMillis(),
            isCollecting = false
        )

        connectedSensors[sensorId] = sensor
        updateSensorStatusFlow()

        // Notify listeners
        notifyListeners { it.onSensorConnected(sensor.toSensorStatus()) }

        // Auto-start data collection if enabled
        startDataCollection(sensor)

        Timber.i("Sensor connected: $sensorId ($sensorType via $connectionType)")
    }

    /**
     * Handle sensor disconnected event
     */
    fun onSensorDisconnected(sensorId: String, sensorType: SensorType) {
        val sensor = connectedSensors.remove(sensorId)

        if (sensor != null) {
            updateSensorStatusFlow()

            // Stop data collection
            dataCollector.stopCollection(sensorId)

            // Notify listeners
            notifyListeners { it.onSensorDisconnected(sensor.toSensorStatus()) }

            Timber.i("Sensor disconnected: $sensorId ($sensorType)")
        }
    }

    /**
     * Start data collection for sensor
     */
    private fun startDataCollection(sensor: Sensor) {
        dataCollector.startCollection(
            sensorId = sensor.id,
            sensorType = sensor.type,
            driver = sensor.driver,
            onDataCollected = { data ->
                // Update sensor status
                sensor.isCollecting = true
                sensor.lastDataTimestamp = System.currentTimeMillis()
                updateSensorStatusFlow()

                // Notify listeners
                notifyListeners { it.onDataReceived(sensor.id, sensor.type, data) }
            },
            onError = { error ->
                Timber.e(error, "Data collection error for sensor: ${sensor.id}")
                notifyListeners { it.onSensorError(sensor.id, error.message ?: "Unknown error") }
            }
        )
    }

    /**
     * Get all connected sensors
     */
    fun getConnectedSensors(): List<Sensor> = connectedSensors.values.toList()

    /**
     * Get sensor by ID
     */
    fun getSensor(sensorId: String): Sensor? = connectedSensors[sensorId]

    /**
     * Check if specific sensor type is connected
     */
    fun isSensorConnected(sensorType: SensorType): Boolean {
        return connectedSensors.values.any { it.type == sensorType }
    }

    /**
     * Register UI listener
     */
    fun registerListener(listener: SensorStatusListener) {
        statusListeners.add(listener)
        Timber.d("Sensor status listener registered (total: ${statusListeners.size})")
    }

    /**
     * Unregister UI listener
     */
    fun unregisterListener(listener: SensorStatusListener) {
        statusListeners.remove(listener)
        Timber.d("Sensor status listener unregistered (total: ${statusListeners.size})")
    }

    /**
     * Update sensor status flow for UI observation
     */
    private fun updateSensorStatusFlow() {
        val statusList = connectedSensors.values.map { it.toSensorStatus() }
        _connectedSensorsFlow.value = statusList
    }

    /**
     * Notify all registered listeners
     */
    private fun notifyListeners(action: (SensorStatusListener) -> Unit) {
        statusListeners.forEach { listener ->
            try {
                action(listener)
            } catch (e: Exception) {
                Timber.e(e, "Error notifying sensor status listener")
            }
        }
    }

    /**
     * Start manual sensor scan
     */
    fun startScan() {
        // Trigger manual scan via SensorAutoDetector
        // This is called by EdgeAIManager.scanForSensors()
        Timber.i("Manual sensor scan triggered")
    }
}

/**
 * Sensor data class
 */
data class Sensor(
    val id: String,                    // Unique ID (USB device name or BLE MAC address)
    val type: SensorType,              // Sensor type
    val connectionType: ConnectionType, // USB or Bluetooth
    val driver: Any,                   // Driver instance (USBSensorDriver or BLESensorDriver)
    val connectedAt: Long,             // Connection timestamp
    var isCollecting: Boolean = false, // Is data collection active?
    var lastDataTimestamp: Long = 0    // Last data received timestamp
) {
    fun toSensorStatus() = SensorStatus(
        sensorId = id,
        sensorType = type,
        connectionType = connectionType,
        isConnected = true,
        isCollecting = isCollecting,
        connectedAt = connectedAt,
        lastDataTimestamp = lastDataTimestamp
    )
}
```

---

## 6. Auto Data Collection & Analysis

### 6.1 AutoDataCollector

**edgeai-sdk/src/main/java/com/glec/edgeai/collection/AutoDataCollector.kt**:

```kotlin
package com.glec.edgeai.collection

import android.content.Context
import com.glec.edgeai.sensor.SensorType
import com.glec.edgeai.sensor.usb.USBSensorDriver
import com.glec.edgeai.sensor.ble.BLESensorDriver
import kotlinx.coroutines.*
import timber.log.Timber
import java.util.concurrent.ConcurrentHashMap

/**
 * Automatic Data Collector
 *
 * Automatically starts data collection when sensors connect.
 * Each sensor type has dedicated collection logic.
 */
class AutoDataCollector(private val context: Context) {

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val collectionJobs = ConcurrentHashMap<String, Job>()

    /**
     * Start data collection for sensor
     *
     * @param sensorId Unique sensor ID
     * @param sensorType Type of sensor
     * @param driver Sensor driver (USB or BLE)
     * @param onDataCollected Callback when data is collected
     * @param onError Callback on collection error
     */
    fun startCollection(
        sensorId: String,
        sensorType: SensorType,
        driver: Any,
        onDataCollected: (data: Any) -> Unit,
        onError: (error: Throwable) -> Unit
    ) {
        // Cancel existing collection job if any
        stopCollection(sensorId)

        val job = scope.launch {
            try {
                when (sensorType) {
                    SensorType.CAN_BUS -> collectCANData(driver as USBSensorDriver, onDataCollected)
                    SensorType.PARKING_SENSOR -> collectParkingData(driver as USBSensorDriver, onDataCollected)
                    SensorType.DASHCAM -> collectDashcamData(driver as USBSensorDriver, onDataCollected)
                    SensorType.REFRIGERATION_TEMP -> collectTempData(driver as BLESensorDriver, onDataCollected)
                    SensorType.LOAD_WEIGHT -> collectWeightData(driver as USBSensorDriver, onDataCollected)
                    SensorType.TIRE_SENSOR -> collectTireData(driver as BLESensorDriver, onDataCollected)
                    SensorType.DRIVER_APP -> collectDriverAppData(driver as BLESensorDriver, onDataCollected)
                    else -> Timber.w("Unknown sensor type: $sensorType")
                }
            } catch (e: CancellationException) {
                Timber.d("Data collection cancelled for sensor: $sensorId")
                throw e
            } catch (e: Exception) {
                Timber.e(e, "Data collection error for sensor: $sensorId")
                onError(e)
            }
        }

        collectionJobs[sensorId] = job
        Timber.i("Data collection started for sensor: $sensorId ($sensorType)")
    }

    /**
     * Stop data collection for sensor
     */
    fun stopCollection(sensorId: String) {
        collectionJobs.remove(sensorId)?.cancel()
        Timber.d("Data collection stopped for sensor: $sensorId")
    }

    // ========== Sensor-Specific Collection Logic ==========

    /**
     * Collect CAN bus data (1Hz)
     */
    private suspend fun collectCANData(
        driver: USBSensorDriver,
        onDataCollected: (data: Any) -> Unit
    ) {
        while (isActive) {
            val canData = driver.readData()  // Blocking read
            onDataCollected(canData)
            delay(1000)  // 1Hz
        }
    }

    /**
     * Collect parking sensor data (10Hz)
     */
    private suspend fun collectParkingData(
        driver: USBSensorDriver,
        onDataCollected: (data: Any) -> Unit
    ) {
        while (isActive) {
            val parkingData = driver.readData()
            onDataCollected(parkingData)
            delay(100)  // 10Hz
        }
    }

    /**
     * Collect dashcam data (event-based sampling)
     */
    private suspend fun collectDashcamData(
        driver: USBSensorDriver,
        onDataCollected: (data: Any) -> Unit
    ) {
        // Event-based sampling only (harsh braking, collision, etc.)
        while (isActive) {
            val eventData = driver.readData()  // Blocks until event occurs
            onDataCollected(eventData)
        }
    }

    /**
     * Collect refrigeration temperature data (0.1Hz, every 10 seconds)
     */
    private suspend fun collectTempData(
        driver: BLESensorDriver,
        onDataCollected: (data: Any) -> Unit
    ) {
        while (isActive) {
            val tempData = driver.readCharacteristic()
            onDataCollected(tempData)
            delay(10000)  // 0.1Hz (every 10 seconds)
        }
    }

    /**
     * Collect load weight data (1Hz)
     */
    private suspend fun collectWeightData(
        driver: USBSensorDriver,
        onDataCollected: (data: Any) -> Unit
    ) {
        while (isActive) {
            val weightData = driver.readData()
            onDataCollected(weightData)
            delay(1000)  // 1Hz
        }
    }

    /**
     * Collect tire sensor data (TPMS) (0.2Hz, every 5 seconds)
     */
    private suspend fun collectTireData(
        driver: BLESensorDriver,
        onDataCollected: (data: Any) -> Unit
    ) {
        while (isActive) {
            val tireData = driver.readCharacteristic()
            onDataCollected(tireData)
            delay(5000)  // 0.2Hz (every 5 seconds)
        }
    }

    /**
     * Collect driver app data (event-based)
     */
    private suspend fun collectDriverAppData(
        driver: BLESensorDriver,
        onDataCollected: (data: Any) -> Unit
    ) {
        // Subscribe to BLE notifications (event-based)
        driver.setNotificationCallback { data ->
            onDataCollected(data)
        }
    }
}
```

---

## 7. Driver UI Requirements

### 7.1 SensorStatusListener (UI Callbacks)

**edgeai-sdk/src/main/java/com/glec/edgeai/ui/SensorStatusListener.kt**:

```kotlin
package com.glec.edgeai.ui

import com.glec.edgeai.sensor.SensorType
import com.glec.edgeai.ui.models.SensorStatus

/**
 * Listener interface for sensor status updates
 *
 * Implement this interface in your Launcher App Activity/Fragment
 * to receive real-time sensor connection status updates.
 *
 * Example usage:
 * ```
 * class MainActivity : AppCompatActivity(), SensorStatusListener {
 *     override fun onSensorConnected(status: SensorStatus) {
 *         runOnUiThread {
 *             addSensorToUI(status)
 *             showToast("센서 연결: ${status.sensorType.displayName}")
 *         }
 *     }
 * }
 * ```
 */
interface SensorStatusListener {

    /**
     * Called when a sensor is connected
     *
     * @param status Sensor status information
     */
    fun onSensorConnected(status: SensorStatus)

    /**
     * Called when a sensor is disconnected
     *
     * @param status Sensor status information
     */
    fun onSensorDisconnected(status: SensorStatus)

    /**
     * Called when data is received from sensor
     *
     * @param sensorId Sensor unique ID
     * @param sensorType Type of sensor
     * @param data Raw data received
     */
    fun onDataReceived(sensorId: String, sensorType: SensorType, data: Any)

    /**
     * Called when sensor error occurs
     *
     * @param sensorId Sensor unique ID
     * @param error Error message
     */
    fun onSensorError(sensorId: String, error: String)

    /**
     * Called when AI inference completes
     *
     * @param result Inference result (behavior classification, anomaly detection, etc.)
     */
    fun onInferenceComplete(result: Any)
}
```

### 7.2 SensorStatus (UI Data Model)

**edgeai-sdk/src/main/java/com/glec/edgeai/ui/models/SensorStatus.kt**:

```kotlin
package com.glec.edgeai.ui.models

import com.glec.edgeai.sensor.ConnectionType
import com.glec.edgeai.sensor.SensorType

/**
 * Sensor status information for UI display
 */
data class SensorStatus(
    val sensorId: String,               // Unique ID
    val sensorType: SensorType,         // Type (CAN, parking, dashcam, etc.)
    val connectionType: ConnectionType, // USB or Bluetooth
    val isConnected: Boolean,           // Connection status
    val isCollecting: Boolean,          // Is data collection active?
    val connectedAt: Long,              // Connection timestamp
    val lastDataTimestamp: Long,        // Last data received timestamp
    val signalStrength: Int = 0         // BLE RSSI (for Bluetooth sensors)
) {
    /**
     * Get connection duration in seconds
     */
    fun getConnectionDurationSeconds(): Long {
        return (System.currentTimeMillis() - connectedAt) / 1000
    }

    /**
     * Get time since last data received (seconds)
     */
    fun getTimeSinceLastDataSeconds(): Long {
        if (lastDataTimestamp == 0L) return -1
        return (System.currentTimeMillis() - lastDataTimestamp) / 1000
    }

    /**
     * Check if sensor is healthy (receiving data regularly)
     */
    fun isHealthy(): Boolean {
        if (!isConnected || !isCollecting) return false
        if (lastDataTimestamp == 0L) return false

        // Healthy if data received within last 30 seconds
        return getTimeSinceLastDataSeconds() < 30
    }

    /**
     * Get display name for UI
     */
    fun getDisplayName(): String = sensorType.displayName

    /**
     * Get icon for UI
     */
    fun getIcon(): String = sensorType.icon
}
```

### 7.3 Sample UI Implementation (Launcher App)

**dtg-launcher/src/main/java/com/glec/dtglauncher/MainActivity.kt**:

```kotlin
package com.glec.dtglauncher

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.glec.edgeai.EdgeAIManager
import com.glec.edgeai.config.SDKConfig
import com.glec.edgeai.sensor.SensorType
import com.glec.edgeai.ui.SensorStatusListener
import com.glec.edgeai.ui.models.SensorStatus
import timber.log.Timber

/**
 * DTG Launcher App Main Activity
 *
 * Displays connected sensor status in real-time
 */
class MainActivity : AppCompatActivity(), SensorStatusListener {

    private lateinit var sensorAdapter: SensorStatusAdapter
    private val connectedSensors = mutableListOf<SensorStatus>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize SDK
        EdgeAIManager.initialize(
            context = applicationContext,
            config = SDKConfig(
                autoStart = true,
                autoSensorDetection = true,
                autoDataCollection = true
            )
        )

        // Start service
        EdgeAIManager.startService()

        // Register listener
        EdgeAIManager.registerSensorStatusListener(this)

        // Setup RecyclerView
        sensorAdapter = SensorStatusAdapter(connectedSensors)
        findViewById<RecyclerView>(R.id.recyclerViewSensors).apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = sensorAdapter
        }

        Timber.i("DTG Launcher initialized")
    }

    override fun onDestroy() {
        super.onDestroy()
        EdgeAIManager.unregisterSensorStatusListener(this)
    }

    // ========== SensorStatusListener Implementation ==========

    override fun onSensorConnected(status: SensorStatus) {
        runOnUiThread {
            connectedSensors.add(status)
            sensorAdapter.notifyItemInserted(connectedSensors.size - 1)

            showToast("센서 연결됨: ${status.getDisplayName()} ${status.getIcon()}")
            Timber.i("Sensor connected in UI: ${status.sensorType}")
        }
    }

    override fun onSensorDisconnected(status: SensorStatus) {
        runOnUiThread {
            val index = connectedSensors.indexOfFirst { it.sensorId == status.sensorId }
            if (index >= 0) {
                connectedSensors.removeAt(index)
                sensorAdapter.notifyItemRemoved(index)

                showToast("센서 연결 해제됨: ${status.getDisplayName()}")
                Timber.i("Sensor disconnected in UI: ${status.sensorType}")
            }
        }
    }

    override fun onDataReceived(sensorId: String, sensorType: SensorType, data: Any) {
        runOnUiThread {
            val index = connectedSensors.indexOfFirst { it.sensorId == sensorId }
            if (index >= 0) {
                val updatedStatus = connectedSensors[index].copy(
                    lastDataTimestamp = System.currentTimeMillis()
                )
                connectedSensors[index] = updatedStatus
                sensorAdapter.notifyItemChanged(index)
            }
        }
    }

    override fun onSensorError(sensorId: String, error: String) {
        runOnUiThread {
            showToast("센서 오류: $error")
            Timber.e("Sensor error in UI: $sensorId - $error")
        }
    }

    override fun onInferenceComplete(result: Any) {
        runOnUiThread {
            // Display AI inference result in UI
            Timber.i("AI inference result: $result")
        }
    }

    private fun showToast(message: String) {
        android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_SHORT).show()
    }
}
```

---

## 8. Launcher App Integration

### 8.1 BootReceiver (Auto-Start on Boot)

**dtg-launcher/src/main/java/com/glec/dtglauncher/BootReceiver.kt**:

```kotlin
package com.glec.dtglauncher

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.glec.edgeai.EdgeAIManager
import com.glec.edgeai.config.SDKConfig
import timber.log.Timber

/**
 * Boot Receiver - Auto-start DTG service on device boot
 *
 * Requires:
 * - RECEIVE_BOOT_COMPLETED permission
 * - Receiver registered in AndroidManifest.xml
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            Timber.i("Device boot completed - Starting DTG service")

            // Initialize SDK
            EdgeAIManager.initialize(
                context = context.applicationContext,
                config = SDKConfig(
                    autoStart = true,
                    autoSensorDetection = true,
                    autoDataCollection = true
                )
            )

            // Start service
            EdgeAIManager.startService()

            Timber.i("DTG service started on boot")
        }
    }
}
```

### 8.2 Launcher App AndroidManifest.xml

**dtg-launcher/src/main/AndroidManifest.xml**:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.glec.dtglauncher">

    <!-- SDK Permissions -->
    <uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />

    <application
        android:name=".DTGLauncherApplication"
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.DTGLauncher">

        <!-- Main Activity (Launcher) -->
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTask">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
                <category android:name="android.intent.category.HOME" />
                <category android:name="android.intent.category.DEFAULT" />
            </intent-filter>
        </activity>

        <!-- Boot Receiver -->
        <receiver
            android:name=".BootReceiver"
            android:enabled="true"
            android:exported="true">
            <intent-filter android:priority="999">
                <action android:name="android.intent.action.BOOT_COMPLETED" />
                <action android:name="android.intent.action.QUICKBOOT_POWERON" />
            </intent-filter>
        </receiver>

    </application>

</manifest>
```

---

## 9. Data Flow Architecture

### 9.1 System-Wide Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        GLEC DTG Edge AI SDK                              │
│                     Multi-Sensor Hub Architecture                        │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         SENSOR LAYER                                    │
└─────────────────────────────────────────────────────────────────────────┘
   │
   │  USB OTG                          Bluetooth Low Energy
   │  ├─ STM32 CAN Bus (1Hz)          ├─ Temperature Sensor (0.1Hz)
   │  ├─ Parking Sensor (10Hz)        ├─ Tire Sensor (TPMS) (0.2Hz)
   │  ├─ Load Weight Sensor (1Hz)     └─ Driver App (Event-based)
   │  └─ Dashcam (Event-based)
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUTO-DETECTION LAYER                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  SensorAutoDetector                                            │    │
│  │  - USB device enumeration (VID/PID matching)                   │    │
│  │  - BLE device scanning (UUID/name matching)                    │    │
│  │  - Automatic permission request & connection                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ▼ (onSensorConnected)
┌─────────────────────────────────────────────────────────────────────────┐
│                  MULTI-SENSOR MANAGEMENT LAYER                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  MultiSensorManager                                            │    │
│  │  - Track all connected sensors                                 │    │
│  │  - Manage sensor lifecycle                                     │    │
│  │  - Notify UI listeners                                         │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ▼ (Auto-start collection)
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                                │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  AutoDataCollector                                             │    │
│  │  - Per-sensor collection logic                                 │    │
│  │  - Sampling rate control                                       │    │
│  │  - Data validation & preprocessing                             │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ▼ (Aggregated data every 60s)
┌─────────────────────────────────────────────────────────────────────────┐
│                      AI INFERENCE LAYER                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  EdgeAIInferenceService                                        │    │
│  │  - Feature extraction (60 samples × 18 features)               │    │
│  │  - LightGBM behavior classification (ONNX)                     │    │
│  │  - TCN fuel prediction (ONNX)                                  │    │
│  │  - LSTM-AE anomaly detection (ONNX)                            │    │
│  │  - YOLOv5 Nano (dashcam event analysis)                        │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ▼ (Inference results)
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRANSMISSION LAYER                                   │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  - MQTT Publisher (TLS to Fleet AI Platform)                   │    │
│  │  - BLE GATT Server (to Driver App)                             │    │
│  │  - Local SQLite queue (offline resilience)                     │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         UI LAYER                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  DTG Launcher App                                              │    │
│  │  - Real-time sensor connection status                          │    │
│  │  - Data collection indicators                                  │    │
│  │  - AI inference results                                        │    │
│  │  - Sensor health monitoring                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Sensor Connection Flow

```
1. Device Boot
   └─> BootReceiver.onReceive(ACTION_BOOT_COMPLETED)
       └─> EdgeAIManager.initialize()
           └─> EdgeAIManager.startService()
               └─> DTGForegroundService.onCreate()

2. Service Start
   └─> DTGForegroundService.onStartCommand()
       └─> SensorAutoDetector.startAutoDetection()
           ├─> detectExistingUSBDevices()  # Scan already connected
           └─> startBLEScanning()          # Start BLE scan

3. USB Device Attached (Example: STM32 CAN Bus)
   └─> USBDeviceReceiver.onReceive(USB_DEVICE_ATTACHED)
       └─> SensorAutoDetector.onUSBDeviceAttached(device)
           └─> isRecognizedUSBDevice(device)  # Check VID/PID
               └─> requestUSBPermission(device)
                   └─> [User grants permission]
                       └─> connectUSBDevice(device)
                           └─> STM32Driver.connect()
                               └─> MultiSensorManager.onSensorConnected(
                                       sensorId = "/dev/bus/usb/001/002",
                                       sensorType = SensorType.CAN_BUS,
                                       connectionType = ConnectionType.USB,
                                       driver = STM32Driver
                                   )

4. Multi-Sensor Manager Processing
   └─> MultiSensorManager.onSensorConnected()
       ├─> connectedSensors[sensorId] = sensor
       ├─> updateSensorStatusFlow()  # Update UI StateFlow
       ├─> notifyListeners { it.onSensorConnected(status) }
       └─> startDataCollection(sensor)
           └─> AutoDataCollector.startCollection(
                   sensorId, sensorType, driver,
                   onDataCollected = { data -> ... },
                   onError = { error -> ... }
               )

5. Data Collection Loop (1Hz for CAN Bus)
   └─> AutoDataCollector.collectCANData()
       └─> while (isActive) {
               val canData = driver.readData()  # Blocking UART read
               onDataCollected(canData)  # Notify manager
               delay(1000)  # 1Hz
           }

6. UI Update
   └─> MainActivity.onSensorConnected(status)
       └─> runOnUiThread {
               connectedSensors.add(status)
               sensorAdapter.notifyItemInserted()
               showToast("센서 연결됨: CAN 버스 🚛")
           }

7. AI Inference (Every 60 seconds)
   └─> DTGForegroundService.inferenceTimer.tick()
       └─> EdgeAIInferenceService.runInference()
           ├─> Extract features from 60 CAN samples
           ├─> LightGBM behavior classification
           ├─> TCN fuel prediction
           └─> LSTM-AE anomaly detection
               └─> onInferenceComplete(result)
                   ├─> MQTT publish to Fleet AI
                   └─> BLE notify to Driver App
```

---

## 10. Implementation Roadmap

### Phase A: SDK Foundation (2-3 Days)

**Goal**: Create EdgeAI SDK module structure

**Tasks**:
1. ✅ Create `edgeai-sdk` module in Android Studio
2. ✅ Configure AAR build (`build.gradle.kts`)
3. ✅ Setup SDK permissions (`AndroidManifest.xml`)
4. ✅ Implement `EdgeAIManager` (public API)
5. ✅ Implement `SDKConfig` (configuration)
6. ✅ Write unit tests (EdgeAIManagerTest, 80% coverage)

**Deliverable**: `edgeai-sdk-1.0.0.aar` (functional SDK library)

---

### Phase B: Auto-Detection (3-4 Days)

**Goal**: Implement automatic sensor detection (USB + BLE)

**Tasks**:
1. ✅ Implement `SensorAutoDetector`
2. ✅ USB device enumeration & VID/PID matching
3. ✅ USB permission request flow
4. ✅ BLE scanning with Nordic library
5. ✅ BLE device identification (UUID/name)
6. ✅ USB attach/detach event handling
7. ✅ BLE reconnection logic
8. ✅ Write unit tests (80% coverage)

**Deliverable**: Automatic sensor detection working for STM32 + BLE sensors

---

### Phase C: Multi-Sensor Support (3-4 Days)

**Goal**: Support all 5+ sensor types

**Tasks**:
1. ✅ Implement `MultiSensorManager`
2. ✅ Define `SensorType` enum (6 types)
3. ✅ Implement `AutoDataCollector`
4. ✅ Per-sensor collection logic:
   - CAN Bus (1Hz)
   - Parking (10Hz)
   - Dashcam (event-based)
   - Temperature (0.1Hz)
   - Weight (1Hz)
   - Tire (0.2Hz)
5. ✅ Data aggregation & validation
6. ✅ Write integration tests

**Deliverable**: Multi-sensor data collection working simultaneously

---

### Phase D: Driver UI (2 Days)

**Goal**: Driver visibility of connected devices

**Tasks**:
1. ✅ Implement `SensorStatusListener` interface
2. ✅ Implement `SensorStatus` data model
3. ✅ Create sample Launcher App UI
4. ✅ RecyclerView adapter for sensor list
5. ✅ Real-time status updates (StateFlow)
6. ✅ Connection health indicators

**Deliverable**: Launcher App showing real-time sensor connections

---

### Phase E: Launcher App Integration (2-3 Days)

**Goal**: Auto-start on boot, always-on operation

**Tasks**:
1. ✅ Create `dtg-launcher` app module
2. ✅ Implement `BootReceiver`
3. ✅ Configure launcher intent filters
4. ✅ Kiosk mode (optional)
5. ✅ Integrate EdgeAI SDK
6. ✅ E2E testing (boot → sensor detection → data collection → inference)

**Deliverable**: Production-ready Launcher App with SDK integration

---

### Phase F: Refactoring & Testing (2 Days)

**Goal**: Refactor existing code into SDK structure

**Tasks**:
1. ✅ Move `DTGForegroundService` into SDK
2. ✅ Move `EdgeAIInferenceService` into SDK
3. ✅ Move `BLEManager` into SDK (refactor for multi-device)
4. ✅ Update MQTT client integration
5. ✅ Update all tests
6. ✅ Full regression testing (144+ tests)

**Deliverable**: Clean SDK architecture with 100% test pass rate

---

### Phase G: Documentation & Deployment (1 Day)

**Goal**: Production deployment preparation

**Tasks**:
1. ✅ API documentation (KDoc)
2. ✅ Integration guide for app developers
3. ✅ Troubleshooting guide
4. ✅ Performance benchmarking
5. ✅ AAR publishing (Maven)

**Deliverable**: Production-ready SDK with documentation

---

## 11. API Reference

### 11.1 Public SDK API

#### EdgeAIManager

```kotlin
object EdgeAIManager {
    // Initialization
    fun initialize(context: Context, config: SDKConfig = SDKConfig())

    // Service control
    fun startService()
    fun stopService()

    // Listener management
    fun registerSensorStatusListener(listener: SensorStatusListener)
    fun unregisterSensorStatusListener(listener: SensorStatusListener)

    // Sensor observation
    fun getConnectedSensors(): StateFlow<List<SensorStatus>>
    fun scanForSensors()

    // Utility
    fun getVersion(): String
}
```

#### SDKConfig

```kotlin
data class SDKConfig(
    val autoStart: Boolean = true,
    val autoSensorDetection: Boolean = true,
    val autoDataCollection: Boolean = true,
    val inferenceIntervalSeconds: Int = 60,
    val canDataFrequencyHz: Int = 1,
    val enableLogging: Boolean = BuildConfig.DEBUG,
    val mqttEnabled: Boolean = true,
    val bleEnabled: Boolean = true,
    val sensorConfigs: Map<SensorType, SensorConfig> = emptyMap()
)
```

#### SensorStatusListener

```kotlin
interface SensorStatusListener {
    fun onSensorConnected(status: SensorStatus)
    fun onSensorDisconnected(status: SensorStatus)
    fun onDataReceived(sensorId: String, sensorType: SensorType, data: Any)
    fun onSensorError(sensorId: String, error: String)
    fun onInferenceComplete(result: Any)
}
```

---

## 12. Security & Privacy

### 12.1 USB Security

**VID/PID Whitelisting**:
- Only connect to recognized sensor VID/PID combinations
- Reject unknown USB devices automatically
- User permission required for first-time connection

**Data Validation**:
- Validate all incoming USB data (CRC, range checks)
- Timeout on blocking reads (prevent DoS)
- Rate limiting on data ingestion

### 12.2 BLE Security

**Pairing & Authentication**:
- BLE pairing required for all sensors
- MITM protection enabled
- Secure Connection (LE Secure Connections)

**Characteristic Security**:
- Encrypted reads/writes
- Authentication required
- Bonding for persistent pairing

### 12.3 Data Privacy

**PII Protection**:
- No PII stored in local database
- GPS coordinates truncated to 100m precision
- Driver identity anonymized (SHA-256 hash)

**TLS/SSL**:
- MQTT over TLS 1.3
- Certificate pinning for Fleet AI Platform
- No plaintext data transmission

---

## 13. Testing Strategy

### 13.1 Unit Tests

**Target Coverage**: ≥80%

**Test Suites**:
- `EdgeAIManagerTest` - SDK initialization, service control
- `SensorAutoDetectorTest` - USB/BLE detection logic
- `MultiSensorManagerTest` - Sensor lifecycle management
- `AutoDataCollectorTest` - Data collection flows
- `SensorTypeTest` - Enum logic, connection type validation

### 13.2 Integration Tests

**Test Scenarios**:
1. **Boot to Service Start** - BootReceiver → EdgeAIManager → DTGForegroundService
2. **USB Sensor Detection** - Attach STM32 → Auto-detect → Connect → Collect data
3. **BLE Sensor Detection** - Scan → Connect → Pair → Collect data
4. **Multi-Sensor Simultaneous** - 3+ sensors connected, data collection parallel
5. **Reconnection** - Disconnect sensor → Auto-reconnect after 5s
6. **AI Inference** - Collect 60s data → Inference → MQTT publish

### 13.3 Hardware-in-Loop Tests

**Required Hardware**:
- STM32 DTG device
- Parking sensor (USB)
- BLE temperature sensor
- BLE tire sensor (TPMS)
- Android tablet (Snapdragon 865)

**Test Plan**:
1. Connect all sensors simultaneously
2. Run for 24 hours continuous
3. Monitor memory, CPU, battery
4. Verify data quality
5. Stress test (disconnect/reconnect cycles)

---

## 14. Success Criteria

### 14.1 Functional

- ✅ SDK AAR build successful
- ✅ Auto-detection works for USB and BLE sensors
- ✅ All 5+ sensor types supported
- ✅ Auto data collection starts on sensor connection
- ✅ Driver UI shows connected devices in real-time
- ✅ Launcher App auto-starts on boot
- ✅ AI inference runs every 60 seconds
- ✅ Results transmitted via MQTT/BLE

### 14.2 Non-Functional

- ⏱️ Sensor detection latency < 2 seconds
- 🔌 USB reconnection < 5 seconds
- 📡 BLE reconnection < 10 seconds
- 💾 Memory footprint < 150MB (all sensors active)
- ⚡ CPU usage < 15% average (excluding inference)
- 🔋 Battery drain < 3W average
- 🧪 Test coverage ≥ 80%
- ✅ All 144+ tests passing

---

## 15. Next Steps

**Immediate Actions**:

1. **Review & Approval** - Stakeholder review of this architecture document
2. **Environment Setup** - Local Android SDK environment (web environment cannot implement)
3. **Phase A Start** - Create `edgeai-sdk` module structure
4. **Hardware Procurement** - Order test sensors (parking, temperature, tire, weight)
5. **VID/PID Collection** - Document actual VID/PID for all sensor types

**Open Questions**:

1. Which specific parking sensor model will be used? (Need VID/PID)
2. Which load weight sensor model? (Need communication protocol)
3. Which refrigeration temperature sensor? (Need BLE UUID)
4. Which tire sensor (TPMS) model? (Need BLE UUID)
5. Should dashcam video analysis use USB or Wi-Fi connection?

---

## 16. References

**External Documentation**:
- Android USB Host API: https://developer.android.com/guide/topics/connectivity/usb/host
- Android BLE Guide: https://developer.android.com/guide/topics/connectivity/bluetooth-le
- Nordic BLE Library: https://github.com/NordicSemiconductor/Android-BLE-Library
- ONNX Runtime Mobile: https://onnxruntime.ai/docs/tutorials/mobile/

**Project Documentation**:
- `README.md` - Project overview
- `PROJECT_STATUS.md` - Development status
- `GPU_REQUIRED_TASKS.md` - Phase 2 implementation tasks
- `BLACKBOX_INTEGRATION_FEASIBILITY.md` - Dashcam integration analysis

---

**Document Status**: ✅ Complete - Ready for Review
**Total Estimated Implementation Time**: 14-19 days (local Android SDK environment required)
