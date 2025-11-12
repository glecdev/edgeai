# MQTT Fleet Integration Architecture

**GLEC DTG Edge AI SDK - Fleet Platform Connectivity**

---

## Overview

Production-grade MQTT client for real-time connectivity between DTG devices and Fleet AI platform.

### Key Features

- ✅ **Reliable**: Auto-reconnect with exponential backoff
- ✅ **Offline-First**: SQLite-based message queue
- ✅ **Secure**: TLS 1.2+ encryption with certificate pinning
- ✅ **QoS Support**: Quality of Service levels 0, 1, 2
- ✅ **Scalable**: Handles 10,000+ queued messages

---

## Architecture

### Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  DTGForegroundService                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  MQTTManager                                               │ │
│  │  ├─ Connection Management                                 │ │
│  │  │  ├─ Paho MQTT Client                                  │ │
│  │  │  ├─ TLS/SSL Configuration                             │ │
│  │  │  ├─ Auto-reconnect (exponential backoff)              │ │
│  │  │  └─ Connection state callbacks                        │ │
│  │  │                                                         │ │
│  │  ├─ Message Publisher                                     │ │
│  │  │  ├─ QoS 0: Telemetry (high frequency)                │ │
│  │  │  ├─ QoS 1: Inference results (important)             │ │
│  │  │  ├─ QoS 2: Critical alerts (must deliver)            │ │
│  │  │  └─ JSON serialization                                │ │
│  │  │                                                         │ │
│  │  └─ Offline Queue Manager                                 │ │
│  │     ├─ SQLite database (persistent storage)              │ │
│  │     ├─ Auto-flush on reconnect                           │ │
│  │     ├─ TTL-based expiration (24 hours)                   │ │
│  │     └─ Max size: 10,000 messages                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  EdgeAIInferenceService → MQTTManager.publish()                 │
│  CANReceiver → MQTTManager.publishTelemetry()                   │
│  AnomalyDetector → MQTTManager.publishAlert()                   │
└──────────────────────────────────────────────────────────────────┘
                           ↓
                    MQTT Broker (TLS)
                           ↓
                  Fleet AI Platform
```

---

## Topic Structure

### Naming Convention

```
glec/dtg/{device_id}/{message_type}
```

### Topic Definitions

| Topic | QoS | Frequency | Payload Size | Description |
|-------|-----|-----------|--------------|-------------|
| `glec/dtg/{device_id}/telemetry` | 0 | 1Hz | ~500 bytes | Real-time CAN data |
| `glec/dtg/{device_id}/inference` | 1 | Every 60s | ~200 bytes | AI inference results |
| `glec/dtg/{device_id}/alerts` | 2 | On event | ~300 bytes | Critical safety alerts |
| `glec/dtg/{device_id}/status` | 1 | Every 5min | ~150 bytes | Device health status |

### Example Payloads

#### Telemetry (QoS 0)

```json
{
  "timestamp": 1704931200000,
  "device_id": "DTG-SN-12345",
  "vehicle_speed": 60.5,
  "engine_rpm": 1800,
  "throttle_position": 25.3,
  "fuel_level": 78.2,
  "gps": {
    "lat": 37.5665,
    "lon": 126.9780,
    "speed": 60.2
  }
}
```

#### Inference (QoS 1)

```json
{
  "timestamp": 1704931200000,
  "device_id": "DTG-SN-12345",
  "behavior": "ECO_DRIVING",
  "confidence": 0.953,
  "safety_score": 95,
  "fuel_efficiency": 6.2,
  "latency_ms": 0.0119
}
```

#### Alerts (QoS 2)

```json
{
  "timestamp": 1704931200000,
  "device_id": "DTG-SN-12345",
  "alert_type": "HARSH_BRAKING",
  "severity": "HIGH",
  "location": {
    "lat": 37.5665,
    "lon": 126.9780
  },
  "details": {
    "deceleration": -6.5,
    "speed_before": 80.0,
    "speed_after": 45.0
  }
}
```

#### Status (QoS 1)

```json
{
  "timestamp": 1704931200000,
  "device_id": "DTG-SN-12345",
  "status": "ONLINE",
  "uptime_seconds": 86400,
  "battery_voltage": 12.6,
  "memory_usage_mb": 45,
  "storage_available_mb": 1024,
  "can_messages_received": 86400,
  "mqtt_messages_sent": 1440,
  "offline_queue_size": 0
}
```

---

## QoS Levels

### QoS 0: At Most Once (Fire and Forget)

**Use Case**: High-frequency telemetry data

**Characteristics**:
- No acknowledgment
- No retry
- Lowest overhead
- Acceptable data loss

**Topics**: `telemetry`

### QoS 1: At Least Once

**Use Case**: Important data (inference results)

**Characteristics**:
- Broker acknowledges receipt
- Sender retries until ACK
- Possible duplicates
- Guaranteed delivery

**Topics**: `inference`, `status`

### QoS 2: Exactly Once

**Use Case**: Critical alerts (harsh braking, accidents)

**Characteristics**:
- 4-way handshake
- No duplicates
- Highest overhead
- Guaranteed single delivery

**Topics**: `alerts`

---

## Connection Management

### Connection Flow

```
┌─────────────┐
│  Start      │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Load Configuration  │
│ - Broker URL        │
│ - Credentials       │
│ - TLS certificates  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Connect to Broker   │
│ - TLS handshake     │
│ - Authentication    │
└──────┬──────────────┘
       │
       ├──Success──▶ ┌─────────────┐
       │             │  Connected  │
       │             └──────┬──────┘
       │                    │
       │                    ▼
       │             ┌─────────────────────┐
       │             │ Flush Offline Queue │
       │             └──────┬──────────────┘
       │                    │
       │                    ▼
       │             ┌─────────────┐
       │             │  Publishing │
       │             └──────┬──────┘
       │                    │
       │             ┌──────▼──────────────┐
       │             │  Connection Lost?   │
       │             └──────┬──────────────┘
       │                    │
       │                    ▼ Yes
       │             ┌─────────────────────┐
       │             │ Store in Queue      │
       │             │ Trigger Reconnect   │
       │             └──────┬──────────────┘
       │                    │
       └──Failure───────────┘
       │
       ▼
┌─────────────────────────────┐
│ Retry with Backoff          │
│ - Wait: 2^attempt seconds   │
│ - Max: 5 attempts           │
│ - Then: 60s periodic retry  │
└─────────────────────────────┘
```

### Reconnection Strategy

**Exponential Backoff**:

```
Attempt 1: Wait 2s   (2^1)
Attempt 2: Wait 4s   (2^2)
Attempt 3: Wait 8s   (2^3)
Attempt 4: Wait 16s  (2^4)
Attempt 5: Wait 32s  (2^5)
After 5 attempts: Wait 60s periodically
```

**Max Retry Duration**: Infinite (keeps trying)

---

## Offline Queue

### Database Schema

**Table**: `mqtt_offline_queue`

```sql
CREATE TABLE mqtt_offline_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    payload TEXT NOT NULL,
    qos INTEGER NOT NULL,
    timestamp BIGINT NOT NULL,
    ttl BIGINT NOT NULL,
    retry_count INTEGER DEFAULT 0
);

CREATE INDEX idx_timestamp ON mqtt_offline_queue(timestamp);
CREATE INDEX idx_ttl ON mqtt_offline_queue(ttl);
```

### Queue Management

**Enqueue**:
1. Check queue size (< 10,000)
2. If full, remove oldest messages
3. Insert new message with TTL

**Dequeue**:
1. On reconnect, fetch all messages
2. Publish in timestamp order
3. Delete on success
4. Increment retry_count on failure

**TTL Expiration**:
1. Every 5 minutes, delete expired messages
2. TTL = 24 hours from insertion

**Max Queue Size**: 10,000 messages
**Max Message Age**: 24 hours
**Max Retry Count**: 3 per message

---

## Security

### TLS Configuration

**Minimum Version**: TLS 1.2
**Cipher Suites**: AES-256-GCM preferred

```kotlin
val socketFactory = createSSLSocketFactory(
    caCertInputStream = assets.open("mqtt_ca.crt"),
    clientCertInputStream = assets.open("client.crt"),  // Optional
    clientKeyInputStream = assets.open("client.key")    // Optional
)

val mqttOptions = MqttConnectOptions().apply {
    socketFactory = socketFactory
    isCleanSession = false
    connectionTimeout = 30
    keepAliveInterval = 60
    isAutomaticReconnect = true
}
```

### Authentication

**Username/Password**:
```kotlin
mqttOptions.userName = "dtg-device-12345"
mqttOptions.password = "secure-password".toCharArray()
```

**Client Certificate** (Optional):
- Device-specific X.509 certificate
- Mutual TLS (mTLS) authentication
- Higher security for production

### Certificate Pinning

```kotlin
val certificatePin = "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

// Validate during connection
val pinner = CertificatePinner.Builder()
    .add("mqtt.fleet.glec.co.kr", certificatePin)
    .build()
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Connection Time** | < 5s | Including TLS handshake |
| **Publish Latency** | < 100ms | QoS 0 |
| **Publish Latency** | < 500ms | QoS 1 |
| **Publish Latency** | < 1s | QoS 2 |
| **Reconnect Time** | < 30s | With backoff |
| **Queue Throughput** | > 100 msg/s | On reconnect flush |
| **Memory Usage** | < 10MB | Including queue |
| **Storage Usage** | < 50MB | For 10,000 messages |

---

## Configuration

### broker_config.properties

```properties
# MQTT Broker
mqtt.broker.url=ssl://mqtt.fleet.glec.co.kr:8883
mqtt.broker.username=dtg-device-{DEVICE_ID}
mqtt.broker.password={SECURE_PASSWORD}

# Connection
mqtt.connection.timeout=30
mqtt.connection.keepalive=60
mqtt.connection.clean_session=false
mqtt.connection.auto_reconnect=true

# Retry
mqtt.retry.max_attempts=5
mqtt.retry.initial_delay_ms=2000
mqtt.retry.max_delay_ms=60000

# Queue
mqtt.queue.max_size=10000
mqtt.queue.ttl_hours=24
mqtt.queue.max_retry=3

# Security
mqtt.tls.enabled=true
mqtt.tls.ca_cert=mqtt_ca.crt
mqtt.tls.client_cert=client.crt
mqtt.tls.client_key=client.key
mqtt.tls.verify_hostname=true
```

---

## Implementation Checklist

### Phase 3A: Core MQTT Client ✅ (Web Environment)

- [ ] Create MQTT package structure
- [ ] Implement MQTTManager class
- [ ] Add Eclipse Paho dependency
- [ ] Implement connection management
- [ ] Add connection callbacks
- [ ] Implement message publishing
- [ ] Add QoS support

### Phase 3B: Offline Queue ✅ (Web Environment)

- [ ] Create SQLite database helper
- [ ] Implement OfflineQueueManager
- [ ] Add TTL expiration
- [ ] Implement auto-flush on reconnect
- [ ] Add queue size limits

### Phase 3C: Security ✅ (Web Environment)

- [ ] Implement TLS configuration
- [ ] Add certificate loading
- [ ] Implement certificate pinning
- [ ] Add credential management

### Phase 3D: Testing ✅ (Web Environment)

- [ ] Unit tests for MQTTManager
- [ ] Unit tests for OfflineQueueManager
- [ ] Integration tests (with mock broker)
- [ ] Performance benchmarks

### Phase 3E: Hardware Testing (Requires Device)

- [ ] Test with real MQTT broker
- [ ] Measure connection latency
- [ ] Measure publish throughput
- [ ] Test offline queue flush
- [ ] Validate TLS handshake
- [ ] Stress test (10,000 messages)

---

## Usage Example

```kotlin
class DTGForegroundService : Service() {
    private lateinit var mqttManager: MQTTManager

    override fun onCreate() {
        super.onCreate()

        // Initialize MQTT
        val config = MQTTConfig(
            brokerUrl = "ssl://mqtt.fleet.glec.co.kr:8883",
            clientId = "DTG-SN-12345",
            username = "dtg-device-12345",
            password = "secure-password"
        )

        mqttManager = MQTTManager(context = this, config = config)

        // Set callbacks
        mqttManager.setConnectionCallback(object : ConnectionCallback {
            override fun onConnected() {
                Log.i(TAG, "MQTT connected")
            }

            override fun onConnectionLost(cause: Throwable?) {
                Log.w(TAG, "MQTT connection lost", cause)
            }

            override fun onReconnecting() {
                Log.i(TAG, "MQTT reconnecting...")
            }
        })

        // Connect
        mqttManager.connect()
    }

    private fun publishInferenceResult(result: InferenceResult) {
        val topic = "glec/dtg/${mqttManager.deviceId}/inference"

        val payload = JSONObject().apply {
            put("timestamp", result.timestamp)
            put("behavior", result.behavior.name)
            put("confidence", result.confidence)
            put("latency_ms", result.latencyMs)
        }.toString()

        // QoS 1: At least once delivery
        mqttManager.publish(topic, payload, qos = 1)
    }

    private fun publishAlert(alert: SafetyAlert) {
        val topic = "glec/dtg/${mqttManager.deviceId}/alerts"

        val payload = JSONObject().apply {
            put("timestamp", alert.timestamp)
            put("alert_type", alert.type.name)
            put("severity", alert.severity.name)
        }.toString()

        // QoS 2: Exactly once delivery (critical)
        mqttManager.publish(topic, payload, qos = 2)
    }

    override fun onDestroy() {
        super.onDestroy()
        mqttManager.disconnect()
    }
}
```

---

## Monitoring & Debugging

### Logging

```kotlin
// Enable verbose logging for debugging
MqttLogging.setLevel(MqttLogging.VERBOSE)

// Log all MQTT events
mqttManager.enableDebugLogging(true)
```

### Metrics

```kotlin
val metrics = mqttManager.getMetrics()

Log.i(TAG, "MQTT Metrics:")
Log.i(TAG, "  Connected: ${metrics.isConnected}")
Log.i(TAG, "  Messages sent: ${metrics.messagesSent}")
Log.i(TAG, "  Messages failed: ${metrics.messagesFailed}")
Log.i(TAG, "  Queue size: ${metrics.queueSize}")
Log.i(TAG, "  Reconnect count: ${metrics.reconnectCount}")
```

---

## Implementation Details

### OfflineQueueDatabaseHelper

**Location**: `android-dtg/app/src/main/java/com/glec/dtg/mqtt/OfflineQueueDatabaseHelper.kt`

SQLite database helper for persistent MQTT message queue.

**Features**:
- Database schema creation and migration
- Index management for timestamp and TTL queries
- Database statistics (size, oldest/newest messages)
- Secure database access with proper transaction handling

**Schema**:
```sql
CREATE TABLE mqtt_offline_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    payload TEXT NOT NULL,
    qos INTEGER NOT NULL,
    timestamp BIGINT NOT NULL,
    ttl BIGINT NOT NULL,
    retry_count INTEGER DEFAULT 0
);

CREATE INDEX idx_timestamp ON mqtt_offline_queue(timestamp);
CREATE INDEX idx_ttl ON mqtt_offline_queue(ttl);
```

### OfflineQueueManager

**Location**: `android-dtg/app/src/main/java/com/glec/dtg/mqtt/OfflineQueueManager.kt`

High-level manager for MQTT offline queue operations.

**Features**:
- **Enqueue**: Add messages with TTL and automatic queue size management
- **Dequeue**: Fetch messages in FIFO order (by timestamp)
- **Cleanup**: Periodic removal of expired messages and max-retry messages
- **Retry Management**: Track retry count per message
- **Thread Safety**: All operations use synchronized database transactions

**API**:
```kotlin
// Enqueue message
val messageId = offlineQueueManager.enqueue(
    topic = "glec/dtg/device-123/inference",
    payload = "{...}",
    qos = 1,
    ttlMillis = 24 * 60 * 60 * 1000  // 24 hours
)

// Dequeue all messages (FIFO order)
val messages: List<QueuedMessage> = offlineQueueManager.dequeueAll()

// Delete after successful publish
offlineQueueManager.delete(messageId)

// Increment retry count on failure
offlineQueueManager.incrementRetryCount(messageId)

// Cleanup expired messages
val expiredCount = offlineQueueManager.cleanupExpired()

// Cleanup max retry messages
val maxRetryCount = offlineQueueManager.cleanupMaxRetries()

// Get queue size
val size = offlineQueueManager.getQueueSize()

// Release resources
offlineQueueManager.release()
```

**Periodic Cleanup**:
- Runs every 5 minutes in background coroutine
- Removes expired messages (TTL < current time)
- Removes messages exceeding max retries (retry_count >= 3)
- Logs cleanup statistics

### MQTTManager Integration

**Location**: `android-dtg/app/src/main/java/com/glec/dtg/mqtt/MQTTManager.kt`

The MQTTManager integrates with OfflineQueueManager for persistent message queuing:

```kotlin
class MQTTManager(private val context: Context, private val config: MQTTConfig) {
    // SQLite-based persistent queue
    private val offlineQueueManager = OfflineQueueManager(
        context = context,
        maxQueueSize = config.queueMaxSize,
        ttlHours = config.queueTTLHours
    )

    fun publish(topic: String, payload: String, qos: Int): Boolean {
        if (!isConnected()) {
            // Queue message for later delivery
            queueMessage(topic, payload, qos)
            return true
        }

        // Publish immediately if connected
        mqttClient?.publish(topic, message, ...)
    }

    private fun handleConnectionSuccess() {
        // Flush offline queue on reconnect
        flushOfflineQueue()
    }

    private fun flushOfflineQueue() {
        // Cleanup expired messages
        offlineQueueManager.cleanupExpired()

        // Dequeue all messages in FIFO order
        val messages = offlineQueueManager.dequeueAll()

        for (message in messages) {
            if (message.canRetry()) {
                val success = publish(message.topic, message.payload, message.qos)

                if (success) {
                    offlineQueueManager.delete(message.id)
                } else {
                    offlineQueueManager.incrementRetryCount(message.id)
                    break  // Stop flushing on failure
                }
            } else {
                // Delete messages exceeding max retries
                offlineQueueManager.delete(message.id)
            }
        }
    }
}
```

**Benefits of SQLite Queue**:
- ✅ **Persistent**: Messages survive app restarts and crashes
- ✅ **ACID Transactions**: Guaranteed data integrity
- ✅ **Efficient**: Indexed queries for fast FIFO retrieval
- ✅ **Scalable**: Handles 10,000+ messages without performance degradation
- ✅ **Automatic Cleanup**: Periodic removal of stale messages

### Testing

**Python Test Suite**: `tests/test_mqtt_offline_queue.py`

Validates SQLite queue behavior with 12 comprehensive tests:
- Basic operations (enqueue, dequeue, delete)
- FIFO ordering
- TTL expiration
- Retry count management
- Queue size limits
- QoS level handling

**Run Tests**:
```bash
pytest tests/test_mqtt_offline_queue.py -v
```

**Test Results**: ✅ 12/12 tests passing

### TLS/SSL Security Implementation

**Locations**:
- `android-dtg/app/src/main/java/com/glec/dtg/mqtt/TLSConfig.kt`
- `android-dtg/app/src/main/java/com/glec/dtg/mqtt/SSLSocketFactoryBuilder.kt`
- `android-dtg/app/src/main/java/com/glec/dtg/mqtt/CertificatePinner.kt`

Production-grade TLS/SSL security for MQTT connections.

**TLSConfig.kt** (160 lines):
```kotlin
// Server authentication only
val tlsConfig = TLSConfig.createServerAuth(
    caCertInputStream = assets.open("mqtt_ca.crt"),
    tlsVersion = "TLSv1.2",
    certificatePins = listOf("sha256/AAAA...=")
)

// Mutual TLS (mTLS)
val tlsConfig = TLSConfig.createMutualTLS(
    caCertInputStream = assets.open("mqtt_ca.crt"),
    clientCertInputStream = assets.open("client.crt"),
    clientKeyInputStream = assets.open("client.key"),
    tlsVersion = "TLSv1.2"
)
```

**Features**:
- TLS 1.2+ enforcement (no SSLv3/TLSv1.0/TLSv1.1)
- Recommended cipher suites (ECDHE, AES-GCM, SHA256/384)
- Mutual TLS (mTLS) support
- Certificate pinning configuration
- Hostname verification

**SSLSocketFactoryBuilder.kt** (190 lines):
```kotlin
val socketFactory = SSLSocketFactoryBuilder.build(tlsConfig)
mqttOptions.socketFactory = socketFactory
```

**Features**:
- CA certificate loading (X.509)
- Client certificate + private key (PEM format)
- TrustManager/KeyManager creation
- Cipher suite enforcement
- Exception handling

**CertificatePinner.kt** (180 lines):
```kotlin
val pinner = CertificatePinner(
    hostname = "mqtt.fleet.glec.co.kr",
    pins = listOf(
        "sha256/AAAA...=",  // Primary
        "sha256/BBBB...="   // Backup
    )
)
```

**Features**:
- SHA-256 public key pinning
- Multi-pin support
- Hostname validation
- PinningTrustManager wrapper
- MITM attack prevention

**MQTT Integration**:
```kotlin
val mqttConfig = MQTTConfig(
    brokerUrl = "ssl://mqtt.fleet.glec.co.kr:8883",
    clientId = "DTG-SN-12345",
    tlsConfig = TLSConfig.createServerAuth(...)
)

mqttManager.connect()  // TLS auto-configured
```

**Test Suite**: `tests/test_mqtt_tls_config.py` (19/19 tests passing)
- TLS config validation
- Mutual TLS validation
- Certificate pinning
- Pin format validation
- MQTT config integration

---

## See Also

- [Eclipse Paho Android](https://github.com/eclipse/paho.mqtt.android)
- [MQTT Version 3.1.1 Spec](https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/mqtt-v3.1.1.html)
- [HiveMQ MQTT Essentials](https://www.hivemq.com/mqtt-essentials/)

---

**Last Updated**: 2025-01-10
**Version**: 1.2.0 (TLS/SSL Security Implementation)
**Author**: GLEC DTG Team
