# Fleet AI Platform Integration

## Overview

This module handles communication between DTG devices and the GLEC Fleet AI platform:
- MQTT over TLS (1.2/1.3) connection
- Vehicle telemetry upload (QoS 1)
- Command reception (dispatch, firmware updates)
- Offline message queuing (SQLite buffer)
- Gzip compression (60-80% reduction)

## Architecture

```
DTG Device
    ↓ (MQTT over TLS)
Fleet AI Platform (mqtt.glec.ai:8883)
    ├── Topic: fleet/vehicles/{vehicle_id}/telemetry (Publish)
    ├── Topic: fleet/vehicles/{vehicle_id}/commands (Subscribe)
    └── Topic: fleet/vehicles/{vehicle_id}/status (Publish)
```

## Message Protocol

### Telemetry Message (Publish)

```json
{
  "vehicle_id": "GLEC-DTG-001",
  "timestamp": 1699564800000,
  "location": {
    "lat": 37.5665,
    "lon": 126.9780,
    "speed": 80.5,
    "heading": 45.2
  },
  "diagnostics": {
    "engine_rpm": 2500,
    "fuel_level": 75.3,
    "battery_voltage": 12.6,
    "coolant_temp": 85.0
  },
  "ai_results": {
    "fuel_efficiency": 12.5,
    "safety_score": 85,
    "carbon_emission": 120.3,
    "anomalies": ["harsh_braking"]
  }
}
```

### Command Message (Subscribe)

```json
{
  "command": "ASSIGN_DISPATCH",
  "dispatch_id": "D123456",
  "destination": {
    "lat": 37.5012,
    "lon": 127.0396,
    "address": "서울시 강남구..."
  },
  "cargo_weight": 5000,
  "deadline": 1699568400000
}
```

## QoS Levels

| Message Type | QoS | Reason |
|--------------|-----|--------|
| Telemetry | 0/1 | Occasional loss acceptable / Recommended |
| AI Results | 1 | Must be delivered |
| Commands | 1 | Critical for dispatch management |
| Safety Alerts | 2 | Exactly-once delivery required |

## Directory Structure

```
fleet-integration/
├── mqtt-client/
│   ├── mqtt_client.py        # Python MQTT client
│   ├── message_buffer.py     # Offline queuing
│   └── compression.py        # Gzip compression
└── protocol/
    ├── schemas.json          # JSON schemas
    ├── telemetry_schema.json
    └── command_schema.json
```

## Configuration

```python
# MQTT Connection
MQTT_BROKER = "mqtt.glec.ai"
MQTT_PORT = 8883  # TLS
MQTT_KEEPALIVE = 60
MQTT_QOS = 1
MQTT_CLEAN_SESSION = False

# Topics
TELEMETRY_TOPIC = "fleet/vehicles/{vehicle_id}/telemetry"
COMMAND_TOPIC = "fleet/vehicles/{vehicle_id}/commands"
STATUS_TOPIC = "fleet/vehicles/{vehicle_id}/status"

# Compression
ENABLE_COMPRESSION = True  # 60-80% size reduction
```

## Usage

```python
# Initialize MQTT client
from mqtt_client import FleetMQTTClient

client = FleetMQTTClient(
    vehicle_id="GLEC-DTG-001",
    broker="mqtt.glec.ai",
    port=8883,
    ca_cert="ca.crt"
)

# Connect
client.connect()

# Publish telemetry
telemetry = {
    "vehicle_id": "GLEC-DTG-001",
    "timestamp": int(time.time() * 1000),
    "location": {...},
    "diagnostics": {...},
    "ai_results": {...}
}
client.publish_telemetry(telemetry)

# Subscribe to commands
def on_command(command):
    print(f"Received command: {command['command']}")

client.subscribe_commands(callback=on_command)
```

## Next Steps

1. Implement Python MQTT client with Eclipse Paho
2. Create message schemas and validation
3. Implement offline queuing with SQLite
4. Add compression and encryption
5. Write integration tests
