"""
GLEC DTG - Fleet AI Platform MQTT Client
Secure MQTT client with TLS, offline queuing, and compression
"""

import json
import gzip
import sqlite3
import time
import ssl
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

import paho.mqtt.client as mqtt


@dataclass
class TelemetryMessage:
    """Vehicle telemetry message"""
    vehicle_id: str
    timestamp: int
    location: Dict[str, float]  # lat, lon, speed, heading
    diagnostics: Dict[str, float]  # engine_rpm, fuel_level, etc.
    ai_results: Dict[str, any]  # fuel_efficiency, safety_score, etc.


@dataclass
class CommandMessage:
    """Fleet command message"""
    command: str
    dispatch_id: Optional[str] = None
    destination: Optional[Dict[str, float]] = None
    cargo_weight: Optional[float] = None
    deadline: Optional[int] = None


class OfflineMessageBuffer:
    """
    SQLite-based message buffer for offline queueing

    Stores messages when connection is lost and replays when restored
    """

    def __init__(self, db_path: str = 'mqtt_buffer.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mqtt_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                payload BLOB NOT NULL,
                qos INTEGER NOT NULL,
                timestamp INTEGER NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def enqueue(self, topic: str, payload: bytes, qos: int):
        """Add message to queue"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO mqtt_queue (topic, payload, qos, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (topic, payload, qos, int(time.time() * 1000)))

        conn.commit()
        conn.close()

    def dequeue_all(self):
        """Get all pending messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, topic, payload, qos FROM mqtt_queue
            ORDER BY timestamp ASC
        ''')

        messages = cursor.fetchall()
        conn.close()

        return messages

    def remove(self, message_ids: list):
        """Remove messages from queue"""
        if not message_ids:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join(['?'] * len(message_ids))
        cursor.execute(f'DELETE FROM mqtt_queue WHERE id IN ({placeholders})', message_ids)

        conn.commit()
        conn.close()

    def count(self) -> int:
        """Get number of pending messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM mqtt_queue')
        count = cursor.fetchone()[0]

        conn.close()
        return count


class FleetMQTTClient:
    """
    Fleet AI Platform MQTT Client

    Features:
    - TLS 1.2/1.3 encryption
    - Automatic reconnection
    - Offline message queuing
    - Gzip compression (60-80% reduction)
    - QoS 0/1/2 support
    """

    def __init__(self,
                 vehicle_id: str,
                 broker: str = 'mqtt.glec.ai',
                 port: int = 8883,
                 ca_cert: Optional[str] = None,
                 client_cert: Optional[str] = None,
                 client_key: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 enable_compression: bool = True):
        """
        Initialize MQTT client

        Args:
            vehicle_id: Unique vehicle identifier
            broker: MQTT broker hostname
            port: MQTT broker port (8883 for TLS)
            ca_cert: Path to CA certificate
            client_cert: Path to client certificate (optional)
            client_key: Path to client key (optional)
            username: MQTT username (optional)
            password: MQTT password (optional)
            enable_compression: Enable gzip compression
        """
        self.vehicle_id = vehicle_id
        self.broker = broker
        self.port = port
        self.enable_compression = enable_compression

        # MQTT topics
        self.telemetry_topic = f"fleet/vehicles/{vehicle_id}/telemetry"
        self.command_topic = f"fleet/vehicles/{vehicle_id}/commands"
        self.status_topic = f"fleet/vehicles/{vehicle_id}/status"

        # Create MQTT client
        self.client = mqtt.Client(
            client_id=f"dtg_{vehicle_id}",
            clean_session=False,  # Persist session
            protocol=mqtt.MQTTv311
        )

        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish

        # Configure TLS
        if ca_cert:
            self.client.tls_set(
                ca_certs=ca_cert,
                certfile=client_cert,
                keyfile=client_key,
                tls_version=ssl.PROTOCOL_TLSv1_2
            )
            self.client.tls_insecure_set(False)

        # Set authentication
        if username and password:
            self.client.username_pw_set(username, password)

        # Offline message buffer
        self.message_buffer = OfflineMessageBuffer()

        # Connection state
        self.connected = False

        # Command callback
        self.command_callback: Optional[Callable[[CommandMessage], None]] = None

    def connect(self):
        """Connect to MQTT broker"""
        print(f"Connecting to MQTT broker: {self.broker}:{self.port}")

        try:
            self.client.connect(
                host=self.broker,
                port=self.port,
                keepalive=60
            )

            # Start network loop in background thread
            self.client.loop_start()

            print("✅ MQTT client started")

        except Exception as e:
            print(f"❌ MQTT connection failed: {e}")
            raise

    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        print("🔌 MQTT client disconnected")

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker"""
        if rc == 0:
            print(f"✅ Connected to MQTT broker")
            self.connected = True

            # Subscribe to command topic
            self.client.subscribe(self.command_topic, qos=1)
            print(f"✅ Subscribed to: {self.command_topic}")

            # Publish online status
            self.publish_status("online")

            # Replay buffered messages
            self._replay_buffered_messages()

        else:
            error_messages = {
                1: "Incorrect protocol version",
                2: "Invalid client identifier",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized"
            }
            print(f"❌ Connection failed: {error_messages.get(rc, f'Unknown error ({rc})')}")
            self.connected = False

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker"""
        self.connected = False

        if rc != 0:
            print(f"⚠️  Unexpected disconnection (rc={rc}). Reconnecting...")
        else:
            print("🔌 Disconnected from broker")

    def _on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            # Decompress if needed
            payload = msg.payload
            if self.enable_compression:
                try:
                    payload = gzip.decompress(payload)
                except:
                    pass  # Not compressed

            # Parse JSON
            data = json.loads(payload.decode('utf-8'))

            # Handle command
            if msg.topic == self.command_topic:
                command = CommandMessage(**data)
                print(f"📨 Received command: {command.command}")

                if self.command_callback:
                    self.command_callback(command)

        except Exception as e:
            print(f"❌ Error processing message: {e}")

    def _on_publish(self, client, userdata, mid):
        """Callback when message published"""
        pass  # Can log successful publishes if needed

    def _replay_buffered_messages(self):
        """Replay messages that were buffered during offline period"""
        pending = self.message_buffer.count()

        if pending > 0:
            print(f"📤 Replaying {pending} buffered messages...")

            messages = self.message_buffer.dequeue_all()
            success_ids = []

            for msg_id, topic, payload, qos in messages:
                try:
                    result = self.client.publish(topic, payload, qos=qos)

                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        success_ids.append(msg_id)

                except Exception as e:
                    print(f"⚠️  Failed to replay message {msg_id}: {e}")

            # Remove successfully sent messages
            if success_ids:
                self.message_buffer.remove(success_ids)
                print(f"✅ Replayed {len(success_ids)} messages")

    def _compress_payload(self, payload: bytes) -> bytes:
        """Compress payload with gzip"""
        if not self.enable_compression:
            return payload

        compressed = gzip.compress(payload)

        # Only use compression if it actually reduces size
        if len(compressed) < len(payload):
            return compressed
        else:
            return payload

    def publish_telemetry(self, telemetry: TelemetryMessage, qos: int = 1):
        """
        Publish vehicle telemetry

        Args:
            telemetry: Telemetry message
            qos: Quality of Service (0, 1, or 2)
        """
        # Convert to JSON
        payload_json = json.dumps(asdict(telemetry))
        payload_bytes = payload_json.encode('utf-8')

        # Compress
        payload_compressed = self._compress_payload(payload_bytes)

        # Publish
        if self.connected:
            try:
                result = self.client.publish(
                    self.telemetry_topic,
                    payload_compressed,
                    qos=qos
                )

                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                    print(f"⚠️  Publish failed, buffering message")
                    self.message_buffer.enqueue(self.telemetry_topic, payload_compressed, qos)

            except Exception as e:
                print(f"❌ Publish error: {e}")
                self.message_buffer.enqueue(self.telemetry_topic, payload_compressed, qos)

        else:
            # Buffer message for later
            print(f"📥 Offline - buffering telemetry message")
            self.message_buffer.enqueue(self.telemetry_topic, payload_compressed, qos)

    def publish_status(self, status: str):
        """Publish vehicle status (online/offline)"""
        payload = json.dumps({
            "vehicle_id": self.vehicle_id,
            "status": status,
            "timestamp": int(time.time() * 1000)
        })

        self.client.publish(self.status_topic, payload, qos=1)

    def set_command_callback(self, callback: Callable[[CommandMessage], None]):
        """Set callback for received commands"""
        self.command_callback = callback


# Example usage
if __name__ == "__main__":
    # Create client
    client = FleetMQTTClient(
        vehicle_id="GLEC-DTG-001",
        broker="mqtt.glec.ai",
        port=8883,
        ca_cert="ca.crt",  # Path to CA certificate
        username="dtg_device",
        password="secure_password",
        enable_compression=True
    )

    # Set command callback
    def handle_command(command: CommandMessage):
        print(f"Handling command: {command.command}")
        if command.command == "ASSIGN_DISPATCH":
            print(f"  Dispatch ID: {command.dispatch_id}")
            print(f"  Destination: {command.destination}")

    client.set_command_callback(handle_command)

    # Connect
    client.connect()

    try:
        # Simulate telemetry publishing
        while True:
            telemetry = TelemetryMessage(
                vehicle_id="GLEC-DTG-001",
                timestamp=int(time.time() * 1000),
                location={
                    "lat": 37.5665,
                    "lon": 126.9780,
                    "speed": 80.5,
                    "heading": 45.2
                },
                diagnostics={
                    "engine_rpm": 2500,
                    "fuel_level": 75.3,
                    "battery_voltage": 12.6
                },
                ai_results={
                    "fuel_efficiency": 12.5,
                    "safety_score": 85,
                    "carbon_emission": 120.3,
                    "anomalies": []
                }
            )

            client.publish_telemetry(telemetry, qos=1)

            time.sleep(60)  # Publish every 60 seconds

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    finally:
        client.disconnect()
