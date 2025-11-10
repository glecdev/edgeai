#!/usr/bin/env python3
"""
GLEC DTG - MQTT Offline Queue Tests

Tests for SQLite-based MQTT offline queue functionality.
Validates queue operations, TTL expiration, retry limits, and FIFO ordering.

Run: pytest tests/test_mqtt_offline_queue.py -v
"""

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Optional


class QueuedMessage:
    """Data class for queued MQTT messages"""

    def __init__(
        self,
        id: int,
        topic: str,
        payload: str,
        qos: int,
        timestamp: int,
        ttl: int,
        retry_count: int = 0
    ):
        self.id = id
        self.topic = topic
        self.payload = payload
        self.qos = qos
        self.timestamp = timestamp
        self.ttl = ttl
        self.retry_count = retry_count

    def is_expired(self) -> bool:
        """Check if message is expired based on TTL"""
        return int(time.time() * 1000) > self.ttl

    def can_retry(self, max_retries: int = 3) -> bool:
        """Check if message can be retried"""
        return self.retry_count < max_retries


class PythonOfflineQueueManager:
    """
    Python implementation of OfflineQueueManager for testing.
    Mirrors Kotlin implementation behavior.
    """

    def __init__(
        self,
        db_path: str,
        max_queue_size: int = 10_000,
        ttl_hours: int = 24,
        max_retries: int = 3
    ):
        self.db_path = db_path
        self.max_queue_size = max_queue_size
        self.ttl_hours = ttl_hours
        self.max_retries = max_retries

        # Create database
        self._create_database()

    def _create_database(self):
        """Create SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mqtt_offline_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                payload TEXT NOT NULL,
                qos INTEGER NOT NULL,
                timestamp BIGINT NOT NULL,
                ttl BIGINT NOT NULL,
                retry_count INTEGER DEFAULT 0
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON mqtt_offline_queue(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON mqtt_offline_queue(ttl)")

        conn.commit()
        conn.close()

    def enqueue(
        self,
        topic: str,
        payload: str,
        qos: int,
        ttl_millis: Optional[int] = None
    ) -> int:
        """
        Enqueue a message to the offline queue

        Returns:
            Message ID, or -1 if failed
        """
        if ttl_millis is None:
            ttl_millis = self.ttl_hours * 60 * 60 * 1000

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check queue size
            current_size = self.get_queue_size()
            if current_size >= self.max_queue_size:
                # Remove oldest message
                self._remove_oldest_message()

            # Insert message
            current_time = int(time.time() * 1000)
            ttl = current_time + ttl_millis

            cursor.execute("""
                INSERT INTO mqtt_offline_queue (topic, payload, qos, timestamp, ttl, retry_count)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (topic, payload, qos, current_time, ttl))

            message_id = cursor.lastrowid
            conn.commit()

            return message_id

        except Exception as e:
            print(f"Error enqueuing message: {e}")
            return -1

        finally:
            conn.close()

    def dequeue_all(self) -> List[QueuedMessage]:
        """
        Dequeue all messages in FIFO order (by timestamp ASC)

        Returns:
            List of queued messages
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, topic, payload, qos, timestamp, ttl, retry_count
                FROM mqtt_offline_queue
                ORDER BY timestamp ASC
            """)

            rows = cursor.fetchall()

            messages = []
            for row in rows:
                message = QueuedMessage(
                    id=row[0],
                    topic=row[1],
                    payload=row[2],
                    qos=row[3],
                    timestamp=row[4],
                    ttl=row[5],
                    retry_count=row[6]
                )
                messages.append(message)

            return messages

        finally:
            conn.close()

    def delete(self, message_id: int) -> bool:
        """
        Delete a message from the queue

        Returns:
            True if deleted, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM mqtt_offline_queue WHERE id = ?", (message_id,))
            rows_deleted = cursor.rowcount
            conn.commit()

            return rows_deleted > 0

        finally:
            conn.close()

    def increment_retry_count(self, message_id: int) -> bool:
        """
        Increment retry count for a message

        Returns:
            True if updated, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE mqtt_offline_queue
                SET retry_count = retry_count + 1
                WHERE id = ?
            """, (message_id,))

            rows_updated = cursor.rowcount
            conn.commit()

            return rows_updated > 0

        finally:
            conn.close()

    def cleanup_expired(self) -> int:
        """
        Cleanup expired messages

        Returns:
            Number of messages deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            current_time = int(time.time() * 1000)

            cursor.execute("DELETE FROM mqtt_offline_queue WHERE ttl < ?", (current_time,))
            rows_deleted = cursor.rowcount
            conn.commit()

            return rows_deleted

        finally:
            conn.close()

    def cleanup_max_retries(self) -> int:
        """
        Cleanup messages that exceeded max retries

        Returns:
            Number of messages deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM mqtt_offline_queue WHERE retry_count >= ?",
                (self.max_retries,)
            )
            rows_deleted = cursor.rowcount
            conn.commit()

            return rows_deleted

        finally:
            conn.close()

    def get_queue_size(self) -> int:
        """
        Get queue size

        Returns:
            Number of messages in queue
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM mqtt_offline_queue")
            count = cursor.fetchone()[0]
            return count

        finally:
            conn.close()

    def clear(self):
        """Clear all messages (for testing)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM mqtt_offline_queue")
            conn.commit()

        finally:
            conn.close()

    def _remove_oldest_message(self):
        """Remove oldest message to make room"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                DELETE FROM mqtt_offline_queue
                WHERE id = (
                    SELECT id FROM mqtt_offline_queue
                    ORDER BY timestamp ASC
                    LIMIT 1
                )
            """)
            conn.commit()

        finally:
            conn.close()


class TestOfflineQueueBasicOperations(unittest.TestCase):
    """Test basic queue operations (enqueue, dequeue, delete)"""

    def setUp(self):
        """Create temporary database for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"{self.temp_dir}/test_queue.db"
        self.queue = PythonOfflineQueueManager(self.db_path, max_queue_size=100)

    def tearDown(self):
        """Cleanup temporary database"""
        Path(self.db_path).unlink(missing_ok=True)
        Path(self.temp_dir).rmdir()

    def test_enqueue_single_message(self):
        """Test enqueuing a single message"""
        message_id = self.queue.enqueue(
            topic="glec/dtg/device-123/inference",
            payload='{"behavior": "normal"}',
            qos=1,
            ttl_millis=60 * 60 * 1000  # 1 hour
        )

        self.assertNotEqual(message_id, -1)
        self.assertEqual(self.queue.get_queue_size(), 1)

    def test_enqueue_multiple_messages(self):
        """Test enqueuing multiple messages"""
        for i in range(10):
            message_id = self.queue.enqueue(
                topic=f"glec/dtg/device-123/telemetry",
                payload=f'{{"index": {i}}}',
                qos=0,
                ttl_millis=60 * 60 * 1000
            )
            self.assertNotEqual(message_id, -1)

        self.assertEqual(self.queue.get_queue_size(), 10)

    def test_dequeue_all_fifo_order(self):
        """Test dequeuing messages in FIFO order"""
        # Enqueue 5 messages with slight delay
        expected_order = []
        for i in range(5):
            message_id = self.queue.enqueue(
                topic="glec/dtg/device-123/test",
                payload=f'{{"index": {i}}}',
                qos=1,
                ttl_millis=60 * 60 * 1000
            )
            expected_order.append(i)
            time.sleep(0.01)  # Ensure different timestamps

        # Dequeue all
        messages = self.queue.dequeue_all()

        self.assertEqual(len(messages), 5)

        # Verify FIFO order
        for idx, message in enumerate(messages):
            self.assertIn(f'{{"index": {idx}}}', message.payload)

    def test_delete_message(self):
        """Test deleting a message"""
        message_id = self.queue.enqueue(
            topic="glec/dtg/device-123/test",
            payload='{"test": true}',
            qos=1,
            ttl_millis=60 * 60 * 1000
        )

        self.assertEqual(self.queue.get_queue_size(), 1)

        # Delete message
        success = self.queue.delete(message_id)
        self.assertTrue(success)
        self.assertEqual(self.queue.get_queue_size(), 0)

    def test_delete_nonexistent_message(self):
        """Test deleting a non-existent message"""
        success = self.queue.delete(999999)
        self.assertFalse(success)


class TestOfflineQueueTTLAndRetries(unittest.TestCase):
    """Test TTL expiration and retry count functionality"""

    def setUp(self):
        """Create temporary database for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"{self.temp_dir}/test_queue.db"
        self.queue = PythonOfflineQueueManager(self.db_path, max_retries=3)

    def tearDown(self):
        """Cleanup temporary database"""
        Path(self.db_path).unlink(missing_ok=True)
        Path(self.temp_dir).rmdir()

    def test_message_expiration(self):
        """Test message expiration based on TTL"""
        # Enqueue message with 100ms TTL (very short)
        message_id = self.queue.enqueue(
            topic="glec/dtg/device-123/test",
            payload='{"test": true}',
            qos=1,
            ttl_millis=100  # 100ms TTL
        )

        self.assertEqual(self.queue.get_queue_size(), 1)

        # Wait for expiration
        time.sleep(0.2)

        # Cleanup expired
        expired_count = self.queue.cleanup_expired()
        self.assertEqual(expired_count, 1)
        self.assertEqual(self.queue.get_queue_size(), 0)

    def test_increment_retry_count(self):
        """Test incrementing retry count"""
        message_id = self.queue.enqueue(
            topic="glec/dtg/device-123/test",
            payload='{"test": true}',
            qos=1,
            ttl_millis=60 * 60 * 1000
        )

        # Increment retry count 3 times
        for i in range(3):
            success = self.queue.increment_retry_count(message_id)
            self.assertTrue(success)

        # Verify retry count
        messages = self.queue.dequeue_all()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].retry_count, 3)

    def test_cleanup_max_retries(self):
        """Test cleanup of messages exceeding max retries"""
        # Enqueue message
        message_id = self.queue.enqueue(
            topic="glec/dtg/device-123/test",
            payload='{"test": true}',
            qos=1,
            ttl_millis=60 * 60 * 1000
        )

        # Increment retry count to max (3)
        for i in range(3):
            self.queue.increment_retry_count(message_id)

        self.assertEqual(self.queue.get_queue_size(), 1)

        # Cleanup max retries
        cleaned_count = self.queue.cleanup_max_retries()
        self.assertEqual(cleaned_count, 1)
        self.assertEqual(self.queue.get_queue_size(), 0)

    def test_can_retry_method(self):
        """Test QueuedMessage.can_retry() method"""
        message = QueuedMessage(
            id=1,
            topic="test",
            payload="{}",
            qos=1,
            timestamp=int(time.time() * 1000),
            ttl=int(time.time() * 1000) + 60000,
            retry_count=0
        )

        # Should be able to retry with count < 3
        self.assertTrue(message.can_retry(max_retries=3))

        # Increment to max
        message.retry_count = 3
        self.assertFalse(message.can_retry(max_retries=3))


class TestOfflineQueueSizeLimit(unittest.TestCase):
    """Test queue size limit and oldest message removal"""

    def setUp(self):
        """Create temporary database for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"{self.temp_dir}/test_queue.db"
        self.queue = PythonOfflineQueueManager(self.db_path, max_queue_size=10)

    def tearDown(self):
        """Cleanup temporary database"""
        Path(self.db_path).unlink(missing_ok=True)
        Path(self.temp_dir).rmdir()

    def test_queue_size_limit(self):
        """Test that queue respects size limit"""
        # Fill queue to max (10)
        for i in range(10):
            self.queue.enqueue(
                topic="test",
                payload=f'{{"index": {i}}}',
                qos=1,
                ttl_millis=60 * 60 * 1000
            )

        self.assertEqual(self.queue.get_queue_size(), 10)

        # Enqueue 11th message (should remove oldest)
        self.queue.enqueue(
            topic="test",
            payload='{"index": 10}',
            qos=1,
            ttl_millis=60 * 60 * 1000
        )

        # Queue should still be 10
        self.assertEqual(self.queue.get_queue_size(), 10)

        # Oldest message (index 0) should be removed
        messages = self.queue.dequeue_all()
        payloads = [msg.payload for msg in messages]

        self.assertNotIn('{"index": 0}', payloads)
        self.assertIn('{"index": 10}', payloads)

    def test_oldest_message_removal_order(self):
        """Test that oldest message (by timestamp) is removed first"""
        # Enqueue messages with different timestamps
        message_ids = []
        for i in range(10):
            message_id = self.queue.enqueue(
                topic="test",
                payload=f'{{"index": {i}}}',
                qos=1,
                ttl_millis=60 * 60 * 1000
            )
            message_ids.append(message_id)
            time.sleep(0.01)  # Ensure different timestamps

        # Enqueue 11th message
        new_message_id = self.queue.enqueue(
            topic="test",
            payload='{"index": 10}',
            qos=1,
            ttl_millis=60 * 60 * 1000
        )

        # Oldest message (first inserted) should be removed
        messages = self.queue.dequeue_all()
        message_ids_after = [msg.id for msg in messages]

        # First message ID should be missing
        self.assertNotIn(message_ids[0], message_ids_after)
        self.assertIn(new_message_id, message_ids_after)


class TestOfflineQueueQoSHandling(unittest.TestCase):
    """Test QoS level handling"""

    def setUp(self):
        """Create temporary database for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"{self.temp_dir}/test_queue.db"
        self.queue = PythonOfflineQueueManager(self.db_path)

    def tearDown(self):
        """Cleanup temporary database"""
        Path(self.db_path).unlink(missing_ok=True)
        Path(self.temp_dir).rmdir()

    def test_qos_levels(self):
        """Test enqueuing messages with different QoS levels"""
        # QoS 0: Fire and forget
        id0 = self.queue.enqueue("test", "{}", qos=0, ttl_millis=60000)

        # QoS 1: At least once
        id1 = self.queue.enqueue("test", "{}", qos=1, ttl_millis=60000)

        # QoS 2: Exactly once
        id2 = self.queue.enqueue("test", "{}", qos=2, ttl_millis=60000)

        self.assertNotEqual(id0, -1)
        self.assertNotEqual(id1, -1)
        self.assertNotEqual(id2, -1)

        # Verify QoS values stored correctly
        messages = self.queue.dequeue_all()
        qos_values = [msg.qos for msg in messages]

        self.assertIn(0, qos_values)
        self.assertIn(1, qos_values)
        self.assertIn(2, qos_values)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestOfflineQueueBasicOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestOfflineQueueTTLAndRetries))
    suite.addTests(loader.loadTestsFromTestCase(TestOfflineQueueSizeLimit))
    suite.addTests(loader.loadTestsFromTestCase(TestOfflineQueueQoSHandling))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
