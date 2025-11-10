package com.glec.dtg.mqtt

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import kotlinx.coroutines.*

/**
 * GLEC DTG - Offline Queue Manager
 *
 * Manages persistent MQTT message queue using SQLite.
 *
 * Features:
 * - Enqueue/dequeue messages
 * - TTL-based expiration
 * - FIFO ordering
 * - Automatic cleanup
 * - Size limits
 *
 * Usage:
 * ```kotlin
 * val queueManager = OfflineQueueManager(context)
 *
 * // Enqueue message
 * queueManager.enqueue(
 *     topic = "glec/dtg/device-123/inference",
 *     payload = "{...}",
 *     qos = 1,
 *     ttl = 24 * 60 * 60 * 1000  // 24 hours
 * )
 *
 * // Dequeue messages
 * val messages = queueManager.dequeueAll()
 * messages.forEach { message ->
 *     // Publish message
 *     if (success) {
 *         queueManager.delete(message.id)
 *     } else {
 *         queueManager.incrementRetryCount(message.id)
 *     }
 * }
 *
 * // Cleanup expired messages
 * queueManager.cleanupExpired()
 * ```
 */
class OfflineQueueManager(
    private val context: Context,
    private val maxQueueSize: Int = 10_000,
    private val ttlHours: Long = 24,
    private val maxRetries: Int = 3
) {
    companion object {
        private const val TAG = "OfflineQueueManager"
        private const val CLEANUP_INTERVAL_MS = 5 * 60 * 1000L  // 5 minutes
    }

    private val dbHelper = OfflineQueueDatabaseHelper(context)
    private val cleanupScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var cleanupJob: Job? = null

    init {
        // Start periodic cleanup
        startPeriodicCleanup()
    }

    /**
     * Enqueue a message to the offline queue
     *
     * @param topic MQTT topic
     * @param payload Message payload (JSON string)
     * @param qos Quality of Service level
     * @param ttlMillis Time-to-live in milliseconds (default: 24 hours)
     * @return Message ID, or -1 if failed
     */
    fun enqueue(
        topic: String,
        payload: String,
        qos: Int,
        ttlMillis: Long = ttlHours * 60 * 60 * 1000
    ): Long {
        val db = dbHelper.writableDatabase

        try {
            // Check queue size
            val currentSize = getQueueSize()
            if (currentSize >= maxQueueSize) {
                Log.w(TAG, "Queue full ($currentSize/$maxQueueSize), removing oldest message")
                removeOldestMessage()
            }

            // Insert message
            val values = ContentValues().apply {
                put(OfflineQueueDatabaseHelper.COLUMN_TOPIC, topic)
                put(OfflineQueueDatabaseHelper.COLUMN_PAYLOAD, payload)
                put(OfflineQueueDatabaseHelper.COLUMN_QOS, qos)
                put(OfflineQueueDatabaseHelper.COLUMN_TIMESTAMP, System.currentTimeMillis())
                put(OfflineQueueDatabaseHelper.COLUMN_TTL, System.currentTimeMillis() + ttlMillis)
                put(OfflineQueueDatabaseHelper.COLUMN_RETRY_COUNT, 0)
            }

            val id = db.insert(OfflineQueueDatabaseHelper.TABLE_NAME, null, values)

            if (id != -1L) {
                Log.d(TAG, "✅ Message enqueued (id=$id, queue size=${currentSize + 1})")
            } else {
                Log.e(TAG, "❌ Failed to enqueue message")
            }

            return id

        } catch (e: Exception) {
            Log.e(TAG, "Error enqueuing message", e)
            return -1
        }
    }

    /**
     * Dequeue all messages (FIFO order)
     *
     * @return List of queued messages
     */
    fun dequeueAll(): List<QueuedMessage> {
        val db = dbHelper.readableDatabase
        val messages = mutableListOf<QueuedMessage>()

        try {
            val cursor = db.query(
                OfflineQueueDatabaseHelper.TABLE_NAME,
                null,  // All columns
                null,  // No selection (get all)
                null,
                null,
                null,
                "${OfflineQueueDatabaseHelper.COLUMN_TIMESTAMP} ASC"  // FIFO
            )

            cursor.use {
                while (it.moveToNext()) {
                    val message = cursorToMessage(it)
                    messages.add(message)
                }
            }

            Log.d(TAG, "Dequeued ${messages.size} messages")

        } catch (e: Exception) {
            Log.e(TAG, "Error dequeuing messages", e)
        }

        return messages
    }

    /**
     * Dequeue messages up to a limit
     *
     * @param limit Maximum number of messages to dequeue
     * @return List of queued messages
     */
    fun dequeue(limit: Int): List<QueuedMessage> {
        val db = dbHelper.readableDatabase
        val messages = mutableListOf<QueuedMessage>()

        try {
            val cursor = db.query(
                OfflineQueueDatabaseHelper.TABLE_NAME,
                null,
                null,
                null,
                null,
                null,
                "${OfflineQueueDatabaseHelper.COLUMN_TIMESTAMP} ASC",
                limit.toString()
            )

            cursor.use {
                while (it.moveToNext()) {
                    val message = cursorToMessage(it)
                    messages.add(message)
                }
            }

            Log.d(TAG, "Dequeued ${messages.size} messages (limit=$limit)")

        } catch (e: Exception) {
            Log.e(TAG, "Error dequeuing messages", e)
        }

        return messages
    }

    /**
     * Delete a message from the queue
     *
     * @param messageId Message ID
     * @return true if deleted, false otherwise
     */
    fun delete(messageId: Long): Boolean {
        val db = dbHelper.writableDatabase

        try {
            val rowsDeleted = db.delete(
                OfflineQueueDatabaseHelper.TABLE_NAME,
                "${OfflineQueueDatabaseHelper.COLUMN_ID} = ?",
                arrayOf(messageId.toString())
            )

            if (rowsDeleted > 0) {
                Log.d(TAG, "✅ Message deleted (id=$messageId)")
                return true
            } else {
                Log.w(TAG, "⚠️ Message not found (id=$messageId)")
                return false
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error deleting message", e)
            return false
        }
    }

    /**
     * Increment retry count for a message
     *
     * @param messageId Message ID
     * @return true if updated, false otherwise
     */
    fun incrementRetryCount(messageId: Long): Boolean {
        val db = dbHelper.writableDatabase

        try {
            val values = ContentValues().apply {
                // Increment retry_count by 1
                put(
                    OfflineQueueDatabaseHelper.COLUMN_RETRY_COUNT,
                    "(${OfflineQueueDatabaseHelper.COLUMN_RETRY_COUNT} + 1)"
                )
            }

            // Use raw SQL for increment
            db.execSQL(
                "UPDATE ${OfflineQueueDatabaseHelper.TABLE_NAME} " +
                        "SET ${OfflineQueueDatabaseHelper.COLUMN_RETRY_COUNT} = " +
                        "${OfflineQueueDatabaseHelper.COLUMN_RETRY_COUNT} + 1 " +
                        "WHERE ${OfflineQueueDatabaseHelper.COLUMN_ID} = ?",
                arrayOf(messageId.toString())
            )

            Log.d(TAG, "Incremented retry count for message $messageId")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "Error incrementing retry count", e)
            return false
        }
    }

    /**
     * Cleanup expired messages
     *
     * @return Number of messages deleted
     */
    fun cleanupExpired(): Int {
        val db = dbHelper.writableDatabase
        val currentTime = System.currentTimeMillis()

        try {
            val rowsDeleted = db.delete(
                OfflineQueueDatabaseHelper.TABLE_NAME,
                "${OfflineQueueDatabaseHelper.COLUMN_TTL} < ?",
                arrayOf(currentTime.toString())
            )

            if (rowsDeleted > 0) {
                Log.i(TAG, "🗑️ Cleaned up $rowsDeleted expired messages")
            }

            return rowsDeleted

        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up expired messages", e)
            return 0
        }
    }

    /**
     * Cleanup messages that exceeded max retries
     *
     * @return Number of messages deleted
     */
    fun cleanupMaxRetries(): Int {
        val db = dbHelper.writableDatabase

        try {
            val rowsDeleted = db.delete(
                OfflineQueueDatabaseHelper.TABLE_NAME,
                "${OfflineQueueDatabaseHelper.COLUMN_RETRY_COUNT} >= ?",
                arrayOf(maxRetries.toString())
            )

            if (rowsDeleted > 0) {
                Log.i(TAG, "🗑️ Cleaned up $rowsDeleted messages with max retries")
            }

            return rowsDeleted

        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up max retry messages", e)
            return 0
        }
    }

    /**
     * Get queue size
     *
     * @return Number of messages in queue
     */
    fun getQueueSize(): Int {
        val db = dbHelper.readableDatabase

        try {
            db.rawQuery(
                "SELECT COUNT(*) FROM ${OfflineQueueDatabaseHelper.TABLE_NAME}",
                null
            ).use { cursor ->
                if (cursor.moveToFirst()) {
                    return cursor.getInt(0)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting queue size", e)
        }

        return 0
    }

    /**
     * Clear all messages (for testing)
     */
    fun clear() {
        dbHelper.clearAll()
    }

    /**
     * Get database statistics
     */
    fun getStatistics(): DatabaseStatistics {
        return dbHelper.getStatistics()
    }

    /**
     * Release resources
     */
    fun release() {
        cleanupJob?.cancel()
        cleanupScope.cancel()
        dbHelper.close()
    }

    // Private methods

    private fun removeOldestMessage() {
        val db = dbHelper.writableDatabase

        try {
            // Delete oldest message (by timestamp)
            db.execSQL(
                "DELETE FROM ${OfflineQueueDatabaseHelper.TABLE_NAME} " +
                        "WHERE ${OfflineQueueDatabaseHelper.COLUMN_ID} = (" +
                        "SELECT ${OfflineQueueDatabaseHelper.COLUMN_ID} " +
                        "FROM ${OfflineQueueDatabaseHelper.TABLE_NAME} " +
                        "ORDER BY ${OfflineQueueDatabaseHelper.COLUMN_TIMESTAMP} ASC " +
                        "LIMIT 1)"
            )

            Log.d(TAG, "Removed oldest message to make room")

        } catch (e: Exception) {
            Log.e(TAG, "Error removing oldest message", e)
        }
    }

    private fun cursorToMessage(cursor: Cursor): QueuedMessage {
        return QueuedMessage(
            id = cursor.getLong(cursor.getColumnIndexOrThrow(OfflineQueueDatabaseHelper.COLUMN_ID)),
            topic = cursor.getString(cursor.getColumnIndexOrThrow(OfflineQueueDatabaseHelper.COLUMN_TOPIC)),
            payload = cursor.getString(cursor.getColumnIndexOrThrow(OfflineQueueDatabaseHelper.COLUMN_PAYLOAD)),
            qos = cursor.getInt(cursor.getColumnIndexOrThrow(OfflineQueueDatabaseHelper.COLUMN_QOS)),
            timestamp = cursor.getLong(cursor.getColumnIndexOrThrow(OfflineQueueDatabaseHelper.COLUMN_TIMESTAMP)),
            ttl = cursor.getLong(cursor.getColumnIndexOrThrow(OfflineQueueDatabaseHelper.COLUMN_TTL)),
            retryCount = cursor.getInt(cursor.getColumnIndexOrThrow(OfflineQueueDatabaseHelper.COLUMN_RETRY_COUNT))
        )
    }

    private fun startPeriodicCleanup() {
        cleanupJob = cleanupScope.launch {
            while (isActive) {
                delay(CLEANUP_INTERVAL_MS)

                try {
                    val expired = cleanupExpired()
                    val maxRetries = cleanupMaxRetries()

                    if (expired > 0 || maxRetries > 0) {
                        Log.i(TAG, "Periodic cleanup: $expired expired, $maxRetries max retries")
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Error in periodic cleanup", e)
                }
            }
        }

        Log.i(TAG, "✅ Periodic cleanup started (every ${CLEANUP_INTERVAL_MS / 1000}s)")
    }
}
