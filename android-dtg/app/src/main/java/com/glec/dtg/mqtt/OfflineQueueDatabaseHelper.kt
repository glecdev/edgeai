package com.glec.dtg.mqtt

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log

/**
 * GLEC DTG - Offline Queue Database Helper
 *
 * SQLite database for persistent MQTT message queue.
 *
 * Features:
 * - Persistent storage (survives app restart)
 * - TTL-based expiration
 * - Retry count tracking
 * - FIFO ordering by timestamp
 *
 * Database Schema:
 * ```sql
 * CREATE TABLE mqtt_offline_queue (
 *     id INTEGER PRIMARY KEY AUTOINCREMENT,
 *     topic TEXT NOT NULL,
 *     payload TEXT NOT NULL,
 *     qos INTEGER NOT NULL,
 *     timestamp BIGINT NOT NULL,
 *     ttl BIGINT NOT NULL,
 *     retry_count INTEGER DEFAULT 0
 * );
 * CREATE INDEX idx_timestamp ON mqtt_offline_queue(timestamp);
 * CREATE INDEX idx_ttl ON mqtt_offline_queue(ttl);
 * ```
 *
 * Usage:
 * ```kotlin
 * val dbHelper = OfflineQueueDatabaseHelper(context)
 * val db = dbHelper.writableDatabase
 * ```
 */
class OfflineQueueDatabaseHelper(context: Context) : SQLiteOpenHelper(
    context,
    DATABASE_NAME,
    null,
    DATABASE_VERSION
) {
    companion object {
        private const val TAG = "OfflineQueueDB"

        // Database info
        const val DATABASE_NAME = "mqtt_offline_queue.db"
        const val DATABASE_VERSION = 1

        // Table name
        const val TABLE_NAME = "mqtt_offline_queue"

        // Column names
        const val COLUMN_ID = "id"
        const val COLUMN_TOPIC = "topic"
        const val COLUMN_PAYLOAD = "payload"
        const val COLUMN_QOS = "qos"
        const val COLUMN_TIMESTAMP = "timestamp"
        const val COLUMN_TTL = "ttl"
        const val COLUMN_RETRY_COUNT = "retry_count"

        // SQL statements
        private const val SQL_CREATE_TABLE = """
            CREATE TABLE $TABLE_NAME (
                $COLUMN_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                $COLUMN_TOPIC TEXT NOT NULL,
                $COLUMN_PAYLOAD TEXT NOT NULL,
                $COLUMN_QOS INTEGER NOT NULL,
                $COLUMN_TIMESTAMP BIGINT NOT NULL,
                $COLUMN_TTL BIGINT NOT NULL,
                $COLUMN_RETRY_COUNT INTEGER DEFAULT 0
            )
        """

        private const val SQL_CREATE_INDEX_TIMESTAMP = """
            CREATE INDEX idx_timestamp ON $TABLE_NAME($COLUMN_TIMESTAMP)
        """

        private const val SQL_CREATE_INDEX_TTL = """
            CREATE INDEX idx_ttl ON $TABLE_NAME($COLUMN_TTL)
        """

        private const val SQL_DROP_TABLE = "DROP TABLE IF EXISTS $TABLE_NAME"
    }

    override fun onCreate(db: SQLiteDatabase) {
        Log.i(TAG, "Creating offline queue database")

        try {
            db.execSQL(SQL_CREATE_TABLE)
            db.execSQL(SQL_CREATE_INDEX_TIMESTAMP)
            db.execSQL(SQL_CREATE_INDEX_TTL)

            Log.i(TAG, "✅ Database created successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error creating database", e)
            throw e
        }
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        Log.i(TAG, "Upgrading database from version $oldVersion to $newVersion")

        try {
            // For now, just drop and recreate
            // TODO: Add migration logic for production
            db.execSQL(SQL_DROP_TABLE)
            onCreate(db)

            Log.i(TAG, "✅ Database upgraded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error upgrading database", e)
            throw e
        }
    }

    override fun onDowngrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Same as upgrade for now
        onUpgrade(db, oldVersion, newVersion)
    }

    /**
     * Get database statistics
     */
    fun getStatistics(): DatabaseStatistics {
        val db = readableDatabase
        var totalMessages = 0
        var expiredMessages = 0

        try {
            // Count total messages
            db.rawQuery("SELECT COUNT(*) FROM $TABLE_NAME", null).use { cursor ->
                if (cursor.moveToFirst()) {
                    totalMessages = cursor.getInt(0)
                }
            }

            // Count expired messages
            val currentTime = System.currentTimeMillis()
            db.rawQuery(
                "SELECT COUNT(*) FROM $TABLE_NAME WHERE $COLUMN_TTL < ?",
                arrayOf(currentTime.toString())
            ).use { cursor ->
                if (cursor.moveToFirst()) {
                    expiredMessages = cursor.getInt(0)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error getting statistics", e)
        }

        return DatabaseStatistics(
            totalMessages = totalMessages,
            expiredMessages = expiredMessages,
            activeMessages = totalMessages - expiredMessages
        )
    }

    /**
     * Clear all data (for testing)
     */
    fun clearAll() {
        writableDatabase.execSQL("DELETE FROM $TABLE_NAME")
        Log.i(TAG, "All messages cleared from database")
    }
}

/**
 * Database statistics data class
 */
data class DatabaseStatistics(
    val totalMessages: Int,
    val expiredMessages: Int,
    val activeMessages: Int
)
