package com.glec.dtg

import android.content.Context
import android.widget.Toast
import timber.log.Timber
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Simple Voice Test
 *
 * Minimal test without complex dependencies
 */
object SimpleVoiceTest {

    suspend fun testVoice(context: Context): String = withContext(Dispatchers.IO) {
        try {
            Timber.d("Starting simple voice test...")

            // Simulate voice query processing
            val queries = listOf(
                "적재 중량이 얼마인가요?",
                "타이어 공기압 확인해줘",
                "엔진 온도 어때?",
                "연비가 어떻게 돼?",
                "차량 상태 알려줘"
            )

            val query = queries.random()

            // Simulate processing time
            Thread.sleep(300)

            // Generate mock response
            val responses = mapOf(
                "적재" to "현재 적재 중량은 5200킬로그램입니다. 안전한 적재 상태입니다.",
                "타이어" to "타이어 공기압은 220kPa입니다. 정상 범위입니다.",
                "엔진" to "엔진 냉각수 온도는 85도입니다. 정상 범위입니다.",
                "연비" to "현재 연비는 6.8km/L입니다. 양호한 연비입니다.",
                "상태" to "차량 상태는 전반적으로 양호합니다. 적재 중량 5200kg, 연비 6.8km/L로 운행 중입니다."
            )

            val response = responses.entries.firstOrNull { (key, _) ->
                query.contains(key)
            }?.value ?: "질문을 이해했습니다. 차량 데이터를 기반으로 답변드리겠습니다."

            Timber.i("Voice test completed: $query -> $response")

            """
            Query: $query

            Response: $response

            Duration: ~300ms
            Status: SUCCESS ✓
            """.trimIndent()

        } catch (e: Exception) {
            Timber.e(e, "Voice test failed")
            "Error: ${e.message}"
        }
    }
}
