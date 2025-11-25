package com.glec.dtg.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.ui.graphics.vector.ImageVector

/**
 * GLEC DTG Dashboard Navigation Destinations
 *
 * Main navigation tabs for truck telematics dashboard
 */
sealed class DashboardDestination(
    val route: String,
    val title: String,
    val icon: ImageVector,
    val description: String
) {
    /**
     * 3D Truck Visualization
     * - Real-time 3D truck model
     * - Fault highlighting (red alerts)
     * - Camera controls
     */
    data object Truck3D : DashboardDestination(
        route = "truck3d",
        title = "3D 뷰",
        icon = Icons.Default.DirectionsCar,
        description = "3D 트럭 시각화 및 고장 표시"
    )

    /**
     * Real-time Data Charts
     * - Speed, RPM, fuel line charts
     * - Historical data visualization
     * - Multi-metric view
     */
    data object Charts : DashboardDestination(
        route = "charts",
        title = "차트",
        icon = Icons.Default.ShowChart,
        description = "실시간 데이터 차트"
    )

    /**
     * Voice AI Interface
     * - Wake word detection
     * - Speech-to-text
     * - LLM conversation
     * - Text-to-speech
     */
    data object Voice : DashboardDestination(
        route = "voice",
        title = "음성 AI",
        icon = Icons.Default.Mic,
        description = "음성 대화형 인터페이스"
    )

    /**
     * Diagnostics & Fault Codes
     * - OBD-II/J1939 codes
     * - Fault history
     * - Recommended actions
     */
    data object Diagnostics : DashboardDestination(
        route = "diagnostics",
        title = "진단",
        icon = Icons.Default.Warning,
        description = "고장 진단 코드 및 이력"
    )

    companion object {
        val items = listOf(Truck3D, Charts, Voice, Diagnostics)
    }
}
