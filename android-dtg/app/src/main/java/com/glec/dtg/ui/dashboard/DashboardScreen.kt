package com.glec.dtg.ui.dashboard

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.glec.dtg.models.CANData
import com.glec.dtg.ui.navigation.DashboardDestination
import com.glec.dtg.ui.truck3d.Truck3DViewer
import com.glec.dtg.ui.truck3d.Truck3DViewerController

/**
 * GLEC DTG Main Dashboard Screen
 *
 * Features:
 * - Bottom navigation (3D, Charts, Voice, Diagnostics)
 * - Top app bar with system status
 * - Real-time CAN data integration
 * - Floating action button for quick actions
 *
 * Architecture:
 * - Scaffold with Material 3 design
 * - Navigation controller for tab switching
 * - Shared state for CAN data updates
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DashboardScreen(
    modifier: Modifier = Modifier,
    canData: CANData? = null
) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    // Get current route for bottom nav selection
    val currentRoute = currentDestination?.route ?: DashboardDestination.Truck3D.route

    Scaffold(
        modifier = modifier,
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "GLEC DTG",
                        style = MaterialTheme.typography.titleLarge
                    )
                },
                actions = {
                    // Connection status indicator
                    IconButton(onClick = { /* TODO: Show connection details */ }) {
                        Icon(
                            imageVector = if (canData != null) {
                                Icons.Default.Bluetooth
                            } else {
                                Icons.Default.BluetoothDisabled
                            },
                            contentDescription = "Connection status",
                            tint = if (canData != null) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.error
                            }
                        )
                    }

                    // Notifications
                    IconButton(onClick = { /* TODO: Show notifications */ }) {
                        Badge {
                            Icon(
                                imageVector = Icons.Default.Notifications,
                                contentDescription = "Notifications"
                            )
                        }
                    }

                    // Settings
                    IconButton(onClick = { /* TODO: Navigate to settings */ }) {
                        Icon(
                            imageVector = Icons.Default.Settings,
                            contentDescription = "Settings"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onSurface
                )
            )
        },
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surface,
                tonalElevation = 8.dp
            ) {
                DashboardDestination.items.forEach { destination ->
                    NavigationBarItem(
                        icon = {
                            Icon(
                                imageVector = destination.icon,
                                contentDescription = destination.description
                            )
                        },
                        label = {
                            Text(
                                text = destination.title,
                                style = MaterialTheme.typography.labelMedium
                            )
                        },
                        selected = currentRoute == destination.route,
                        onClick = {
                            navController.navigate(destination.route) {
                                // Pop up to start destination to avoid building large stack
                                popUpTo(DashboardDestination.Truck3D.route) {
                                    saveState = true
                                }
                                // Avoid multiple copies
                                launchSingleTop = true
                                // Restore state when reselecting
                                restoreState = true
                            }
                        },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = MaterialTheme.colorScheme.primary,
                            selectedTextColor = MaterialTheme.colorScheme.primary,
                            unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                            unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
                            indicatorColor = MaterialTheme.colorScheme.primaryContainer
                        )
                    )
                }
            }
        },
        floatingActionButton = {
            // Quick action FAB (context-dependent)
            when (currentRoute) {
                DashboardDestination.Voice.route -> {
                    FloatingActionButton(
                        onClick = { /* TODO: Start voice recording */ },
                        containerColor = MaterialTheme.colorScheme.primary
                    ) {
                        Icon(
                            imageVector = Icons.Default.Mic,
                            contentDescription = "Start voice command"
                        )
                    }
                }
                DashboardDestination.Diagnostics.route -> {
                    FloatingActionButton(
                        onClick = { /* TODO: Clear fault codes */ },
                        containerColor = MaterialTheme.colorScheme.error
                    ) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Clear fault codes"
                        )
                    }
                }
                // No FAB for 3D and Charts screens
                else -> {}
            }
        }
    ) { paddingValues ->
        DashboardNavHost(
            navController = navController,
            canData = canData,
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        )
    }
}

/**
 * Navigation host for dashboard destinations
 */
@Composable
fun DashboardNavHost(
    navController: NavHostController,
    canData: CANData?,
    modifier: Modifier = Modifier
) {
    NavHost(
        navController = navController,
        startDestination = DashboardDestination.Truck3D.route,
        modifier = modifier
    ) {
        // 3D Truck Visualization
        composable(DashboardDestination.Truck3D.route) {
            Truck3DViewerScreen(canData = canData)
        }

        // Real-time Charts
        composable(DashboardDestination.Charts.route) {
            ChartsScreen(canData = canData)
        }

        // Voice AI Interface
        composable(DashboardDestination.Voice.route) {
            VoiceScreen(canData = canData)
        }

        // Diagnostics & Fault Codes
        composable(DashboardDestination.Diagnostics.route) {
            DiagnosticsScreen(canData = canData)
        }
    }
}

/**
 * 3D Truck Viewer Screen
 */
@Composable
fun Truck3DViewerScreen(
    canData: CANData?,
    modifier: Modifier = Modifier
) {
    val viewerController = remember { Truck3DViewerController() }

    // Update viewer with real-time CAN data
    LaunchedEffect(canData) {
        canData?.let {
            viewerController.updateVehicleData(it)
        }
    }

    Truck3DViewer(
        controller = viewerController,
        modifier = modifier.fillMaxSize()
    )
}

/**
 * Real-time Charts Screen (Placeholder)
 */
@Composable
fun ChartsScreen(
    canData: CANData?,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = androidx.compose.ui.Alignment.Center
    ) {
        Column(
            horizontalAlignment = androidx.compose.ui.Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Icon(
                imageVector = Icons.Default.ShowChart,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "실시간 차트",
                style = MaterialTheme.typography.headlineSmall
            )
            Text(
                text = "Speed, RPM, Fuel 라인 차트\n(구현 예정)",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/**
 * Voice AI Interface Screen (Placeholder)
 */
@Composable
fun VoiceScreen(
    canData: CANData?,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = androidx.compose.ui.Alignment.Center
    ) {
        Column(
            horizontalAlignment = androidx.compose.ui.Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Mic,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "음성 AI 대화",
                style = MaterialTheme.typography.headlineSmall
            )
            Text(
                text = "Whisper STT + Qwen LLM + Kokoro TTS\n(구현 예정)",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/**
 * Diagnostics & Fault Codes Screen (Placeholder)
 */
@Composable
fun DiagnosticsScreen(
    canData: CANData?,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = androidx.compose.ui.Alignment.Center
    ) {
        Column(
            horizontalAlignment = androidx.compose.ui.Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.error
            )
            Text(
                text = "고장 진단 코드",
                style = MaterialTheme.typography.headlineSmall
            )
            Text(
                text = "OBD-II/J1939 진단 코드 이력\n(구현 예정)",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
