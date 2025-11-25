package com.glec.dtg.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/**
 * GLEC DTG Dark Theme
 *
 * Optimized for:
 * - Night driving visibility
 * - Reduced eye strain
 * - High contrast for critical information
 * - Professional commercial vehicle aesthetic
 */
private val DarkColorScheme = darkColorScheme(
    // Primary colors (Cyan accent)
    primary = CyanPrimary,
    onPrimary = Color.Black,
    primaryContainer = CyanPrimaryDark,
    onPrimaryContainer = Color.White,

    // Secondary colors
    secondary = StatusWarning,
    onSecondary = Color.Black,
    secondaryContainer = Color(0xFF804A00),
    onSecondaryContainer = Color.White,

    // Tertiary colors (for accents)
    tertiary = StatusNormal,
    onTertiary = Color.Black,
    tertiaryContainer = Color(0xFF1A6B2E),
    onTertiaryContainer = Color.White,

    // Error colors
    error = StatusDanger,
    onError = Color.White,
    errorContainer = Color(0xFF8C0000),
    onErrorContainer = Color.White,

    // Background colors
    background = BackgroundDark,
    onBackground = TextPrimary,
    surface = SurfaceDark,
    onSurface = TextPrimary,
    surfaceVariant = SurfaceVariant,
    onSurfaceVariant = TextSecondary,

    // Outline colors
    outline = DividerColor,
    outlineVariant = Color(0xFF2A2A3E),

    // Inverse colors
    inverseSurface = Color(0xFFE0E0E0),
    inverseOnSurface = Color(0xFF1A1A2E),
    inversePrimary = Color(0xFF006B8C)
)

@Composable
fun DTGTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    // GLEC DTG always uses dark theme for optimal night driving visibility
    val colorScheme = DarkColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
