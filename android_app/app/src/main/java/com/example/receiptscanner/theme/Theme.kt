package com.example.receiptscanner.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary         = Color(0xFF2ECC71),
    onPrimary       = Color(0xFF0D1117),
    secondary       = Color(0xFF3498DB),
    onSecondary     = Color(0xFFFFFFFF),
    tertiary        = Color(0xFF9B59B6),
    background      = Color(0xFF0D1117),
    onBackground    = Color(0xFFE6EDF3),
    surface         = Color(0xFF161B22),
    onSurface       = Color(0xFFE6EDF3),
    surfaceVariant  = Color(0xFF21262D),
    onSurfaceVariant= Color(0xFF8B949E),
    error           = Color(0xFFFF5252),
    onError         = Color(0xFFFFFFFF),
)

@Composable
fun ReceiptScannerTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        typography  = Typography,
        content     = content,
    )
}
