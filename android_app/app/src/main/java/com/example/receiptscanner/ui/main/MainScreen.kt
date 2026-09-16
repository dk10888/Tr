package com.example.receiptscanner.ui.main

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.receiptscanner.ml.ReceiptProcessor
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// ── Color Palette ─────────────────────────────────────────────────────────────
private val DarkBg      = Color(0xFF0D1117)
private val CardBg      = Color(0xFF161B22)
private val AccentGreen = Color(0xFF2ECC71)
private val AccentBlue  = Color(0xFF3498DB)
private val AccentPurple= Color(0xFF9B59B6)
private val TextPrimary = Color(0xFFE6EDF3)
private val TextSecond  = Color(0xFF8B949E)
private val FoodColor   = Color(0xFF2ECC71)
private val DiscardColor= Color(0xFF6E7681)

@Composable
fun MainScreen(
    onItemClick: ((Any) -> Unit)? = null,
    modifier: Modifier = Modifier,
    viewModel: MainScreenViewModel = viewModel()
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()
    val bitmap  by viewModel.selectedBitmap.collectAsState()
    val threshold by viewModel.confidenceThreshold.collectAsState()

    LaunchedEffect(Unit) { viewModel.initProcessor(context) }

    var showCamera by remember { mutableStateOf(false) }

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.onImageSelected(context, it) }
    }

    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) showCamera = true
        else Toast.makeText(context, "Camera permission required", Toast.LENGTH_SHORT).show()
    }

    Surface(color = DarkBg, modifier = modifier.fillMaxSize()) {
        if (showCamera) {
            CameraScreen(
                onPhotoTaken = { bmp ->
                    showCamera = false
                    viewModel.onCapturedBitmap(bmp)
                },
                onDismiss = { showCamera = false }
            )
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                item { HeaderSection() }

                // ── Image Selector Card ──────────────────────────────
                item {
                    ImageSelectorCard(
                        bitmap = bitmap,
                        onCameraClick = {
                            if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                                == PackageManager.PERMISSION_GRANTED
                            ) showCamera = true
                            else cameraPermission.launch(Manifest.permission.CAMERA)
                        },
                        onGalleryClick = { galleryLauncher.launch("image/*") },
                        onClear = { viewModel.reset() }
                    )
                }

                // ── Settings ─────────────────────────────────────────
                if (bitmap != null) {
                    item {
                        SettingsCard(
                            threshold = threshold,
                            onThresholdChange = { viewModel.setConfidenceThreshold(it) }
                        )
                    }
                    item {
                        Button(
                            onClick = { viewModel.processCurrentImage() },
                            enabled = uiState !is MainScreenViewModel.UiState.Processing,
                            modifier = Modifier.fillMaxWidth().height(56.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = AccentGreen,
                                contentColor = DarkBg,
                            ),
                            shape = RoundedCornerShape(12.dp),
                        ) {
                            Icon(Icons.Default.Search, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("Scan Receipt", fontWeight = FontWeight.Bold, fontSize = 16.sp)
                        }
                    }
                }

                // ── States ────────────────────────────────────────────
                item {
                    when (val state = uiState) {
                        is MainScreenViewModel.UiState.Idle         -> {}
                        is MainScreenViewModel.UiState.LoadingModels -> LoadingCard("Loading AI models…")
                        is MainScreenViewModel.UiState.Processing    -> LoadingCard("Scanning receipt…")
                        is MainScreenViewModel.UiState.Error        -> ErrorCard(state.message)
                        is MainScreenViewModel.UiState.Success      -> ResultsSection(state.result)
                    }
                }
            }
        }
    }
}

// ── Header ────────────────────────────────────────────────────────────────────
@Composable
private fun HeaderSection() {
    Column(modifier = Modifier.fillMaxWidth()) {
        Spacer(Modifier.height(8.dp))
        Text(
            "🍜 Receipt Scanner",
            fontSize = 28.sp,
            fontWeight = FontWeight.ExtraBold,
            color = TextPrimary,
        )
        Text(
            "English & Korean • AI-powered food detection",
            fontSize = 13.sp,
            color = TextSecond,
        )
        Spacer(Modifier.height(8.dp))
        // Language chips
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            LanguageChip("🇺🇸 English", AccentBlue)
            LanguageChip("🇰🇷 한국어", AccentPurple)
        }
    }
}

@Composable
private fun LanguageChip(label: String, color: Color) {
    Surface(
        color = color.copy(alpha = 0.15f),
        shape = RoundedCornerShape(50),
        border = BorderStroke(1.dp, color.copy(alpha = 0.4f)),
    ) {
        Text(
            label,
            color = color,
            fontSize = 12.sp,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
        )
    }
}

// ── Image Selector Card ───────────────────────────────────────────────────────
@Composable
private fun ImageSelectorCard(
    bitmap: Bitmap?,
    onCameraClick: () -> Unit,
    onGalleryClick: () -> Unit,
    onClear: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardBg),
        shape = RoundedCornerShape(16.dp),
    ) {
        if (bitmap != null) {
            Box(modifier = Modifier.fillMaxWidth()) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Receipt image",
                    modifier = Modifier.fillMaxWidth().heightIn(max = 280.dp).clip(RoundedCornerShape(16.dp)),
                    contentScale = ContentScale.Fit,
                )
                IconButton(
                    onClick = onClear,
                    modifier = Modifier.align(Alignment.TopEnd).padding(4.dp)
                        .background(DarkBg.copy(alpha = 0.7f), CircleShape),
                ) {
                    Icon(Icons.Default.Close, contentDescription = "Remove", tint = TextPrimary)
                }
            }
        } else {
            Column(
                modifier = Modifier.fillMaxWidth().padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Icon(
                    Icons.Default.Receipt, contentDescription = null,
                    tint = AccentBlue.copy(alpha = 0.6f),
                    modifier = Modifier.size(64.dp),
                )
                Spacer(Modifier.height(16.dp))
                Text("Select a receipt image", color = TextPrimary, fontWeight = FontWeight.SemiBold)
                Text("Camera or Gallery", color = TextSecond, fontSize = 13.sp)
                Spacer(Modifier.height(20.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedButton(
                        onClick = onCameraClick,
                        border = BorderStroke(1.dp, AccentBlue),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = AccentBlue),
                    ) {
                        Icon(Icons.Default.CameraAlt, contentDescription = null, Modifier.size(16.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("Camera")
                    }
                    OutlinedButton(
                        onClick = onGalleryClick,
                        border = BorderStroke(1.dp, AccentPurple),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = AccentPurple),
                    ) {
                        Icon(Icons.Default.PhotoLibrary, contentDescription = null, Modifier.size(16.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("Gallery")
                    }
                }
            }
        }
    }
}

// ── Settings Card ─────────────────────────────────────────────────────────────
@Composable
private fun SettingsCard(threshold: Float, onThresholdChange: (Float) -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBg),
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Confidence Threshold", color = TextPrimary, fontWeight = FontWeight.Medium)
                Text(
                    "${(threshold * 100).toInt()}%",
                    color = AccentGreen, fontWeight = FontWeight.Bold, fontSize = 16.sp,
                )
            }
            Slider(
                value = threshold,
                onValueChange = onThresholdChange,
                valueRange = 0.40f..0.95f,
                colors = SliderDefaults.colors(
                    thumbColor = AccentGreen,
                    activeTrackColor = AccentGreen,
                    inactiveTrackColor = TextSecond.copy(alpha = 0.3f),
                ),
            )
        }
    }
}

// ── Results Section ───────────────────────────────────────────────────────────
@Composable
private fun ResultsSection(result: ReceiptProcessor.ProcessingResult) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // Stats row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            StatChip("Total Lines", result.totalLines.toString(), AccentBlue, Modifier.weight(1f))
            StatChip("Food Items", result.foodItems.size.toString(), AccentGreen, Modifier.weight(1f))
            StatChip("Time", "${result.elapsedMs}ms", AccentPurple, Modifier.weight(1f))
        }

        if (result.foodItems.isNotEmpty()) {
            SectionHeader("🍜 Food Items (${result.foodItems.size})")
            result.foodItems.forEach { item ->
                ItemCard(item, isFood = true)
            }
        }

        if (result.discardedItems.isNotEmpty()) {
            SectionHeader("🧾 Non-Food (${result.discardedItems.size})")
            result.discardedItems.take(10).forEach { item ->
                ItemCard(item, isFood = false)
            }
            if (result.discardedItems.size > 10) {
                Text(
                    "+ ${result.discardedItems.size - 10} more non-food items",
                    color = TextSecond, fontSize = 12.sp,
                    modifier = Modifier.padding(horizontal = 8.dp),
                )
            }
        }

        if (result.foodItems.isEmpty() && result.discardedItems.isEmpty()) {
            Card(
                colors = CardDefaults.cardColors(containerColor = CardBg),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    "No text detected on this image.\nTry a clearer photo with better lighting.",
                    color = TextSecond, textAlign = TextAlign.Center,
                    modifier = Modifier.padding(24.dp).fillMaxWidth(),
                )
            }
        }
    }
}

@Composable
private fun StatChip(label: String, value: String, color: Color, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.1f)),
        shape = RoundedCornerShape(10.dp),
    ) {
        Column(
            modifier = Modifier.padding(12.dp).fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(value, color = color, fontWeight = FontWeight.ExtraBold, fontSize = 20.sp)
            Text(label, color = TextSecond, fontSize = 11.sp)
        }
    }
}

@Composable
private fun SectionHeader(title: String) {
    Text(
        title, color = TextPrimary,
        fontWeight = FontWeight.Bold, fontSize = 16.sp,
        modifier = Modifier.padding(vertical = 4.dp),
    )
}

@Composable
private fun ItemCard(item: ReceiptProcessor.FoodItem, isFood: Boolean) {
    val langFlag = if (item.language == "ko") "🇰🇷" else "🇺🇸"
    val color = if (isFood) FoodColor else DiscardColor

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CardBg),
        shape = RoundedCornerShape(8.dp),
    ) {
        Row(
            modifier = Modifier.padding(12.dp).fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                Text(langFlag, fontSize = 16.sp)
                Spacer(Modifier.width(8.dp))
                Text(item.text, color = TextPrimary, fontSize = 14.sp)
            }
            Text(
                "${(item.confidence * 100).toInt()}%",
                color = color, fontWeight = FontWeight.Bold, fontSize = 13.sp,
            )
        }
    }
}

// ── Loading / Error Cards ─────────────────────────────────────────────────────
@Composable
private fun LoadingCard(message: String) {
    Card(
        colors = CardDefaults.cardColors(containerColor = CardBg),
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
    ) {
        Row(
            modifier = Modifier.padding(20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            CircularProgressIndicator(color = AccentGreen, modifier = Modifier.size(24.dp))
            Text(message, color = TextPrimary)
        }
    }
}

@Composable
private fun ErrorCard(message: String) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Color(0xFF3D1A1A)),
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Icon(Icons.Default.Error, contentDescription = null, tint = Color(0xFFFF5252))
            Text(message, color = Color(0xFFFFCDD2), fontSize = 13.sp)
        }
    }
}

// ── CameraX Preview ───────────────────────────────────────────────────────────
@Composable
private fun CameraScreen(
    onPhotoTaken: (Bitmap) -> Unit,
    onDismiss: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    var imageCapture: ImageCapture? by remember { mutableStateOf(null) }
    val executor: ExecutorService = remember { Executors.newSingleThreadExecutor() }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                    val capture = ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build()
                    imageCapture = capture
                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture
                        )
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Camera UI overlays
        Column(
            modifier = Modifier.fillMaxSize().padding(24.dp),
            verticalArrangement = Arrangement.SpaceBetween,
        ) {
            // Back button
            IconButton(
                onClick = onDismiss,
                modifier = Modifier.background(Color.Black.copy(0.5f), CircleShape)
            ) {
                Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Color.White)
            }
            // Capture button
            Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                IconButton(
                    onClick = {
                        imageCapture?.let { capture ->
                            capture.takePicture(
                                executor,
                                object : ImageCapture.OnImageCapturedCallback() {
                                    override fun onCaptureSuccess(image: ImageProxy) {
                                        val bitmap = image.toBitmap()
                                        image.close()
                                        onPhotoTaken(bitmap)
                                    }
                                    override fun onError(exception: ImageCaptureException) {
                                        Toast.makeText(context, "Capture failed", Toast.LENGTH_SHORT).show()
                                    }
                                }
                            )
                        }
                    },
                    modifier = Modifier.size(80.dp)
                        .background(AccentGreen, CircleShape)
                        .border(4.dp, Color.White, CircleShape),
                ) {
                    Icon(Icons.Default.Camera, contentDescription = "Capture",
                        tint = DarkBg, modifier = Modifier.size(32.dp))
                }
            }
        }
    }
}
