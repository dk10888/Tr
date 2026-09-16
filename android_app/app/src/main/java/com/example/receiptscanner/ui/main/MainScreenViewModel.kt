package com.example.receiptscanner.ui.main

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.receiptscanner.ml.ReceiptProcessor
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class MainScreenViewModel : ViewModel() {
    private val TAG = "MainScreenViewModel"

    // Lazy-initialized so we don't hold context in the ViewModel constructor
    private var processor: ReceiptProcessor? = null

    sealed class UiState {
        object Idle : UiState()
        object LoadingModels : UiState()
        object Processing : UiState()
        data class Success(val result: ReceiptProcessor.ProcessingResult) : UiState()
        data class Error(val message: String) : UiState()
    }

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _selectedBitmap = MutableStateFlow<Bitmap?>(null)
    val selectedBitmap: StateFlow<Bitmap?> = _selectedBitmap.asStateFlow()

    private val _confidenceThreshold = MutableStateFlow(0.60f)
    val confidenceThreshold: StateFlow<Float> = _confidenceThreshold.asStateFlow()

    fun initProcessor(context: Context) {
        if (processor != null) return
        _uiState.value = UiState.LoadingModels
        viewModelScope.launch {
            try {
                processor = ReceiptProcessor(context.applicationContext)
                _uiState.value = UiState.Idle
                Log.i(TAG, "ReceiptProcessor initialized")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize processor: ${e.message}", e)
                _uiState.value = UiState.Error("Failed to load models: ${e.message}")
            }
        }
    }

    fun onImageSelected(context: Context, uri: Uri) {
        viewModelScope.launch {
            try {
                val bitmap = loadBitmap(context, uri)
                _selectedBitmap.value = bitmap
                _uiState.value = UiState.Idle
            } catch (e: Exception) {
                _uiState.value = UiState.Error("Failed to load image: ${e.message}")
            }
        }
    }

    fun onCapturedBitmap(bitmap: Bitmap) {
        _selectedBitmap.value = bitmap
        _uiState.value = UiState.Idle
    }

    fun processCurrentImage() {
        val bmp = _selectedBitmap.value ?: return
        val proc = processor ?: return

        _uiState.value = UiState.Processing
        viewModelScope.launch {
            try {
                val result = proc.processImage(bmp, _confidenceThreshold.value)
                _uiState.value = UiState.Success(result)
            } catch (e: Exception) {
                Log.e(TAG, "Processing failed: ${e.message}", e)
                _uiState.value = UiState.Error("Processing failed: ${e.message}")
            }
        }
    }

    fun setConfidenceThreshold(threshold: Float) {
        _confidenceThreshold.value = threshold
    }

    fun reset() {
        _selectedBitmap.value = null
        _uiState.value = UiState.Idle
    }

    private fun loadBitmap(context: Context, uri: Uri): Bitmap {
        return context.contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream)
        } ?: throw IllegalArgumentException("Could not decode image from URI")
    }

    override fun onCleared() {
        super.onCleared()
        processor?.close()
    }
}
