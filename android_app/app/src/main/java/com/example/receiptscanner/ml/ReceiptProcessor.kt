package com.example.receiptscanner.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Central receipt processing pipeline.
 * Orchestrates: OCR → Language Detection → BERT Classification
 *
 * English lines → English BERT (bert_tiny / bert_mini)
 * Korean lines  → Korean ELECTRA-small
 */
class ReceiptProcessor(context: Context) : AutoCloseable {
    private val TAG = "ReceiptProcessor"

    private val ocr = MlKitOcrHelper()

    // English BERT (bert_tiny — lighter, faster)
    private val englishClassifier = OnnxBertClassifier(
        context         = context,
        modelFileName   = "english_food_bert.onnx",
        vocabFileName   = "english_vocab.txt",
        clsTokenId      = 101,   // standard BERT [CLS]
        sepTokenId      = 102,
        padTokenId      = 0,
        unkTokenId      = 100,
        maxLen          = 64,
        useGpu          = true,
    )

    // Korean KoELECTRA-small
    private val koreanClassifier = OnnxBertClassifier(
        context         = context,
        modelFileName   = "korean_food_classifier_quant.onnx",
        vocabFileName   = "korean_vocab.txt",
        clsTokenId      = 2,    // KoELECTRA [CLS]
        sepTokenId      = 3,
        padTokenId      = 0,
        unkTokenId      = 1,
        maxLen          = 64,
        useGpu          = true,
    )

    data class ProcessingResult(
        val totalLines: Int,
        val foodItems: List<FoodItem>,
        val discardedItems: List<FoodItem>,
        val elapsedMs: Long,
    )

    data class FoodItem(
        val text: String,
        val confidence: Float,
        val language: String,
        val isFood: Boolean,
    )

    /**
     * Full pipeline: OCR → classify → return food items.
     * Call on IO dispatcher.
     */
    suspend fun processImage(
        bitmap: Bitmap,
        confidenceThreshold: Float = 0.60f,
    ): ProcessingResult = withContext(Dispatchers.IO) {
        val start = System.currentTimeMillis()

        // Step 1: OCR
        val lines = ocr.extractLines(bitmap)
        Log.i(TAG, "OCR returned ${lines.size} lines")

        if (lines.isEmpty()) {
            return@withContext ProcessingResult(0, emptyList(), emptyList(), System.currentTimeMillis() - start)
        }

        // Step 2: Partition by language
        val englishLines = lines.filter { it.language == "en" }
        val koreanLines  = lines.filter { it.language == "ko" }
        Log.i(TAG, "English: ${englishLines.size}, Korean: ${koreanLines.size}")

        // Step 3: Classify in batch per language
        val englishResults = if (englishLines.isNotEmpty())
            englishClassifier.classifyBatch(englishLines.map { it.text }, confidenceThreshold)
        else emptyList()

        val koreanResults = if (koreanLines.isNotEmpty())
            koreanClassifier.classifyBatch(koreanLines.map { it.text }, confidenceThreshold)
        else emptyList()

        // Step 4: Build output preserving original order
        val food      = mutableListOf<FoodItem>()
        val discarded = mutableListOf<FoodItem>()

        englishResults.forEach { r ->
            val item = FoodItem(r.text, r.confidence, "en", r.isFood)
            if (r.isFood) food.add(item) else discarded.add(item)
        }
        koreanResults.forEach { r ->
            val item = FoodItem(r.text, r.confidence, "ko", r.isFood)
            if (r.isFood) food.add(item) else discarded.add(item)
        }

        val elapsed = System.currentTimeMillis() - start
        Log.i(TAG, "Done in ${elapsed}ms — ${food.size} food, ${discarded.size} discarded")

        ProcessingResult(
            totalLines    = lines.size,
            foodItems     = food,
            discardedItems = discarded,
            elapsedMs     = elapsed,
        )
    }

    override fun close() {
        ocr.close()
        englishClassifier.close()
        koreanClassifier.close()
    }
}
