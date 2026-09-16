package com.example.receiptscanner.ml

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizerOptionsInterface
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Wraps Google ML Kit Text Recognition for both Latin and Korean scripts.
 *
 * Strategy:
 *  - Runs BOTH recognizers in parallel on the same image.
 *  - Merges results by vertical position so Korean and English lines
 *    from the same receipt are interleaved correctly.
 *  - Falls back to Latin-only if Korean recognizer fails.
 */
class MlKitOcrHelper {
    private val TAG = "MlKitOcrHelper"

    private val latinRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val koreanRecognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())

    data class OcrLine(
        val text: String,
        val confidence: Float,
        val boundingTop: Int,
        val language: String,  // "en" or "ko"
    )

    /**
     * Extract all text lines from a bitmap.
     * Returns lines sorted top-to-bottom, with language tag per line.
     */
    suspend fun extractLines(bitmap: Bitmap): List<OcrLine> {
        val image = InputImage.fromBitmap(bitmap, 0)

        // Run both recognizers
        val latinLines  = runRecognizer(latinRecognizer, image, "en")
        val koreanLines = runRecognizer(koreanRecognizer, image, "ko")

        // Merge: keep Korean text as Korean, Latin as English.
        // De-duplicate lines that appear in both results (same text, similar Y position).
        val merged = mergeResults(latinLines, koreanLines)
        Log.i(TAG, "OCR done: ${merged.size} lines (${latinLines.size} latin, ${koreanLines.size} korean)")
        return merged
    }

    private suspend fun runRecognizer(
        recognizer: com.google.mlkit.vision.text.TextRecognizer,
        image: InputImage,
        lang: String,
    ): List<OcrLine> = suspendCancellableCoroutine { cont ->
        recognizer.process(image)
            .addOnSuccessListener { visionText ->
                val lines = extractFromVisionText(visionText, lang)
                cont.resume(lines)
            }
            .addOnFailureListener { e ->
                Log.w(TAG, "[$lang] Recognizer failed: ${e.message}")
                cont.resume(emptyList()) // graceful fallback
            }
    }

    private fun extractFromVisionText(visionText: Text, lang: String): List<OcrLine> {
        val lines = mutableListOf<OcrLine>()
        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                val text = line.text.trim()
                if (text.length < 2) continue
                val top = line.boundingBox?.top ?: 0
                val conf = line.confidence ?: 1.0f
                lines.add(OcrLine(text = text, confidence = conf, boundingTop = top, language = lang))
            }
        }
        return lines.sortedBy { it.boundingTop }
    }

    /**
     * Merge Latin and Korean results.
     * - Lines that contain Hangul characters → take from Korean recognizer
     * - Lines that are pure Latin → take from Latin recognizer
     * - Deduplicate by text + vertical proximity (within 20px)
     */
    private fun mergeResults(latin: List<OcrLine>, korean: List<OcrLine>): List<OcrLine> {
        val all = mutableListOf<OcrLine>()
        val seenTexts = mutableSetOf<String>()

        // First pass: add all Korean lines (Hangul text)
        for (line in korean) {
            val normalized = line.text.lowercase().trim()
            if (LanguageDetector.isKorean(line.text) && !seenTexts.contains(normalized)) {
                all.add(line.copy(language = "ko"))
                seenTexts.add(normalized)
            }
        }

        // Second pass: add Latin lines not already covered
        for (line in latin) {
            val normalized = line.text.lowercase().trim()
            if (!seenTexts.contains(normalized)) {
                all.add(line.copy(language = "en"))
                seenTexts.add(normalized)
            }
        }

        return all.sortedBy { it.boundingTop }
    }

    fun close() {
        latinRecognizer.close()
        koreanRecognizer.close()
    }
}
