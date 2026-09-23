package com.example.receiptscanner.ml

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/**
 * Wraps Google ML Kit Text Recognition (Latin + Korean).
 *
 * Key improvement — word-level grouping with gap detection:
 *   • Descends to Text.Element (individual word boxes) rather than Text.Line.
 *   • Words that are on the same row (Y-overlap) and close horizontally
 *     (gap < COLUMN_GAP_PX) are merged into a single candidate string.
 *   • A **large** horizontal gap signals the price column on the right side
 *     of the receipt → the current candidate is emitted and a new one starts.
 *   • This fixes the two failure modes we observed:
 *       (a) Multi-word names fragmented across separate ML-Kit lines → now merged.
 *       (b) "Pan Fried Gyoza Pork  15,000원" all on one row → split at the gap.
 */
class MlKitOcrHelper {

    private val TAG = "MlKitOcrHelper"

    // ── Tuneable thresholds ──────────────────────────────────────────────────
    /** Two word-boxes are on the "same row" if their vertical-overlap ratio ≥ this. */
    private val ROW_OVERLAP_RATIO = 0.4f

    /**
     * Absolute horizontal gap between two word-boxes that signals a price
     * column split.  Receipt layouts vary, so we also check a fraction of the
     * image width below.
     */
    private val COLUMN_GAP_ABS_PX = 80

    /** If the gap is also ≥ this fraction of image width it is a column gap. */
    private val COLUMN_GAP_REL = 0.18f

    /** Words whose horizontal gap is < this are always merged (continuation). */
    private val WORD_JOIN_GAP_PX = 40
    // ────────────────────────────────────────────────────────────────────────

    private val latinRecognizer  = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val koreanRecognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())

    /**
     * One logical line (after word-grouping) ready for the classifier.
     *
     * @param text        The merged text of consecutive close-by words.
     * @param confidence  Average element confidence.
     * @param boundingTop Top-Y of the first word in the group (for ordering).
     * @param language    "en" or "ko"
     */
    data class OcrLine(
        val text: String,
        val confidence: Float,
        val boundingTop: Int,
        val language: String,
    )

    // ── Word-box extracted from a Text.Element ────────────────────────────
    private data class WordBox(
        val text: String,
        val confidence: Float,
        val box: Rect,
        val language: String,
    )

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Extract all logical item candidates from a bitmap.
     * Returns candidates sorted top-to-bottom, language-tagged.
     */
    suspend fun extractLines(bitmap: Bitmap): List<OcrLine> {
        val image = InputImage.fromBitmap(bitmap, 0)
        val imgW   = bitmap.width

        val latinWords  = runRecognizer(latinRecognizer,  image, "en")
        val koreanWords = runRecognizer(koreanRecognizer, image, "ko")

        val merged = mergeWordLists(latinWords, koreanWords)
        val groups = groupWordsIntoLines(merged, imgW)

        Log.i(TAG, "OCR done: ${groups.size} candidate lines from ${merged.size} words")
        return groups
    }

    // ── Recognizer runner ─────────────────────────────────────────────────

    private suspend fun runRecognizer(
        recognizer: com.google.mlkit.vision.text.TextRecognizer,
        image: InputImage,
        lang: String,
    ): List<WordBox> = suspendCancellableCoroutine { cont ->
        recognizer.process(image)
            .addOnSuccessListener { visionText ->
                cont.resume(extractWords(visionText, lang))
            }
            .addOnFailureListener { e ->
                Log.w(TAG, "[$lang] Recognizer failed: ${e.message}")
                cont.resume(emptyList())
            }
    }

    // ── Extract individual word boxes from ML Kit result ──────────────────

    private fun extractWords(visionText: Text, lang: String): List<WordBox> {
        val words = mutableListOf<WordBox>()
        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                for (element in line.elements) {
                    val t   = element.text.trim()
                    if (t.isEmpty()) continue
                    val box = element.boundingBox ?: continue
                    val conf = element.confidence ?: 1.0f
                    words.add(WordBox(t, conf, box, lang))
                }
            }
        }
        return words.sortedWith(compareBy({ it.box.top }, { it.box.left }))
    }

    // ── De-duplicate across Latin & Korean word lists ─────────────────────

    /**
     * Prefer Korean-recognizer words for Hangul text; keep Latin words for
     * everything else.  De-duplicate by checking bounding-box overlap.
     */
    private fun mergeWordLists(latin: List<WordBox>, korean: List<WordBox>): List<WordBox> {
        val result   = mutableListOf<WordBox>()
        val usedBoxes = mutableListOf<Rect>()

        // Add Korean words first (Hangul-containing)
        for (w in korean) {
            if (LanguageDetector.isKorean(w.text)) {
                result.add(w.copy(language = "ko"))
                usedBoxes.add(w.box)
            }
        }

        // Add Latin words whose box doesn't heavily overlap an already-added box
        for (w in latin) {
            val overlap = usedBoxes.any { used ->
                val ix = maxOf(w.box.left, used.left)
                val iy = maxOf(w.box.top, used.top)
                val ax = minOf(w.box.right, used.right)
                val ay = minOf(w.box.bottom, used.bottom)
                val inter = maxOf(0, ax - ix).toLong() * maxOf(0, ay - iy).toLong()
                val wArea = w.box.width().toLong() * w.box.height().toLong()
                wArea > 0 && inter.toFloat() / wArea > 0.5f
            }
            if (!overlap) {
                result.add(w.copy(language = "en"))
                usedBoxes.add(w.box)
            }
        }

        return result.sortedWith(compareBy({ it.box.top }, { it.box.left }))
    }

    // ── Core grouping logic ───────────────────────────────────────────────

    /**
     * Groups word-boxes into logical candidate lines using two rules:
     *
     *  1. **Same-row check**: two words belong to the same row if the vertical
     *     overlap of their bounding boxes is ≥ ROW_OVERLAP_RATIO of the
     *     shorter word's height.
     *
     *  2. **Gap check (applied only within the same row)**:
     *       • gap < WORD_JOIN_GAP_PX           → always merge (same token)
     *       • gap ≥ COLUMN_GAP_ABS_PX  AND
     *         gap ≥ imgW * COLUMN_GAP_REL       → column split (emit current group)
     *       • in between                         → merge (intra-word spacing)
     */
    private fun groupWordsIntoLines(words: List<WordBox>, imgW: Int): List<OcrLine> {
        if (words.isEmpty()) return emptyList()

        val colGapThreshold = maxOf(COLUMN_GAP_ABS_PX, (imgW * COLUMN_GAP_REL).toInt())

        val result = mutableListOf<OcrLine>()

        // Accumulator for the current group
        var groupWords   = mutableListOf(words[0])
        var groupLang    = words[0].language

        for (i in 1 until words.size) {
            val prev = groupWords.last()
            val curr = words[i]

            val sameRow = isSameRow(prev.box, curr.box)
            val hGap    = curr.box.left - prev.box.right   // can be negative if overlapping

            val shouldSplit = when {
                !sameRow                             -> true   // clearly different row
                hGap >= colGapThreshold              -> true   // price-column gap
                else                                 -> false  // merge
            }

            if (shouldSplit) {
                // Emit the accumulated group
                result.add(buildOcrLine(groupWords, groupLang))
                groupWords = mutableListOf(curr)
                groupLang  = curr.language
            } else {
                groupWords.add(curr)
                // If mixed scripts, the Korean language wins (more informative for classifier)
                if (LanguageDetector.isKorean(curr.text)) groupLang = "ko"
            }
        }
        // Emit final group
        result.add(buildOcrLine(groupWords, groupLang))

        return result
            .filter { it.text.length >= 2 }
            .sortedBy { it.boundingTop }
    }

    /** True when two bounding boxes share enough vertical overlap to be on the same row. */
    private fun isSameRow(a: Rect, b: Rect): Boolean {
        val overlapTop    = maxOf(a.top, b.top)
        val overlapBottom = minOf(a.bottom, b.bottom)
        val overlap       = maxOf(0, overlapBottom - overlapTop)
        val shorter       = minOf(a.height(), b.height())
        return shorter > 0 && overlap.toFloat() / shorter >= ROW_OVERLAP_RATIO
    }

    /** Converts a list of WordBoxes into a single OcrLine. */
    private fun buildOcrLine(words: List<WordBox>, lang: String): OcrLine {
        val text = words.joinToString(" ") { it.text }
        val avgConf = words.map { it.confidence }.average().toFloat()
        val top = words.minOf { it.box.top }
        return OcrLine(text = text, confidence = avgConf, boundingTop = top, language = lang)
    }

    fun close() {
        latinRecognizer.close()
        koreanRecognizer.close()
    }
}
