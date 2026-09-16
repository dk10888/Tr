package com.example.receiptscanner.ml

/**
 * Detects whether a string is predominantly Korean (Hangul) or Latin.
 * Used to route each OCR'd line to the correct BERT classifier.
 */
object LanguageDetector {

    // Unicode range for Hangul syllables (가–힣) + Hangul Jamo + Compatibility Jamo
    private val HANGUL_RANGE = Regex("[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")

    /**
     * Returns true if the string contains enough Korean characters
     * to be classified as a Korean receipt line.
     *
     * Strategy: if ≥ 30% of alphabetic/CJK chars are Hangul → Korean.
     */
    fun isKorean(text: String): Boolean {
        if (text.isBlank()) return false
        val hangulCount = HANGUL_RANGE.findAll(text).count()
        val totalMeaningful = text.count { it.isLetter() }
        if (totalMeaningful == 0) return false
        return hangulCount.toDouble() / totalMeaningful >= 0.30
    }

    /** Convenience: returns "ko" or "en" */
    fun detectLanguage(text: String): String = if (isKorean(text)) "ko" else "en"
}
