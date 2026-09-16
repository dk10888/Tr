package com.example.receiptscanner.ml

/**
 * Pure Kotlin WordPiece tokenizer for BERT/ELECTRA models.
 *
 * Handles both English (Latin) and Korean (Hangul) tokens via
 * the same WordPiece algorithm — works with any HuggingFace vocab.txt.
 *
 * Usage:
 *   val tokenizer = WordPieceTokenizer(vocab, clsId, sepId, padId, unkId)
 *   val (inputIds, attentionMask, tokenTypeIds) = tokenizer.encode("김치찌개", maxLen = 64)
 */
class WordPieceTokenizer(
    private val vocab: Map<String, Int>,
    private val clsTokenId: Int,
    private val sepTokenId: Int,
    private val padTokenId: Int,
    private val unkTokenId: Int,
) {
    data class Encoding(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds: LongArray,
    )

    /**
     * Encode a single text into fixed-length token arrays.
     * Returns arrays of size [maxLen] padded with padTokenId.
     */
    fun encode(text: String, maxLen: Int = 64): Encoding {
        val tokens = tokenize(text)
        val ids = mutableListOf<Int>().apply {
            add(clsTokenId)
            // Reserve 1 slot for [SEP]
            addAll(tokens.take(maxLen - 2).map { vocab[it] ?: unkTokenId })
            add(sepTokenId)
        }

        val inputIds      = LongArray(maxLen) { padTokenId.toLong() }
        val attentionMask = LongArray(maxLen) { 0L }
        val tokenTypeIds  = LongArray(maxLen) { 0L }

        ids.forEachIndexed { i, id ->
            if (i < maxLen) {
                inputIds[i] = id.toLong()
                attentionMask[i] = 1L
            }
        }
        return Encoding(inputIds, attentionMask, tokenTypeIds)
    }

    // ── WordPiece tokenization ───────────────────────────────────────

    private fun tokenize(text: String): List<String> {
        val cleanText = text.lowercase().trim()
        val result = mutableListOf<String>()
        // Split on whitespace, then WordPiece each word
        for (word in splitOnWhitespace(cleanText)) {
            result.addAll(wordpieceWord(word))
        }
        return result
    }

    /**
     * Split text on whitespace. For Korean text, also split individual
     * Hangul syllables into their own "words" for WordPiece processing.
     */
    private fun splitOnWhitespace(text: String): List<String> {
        return text.split(Regex("\\s+")).filter { it.isNotEmpty() }
    }

    private fun wordpieceWord(word: String): List<String> {
        // If the whole word is in vocab, return it directly
        if (vocab.containsKey(word)) return listOf(word)

        val subTokens = mutableListOf<String>()
        var start = 0
        var isBad = false

        while (start < word.length) {
            var end = word.length
            var curSubStr: String? = null

            while (start < end) {
                val substr = word.substring(start, end)
                val candidate = if (start > 0) "##$substr" else substr
                if (vocab.containsKey(candidate)) {
                    curSubStr = candidate
                    break
                }
                end--
            }

            if (curSubStr == null) {
                isBad = true
                break
            }
            subTokens.add(curSubStr)
            start = end
        }

        return if (isBad) listOf("[UNK]") else subTokens
    }

    companion object {
        /**
         * Load vocab.txt from assets and build the vocab map.
         * vocab.txt format: one token per line, index = line number.
         */
        fun loadVocab(vocabText: String): Map<String, Int> {
            val vocab = mutableMapOf<String, Int>()
            vocabText.lines().forEachIndexed { index, line ->
                val token = line.trim()
                if (token.isNotEmpty()) vocab[token] = index
            }
            return vocab
        }
    }
}
