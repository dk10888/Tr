package com.example.receiptscanner.ml

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.LongBuffer

/**
 * FastTextOnnxClassifier
 * ======================
 * Runs the FastText v5.0 Android ONNX model (8.8 MB) using ONNX Runtime.
 *
 * FastText inference = character n-gram hashing → EmbeddingBag lookup → sigmoid
 *
 * The ONNX model expects two inputs:
 *   - token_ids  : LongTensor[num_tokens] — word vocab IDs + n-gram bucket IDs
 *   - offsets    : LongTensor[batch_size] — start index of each sample (always [0] for single inference)
 *
 * Model hyperparams (baked in, must match training):
 *   minn=2, maxn=5, wordNgrams=2, bucket=50000, vocab_size from fasttext_vocab.json
 *
 * Assets required (copy to app/src/main/assets/):
 *   - fasttext_v50_android.onnx   (8.8 MB)
 *   - fasttext_vocab.json          (~500 KB)
 *
 * Usage:
 *   val classifier = FastTextOnnxClassifier(context)
 *   val result = classifier.classify("갈비세트")
 *   // result.label = "food" | "not_food", result.confidence = 0.0..1.0
 */
class FastTextOnnxClassifier(private val context: Context) {

    data class ClassificationResult(
        val label: String,
        val confidence: Float,
        val timeMs: Long
    )

    private val TAG = "FastTextOnnxClassifier"

    // Model hyperparams — must match build_v50_android.py training settings
    private val MINN = 2
    private val MAXN = 5
    private val BUCKET = 50000
    private val WORD_NGRAMS = 2  // bigram word n-grams
    private val MODEL_ASSET = "fasttext_v50_android.onnx"
    private val VOCAB_ASSET  = "fasttext_vocab.json"

    // Loaded at init
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private lateinit var wordIndex: Map<String, Int>   // word → embedding row index
    private var vocabSize: Int = 0                      // number of word rows in embedding
    private var labels: List<String> = emptyList()

    init {
        loadModel()
        loadVocab()
    }

    // -------------------------------------------------------------------------
    // Model & Vocab Loading
    // -------------------------------------------------------------------------

    private fun loadModel() {
        ortEnv = OrtEnvironment.getEnvironment()
        val modelBytes = context.assets.open(MODEL_ASSET).readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        ortSession = ortEnv.createSession(modelBytes, opts)
        Log.i(TAG, "ONNX session created: $MODEL_ASSET")
    }

    private fun loadVocab() {
        val raw = context.assets.open(VOCAB_ASSET)
            .bufferedReader().use { it.readText() }
        val json = JSONObject(raw)
        val words = json.getJSONArray("words")
        vocabSize = words.length()
        val map = HashMap<String, Int>(vocabSize * 2)
        for (i in 0 until words.length()) {
            map[words.getString(i)] = i
        }
        wordIndex = map

        val labelsArr = json.getJSONArray("labels")
        labels = (0 until labelsArr.length()).map {
            labelsArr.getString(it).removePrefix("__label__")
        }

        Log.i(TAG, "Vocab loaded: $vocabSize words, ${labels.size} labels")
    }

    // -------------------------------------------------------------------------
    // FastText Tokenisation
    // Replicates FastText C++ tokenizer:
    //   1. Split text into words (whitespace)
    //   2. For each word: look up word ID + generate n-gram bucket IDs
    //   3. If wordNgrams > 1: add bigram bucket IDs across adjacent words
    // -------------------------------------------------------------------------

    /**
     * FNV-1a hash — exact match to FastText C++ implementation.
     *
     * FastText C++ uses int8_t (SIGNED byte) cast:
     *   h = h ^ (uint32_t)(int8_t)(char)byte;
     *   h = h * 16777619;
     *
     * Kotlin Byte is already signed (-128..127), so b.toLong() sign-extends
     * correctly for bytes > 127 (e.g. 0xC0 → -64 → 0xFFFFFFFFFFFFFFC0).
     * We must mask to 32-bit AFTER each XOR and multiply to replicate uint32_t.
     */
    private fun fnvHash(s: String): Int {
        var h = 2166136261L                              // uint32 start value, fits in Long
        for (b in s.toByteArray(Charsets.UTF_8)) {
            h = (h xor b.toLong()) and 0xFFFFFFFFL       // XOR signed byte, mask to 32-bit
            h = (h * 16777619L) and 0xFFFFFFFFL          // multiply, mask to uint32 wraparound
        }
        val idx = h % BUCKET.toLong()
        return (vocabSize.toLong() + idx).toInt()
    }

    /**
     * Generate all character n-grams for a single word.
     * FastText wraps the word with '<' and '>' markers.
     */
    private fun charNgrams(word: String): List<Int> {
        val wrapped = "<$word>"
        val ids = mutableListOf<Int>()
        val bytes = wrapped.toByteArray(Charsets.UTF_8)
        val n = bytes.size

        for (i in bytes.indices) {
            val sb = StringBuilder()
            var j = i
            var ngramLen = 0
            while (j < n && ngramLen < MAXN) {
                sb.append(bytes[j].toInt().and(0xFF).toChar())
                j++
                ngramLen++
                if (ngramLen >= MINN) {
                    ids.add(fnvHash(sb.toString()))
                }
            }
        }
        return ids
    }

    /**
     * Converts a receipt line string into a list of embedding row indices.
     * These are fed directly to the ONNX EmbeddingBag input.
     */
    private fun tokenise(text: String): LongArray {
        val tokens = mutableListOf<Long>()
        val words = text.lowercase().trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        val wordIds = mutableListOf<Int>()

        for (word in words) {
            // Word-level lookup
            val wordId = wordIndex[word]
            if (wordId != null) {
                tokens.add(wordId.toLong())
                wordIds.add(wordId)
            } else {
                wordIds.add(-1)
            }
            // Character n-gram bucket IDs
            for (bucketId in charNgrams(word)) {
                tokens.add(bucketId.toLong())
            }
        }

        // Word bigrams (wordNgrams=2) — hash of consecutive word pairs
        if (WORD_NGRAMS >= 2 && words.size >= 2) {
            for (i in 0 until words.size - 1) {
                val bigram = "${words[i]} ${words[i+1]}"
                tokens.add(fnvHash(bigram).toLong())
            }
        }

        return if (tokens.isEmpty()) longArrayOf(0L) else tokens.toLongArray()
    }

    // -------------------------------------------------------------------------
    // Inference
    // -------------------------------------------------------------------------

    fun classify(text: String): ClassificationResult {
        val t0 = System.currentTimeMillis()

        val tokenIds = tokenise(text)
        val offsets  = longArrayOf(0L)   // single sample

        // Build ONNX tensors
        val idsTensor = OnnxTensor.createTensor(
            ortEnv,
            LongBuffer.wrap(tokenIds),
            longArrayOf(tokenIds.size.toLong())
        )
        val offsTensor = OnnxTensor.createTensor(
            ortEnv,
            LongBuffer.wrap(offsets),
            longArrayOf(offsets.size.toLong())
        )

        val inputs = mapOf("token_ids" to idsTensor, "offsets" to offsTensor)
        val output = ortSession.run(inputs)

        // Output: float[1][2] probabilities — sigmoid(logits) from OVA loss
        @Suppress("UNCHECKED_CAST")
        val probs = (output[0].value as Array<FloatArray>)[0]

        idsTensor.close()
        offsTensor.close()
        output.close()

        val elapsed = System.currentTimeMillis() - t0

        // labels list from vocab matches output neuron order
        // Find highest probability label
        var maxProb = probs[0]
        var maxIdx  = 0
        for (i in 1 until probs.size) {
            if (probs[i] > maxProb) { maxProb = probs[i]; maxIdx = i }
        }

        val label = if (maxIdx < labels.size) labels[maxIdx] else "not_food"
        return ClassificationResult(label, maxProb, elapsed)
    }

    fun isFood(text: String): Boolean = classify(text).label == "food"

    fun close() {
        ortSession.close()
        ortEnv.close()
    }
}
