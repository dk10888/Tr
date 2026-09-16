package com.example.receiptscanner.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import android.content.Context
import android.util.Log
import java.nio.LongBuffer

/**
 * BERT/ELECTRA food classifier powered by ONNX Runtime for Android.
 *
 * Supports GPU acceleration via NNAPI delegate (Android Neural Networks API).
 * Falls back to CPU automatically if NNAPI is unavailable.
 *
 * @param context  Android context (for assets access)
 * @param modelFileName  ONNX model filename in assets/ (e.g. "english_food_bert.onnx")
 * @param vocabFileName  vocab.txt filename in assets/ (e.g. "english_vocab.txt")
 * @param clsTokenId  [CLS] token ID (2 for most BERT models)
 * @param sepTokenId  [SEP] token ID (3)
 * @param padTokenId  [PAD] token ID (0)
 * @param unkTokenId  [UNK] token ID (1)
 * @param maxLen      Maximum sequence length (default 64)
 * @param useGpu      Try NNAPI GPU delegate if true (default true)
 */
class OnnxBertClassifier(
    private val context: Context,
    private val modelFileName: String,
    private val vocabFileName: String,
    private val clsTokenId: Int = 2,
    private val sepTokenId: Int = 3,
    private val padTokenId: Int = 0,
    private val unkTokenId: Int = 1,
    private val maxLen: Int = 64,
    private val useGpu: Boolean = true,
) : AutoCloseable {

    private val TAG = "OnnxBertClassifier"

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: WordPieceTokenizer

    init {
        // ── Load ONNX session ────────────────────────────────────────
        val opts = SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)

            if (useGpu) {
                try {
                    // NNAPI = Android Neural Networks API = GPU/DSP/NPU acceleration
                    addNnapi()
                    Log.i(TAG, "[$modelFileName] NNAPI (GPU) delegate enabled ✅")
                } catch (e: Exception) {
                    Log.w(TAG, "[$modelFileName] NNAPI unavailable, using CPU: ${e.message}")
                }
            }
        }

        val modelBytes = context.assets.open(modelFileName).readBytes()
        session = env.createSession(modelBytes, opts)
        Log.i(TAG, "[$modelFileName] ONNX session loaded. Inputs: ${session.inputNames}")

        // ── Load tokenizer vocab ─────────────────────────────────────
        val vocabText = context.assets.open(vocabFileName).bufferedReader().readText()
        val vocab     = WordPieceTokenizer.loadVocab(vocabText)
        tokenizer     = WordPieceTokenizer(vocab, clsTokenId, sepTokenId, padTokenId, unkTokenId)
        Log.i(TAG, "[$vocabFileName] Vocab loaded: ${vocab.size} tokens")
    }

    data class ClassificationResult(
        val text: String,
        val label: String,          // "food" or "not_food"
        val confidence: Float,      // 0.0 – 1.0
        val isFood: Boolean,
    )

    /**
     * Classify a single receipt line.
     * @param text Raw OCR text
     * @param threshold Minimum confidence to call something "food" (default 0.60)
     */
    fun classify(text: String, threshold: Float = 0.60f): ClassificationResult {
        val results = classifyBatch(listOf(text), threshold)
        return results.first()
    }

    /**
     * Classify a batch of receipt lines.
     * More efficient than calling classify() in a loop.
     */
    fun classifyBatch(
        texts: List<String>,
        threshold: Float = 0.60f,
    ): List<ClassificationResult> {
        if (texts.isEmpty()) return emptyList()

        val batchSize = texts.size
        val flatInputIds      = LongArray(batchSize * maxLen)
        val flatAttentionMask = LongArray(batchSize * maxLen)
        val flatTokenTypeIds  = LongArray(batchSize * maxLen)

        texts.forEachIndexed { batchIdx, text ->
            val enc = tokenizer.encode(text, maxLen)
            val offset = batchIdx * maxLen
            enc.inputIds.copyInto(flatInputIds,      offset)
            enc.attentionMask.copyInto(flatAttentionMask, offset)
            enc.tokenTypeIds.copyInto(flatTokenTypeIds,   offset)
        }

        val shape = longArrayOf(batchSize.toLong(), maxLen.toLong())

        val inputIdsTensor      = OnnxTensor.createTensor(env, LongBuffer.wrap(flatInputIds), shape)
        val attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(flatAttentionMask), shape)
        val tokenTypeIdsTensor  = OnnxTensor.createTensor(env, LongBuffer.wrap(flatTokenTypeIds), shape)

        val inputs = mapOf(
            "input_ids"      to inputIdsTensor,
            "attention_mask" to attentionMaskTensor,
            "token_type_ids" to tokenTypeIdsTensor,
        )

        val output = session.run(inputs)
        // logits shape: [batchSize, 2]
        @Suppress("UNCHECKED_CAST")
        val logits = (output[0].value as Array<FloatArray>)

        val results = mutableListOf<ClassificationResult>()
        logits.forEachIndexed { i, logit ->
            val probs = softmax(logit)
            val predIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val label = if (predIdx == 1) "food" else "not_food"
            val conf = probs[predIdx]
            results.add(
                ClassificationResult(
                    text       = texts[i],
                    label      = label,
                    confidence = conf,
                    isFood     = (predIdx == 1) && (conf >= threshold),
                )
            )
        }

        // Clean up tensors
        inputIdsTensor.close()
        attentionMaskTensor.close()
        tokenTypeIdsTensor.close()
        output.close()

        return results
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxVal = logits.max()
        val exps   = logits.map { Math.exp((it - maxVal).toDouble()).toFloat() }
        val sum    = exps.sum()
        return exps.map { it / sum }.toFloatArray()
    }

    override fun close() {
        session.close()
        env.close()
    }
}
