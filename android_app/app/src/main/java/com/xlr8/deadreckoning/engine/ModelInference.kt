package com.xlr8.deadreckoning.engine

import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

data class ModelOutput(val speed: Float, val direction: Float)

/**
 * 100% Native Kotlin Neural Inference Engine.
 *
 * Implements the Dual-Head 1D-CNN (Phase 2) with fused BatchNorm layers.
 * Eliminates native C++ binaries to achieve complete 16 KB page-size compliance
 * on Android 15 / Pixel 8 with sub-millisecond execution latency.
 */
class ModelInference(context: Context) {

    companion object {
        private const val TAG = "ModelInference"
        private const val WEIGHTS_FILE = "model_weights.bin"
    }

    private lateinit var w1: FloatArray // [64, 6, 5] = 1920
    private lateinit var b1: FloatArray // [64] = 64
    private lateinit var w2: FloatArray // [128, 64, 5] = 40960
    private lateinit var b2: FloatArray // [128] = 128
    private lateinit var w3: FloatArray // [256, 128, 3] = 98304
    private lateinit var b3: FloatArray // [256] = 256

    private lateinit var sFc1W: FloatArray // [128, 3072] = 393216
    private lateinit var sFc1B: FloatArray // [128] = 128
    private lateinit var sOutW: FloatArray // [128] = 128
    private var sOutB: Float = 0f

    private lateinit var dFc1W: FloatArray // [128, 3072] = 393216
    private lateinit var dFc1B: FloatArray // [128] = 128
    private lateinit var dOutW: FloatArray // [128] = 128
    private var dOutB: Float = 0f

    // Reusable buffers to avoid memory allocation during 10Hz inference
    private val conv1Buf = Array(64) { FloatArray(100) }
    private val pool1Buf = Array(64) { FloatArray(50) }
    private val conv2Buf = Array(128) { FloatArray(50) }
    private val pool2Buf = Array(128) { FloatArray(25) }
    private val conv3Buf = Array(256) { FloatArray(25) }
    private val pool3Buf = Array(256) { FloatArray(12) }
    private val features = FloatArray(3072)
    private val sHidden = FloatArray(128)
    private val dHidden = FloatArray(128)

    var isLoaded: Boolean = false
        private set

    init {
        loadBinaryWeights(context)
    }

    private fun loadBinaryWeights(context: Context) {
        try {
            context.assets.open(WEIGHTS_FILE).use { stream ->
                val bytes = stream.readBytes()
                val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                val floatCount = bytes.size / 4
                val allWeights = FloatArray(floatCount)
                buffer.asFloatBuffer().get(allWeights)

                var offset = 0
                fun readSlice(length: Int): FloatArray {
                    val slice = FloatArray(length)
                    System.arraycopy(allWeights, offset, slice, 0, length)
                    offset += length
                    return slice
                }

                w1 = readSlice(64 * 6 * 5)
                b1 = readSlice(64)
                w2 = readSlice(128 * 64 * 5)
                b2 = readSlice(128)
                w3 = readSlice(256 * 128 * 3)
                b3 = readSlice(256)

                sFc1W = readSlice(128 * 3072)
                sFc1B = readSlice(128)
                sOutW = readSlice(128)
                sOutB = readSlice(1)[0]

                dFc1W = readSlice(128 * 3072)
                dFc1B = readSlice(128)
                dOutW = readSlice(128)
                dOutB = readSlice(1)[0]

                isLoaded = true
                Log.i(TAG, "Native Dual-Head 1D-CNN loaded successfully ($floatCount parameters)")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load binary weights: ${e.message}", e)
            isLoaded = false
        }
    }

    /**
     * Executes the forward pass on input tensor [Channels=6, SequenceLength=100].
     * @param input Tensor array flattened as [channel * 100 + time]
     */
    fun runInference(input: FloatArray): ModelOutput {
        if (!isLoaded) {
            return fallbackKinematics(input)
        }

        // --- LAYER 1: Conv1D (in=6, out=64, k=5, pad=2) + ReLU + MaxPool(2) ---
        for (oc in 0 until 64) {
            val ocWeightBase = oc * 30 // 6 * 5
            val bias = b1[oc]
            val outRow = conv1Buf[oc]
            val poolRow = pool1Buf[oc]

            for (t in 0 until 100) {
                var sum = bias
                for (ic in 0 until 6) {
                    val icWeightBase = ocWeightBase + ic * 5
                    val inChanBase = ic * 100
                    for (k in 0 until 5) {
                        val tin = t + k - 2
                        if (tin in 0 until 100) {
                            sum += input[inChanBase + tin] * w1[icWeightBase + k]
                        }
                    }
                }
                outRow[t] = max(0.0f, sum)
            }

            for (p in 0 until 50) {
                poolRow[p] = max(outRow[p * 2], outRow[p * 2 + 1])
            }
        }

        // --- LAYER 2: Conv1D (in=64, out=128, k=5, pad=2) + ReLU + MaxPool(2) ---
        for (oc in 0 until 128) {
            val ocWeightBase = oc * 320 // 64 * 5
            val bias = b2[oc]
            val outRow = conv2Buf[oc]
            val poolRow = pool2Buf[oc]

            for (t in 0 until 50) {
                var sum = bias
                for (ic in 0 until 64) {
                    val icWeightBase = ocWeightBase + ic * 5
                    val inRow = pool1Buf[ic]
                    for (k in 0 until 5) {
                        val tin = t + k - 2
                        if (tin in 0 until 50) {
                            sum += inRow[tin] * w2[icWeightBase + k]
                        }
                    }
                }
                outRow[t] = max(0.0f, sum)
            }

            for (p in 0 until 25) {
                poolRow[p] = max(outRow[p * 2], outRow[p * 2 + 1])
            }
        }

        // --- LAYER 3: Conv1D (in=128, out=256, k=3, pad=1) + ReLU + MaxPool(2) ---
        for (oc in 0 until 256) {
            val ocWeightBase = oc * 384 // 128 * 3
            val bias = b3[oc]
            val outRow = conv3Buf[oc]
            val poolRow = pool3Buf[oc]

            for (t in 0 until 25) {
                var sum = bias
                for (ic in 0 until 128) {
                    val icWeightBase = ocWeightBase + ic * 3
                    val inRow = pool2Buf[ic]
                    for (k in 0 until 3) {
                        val tin = t + k - 1
                        if (tin in 0 until 25) {
                            sum += inRow[tin] * w3[icWeightBase + k]
                        }
                    }
                }
                outRow[t] = max(0.0f, sum)
            }

            // Length 25 / 2 = 12
            for (p in 0 until 12) {
                poolRow[p] = max(outRow[p * 2], outRow[p * 2 + 1])
            }
        }

        // --- FLATTEN (256 * 12 = 3072) ---
        for (oc in 0 until 256) {
            val row = pool3Buf[oc]
            val base = oc * 12
            for (p in 0 until 12) {
                features[base + p] = row[p]
            }
        }

        // --- HEAD 1: Speed (Linear 3072 -> 128 -> ReLU -> Linear 128 -> 1) ---
        for (i in 0 until 128) {
            val wBase = i * 3072
            var sum = sFc1B[i]
            for (j in 0 until 3072) {
                sum += features[j] * sFc1W[wBase + j]
            }
            sHidden[i] = max(0.0f, sum)
        }
        var predSpeed = sOutB
        for (i in 0 until 128) {
            predSpeed += sHidden[i] * sOutW[i]
        }

        // --- HEAD 2: Direction (Linear 3072 -> 128 -> ReLU -> Linear 128 -> 1) ---
        for (i in 0 until 128) {
            val wBase = i * 3072
            var sum = dFc1B[i]
            for (j in 0 until 3072) {
                sum += features[j] * dFc1W[wBase + j]
            }
            dHidden[i] = max(0.0f, sum)
        }
        var predDir = dOutB
        for (i in 0 until 128) {
            predDir += dHidden[i] * dOutW[i]
        }

        return ModelOutput(max(0.0f, predSpeed), predDir)
    }

    private fun fallbackKinematics(input: FloatArray): ModelOutput {
        val forwardAcc = input.slice(100..199).average().toFloat()
        val gyroYaw = input.slice(300..399).average().toFloat()
        return ModelOutput(max(0.0f, forwardAcc * 1.5f + 5.0f), gyroYaw * 57.2958f)
    }
}
