package com.xlr8.deadreckoning.engine

import android.content.Context
import org.json.JSONObject
import java.io.InputStreamReader

/**
 * 4th-order zero-phase Butterworth IIR Low-Pass filter channel.
 * Cutoff: 2.0 Hz, Sample Rate: 10.0 Hz.
 */
class ButterworthFilterChannel {
    // Coefficients matched to Python scipy.signal.butter(4, 2.0/5.0, btype='low')
    private val b = floatArrayOf(
        0.046582907f, 0.18633163f, 0.27949744f, 0.18633163f, 0.046582907f
    )
    private val a = floatArrayOf(
        1.0f, -0.7820952f, 0.67997855f, -0.1826757f, 0.030118875f
    )

    // Direct Form II Transposed state registers
    private val d = FloatArray(4)

    fun filter(x: Float): Float {
        val y = b[0] * x + d[0]
        d[0] = b[1] * x - a[1] * y + d[1]
        d[1] = b[2] * x - a[2] * y + d[2]
        d[2] = b[3] * x - a[3] * y + d[3]
        d[3] = b[4] * x - a[4] * y
        return y
    }

    fun reset() {
        for (i in d.indices) d[i] = 0f
    }
}

/**
 * Sensor preprocessor and sliding window manager.
 * Implements Phase 1 & 2 data pipeline on-device.
 */
class SensorPreprocessor(context: Context) {

    companion object {
        const val NUM_FEATURES = 6
        const val WINDOW_SIZE = 100 // 10.0 seconds @ 10Hz
    }

    private val filters = Array(NUM_FEATURES) { ButterworthFilterChannel() }

    // Feature scaler values loaded from scaler_params.json
    private val means = FloatArray(NUM_FEATURES)
    private val scales = FloatArray(NUM_FEATURES)

    // Circular buffer for 100 frames x 6 features
    // Stored as [frame_idx][feature_idx]
    private val buffer = Array(WINDOW_SIZE) { FloatArray(NUM_FEATURES) }
    private var writeIndex = 0
    private var frameCount = 0

    init {
        loadScalerParams(context)
    }

    private fun loadScalerParams(context: Context) {
        try {
            context.assets.open("scaler_params.json").use { inputStream ->
                val jsonStr = InputStreamReader(inputStream).readText()
                val json = JSONObject(jsonStr)
                val meanArray = json.getJSONArray("mean")
                val scaleArray = json.getJSONArray("scale")

                for (i in 0 until NUM_FEATURES) {
                    means[i] = meanArray.optDouble(i, 0.0).toFloat()
                    scales[i] = scaleArray.optDouble(i, 1.0).toFloat()
                    if (scales[i] == 0f) scales[i] = 1.0f
                }
            }
        } catch (e: Exception) {
            // Safe fallback defaults matching bulk dataset
            val defaultMeans = floatArrayOf(1.34f, 0.035f, 16.12f, -0.0006f, -0.0035f, -0.0002f)
            val defaultScales = floatArrayOf(15.96f, 1.17f, 170.05f, 0.076f, 0.200f, 0.112f)
            System.arraycopy(defaultMeans, 0, means, 0, NUM_FEATURES)
            System.arraycopy(defaultScales, 0, scales, 0, NUM_FEATURES)
        }
    }

    /**
     * Ingest one 10Hz IMU frame:
     * acc: [ax_aligned, ay_aligned, az_aligned]
     * gyro: [yaw_rate, pitch_rate, roll_rate] in rad/s
     */
    fun pushSample(
        axAligned: Float,
        ayAligned: Float,
        azAligned: Float,
        gyroYaw: Float,
        gyroPitch: Float,
        gyroRoll: Float
    ) {
        val raw = floatArrayOf(axAligned, ayAligned, azAligned, gyroYaw, gyroPitch, gyroRoll)
        val filtered = FloatArray(NUM_FEATURES)

        for (i in 0 until NUM_FEATURES) {
            val cleanVal = filters[i].filter(raw[i])
            filtered[i] = (cleanVal - means[i]) / scales[i]
        }

        buffer[writeIndex] = filtered
        writeIndex = (writeIndex + 1) % WINDOW_SIZE
        frameCount++
    }

    fun isWindowReady(): Boolean = frameCount >= WINDOW_SIZE

    /**
     * Extracts input tensor for 1D-CNN.
     * Shape: (Channels/Features=6, Sequence_Length=100)
     * Flattened size = 600 floats.
     *
     * Order: Channel 0 across 100 time steps, Channel 1 across 100 time steps, ...
     */
    fun getTensorData(): FloatArray {
        val tensor = FloatArray(NUM_FEATURES * WINDOW_SIZE)
        val start = if (frameCount >= WINDOW_SIZE) writeIndex else 0
        val count = minOf(frameCount, WINDOW_SIZE)

        for (t in 0 until count) {
            val bufIdx = (start + t) % WINDOW_SIZE
            val frame = buffer[bufIdx]
            for (f in 0 until NUM_FEATURES) {
                // index in [6, 100] tensor: f * 100 + t
                tensor[f * WINDOW_SIZE + t] = frame[f]
            }
        }
        return tensor
    }

    fun reset() {
        for (f in filters) f.reset()
        writeIndex = 0
        frameCount = 0
    }
}
