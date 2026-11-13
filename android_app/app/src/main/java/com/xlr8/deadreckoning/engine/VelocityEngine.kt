package com.xlr8.deadreckoning.engine

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * ONNX Dual-Head 1D-CNN Velocity & Heading Inference Engine.
 *
 * Pipeline (matches Python Phase 2 training exactly):
 *   1. IMUSensorManager delivers Butterworth-filtered sensor frames (2 Hz LPF)
 *   2. This engine applies StandardScaler normalization (scaler_params.json)
 *   3. ZUPT gate: if the filtered signal variance is below the noise floor,
 *      hard-lock output to (0, 0) without running the neural network
 *   4. Format tensor as [1, 6, 100] (Channels-first for 1D-CNN)
 *   5. ONNX inference produces speed (m/s) and heading change (degrees)
 *   6. Post-process: clamp negative speed, cap maximum plausible speed
 */
class VelocityEngine(context: Context) {

    companion object {
        private const val TAG = "VelocityEngine"

        // ZUPT thresholds (tuned for Butterworth-filtered LINEAR_ACCELERATION)
        //
        // After the 2 Hz low-pass filter, a stationary phone on a desk produces
        // acceleration variance on the order of 1e-5 to 1e-4 m^2/s^4.
        // A phone in a gently moving vehicle produces variance > 0.01.
        // We set the threshold conservatively between these two regimes.
        private const val ZUPT_VARIANCE_THRESHOLD = 0.005f   // m^2/s^4
        private const val ZUPT_MAX_ACCEL_THRESHOLD = 0.10f   // m/s^2 (per-axis peak)

        // Plausibility cap: 200 km/h = 55.6 m/s -- anything above is a hallucination
        private const val MAX_PLAUSIBLE_SPEED_MS = 55.6f
    }

    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: ai.onnxruntime.OrtSession
    private val inputName: String

    private var means: FloatArray
    private var scales: FloatArray

    init {
        // Load ONNX model
        val modelBytes = context.assets.open("velocity_engine.onnx").readBytes()
        ortSession = ortEnvironment.createSession(modelBytes)
        inputName = ortSession.inputNames.iterator().next()
        Log.i(TAG, "ONNX session loaded. Input node: '$inputName'")

        // Load StandardScaler parameters exported from Phase 2 Python training
        val jsonString = context.assets.open("scaler_params.json").bufferedReader().use { it.readText() }
        val mapType = object : TypeToken<Map<String, List<Float>>>() {}.type
        val scalerData: Map<String, List<Float>> = Gson().fromJson(jsonString, mapType)

        means = scalerData["mean"]!!.toFloatArray()
        scales = scalerData["scale"]!!.toFloatArray()

        Log.i(TAG, "Scaler loaded: ${means.size} features, " +
              "mean[0..2]=[${means.take(3).joinToString()}], " +
              "scale[0..2]=[${scales.take(3).joinToString()}]")
    }

    /**
     * Predict speed (m/s) and heading change (degrees) from a 100-frame
     * Butterworth-filtered sensor window.
     *
     * @param sensorWindow List of 100 FloatArrays, each [ax, ay, az, gx, gy, gz]
     *        (already low-pass filtered by IMUSensorManager)
     * @return Pair(speed_m_s, heading_change_deg). Returns (0, 0) if ZUPT fires.
     */
    fun predict(sensorWindow: List<FloatArray>): Pair<Float, Float> {
        if (sensorWindow.size < 100) return Pair(0f, 0f)

        // ---- ZUPT Gate: Dual-criteria stationarity detection ----

        // Criterion 1: Max absolute acceleration across the entire window.
        // If the Butterworth-filtered signal never exceeds the noise floor,
        // the device is absolutely stationary.
        var maxAbsAccel = 0f
        for (frame in sensorWindow) {
            for (axis in 0..2) {
                val absVal = abs(frame[axis])
                if (absVal > maxAbsAccel) maxAbsAccel = absVal
            }
        }

        if (maxAbsAccel < ZUPT_MAX_ACCEL_THRESHOLD) {
            return Pair(0f, 0f)
        }

        // Criterion 2: Variance of acceleration magnitude across the window.
        // Even if individual filtered samples drift slightly above the threshold
        // (e.g. from filter transients), variance catches steady-state stillness.
        val accelMags = FloatArray(sensorWindow.size)
        var sumMag = 0f
        for (i in sensorWindow.indices) {
            val frame = sensorWindow[i]
            val mag = sqrt(frame[0] * frame[0] + frame[1] * frame[1] + frame[2] * frame[2])
            accelMags[i] = mag
            sumMag += mag
        }
        val meanMag = sumMag / sensorWindow.size

        var varianceAccum = 0f
        for (m in accelMags) {
            val diff = m - meanMag
            varianceAccum += diff * diff
        }
        val variance = varianceAccum / sensorWindow.size

        if (variance < ZUPT_VARIANCE_THRESHOLD) {
            return Pair(0f, 0f)
        }

        // ---- StandardScaler normalization + Channels-first tensor layout ----
        // Format: [batch=1, channels=6, time=100]
        val flatBuffer = FloatBuffer.allocate(1 * 6 * 100)
        for (featureIdx in 0..5) {
            for (frameIdx in 0..99) {
                val rawValue = sensorWindow[frameIdx][featureIdx]
                val scaledValue = (rawValue - means[featureIdx]) / scales[featureIdx]
                flatBuffer.put(scaledValue)
            }
        }
        flatBuffer.rewind()

        // ---- ONNX Inference ----
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, flatBuffer, longArrayOf(1, 6, 100))
        val results = ortSession.run(mapOf(inputName to inputTensor))

        // Dual-head output: [0] = speed, [1] = heading
        val speedPrediction  = (results[0].value as Array<FloatArray>)[0][0]
        val headingPrediction = (results[1].value as Array<FloatArray>)[0][0]

        inputTensor.close()
        results.close()

        // ---- Post-processing ----
        // Clamp negative speed (model regression artifact)
        // Cap at maximum plausible vehicle speed
        val finalSpeed = when {
            speedPrediction < 0f              -> 0f
            speedPrediction > MAX_PLAUSIBLE_SPEED_MS -> MAX_PLAUSIBLE_SPEED_MS
            else                              -> speedPrediction
        }

        return Pair(finalSpeed, headingPrediction)
    }
}
