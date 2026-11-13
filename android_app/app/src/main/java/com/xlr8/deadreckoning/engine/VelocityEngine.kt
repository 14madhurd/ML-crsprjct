package com.xlr8.deadreckoning.engine

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer
import kotlin.math.abs

class VelocityEngine(context: Context) {
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession = ortEnvironment.createSession(context.assets.open("velocity_engine.onnx").readBytes())
    private val inputName = ortSession.inputNames.iterator().next() // Dynamically fetch ONNX input node
    
    private var means: FloatArray
    private var scales: FloatArray

    init {
        // FIX 2: Load the exact normalization parameters from Phase 2 Python export
        val jsonString = context.assets.open("scaler_params.json").bufferedReader().use { it.readText() }
        val mapType = object : TypeToken<Map<String, List<Float>>>() {}.type
        val scalerData: Map<String, List<Float>> = Gson().fromJson(jsonString, mapType)
        
        means = scalerData["mean"]!!.toFloatArray()
        scales = scalerData["scale"]!!.toFloatArray()
    }

    fun predict(sensorWindow: List<FloatArray>): Pair<Float, Float> {
        if (sensorWindow.size < 100) return Pair(0f, 0f)

        // FIX 3: Zero Velocity Update (ZUPT) Threshold
        // If the maximum linear acceleration across the 10-second window is microscopic, the vehicle is stationary.
        var maxAccel = 0f
        for (frame in sensorWindow) {
            val accelMagnitude = abs(frame[0]) + abs(frame[1]) + abs(frame[2])
            if (accelMagnitude > maxAccel) maxAccel = accelMagnitude
        }
        
        // 0.15 m/s^2 is the standard smartphone desk-vibration noise floor
        if (maxAccel < 0.15f) { 
            return Pair(0f, 0f) // Hard lock predictions to zero without running the neural network
        }

        // Flatten, scale, and format the 100x6 window for the PyTorch 1D-CNN format (Channels x Time)
        val flatBuffer = FloatBuffer.allocate(1 * 6 * 100)
        for (featureIdx in 0..5) {
            for (frameIdx in 0..99) {
                val rawValue = sensorWindow[frameIdx][featureIdx]
                val scaledValue = (rawValue - means[featureIdx]) / scales[featureIdx]
                flatBuffer.put(scaledValue)
            }
        }
        flatBuffer.rewind()

        // Run ONNX Inference
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, flatBuffer, longArrayOf(1, 6, 100))
        val results = ortSession.run(mapOf(inputName to inputTensor))
        
        // Extract Dual-Head Predictions (Speed and Heading)
        val speedPrediction = (results[0].value as Array<FloatArray>)[0][0]
        val headingPrediction = (results[1].value as Array<FloatArray>)[0][0]

        inputTensor.close()
        results.close()

        // Ensure speed does not output negative reversing artifacts
        val finalSpeed = if (speedPrediction < 0f) 0f else speedPrediction

        return Pair(finalSpeed, headingPrediction)
    }
}
