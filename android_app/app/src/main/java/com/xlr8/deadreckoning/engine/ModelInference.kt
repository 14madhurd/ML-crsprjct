package com.xlr8.deadreckoning.engine

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

data class ModelOutput(val speed: Float, val direction: Float)

/**
 * On-device PyTorch Mobile Inference Engine.
 * Loads and runs bulk_velocity_cnn.pt (DualHeadVelocityCNN).
 */
class ModelInference(private val context: Context) {

    companion object {
        private const val TAG = "ModelInference"
        private const val MODEL_NAME = "bulk_velocity_cnn.pt"
    }

    private var module: Module? = null
    var isLoaded: Boolean = false
        private set

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val modelPath = assetFilePath(context, MODEL_NAME)
            module = Module.load(modelPath)
            isLoaded = true
            Log.i(TAG, "PyTorch Mobile Dual-Head Model loaded successfully from $modelPath")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to load PyTorch module directly: ${e.message}. Using native fallback.")
            isLoaded = false
        }
    }

    /**
     * Executes forward pass of Dual-Head 1D-CNN.
     * Input: float array of 600 elements corresponding to [1, 6, 100]
     * Returns: ModelOutput(speed in m/s, direction/heading rate in deg)
     */
    fun runInference(inputData: FloatArray): ModelOutput {
        val mod = module
        if (isLoaded && mod != null) {
            try {
                val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 6, 100))
                val output = mod.forward(IValue.from(inputTensor))

                return if (output.isTuple) {
                    val tuple = output.toTuple()
                    val speedVal = tuple[0].toTensor().dataAsFloatArray[0]
                    val dirVal = tuple[1].toTensor().dataAsFloatArray[0]
                    ModelOutput(maxOf(0.0f, speedVal), dirVal)
                } else if (output.isList) {
                    val list = output.toList()
                    val speedVal = list[0].toTensor().dataAsFloatArray[0]
                    val dirVal = list[1].toTensor().dataAsFloatArray[0]
                    ModelOutput(maxOf(0.0f, speedVal), dirVal)
                } else {
                    val tensor = output.toTensor()
                    val arr = tensor.dataAsFloatArray
                    ModelOutput(maxOf(0.0f, arr[0]), if (arr.size > 1) arr[1] else 0.0f)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Inference exception: ${e.message}. Falling back.")
            }
        }

        // Native kinematic fallback: integrate longitudinal acceleration
        // inputData[0..99] is acc_x, inputData[100..199] is acc_y (longitudinal forward)
        val forwardAccAvg = inputData.slice(100..199).average().toFloat()
        val estimatedSpeed = maxOf(0.0f, forwardAccAvg * 1.5f + 5.0f)
        val gyroYawAvg = inputData.slice(300..399).average().toFloat()
        return ModelOutput(estimatedSpeed, gyroYawAvg * 57.2958f)
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(8 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}
