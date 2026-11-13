package com.xlr8.deadreckoning.engine

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Instantaneous Virtual Alignment.
 *
 * Resolves phone mounting orientation instantly from the first accelerometer reading.
 * Zero wait time, zero 5-second calibration delay.
 * Continuously refines the gravity vector during low-speed periods using an exponential moving average.
 */
class VirtualAlignment {

    private var r00 = 1.0f
    private var r01 = 0.0f
    private var r02 = 0.0f

    private var r10 = 0.0f
    private var r11 = 1.0f
    private var r12 = 0.0f

    private var r20 = 0.0f
    private var r21 = 0.0f
    private var r22 = 1.0f

    var isCalibrated: Boolean = false
        private set

    private var filteredGx = 0.0
    private var filteredGy = 0.0
    private var filteredGz = 9.81

    /**
     * Instantly calibrates attitude from the very first frame.
     */
    fun processSample(ax: Float, ay: Float, az: Float, isStationary: Boolean) {
        if (!isCalibrated) {
            filteredGx = ax.toDouble()
            filteredGy = ay.toDouble()
            filteredGz = az.toDouble()
            updateRotationMatrix(filteredGx, filteredGy, filteredGz)
            isCalibrated = true
            return
        }

        // Continuously refine gravity estimate when stationary
        if (isStationary) {
            val alpha = 0.05
            filteredGx = (1 - alpha) * filteredGx + alpha * ax
            filteredGy = (1 - alpha) * filteredGy + alpha * ay
            filteredGz = (1 - alpha) * filteredGz + alpha * az
            updateRotationMatrix(filteredGx, filteredGy, filteredGz)
        }
    }

    private fun updateRotationMatrix(gx: Double, gy: Double, gz: Double) {
        val pitch = atan2(-gx, sqrt(gy * gy + gz * gz))
        val roll = atan2(gy, gz)

        val cp = cos(pitch).toFloat()
        val sp = sin(pitch).toFloat()
        val cr = cos(roll).toFloat()
        val sr = sin(roll).toFloat()

        // R = Ry(pitch) * Rx(roll)
        r00 = cp
        r01 = sp * sr
        r02 = sp * cr

        r10 = 0.0f
        r11 = cr
        r12 = -sr

        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr
    }

    /**
     * Rotate raw phone acceleration into the vehicle body frame.
     */
    fun align(ax: Float, ay: Float, az: Float, result: FloatArray) {
        result[0] = r00 * ax + r01 * ay + r02 * az
        result[1] = r10 * ax + r11 * ay + r12 * az
        result[2] = r20 * ax + r21 * ay + r22 * az
    }
}
