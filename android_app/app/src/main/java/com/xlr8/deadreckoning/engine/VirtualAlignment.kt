package com.xlr8.deadreckoning.engine

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Dynamic Virtual Alignment module.
 *
 * Implements stationary gravity vector attitude estimation (Phase 1).
 * Computes Pitch and Roll Euler angles relative to gravity and projects
 * arbitrary smartphone mounting acceleration into the vehicle body frame.
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

    private val calibrationSamples = ArrayList<FloatArray>()
    private val requiredSamples = 50 // ~5 seconds @ 10Hz

    fun resetCalibration() {
        calibrationSamples.clear()
        isCalibrated = false
        r00 = 1.0f; r01 = 0.0f; r02 = 0.0f
        r10 = 0.0f; r11 = 1.0f; r12 = 0.0f
        r20 = 0.0f; r21 = 0.0f; r22 = 1.0f
    }

    /**
     * Feed stationary acceleration sample (ax, ay, az).
     * Returns true when calibration finishes.
     */
    fun addCalibrationSample(ax: Float, ay: Float, az: Float): Boolean {
        calibrationSamples.add(floatArrayOf(ax, ay, az))
        if (calibrationSamples.size >= requiredSamples) {
            computeMatrixFromSamples()
            isCalibrated = true
            return true
        }
        return false
    }

    private fun computeMatrixFromSamples() {
        var sumX = 0.0
        var sumY = 0.0
        var sumZ = 0.0
        for (s in calibrationSamples) {
            sumX += s[0]
            sumY += s[1]
            sumZ += s[2]
        }
        val n = calibrationSamples.size.toDouble()
        val gx = sumX / n
        val gy = sumY / n
        val gz = sumZ / n

        // Phase 1 Euler angles
        val pitch = atan2(-gx, sqrt(gy * gy + gz * gz))
        val roll = atan2(gy, gz)

        val cp = cos(pitch).toFloat()
        val sp = sin(pitch).toFloat()
        val cr = cos(roll).toFloat()
        val sr = sin(roll).toFloat()

        // R = Ry(pitch) * Rx(roll)
        // Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
        // Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
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
     * Transform raw acceleration vector into vehicle aligned coordinates.
     * Output: FloatArray of size 3 [acc_x_aligned, acc_y_aligned, acc_z_aligned]
     */
    fun align(ax: Float, ay: Float, az: Float, result: FloatArray) {
        result[0] = r00 * ax + r01 * ay + r02 * az
        result[1] = r10 * ax + r11 * ay + r12 * az
        result[2] = r20 * ax + r21 * ay + r22 * az
    }
}
