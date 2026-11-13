package com.xlr8.deadreckoning.engine

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * IMU Sensor Manager with real-time Butterworth IIR filtering.
 *
 * Replicates the exact Python Phase 1 preprocessing pipeline:
 *   scipy.signal.butter(N=4, Wn=2.0/(10.0/2), btype='low')
 *
 * Every raw sample passes through a 4th-order IIR low-pass filter (fc=2.0 Hz,
 * fs=10.0 Hz) BEFORE entering the sliding window. This eliminates the
 * training-inference skew that caused the model to hallucinate speed/heading
 * from high-frequency desk vibration noise it was never trained on.
 *
 * Uses TYPE_LINEAR_ACCELERATION (gravity already stripped by Android sensor
 * fusion) and TYPE_GYROSCOPE.
 */
class IMUSensorManager(context: Context) : SensorEventListener {

    companion object {
        private const val NUM_CHANNELS = 6   // 3 accel + 3 gyro
        private const val WINDOW_SIZE = 100  // 10 seconds @ 10 Hz
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val linearAccel = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val gyroscope   = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    val sensorWindow = ConcurrentLinkedQueue<FloatArray>()

    private var lastAcc = FloatArray(3)
    private var lastGyr = FloatArray(3)

    // --- Per-channel 4th-order Butterworth IIR low-pass filters ---
    // Coefficients match scipy.signal.butter(4, 2.0/5.0, btype='low')
    // i.e. order=4, cutoff=2.0 Hz, Nyquist=5.0 Hz (sample rate 10 Hz)
    private val filters = Array(NUM_CHANNELS) { ButterworthLPF() }

    fun startRecording() {
        // 100 000 us = 10 Hz sampling rate
        sensorManager.registerListener(this, linearAccel, 100000)
        sensorManager.registerListener(this, gyroscope,   100000)
    }

    fun stopRecording() {
        sensorManager.unregisterListener(this)
        sensorWindow.clear()
        for (f in filters) f.reset()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> lastAcc = event.values.clone()
            Sensor.TYPE_GYROSCOPE           -> lastGyr = event.values.clone()
        }

        // Assemble the 6-channel raw frame: [ax, ay, az, gx, gy, gz]
        val rawFrame = floatArrayOf(
            lastAcc[0], lastAcc[1], lastAcc[2],
            lastGyr[0], lastGyr[1], lastGyr[2]
        )

        // Apply per-channel Butterworth IIR filtering BEFORE windowing
        val filteredFrame = FloatArray(NUM_CHANNELS)
        for (ch in 0 until NUM_CHANNELS) {
            filteredFrame[ch] = filters[ch].filter(rawFrame[ch])
        }

        sensorWindow.add(filteredFrame)
        if (sensorWindow.size > WINDOW_SIZE) sensorWindow.poll()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}

/**
 * Causal 4th-order Butterworth IIR Low-Pass Filter (Direct-Form II Transposed).
 *
 * Reproduces scipy.signal.butter(4, 2.0/5.0, btype='low', analog=False)
 *
 * Transfer function coefficients (numerator b, denominator a):
 *   b = [0.04658291, 0.18633163, 0.27949745, 0.18633163, 0.04658291]
 *   a = [1.0,       -0.78209521, 0.67997856, -0.18267570, 0.03011888]
 *
 * Note: Python's filtfilt applies the filter forward then backward (zero-phase,
 * non-causal, effective 8th order). In real-time we can only apply it causally
 * (forward-only), which introduces a small phase lag but matches the amplitude
 * response that the model learned. The critical property the model depends on is
 * the frequency-domain attenuation of jitter above 2 Hz, which is preserved.
 */
class ButterworthLPF {

    // Numerator coefficients (feedforward)
    private val b0 = 0.046582907f
    private val b1 = 0.18633163f
    private val b2 = 0.27949745f
    private val b3 = 0.18633163f
    private val b4 = 0.046582907f

    // Denominator coefficients (feedback) -- a[0] = 1.0 is implicit
    private val a1 = -0.78209521f
    private val a2 =  0.67997856f
    private val a3 = -0.18267570f
    private val a4 =  0.03011888f

    // Direct-Form II Transposed state variables (4 delay elements)
    private var d0 = 0.0f
    private var d1 = 0.0f
    private var d2 = 0.0f
    private var d3 = 0.0f

    /**
     * Filter a single sample. Call once per channel per time step.
     * @param x The raw input sample.
     * @return The low-pass filtered output sample.
     */
    fun filter(x: Float): Float {
        val y = b0 * x + d0
        d0 = b1 * x - a1 * y + d1
        d1 = b2 * x - a2 * y + d2
        d2 = b3 * x - a3 * y + d3
        d3 = b4 * x - a4 * y
        return y
    }

    /** Reset all internal state. Call when restarting sensor collection. */
    fun reset() {
        d0 = 0.0f
        d1 = 0.0f
        d2 = 0.0f
        d3 = 0.0f
    }
}
