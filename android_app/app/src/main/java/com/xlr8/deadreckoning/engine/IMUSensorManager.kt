package com.xlr8.deadreckoning.engine

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.util.concurrent.ConcurrentLinkedQueue

class IMUSensorManager(context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    // FIX 1: Use Linear Acceleration to mathematically exclude Earth's 1-G gravity
    private val linearAccel = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    val sensorWindow = ConcurrentLinkedQueue<FloatArray>()
    private var lastAcc = FloatArray(3)
    private var lastGyr = FloatArray(3)

    fun startRecording() {
        sensorManager.registerListener(this, linearAccel, 100000)
        sensorManager.registerListener(this, gyroscope, 100000)
    }

    fun stopRecording() {
        sensorManager.unregisterListener(this)
        sensorWindow.clear()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> lastAcc = event.values.clone()
            Sensor.TYPE_GYROSCOPE -> lastGyr = event.values.clone()
        }

        val currentFrame = floatArrayOf(
            lastAcc[0], lastAcc[1], lastAcc[2],
            lastGyr[0], lastGyr[1], lastGyr[2]
        )

        sensorWindow.add(currentFrame)
        if (sensorWindow.size > 100) sensorWindow.poll()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
