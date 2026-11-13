package com.xlr8.deadreckoning.gnss

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper

interface SensorStreamListener {
    fun onImuFrame(ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float)
    fun onGnssLocation(location: Location)
}

/**
 * Manages raw hardware sensors (Accelerometer + Gyroscope @ 10Hz)
 * and Android GNSS positioning provider.
 */
class LocationTracker(
    private val context: Context,
    private val listener: SensorStreamListener
) : SensorEventListener, LocationListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private var rawAx = 0f; private var rawAy = 0f; private var rawAz = 9.8f
    private var rawGx = 0f; private var rawGy = 0f; private var rawGz = 0f

    private val handler = Handler(Looper.getMainLooper())
    private var isStreaming = false

    // Periodic 10Hz timer (100ms interval) to dispatch synced frames to the pipeline
    private val imuSampler = object : Runnable {
        override fun run() {
            if (!isStreaming) return
            listener.onImuFrame(rawAx, rawAy, rawAz, rawGx, rawGy, rawGz)
            handler.postDelayed(this, 100L) // 10Hz exactly
        }
    }

    @SuppressLint("MissingPermission")
    fun start() {
        if (isStreaming) return
        isStreaming = true

        // Register sensors
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        // Register GNSS
        try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1000L,
                    0.0f,
                    this,
                    Looper.getMainLooper()
                )
            }
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.NETWORK_PROVIDER,
                    1000L,
                    0.0f,
                    this,
                    Looper.getMainLooper()
                )
            }
        } catch (e: SecurityException) {
            // Handled via permission check in Activity
        }

        handler.post(imuSampler)
    }

    fun stop() {
        isStreaming = false
        handler.removeCallbacks(imuSampler)
        sensorManager.unregisterListener(this)
        try {
            locationManager.removeUpdates(this)
        } catch (e: SecurityException) {
            // Ignore
        }
    }

    // --- SensorEventListener ---
    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                rawAx = event.values[0]
                rawAy = event.values[1]
                rawAz = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                rawGx = event.values[0]
                rawGy = event.values[1]
                rawGz = event.values[2]
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // --- LocationListener ---
    override fun onLocationChanged(location: Location) {
        listener.onGnssLocation(location)
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}
