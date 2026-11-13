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
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

interface SensorStreamListener {
    fun onImuFrame(ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float)
    fun onGnssLocation(location: Location)
}

/**
 * Manages IMU sensors at 10Hz and dual-pipeline GNSS positioning
 * (Google Play Services Fused Location + Hardware GPS Provider).
 */
class LocationTracker(
    private val context: Context,
    private val listener: SensorStreamListener
) : SensorEventListener, LocationListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private val fusedLocationClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private var rawAx = 0f; private var rawAy = 0f; private var rawAz = 9.81f
    private var rawGx = 0f; private var rawGy = 0f; private var rawGz = 0f

    private val handler = Handler(Looper.getMainLooper())
    private var isStreaming = false

    private val fusedLocationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val location = result.lastLocation ?: return
            listener.onGnssLocation(location)
        }
    }

    // 10Hz synced IMU dispatcher (100ms ticks)
    private val imuSampler = object : Runnable {
        override fun run() {
            if (!isStreaming) return
            listener.onImuFrame(rawAx, rawAy, rawAz, rawGx, rawGy, rawGz)
            handler.postDelayed(this, 100L)
        }
    }

    @SuppressLint("MissingPermission")
    fun start() {
        if (isStreaming) return
        isStreaming = true

        // 1. Register hardware sensors
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        // 2. Immediate Last Known Location query (instant map pin on launch)
        fusedLocationClient.lastLocation.addOnSuccessListener { loc ->
            if (loc != null) listener.onGnssLocation(loc)
        }

        val lastGps = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        val lastNet = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
        val bestLast = lastGps ?: lastNet
        if (bestLast != null) {
            listener.onGnssLocation(bestLast)
        }

        // 3. Fused Location continuous updates (500ms interval for high response)
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 500L)
            .setMinUpdateIntervalMillis(250L)
            .build()

        try {
            fusedLocationClient.requestLocationUpdates(
                request,
                fusedLocationCallback,
                Looper.getMainLooper()
            )
        } catch (e: SecurityException) {
            // Handled via permission flow
        }

        // 4. Fallback hardware GPS provider
        try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    500L,
                    0.0f,
                    this,
                    Looper.getMainLooper()
                )
            }
        } catch (e: SecurityException) {
            // Handled via permission flow
        }

        handler.post(imuSampler)
    }

    fun stop() {
        isStreaming = false
        handler.removeCallbacks(imuSampler)
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(fusedLocationCallback)
        try {
            locationManager.removeUpdates(this)
        } catch (e: Exception) {
            // Ignore
        }
    }

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

    override fun onLocationChanged(location: Location) {
        listener.onGnssLocation(location)
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}
