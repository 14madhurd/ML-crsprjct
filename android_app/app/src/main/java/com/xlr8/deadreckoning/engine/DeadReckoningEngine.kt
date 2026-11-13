package com.xlr8.deadreckoning.engine

import android.content.Context
import android.location.Location
import kotlin.math.cos
import kotlin.math.sin

data class NavigationTelemetry(
    val latitude: Double,
    val longitude: Double,
    val speedMs: Float,
    val speedKmh: Float,
    val headingDegrees: Float,
    val headingRateDegPerSec: Float,
    val isGnssActive: Boolean,
    val drDistanceMeters: Double,
    val totalTripDistanceMeters: Double,
    val outageDurationSeconds: Int,
    val accuracyMeters: Float
)

/**
 * Dead Reckoning Engine powered by:
 * 1. IMUSensorManager (Linear Acceleration gravity stripping)
 * 2. VelocityEngine (ONNX Dual-Head 1D-CNN with ZUPT noise gate)
 * 3. 1D Velocity Kalman Filter for smoothing
 */
class DeadReckoningEngine(context: Context) {

    val imuSensorManager = IMUSensorManager(context)
    val velocityEngine = VelocityEngine(context)
    private val velocityKalman = VelocityKalmanFilter(initialVelocity = 0.0f)

    var currentLatitude: Double = 0.0
        private set
    var currentLongitude: Double = 0.0
        private set
    var currentHeadingDeg: Float = 0.0f
        private set
    var currentSpeedMs: Float = 0.0f
        private set
    var drDistanceMeters: Double = 0.0
        private set
    var totalDistanceMeters: Double = 0.0
        private set
    var outageSeconds: Int = 0
        private set

    var hasPositionFix: Boolean = false
        private set

    companion object {
        const val DT: Float = 0.1f // 10Hz tick
        const val METERS_PER_LAT_DEGREE = 111139.0
    }

    fun start() {
        imuSensorManager.startRecording()
    }

    fun stop() {
        imuSensorManager.stopRecording()
    }

    fun onGnssLocationUpdate(location: Location) {
        val prevLat = currentLatitude
        val prevLon = currentLongitude

        currentLatitude = location.latitude
        currentLongitude = location.longitude

        if (location.hasBearing() && location.speed > 0.8f) {
            currentHeadingDeg = location.bearing
        }
        if (location.hasSpeed()) {
            currentSpeedMs = location.speed
            velocityKalman.reset(currentSpeedMs)
        }

        if (hasPositionFix && prevLat != 0.0 && prevLon != 0.0) {
            val dist = calculateDistance(prevLat, prevLon, currentLatitude, currentLongitude)
            if (dist < 100.0) {
                totalDistanceMeters += dist
            }
        }

        hasPositionFix = true
        drDistanceMeters = 0.0
        outageSeconds = 0
    }

    /**
     * Executes 10Hz navigation update.
     */
    fun tick(isGnssOutage: Boolean, gnssAccuracy: Float): NavigationTelemetry {
        var headingRate = 0f

        if (isGnssOutage && hasPositionFix) {
            val windowList = imuSensorManager.sensorWindow.toList()
            val (predSpeed, predHeading) = velocityEngine.predict(windowList)

            // Extract last gyro Z (yaw rate in rad/s)
            val lastFrame = windowList.lastOrNull()
            if (lastFrame != null) {
                headingRate = lastFrame[5] * 57.2957795f
            }

            // Zero Velocity Update (ZUPT) check
            if (predSpeed <= 0f) {
                currentSpeedMs = 0f
                velocityKalman.reset(0f)
            } else {
                currentSpeedMs = velocityKalman.update(predSpeed)

                // Integrate heading from yaw rate
                currentHeadingDeg = (currentHeadingDeg + (headingRate * DT)) % 360f
                if (currentHeadingDeg < 0f) currentHeadingDeg += 360f

                val stepDistance = currentSpeedMs * DT
                drDistanceMeters += stepDistance
                totalDistanceMeters += stepDistance

                val headingRad = Math.toRadians(currentHeadingDeg.toDouble())
                val deltaNorthMeters = stepDistance * cos(headingRad)
                val deltaEastMeters = stepDistance * sin(headingRad)

                val latRad = Math.toRadians(currentLatitude)
                val deltaLat = deltaNorthMeters / METERS_PER_LAT_DEGREE
                val deltaLon = deltaEastMeters / (METERS_PER_LAT_DEGREE * cos(latRad))

                currentLatitude += deltaLat
                currentLongitude += deltaLon
            }
        }

        return NavigationTelemetry(
            latitude = currentLatitude,
            longitude = currentLongitude,
            speedMs = currentSpeedMs,
            speedKmh = currentSpeedMs * 3.6f,
            headingDegrees = currentHeadingDeg,
            headingRateDegPerSec = headingRate,
            isGnssActive = !isGnssOutage,
            drDistanceMeters = drDistanceMeters,
            totalTripDistanceMeters = totalDistanceMeters,
            outageDurationSeconds = outageSeconds,
            accuracyMeters = if (isGnssOutage) (drDistanceMeters * 0.05 + 5.0).toFloat() else gnssAccuracy
        )
    }

    fun incrementOutageSecond() {
        outageSeconds++
    }

    private fun calculateDistance(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2) * sin(dLat / 2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
                sin(dLon / 2) * sin(dLon / 2)
        val c = 2 * kotlin.math.atan2(kotlin.math.sqrt(a), kotlin.math.sqrt(1 - a))
        return 6371000.0 * c
    }

    fun reset() {
        imuSensorManager.stopRecording()
        velocityKalman.reset()
        drDistanceMeters = 0.0
        totalDistanceMeters = 0.0
        outageSeconds = 0
    }
}
