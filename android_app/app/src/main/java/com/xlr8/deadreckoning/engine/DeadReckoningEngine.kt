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
 * Core Dead Reckoning Engine.
 * Integrates instant virtual alignment, 10Hz sensor preprocessor,
 * pure Kotlin 1D-CNN neural network, and continuous kinematic positioning.
 */
class DeadReckoningEngine(context: Context) {

    val virtualAlignment = VirtualAlignment()
    val preprocessor = SensorPreprocessor(context)
    val model = ModelInference(context)
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
        const val DT: Float = 0.1f // 10Hz sampling
        const val METERS_PER_LAT_DEGREE = 111139.0
    }

    /**
     * Called whenever a fresh valid GNSS fix arrives from the satellites.
     */
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
            if (dist < 100.0) { // filter GPS teleport jumps
                totalDistanceMeters += dist
            }
        }

        hasPositionFix = true
        drDistanceMeters = 0.0
        outageSeconds = 0
    }

    /**
     * 10Hz sensor tick.
     */
    fun processImuTick(
        ax: Float, ay: Float, az: Float,
        gx: Float, gy: Float, gz: Float,
        isGnssOutage: Boolean,
        gnssAccuracy: Float
    ): NavigationTelemetry {
        // 1. Instant Virtual Alignment (Refines gravity vector when stationary)
        virtualAlignment.processSample(ax, ay, az, currentSpeedMs < 0.5f)

        val alignedAcc = FloatArray(3)
        virtualAlignment.align(ax, ay, az, alignedAcc)

        // 2. Preprocessor (2Hz Butterworth low-pass + Standardizer)
        preprocessor.pushSample(
            alignedAcc[0], alignedAcc[1], alignedAcc[2],
            gz, gy, gx
        )

        val headingRate = gz * 57.2957795f // convert rad/s to deg/s

        // 3. Model Inference & Dead Reckoning Navigation
        if (isGnssOutage && hasPositionFix) {
            val tensor = preprocessor.getTensorData()
            val prediction = model.runInference(tensor)

            // Smooth AI speed with Velocity Kalman Filter
            currentSpeedMs = velocityKalman.update(prediction.speed)

            // Integrate heading using gyroscope yaw rate
            currentHeadingDeg = (currentHeadingDeg + (headingRate * DT)) % 360f
            if (currentHeadingDeg < 0f) currentHeadingDeg += 360f

            // Forward non-holonomic kinematic step
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

    fun setInitialCoordinate(lat: Double, lon: Double, heading: Float = 0f) {
        currentLatitude = lat
        currentLongitude = lon
        currentHeadingDeg = heading
        hasPositionFix = true
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
        preprocessor.reset()
        velocityKalman.reset()
        drDistanceMeters = 0.0
        totalDistanceMeters = 0.0
        outageSeconds = 0
    }
}
