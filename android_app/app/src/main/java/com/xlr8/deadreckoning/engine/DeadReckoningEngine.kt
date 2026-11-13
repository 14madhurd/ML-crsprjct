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
    val outageDurationSeconds: Int,
    val accuracyMeters: Float
)

/**
 * Core Dead Reckoning Engine.
 * Integrates Virtual Alignment, Sensor Preprocessing, Dual-Head 1D-CNN Model,
 * Kalman Filter, and Non-Holonomic Kinematic Track Integrator.
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
    var outageSeconds: Int = 0
        private set

    private var hasInitialFix: Boolean = false
    private var lastGyroZ: Float = 0.0f

    companion object {
        const val DT: Float = 0.1f // 10Hz sampling
        const val METERS_PER_LAT_DEGREE = 111139.0
    }

    /**
     * Called whenever a fresh valid GNSS fix arrives from the satellites.
     */
    fun onGnssLocationUpdate(location: Location) {
        currentLatitude = location.latitude
        currentLongitude = location.longitude

        if (location.hasBearing() && location.speed > 1.0f) {
            currentHeadingDeg = location.bearing
        }
        if (location.hasSpeed()) {
            currentSpeedMs = location.speed
            velocityKalman.reset(currentSpeedMs)
        }

        hasInitialFix = true
        drDistanceMeters = 0.0
        outageSeconds = 0
    }

    /**
     * Called every 10Hz tick when IMU sensor samples are ready.
     * @param ax Raw smartphone accelerometer X
     * @param ay Raw smartphone accelerometer Y
     * @param az Raw smartphone accelerometer Z
     * @param gx Gyroscope roll rate (rad/s)
     * @param gy Gyroscope pitch rate (rad/s)
     * @param gz Gyroscope yaw rate (rad/s)
     * @param isGnssOutage Whether the GNSS detector has flagged an outage
     * @return Fresh telemetry update
     */
    fun processImuTick(
        ax: Float, ay: Float, az: Float,
        gx: Float, gy: Float, gz: Float,
        isGnssOutage: Boolean,
        gnssAccuracy: Float
    ): NavigationTelemetry {
        lastGyroZ = gz

        // 1. If calibrating, accumulate gravity
        if (!virtualAlignment.isCalibrated) {
            virtualAlignment.addCalibrationSample(ax, ay, az)
        }

        // 2. Virtual Alignment
        val alignedAcc = FloatArray(3)
        virtualAlignment.align(ax, ay, az, alignedAcc)

        // 3. Preprocessor (2Hz Butterworth low-pass + Standardizer + Sliding window)
        preprocessor.pushSample(
            alignedAcc[0], alignedAcc[1], alignedAcc[2],
            gz, gy, gx
        )

        // 4. Model Inference & Kinematics
        var headingRate = gz * 57.2957795f // convert rad/s to deg/s

        if (isGnssOutage && hasInitialFix && preprocessor.isWindowReady()) {
            val tensor = preprocessor.getTensorData()
            val prediction = model.runInference(tensor)

            // Smooth AI speed with Velocity Kalman Filter (Phase 6)
            currentSpeedMs = velocityKalman.update(prediction.speed)

            // Update heading from gyroscope yaw rate
            currentHeadingDeg = (currentHeadingDeg + (headingRate * DT)) % 360f
            if (currentHeadingDeg < 0f) currentHeadingDeg += 360f

            // Forward non-holonomic kinematic step
            val stepDistance = currentSpeedMs * DT
            drDistanceMeters += stepDistance

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
        hasInitialFix = true
    }

    fun reset() {
        preprocessor.reset()
        velocityKalman.reset()
        drDistanceMeters = 0.0
        outageSeconds = 0
        hasInitialFix = false
    }
}
