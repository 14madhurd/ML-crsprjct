package com.xlr8.deadreckoning.engine

/**
 * 1D Kalman Filter for Position Sensor Fusion (Phase 4).
 * Fuses 10Hz AI Dead Reckoning speed steps with 1Hz noisy GNSS.
 */
class PositionKalmanFilter(
    private var processNoiseQ: Double = 0.5,
    private var measurementNoiseR: Double = 4.0 // 2.0m standard deviation squared
) {
    var stateX: Double = 0.0
        private set
    private var covarianceP: Double = 10.0

    fun reset(initialPosition: Double = 0.0) {
        stateX = initialPosition
        covarianceP = 10.0
    }

    /**
     * Prediction step: Propagates state with AI displacement u = v_ai * dt
     */
    fun predict(aiDisplacement: Double) {
        stateX += aiDisplacement
        covarianceP += processNoiseQ
    }

    /**
     * Update step: Fuses GNSS measurement when available.
     * Skips update when GNSS is in outage/blackout (deficit handling).
     */
    fun update(gnssMeasurement: Double) {
        val kalmanGain = covarianceP / (covarianceP + measurementNoiseR)
        stateX += kalmanGain * (gnssMeasurement - stateX)
        covarianceP *= (1.0 - kalmanGain)
    }
}

/**
 * 1D Responsive Kalman Filter for Velocity Smoothing (Phase 6).
 * Reduces high-frequency AI speed estimation jitter.
 */
class VelocityKalmanFilter(
    initialVelocity: Float = 0.0f,
    private val processNoiseQ: Float = 2.0f,
    private val measurementNoiseR: Float = 0.5f
) {
    var stateEstimate: Float = initialVelocity
        private set
    private var errorCovariance: Float = 1.0f

    fun reset(initialVelocity: Float = 0.0f) {
        stateEstimate = initialVelocity
        errorCovariance = 1.0f
    }

    fun update(aiSpeedMeasurement: Float): Float {
        val predictedState = stateEstimate
        val predictedErrorCov = errorCovariance + processNoiseQ

        val kalmanGain = predictedErrorCov / (predictedErrorCov + measurementNoiseR)
        stateEstimate = predictedState + kalmanGain * (aiSpeedMeasurement - predictedState)
        errorCovariance = (1.0f - kalmanGain) * predictedErrorCov

        // Speed cannot be negative in standard forward automotive driving
        if (stateEstimate < 0.0f) stateEstimate = 0.0f
        return stateEstimate
    }
}
