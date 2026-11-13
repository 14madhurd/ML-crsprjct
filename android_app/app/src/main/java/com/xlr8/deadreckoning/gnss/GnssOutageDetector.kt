package com.xlr8.deadreckoning.gnss

import android.location.Location
import android.os.Handler
import android.os.Looper

interface GnssOutageListener {
    fun onGnssOutageStarted(reason: String)
    fun onGnssSignalRestored(location: Location)
    fun onOutageTick(durationSeconds: Int)
}

/**
 * Monitors satellite fixes in real time.
 * Automatically flags GNSS outage if fixes cease for > 2.0 seconds
 * or when tunnel simulation is toggled.
 */
class GnssOutageDetector(
    private val outageThresholdMs: Long = 2000L,
    private val listener: GnssOutageListener
) {

    private var lastFixTimestamp: Long = 0L
    private var isCurrentlyInOutage: Boolean = false
    private var outageStartTimestamp: Long = 0L

    var isSimulatedOutage: Boolean = false
        set(value) {
            field = value
            if (value && !isCurrentlyInOutage) {
                triggerOutage("Manual Tunnel Blackout Simulation")
            } else if (!value && isCurrentlyInOutage && lastFixTimestamp > 0L) {
                // If we have a recent fix, restore immediately
                if (System.currentTimeMillis() - lastFixTimestamp < outageThresholdMs) {
                    restoreSignal(null)
                }
            }
        }

    private val handler = Handler(Looper.getMainLooper())
    private var isRunning = false

    private val watchdogRunnable = object : Runnable {
        override fun run() {
            if (!isRunning) return

            val now = System.currentTimeMillis()

            if (isSimulatedOutage) {
                if (!isCurrentlyInOutage) {
                    triggerOutage("Simulated Outage Active")
                } else {
                    val durationSec = ((now - outageStartTimestamp) / 1000).toInt()
                    listener.onOutageTick(durationSec)
                }
            } else if (lastFixTimestamp > 0L) {
                val elapsedSinceLastFix = now - lastFixTimestamp
                if (elapsedSinceLastFix > outageThresholdMs) {
                    if (!isCurrentlyInOutage) {
                        triggerOutage("GNSS Fix Timeout (${elapsedSinceLastFix}ms)")
                    } else {
                        val durationSec = ((now - outageStartTimestamp) / 1000).toInt()
                        listener.onOutageTick(durationSec)
                    }
                }
            }

            handler.postDelayed(this, 500)
        }
    }

    fun start() {
        isRunning = true
        lastFixTimestamp = System.currentTimeMillis() // grace period
        handler.post(watchdogRunnable)
    }

    fun stop() {
        isRunning = false
        handler.removeCallbacks(watchdogRunnable)
        isCurrentlyInOutage = false
    }

    /**
     * Feed live location from Android LocationManager / FusedLocationProviderClient.
     */
    fun onLocationReceived(location: Location) {
        if (isSimulatedOutage) {
            // Keep record of background fix, but ignore for active state
            return
        }

        lastFixTimestamp = System.currentTimeMillis()

        if (isCurrentlyInOutage) {
            restoreSignal(location)
        }
    }

    private fun triggerOutage(reason: String) {
        isCurrentlyInOutage = true
        outageStartTimestamp = System.currentTimeMillis()
        listener.onGnssOutageStarted(reason)
    }

    private fun restoreSignal(location: Location?) {
        isCurrentlyInOutage = false
        if (location != null) {
            listener.onGnssSignalRestored(location)
        }
    }

    fun isOutage(): Boolean = isCurrentlyInOutage
}
