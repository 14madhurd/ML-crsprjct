package com.xlr8.deadreckoning.gnss

import android.annotation.SuppressLint
import android.content.Context
import android.location.GnssStatus
import android.location.Location
import android.location.LocationManager
import android.os.Handler
import android.os.Looper

interface GnssOutageListener {
    fun onGnssOutageStarted(reason: String)
    fun onGnssSignalRestored(location: Location)
    fun onOutageTick(durationSeconds: Int)
    fun onSatelliteStatusUpdate(usedInFix: Int, totalInView: Int)
}

/**
 * Fully Automatic GNSS Outage Detector.
 *
 * Detects satellite loss via Android GnssStatus hardware callback
 * and an active 1.5-second satellite fix timeout watchdog.
 * Operates 100% autonomously without any manual buttons.
 */
class GnssOutageDetector(
    private val context: Context,
    private val outageThresholdMs: Long = 1800L,
    private val listener: GnssOutageListener
) {

    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    private var lastFixTimestamp: Long = 0L
    private var isCurrentlyInOutage: Boolean = false
    private var outageStartTimestamp: Long = 0L

    private var satellitesUsed: Int = 0
    private var satellitesTotal: Int = 0

    private val handler = Handler(Looper.getMainLooper())
    private var isRunning = false

    private val gnssStatusCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(status: GnssStatus) {
            satellitesTotal = status.satelliteCount
            satellitesUsed = (0 until status.satelliteCount).count { status.usedInFix(it) }

            listener.onSatelliteStatusUpdate(satellitesUsed, satellitesTotal)

            // If zero satellites are used in fix, satellite signal has vanished (tunnel / indoor)
            if (satellitesUsed == 0 && !isCurrentlyInOutage && lastFixTimestamp > 0L) {
                val elapsed = System.currentTimeMillis() - lastFixTimestamp
                if (elapsed > 1000L) {
                    triggerOutage("0 Satellites in Fix (Tunnel / Obstruction)")
                }
            }
        }
    }

    private val watchdogRunnable = object : Runnable {
        override fun run() {
            if (!isRunning) return
            val now = System.currentTimeMillis()

            if (lastFixTimestamp > 0L) {
                val elapsedSinceLastFix = now - lastFixTimestamp
                val isGpsEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)

                if (!isGpsEnabled) {
                    if (!isCurrentlyInOutage) triggerOutage("GPS Hardware Disabled")
                } else if (elapsedSinceLastFix > outageThresholdMs) {
                    if (!isCurrentlyInOutage) {
                        triggerOutage("Signal Lost (${elapsedSinceLastFix}ms timeout)")
                    } else {
                        val durationSec = ((now - outageStartTimestamp) / 1000).toInt()
                        listener.onOutageTick(durationSec)
                    }
                }
            }

            handler.postDelayed(this, 500L)
        }
    }

    @SuppressLint("MissingPermission")
    fun start() {
        if (isRunning) return
        isRunning = true
        lastFixTimestamp = System.currentTimeMillis()

        try {
            locationManager.registerGnssStatusCallback(gnssStatusCallback, handler)
        } catch (e: SecurityException) {
            // Handled via permission flow
        }

        handler.post(watchdogRunnable)
    }

    fun stop() {
        isRunning = false
        handler.removeCallbacks(watchdogRunnable)
        try {
            locationManager.unregisterGnssStatusCallback(gnssStatusCallback)
        } catch (e: Exception) {
            // Ignore
        }
        isCurrentlyInOutage = false
    }

    /**
     * Called whenever a fresh location arrives from GPS or FusedLocationProvider.
     */
    fun onLocationReceived(location: Location) {
        lastFixTimestamp = System.currentTimeMillis()

        if (isCurrentlyInOutage) {
            restoreSignal(location)
        }
    }

    private fun triggerOutage(reason: String) {
        if (isCurrentlyInOutage) return
        isCurrentlyInOutage = true
        outageStartTimestamp = System.currentTimeMillis()
        listener.onGnssOutageStarted(reason)
    }

    private fun restoreSignal(location: Location) {
        isCurrentlyInOutage = false
        listener.onGnssSignalRestored(location)
    }

    fun isOutage(): Boolean = isCurrentlyInOutage
}
