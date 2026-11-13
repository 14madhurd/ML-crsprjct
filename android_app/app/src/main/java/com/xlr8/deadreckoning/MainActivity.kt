package com.xlr8.deadreckoning

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.location.Location
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.preference.PreferenceManager
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.xlr8.deadreckoning.databinding.ActivityMainBinding
import com.xlr8.deadreckoning.engine.DeadReckoningEngine
import com.xlr8.deadreckoning.engine.NavigationTelemetry
import com.xlr8.deadreckoning.gnss.GnssOutageDetector
import com.xlr8.deadreckoning.gnss.GnssOutageListener
import com.xlr8.deadreckoning.gnss.LocationTracker
import com.xlr8.deadreckoning.gnss.SensorStreamListener
import org.osmdroid.config.Configuration
import org.osmdroid.events.MapListener
import org.osmdroid.events.ScrollEvent
import org.osmdroid.events.ZoomEvent
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.CustomZoomButtonsController
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import java.util.Locale

class MainActivity : AppCompatActivity(), SensorStreamListener, GnssOutageListener {

    private lateinit var binding: ActivityMainBinding

    private lateinit var engine: DeadReckoningEngine
    private lateinit var outageDetector: GnssOutageDetector
    private lateinit var locationTracker: LocationTracker

    private var isNavigating = false

    // Map Overlays
    private var vehicleMarker: Marker? = null
    private var gnssPolyline: Polyline? = null
    private var drPolyline: Polyline? = null
    private var lastKnownGeoPoint: GeoPoint? = null
    private var autoFollowCamera = true

    private var currentGnssAccuracy = 0.0f
    private var lastPlottedGeoPoint: GeoPoint? = null

    private val navHandler = Handler(Looper.getMainLooper())
    private val navTickRunnable = object : Runnable {
        override fun run() {
            if (!isNavigating) return
            val isOutage = outageDetector.isOutage()
            val telemetry = engine.tick(isOutage, currentGnssAccuracy)
            renderTelemetry(telemetry)
            navHandler.postDelayed(this, 100L) // 10Hz
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val fineGranted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] == true
        val coarseGranted = permissions[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        if (fineGranted || coarseGranted) {
            startAutonomousNavigation()
        } else {
            Toast.makeText(this, "High-precision location required for map navigation.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Keep screen awake while navigating (standard automotive UX)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Initialize OSMDroid
        Configuration.getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this))
        Configuration.getInstance().userAgentValue = packageName

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initEngines()
        setupMapView()
        setupControls()

        checkPermissionsAndStart()
    }

    private fun initEngines() {
        engine = DeadReckoningEngine(this)
        outageDetector = GnssOutageDetector(this, outageThresholdMs = 1800L, listener = this)
        locationTracker = LocationTracker(this, this)
    }

    private fun setupMapView() {
        binding.mapView.apply {
            setTileSource(TileSourceFactory.MAPNIK)
            zoomController.setVisibility(CustomZoomButtonsController.Visibility.NEVER)
            setMultiTouchControls(true)
            controller.setZoom(18.0)

            addMapListener(object : MapListener {
                override fun onScroll(event: ScrollEvent?): Boolean = false
                override fun onZoom(event: ZoomEvent?): Boolean = false
            })
        }

        // Real-Time Polyline Trails (High-visibility 14f stroke width)
        gnssPolyline = Polyline(binding.mapView).apply {
            outlinePaint.color = ContextCompat.getColor(this@MainActivity, R.color.route_gnss)
            outlinePaint.strokeWidth = 14f
        }
        drPolyline = Polyline(binding.mapView).apply {
            outlinePaint.color = ContextCompat.getColor(this@MainActivity, R.color.route_dead_reckoning)
            outlinePaint.strokeWidth = 14f
        }

        binding.mapView.overlays.add(gnssPolyline)
        binding.mapView.overlays.add(drPolyline)

        // Vehicle Directional Marker
        vehicleMarker = Marker(binding.mapView).apply {
            icon = ContextCompat.getDrawable(this@MainActivity, R.drawable.ic_navigation_arrow)
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
            title = "Vehicle"
        }
        binding.mapView.overlays.add(vehicleMarker)
    }

    private fun setupControls() {
        binding.fabRecenter.setOnClickListener {
            autoFollowCamera = true
            lastKnownGeoPoint?.let {
                binding.mapView.controller.animateTo(it)
            }
            Toast.makeText(this, "Centering map on vehicle", Toast.LENGTH_SHORT).show()
        }

        binding.fabClearTrack.setOnClickListener {
            gnssPolyline?.actualPoints?.clear()
            drPolyline?.actualPoints?.clear()
            binding.mapView.invalidate()
            Toast.makeText(this, "Track polyline cleared", Toast.LENGTH_SHORT).show()
        }
    }

    private fun checkPermissionsAndStart() {
        val fineLocation = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
        if (fineLocation == PackageManager.PERMISSION_GRANTED) {
            startAutonomousNavigation()
        } else {
            permissionLauncher.launch(
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                )
            )
        }
    }

    private fun startAutonomousNavigation() {
        if (isNavigating) return
        isNavigating = true

        engine.start()
        locationTracker.start()
        outageDetector.start()

        navHandler.post(navTickRunnable)

        updateStatusUI(isGnssOutage = false, reason = "Acquiring Satellites")
    }

    // --- SensorStreamListener Callbacks ---
    override fun onImuFrame(ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float) {
        // Linear acceleration is handled by IMUSensorManager
    }

    override fun onGnssLocation(location: Location) {
        if (!isNavigating) return
        currentGnssAccuracy = location.accuracy

        outageDetector.onLocationReceived(location)

        if (!outageDetector.isOutage()) {
            engine.onGnssLocationUpdate(location)
        }

        val geoPoint = GeoPoint(location.latitude, location.longitude)
        lastKnownGeoPoint = geoPoint
        if (lastPlottedGeoPoint == null) {
            binding.mapView.controller.setCenter(geoPoint)
            vehicleMarker?.position = geoPoint
            binding.mapView.invalidate()
        }
    }

    // --- GnssOutageListener Callbacks (Fully Automatic) ---
    override fun onGnssOutageStarted(reason: String) {
        runOnUiThread {
            updateStatusUI(isGnssOutage = true, reason = reason)
        }
    }

    override fun onGnssSignalRestored(location: Location) {
        runOnUiThread {
            updateStatusUI(isGnssOutage = false, reason = "Signal Restored")
        }
    }

    override fun onOutageTick(durationSeconds: Int) {
        engine.incrementOutageSecond()
    }

    override fun onSatelliteStatusUpdate(usedInFix: Int, totalInView: Int) {
        runOnUiThread {
            if (outageDetector.isOutage()) {
                binding.tvGnssFixCount.text = String.format(Locale.US, "Satellites: %d/%d (Lost)", usedInFix, totalInView)
            } else {
                binding.tvGnssFixCount.text = String.format(Locale.US, "Satellites: %d/%d in Fix", usedInFix, totalInView)
            }
        }
    }

    private fun updateStatusUI(isGnssOutage: Boolean, reason: String) {
        if (isGnssOutage) {
            binding.tvStatusBadge.text = getString(R.string.status_dr_active)
            binding.tvStatusBadge.setBackgroundResource(R.drawable.bg_badge_outage)
            binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.status_gnss_outage))

            binding.tvSourceSubtitle.text = "GNSS Lost ($reason) - ONNX ZUPT Dead Reckoning Active"
            vehicleMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_navigation_arrow_outage)
        } else {
            binding.tvStatusBadge.text = getString(R.string.status_gnss_active)
            binding.tvStatusBadge.setBackgroundResource(R.drawable.bg_badge_active)
            binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.status_gnss_active))

            binding.tvSourceSubtitle.text = "Connected to Satellites (±%.1fm)".format(currentGnssAccuracy)
            vehicleMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_navigation_arrow)
        }
    }

    private fun renderTelemetry(t: NavigationTelemetry) {
        binding.tvSpeedValue.text = String.format(Locale.US, "%.1f km/h", t.speedKmh)
        binding.tvSpeedMs.text = String.format(Locale.US, "%.1f m/s", t.speedMs)

        binding.tvHeadingValue.text = String.format(Locale.US, "%.0f° %s", t.headingDegrees, getCardinalDirection(t.headingDegrees))
        binding.tvHeadingRate.text = String.format(Locale.US, "%+.1f°/s", t.headingRateDegPerSec)

        if (t.totalTripDistanceMeters >= 1000.0) {
            binding.tvTotalDistance.text = String.format(Locale.US, "%.2f km", t.totalTripDistanceMeters / 1000.0)
        } else {
            binding.tvTotalDistance.text = String.format(Locale.US, "%.0f m", t.totalTripDistanceMeters)
        }

        binding.tvDrDistance.text = String.format(Locale.US, "%.1f m", t.drDistanceMeters)
        binding.tvAccuracyValue.text = String.format(Locale.US, "± %.1f m", t.accuracyMeters)

        if (t.latitude != 0.0 && t.longitude != 0.0) {
            val currentPoint = GeoPoint(t.latitude, t.longitude)
            lastKnownGeoPoint = currentPoint

            vehicleMarker?.position = currentPoint
            vehicleMarker?.rotation = t.headingDegrees

            val lastP = lastPlottedGeoPoint
            val distanceSinceLast = if (lastP != null) currentPoint.distanceToAsDouble(lastP) else 100.0

            if (distanceSinceLast >= 0.8) {
                if (t.isGnssActive) {
                    gnssPolyline?.addPoint(currentPoint)
                } else {
                    drPolyline?.addPoint(currentPoint)
                }
                lastPlottedGeoPoint = currentPoint
            }

            if (autoFollowCamera) {
                binding.mapView.controller.animateTo(currentPoint)
            }
            binding.mapView.invalidate()
        }
    }

    private fun getCardinalDirection(deg: Float): String {
        val directions = arrayOf("N", "NE", "E", "SE", "S", "SW", "W", "NW", "N")
        val index = Math.round((deg % 360) / 45.0).toInt()
        return directions[index]
    }

    override fun onResume() {
        super.onResume()
        binding.mapView.onResume()
    }

    override fun onPause() {
        super.onPause()
        binding.mapView.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        isNavigating = false
        navHandler.removeCallbacks(navTickRunnable)
        engine.stop()
        locationTracker.stop()
        outageDetector.stop()
    }
}
