package com.xlr8.deadreckoning

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.location.Location
import android.os.Bundle
import android.preference.PreferenceManager
import android.view.View
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
    private var isCalibrating = false

    // Map Overlays
    private var vehicleMarker: Marker? = null
    private var gnssPolyline: Polyline? = null
    private var drPolyline: Polyline? = null
    private var lastKnownGeoPoint: GeoPoint? = null
    private var autoCenterMap = true

    private var currentGnssAccuracy = 0.0f

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val fineGranted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] == true
        val coarseGranted = permissions[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        if (fineGranted || coarseGranted) {
            startNavigation()
        } else {
            Toast.makeText(this, getString(R.string.perm_location_rationale), Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OSMDroid configuration before inflating layout
        Configuration.getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this))
        Configuration.getInstance().userAgentValue = packageName

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initEngines()
        setupMapView()
        setupListeners()
    }

    private fun initEngines() {
        engine = DeadReckoningEngine(this)
        outageDetector = GnssOutageDetector(outageThresholdMs = 2000L, listener = this)
        locationTracker = LocationTracker(this, this)
    }

    private fun setupMapView() {
        binding.mapView.apply {
            setTileSource(TileSourceFactory.MAPNIK)
            zoomController.setVisibility(CustomZoomButtonsController.Visibility.NEVER)
            setMultiTouchControls(true)
            controller.setZoom(17.5)

            // Default fallback coordinate: KLE Technological University Campus
            val defaultPoint = GeoPoint(15.4226, 75.0076)
            controller.setCenter(defaultPoint)
            lastKnownGeoPoint = defaultPoint
        }

        // Initialize Polyline tracks
        gnssPolyline = Polyline(binding.mapView).apply {
            outlinePaint.color = ContextCompat.getColor(this@MainActivity, R.color.route_gnss)
            outlinePaint.strokeWidth = 10f
        }
        drPolyline = Polyline(binding.mapView).apply {
            outlinePaint.color = ContextCompat.getColor(this@MainActivity, R.color.route_dead_reckoning)
            outlinePaint.strokeWidth = 10f
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

    private fun setupListeners() {
        binding.btnToggleNavigation.setOnClickListener {
            if (isNavigating) {
                stopNavigation()
            } else {
                checkPermissionsAndStart()
            }
        }

        binding.btnCalibrate.setOnClickListener {
            triggerCalibration()
        }

        binding.switchSimulateOutage.setOnCheckedChangeListener { _, isChecked ->
            outageDetector.isSimulatedOutage = isChecked
        }

        binding.fabRecenter.setOnClickListener {
            autoCenterMap = true
            lastKnownGeoPoint?.let {
                binding.mapView.controller.animateTo(it)
            }
        }
    }

    private fun checkPermissionsAndStart() {
        val fineLocation = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
        if (fineLocation == PackageManager.PERMISSION_GRANTED) {
            startNavigation()
        } else {
            permissionLauncher.launch(
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                )
            )
        }
    }

    private fun startNavigation() {
        isNavigating = true
        binding.btnToggleNavigation.text = getString(R.string.btn_stop)
        binding.btnToggleNavigation.setBackgroundColor(ContextCompat.getColor(this, R.color.status_error))

        locationTracker.start()
        outageDetector.start()

        updateStatusUI(isGnssOutage = false, reason = "Acquiring Fix")
    }

    private fun stopNavigation() {
        isNavigating = false
        binding.btnToggleNavigation.text = getString(R.string.btn_start)
        binding.btnToggleNavigation.setBackgroundColor(ContextCompat.getColor(this, R.color.accent_blue))

        locationTracker.stop()
        outageDetector.stop()
        engine.reset()

        binding.tvStatusBadge.text = getString(R.string.status_ready)
        binding.tvStatusBadge.setBackgroundResource(R.drawable.bg_badge_active)
        binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.status_gnss_active))
        binding.tvSourceSubtitle.text = getString(R.string.source_idle)
    }

    private fun triggerCalibration() {
        isCalibrating = true
        engine.virtualAlignment.resetCalibration()
        binding.tvStatusBadge.text = getString(R.string.status_calibrating)
        binding.tvStatusBadge.setBackgroundResource(R.drawable.bg_badge_outage)
        binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.status_gnss_outage))
        Toast.makeText(this, "Calibrating: keep phone stationary for 5 seconds", Toast.LENGTH_SHORT).show()
    }

    // --- SensorStreamListener Callbacks ---
    override fun onImuFrame(ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float) {
        if (!isNavigating) return

        if (isCalibrating && engine.virtualAlignment.isCalibrated) {
            isCalibrating = false
            Toast.makeText(this, "Virtual alignment calibrated", Toast.LENGTH_SHORT).show()
        }

        val isOutage = outageDetector.isOutage()
        val telemetry = engine.processImuTick(ax, ay, az, gx, gy, gz, isOutage, currentGnssAccuracy)

        runOnUiThread {
            renderTelemetry(telemetry)
        }
    }

    override fun onGnssLocation(location: Location) {
        if (!isNavigating) return
        currentGnssAccuracy = location.accuracy

        outageDetector.onLocationReceived(location)

        if (!outageDetector.isOutage()) {
            engine.onGnssLocationUpdate(location)
            binding.tvGnssFixCount.text = String.format(Locale.US, "GNSS Fix: ±%.1fm", location.accuracy)
        }
    }

    // --- GnssOutageListener Callbacks ---
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

    private fun updateStatusUI(isGnssOutage: Boolean, reason: String) {
        if (isGnssOutage) {
            binding.tvStatusBadge.text = getString(R.string.status_dr_active)
            binding.tvStatusBadge.setBackgroundResource(R.drawable.bg_badge_outage)
            binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.status_gnss_outage))

            binding.tvSourceSubtitle.text = getString(R.string.source_dr)
            binding.tvGnssFixCount.text = "GNSS: Outage ($reason)"

            vehicleMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_navigation_arrow_outage)
        } else {
            binding.tvStatusBadge.text = getString(R.string.status_gnss_active)
            binding.tvStatusBadge.setBackgroundResource(R.drawable.bg_badge_active)
            binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.status_gnss_active))

            binding.tvSourceSubtitle.text = getString(R.string.source_gnss)
            vehicleMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_navigation_arrow)
        }
    }

    private fun renderTelemetry(t: NavigationTelemetry) {
        binding.tvSpeedValue.text = String.format(Locale.US, "%.1f km/h", t.speedKmh)
        binding.tvSpeedMs.text = String.format(Locale.US, "%.1f m/s", t.speedMs)

        binding.tvHeadingValue.text = String.format(Locale.US, "%.0f° %s", t.headingDegrees, getCardinalDirection(t.headingDegrees))
        binding.tvHeadingRate.text = String.format(Locale.US, "%+.1f°/s", t.headingRateDegPerSec)

        binding.tvDrDistance.text = String.format(Locale.US, "%.1f m", t.drDistanceMeters)
        binding.tvDrTime.text = String.format(Locale.US, "Outage: %ds", t.outageDurationSeconds)

        binding.tvAccuracyValue.text = String.format(Locale.US, "± %.1f m", t.accuracyMeters)
        binding.tvStatusDetail.text = if (t.isGnssActive) "Satellite Fix" else "Dead Reckoning"

        // Update Map Marker and Path
        if (t.latitude != 0.0 && t.longitude != 0.0) {
            val currentPoint = GeoPoint(t.latitude, t.longitude)
            lastKnownGeoPoint = currentPoint

            vehicleMarker?.position = currentPoint
            vehicleMarker?.rotation = t.headingDegrees

            if (t.isGnssActive) {
                gnssPolyline?.addPoint(currentPoint)
            } else {
                drPolyline?.addPoint(currentPoint)
            }

            if (autoCenterMap) {
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
        stopNavigation()
    }
}
