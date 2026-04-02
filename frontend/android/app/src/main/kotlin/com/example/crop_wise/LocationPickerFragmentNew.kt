package com.example.crop_wise

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.DialogFragment
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.Marker
import com.google.android.gms.maps.model.MarkerOptions

interface LocationPickerCallbackNew {
    fun onLocationSelected(latitude: Double, longitude: Double)
}

class LocationPickerFragmentNew : DialogFragment(), OnMapReadyCallback, GoogleMap.OnMarkerDragListener {

    private val TAG = "LocationPickerFragment"
    private var googleMap: GoogleMap? = null
    private var currentMarker: Marker? = null
    private var selectedLocation: LatLng? = null
    private var callback: LocationPickerCallbackNew? = null

    // Default coordinates - Delhi, India
    private val defaultLocation = LatLng(28.6139, 77.2090)
    private val defaultZoom = 12.0f

    companion object {
        fun newInstance(callback: LocationPickerCallbackNew): LocationPickerFragmentNew {
            val fragment = LocationPickerFragmentNew()
            fragment.callback = callback
            return fragment
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setStyle(STYLE_NO_FRAME, android.R.style.Theme_DeviceDefault_Light_NoActionBar_Fullscreen)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.location_picker_new, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        Log.d(TAG, "onViewCreated called")

        val mapFragment = childFragmentManager.findFragmentById(R.id.map_fragment) as SupportMapFragment
        mapFragment.getMapAsync(this)
        Log.d(TAG, "getMapAsync called")

        val confirmButton = view.findViewById<Button>(R.id.confirm_location_btn)
        confirmButton.setOnClickListener {
            selectedLocation?.let { location ->
                callback?.onLocationSelected(location.latitude, location.longitude)
                dismiss()
            } ?: run {
                // If no drag happened, use default location
                callback?.onLocationSelected(defaultLocation.latitude, defaultLocation.longitude)
                dismiss()
            }
        }
    }

    override fun onMapReady(map: GoogleMap) {
        Log.d(TAG, "onMapReady called - Map is ready")
        googleMap = map

        // Configure map UI
        map.uiSettings.apply {
            isZoomControlsEnabled = true
            isCompassEnabled = true
            isMyLocationButtonEnabled = false
            isMapToolbarEnabled = true
        }
        Log.d(TAG, "Map UI configured")

        // Move camera to default location (Delhi)
        map.moveCamera(CameraUpdateFactory.newLatLngZoom(defaultLocation, defaultZoom))

        // Add draggable marker with user hint
        currentMarker = map.addMarker(
            MarkerOptions()
                .position(defaultLocation)
                .draggable(true)
                .title("Long-press to drag")
                .snippet("Hold and move to adjust")
        )

        // Set initial selected location
        selectedLocation = defaultLocation

        // Set drag listener
        map.setOnMarkerDragListener(this)

        // Update coordinates display
        updateCoordinatesDisplay(defaultLocation)
    }

    override fun onMarkerDragStart(marker: Marker) {
        // Prevent Flutter ScrollView/Tab from stealing touch events
        view?.parent?.requestDisallowInterceptTouchEvent(true)
        view?.findViewById<TextView>(R.id.coordinates_text)?.text = "Dragging..."
    }

    override fun onMarkerDrag(marker: Marker) {
        // Update coordinates in real-time
        updateCoordinatesDisplay(marker.position)
    }

    override fun onMarkerDragEnd(marker: Marker) {
        // Allow Flutter to intercept touch events again
        view?.parent?.requestDisallowInterceptTouchEvent(false)
        // Save final position
        selectedLocation = marker.position
        updateCoordinatesDisplay(marker.position)
        Log.d(TAG, "Marker drag ended at: ${marker.position.latitude}, ${marker.position.longitude}")
    }

    private fun updateCoordinatesDisplay(location: LatLng) {
        val coordinatesText = view?.findViewById<TextView>(R.id.coordinates_text)
        coordinatesText?.text = String.format("%.6f, %.6f", location.latitude, location.longitude)
    }

    fun setCallback(callback: LocationPickerCallbackNew) {
        this.callback = callback
    }
}
