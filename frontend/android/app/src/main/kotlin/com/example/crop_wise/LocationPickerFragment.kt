package com.example.crop_wise

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import androidx.fragment.app.DialogFragment
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.Marker
import com.google.android.gms.maps.model.MarkerOptions

class LocationPickerFragment : DialogFragment(), OnMapReadyCallback, GoogleMap.OnMarkerDragListener {

    private var googleMap: GoogleMap? = null
    private var currentMarker: Marker? = null
    private var callback: LocationPickerCallback? = null
    
    companion object {
        private const val DEFAULT_LAT = 40.7128  // New York
        private const val DEFAULT_LNG = -74.0060
        private const val DEFAULT_ZOOM = 10f
        
        fun newInstance(callback: LocationPickerCallback): LocationPickerFragment {
            val fragment = LocationPickerFragment()
            fragment.callback = callback
            return fragment
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.location_picker_layout, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        dialog?.window?.setLayout(
            (resources.displayMetrics.widthPixels * 0.9).toInt(),
            (resources.displayMetrics.heightPixels * 0.8).toInt()
        )
        
        val mapFragment = childFragmentManager.findFragmentById(R.id.map_fragment) as SupportMapFragment
        mapFragment.getMapAsync(this)
        
        view.findViewById<Button>(R.id.btn_cancel).setOnClickListener {
            callback?.onLocationPickerDismissed()
            dismiss()
        }
        
        view.findViewById<Button>(R.id.btn_confirm).setOnClickListener {
            currentMarker?.position?.let { position ->
                callback?.onLocationSelected(position.latitude, position.longitude)
                dismiss()
            }
        }
    }

    override fun onMapReady(map: GoogleMap) {
        googleMap = map
        
        val defaultLocation = LatLng(DEFAULT_LAT, DEFAULT_LNG)
        
        map.moveCamera(CameraUpdateFactory.newLatLngZoom(defaultLocation, DEFAULT_ZOOM))
        
        map.setOnMarkerDragListener(this)
        
        currentMarker = map.addMarker(
            MarkerOptions()
                .position(defaultLocation)
                .draggable(true)
                .title("Selected Location")
        )
        
        map.uiSettings.apply {
            isZoomControlsEnabled = true
            isCompassEnabled = true
            isMyLocationButtonEnabled = false
            isMapToolbarEnabled = false
        }
        
        map.mapType = GoogleMap.MAP_TYPE_NORMAL
    }

    override fun onMarkerDragStart(marker: Marker) {
        // Optional: Handle drag start
    }

    override fun onMarkerDrag(marker: Marker) {
        // Optional: Handle drag in progress
    }

    override fun onMarkerDragEnd(marker: Marker) {
        val position = marker.position
        // The final position is captured here
        // You could add visual feedback or validation here
    }

    fun setCallback(callback: LocationPickerCallback) {
        this.callback = callback
    }
}
