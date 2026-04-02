package com.example.map_test

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.embedding.android.FlutterFragmentActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.MapsInitializer
import com.google.android.gms.maps.OnMapsSdkInitializedCallback
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.MarkerOptions

class MainActivity: FlutterFragmentActivity(), OnMapsSdkInitializedCallback {
    
    private val CHANNEL = "map_test/location_picker"
    private val TAG = "MapTest"
    private var pendingResult: MethodChannel.Result? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate - Initializing Maps SDK with LATEST renderer")
        MapsInitializer.initialize(applicationContext, MapsInitializer.Renderer.LATEST, this)
    }
    
    override fun onMapsSdkInitialized(renderer: MapsInitializer.Renderer) {
        Log.d(TAG, "Maps SDK initialized with renderer: $renderer")
    }
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        Log.d(TAG, "Registering method channel")
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            Log.d(TAG, "Method called: ${call.method}")
            when (call.method) {
                "showLocationPicker" -> {
                    pendingResult = result
                    showLocationPicker()
                }
                else -> result.notImplemented()
            }
        }
    }
    
    private fun showLocationPicker() {
        val defaultLocation = LatLng(28.6139, 77.2090)
        
        val mapFragment = SupportMapFragment()
        mapFragment.getMapAsync { googleMap: GoogleMap ->
            Log.d(TAG, "Map ready")
            googleMap.moveCamera(CameraUpdateFactory.newLatLngZoom(defaultLocation, 12f))
            googleMap.addMarker(
                MarkerOptions()
                    .position(defaultLocation)
                    .draggable(true)
                    .title("Drag to select")
            )
        }
        
        supportFragmentManager.beginTransaction()
            .add(android.R.id.content, mapFragment, "map")
            .addToBackStack("map")
            .commit()
            
        Handler(Looper.getMainLooper()).postDelayed({
            supportFragmentManager.popBackStack()
            pendingResult?.success(mapOf(
                "latitude" to defaultLocation.latitude,
                "longitude" to defaultLocation.longitude
            ))
            pendingResult = null
        }, 5000)
    }
}
