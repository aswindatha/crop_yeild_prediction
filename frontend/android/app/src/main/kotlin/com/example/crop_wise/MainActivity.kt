package com.example.crop_wise

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.embedding.android.FlutterFragmentActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterFragmentActivity() {
    
    private val CHANNEL = "crop_wise/location_picker"
    private val TAG = "MainActivity"
    private var pendingResult: MethodChannel.Result? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "MainActivity onCreate called")
    }
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        Log.d(TAG, "configureFlutterEngine called - registering method channel")
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            Log.d(TAG, "Method call received: ${call.method}")
            when (call.method) {
                "showLocationPicker" -> {
                    Log.d(TAG, "showLocationPicker called")
                    pendingResult = result
                    showLocationPicker()
                }
                else -> {
                    Log.d(TAG, "Method not implemented: ${call.method}")
                    result.notImplemented()
                }
            }
        }
        Log.d(TAG, "Method channel registered successfully")
    }
    
    private fun showLocationPicker() {
        Log.d(TAG, "Creating LocationPickerFragmentNew")
        val callback = object : LocationPickerCallbackNew {
            override fun onLocationSelected(latitude: Double, longitude: Double) {
                Log.d(TAG, "Location selected: $latitude, $longitude")
                // Return location data to Flutter on main thread
                Handler(Looper.getMainLooper()).post {
                    pendingResult?.success(mapOf(
                        "latitude" to latitude,
                        "longitude" to longitude,
                        "locationData" to mapOf<String, Any>()
                    ))
                    pendingResult = null
                }
            }
        }
        
        val fragment = LocationPickerFragmentNew.newInstance(callback)
        fragment.show(supportFragmentManager, "location_picker")
    }
}
