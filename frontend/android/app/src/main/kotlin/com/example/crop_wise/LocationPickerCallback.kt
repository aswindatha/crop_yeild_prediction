package com.example.crop_wise

interface LocationPickerCallback {
    fun onLocationSelected(latitude: Double, longitude: Double)
    fun onLocationPickerDismissed()
}
