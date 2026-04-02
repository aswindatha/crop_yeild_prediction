package com.example.crop_wise

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

class LocationDataManager(private val context: Context) {
    
    private val apiKey = "your_api_key_here"
    private val TAG = "LocationDataManager"
    
    data class LocationData(
        val latitude: Double,
        val longitude: Double,
        val address: String? = null,
        val weather: WeatherData? = null,
        val timeZone: String? = null
    )
    
    data class WeatherData(
        val temperature: Double,
        val condition: String,
        val humidity: Int,
        val windSpeed: Double
    )

    suspend fun fetchLocationData(latitude: Double, longitude: Double): Result<LocationData> {
        return try {
            withContext(Dispatchers.IO) {
                val address = fetchGeocodingData(latitude, longitude)
                val weather = fetchWeatherData(latitude, longitude)
                val timeZone = fetchTimeZoneData(latitude, longitude)
                
                Result.success(
                    LocationData(
                        latitude = latitude,
                        longitude = longitude,
                        address = address,
                        weather = weather,
                        timeZone = timeZone
                    )
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching location data", e)
            Result.failure(e)
        }
    }

    private suspend fun fetchGeocodingData(latitude: Double, longitude: Double): String? {
        return try {
            val url = URL(
                "https://maps.googleapis.com/maps/api/geocode/json?latlng=$latitude,$longitude&key=$apiKey"
            )
            
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 10000
            connection.readTimeout = 10000
            
            if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                val response = BufferedReader(InputStreamReader(connection.inputStream))
                    .use { it.readText() }
                
                val json = JSONObject(response)
                val results = json.getJSONArray("results")
                
                if (results.length() > 0) {
                    results.getJSONObject(0).getString("formatted_address")
                } else null
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "Geocoding error", e)
            null
        }
    }

    private suspend fun fetchWeatherData(latitude: Double, longitude: Double): WeatherData? {
        return try {
            val url = URL(
                "https://api.openweathermap.org/data/2.5/weather?lat=$latitude&lon=$longitude&appid=$apiKey&units=metric"
            )
            
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 10000
            connection.readTimeout = 10000
            
            if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                val response = BufferedReader(InputStreamReader(connection.inputStream))
                    .use { it.readText() }
                
                val json = JSONObject(response)
                val main = json.getJSONObject("main")
                val weather = json.getJSONArray("weather").getJSONObject(0)
                val wind = json.getJSONObject("wind")
                
                WeatherData(
                    temperature = main.getDouble("temp"),
                    condition = weather.getString("description"),
                    humidity = main.getInt("humidity"),
                    windSpeed = wind.getDouble("speed")
                )
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "Weather API error", e)
            null
        }
    }

    private suspend fun fetchTimeZoneData(latitude: Double, longitude: Double): String? {
        return try {
            val timestamp = System.currentTimeMillis() / 1000
            val url = URL(
                "https://maps.googleapis.com/maps/api/timezone/json?location=$latitude,$longitude&timestamp=$timestamp&key=$apiKey"
            )
            
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 10000
            connection.readTimeout = 10000
            
            if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                val response = BufferedReader(InputStreamReader(connection.inputStream))
                    .use { it.readText() }
                
                val json = JSONObject(response)
                json.getString("timeZoneName")
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "Timezone API error", e)
            null
        }
    }
}
