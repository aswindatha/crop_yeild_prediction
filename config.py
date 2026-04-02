# Google Maps APIs Configuration
# Replace YOUR_GOOGLE_MAPS_API_KEY_HERE with your actual API key

# To get a Google Maps API key:
# 1. Go to Google Cloud Console (https://console.cloud.google.com/)
# 2. Create a new project or select an existing one
# 3. Enable these APIs: Geolocation API, Geocoding API, Weather API, Time Zone API
# 4. Create credentials -> API Key
# 5. Copy the API key and replace the placeholder below

GOOGLE_MAPS_API_KEY = "your_api_key_here"

# This single key works for all enabled APIs:
# - Geolocation API (backup location detection)
# - Geocoding API (address lookup) 
# - Weather API (weather data)
# - Time Zone API (timezone context)

# Optional: You can also set this as an environment variable
# In your terminal: export GOOGLE_MAPS_API_KEY="your_actual_api_key_here"
# Then modify server.py to use: os.environ.get('GOOGLE_MAPS_API_KEY', 'YOUR_ACTUAL_API_KEY_HERE')
