# Crop Yield Prediction Flutter App

## Features

- 📍 **Location Services**: Automatically gets user's exact GPS location
- 🌾 **Crop Selection**: Dropdown with 7 crop types (rice, sugarcane, cotton, pulses, millet, groundnut, coconut)
- 🧪 **Nutrient Inputs**: Text fields for phosphorus and potassium with sample placeholders
- 💧 **Irrigation**: Yes/No radio buttons
- 📏 **Farm Size**: Text field with sample placeholder
- ⚙️ **Settings**: Backend API URL configuration with connection test
- 🔄 **Real-time Prediction**: Calls backend API and shows results

## Setup Instructions

### 1. Install Dependencies
```bash
cd frontend
flutter pub get
```

### 2. Run the App
```bash
flutter run
```

### 3. Configure Backend
1. Click the settings button (⚙️) in the top-right
2. Enter your backend API URL (default: http://localhost:5000)
3. Click "Test Connection" to verify
4. Click "Save" to store the URL

## App Structure

### Single Screen Layout:
- **AppBar**: Title + Settings button
- **Location Card**: Shows GPS coordinates and status
- **Crop Type Card**: Dropdown for crop selection
- **Nutrient Card**: Phosphorus & Potassium input fields
- **Irrigation Card**: Yes/No radio buttons
- **Farm Size Card**: Farm size input field
- **Predict Button**: Calls backend API

### Settings Popup:
- API URL text input
- Test Connection button (curl equivalent)
- Save button

## API Integration

### Prediction Request:
```json
{
  "latitude": 13.05,
  "longitude": 80.15,
  "crop_type": "rice",
  "phosphorus": 25.0,
  "potassium": 150.0,
  "irrigation_available": 1,
  "farm_size_ha": 3.0
}
```

### Health Check:
```
GET {api_url}/health
```

## Permissions Required

- `INTERNET`: For API calls
- `ACCESS_FINE_LOCATION`: For GPS location
- `ACCESS_COARSE_LOCATION`: For location fallback

## Dependencies

- `http`: API calls
- `geolocator`: GPS location services
- `permission_handler`: Runtime permissions

## Notes

- App automatically requests location permissions on startup
- All input fields have sample placeholders for guidance
- Connection test validates backend availability
- Results show predicted yield with input details
