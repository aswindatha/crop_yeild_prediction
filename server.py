from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
# Allow all origins for development
CORS(app, resources={r"/*": {"origins": "*"}})

# ML Model Classes (same as in predict.py)
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], n_gaussians=5):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.n_gaussians = n_gaussians
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.pi_net = nn.Linear(prev_dim, n_gaussians)
        self.mu_net = nn.Linear(prev_dim, n_gaussians)
        self.sigma_net = nn.Linear(prev_dim, n_gaussians)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        pi = torch.softmax(self.pi_net(features), dim=1)
        mu = self.mu_net(features)
        sigma = F.softplus(self.sigma_net(features)) + 1e-6
        return pi, mu, sigma

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_projection(x)
        return x.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

import threading
import time
from datetime import datetime

# Global variables for models and preprocessors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mdn_model = None
transformer_model = None
preprocessors = None

# Agentic state management
agent_state = {
    'monitoring_active': False,
    'monitored_locations': [],
    'alerts': [],
    'last_check': None
}

def load_models():
    """Load trained models and preprocessors"""
    global mdn_model, transformer_model, preprocessors
    
    try:
        import os
        model_dir = 'models'
        mdn_path = os.path.join(model_dir, 'mdn_crop_yield_model.pth')
        transformer_path = os.path.join(model_dir, 'transformer_crop_yield_model.pth')
        
        # Check if model files exist
        if not os.path.exists(mdn_path):
            raise FileNotFoundError(f"MDN model file not found at: {os.path.abspath(mdn_path)}")
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"Transformer model file not found at: {os.path.abspath(transformer_path)}")
        
        print(f"Loading MDN model from: {os.path.abspath(mdn_path)}")
        print(f"Loading Transformer model from: {os.path.abspath(transformer_path)}")
        
        # Add safe globals for scikit-learn objects
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        import torch.serialization
        
        # Allow scikit-learn objects that might be in the checkpoint
        torch.serialization.add_safe_globals([StandardScaler, LabelEncoder])
        
        # Load MDN model with weights_only=False to handle scikit-learn objects
        print("Loading MDN model...")
        mdn_checkpoint = torch.load(mdn_path, map_location=device, weights_only=False)
        input_dim = len(mdn_checkpoint['feature_columns'])
        print(f"MDN model input dimension: {input_dim}")
        
        mdn_model = MDN(
            input_dim=input_dim,
            hidden_dims=mdn_checkpoint.get('hidden_dims', [256, 128, 64]),
            n_gaussians=mdn_checkpoint.get('n_gaussians', 5)
        ).to(device)
        mdn_model.load_state_dict(mdn_checkpoint['model_state_dict'])
        mdn_model.eval()
        print("MDN model loaded successfully!")
        
        # Load Transformer model with weights_only=False to handle scikit-learn objects
        print("Loading Transformer model...")
        transformer_checkpoint = torch.load(transformer_path, map_location=device, weights_only=False)
        
        transformer_model = TransformerRegressor(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.1
        ).to(device)
        transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
        transformer_model.eval()
        print("Transformer model loaded successfully!")
        
        # Store preprocessors
        required_keys = ['scaler_X', 'scaler_y', 'le_crop', 'le_region', 'feature_columns']
        missing_keys = [key for key in required_keys if key not in mdn_checkpoint]
        if missing_keys:
            raise KeyError(f"Missing required keys in model checkpoint: {', '.join(missing_keys)}")
            
        preprocessors = {key: mdn_checkpoint[key] for key in required_keys}
        print("Preprocessors loaded successfully!")
        
        # Verify model output
        print("Verifying models with test input...")
        test_input = torch.randn(1, input_dim).to(device)
        with torch.no_grad():
            mdn_out = mdn_model(test_input)
            transformer_out = transformer_model(test_input)
        print(f"MDN output shape: {[x.shape for x in mdn_out] if isinstance(mdn_out, tuple) else mdn_out.shape}")
        print(f"Transformer output shape: {transformer_out.shape}")
        
        print("All models loaded and verified successfully!")
        return True
        
    except Exception as e:
        import traceback
        print("\n" + "="*50)
        print("ERROR LOADING MODELS:")
        print(f"Type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        print("="*50 + "\n")
        return False

def autonomous_monitoring_loop():
    """Agentic monitoring loop that takes proactive actions"""
    while agent_state['monitoring_active']:
        try:
            current_time = datetime.now()
            
            for location in agent_state['monitored_locations']:
                lat, lon, crop_type, phosphorus, potassium, irrigation, farm_size = location
                
                # Get current conditions
                avg_temp, rainfall, humidity = get_weather_data(lat, lon)
                soil_data = get_soil_data(lat, lon)
                
                # Agentic decision making
                alerts = []
                
                # 1. Drought risk assessment
                if rainfall < 10 and avg_temp > 30:
                    alert = {
                        'type': 'drought_risk',
                        'location': {'lat': lat, 'lon': lon},
                        'severity': 'high',
                        'message': f'High drought risk detected: {avg_temp}°C, {rainfall}mm rainfall',
                        'recommendation': 'Increase irrigation frequency',
                        'timestamp': current_time.isoformat()
                    }
                    alerts.append(alert)
                
                # 2. Nutrient deficiency check
                if soil_data['nitrogen'] < 0.5:
                    alert = {
                        'type': 'nutrient_deficiency',
                        'location': {'lat': lat, 'lon': lon},
                        'severity': 'medium',
                        'message': f'Low nitrogen levels: {soil_data["nitrogen"]:.2f}%',
                        'recommendation': 'Apply nitrogen fertilizer',
                        'timestamp': current_time.isoformat()
                    }
                    alerts.append(alert)
                
                # 3. Optimal planting window
                if 25 <= avg_temp <= 30 and 50 <= rainfall <= 150:
                    alert = {
                        'type': 'optimal_conditions',
                        'location': {'lat': lat, 'lon': lon},
                        'severity': 'info',
                        'message': f'Optimal conditions for {crop_type}: {avg_temp}°C, {rainfall}mm',
                        'recommendation': 'Good time for planting/field activities',
                        'timestamp': current_time.isoformat()
                    }
                    alerts.append(alert)
                
                # Store alerts
                agent_state['alerts'].extend(alerts)
                
                # Keep only last 50 alerts
                if len(agent_state['alerts']) > 50:
                    agent_state['alerts'] = agent_state['alerts'][-50:]
            
            agent_state['last_check'] = current_time.isoformat()
            
            # Check every hour (3600 seconds)
            time.sleep(3600)
            
        except Exception as e:
            print(f"Monitoring loop error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

def start_monitoring():
    """Start the agentic monitoring system"""
    if not agent_state['monitoring_active']:
        agent_state['monitoring_active'] = True
        monitoring_thread = threading.Thread(target=autonomous_monitoring_loop, daemon=True)
        monitoring_thread.start()
        return True
    return False

def stop_monitoring():
    """Stop the agentic monitoring system"""
    agent_state['monitoring_active'] = False
    return True

def get_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': 'temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean',
            'timezone': 'auto'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Get today's data (first day in response)
        daily = data.get('daily', {})
        avg_temp = daily.get('temperature_2m_mean', [25.0])[0]
        rainfall = daily.get('precipitation_sum', [50.0])[0]
        humidity_percent = daily.get('relative_humidity_2m_mean', [70.0])[0]
        
        return avg_temp, rainfall, humidity_percent
        
    except Exception as e:
        print(f"Weather API error: {e}")
        # Return default values
        return 25.0, 50.0, 70.0

def get_soil_data(lat, lon):
    """Fetch soil data from ISRIC SoilGrids API"""
    soil_properties = {
        'phh2o': 6.5,      # pH
        'soc': 1.5,        # Soil Organic Carbon
        'nitrogen': 0.8,   # Total Nitrogen
        'cec': 15.0,       # Cation Exchange Capacity
        'clay': 30.0,      # Clay content
        'sand': 35.0,      # Sand content
        'silt': 35.0       # Silt content
    }
    
    try:
        base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        
        for prop in ['phh2o', 'soc', 'nitrogen', 'cec', 'clay', 'sand', 'silt']:
            try:
                params = {
                    'lat': lat,
                    'lon': lon,
                    'property': prop,
                    'depth': '0-30cm'
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                data = response.json()
                
                # Extract mean value from the response
                layers = data.get('properties', {}).get('layers', [])
                if layers and layers[0].get('depths'):
                    values = layers[0].get('depths')[0].get('values', {})
                    mean_val = values.get('mean')
                    if mean_val is not None:
                        if prop == 'phh2o':
                            soil_properties['phh2o'] = mean_val
                        elif prop == 'soc':
                            soil_properties['soc'] = mean_val
                        elif prop == 'nitrogen':
                            soil_properties['nitrogen'] = mean_val
                        elif prop == 'cec':
                            soil_properties['cec'] = mean_val
                        elif prop == 'clay':
                            soil_properties['clay'] = mean_val
                        elif prop == 'sand':
                            soil_properties['sand'] = mean_val
                        elif prop == 'silt':
                            soil_properties['silt'] = mean_val
                        
            except Exception as e:
                print(f"Soil API error for {prop}: {e}")
                continue
                
    except Exception as e:
        print(f"Soil API error: {e}")
    
    return soil_properties

def get_location_name(lat, lon):
    """Get location name from coordinates using Nominatim API"""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1,
            'zoom': 10
        }
        
        headers = {
            'User-Agent': 'CropYieldPredictor/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if response.status_code == 200 and 'address' in data:
            address = data['address']
            
            # Build location name from available components
            location_parts = []
            
            # Add village/town/city if available
            if 'village' in address:
                location_parts.append(address['village'])
            elif 'town' in address:
                location_parts.append(address['town'])
            elif 'city' in address:
                location_parts.append(address['city'])
            
            # Add district if available
            if 'county' in address:
                location_parts.append(address['county'])
            elif 'district' in address:
                location_parts.append(address['district'])
            
            # Add state if available
            if 'state' in address:
                location_parts.append(address['state'])
            
            # If no specific parts, use display name
            if not location_parts and 'display_name' in data:
                # Take first few parts of display name
                display_name = data['display_name'].split(',')
                location_parts = [part.strip() for part in display_name[:3]]
            
            location_name = ', '.join(location_parts) if location_parts else data.get('display_name', 'Unknown Location')
            
            return {
                'name': location_name,
                'display_name': data.get('display_name', location_name),
                'address': address
            }
        else:
            return {
                'name': f'Lat: {lat:.4f}, Lon: {lon:.4f}',
                'display_name': f'Lat: {lat:.4f}, Lon: {lon:.4f}',
                'address': {}
            }
            
    except Exception as e:
        print(f"Geocoding API error: {e}")
        return {
            'name': f'Lat: {lat:.4f}, Lon: {lon:.4f}',
            'display_name': f'Lat: {lat:.4f}, Lon: {lon:.4f}',
            'address': {}
        }

def determine_region(lat):
    """Determine region based on latitude"""
    # Simple logic based on Indian geography
    if lat < 8:  # Very south - coastal
        return 'coastal'
    elif lat < 20:  # Central - inland
        return 'inland'
    else:  # North - hills
        return 'hills'

def predict_yield(input_features):
    """Make prediction using both models"""
    if not mdn_model or not transformer_model or not preprocessors:
        return None, None
    
    try:
        # Create DataFrame for preprocessing
        import pandas as pd
        df = pd.DataFrame([input_features])
        
        # Encode categorical variables
        df['crop_type_encoded'] = preprocessors['le_crop'].transform([df['crop_type'].iloc[0]])
        df['region_encoded'] = preprocessors['le_region'].transform([df['region'].iloc[0]])
        df['irrigation_available'] = df['irrigation_available'].astype(int)
        
        # Select features in correct order
        feature_columns = preprocessors['feature_columns']
        features = df[feature_columns].values
        features_scaled = preprocessors['scaler_X'].transform(features)
        
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        
        # MDN Prediction
        with torch.no_grad():
            pi, mu, sigma = mdn_model(features_tensor)
            # Simple prediction using mean of mixture
            mdn_pred_scaled = (pi * mu).sum(dim=1)
            mdn_pred = preprocessors['scaler_y'].inverse_transform(
                mdn_pred_scaled.cpu().numpy().reshape(-1, 1)
            ).flatten()[0]
        
        # Transformer Prediction
        with torch.no_grad():
            transformer_pred_scaled = transformer_model(features_tensor)
            transformer_pred = preprocessors['scaler_y'].inverse_transform(
                transformer_pred_scaled.cpu().numpy().reshape(-1, 1)
            ).flatten()[0]
        
        # Average prediction
        avg_pred = (mdn_pred + transformer_pred) / 2
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            return obj
        
        return float(avg_pred), {
            'mdn_prediction': float(mdn_pred),
            'transformer_prediction': float(transformer_pred)
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

@app.route('/agent/start', methods=['POST'])
def start_agent():
    """Start agentic monitoring"""
    try:
        data = request.get_json()
        locations = data.get('locations', [])
        
        # Add locations to monitoring
        for loc in locations:
            location_tuple = (
                loc['latitude'], loc['longitude'], loc['crop_type'],
                loc.get('phosphorus', 25.0), loc.get('potassium', 150.0),
                loc.get('irrigation_available', 1), loc.get('farm_size_ha', 3.0)
            )
            agent_state['monitored_locations'].append(location_tuple)
        
        # Start monitoring
        started = start_monitoring()
        
        return jsonify({
            'status': 'started' if started else 'already_running',
            'monitoring_locations': len(agent_state['monitored_locations']),
            'message': 'Agentic monitoring started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/agent/stop', methods=['POST'])
def stop_agent():
    """Stop agentic monitoring"""
    try:
        stopped = stop_monitoring()
        agent_state['monitored_locations'] = []
        
        return jsonify({
            'status': 'stopped' if stopped else 'not_running',
            'message': 'Agentic monitoring stopped'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/agent/alerts', methods=['GET'])
def get_alerts():
    """Get all agentic alerts"""
    try:
        return jsonify({
            'alerts': agent_state['alerts'],
            'monitoring_active': agent_state['monitoring_active'],
            'monitored_locations': len(agent_state['monitored_locations']),
            'last_check': agent_state['last_check']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/agent/status', methods=['GET'])
def agent_status():
    """Get agentic system status"""
    try:
        return jsonify({
            'monitoring_active': agent_state['monitoring_active'],
            'monitored_locations': agent_state['monitored_locations'],
            'total_alerts': len(agent_state['alerts']),
            'last_check': agent_state['last_check'],
            'models_loaded': mdn_model is not None and transformer_model is not None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    else:
        return obj

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    print("\n" + "="*80)
    print(" CROP YIELD PREDICTION REQUEST")
    print("="*80)
    
    try:
        data = request.get_json()
        
        # Print user input
        print("\n USER INPUT:")
        print("-" * 40)
        print(f"  Latitude: {data.get('latitude')}")
        print(f"  Longitude: {data.get('longitude')}")
        print(f"  Crop Type: {data.get('crop_type')}")
        print(f"  Phosphorus: {data.get('phosphorus')} mg/kg")
        print(f"  Potassium: {data.get('potassium')} mg/kg")
        print(f"  Irrigation Available: {data.get('irrigation_available')}")
        print(f"  Farm Size: {data.get('farm_size_ha')} hectares")
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'crop_type', 'phosphorus', 'potassium', 'irrigation_available', 'farm_size_ha']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        crop_type = data['crop_type']
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        irrigation_available = int(data['irrigation_available'])
        farm_size_ha = float(data['farm_size_ha'])
        
        # Get current month
        current_month = datetime.now().month
        
        print(f"\n Current Month: {current_month}")
        
        # Get weather data
        print("\n FETCHING WEATHER DATA...")
        print("-" * 40)
        avg_temp, rainfall, humidity_percent = get_weather_data(lat, lon)
        print(f"  Average Temperature: {avg_temp}°C")
        print(f"  Rainfall: {rainfall}mm")
        print(f"  Humidity: {humidity_percent}%")
        
        # Get soil data
        print("\n FETCHING SOIL DATA...")
        print("-" * 40)
        soil_data = get_soil_data(lat, lon)
        print(f"  pH: {soil_data['phh2o']}")
        print(f"  Soil Organic Carbon (SOC): {soil_data['soc']}%")
        print(f"  Total Nitrogen: {soil_data['nitrogen']}%")
        print(f"  CEC: {soil_data['cec']} cmol/kg")
        print(f"  Clay: {soil_data['clay']}%")
        print(f"  Sand: {soil_data['sand']}%")
        print(f"  Silt: {soil_data['silt']}%")
        
        # Get location name
        print("\n FETCHING LOCATION NAME...")
        print("-" * 40)
        location_data = get_location_name(lat, lon)
        print(f"  Location Name: {location_data['name']}")
        
        # Determine region
        print("\n REGION DETERMINATION...")
        print("-" * 40)
        region = determine_region(lat)
        print(f"  Latitude: {lat}°")
        print(f"  Longitude: {lon}°")
        print(f"  Determined Region: {region}")
        
        # Combine all features
        input_features = {
            'avg_temp': avg_temp,
            'rainfall': rainfall,
            'crop_type': crop_type,
            'pH': soil_data['phh2o'],
            'SOC': soil_data['soc'],
            'Total_Nitrogen': soil_data['nitrogen'],
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'CEC': soil_data['cec'],
            'Clay': soil_data['clay'],
            'Sand': soil_data['sand'],
            'Silt': soil_data['silt'],
            'soil_depth_cm': 30.0,
            'humidity_percent': humidity_percent,
            'region': region,
            'month': current_month,
            'irrigation_available': irrigation_available,
            'farm_size_ha': farm_size_ha
        }
        
        print("\n COMBINED FEATURES FOR MODEL:")
        print("-" * 40)
        for key, value in input_features.items():
            print(f"  {key}: {value}")
        
        # Make prediction
        print("\n MAKING PREDICTIONS...")
        print("-" * 40)
        predicted_yield, model_details = predict_yield(input_features)
        
        if predicted_yield is None:
            # Fallback to simple formula if models fail
            print(" Models failed, using fallback formula")
            predicted_yield = (farm_size_ha * 1000) + (rainfall * 5) + (avg_temp * 50)
            model_details = {'note': 'Using fallback formula due to model error'}
            predicted_yield = float(predicted_yield)  # Ensure it's a Python float
        else:
            print(" Models prediction successful!")
            if model_details:
                print(f"  MDN Prediction: {model_details.get('mdn_prediction', 'N/A'):.2f} kg")
                print(f"  Transformer Prediction: {model_details.get('transformer_prediction', 'N/A'):.2f} kg")
                print(f"  Average Prediction: {predicted_yield:.2f} kg")
        
        # Prepare response with proper JSON serialization
        try:
            response = {
                'predicted_yield_kg': round(float(predicted_yield), 2),
                'input_features': convert_to_native(input_features),
                'model_details': convert_to_native(model_details) if model_details else {},
                'api_data': {
                    'weather': {
                        'avg_temp': float(avg_temp),
                        'rainfall': float(rainfall),
                        'humidity_percent': float(humidity_percent)
                    },
                    'soil': convert_to_native(soil_data),
                    'region': region,
                    'month': int(current_month),
                    'location': {
                        'name': location_data['name'],
                        'display_name': location_data['display_name'],
                        'coordinates': {'lat': float(lat), 'lon': float(lon)}
                    }
                }
            }
        except Exception as json_error:
            print(f"JSON serialization error: {json_error}")
            # Fallback response with minimal data
            response = {
                'predicted_yield_kg': round(float(predicted_yield), 2),
                'input_features': {
                    'crop_type': crop_type,
                    'farm_size_ha': farm_size_ha,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'irrigation_available': irrigation_available
                },
                'model_details': {'note': 'Detailed data unavailable due to serialization error'},
                'api_data': {
                    'weather': {'avg_temp': float(avg_temp), 'rainfall': float(rainfall)},
                    'soil': {'phh2o': soil_data.get('phh2o', 6.5)},
                    'region': region,
                    'month': int(current_month),
                    'location': {
                        'name': location_data['name'],
                        'display_name': location_data['display_name'],
                        'coordinates': {'lat': float(lat), 'lon': float(lon)}
                    }
                }
            }
        
        print("\n📤 RESPONSE SENT TO CLIENT:")
        print("-" * 40)
        print(f"  Predicted Yield: {response['predicted_yield_kg']} kg")
        print(f"  Status: Success")
        print("="*80 + "\n")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': mdn_model is not None and transformer_model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    print("="*50)
    print("Starting Crop Yield Prediction Server...")
    print("="*50 + "\n")
    
    # Load ML models
    print("\nLoading ML models...")
    if not load_models():
        print("\n" + "!"*50)
        print("ERROR: Failed to load one or more models. Server cannot start.")
        print("Please check the error messages above for details.")
        print("!"*50 + "\n")
        exit(1)
    
    # Start the server
    print("\n" + "="*50)
    print("Starting Flask server...")
    print("Server running at http://0.0.0.0:5000")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print("\n" + "!"*50)
        print(f"ERROR: Failed to start server: {e}")
        print("!"*50 + "\n")
