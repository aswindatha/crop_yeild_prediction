import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], n_gaussians=5):
        super(MDN, self).__init__()
        
        self.input_dim = input_dim
        self.n_gaussians = n_gaussians
        
        # Feature extractor layers
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
        
        # MDN output layers
        self.pi_net = nn.Linear(prev_dim, n_gaussians)  # Mixture weights
        self.mu_net = nn.Linear(prev_dim, n_gaussians)  # Means
        self.sigma_net = nn.Linear(prev_dim, n_gaussians)  # Standard deviations
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # MDN parameters
        pi = torch.softmax(self.pi_net(features), dim=1)  # Mixture weights
        mu = self.mu_net(features)  # Means
        sigma = F.softplus(self.sigma_net(features)) + 1e-6  # Std dev (positive)
        
        return pi, mu, sigma

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        # Add sequence dimension (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Project to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling and output projection
        x = x.mean(dim=1)  # (batch_size, d_model)
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

def sample_from_mixture(pi, mu, sigma, n_samples=1):
    """Sample from the mixture distribution"""
    batch_size = pi.size(0)
    
    samples = []
    for _ in range(n_samples):
        # Select mixture component
        component = torch.multinomial(pi, 1).squeeze()
        
        # Sample from selected component
        indices = torch.arange(batch_size)
        mu_sample = mu[indices, component]
        sigma_sample = sigma[indices, component]
        
        sample = torch.normal(mu_sample, sigma_sample)
        samples.append(sample)
    
    return torch.stack(samples, dim=1) if n_samples > 1 else samples[0]

class CropYieldPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mdn_model = None
        self.transformer_model = None
        self.mdn_preprocessors = None
        self.transformer_preprocessors = None
        
    def load_models(self):
        """Load both trained models and their preprocessors"""
        print("Loading models...")
        
        # Load MDN model
        mdn_checkpoint = torch.load('models/mdn_crop_yield_model.pth', map_location=self.device, weights_only=False)
        input_dim = len(mdn_checkpoint['feature_columns'])
        
        self.mdn_model = MDN(
            input_dim=input_dim,
            hidden_dims=mdn_checkpoint['hidden_dims'],
            n_gaussians=mdn_checkpoint['n_gaussians']
        ).to(self.device)
        
        self.mdn_model.load_state_dict(mdn_checkpoint['model_state_dict'])
        self.mdn_model.eval()
        
        self.mdn_preprocessors = {
            'scaler_X': mdn_checkpoint['scaler_X'],
            'scaler_y': mdn_checkpoint['scaler_y'],
            'le_crop': mdn_checkpoint['le_crop'],
            'le_region': mdn_checkpoint['le_region'],
            'feature_columns': mdn_checkpoint['feature_columns']
        }
        
        # Load Transformer model
        transformer_checkpoint = torch.load('models/transformer_crop_yield_model.pth', map_location=self.device, weights_only=False)
        input_dim = len(transformer_checkpoint['feature_columns'])
        
        self.transformer_model = TransformerRegressor(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.1
        ).to(self.device)
        
        self.transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
        self.transformer_model.eval()
        
        self.transformer_preprocessors = {
            'scaler_X': transformer_checkpoint['scaler_X'],
            'scaler_y': transformer_checkpoint['scaler_y'],
            'le_crop': transformer_checkpoint['le_crop'],
            'le_region': transformer_checkpoint['le_region'],
            'feature_columns': transformer_checkpoint['feature_columns']
        }
        
        print("Models loaded successfully!")
        
    def preprocess_input(self, input_data, preprocessors):
        """Preprocess input data using the same pipeline as training"""
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        df['crop_type_encoded'] = preprocessors['le_crop'].transform([df['crop_type'].iloc[0]])
        df['region_encoded'] = preprocessors['le_region'].transform([df['region'].iloc[0]])
        
        # Convert boolean to int
        df['irrigation_available'] = df['irrigation_available'].astype(int)
        
        # Select features in the correct order
        feature_columns = preprocessors['feature_columns']
        features = df[feature_columns].values
        
        # Scale features
        features_scaled = preprocessors['scaler_X'].transform(features)
        
        return features_scaled
    
    def predict_mdn(self, input_data, n_samples=10):
        """Make prediction using MDN model with uncertainty"""
        if self.mdn_model is None:
            raise ValueError("MDN model not loaded. Call load_models() first.")
        
        # Preprocess input
        features = self.preprocess_input(input_data, self.mdn_preprocessors)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            # Get MDN parameters
            pi, mu, sigma = self.mdn_model(features_tensor)
            
            # Sample multiple times to get prediction and uncertainty
            samples = []
            for _ in range(n_samples):
                sample = sample_from_mixture(pi, mu, sigma)
                samples.append(sample)
            
            samples = torch.stack(samples, dim=1)  # (1, n_samples)
            
            # Calculate mean and std across samples
            pred_mean_scaled = samples.mean(dim=1)
            pred_std_scaled = samples.std(dim=1)
            
            # Convert back to original scale
            pred_mean = self.mdn_preprocessors['scaler_y'].inverse_transform(
                pred_mean_scaled.cpu().numpy().reshape(-1, 1)
            ).flatten()[0]
            
            pred_std = pred_std_scaled.cpu().numpy()[0] * self.mdn_preprocessors['scaler_y'].scale_[0]
            
        return pred_mean, pred_std
    
    def predict_transformer(self, input_data):
        """Make prediction using Transformer model"""
        if self.transformer_model is None:
            raise ValueError("Transformer model not loaded. Call load_models() first.")
        
        # Preprocess input
        features = self.preprocess_input(input_data, self.transformer_preprocessors)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            # Get prediction
            pred_scaled = self.transformer_model(features_tensor)
            
            # Convert back to original scale
            pred = self.transformer_preprocessors['scaler_y'].inverse_transform(
                pred_scaled.cpu().numpy().reshape(-1, 1)
            ).flatten()[0]
            
        return pred
    
    def predict_all(self, input_data):
        """Get predictions from both models"""
        mdn_pred, mdn_uncertainty = self.predict_mdn(input_data)
        transformer_pred = self.predict_transformer(input_data)
        
        return {
            'MDN_Prediction': mdn_pred,
            'MDN_Uncertainty': mdn_uncertainty,
            'Transformer_Prediction': transformer_pred,
            'Average_Prediction': (mdn_pred + transformer_pred) / 2
        }

def main():
    # Initialize predictor
    predictor = CropYieldPredictor()
    
    # Load models
    predictor.load_models()
    
    # Sample test cases
    test_cases = [
        {
            'name': 'Rice Farm in Coastal Region',
            'data': {
                'avg_temp': 28.5,
                'rainfall': 120.0,
                'crop_type': 'rice',
                'pH': 6.5,
                'SOC': 1.8,
                'Total_Nitrogen': 0.8,
                'Phosphorus': 25.0,
                'Potassium': 150.0,
                'CEC': 18.0,
                'Clay': 35.0,
                'Sand': 30.0,
                'Silt': 35.0,
                'soil_depth_cm': 100.0,
                'humidity_percent': 75.0,
                'region': 'coastal',
                'month': 7,
                'irrigation_available': 1,
                'farm_size_ha': 3.0
            }
        },
        {
            'name': 'Sugarcane in Inland Region',
            'data': {
                'avg_temp': 32.0,
                'rainfall': 80.0,
                'crop_type': 'sugarcane',
                'pH': 7.2,
                'SOC': 2.2,
                'Total_Nitrogen': 1.2,
                'Phosphorus': 35.0,
                'Potassium': 200.0,
                'CEC': 22.0,
                'Clay': 40.0,
                'Sand': 25.0,
                'Silt': 35.0,
                'soil_depth_cm': 150.0,
                'humidity_percent': 65.0,
                'region': 'inland',
                'month': 10,
                'irrigation_available': 1,
                'farm_size_ha': 5.0
            }
        },
        {
            'name': 'Cotton in Hills Region',
            'data': {
                'avg_temp': 26.0,
                'rainfall': 60.0,
                'crop_type': 'cotton',
                'pH': 6.8,
                'SOC': 1.5,
                'Total_Nitrogen': 0.6,
                'Phosphorus': 20.0,
                'Potassium': 120.0,
                'CEC': 15.0,
                'Clay': 30.0,
                'Sand': 40.0,
                'Silt': 30.0,
                'soil_depth_cm': 80.0,
                'humidity_percent': 60.0,
                'region': 'hills',
                'month': 9,
                'irrigation_available': 0,
                'farm_size_ha': 2.0
            }
        },
        {
            'name': 'Groundnut Small Farm',
            'data': {
                'avg_temp': 30.0,
                'rainfall': 90.0,
                'crop_type': 'groundnut',
                'pH': 6.2,
                'SOC': 1.2,
                'Total_Nitrogen': 0.5,
                'Phosphorus': 18.0,
                'Potassium': 100.0,
                'CEC': 12.0,
                'Clay': 25.0,
                'Sand': 45.0,
                'Silt': 30.0,
                'soil_depth_cm': 60.0,
                'humidity_percent': 70.0,
                'region': 'coastal',
                'month': 6,
                'irrigation_available': 1,
                'farm_size_ha': 1.5
            }
        },
        {
            'name': 'Coconut Plantation',
            'data': {
                'avg_temp': 31.0,
                'rainfall': 150.0,
                'crop_type': 'coconut',
                'pH': 7.0,
                'SOC': 2.5,
                'Total_Nitrogen': 1.0,
                'Phosphorus': 30.0,
                'Potassium': 180.0,
                'CEC': 20.0,
                'Clay': 38.0,
                'Sand': 32.0,
                'Silt': 30.0,
                'soil_depth_cm': 120.0,
                'humidity_percent': 80.0,
                'region': 'coastal',
                'month': 8,
                'irrigation_available': 1,
                'farm_size_ha': 4.0
            }
        }
    ]
    
    # Make predictions for all test cases
    print("\n" + "="*80)
    print("CROP YIELD PREDICTION RESULTS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 50)
        
        # Get predictions
        results = predictor.predict_all(test_case['data'])
        
        # Print results
        print(f"MDN Prediction:       {results['MDN_Prediction']:.2f} kg ± {results['MDN_Uncertainty']:.2f} kg")
        print(f"Transformer Prediction: {results['Transformer_Prediction']:.2f} kg")
        print(f"Average Prediction:    {results['Average_Prediction']:.2f} kg")
        
        # Print input summary
        print(f"\nInput Summary:")
        print(f"  Crop: {test_case['data']['crop_type']}")
        print(f"  Region: {test_case['data']['region']}")
        print(f"  Temperature: {test_case['data']['avg_temp']}°C")
        print(f"  Rainfall: {test_case['data']['rainfall']}mm")
        print(f"  Farm Size: {test_case['data']['farm_size_ha']} ha")
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
