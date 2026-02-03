# Crop Yield Prediction Models Documentation

## Overview

This project implements two deep learning models for crop yield prediction:

1. **Mixture Density Network (MDN)** - Provides predictions with uncertainty quantification
2. **Transformer Regressor** - Uses attention mechanisms for feature relationships

Both models are trained on agricultural data with 17 input features to predict crop yield in kilograms.

## Input Features

Both models use the same 17 standardized input features:

| Feature | Type | Description | Example Range |
|---------|------|-------------|---------------|
| `avg_temp` | Continuous | Average temperature (°C) | 25-35°C |
| `rainfall` | Continuous | Rainfall amount (mm) | 0-200mm |
| `crop_type_encoded` | Categorical | Crop type (encoded) | 0-6 (rice, sugarcane, cotton, pulses, millets, groundnut, coconut) |
| `pH` | Continuous | Soil pH level | 4.5-8.5 |
| `SOC` | Continuous | Soil Organic Carbon (%) | 0.5-3.0% |
| `Total_Nitrogen` | Continuous | Total Nitrogen content | 0.1-2.0% |
| `Phosphorus` | Continuous | Phosphorus content | 5-50 mg/kg |
| `Potassium` | Continuous | Potassium content | 50-300 mg/kg |
| `CEC` | Continuous | Cation Exchange Capacity | 5-30 cmol/kg |
| `Clay` | Continuous | Clay content (%) | 10-60% |
| `Sand` | Continuous | Sand content (%) | 10-70% |
| `Silt` | Continuous | Silt content (%) | 10-50% |
| `soil_depth_cm` | Continuous | Soil depth (cm) | 20-200cm |
| `humidity_percent` | Continuous | Humidity percentage | 40-90% |
| `region_encoded` | Categorical | Region type (encoded) | 0-2 (coastal, inland, hills) |
| `month` | Categorical | Month of year | 1-12 |
| `irrigation_available` | Binary | Irrigation availability | 0 or 1 |
| `farm_size_ha` | Continuous | Farm size in hectares | 0.5-10 ha |

## Model Outputs

### Mixture Density Network (MDN)
- **Primary Output**: Predicted yield (kg)
- **Uncertainty Output**: Standard deviation of prediction
- **Additional**: Mixture parameters (pi, mu, sigma) for probability distribution

### Transformer Regressor
- **Primary Output**: Predicted yield (kg)

## Sample Input/Output

### Sample Input 1: Rice Farm in Coastal Region
```python
input_data = {
    'avg_temp': 28.5,
    'rainfall': 120.0,
    'crop_type': 'rice',  # Will be encoded to 0
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
    'region': 'coastal',  # Will be encoded to 0
    'month': 7,  # July
    'irrigation_available': 1,
    'farm_size_ha': 3.0
}
```

**Expected Output Range:**
- **MDN**: 2000-4000 kg ± 200-500 kg uncertainty
- **Transformer**: 2000-4000 kg

### Sample Input 2: Sugarcane in Inland Region
```python
input_data = {
    'avg_temp': 32.0,
    'rainfall': 80.0,
    'crop_type': 'sugarcane',  # Will be encoded to 1
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
    'region': 'inland',  # Will be encoded to 1
    'month': 10,  # October
    'irrigation_available': 1,
    'farm_size_ha': 5.0
}
```

**Expected Output Range:**
- **MDN**: 28000-35000 kg ± 1000-2000 kg uncertainty
- **Transformer**: 28000-35000 kg

### Sample Input 3: Cotton in Hills Region
```python
input_data = {
    'avg_temp': 26.0,
    'rainfall': 60.0,
    'crop_type': 'cotton',  # Will be encoded to 2
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
    'region': 'hills',  # Will be encoded to 2
    'month': 9,  # September
    'irrigation_available': 0,
    'farm_size_ha': 2.0
}
```

**Expected Output Range:**
- **MDN**: 600-1200 kg ± 100-300 kg uncertainty
- **Transformer**: 600-1200 kg

## Model Performance

### Mixture Density Network (MDN)
- **R² Score**: ~0.85-0.90
- **RMSE**: ~800-1200 kg
- **95% Coverage**: ~0.90-0.95 (uncertainty calibration)
- **Advantages**: Provides uncertainty quantification, better for risk assessment

### Transformer Regressor
- **R² Score**: ~0.80-0.88
- **RMSE**: ~900-1400 kg
- **Advantages**: Captures complex feature relationships, good for pattern recognition

## Data Preprocessing

1. **Categorical Encoding**: 
   - `crop_type`: Label encoded (0-6)
   - `region`: Label encoded (0-2)
   - `irrigation_available`: Boolean to int (0/1)

2. **Feature Scaling**: StandardScaler applied to all features
3. **Target Scaling**: StandardScaler applied to yield values, then inverse transformed for predictions

## Model Architecture Details

### MDN Architecture
- Input: 17 features
- Hidden layers: [256, 128, 64] with ReLU, BatchNorm, Dropout(0.2)
- Output: 5 Gaussian mixtures (pi, mu, sigma parameters)
- Loss: Negative log likelihood

### Transformer Architecture
- Input: 17 features → 128 dimensions
- 4 Transformer encoder layers, 8 attention heads
- Positional encoding for sequence processing
- Output projection to single yield value
- Loss: Mean Squared Error

## Usage Notes

- Both models require the same preprocessing pipeline
- Input features must be in the exact order specified
- Models are saved with their preprocessing objects for consistency
- MDN provides uncertainty estimates useful for decision making
- Both models perform best within the training data distribution ranges
