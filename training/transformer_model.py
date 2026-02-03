import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CropYieldDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

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

def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Handle categorical variables
    le_crop = LabelEncoder()
    le_region = LabelEncoder()
    
    df['crop_type_encoded'] = le_crop.fit_transform(df['crop_type'])
    df['region_encoded'] = le_region.fit_transform(df['region'])
    
    # Convert boolean to int
    df['irrigation_available'] = df['irrigation_available'].astype(int)
    
    # Select features for training (removed tech_adoption_score)
    feature_columns = [
        'avg_temp', 'rainfall', 'crop_type_encoded', 'pH', 'SOC',
        'Total_Nitrogen', 'Phosphorus', 'Potassium', 'CEC', 'Clay', 'Sand', 'Silt',
        'soil_depth_cm', 'humidity_percent', 'region_encoded', 'month',
        'irrigation_available', 'farm_size_ha'
    ]
    
    X = df[feature_columns].values
    y = df['yield_kg'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            scaler_X, scaler_y, le_crop, le_region, feature_columns)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, scaler_y, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            
            outputs = model(batch_features)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_targets.numpy())
    
    # Convert back to original scale
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_results(predictions, actuals, train_losses, val_losses):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actual vs Predicted scatter plot
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=1)
    axes[0, 0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Yield (kg)')
    axes[0, 0].set_ylabel('Predicted Yield (kg)')
    axes[0, 0].set_title('Actual vs Predicted Crop Yield')
    axes[0, 0].grid(True)
    
    # Residual plot
    residuals = actuals - predictions
    axes[0, 1].scatter(predictions, residuals, alpha=0.5, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Yield (kg)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True)
    
    # Training loss curves
    axes[1, 0].plot(train_losses, label='Training Loss')
    axes[1, 0].plot(val_losses, label='Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training and Validation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Distribution of predictions vs actuals
    axes[1, 1].hist(actuals, bins=50, alpha=0.5, label='Actual', density=True)
    axes[1, 1].hist(predictions, bins=50, alpha=0.5, label='Predicted', density=True)
    axes[1, 1].set_xlabel('Yield (kg)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution of Actual vs Predicted Yields')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('transformer_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    DATA_FILE = 'dataset.csv'
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 4
    DROPOUT = 0.1
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    (X_train, X_test, y_train, y_test, scaler_X, scaler_y, 
     le_crop, le_region, feature_columns) = load_and_preprocess_data(DATA_FILE)
    
    # Create datasets and dataloaders
    train_dataset = CropYieldDataset(X_train, y_train)
    test_dataset = CropYieldDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = TransformerRegressor(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device
    )
    
    # Evaluate model
    predictions, actuals, metrics = evaluate_model(model, test_loader, scaler_y, device)
    
    # Plot results
    plot_results(predictions, actuals, train_losses, val_losses)
    
    # Save model to models directory
    import os
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'le_crop': le_crop,
        'le_region': le_region,
        'feature_columns': feature_columns,
        'metrics': metrics
    }, os.path.join(models_dir, 'transformer_crop_yield_model.pth'))
    
    print(f"\nTraining completed! Model saved as '{models_dir}/transformer_crop_yield_model.pth'")
    print(f"Final R² Score: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()
