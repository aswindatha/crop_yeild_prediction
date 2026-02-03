import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        pi = F.softmax(self.pi_net(features), dim=1)  # Mixture weights
        mu = self.mu_net(features)  # Means
        sigma = F.softplus(self.sigma_net(features)) + 1e-6  # Std dev (positive)
        
        return pi, mu, sigma

def mdn_loss(pi, mu, sigma, target):
    """Calculate MDN negative log likelihood loss"""
    # Expand target to match mixture dimensions
    target = target.unsqueeze(1).expand_as(mu)
    
    # Calculate normal distribution probabilities
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(target)
    
    # Weight by mixture probabilities
    weighted_log_prob = log_prob + torch.log(pi + 1e-10)
    
    # Log sum exp across mixtures
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)
    
    # Negative log likelihood
    return -log_sum.mean()

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

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    
    print("Starting MDN training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            pi, mu, sigma = model(batch_features)
            loss = mdn_loss(pi, mu, sigma, batch_targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                
                pi, mu, sigma = model(batch_features)
                loss = mdn_loss(pi, mu, sigma, batch_targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, scaler_y, device, n_samples=10):
    model.eval()
    predictions = []
    uncertainties = []
    actuals = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            
            pi, mu, sigma = model(batch_features)
            
            # Sample multiple times from mixture
            samples = []
            for _ in range(n_samples):
                sample = sample_from_mixture(pi, mu, sigma)
                samples.append(sample)
            
            samples = torch.stack(samples, dim=1)  # (batch_size, n_samples)
            
            # Calculate mean and std across samples
            pred_mean = samples.mean(dim=1)
            pred_std = samples.std(dim=1)
            
            predictions.extend(pred_mean.cpu().numpy())
            uncertainties.extend(pred_std.cpu().numpy())
            actuals.extend(batch_targets.numpy())
    
    # Convert back to original scale
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    uncertainties = np.array(uncertainties) * scaler_y.scale_[0]  # Scale uncertainty
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Calculate uncertainty calibration
    coverage_95 = np.mean(np.abs(actuals - predictions) <= 1.96 * uncertainties)
    
    print(f"\nMDN Model Evaluation Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Uncertainty: {np.mean(uncertainties):.2f}")
    print(f"95% Coverage: {coverage_95:.3f}")
    
    return predictions, actuals, uncertainties, {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'mean_uncertainty': np.mean(uncertainties),
        'coverage_95': coverage_95
    }

def plot_mdn_results(predictions, actuals, uncertainties, train_losses, val_losses):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Actual vs Predicted scatter plot
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=1)
    axes[0, 0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Yield (kg)')
    axes[0, 0].set_ylabel('Predicted Yield (kg)')
    axes[0, 0].set_title('Actual vs Predicted Crop Yield')
    axes[0, 0].grid(True)
    
    # Prediction with uncertainty bands
    sorted_indices = np.argsort(actuals)
    sorted_actuals = actuals[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    sorted_uncertainties = uncertainties[sorted_indices]
    
    axes[0, 1].plot(sorted_actuals, sorted_predictions, 'b.', alpha=0.3, markersize=2)
    axes[0, 1].fill_between(sorted_actuals, 
                           sorted_predictions - 1.96*sorted_uncertainties,
                           sorted_predictions + 1.96*sorted_uncertainties,
                           alpha=0.2, color='blue', label='95% CI')
    axes[0, 1].plot([sorted_actuals.min(), sorted_actuals.max()], 
                    [sorted_actuals.min(), sorted_actuals.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Yield (kg)')
    axes[0, 1].set_ylabel('Predicted Yield (kg)')
    axes[0, 1].set_title('Predictions with Uncertainty Bands')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Residual plot
    residuals = actuals - predictions
    axes[0, 2].scatter(predictions, residuals, alpha=0.5, s=1)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Predicted Yield (kg)')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Residual Plot')
    axes[0, 2].grid(True)
    
    # Training loss curves
    axes[1, 0].plot(train_losses, label='Training Loss')
    axes[1, 0].plot(val_losses, label='Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training and Validation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Uncertainty vs Error
    errors = np.abs(residuals)
    axes[1, 1].scatter(uncertainties, errors, alpha=0.5, s=1)
    axes[1, 1].set_xlabel('Prediction Uncertainty (kg)')
    axes[1, 1].set_ylabel('Absolute Error (kg)')
    axes[1, 1].set_title('Uncertainty vs Prediction Error')
    axes[1, 1].grid(True)
    
    # Distribution of uncertainties
    axes[1, 2].hist(uncertainties, bins=50, alpha=0.7, density=True)
    axes[1, 2].set_xlabel('Prediction Uncertainty (kg)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Distribution of Prediction Uncertainties')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('mdn_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    DATA_FILE = 'dataset.csv'
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    N_GAUSSIANS = 5
    HIDDEN_DIMS = [256, 128, 64]
    
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
    
    # Initialize MDN model
    input_dim = X_train.shape[1]
    model = MDN(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        n_gaussians=N_GAUSSIANS
    ).to(device)
    
    print(f"MDN Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of Gaussian mixtures: {N_GAUSSIANS}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, optimizer, scheduler, NUM_EPOCHS, device
    )
    
    # Evaluate model
    predictions, actuals, uncertainties, metrics = evaluate_model(
        model, test_loader, scaler_y, device
    )
    
    # Plot results
    plot_mdn_results(predictions, actuals, uncertainties, train_losses, val_losses)
    
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
        'n_gaussians': N_GAUSSIANS,
        'hidden_dims': HIDDEN_DIMS,
        'metrics': metrics
    }, os.path.join(models_dir, 'mdn_crop_yield_model.pth'))
    
    print(f"\nMDN Training completed! Model saved as '{models_dir}/mdn_crop_yield_model.pth'")
    print(f"Final R² Score: {metrics['r2']:.4f}")
    print(f"95% Coverage: {metrics['coverage_95']:.3f}")

if __name__ == "__main__":
    main()
