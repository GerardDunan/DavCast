import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from scipy import stats
import os
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR  # Add this import
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import datetime
import warnings
import copy
import requests
from datetime import datetime
import traceback
warnings.filterwarnings('ignore')

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Define global variables
base_features = [
    'ramp_up_rate',        # Ramp features (3)
    'clear_sky_ratio',
    'hour_ratio',
    'prev_hour',           # History features (3)
    'rolling_mean_3h',
    'prev_day_same_hour',
    'UV Index',            # Environmental features (4)
    'Average Temperature',
    'Average Humidity',
    'clear_sky_radiation'
]  # Total 10 base features + 2 time features = 12 features

# Initialize scalers as global variables
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Constants
DAVAO_LATITUDE = 7.0707
DAVAO_LONGITUDE = 125.6087
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class SolarDataset(Dataset):
    def __init__(self, X, time_features, y):
        self.X = torch.FloatTensor(X)
        self.time_features = torch.FloatTensor(time_features)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.time_features[idx], self.y[idx]

class WeightedSolarDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.weights = torch.FloatTensor(weights)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


def preprocess_data(data_path):
    """Preprocess data with enhanced feature engineering"""
    try:
        # Load the CSV file first
        data = pd.read_csv(data_path)
        
        if 'timestamp' not in data.columns:
            data['timestamp'] = pd.to_datetime(data['Date & Time'])
        
        # Create two versions of the data: hourly and 5-minute intervals
        hourly_data = data[data['timestamp'].dt.minute == 0].copy()
        minute_data = data.copy()  # Keep all 5-minute data
        
        # Process hourly data
        hourly_data['date'] = hourly_data['timestamp'].dt.date
        hourly_data['hour'] = hourly_data['timestamp'].dt.hour
        hourly_data['month'] = hourly_data['timestamp'].dt.month
        hourly_data['day_of_year'] = hourly_data['timestamp'].dt.dayofyear
        
        # Process minute data (keep original timestamps)
        minute_data['date'] = minute_data['timestamp'].dt.date
        minute_data['hour'] = minute_data['timestamp'].dt.hour
        minute_data['month'] = minute_data['timestamp'].dt.month
        minute_data['day_of_year'] = minute_data['timestamp'].dt.dayofyear
        minute_data['minute'] = minute_data['timestamp'].dt.minute  # Add minute information
        
        # Calculate clear sky radiation for both datasets
        for df in [hourly_data, minute_data]:
            df['clear_sky_radiation'] = df.apply(
                lambda row: calculate_clear_sky_radiation(
                    row['hour'] + row['minute']/60 if 'minute' in df.columns else row['hour'],  # Use decimal hours
                    DAVAO_LATITUDE, 
                    DAVAO_LONGITUDE, 
                    row['date']
                ), 
                axis=1
            )
        
        # Calculate feature averages from hourly data
        feature_averages = {
            col: hourly_data[col].mean() 
            for col in hourly_data.select_dtypes(include=[np.number]).columns 
            if col not in ['hour', 'month', 'day_of_year', 'minute']
        }
        
        # Process the rest of the features for hourly data
        process_features(hourly_data)
        process_features(minute_data)
        
        return hourly_data, minute_data, feature_averages
        
    except Exception as e:
        print(f"Error in preprocess_data: {str(e)}")
        traceback.print_exc()
        return None, None, None

def process_features(data):
    """Process features for both hourly and minute data"""
    # Enhanced time features
    data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_year']/365)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_year']/365)
    
    # Calculate rolling statistics
    data['rolling_mean_3h'] = data.groupby('date')['Solar Rad - W/m^2'].transform(
        lambda x: x.rolling(window=36, min_periods=1).mean()  # 36 5-minute intervals = 3 hours
    )
    data['rolling_mean_6h'] = data.groupby('date')['Solar Rad - W/m^2'].transform(
        lambda x: x.rolling(window=72, min_periods=1).mean()  # 72 5-minute intervals = 6 hours
    )
    data['rolling_std_3h'] = data.groupby('date')['Solar Rad - W/m^2'].transform(
        lambda x: x.rolling(window=36, min_periods=1).std()
    )
    
    # Other features
    data['cloud_impact'] = 1 - (data['Solar Rad - W/m^2'] / data['clear_sky_radiation'].clip(lower=1))
    data['solar_trend'] = data.groupby('date')['Solar Rad - W/m^2'].diff()
    data['clear_sky_ratio'] = data['Solar Rad - W/m^2'] / data['clear_sky_radiation'].clip(lower=1)
    data['humidity_impact'] = 1 - (data['Average Humidity'] / 100)
    data['temp_clear_sky_interaction'] = data['Average Temperature'] * data['clear_sky_radiation'] / 1000
    
    # Fill NaN values
    for col in data.select_dtypes(include=[np.number]).columns:
        if col != 'hour':
            if 'prev' in col or 'rolling' in col:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            else:
                data[col] = data[col].fillna(data.groupby('hour')[col].transform('mean'))

def analyze_features(data):
    """Analyze and select the most important features using multiple methods"""
    try:
        print("\nPerforming feature selection analysis...")
        
        # Create a copy of the data and drop non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number]).copy()
        
        # Drop any datetime columns if they exist
        datetime_cols = ['date', 'timestamp']
        numeric_data = numeric_data.drop(columns=[col for col in datetime_cols if col in numeric_data.columns])
        
        # Select target variable - using ground data's solar radiation
        target = numeric_data['Solar Rad - W/m^2']
        features = numeric_data.drop(['Solar Rad - W/m^2'], axis=1)
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(features, target)
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Calculate correlations
        correlations = features.corrwith(target).abs()
        
        # Combine both metrics
        feature_importance['correlation'] = [correlations[feat] for feat in feature_importance['feature']]
        
        # Calculate combined score
        feature_importance['combined_score'] = (
            feature_importance['importance'] * 0.6 + 
            feature_importance['correlation'] * 0.4
        )
        
        # Sort by combined score
        feature_importance = feature_importance.sort_values('combined_score', ascending=False)
        
        print("\nFeature Importance Analysis:")
        print("============================")
        print("\nAll Features Ranked by Importance:")
        print("----------------------------------")
        for idx, row in feature_importance.iterrows():
            print(f"{row['feature']:<30} | MI Score: {row['importance']:.4f} | "
                  f"Correlation: {row['correlation']:.4f} | Combined: {row['combined_score']:.4f}")
        
        # Select top features with higher threshold and remove redundant ones
        threshold = 1.4  # Increased threshold
        initial_selection = feature_importance[feature_importance['combined_score'] > threshold]
        
        # List of feature groups that are likely redundant
        redundant_groups = [
            ['Average Temperature', 'Heat Index', 'Average THW Index', 'Average Wind Chill'],  # Keep THSW Index separate
            ['cloud_impact', 'cloud_cover', 'clearness_index'],  # Keep all cloud-related features
            ['prev_hour', 'prev_2hour'],  # Keep rolling_mean_3h separate
            ['hour_sin', 'hour_cos']  # Keep hour separate
        ]
        
        # Keep only the best feature from each redundant group
        final_features = []
        used_groups = set()
        
        for feat in initial_selection['feature']:
            # Check if feature belongs to any redundant group
            in_redundant_group = False
            for group in redundant_groups:
                if feat in group:
                    group_key = tuple(group)  # Convert list to tuple for set membership
                    if group_key not in used_groups:
                        final_features.append(feat)
                        used_groups.add(group_key)
                    in_redundant_group = True
                    break
            
            # If feature is not in any redundant group, add it
            if not in_redundant_group:
                final_features.append(feat)
        
        # Always include these essential features regardless of threshold
        essential_features = [
            'UV Index', 
            'clear_sky_radiation', 
            'rolling_mean_3h',  # Added as essential
            'Average THSW Index',  # Added as essential
            'prev_hour'  # Added as essential
        ]
        for feat in essential_features:
            if feat not in final_features and feat in features.columns:
                final_features.append(feat)
        
        print("\nSelected Features After Redundancy Removal:")
        print("----------------------------------------")
        for feat in final_features:
            score = feature_importance[feature_importance['feature'] == feat]['combined_score'].iloc[0]
            print(f"{feat:<30} | Combined Score: {score:.4f}")
        
        return final_features
        
    except Exception as e:
        print(f"Error in analyze_features: {str(e)}")
        traceback.print_exc()
        return None

def prepare_features(df):
    try:
        # Filter for exact hour values
        df = df[df['timestamp'].dt.minute == 0].copy()
        
        # Calculate all features first
        df = calculate_all_features(df)
        
        # Prepare feature matrix ensuring all features exist
        feature_matrix = []
        for feature in base_features:
            if feature not in df.columns:
                print(f"Warning: Missing feature {feature}, adding zeros")
                df[feature] = 0.0
            feature_matrix.append(df[feature].values)
        
        # Convert to numpy array
        X = np.column_stack(feature_matrix)
        
        # Add time features
        hour_sin = np.sin(2 * np.pi * df['hour'] / 24)
        hour_cos = np.cos(2 * np.pi * df['hour'] / 24)
        time_features = np.column_stack([hour_sin, hour_cos])
        
        # Combine all features
        X = np.column_stack([X, time_features])
        
        print(f"\nFeature matrix shape: {X.shape}")
        print("Features included:", base_features + ['hour_sin', 'hour_cos'])
        
        return X, df['Solar Rad - W/m^2'].values, base_features
        
    except Exception as e:
        print(f"Error in prepare_features: {str(e)}")
        traceback.print_exc()
        return None, None, None

def calculate_all_features(df):
    """Calculate all required features"""
    # Calculate clear sky radiation
    df['clear_sky_radiation'] = df.apply(
        lambda row: calculate_clear_sky_radiation(
            row['hour'], 
            DAVAO_LATITUDE, 
            DAVAO_LONGITUDE, 
            pd.to_datetime(row['date'])
        ), 
        axis=1
    )
    
    # Calculate ramp features
    df['ramp_up_rate'] = df.groupby('hour')['Solar Rad - W/m^2'].diff().rolling(7, min_periods=1).mean()
    df['clear_sky_ratio'] = df['Solar Rad - W/m^2'] / df['clear_sky_radiation'].clip(lower=1)
    df['hour_ratio'] = df['hour'] / 24.0
    
    # Calculate history features
    df['prev_hour'] = df.groupby('date')['Solar Rad - W/m^2'].shift(1)
    df['rolling_mean_3h'] = df.groupby('date')['Solar Rad - W/m^2'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['prev_day_same_hour'] = df.groupby('hour')['Solar Rad - W/m^2'].shift(1)
    
    # Calculate environmental features
    df['cloud_impact'] = 1 - (df['Solar Rad - W/m^2'] / df['clear_sky_radiation'].clip(lower=1))
    df['solar_trend'] = df.groupby('date')['Solar Rad - W/m^2'].diff()
    
    return df

class ImprovedSolarPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ImprovedSolarPredictor, self).__init__()
        
        # Feature dimensions - must match base_features length
        self.ramp_features = 3     # First 3 features
        self.history_features = 3  # Next 3 features
        self.env_features = 4      # Last 4 features
        self.time_features = 2     # Time features added separately
        
        print(f"\nNetwork Architecture:")
        print(f"Ramp features: {self.ramp_features}")
        print(f"History features: {self.history_features}")
        print(f"Environmental features: {self.env_features}")
        print(f"Time features: {self.time_features}")
        print(f"Total features: {self.ramp_features + self.history_features + self.env_features + self.time_features}")
        
        # Ramp features network
        self.ramp_net = nn.Sequential(
            nn.Linear(self.ramp_features, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        
        # Historical features network
        self.history_net = nn.Sequential(
            nn.Linear(self.history_features, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2))
        
        # Environmental features network
        self.env_net = nn.Sequential(
            nn.Linear(self.env_features, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2))
        
        # Time features network
        self.time_net = nn.Sequential(
            nn.Linear(self.time_features, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4))
        
        # Combination network
        combined_dim = (hidden_dim // 2) * 3 + (hidden_dim // 4)  # All features combined
        self.final_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Ensure non-negative output
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, weather_features, time_features):
        try:
            # Split weather features
            ramp = weather_features[:, :self.ramp_features]
            history = weather_features[:, self.ramp_features:self.ramp_features + self.history_features]
            env = weather_features[:, self.ramp_features + self.history_features:]
            
            # Process each feature group
            ramp_out = self.ramp_net(ramp)
            history_out = self.history_net(history)
            env_out = self.env_net(env)
            time_out = self.time_net(time_features)
            
            # Combine all features
            combined = torch.cat([ramp_out, history_out, env_out, time_out], dim=1)
            
            # Final prediction
            output = self.final_net(combined)
            
            return output
            
        except Exception as e:
            print(f"\nError in forward pass: {str(e)}")
            print(f"Input shapes:")
            print(f"Weather features: {weather_features.shape}")
            print(f"Time features: {time_features.shape}")
            traceback.print_exc()
            return None

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
        
    def forward(self, pred, target, time_features):
        try:
            # Convert inputs to float32 for better numerical stability
            pred = pred.float()
            target = target.float().unsqueeze(1)
            time_features = time_features.float()
            
            # Calculate hour of day
            hour = torch.atan2(time_features[:, 0], time_features[:, 1])
            hour = (hour / np.pi * 12 + 12) % 24
            
            # Base loss with epsilon to prevent division by zero
            epsilon = 1e-8
            mse_loss = self.mse(pred, target)
            mae_loss = self.mae(pred, target)
            base_loss = 0.7 * mse_loss + 0.3 * mae_loss
            
            # Clip predictions to prevent extreme values
            pred_clipped = torch.clamp(pred, min=0.0, max=1500.0)
            
            # Morning ramp-up penalties
            morning_ramp = (hour >= 6) & (hour < 10)
            morning_penalty = torch.where(
                morning_ramp.unsqueeze(1) & (pred_clipped < target),
                torch.abs(target - pred_clipped) * 8.0,
                torch.zeros_like(pred_clipped)
            )
            
            # Peak hours penalties
            peak_hours = (hour >= 10) & (hour <= 14)
            peak_penalty = torch.where(
                peak_hours.unsqueeze(1),
                torch.abs(target - pred_clipped) * 5.0,
                torch.zeros_like(pred_clipped)
            )
            
            # Night hours penalties
            night_hours = (hour < 6) | (hour >= 18)
            night_penalty = torch.where(
                night_hours.unsqueeze(1),
                pred_clipped * 10.0,
                torch.zeros_like(pred_clipped)
            )
            
            # Rapid changes penalty
            diff_penalty = torch.abs(torch.diff(pred_clipped, dim=0, prepend=pred_clipped[:1]))
            rapid_change_penalty = torch.where(
                diff_penalty > 100,
                diff_penalty * 0.1,
                torch.zeros_like(diff_penalty)
            )
            
            # Combine all penalties with gradient clipping
            total_loss = (
                torch.clamp(base_loss.mean(), max=1e6) +
                torch.clamp(morning_penalty.mean(), max=1e6) +
                torch.clamp(peak_penalty.mean(), max=1e6) +
                torch.clamp(night_penalty.mean(), max=1e6) +
                torch.clamp(rapid_change_penalty.mean(), max=1e6))
            
            return total_loss
            
        except Exception as e:
            print(f"Error in loss calculation: {str(e)}")
            traceback.print_exc()
            return None

def train_model(X_train, y_train, X_test, y_test, scaler_y, epochs=200):
    try:
        input_dim = X_train.shape[1]
        print(f"Input dimension: {input_dim}")
        
        # Convert to tensors and ensure no NaN values
        X_train_tensor = torch.FloatTensor(np.nan_to_num(X_train, nan=0.0))
        y_train_tensor = torch.FloatTensor(np.nan_to_num(y_train, nan=0.0)).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(np.nan_to_num(X_test, nan=0.0))
        y_test_tensor = torch.FloatTensor(np.nan_to_num(y_test, nan=0.0)).reshape(-1, 1)
        
        # Split features
        X_train_weather = X_train_tensor[:, :-2]
        X_train_time = X_train_tensor[:, -2:]
        X_test_weather = X_test_tensor[:, :-2]
        X_test_time = X_test_tensor[:, -2:]
        
        # Create datasets
        train_dataset = TensorDataset(X_train_weather, X_train_time, y_train_tensor)
        test_dataset = TensorDataset(X_test_weather, X_test_time, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = ImprovedSolarPredictor(input_dim)
        criterion = nn.MSELoss()
        
        # Use AdamW optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.005,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            div_factor=10,
            final_div_factor=100
        )
        
        early_stopping = EarlyStopping(patience=20)
        best_model = None
        best_val_loss = float('inf')
        corrections = None  # Initialize corrections
        
        print("\nTraining Progress:")
        print("Epoch | Train Loss | Val Loss | Correlation | R² Score")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_weather, batch_time, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_weather, batch_time)
                if outputs is None:
                    continue
                    
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_weather, batch_time, batch_y in val_loader:
                    outputs = model(batch_weather, batch_time)
                    if outputs is None:
                        continue
                        
                    val_loss += criterion(outputs, batch_y).item()
                    val_batches += 1
                    
                    # Store predictions and targets
                    all_preds.extend(outputs.cpu().numpy().flatten())
                    all_targets.extend(batch_y.cpu().numpy().flatten())
            
            # Calculate metrics
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                if len(all_preds) > 0 and len(all_targets) > 0:
                    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
                    r2 = r2_score(all_targets, all_preds)
                    
                    print(f"{epoch+1:3d} | {avg_train_loss:9.4f} | {avg_val_loss:8.4f} | "
                          f"{correlation:10.4f} | {r2:8.4f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model = copy.deepcopy(model)
                        print("Best model updated!")
                    
                    if early_stopping(avg_val_loss):
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
        
        # Return exactly three values: model, corrections, and completed epochs
        return best_model, corrections, epoch + 1
        
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        traceback.print_exc()
        return None, None, 0

def predict_hourly_radiation(model, features, scaler_X, scaler_y, date, base_features):
    """Make predictions for all hours in a day"""
    predictions = []
    timestamps = []
    
    for hour in range(24):
        # Create timestamp
        timestamp = pd.Timestamp.combine(date, pd.Timestamp(f"{hour:02d}:00").time())
        
        # Make prediction for this hour
        prediction = predict_for_hour(model, hour, features, scaler_X, scaler_y, base_features)
        
        # Validate prediction
        prediction = validate_predictions(prediction, hour)
        
        predictions.append(prediction)
        timestamps.append(timestamp)
    
    # Create DataFrame with predictions
    results_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Hour': [t.hour for t in timestamps],
        'Predicted Solar Radiation (W/m²)': predictions
    })
    
    # Save predictions to CSV
    results_df.to_csv('figures/hourly_predictions.csv', index=False)
    
    # Create prediction plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Hour'], results_df['Predicted Solar Radiation (W/m²)'], 
             marker='o', linestyle='-', linewidth=2)
    plt.title(f'Predicted Solar Radiation for {date.strftime("%Y-%m-%d")}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Solar Radiation (W/m²)')
    plt.grid(True)
    plt.xticks(range(24))
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('figures/hourly_predictions.png')
    plt.close()
    
    return results_df

def analyze_feature_distributions(data):
    """Analyze feature distributions"""
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        if col != 'hour':
            sns.histplot(data=data[col], kde=True, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col}', pad=20)
            
            # Rotate x-axis labels for better readability
            axes[idx].tick_params(axis='x', rotation=45)
            
            skewness = stats.skew(data[col].dropna())
            kurtosis = stats.kurtosis(data[col].dropna())
            # Moved text box to upper right
            axes[idx].text(0.95, 0.95, 
                         f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}', 
                         transform=axes[idx].transAxes,
                         bbox=dict(facecolor='white', alpha=0.8),
                         verticalalignment='top',
                         horizontalalignment='right')
    
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout(h_pad=1.0, w_pad=0.5)
    plt.savefig('figures/feature_distributions.png')
    plt.close()

def predict_for_hour(model, hour, features, scaler_X, scaler_y, base_features):
    """Make prediction for a specific hour"""
    # Create feature vector
    feature_vector = []
    
    # Add base features
    for feature in base_features:
        feature_vector.append(features[feature][hour])
    
    # Add engineered features
    hour_sin = np.sin(2 * np.pi * hour/24)
    hour_cos = np.cos(2 * np.pi * hour/24)
    uv_squared = features['UV Index'][hour] ** 2
    uv_temp_interaction = features['UV Index'][hour] * features['Average Temperature'][hour]
    humidity_temp_interaction = features['Average Humidity'][hour] * features['Average Temperature'][hour]
    
    feature_vector.extend([
        hour_sin,
        hour_cos,
        uv_squared,
        uv_temp_interaction,
        humidity_temp_interaction
    ])
    
    # Convert to numpy array and reshape
    X = np.array([feature_vector])
    
    # Scale features
    X_scaled = scaler_X.transform(X)
    
    # Convert to tensor and reshape for RNN
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor, torch.FloatTensor([[hour_sin, hour_cos]]))
        prediction = scaler_y.inverse_transform(prediction.numpy().reshape(-1, 1))
    
    return prediction[0][0]

def feature_selection(data):
    """Select most important features based on mutual information scores"""
    features = data.drop(['hour', 'Solar Rad - W/m^2'], axis=1)
    target = data['Solar Rad - W/m^2']  # Direct solar radiation
    
    mi_scores = mutual_info_regression(features, target)
    important_features = pd.DataFrame({
        'feature': features.columns,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
    # Select top features based on importance threshold
    threshold = 0.3  # Adjust based on mutual information scores
    selected_features = important_features[important_features['importance'] > threshold]['feature'].tolist()
    
    return selected_features

class CloudAwareRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(CloudAwareRNN, self).__init__()
        
        # Cloud detection branch
        self.cloud_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),               
        )
        
        # Main radiation prediction branch
        self.radiation_branch = nn.LSTM(
            input_dim, 
            hidden_dim,  
            layer_dim, 
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for cloud impact
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Combined output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)           
        )
        
    def forward(self, x):
        # Cloud feature processing
        cloud_features = self.cloud_branch(x)
        
        # Main radiation prediction
        radiation_out, _ = self.radiation_branch(x)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(radiation_out, radiation_out, radiation_out)
        
        # Combine features
        combined = torch.cat((attn_out[:, -1, :], cloud_features), dim=1)
        
        # Final prediction
        out = self.fc(combined)
        return out

class CloudAwareLoss(nn.Module):
    def __init__(self, cloud_weight=0.3):
        super(CloudAwareLoss, self).__init__()
        self.cloud_weight = cloud_weight
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, cloud_features):
        # Base MSE loss
        base_loss = self.mse(pred, target)
        
        # Additional loss for sudden changes
        sudden_change_mask = cloud_features['sudden_drop'] | cloud_features['sudden_increase']
        if sudden_change_mask.any():
            cloud_loss = self.mse(
                pred[sudden_change_mask],
                target[sudden_change_mask]
            )
            return base_loss + self.cloud_weight * cloud_loss
        
        return base_loss

def train_cloud_aware_model(model, train_loader, val_loader, epochs=600):
    """Training with cloud awareness"""
    criterion = CloudAwareLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, 
                          steps_per_epoch=len(train_loader))
    
    cloud_features = CloudFeatures()
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            # Calculate cloud features for batch
            cloud_feat = cloud_features.calculate_cloud_features(batch_X)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y, cloud_feat)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    return model

def predict_with_cloud_awareness(model, features, hour):
    """Make predictions with cloud awareness"""
    cloud_features = CloudFeatures()
    cloud_feat = cloud_features.calculate_cloud_features(features)
    
    model.eval()
    with torch.no_grad():
        prediction = model(features)
        
        # Adjust prediction based on cloud features
        if cloud_feat['sudden_drop'].item() > 0:
            prediction *= 0.7  # Reduce prediction for sudden drops
        elif cloud_feat['sudden_increase'].item() > 0:
            prediction *=1.3  # Increase prediction for sudden clearings
            
        # Ensure physical constraints
        prediction = torch.clamp(prediction, min=0, max=1200)
        
    return prediction.item()

def post_process_predictions(predictions, actual_values=None, cloud_cover=None):
    processed = predictions.copy()
    
    # Apply cloud cover adjustment if available
    if cloud_cover is not None:
        cloud_factor = 1.0 - (cloud_cover * 0.01)  # Convert percentage to factor
        processed *= cloud_factor
    
    # Physical constraints
    processed = np.clip(processed, 0, 1200)  # Max realistic solar radiation
    
    # Time-based corrections
    hour = np.arange(len(processed)) % 24
    night_hours = (hour < 6) | (hour > 18)
    processed[night_hours] = 0
    
    # Smooth extreme changes
    for i in range(1, len(processed)):
        max_change = 150  # Max allowed change between hours
        if abs(processed[i] - processed[i-1]) > max_change:
            direction = np.sign(processed[i] - processed[i-1])
            processed[i] = processed[i-1] + direction * max_change
    
    # Adjust based on clear sky model
    clear_sky = calculate_clear_sky_radiation(
        hour, 
        DAVAO_LATITUDE,  # Use constants instead of undefined variables
        DAVAO_LONGITUDE,
        pd.Timestamp.now().date()  # Use current date if not provided
    )
    processed = np.minimum(processed, clear_sky * 1.1)  # Allow 10% above clear sky
    
    # Ensemble with moving average for stability
    window = 3
    ma = np.convolve(processed, np.ones(window)/window, mode='same')
    processed = 0.7 * processed + 0.3 * ma
    
    return processed

def calculate_confidence_intervals(model, X_test, n_samples=100):
    """Calculate prediction confidence intervals using Monte Carlo Dropout"""
    model.train()  # Enable dropout
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_test)
            predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    confidence_95 = 1.96 * std_pred
    
    return mean_pred, confidence_95

def predict_with_correction(model, X, hour, prev_value):
    """Make prediction with time-based corrections"""
    try:
        # Ensure X is a 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Scale the input features
        X_scaled = scaler_X.transform(X)
            
        # Split features - handle both numpy arrays and torch tensors
        if isinstance(X_scaled, torch.Tensor):
            weather_features = X_scaled[:, :-2]
            time_features = X_scaled[:, -2:]
        else:
            weather_features = torch.FloatTensor(X_scaled[:, :-2])
            time_features = torch.FloatTensor(X_scaled[:, -2:])
        
        # Get base prediction
        model.eval()
        with torch.no_grad():
            prediction = model(weather_features, time_features)
            prediction = prediction.numpy()[0][0] if isinstance(prediction, torch.Tensor) else prediction[0][0]
            
            # Inverse transform the prediction
            prediction = scaler_y.inverse_transform([[prediction]])[0][0]
        
        # Print debug information
        print(f"\nDebug Information:")
        print(f"Input features shape: {X.shape}")
        print(f"Weather features shape: {weather_features.shape}")
        print(f"Time features shape: {time_features.shape}")
        print(f"Raw prediction: {prediction:.2f}")
        print(f"Previous value: {prev_value:.2f}")
        print(f"Hour: {hour}")
        
        # Basic validation for daylight hours
        if 6 <= hour <= 18:
            # Ensure prediction is at least 20% of previous value during daylight
            min_value = prev_value * 0.2
            prediction = max(prediction, min_value)
            
            # Calculate clear sky radiation for maximum bound
            clear_sky = calculate_clear_sky_radiation(
                hour, 
                DAVAO_LATITUDE, 
                DAVAO_LONGITUDE, 
                datetime.now().date()
            )
            # Allow up to 110% of clear sky radiation
            max_value = clear_sky * 1.1
            prediction = min(prediction, max_value)
        
        # Validate prediction
        prediction = validate_predictions(prediction, hour)
        
        # Apply time-based corrections
        if 6 <= hour <= 9:  # Morning hours
            prediction = adjust_morning_prediction(prediction, hour, prev_value)
        elif 14 <= hour <= 17:  # Afternoon hours
            prediction = adjust_afternoon_prediction(prediction, hour, prev_value)
            
        # Ensure reasonable change from previous value
        max_increase = prev_value * 2.0  # Maximum 100% increase
        max_decrease = prev_value * 0.3  # Maximum 70% decrease
        prediction = min(max(prediction, max_decrease), max_increase)
        
        print(f"Final adjusted prediction: {prediction:.2f}")
        return prediction
            
    except Exception as e:
        print(f"Error in predict_with_correction: {str(e)}")
        traceback.print_exc()
        return None

def add_peak_features(df):
    """Add features specifically for peak radiation prediction"""
    
    # Calculate clear sky index for peak hours
    df['peak_hour'] = (df['hour'] >= 10) & (df['hour'] <= 14)
    df['clear_sky_ratio'] = df['Solar Rad - W/m^2'] / df['clear_sky_radiation'].clip(lower=1)
    
    # Add features for peak radiation periods
    df['peak_temp_ratio'] = df['Average Temperature'] / df['Average Temperature'].rolling(24).max()
    df['peak_humidity_impact'] = 1 - (df['Average Humidity'] / 100)
    
    # Add interaction terms for peak hours
    df.loc[df['peak_hour'], 'peak_features'] = (    
        df.loc[df['peak_hour'], 'peak_temp_ratio'] * 
        df.loc[df['peak_hour'], 'peak_humidity_impact'] * 
        df.loc[df['peak_hour'], 'clear_sky_ratio']
    )
    
    return df

def residual_based_correction(model, X_train, y_train):
    """Create correction factors based on residual patterns"""
    
    # Get base predictions
    model.eval()
    with torch.no_grad():
        base_pred = model(X_train).numpy()
    
    # Calculate residuals
    residuals = y_train.numpy() - base_pred
    
    # Create correction bins
    bins = np.linspace(0, 1000, 20)  # 20 bins from 0 to 1000 W/m²
    corrections = {}
    
    # Calculate mean correction for each bin
    for i in range(len(bins)-1):
        mask = (base_pred >= bins[i]) & (base_pred < bins[i+1])
        if mask.any():
            corrections[i] = np.mean(residuals[mask])
    
    return corrections, bins

def apply_residual_correction(predictions, corrections, bins):
    """Apply correction factors to predictions"""
    corrected = predictions.copy()
    
    for i in range(len(bins)-1):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if mask.any() and i in corrections:
            corrected[mask] += corrections[i]
    
    return corrected

class HeteroscedasticLoss(nn.Module):
    def __init__(self):
        super(HeteroscedasticLoss, self).__init__()
        
    def forward(self, pred, target):
        # Estimate variance based on prediction magnitude
        predicted_variance = 0.1 + 0.9 * torch.sigmoid(pred/500)
        
        # Calculate weighted loss
        squared_error = torch.pow(pred - target, 2)
        loss = (squared_error / predicted_variance) + torch.log(predicted_variance)
        
        return loss.mean()

def train_with_residual_awareness(model, train_loader, val_loader, epochs=600):
    """Training loop with residual-aware components"""
    
    criterion = HeteroscedasticLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize correction factors
    corrections = None
    bins = None
    update_interval = 50
    
    # Get training data from loader
    X_train_data = []
    y_train_data = []
    for batch_X, batch_y in train_loader:
        X_train_data.append(batch_X)
        y_train_data.append(batch_y)
    X_train = torch.cat(X_train_data)
    y_train = torch.cat(y_train_data)
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Apply current corrections if available
            if corrections is not None:
                outputs = apply_residual_correction(outputs, corrections, bins)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Update correction factors periodically
        if epoch % update_interval == 0:
            corrections, bins = residual_based_correction(model, X_train, y_train)
    
    return model, corrections, bins

def predict_with_uncertainty(model, X, corrections=None, bins=None):
    """Make predictions with uncertainty estimates"""
    
    model.eval()
    with torch.no_grad():
        # Base prediction
        pred = model(X)
        
        # Apply residual correction
        if corrections is not None and bins is not None:
            pred = apply_residual_correction(pred, corrections, bins)
        
        # Estimate uncertainty
        uncertainty = 0.1 + 0.9 * torch.sigmoid(pred/500)
        
        return pred, uncertainty

def validate_data(df):
    """Validate input data"""
    required_columns = [
        'Date & Time', 'Average Barometer', 'Average Temperature',
        'Average Humidity', 'Average Dew Point', 'Average Wet Bulb',
        'Avg Wind Speed - km/h', 'Average Wind Chill', 'Heat Index',
        'Average THW Index', 'Average THSW Index', 'UV Index',
        'Solar Rad - W/m^2',
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for invalid values
    if (df['Solar Rad - W/m^2'] < 0).any():
        raise ValueError("Negative solar radiation values found")
    
    return True

def validate_predictions(prediction, hour):
    """Enhanced validation of predictions based on time of day"""
    try:
        # Night hours (0-4, 19-23) should be near zero - more lenient with early morning
        if hour < 4 or hour >= 19:
            return 0.0
        
        # Get clear sky radiation for the hour
        clear_sky = calculate_clear_sky_radiation(
            hour, 
            DAVAO_LATITUDE, 
            DAVAO_LONGITUDE, 
            datetime.now().date()
        )
        
        # Minimum values based on hour - much more lenient in early morning
        min_values = {
            4: 0,     # Pre-dawn
            5: 0,     # Dawn beginning
            6: 10,    # Dawn
            7: 30,    # Early morning
            8: 100,   # Morning
            9: 200,   # Late morning
            10: 300,  # Late morning
            11: 400,  # Near noon
            12: 400,  # Noon
            13: 400,  # Early afternoon
            14: 300,  # Mid afternoon
            15: 200,  # Late afternoon
            16: 100,  # Evening
            17: 50,   # Dusk
            18: 10    # Late dusk
        }
        
        # Get minimum value for current hour, with more gradual limits
        min_value = min_values.get(hour, 0)
        
        # Special handling for early morning and late afternoon
        if 4 <= hour <= 7:  # Early morning hours
            # Use a percentage of the prediction instead of fixed minimum
            min_value = min(min_value, prediction * 0.1)  # Allow down to 10% of predicted value
            # Also ensure we're not forcing too high a minimum during early hours
            min_value = min(min_value, clear_sky * 0.3)  # Cap at 30% of clear sky
        elif 16 <= hour <= 18:  # Late afternoon
            min_value = min(min_value, clear_sky * 0.3)
        
        # Maximum value based on clear sky and hour
        if 5 <= hour <= 7:  # Early morning
            max_value = clear_sky * 1.2  # More lenient in early morning
        else:
            max_value = clear_sky * 1.1  # Normal hours
        
        # Clip prediction between min and max values
        prediction = np.clip(prediction, min_value, max_value)
        
        return float(prediction)
        
    except Exception as e:
        print(f"Error in validate_predictions: {str(e)}")
        return 0.0

def detect_weather_pattern(df):
    """Detect weather patterns that might affect predictions"""
    patterns = {
        'rainy': (df['Average Humidity'] > 85) & (df['Average Temperature'] < 25),
        'clear': (df['Average Humidity'] < 70) & (df['UV Index'] > 8),
        'cloudy': (df['Average Humidity'] > 75) & (df['UV Index'] < 5)
    }
    
    return patterns

def prepare_sequence(X, sequence_length=1):
    """Prepare sequential data for LSTM"""
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
    return np.array(sequences)

def extract_minute_data(data_path):
    """Extract 5-minute interval data from raw dataset"""
    try:
        # Read the CSV file
        df = pd.read_csv(data_path)
        
        # Convert timestamp and validate
        df['timestamp'] = pd.to_datetime(df['Date & Time'], format='%m/%d/%Y %H:%M')
        
        # Remove future dates
        current_time = pd.Timestamp.now()
        future_dates = df['timestamp'] > current_time
        if future_dates.any():
            print(f"Warning: Removing {future_dates.sum()} future timestamps")
            df = df[~future_dates].copy()
        
        # Print data range for validation
        print("\nData range in dataset:")
        print(f"Start time: {df['timestamp'].min()}")
        print(f"End time: {df['timestamp'].max()}")
        print(f"Total records: {len(df)}")
        
        if df.empty:
            raise ValueError("Dataset is empty")
            
        # Extract components
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Calculate clear sky radiation for each timestamp
        df['clear_sky_radiation'] = df.apply(
            lambda row: calculate_clear_sky_radiation(
                row['hour'] + row['minute']/60,  # Convert to decimal hours
                DAVAO_LATITUDE,
                DAVAO_LONGITUDE,
                row['timestamp'].date()
            ),
            axis=1
        )
        
        # Calculate additional features
        df['cloud_impact'] = 1 - (df['Solar Rad - W/m^2'] / df['clear_sky_radiation'].clip(lower=1))
        df['clear_sky_ratio'] = df['Solar Rad - W/m^2'] / df['clear_sky_radiation'].clip(lower=1)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        print(f"Error in extract_minute_data: {str(e)}")
        traceback.print_exc()
        return None

def save_results(results_df, figure_path, csv_path):
    """Save results with improved plotting and data handling"""
    try:
        # Ensure figures directory exists
        if not os.path.exists('figures'):
            os.makedirs('figures')
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot actual values with better formatting
        plt.plot(results_df['Hour'], results_df['Actual Values'], 
                marker='o', linestyle='-', linewidth=2, 
                label='Actual Values', color='blue')
        
        # Plot next hour prediction with larger marker
        next_hour_mask = results_df['Next Hour Prediction'].notna()
        if next_hour_mask.any():
            plt.plot(results_df.loc[next_hour_mask, 'Hour'], 
                    results_df.loc[next_hour_mask, 'Next Hour Prediction'],
                    marker='*', markersize=20, color='red', linestyle='none',
                    label='Next Hour Prediction')
        
        plt.title('Solar Radiation Predictions vs Actual Values')
        plt.xlabel('Hour of Day')
        plt.ylabel('Solar Radiation (W/m²)')
        plt.grid(True)
        plt.xticks(range(24))
        plt.ylim(bottom=0, top=1200)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save CSV
        results_df.to_csv(csv_path, index=False, float_format='%.2f')
        
        print(f"Successfully saved results to {csv_path}")
        print(f"Successfully saved plot to {figure_path}")
            
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        traceback.print_exc()

def analyze_recent_patterns(minute_data, last_timestamp, lookback_minutes=30):
    """Enhanced pattern detection focusing on rapid changes"""
    try:
        # Get recent data with longer lookback for trend analysis
        start_time = last_timestamp - pd.Timedelta(minutes=lookback_minutes)
        recent_data = minute_data[
            (minute_data['timestamp'] >= start_time) & 
            (minute_data['timestamp'] <= last_timestamp)
        ]
        
        if recent_data.empty:
            print("Warning: No recent data available for pattern analysis")
            return None
            
        # Sort and calculate rolling means for smoother analysis
        recent_data = recent_data.sort_values('timestamp')
        recent_data['radiation_rolling'] = recent_data['Solar Rad - W/m^2'].rolling(3, min_periods=1).mean()
        recent_data['rolling_mean_6h'] = recent_data['Solar Rad - W/m^2'].rolling(6, min_periods=1).mean()
        recent_data['rolling_std_3h'] = recent_data['Solar Rad - W/m^2'].rolling(3, min_periods=1).std()
        
        # Initialize patterns with default values
        patterns = {
            'rapid_drop': False,
            'cloud_cover': False,
            'high_variability': False,
            'trend': 'stable',
            'trend_magnitude': 0.0,
            'recent_radiation': float(recent_data['Solar Rad - W/m^2'].iloc[-1]),
            'radiation_slope': 0.0,
            'likely_rain': False
        }
        
        # Weather indicators with more sensitive thresholds
        humidity_rising = (recent_data['Average Humidity'].diff().mean() > 0.2)  # Even more sensitive
        high_humidity = (recent_data['Average Humidity'].mean() > 75)  # Lower threshold
        pressure_dropping = (recent_data['Average Barometer'].diff().mean() < -0.005)  # More sensitive
        temp_dropping = (recent_data['Average Temperature'].diff().mean() < -0.05)  # More sensitive
        
        # Additional indicators
        dew_point_close = (recent_data['Average Temperature'] - recent_data['Average Dew Point']).mean() < 4
        wind_speed_rising = (recent_data['Avg Wind Speed - km/h'].diff().mean() > 0.3)
        pressure_trend = recent_data['Average Barometer'].diff().rolling(3, min_periods=1).mean().iloc[-1]
        
        # Calculate radiation metrics safely
        radiation_values = recent_data['radiation_rolling'].fillna(method='ffill').values
        raw_radiation = recent_data['Solar Rad - W/m^2'].fillna(method='ffill').values
        
        if len(radiation_values) > 2:
            # Calculate changes
            radiation_change = float(radiation_values[-1] - radiation_values[0])
            patterns['trend_magnitude'] = radiation_change
            
            # Short-term change calculation
            recent_values = raw_radiation[-min(3, len(raw_radiation)):]
            if len(recent_values) > 1 and recent_values[0] != 0:
                short_term_change = (recent_values[-1] - recent_values[0]) / recent_values[0]
            else:
                short_term_change = 0.0
            
            # Variability calculation
            recent_window = radiation_values[-min(5, len(radiation_values)):]
            radiation_variability = np.std(recent_window) / (np.mean(recent_window) + 1)
            patterns['high_variability'] = radiation_variability > 0.12  # More sensitive
            
            # Enhanced cloud detection conditions
            cloud_indicators = [
                high_humidity and temp_dropping,
                humidity_rising and pressure_dropping,
                dew_point_close and humidity_rising,
                wind_speed_rising and humidity_rising,
                radiation_variability > 0.12,
                pressure_trend < -0.01,  # Significant pressure drop
                short_term_change < -0.1  # 10% drop in short term
            ]
            
            patterns['cloud_cover'] = any(cloud_indicators)
            
            # Enhanced rain prediction with more conditions
            rain_indicators = [
                patterns['cloud_cover'] and high_humidity and pressure_dropping,
                patterns['cloud_cover'] and dew_point_close,
                high_humidity and temp_dropping and pressure_dropping,
                dew_point_close and pressure_dropping and humidity_rising,
                pressure_trend < -0.02 and humidity_rising  # Sharp pressure drop with rising humidity
            ]
            
            patterns['likely_rain'] = any(rain_indicators)
            
            # Trend detection with more sensitivity
            if radiation_change < -20 or short_term_change < -0.1:  # More sensitive thresholds
                patterns['trend'] = 'decreasing'
                if radiation_change < -50 or short_term_change < -0.15:
                    patterns['rapid_drop'] = True
            elif radiation_change > 20:
                patterns['trend'] = 'increasing'
            
            # Calculate slope
            time_intervals = np.arange(len(radiation_values))
            slope, _ = np.polyfit(time_intervals, radiation_values, 1)
            patterns['radiation_slope'] = float(slope)
            
            # Add these new indicators
            # 1. UV Index stability
            uv_stability = recent_data['UV Index'].diff().abs()
            uv_unstable = uv_stability.mean() > 0.5
            
            # 2. Short-interval radiation changes (5-minute windows)
            radiation_5min_changes = recent_data['Solar Rad - W/m^2'].diff(periods=1)
            rapid_fluctuation = radiation_5min_changes.abs().max() > 100
            
            # 3. Pressure acceleration (rate of pressure change)
            pressure_acceleration = recent_data['Average Barometer'].diff().diff()
            accelerating_pressure_drop = pressure_acceleration.mean() < -0.001
            
            # Enhanced cloud detection conditions
            cloud_indicators = [
                # ... existing indicators ...
                uv_unstable,
                rapid_fluctuation,
                accelerating_pressure_drop,
                # New specific condition for sharp drops
                any(radiation_5min_changes < -200),  # Detect 200+ W/m² drops
                pressure_trend < -0.1 and uv_unstable  # Combine pressure and UV instability
            ]
            
            # Print additional diagnostics
            print("\nShort-term indicators:")
            print(f"UV stability (diff mean): {uv_stability.mean():.3f}")
            print(f"Max 5-min radiation change: {radiation_5min_changes.min():.1f} W/m²")
            print(f"Pressure acceleration: {pressure_acceleration.mean():.4f}")
            
            # ... rest of the code ...
            
        return patterns
        
    except Exception as e:
        print(f"Error in analyze_recent_patterns: {str(e)}")
        return None

def adjust_prediction(pred, similar_cases_data, current_patterns):
    """Adjust prediction with enhanced consideration of historical low values"""
    if not similar_cases_data or current_patterns is None:
        return pred
        
    similar_cases, pattern_weights = similar_cases_data
    adjusted_pred = pred
    
    # Get the recent radiation value and patterns
    recent_radiation = current_patterns.get('recent_radiation', pred)
    
    # Check for low radiation cases in history
    if pattern_weights['low_radiation_cases']:
        low_rad_avg = np.mean(pattern_weights['low_radiation_cases'])
        low_rad_count = len(pattern_weights['low_radiation_cases'])
        
        # If we have multiple low radiation cases, be more conservative
        if low_rad_count >= 2:
            print(f"Found {low_rad_count} historical cases with low radiation")
            # Use weighted average favoring lower values
            adjusted_pred = (low_rad_avg * 0.6 + adjusted_pred * 0.4)
            print(f"Adjusting prediction based on historical low values: {adjusted_pred:.2f} W/m²")
    
    # Apply existing adjustments
    if current_patterns['likely_rain']:
        adjusted_pred = min(adjusted_pred * 0.3, recent_radiation * 0.5)
        print("Rain likely - applying severe reduction")
    elif current_patterns['cloud_cover']:
        if current_patterns['high_variability']:
            adjusted_pred = min(adjusted_pred * 0.5, recent_radiation * 0.7)
            print("Heavy cloud cover detected - applying significant reduction")
        else:
            adjusted_pred = min(adjusted_pred * 0.7, recent_radiation * 0.9)
            print("Moderate cloud cover detected - applying moderate reduction")
    
    # Consider similar historical cases with pressure drops
    pressure_cases = [x for x in pattern_weights['similar_pressure_cases'] if x is not None]
    if pressure_cases:
        pressure_avg = np.mean(pressure_cases)
        if pressure_avg < adjusted_pred * 0.8:  # If historical cases show lower values
            adjusted_pred = (pressure_avg * 0.7 + adjusted_pred * 0.3)
            print(f"Adjusting for historical pressure patterns: {adjusted_pred:.2f} W/m²")
    
    # Ensure prediction is physically reasonable
    adjusted_pred = max(0, min(adjusted_pred, 1200))
    
    print("\nPrediction adjustment details:")
    print(f"Original prediction: {pred:.2f} W/m²")
    print(f"Recent radiation: {recent_radiation:.2f} W/m²")
    print(f"Final adjusted prediction: {adjusted_pred:.2f} W/m²")
    
    return adjusted_pred

def find_similar_historical_patterns(data, target_hour, target_date, lookback_days=120):
    try:
        start_date = target_date - pd.Timedelta(days=lookback_days)
        
        # Get historical data for the same hour and surrounding hours
        historical_data = data[
            (data['timestamp'].dt.date >= start_date) & 
            (data['timestamp'].dt.date < target_date)
        ].copy()
        
        # Get current conditions
        current_conditions = data[data['timestamp'].dt.date == target_date]
        if current_conditions.empty or historical_data.empty:
            return None
        
        # Get current weather parameters
        current_hour = target_hour
        current_weather = current_conditions[
            current_conditions['hour'] == current_hour - 1
        ].iloc[0]
        
        similar_days = []
        
        # Enhanced similarity calculations
        for date in historical_data['timestamp'].dt.date.unique():
            day_data = historical_data[historical_data['timestamp'].dt.date == date]
            
            # Skip if insufficient data
            if len(day_data) < 24:
                continue
            
            target_hour_data = day_data[day_data['hour'] == target_hour]
            if target_hour_data.empty:
                continue
            
            # Get previous hours' pattern
            morning_data = day_data[
                (day_data['hour'] >= 6) & 
                (day_data['hour'] <= target_hour)]
            
            if len(morning_data) >= 2:
                ramp_rate = np.diff(morning_data['Solar Rad - W/m^2'].values).mean()
            else:
                continue
            
            # Compare weather conditions with enhanced weighting
            weather_at_hour = day_data[day_data['hour'] == current_hour - 1].iloc[0]
            
            # Weather similarity with more emphasis on key indicators
            weather_similarity = (
                (1 - abs(current_weather['Average Temperature'] - weather_at_hour['Average Temperature']) / 50) * 0.2 +
                (1 - abs(current_weather['Average Humidity'] - weather_at_hour['Average Humidity']) / 100) * 0.3 +
                (1 - abs(current_weather['UV Index'] - weather_at_hour['UV Index']) / 20) * 0.3 +
                (1 - abs(current_weather['cloud_impact'] - weather_at_hour['cloud_impact'])) * 0.2
            )
            
            # Enhanced clear sky ratio comparison
            current_clear_sky = max(1, current_weather.get('clear_sky_radiation', 1))
            historical_clear_sky = max(1, weather_at_hour.get('clear_sky_radiation', 1))
            
            current_ratio = current_weather['Solar Rad - W/m^2'] / current_clear_sky
            historical_ratio = weather_at_hour['Solar Rad - W/m^2'] / historical_clear_sky
            
            clear_sky_similarity = 1 - min(1, abs(current_ratio - historical_ratio))
            
            # Enhanced ramp rate comparison
            current_ramp = current_conditions[
                (current_conditions['hour'] >= 6) & 
                (current_conditions['hour'] <= current_hour)
            ]['Solar Rad - W/m^2'].diff().mean()
            
            if pd.isna(current_ramp):
                current_ramp = 0
            
            ramp_similarity = 1 - min(1, abs(current_ramp - ramp_rate) / 100)
            
            # Value-based similarity (new)
            value_diff = abs(target_hour_data['Solar Rad - W/m^2'].iloc[0] - current_weather['Solar Rad - W/m^2'])
            value_similarity = 1 - min(1, value_diff / 1000)
            
            # Calculate overall similarity with adjusted weights
            if 10 <= target_hour <= 14:  # Peak hours
                total_similarity = (
                    weather_similarity * 0.3 +
                    clear_sky_similarity * 0.3 +
                    ramp_similarity * 0.2 +
                    value_similarity * 0.2  # Added value similarity
                )
            else:
                total_similarity = (
                    weather_similarity * 0.4 +
                    clear_sky_similarity * 0.3 +
                    ramp_similarity * 0.3
                )
            
            # Get the actual value
            actual_value = target_hour_data['Solar Rad - W/m^2'].iloc[0]
            
            similar_days.append({
                'date': date,
                'similarity': total_similarity,
                'value': actual_value,
                'weather_similarity': weather_similarity,
                'clear_sky_similarity': clear_sky_similarity,
                'ramp_similarity': ramp_similarity,
                'value_similarity': value_similarity,  # Added
                'ramp_rate': ramp_rate
            })
        
        # Sort by similarity but prioritize higher values during peak hours
        if 10 <= target_hour <= 14:
            similar_days.sort(key=lambda x: (x['similarity'] * 0.7 + (x['value'] / 1000) * 0.3), reverse=True)
        else:
            similar_days.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Print analysis
        print("\nMost Similar Historical Days Analysis:")
        for i, day in enumerate(similar_days[:5]):
            print(f"\nPattern {i+1}:")
            print(f"Date: {day['date']}")
            print(f"Overall Similarity: {day['similarity']:.4f}")
            print(f"Weather Similarity: {day['weather_similarity']:.4f}")
            print(f"Clear Sky Similarity: {day['clear_sky_similarity']:.4f}")
            print(f"Ramp-up Similarity: {day['ramp_similarity']:.4f}")
            print(f"Ramp Rate: {day['ramp_rate']:.2f} W/m²/hour")
            print(f"Actual Value: {day['value']:.2f} W/m²")
        
        return similar_days
        
    except Exception as e:
        print(f"Error in find_similar_historical_patterns: {str(e)}")
        traceback.print_exc()
        return None

class WeatherPatternDetector:
    def __init__(self):
        self.cloud_indicators = {
            'humidity_threshold': 75,
            'pressure_drop_threshold': -0.05,
            'uv_drop_threshold': 2
        }
        
    def detect_patterns(self, historical_data, lookback_hours=3):
        """Enhanced weather pattern detection"""
        recent_data = historical_data.tail(lookback_hours)
        
        patterns = {
            'cloud_formation_likely': False,
            'clearing_likely': False,
            'stability': 'unstable',
            'confidence': 'low'
        }
        
        # Analyze rapid changes
        if len(recent_data) >= 2:
            # Humidity analysis
            humidity_trend = recent_data['Average Humidity'].diff().mean()
            humidity_level = recent_data['Average Humidity'].iloc[-1]
            
            # Pressure analysis
            pressure_trend = recent_data['Average Barometer'].diff().mean()
            
            # UV and radiation correlation
            uv_trend = recent_data['UV Index'].diff().mean()
            radiation_trend = recent_data['Solar Rad - W/m^2'].diff().mean()
            
            # Cloud formation indicators
            cloud_indicators = [
                humidity_trend > 0.2 and humidity_level > self.cloud_indicators['humidity_threshold'],
                pressure_trend < self.cloud_indicators['pressure_drop_threshold'],
                uv_trend < -self.cloud_indicators['uv_drop_threshold']
            ]
            
            # Clearing indicators
            clearing_indicators = [
                humidity_trend < -0.2,
                pressure_trend > 0.05,
                uv_trend > 1 and radiation_trend > 50
            ]
            
            patterns['cloud_formation_likely'] = sum(cloud_indicators) >= 2
            patterns['clearing_likely'] = sum(clearing_indicators) >= 2
            
            # Determine stability and confidence
            patterns['stability'] = 'stable' if abs(radiation_trend) < 50 else 'unstable'
            patterns['confidence'] = 'high' if abs(radiation_trend) > 100 else 'medium'
        
        return patterns

def adjust_morning_prediction(prediction, hour, prev_value, weather_patterns=None):
    """Adjust morning predictions based on patterns"""
    try:
        # Base multipliers for early morning hours
        base_multipliers = {6: 0.15, 7: 0.3, 8: 0.5, 9: 0.7}
        multiplier = base_multipliers.get(hour, 1.0)
        
        # Adjust multiplier based on weather patterns if available
        if weather_patterns is not None:
            if weather_patterns.get('clearing_likely', False):
                multiplier *= 1.3
            elif weather_patterns.get('cloud_formation_likely', False):
                multiplier *= 0.7
        
        # Apply multiplier and ensure reasonable limits
        adjusted_prediction = prediction * multiplier
        
        # Ensure prediction doesn't exceed reasonable limits
        if hour == 6:
            adjusted_prediction = min(max(adjusted_prediction, 10), 30)
        elif hour == 7:
            adjusted_prediction = min(max(adjusted_prediction, 50), 150)
        elif hour == 8:
            adjusted_prediction = min(max(adjusted_prediction, 100), 300)
        elif hour == 9:
            adjusted_prediction = min(max(adjusted_prediction, 200), 500)
        
        return adjusted_prediction
        
    except Exception as e:
        print(f"Error in adjust_morning_prediction: {str(e)}")
        return prediction

def adjust_afternoon_prediction(prediction, hour, prev_value, weather_patterns=None):
    """Adjust afternoon predictions based on patterns"""
    try:
        # Base decline factors
        base_decline = 0.8  # Less steep decline initially
        hours_past_peak = hour - 14  # Past 2 PM
        
        # Adjust decline based on weather patterns if available
        if weather_patterns is not None:
            if weather_patterns.get('cloud_formation_likely', False):
                base_decline = 0.7  # Steeper decline for cloudy conditions
            elif weather_patterns.get('clearing_likely', False):
                base_decline = 0.85  # Gentler decline for clear conditions
        
        # Calculate decline factor
        decline_factor = base_decline ** hours_past_peak
        
        # Apply decline
        adjusted_prediction = prediction * decline_factor
        
        # Ensure prediction doesn't exceed previous hour
        if prev_value > 0:
            max_allowed = prev_value * 0.9  # Maximum 90% of previous hour
            adjusted_prediction = min(adjusted_prediction, max_allowed)
        
        # Ensure reasonable minimum values
        if hour >= 17:  # Late afternoon
            adjusted_prediction = min(adjusted_prediction, 100)
        
        return max(0, adjusted_prediction)  # Ensure non-negative
        
    except Exception as e:
        print(f"Error in adjust_afternoon_prediction: {str(e)}")
        return prediction

def adjust_peak_prediction(prediction, hour, patterns, prev_value):
    """Adjust peak hour predictions"""
    if patterns['stability'] == 'unstable':
        if patterns['cloud_formation_likely']:
            return min(prediction, prev_value * 0.8)
        elif patterns['clearing_likely']:
            return max(prediction, prev_value * 1.2)
    return prediction

def adjust_afternoon_prediction(prediction, hour, patterns, prev_value):
    """Adjust afternoon predictions"""
    # Steeper decline in cloudy conditions
    if patterns['cloud_formation_likely']:
        decline_factor = 0.5
    else:
        decline_factor = 0.7
        
    hours_past_peak = hour - 14  # Past 2 PM
    prediction *= (decline_factor ** hours_past_peak)
    
    return min(prediction, prev_value * 0.9)  # Ensure declining trend

def analyze_early_warning_signs(data, last_timestamp):
    """Analyze early warning signs with more aggressive risk assessment"""
    try:
        # Get the last hour's data and previous data point
        last_hour_data = data[data['timestamp'].dt.hour == last_timestamp.hour].iloc[-1]
        prev_data = data.iloc[-2]
        
        risk_factors = {
            'high_risk': False,
            'risk_score': 0,
            'warning_signs': [],
            'critical_combinations': False
        }
        
        # 1. Enhanced Pressure Analysis
        pressure_trend = last_hour_data['Average Barometer'] - prev_data['Average Barometer']
        pressure_acceleration = pressure_trend - (prev_data['Average Barometer'] - data.iloc[-3]['Average Barometer'])
        
        if pressure_trend < -0.2:
            risk_factors['risk_score'] += 35
            risk_factors['warning_signs'].append(f"Significant pressure drop: {pressure_trend:.3f}")
        elif pressure_trend < -0.1:
            risk_factors['risk_score'] += 20
            risk_factors['warning_signs'].append(f"Moderate pressure drop: {pressure_trend:.3f}")
        
        if pressure_acceleration < -0.05:
            risk_factors['risk_score'] += 15
            risk_factors['warning_signs'].append(f"Accelerating pressure drop: {pressure_acceleration:.3f}")
            
        # 2. Enhanced Humidity Analysis
        current_humidity = last_hour_data['Average Humidity']
        humidity_trend = current_humidity - prev_data['Average Humidity']
        
        if current_humidity > 65:
            risk_score = min(int((current_humidity - 65) * 2), 30)
            risk_factors['risk_score'] += risk_score
            risk_factors['warning_signs'].append(f"Elevated humidity: {current_humidity:.1f}%")
        
        if humidity_trend > 0:
            risk_factors['risk_score'] += int(humidity_trend * 8)
            risk_factors['warning_signs'].append(f"Rising humidity: +{humidity_trend:.1f}%")
            
        # 3. Enhanced Temperature-Dew Point Analysis
        temp_dewpoint_spread = last_hour_data['Average Temperature'] - last_hour_data['Average Dew Point']
        if temp_dewpoint_spread < 5:
            risk_score = min(int((5 - temp_dewpoint_spread) * 15), 35)
            risk_factors['risk_score'] += risk_score
            risk_factors['warning_signs'].append(f"Small temp-dewpoint spread: {temp_dewpoint_spread:.1f}°C")
            
        # 4. Enhanced Wind Analysis
        wind_speed = last_hour_data['Avg Wind Speed - km/h']
        wind_change = wind_speed - prev_data['Avg Wind Speed - km/h']
        
        if wind_change > 2:
            risk_factors['risk_score'] += int(wind_change * 5)
            risk_factors['warning_signs'].append(f"Increasing wind: +{wind_change:.1f} km/h")
            
        # 5. Enhanced UV Index Analysis
        uv_index = last_hour_data['UV Index']
        expected_uv = get_expected_uv_for_hour(last_timestamp.hour)
        uv_ratio = uv_index / expected_uv if expected_uv > 0 else 1
        
        if uv_ratio < 0.9:
            risk_factors['risk_score'] += int((1 - uv_ratio) * 40)
            risk_factors['warning_signs'].append(f"Lower than expected UV: {uv_index} vs {expected_uv:.1f}")
            
        # 6. Critical Combinations Check
        critical_combinations = [
            (pressure_trend < -0.1 and humidity_trend > 0),
            (temp_dewpoint_spread < 4 and humidity_trend > 0),
            (pressure_acceleration < -0.05 and wind_change > 2),
            (uv_ratio < 0.9 and humidity_trend > 0),
            (pressure_trend < -0.15 and temp_dewpoint_spread < 5)
        ]
        
        if any(critical_combinations):
            risk_factors['critical_combinations'] = True
            risk_factors['risk_score'] = max(risk_factors['risk_score'], 75)
            risk_factors['warning_signs'].append("CRITICAL: Multiple high-risk indicators detected")
            
        # Set high risk flag if score is above threshold
        risk_factors['high_risk'] = risk_factors['risk_score'] >= 50
        
        print("\nEnhanced Early Warning Analysis:")
        print(f"Risk Score: {risk_factors['risk_score']}/100")
        print(f"Critical Combinations: {risk_factors['critical_combinations']}")
        print("Warning Signs Detected:")
        for warning in risk_factors['warning_signs']:
            print(f"- {warning}")
            
        return risk_factors
        
    except Exception as e:
        print(f"Error in analyze_early_warning_signs: {str(e)}")
        traceback.print_exc()
        return None

def get_expected_uv_for_hour(hour):
    """Get expected UV index for a given hour based on historical patterns"""
    # Simplified UV expectations for Davao (can be enhanced with historical data)
    uv_expectations = {
        6: 1, 7: 2, 8: 4, 9: 6, 10: 8, 11: 9, 12: 10, 
        13: 9, 14: 8, 15: 6, 16: 4, 17: 2, 18: 1
    }
    return uv_expectations.get(hour, 0)

def adjust_prediction_with_risk(pred, risk_factors, hour):
    """More aggressive prediction adjustment based on risk factors"""
    if not risk_factors or hour < 6 or hour > 18:
        return pred
        
    adjusted_pred = pred
    risk_score = risk_factors['risk_score']
    
    # Critical combinations trigger severe reductions
    if risk_factors['critical_combinations']:
        adjusted_pred *= 0.25  # 75% reduction for critical combinations
        print("CRITICAL combination of risk factors - applying 75% reduction")
    # Otherwise use graduated scale with more aggressive reductions
    elif risk_score >= 80:
        adjusted_pred *= 0.2  # 80% reduction
        print("Very high risk - applying 80% reduction")
    elif risk_score >= 60:
        adjusted_pred *= 0.3  # 70% reduction
        print("High risk - applying 70% reduction")
    elif risk_score >= 40:
        adjusted_pred *= 0.5  # 50% reduction
        print("Moderate risk - applying 50% reduction")
    elif risk_score >= 20:
        adjusted_pred *= 0.7  # 30% reduction
        print("Low risk - applying 30% reduction")
        
    return adjusted_pred

def analyze_extended_patterns(minute_data, timestamp, target_hour):
    """Analyze patterns using 3-hour windows"""
    try:
        # Use the target hour's timestamp
        target_timestamp = pd.Timestamp.combine(
            timestamp.date(),
            pd.Timestamp(f"{target_hour:02d}:00").time()
        )
        
        # Calculate start date for historical analysis
        start_date = target_timestamp.date() - pd.Timedelta(days=120)
        
        # Sort minute data by timestamp
        minute_data = minute_data.sort_values('timestamp')
        
        # Get current pattern including previous 3 hours
        current_pattern = minute_data[
            (minute_data['timestamp'].dt.date == target_timestamp.date()) &
            (minute_data['timestamp'].dt.hour.isin([
                target_hour,
                target_hour - 1,
                target_hour - 2,
                target_hour - 3
            ]))
        ]['Solar Rad - W/m^2'].values
        
        if len(current_pattern) < 2:
            print(f"Not enough current data points: {len(current_pattern)}")
            return None
            
        print(f"\nPattern Analysis Parameters:")
        print(f"Analyzing patterns for hour {target_hour:02d}:00")
        print(f"From {start_date} to {target_timestamp.date()}")
        print(f"\nCurrent pattern ({len(current_pattern)} points):")
        print(current_pattern)
        
        # Get historical data
        similar_patterns = []
        
        # Get all unique dates in the historical data
        historical_dates = minute_data[
            (minute_data['timestamp'].dt.date >= start_date) &
            (minute_data['timestamp'].dt.date < target_timestamp.date())
        ]['timestamp'].dt.date.unique()
        
        for date in historical_dates:
            # Get pattern for this date (including previous 3 hours)
            historical_pattern = minute_data[
                (minute_data['timestamp'].dt.date == date) &
                (minute_data['timestamp'].dt.hour.isin([
                    target_hour,
                    target_hour - 1,
                    target_hour - 2,
                    target_hour - 3
                ]))
            ]['Solar Rad - W/m^2'].values
            
            if len(historical_pattern) >= 2:  # Need at least 2 points for a pattern
                similarity = calculate_pattern_similarity(current_pattern, historical_pattern)
                
                # Get next hour's values for this date
                next_hour = (target_hour + 1) % 24
                next_hour_data = minute_data[
                    (minute_data['timestamp'].dt.date == date) &
                    (minute_data['timestamp'].dt.hour == next_hour)
                ]['Solar Rad - W/m^2'].values
                
                if len(next_hour_data) > 0:
                    similar_patterns.append({
                        'date': date,
                        'hour': target_hour,
                        'similarity': similarity,
                        'next_hour_values': next_hour_data,
                        'pattern': historical_pattern
                    })
        
        if not similar_patterns:
            print("No similar patterns found")
            return None
            
        # Sort patterns by similarity
        similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
        top_patterns = similar_patterns[:3]
        
        # Print top patterns with more detail
        print("\nTop Similar Historical Patterns:")
        for idx, pattern in enumerate(top_patterns, 1):
            print(f"\n{idx}. Date: {pattern['date']}")
            print(f"   Hour: {pattern['hour']:02d}:00")
            print(f"   Similarity: {pattern['similarity']:.3f}")
            print(f"   Next hour average: {np.mean(pattern['next_hour_values']):.1f} W/m²")
            print(f"   Pattern values: {pattern['pattern']}")
        
        # Calculate typical range from all similar patterns
        all_next_values = np.concatenate([p['next_hour_values'] for p in similar_patterns])
        value_range = (np.percentile(all_next_values, 25), np.percentile(all_next_values, 75))
        
        return {
            'patterns': top_patterns,
            'confidence': 'high' if len(similar_patterns) >= 5 else 'low',
            'trend': analyze_trend(current_pattern),
            'range': value_range,
            'best_match': top_patterns[0] if top_patterns else None
        }
        
    except Exception as e:
        print(f"Error in analyze_extended_patterns: {str(e)}")
        traceback.print_exc()
        return None

def get_pattern_fingerprint(data, timestamp, window):
    """Create detailed pattern fingerprint for a time window"""
    try:
        # Get data for the window
        start_time = timestamp - window
        window_data = data[
            (data['timestamp'] >= start_time) & 
            (data['timestamp'] <= timestamp)
        ].copy()
        
        # For 5-minute data, we need at least 2 readings
        min_readings = 2
        if len(window_data) < min_readings:
            print(f"Not enough data points in window: {len(window_data)} (need at least {min_readings})")
            print(f"Window: {start_time} to {timestamp}")
            return None
            
        # Ensure data is sorted by timestamp
        window_data = window_data.sort_values('timestamp')
        
        # Calculate basic metrics without printing debug info
        radiation_metrics = {
            'radiation_changes': window_data['Solar Rad - W/m^2'].diff().dropna().values,
            'clear_sky_ratios': (
                window_data['Solar Rad - W/m^2'] / 
                window_data['clear_sky_radiation'].clip(lower=1)
            ).values,
            'humidity_trend': window_data['Average Humidity'].diff().dropna().values,
            'uv_trend': window_data['UV Index'].values,
            'pressure_changes': window_data['Average Barometer'].diff().dropna().values,
            'temp_humidity_spread': (
                window_data['Average Temperature'] - 
                window_data['Average Dew Point']
            ).values
        }
        
        # Calculate trends without printing debug info
        trends = {}
        for metric in ['Solar Rad - W/m^2', 'Average Humidity', 'UV Index', 'Average Barometer']:
            if len(window_data[metric]) >= 2:
                x = np.arange(len(window_data[metric]))
                slope, intercept = np.polyfit(x, window_data[metric], 1)
                trends[metric] = {
                    'slope': slope,
                    'intercept': intercept,
                    'direction': 'increasing' if slope > 0 else 'decreasing'
                }
            else:
                trends[metric] = {
                    'slope': 0,
                    'intercept': 0,
                    'direction': 'stable'
                }
        
        # Add trends to fingerprint
        radiation_metrics['trends'] = trends
        
        return radiation_metrics
        
    except Exception as e:
        print(f"Error in get_pattern_fingerprint: {str(e)}")
        print(f"Window: {start_time} to {timestamp}")
        traceback.print_exc()
        return None

def calc_trend_metrics(series):
    """Calculate comprehensive trend metrics for a series"""
    try:
        if len(series) < 2:
            return {
                'slope': 0.0,
                'acceleration': 0.0,
                'volatility': 0.0,
                'direction': 'stable'
            }
            
        # Calculate basic trend using linear regression
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        
        # Calculate acceleration (change in slope)
        diffs = np.diff(series)
        acceleration = np.mean(np.diff(diffs)) if len(diffs) > 1 else 0.0
        
        # Calculate volatility
        volatility = np.std(series) / (np.mean(series) + 1e-8)  # Add small epsilon to prevent division by zero
        
        # Determine direction
        if slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'slope': float(slope),
            'acceleration': float(acceleration),
            'volatility': float(volatility),
            'direction': direction
        }
        
    except Exception as e:
        print(f"Error in calc_trend_metrics: {str(e)}")
        return {
            'slope': 0.0,
            'acceleration': 0.0,
            'volatility': 0.0,
            'direction': 'stable'
        }

def calculate_trend_similarity(trend1, trend2):
    """Calculate similarity between trend metrics"""
    try:
        if not isinstance(trend1, dict) or not isinstance(trend2, dict):
            return 0.0
            
        # Ensure all required keys exist
        required_keys = ['slope', 'acceleration', 'volatility']
        if not all(key in trend1 and key in trend2 for key in required_keys):
            return 0.0
            
        # Calculate similarities for each trend component
        slope_sim = 1 - min(1, abs(trend1['slope'] - trend2['slope']) / (max(abs(trend1['slope']), 1e-8)))
        accel_sim = 1 - min(1, abs(trend1['acceleration'] - trend2['acceleration']) / (max(abs(trend1['acceleration']), 1e-8)))
        vol_sim = 1 - min(1, abs(trend1['volatility'] - trend2['volatility']) / (max(trend1['volatility'], 1e-8)))
        
        # Direction similarity (bonus if directions match)
        direction_bonus = 1.0 if trend1.get('direction') == trend2.get('direction') else 0.5
        
        # Weighted combination
        similarity = (
            slope_sim * 0.4 +
            accel_sim * 0.3 +
            vol_sim * 0.2 +
            direction_bonus * 0.1
        )
        
        return max(0.0, min(1.0, float(similarity)))
        
    except Exception as e:
        print(f"Error in calculate_trend_similarity: {str(e)}")
        return 0.0

def analyze_pattern_evolution(data, start_time, end_time):
    """Analyze how pattern evolved in the next hour"""
    try:
        evolution_data = data[
            (data['timestamp'] >= start_time) & 
            (data['timestamp'] <= end_time)
        ].copy()
        
        if evolution_data.empty:
            return None
            
        # Calculate evolution metrics
        evolution = {
            'initial_value': evolution_data['Solar Rad - W/m^2'].iloc[0],
            'final_value': evolution_data['Solar Rad - W/m^2'].iloc[-1],
            'max_value': evolution_data['Solar Rad - W/m^2'].max(),
            'min_value': evolution_data['Solar Rad - W/m^2'].min(),
            'trend': calc_trend_metrics(evolution_data['Solar Rad - W/m^2']),
            'breakpoints': detect_pattern_breakpoints(evolution_data)
        }
        
        return evolution
        
    except Exception as e:
        print(f"Error in analyze_pattern_evolution: {str(e)}")
        return None

def detect_pattern_breakpoints(data):
    """Detect significant pattern changes"""
    try:
        breakpoints = []
        
        # Calculate changes
        radiation_changes = data['Solar Rad - W/m^2'].diff()
        
        # Define thresholds
        change_threshold = np.std(radiation_changes) * 2
        
        # Detect significant changes
        for idx, change in enumerate(radiation_changes):
            if abs(change) > change_threshold:
                breakpoints.append({
                    'index': idx,
                    'timestamp': data.index[idx],
                    'change': change,
                    'type': 'increase' if change > 0 else 'decrease'
                })
                
        return breakpoints
        
    except Exception as e:
        print(f"Error in detect_pattern_breakpoints: {str(e)}")
        return []

def analyze_similar_sequences(sequences):
    """Analyze similar sequences to determine likely evolution"""
    try:
        if not sequences:
            return {
                'confidence': 'low',
                'trend': 'unknown',
                'range': (0, 0)
            }
            
        # Collect evolution metrics
        evolutions = [seq['pattern_evolution'] for seq in sequences if seq['pattern_evolution']]
        
        if not evolutions:
            return {
                'confidence': 'low',
                'trend': 'unknown',
                'range': (0, 0)
            }
            
        # Calculate typical ranges
        final_values = [ev['final_value'] for ev in evolutions]
        value_range = (
            np.percentile(final_values, 25),
            np.percentile(final_values, 75)
        )
        
        # Analyze trends
        trends = [ev['trend']['slope'] for ev in evolutions]
        avg_trend = np.mean(trends)
        trend_consistency = np.std(trends) / max(abs(avg_trend), 1)
        
        # Determine trend direction
        if avg_trend > 10:
            trend = 'increasing'
        elif avg_trend < -10:
            trend = 'decreasing'
        else:
            trend = 'stable'
            
        # Calculate confidence based on consistency
        if len(sequences) >= 5 and trend_consistency < 0.3:
            confidence = 'high'
        elif len(sequences) >= 3 and trend_consistency < 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
            
        return {
            'confidence': confidence,
            'trend': trend,
            'range': value_range
        }
        
    except Exception as e:
        print(f"Error in analyze_similar_sequences: {str(e)}")
        return {
            'confidence': 'low',
            'trend': 'unknown',
            'range': (0, 0)
        }

def assess_peak_stability(short_trends, medium_trends):
    """Assess stability during peak hours"""
    # Calculate volatility metrics
    radiation_volatility = abs(short_trends['radiation_accel'])
    uv_volatility = abs(short_trends['uv'])
    
    # Check for rapid changes
    rapid_changes = [
        abs(short_trends['radiation']) > 100,     # Large radiation change
        abs(short_trends['radiation_accel']) > 8, # High acceleration
        abs(short_trends['uv']) > 0.3,           # Significant UV change
        abs(medium_trends['humidity']) > 2.0      # Rapid humidity change
    ]
    
    # Check for stability indicators
    stability_indicators = [
        abs(short_trends['radiation']) < 30,      # Small radiation change
        abs(short_trends['radiation_accel']) < 3, # Low acceleration
        abs(short_trends['uv']) < 0.1,           # Minimal UV change
        abs(medium_trends['humidity']) < 1.0      # Stable humidity
    ]
    
    # Count indicators
    rapid_count = sum(rapid_changes)
    stability_count = sum(stability_indicators)
    
    # Determine stability
    if rapid_count >= 2:
        return 'unstable'
    elif stability_count >= 3:
        return 'stable'
    else:
        return 'moderate'

def calculate_clear_sky_radiation(hour, latitude, longitude, date):
    """Calculate theoretical clear sky radiation"""
    try:
        # Convert inputs to float
        hour = float(hour)
        latitude = float(latitude)
        longitude = float(longitude)
        
        # Convert latitude and longitude to radians
        lat_rad = np.radians(latitude)
        
        # Calculate day of year
        day_of_year = date.timetuple().tm_yday
        
        # Calculate solar declination
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
        declination_rad = np.radians(declination)
        
        # Calculate hour angle
        hour_angle = 15 * (hour - 12)  # 15 degrees per hour from solar noon
        hour_angle_rad = np.radians(hour_angle)
        
        # Calculate solar altitude
        sin_altitude = (np.sin(lat_rad) * np.sin(declination_rad)) + \
                       (np.cos(lat_rad) * np.cos(declination_rad) * \
                       np.cos(hour_angle_rad))
        solar_altitude = np.arcsin(sin_altitude)
        
        # Calculate air mass
        if sin_altitude > 0:
            air_mass = 1 / (sin_altitude + 0.50572 * pow(6.07995 + solar_altitude, -1.6364))
        else:
            return 0
        
        # Calculate extraterrestrial radiation
        # Solar constant (W/m²)
        solar_constant = 1361
        
        # Calculate eccentricity correction
        day_angle = 2 * np.pi * day_of_year / 365
        eccentricity = (1.00011 + 0.034221 * np.cos(day_angle) + 
                       0.00128 * np.sin(day_angle) + 
                       0.000719 * np.cos(2 * day_angle) + 
                       0.000077 * np.sin(2 * day_angle))
        
        # Calculate clear sky radiation
        clear_sky = (solar_constant * eccentricity * 
                    np.exp(-0.8662 * air_mass * 0.095) * 
                    sin_altitude)
        
        # Add basic cloud impact based on hour
        if hour < 6 or hour > 18:  # Night hours
            return 0
        elif hour < 8 or hour > 16:  # Early morning/late afternoon
            clear_sky *= 0.7
        elif 11 <= hour <= 13:  # Peak hours
            clear_sky *= 0.95
        else:  # Other daylight hours
            clear_sky *= 0.85
        
        return max(0, min(clear_sky, 1200))  # Cap at 1200 W/m²
        
    except Exception as e:
        print(f"Error in calculate_clear_sky_radiation: {str(e)}")
        return 0

class CloudFeatures:
    """Class to handle cloud-related feature calculations"""
    def __init__(self):
        self.thresholds = {
            'sudden_drop': -50,  # W/m² per 5 minutes
            'sudden_increase': 50,
            'variability': 0.2,
            'cloud_impact': 0.3
        }
    
    def calculate_cloud_features(self, data):
        """Calculate cloud-related features from weather data"""
        try:
            features = {
                'sudden_drop': False,
                'sudden_increase': False,
                'high_variability': False,
                'cloud_cover': False
            }
            
            # Calculate radiation changes
            if isinstance(data, torch.Tensor):
                radiation = data[:, 0].numpy()  # Assuming radiation is first feature
            else:
                radiation = data['Solar Rad - W/m^2'].values
                
            # Detect sudden changes
            changes = np.diff(radiation)
            if len(changes) > 0:
                features['sudden_drop'] = any(changes < self.thresholds['sudden_drop'])
                features['sudden_increase'] = any(changes > self.thresholds['sudden_increase'])
                
                # Calculate variability
                if len(radiation) > 1:
                    variability = np.std(radiation) / (np.mean(radiation) + 1)
                    features['high_variability'] = variability > self.thresholds['variability']
                
                # Estimate cloud cover
                if 'clear_sky_radiation' in data.columns:
                    clear_sky_ratio = radiation / data['clear_sky_radiation'].clip(lower=1)
                    features['cloud_cover'] = np.mean(1 - clear_sky_ratio) > self.thresholds['cloud_impact']
            
            return features
            
        except Exception as e:
            print(f"Error in calculate_cloud_features: {str(e)}")
            return {
                'sudden_drop': False,
                'sudden_increase': False,
                'high_variability': False,
                'cloud_cover': False
            }

def predict_next_hour(model, data, minute_data, current_hour, target_date):
    """Predict next hour with extended pattern analysis"""
    try:
        # Get the exact hour value for current hour
        current_data = data[
            (data['timestamp'].dt.date == target_date) & 
            (data['timestamp'].dt.hour == current_hour)
        ].copy()
        
        if current_data.empty:
            print(f"No data found for hour {current_hour:02d}:00")
            return None
            
        # Get previous value
        prev_value = current_data['Solar Rad - W/m^2'].iloc[0]
        
        # Get pattern analysis for next hour once
        next_hour = (current_hour + 1) % 24
        pattern_analysis = analyze_extended_patterns(
            minute_data,
            pd.Timestamp.combine(target_date, pd.Timestamp(f"{next_hour:02d}:00").time()),
            target_hour=next_hour
        )
        
        if pattern_analysis:
            pattern_prediction = np.mean([p['next_hour_values'].mean() for p in pattern_analysis['patterns']])
            
            # Calculate clear sky radiation for next hour
            clear_sky = calculate_clear_sky_radiation(
                next_hour,
                DAVAO_LATITUDE,
                DAVAO_LONGITUDE,
                target_date
            )
            
            # Validate and adjust prediction
            prediction = validate_predictions(pattern_prediction, next_hour)
            
            # Ensure reasonable change from previous value
            max_increase = prev_value * 2.0  # Maximum 100% increase
            max_decrease = prev_value * 0.3  # Maximum 70% decrease
            prediction = min(max(prediction, max_decrease), max_increase)
            
            # Cap at clear sky radiation
            prediction = min(prediction, clear_sky * 1.1)
            
            return prediction
            
        return 0.0
        
    except Exception as e:
        print(f"Error in predict_next_hour: {str(e)}")
        traceback.print_exc()
        return None

def save_detailed_results(results_df, plot_path, csv_path, current_actual, next_hour_prediction):
    """Save detailed prediction results and visualizations"""
    try:
        # Save to CSV
        results_df.to_csv(csv_path, index=False)
        
        # Create detailed plot
        plt.figure(figsize=(12, 6))
        
        # Plot confidence intervals
        plt.fill_between(
            range(len(results_df)),
            results_df['CI_Lower'],
            results_df['CI_Upper'],
            alpha=0.2,
            color='blue',
            label='Confidence Interval'
        )
        
        # Plot predicted values
        plt.plot(
            range(len(results_df)),
            results_df['Predicted'],
            'b-',
            label='Predicted Values'
        )
        
        # Plot current actual value
        plt.axhline(
            y=current_actual,
            color='g',
            linestyle='--',
            label='Current Value'
        )
        
        # Customize plot
        plt.title('Detailed Solar Radiation Prediction')
        plt.xlabel('Minutes (5-minute intervals)')
        plt.ylabel('Solar Radiation (W/m²)')
        plt.legend()
        plt.grid(True)
        
        # Add timestamp labels
        plt.xticks(
            range(len(results_df)),
            [t.strftime('%H:%M') for t in results_df['Timestamp']],
            rotation=45
        )
        
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
    except Exception as e:
        print(f"Error in save_detailed_results: {str(e)}")

def calculate_horizon_similarities(current, historical):
    """Calculate pattern similarities across time horizons"""
    try:
        similarities = {}
        
        # Only use short and medium horizons
        for horizon in ['short', 'medium']:
            if current[horizon] is None or historical[horizon] is None:
                similarities[horizon] = 0
                continue
            
            # Calculate similarities for each metric - fix indentation
            radiation_sim = calculate_sequence_similarity(
                current[horizon]['radiation_changes'],
                historical[horizon]['radiation_changes']
            )
            
            ratio_sim = calculate_sequence_similarity(
                current[horizon]['clear_sky_ratios'],
                historical[horizon]['clear_sky_ratios']
            )
            
            humidity_sim = calculate_sequence_similarity(
                current[horizon]['humidity_trend'],
                historical[horizon]['humidity_trend']
            )
            
            uv_sim = calculate_sequence_similarity(
                current[horizon]['uv_trend'],
                historical[horizon]['uv_trend']
            )
            
            # Calculate trend similarities
            trend_sim = calculate_trend_similarity(
                current[horizon]['trends'],
                historical[horizon]['trends']
            )
            
            # Weighted combination
            similarities[horizon] = (
                radiation_sim * 0.3 +
                ratio_sim * 0.2 +
                humidity_sim * 0.2 +
                uv_sim * 0.15 +
                trend_sim * 0.15
            )
        
        return similarities
        
    except Exception as e:
        print(f"Error in calculate_horizon_similarities: {str(e)}")
        return {'short': 0, 'medium': 0}

def calculate_sequence_similarity(seq1, seq2):
    """Calculate similarity between two sequences"""
    try:
        if len(seq1) != len(seq2):
            return 0
            
        # Calculate normalized difference
        diff = np.abs(seq1 - seq2)
        max_val = max(np.max(np.abs(seq1)), np.max(np.abs(seq2)))
        if max_val == 0:
            return 1
            
        normalized_diff = diff / max_val
        similarity = 1 - np.mean(normalized_diff)
        
        return max(0, similarity)
        
    except Exception as e:
        print(f"Error in calculate_sequence_similarity: {str(e)}")
        return 0

def calculate_sequence_metrics(sequence):
    """Calculate metrics for a sequence with error handling"""
    try:
        # Ensure required columns exist
        required_columns = ['Solar Rad - W/m^2', 'clear_sky_radiation', 'Average Humidity', 'UV Index']
        if not all(col in sequence.columns for col in required_columns):
            print("Missing required columns")
            return None
            
        # Calculate metrics with error checking
        metrics = {
            'short_term': {},
            'medium_term': {},
            'long_term': {}
        }
        
        # Short-term metrics (last 30 minutes)
        short_term = sequence.tail(6)
        if len(short_term) >= 2:
            metrics['short_term'] = {
                'radiation_changes': short_term['Solar Rad - W/m^2'].diff().dropna().tolist(),
                'radiation_ratios': (short_term['Solar Rad - W/m^2'] / 
                                       short_term['clear_sky_radiation']).tolist(),
                'humidity_trend': short_term['Average Humidity'].diff().mean(),
                'uv_trend': short_term['UV Index'].diff().mean()
            }
            
        # Medium-term metrics (last hour)
        medium_term = sequence.tail(12)
        if len(medium_term) >= 2:
            metrics['medium_term'] = {
                'radiation_changes': medium_term['Solar Rad - W/m^2'].diff().dropna().tolist(),
                'radiation_ratios': (medium_term['Solar Rad - W/m^2'] / 
                                       medium_term['clear_sky_radiation']).tolist(),
                'humidity_trend': medium_term['Average Humidity'].diff().mean(),
                'uv_trend': medium_term['UV Index'].diff().mean()
            }
            
        # Long-term metrics (full sequence)
        if len(sequence) >= 2:
            metrics['long_term'] = {
                'radiation_changes': sequence['Solar Rad - W/m^2'].diff().dropna().tolist(),
                'radiation_ratios': (sequence['Solar Rad - W/m^2'] / 
                                       sequence['clear_sky_radiation']).tolist(),
                'humidity_trend': sequence['Average Humidity'].diff().mean(),
                'uv_trend': sequence['UV Index'].diff().mean()
            }
            
        return metrics
        
    except Exception as e:
        print(f"Error in calculate_sequence_metrics: {str(e)}")
        return None

def calculate_multi_horizon_similarity(current, historical):
    """Calculate similarity between multiple horizons"""
    try:
        similarities = {}
        
        for horizon in ['short', 'medium', 'long']:
            if current[horizon] is None or historical[horizon] is None:
                similarities[horizon] = 0
                continue
                
                # Calculate similarities for each metric
                radiation_sim = calculate_sequence_similarity(
                    current[horizon]['radiation_changes'],
                    historical[horizon]['radiation_changes']
                )
                
                ratio_sim = calculate_sequence_similarity(
                    current[horizon]['clear_sky_ratios'],
                    historical[horizon]['clear_sky_ratios']
                )
                
                humidity_sim = calculate_sequence_similarity(
                    current[horizon]['humidity_trend'],
                    historical[horizon]['humidity_trend']
                )
                
                uv_sim = calculate_sequence_similarity(
                    current[horizon]['uv_trend'],
                    historical[horizon]['uv_trend']
                )
                
                # Calculate trend similarities
                trend_sim = calculate_trend_similarity(
                    current[horizon]['trends'],
                    historical[horizon]['trends']
                )
                
                # Weighted combination
                similarities[horizon] = (
                    radiation_sim * 0.3 +
                    ratio_sim * 0.2 +
                    humidity_sim * 0.2 +
                    uv_sim * 0.15 +
                    trend_sim * 0.15
                )
                
        return similarities
        
    except Exception as e:
        print(f"Error in calculate_multi_horizon_similarity: {str(e)}")
        return {'short': 0, 'medium': 0, 'long': 0}

def identify_pattern_type(changes):
    """Identify the type of pattern based on changes"""
    try:
        # Calculate basic statistics
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        max_change = np.max(changes)
        min_change = np.min(changes)
        
        # Determine pattern type based on these statistics
        if max_change - min_change < 0.1:
            return 'stable'
        elif std_change / mean_change < 0.1:
            return 'trend'
        elif max_change - min_change > 0.2:
            return 'sudden_change'
        else:
            return 'mixed'
        
    except Exception as e:
        print(f"Error in identify_pattern_type: {str(e)}")
        return 'unknown'

def create_detailed_forecast(similar_sequences, last_value):
    """Create a detailed forecast based on similar sequences"""
    try:
        # Calculate average changes and trends
        avg_changes = np.mean([seq['changes'] for seq in similar_sequences], axis=0)
        avg_trends = [seq['trend']['slope'] for seq in similar_sequences]
        
        # Calculate overall trend
        overall_trend = np.mean(avg_trends)
        
        # Calculate confidence intervals
        confidence_intervals = calculate_confidence_intervals(similar_sequences)
        
        # Create detailed forecast
        forecast = {
            'confidence': 'high',
            'trend': 'stable' if overall_trend > 0 else 'decreasing',
            'range': (last_value + avg_changes[0], last_value + avg_changes[-1]),
            'confidence_intervals': confidence_intervals,
            'pattern_type': similar_sequences[0]['pattern_type']
        }
        
        return forecast
        
    except Exception as e:
        print(f"Error in create_detailed_forecast: {str(e)}")
        return None

def predict_window_hours(model, data, minute_data, current_hour, target_date, window_size=3):
    """Predict remaining hours of the day"""
    try:
        predictions = []
        hours = []
        actuals = []
        
        # Calculate start and end hours for the window
        start_hour = max(0, current_hour - window_size)
        end_hour = 23  # Changed to predict until end of day
        
        print(f"\nPredicting window from {start_hour:02d}:00 to {end_hour:02d}:00")
        print(f"Current hour: {current_hour:02d}:00")
        
        # Get today's data with correct actual values
        todays_data = data[
            (data['timestamp'].dt.date == target_date)
        ].copy()
        
        # Create a dictionary of actual values for quick lookup
        actual_values = {}
        for hour in range(24):
            hour_data = todays_data[todays_data['timestamp'].dt.hour == hour]
            if not hour_data.empty:
                actual_values[hour] = hour_data['Solar Rad - W/m^2'].iloc[0]
                print(f"Found {hour:02d}:00 value: {actual_values[hour]:.2f}")
        
        # Get predictions for each hour in the window
        for hour in range(start_hour, end_hour + 1):
            # Early morning and night hours should be 0
            if hour < 5 or hour > 18:  # No solar radiation before 5 AM and after 6 PM
                prediction = 0.0
            else:
                # Get pattern analysis for previous hour to get next hour prediction
                prev_hour = hour - 1
                prev_timestamp = pd.Timestamp.combine(
                    target_date,
                    pd.Timestamp(f"{prev_hour:02d}:00").time()
                )
                
                # Get pattern analysis for previous hour
                pattern_analysis = analyze_extended_patterns(
                    minute_data,
                    prev_timestamp,
                    target_hour=prev_hour
                )
                
                if pattern_analysis and pattern_analysis['patterns']:
                    # Use next hour average from previous hour's patterns
                    next_hour_values = [p['next_hour_values'].mean() for p in pattern_analysis['patterns']]
                    pattern_prediction = np.mean(next_hour_values)
                    prediction = validate_predictions(pattern_prediction, hour)
                    
                    # Only apply the prediction if it's greater than 0
                    if prediction > 0:
                        print(f"\nPredicting hour {hour:02d}:00 - Pattern prediction from previous hour: {prediction:.2f} W/m²")
                        formatted_values = [f"{value:.1f}" for value in next_hour_values]
                        print(f"Based on next hour averages: {formatted_values} W/m²")
                    else:
                        prediction = np.mean(next_hour_values)  # Use raw average if validation zeroes it
                else:
                    prediction = 0.0
            
            hours.append(hour)
            predictions.append(prediction)
            actuals.append(actual_values.get(hour))
            
            # Print comparison for current hour
            if hour == current_hour:
                print(f"\nCurrent hour {hour:02d}:00:")
                print(f"Predicted: {prediction:.2f} W/m²")
                print(f"Actual: {actual_values.get(hour, 'N/A')} W/m²")
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'Hour': hours,
            'Timestamp': [f"{target_date} {hour:02d}:00" for hour in hours],
            'Predicted': predictions,
            'Actual': actuals
        })
        
        # Get next hour prediction with enhanced analysis
        next_hour = (current_hour + 1) % 24
        next_hour_prediction = predict_next_hour(model, data, minute_data, current_hour, target_date)
        
        # Update the next hour prediction in results_df
        if next_hour in results_df['Hour'].values:
            results_df.loc[results_df['Hour'] == next_hour, 'Predicted'] = next_hour_prediction
            print(f"\nUpdated next hour ({next_hour:02d}:00) prediction: {next_hour_prediction:.2f} W/m²")
        
        # Print the final DataFrame for verification
        print("\nFinal predictions and actuals:")
        print(results_df.to_string())
        
        # Save results to CSV
        csv_path = 'figures/window_predictions.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved predictions to {csv_path}")
        
        # Create and save plot
        plt.figure(figsize=(12, 6))
        
        # Plot predictions
        plt.plot(results_df['Hour'], results_df['Predicted'], 
                marker='o', linestyle='-', linewidth=2, label='Predicted',
                color='blue')
        
        # Plot actual values where available (only past and current hours)
        actual_mask = results_df['Hour'] <= current_hour
        if actual_mask.any():
            plt.plot(results_df.loc[actual_mask, 'Hour'], 
                    results_df.loc[actual_mask, 'Actual'],
                    marker='s', linestyle='--', linewidth=2, label='Actual',
                    color='green')
        
        # Highlight current hour
        current_hour_data = results_df[results_df['Hour'] == current_hour]
        plt.scatter(current_hour_data['Hour'], current_hour_data['Actual'],
                   color='red', s=100, zorder=5, label='Current Hour')
        
        # Highlight next hour prediction
        next_hour_data = results_df[results_df['Hour'] == next_hour]
        if not next_hour_data.empty:
            plt.scatter(next_hour_data['Hour'], next_hour_data['Predicted'],
                      color='purple', s=100, zorder=5, label='Next Hour',
                      marker='*')
        
        plt.title(f'Solar Radiation Predictions Window for {target_date}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Solar Radiation (W/m²)')
        plt.grid(True)
        plt.xticks(range(start_hour, end_hour + 1))
        plt.ylim(bottom=0)
        plt.legend()
        
        # Add value annotations
        for idx, row in results_df.iterrows():
            if row['Hour'] == next_hour:
                # Special annotation for next hour prediction
                plt.annotate(f"{row['Predicted']:.0f} W/m²\n(Next Hour)", 
                           (row['Hour'], row['Predicted']),
                           textcoords="offset points", 
                           xytext=(0,15), 
                           ha='center',
                           color='purple',
                           fontweight='bold')
            else:
                plt.annotate(f"{row['Predicted']:.0f}", 
                           (row['Hour'], row['Predicted']),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center')
                # Only show actual values for past and current hours
                if pd.notna(row['Actual']) and row['Hour'] <= current_hour:
                    plt.annotate(f"{row['Actual']:.0f}", 
                               (row['Hour'], row['Actual']),
                               textcoords="offset points", 
                               xytext=(0,-15), 
                               ha='center',
                               color='green')
        
        plt.tight_layout()
        plt.savefig('figures/window_predictions.png')
        plt.close()
        
        return results_df
        
    except Exception as e:
        print(f"Error in predict_window_hours: {str(e)}")
        traceback.print_exc()
        return None

# Modify the main function to use the new prediction function
def main():
    try:
        print("Starting enhanced solar radiation prediction pipeline...")
        
        # Load and validate input data
        if not os.path.exists('dataset.csv'):
            raise FileNotFoundError("dataset.csv not found")
        
        print("\nLoading and preprocessing data...")
        hourly_data, minute_data, feature_averages = preprocess_data('dataset.csv')
        if hourly_data is None or minute_data is None:
            raise ValueError("Failed to preprocess data")
        
        # Use hourly_data for model training
        data = hourly_data
        
        # Get the last timestamp from the dataset
        last_timestamp = data['timestamp'].max()
        
        # Process minute data
        minute_data = minute_data[minute_data['timestamp'] <= last_timestamp].copy()
        
        # Process features for minute data
        minute_data['clear_sky_radiation'] = minute_data.apply(
            lambda row: calculate_clear_sky_radiation(
                row['hour'] + row['minute']/60,
                DAVAO_LATITUDE,
                DAVAO_LONGITUDE,
                row['date']
            ),
            axis=1
        )
        
        # Calculate other features for minute data
        process_features(minute_data)
        
        if data.empty or minute_data.empty:
            raise ValueError("No valid data found")
            
        current_hour = last_timestamp.hour
        target_date = last_timestamp.date()
        
        print(f"\nAnalyzing patterns for {target_date} {current_hour:02d}:00")
        
        # Print data info
        print("\nData Range Validation:")
        print(f"Start date: {data['timestamp'].min()}")
        print(f"End date: {last_timestamp}")
        print(f"Total hourly records: {len(data)}")
        print(f"Total minute records: {len(minute_data)}")
        print(f"Minutes per hour: {len(minute_data[minute_data['timestamp'].dt.date == target_date])}")
        
        # Verify minute data for current hour
        current_hour_data = minute_data[
            (minute_data['timestamp'].dt.date == target_date) &
            (minute_data['timestamp'].dt.hour == current_hour)
        ]
        print(f"\nCurrent hour minute data:")
        print(current_hour_data[['timestamp', 'Solar Rad - W/m^2']].to_string())
        
        # Perform extended historical pattern analysis
        historical_patterns = analyze_extended_patterns(
            minute_data,          # Pass minute-level data
            last_timestamp,       # Pass the last timestamp
            target_hour=current_hour  # Pass current hour as target hour
        )
        
        if historical_patterns is None:
            print("Warning: Could not analyze historical patterns")
            
        # Prepare features for model
        print("\nPreparing features...")
        X, y, feature_names = prepare_features(data)
        if X is None or y is None:
            raise ValueError("Feature preparation failed")
            
        # Scale features
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Split data and train model
        train_size = int(0.7 * len(X_scaled))
        X_train = X_scaled[:train_size]
        X_test = X_scaled[train_size:]
        y_train = y_scaled[:train_size]
        y_test = y_scaled[train_size:]
        
        print("\nTraining model...")
        model, corrections, epochs = train_model(X_train, y_train, X_test, y_test, scaler_y)
        
        if model is not None:
            print(f"\nModel trained successfully over {epochs} epochs")
            
            # Get current hour's actual value
            current_data = data[data['timestamp'] == last_timestamp].iloc[0]
            current_actual = current_data['Solar Rad - W/m^2']
            
            # Replace the single prediction with window prediction
            print("\nMaking window predictions...")
            results_df = predict_window_hours(
                model,
                data,
                minute_data,
                current_hour,
                target_date,
                window_size=3  # Predict 3 hours before and after current hour
            )
            
            if results_df is not None:
                print("\nPrediction Results:")
                print(f"Current Hour ({current_hour:02d}:00): {current_actual:.2f} W/m²")
                print("\nWindow Predictions:")
                for _, row in results_df.iterrows():
                    actual_str = f"(Actual: {row['Actual']:.2f})" if pd.notna(row['Actual']) else "(No actual data)"
                    print(f"Hour {row['Hour']:02d}:00 - Predicted: {row['Predicted']:.2f} W/m² {actual_str}")
            else:
                print("\nError: Window predictions failed")
            
        else:
            print("\nError: Model training failed")
            
    except Exception as e:
        print(f"\nError in main function: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')

def calculate_pattern_similarity(pattern1, pattern2):
    """Calculate similarity between two patterns"""
    try:
        # Ensure patterns have same length
        min_len = min(len(pattern1), len(pattern2))
        pattern1 = pattern1[:min_len]
        pattern2 = pattern2[:min_len]
        
        if min_len < 2:
            return 0.0
            
        # Convert to numpy arrays and normalize
        p1 = np.array(pattern1, dtype=float)
        p2 = np.array(pattern2, dtype=float)
        
        # Normalize patterns to 0-1 range
        p1_norm = (p1 - np.min(p1)) / (np.max(p1) - np.min(p1) + 1e-8)
        p2_norm = (p2 - np.min(p2)) / (np.max(p2) - np.min(p2) + 1e-8)
        
        # Calculate similarity using multiple metrics
        euclidean_sim = 1 / (1 + np.sqrt(np.mean((p1_norm - p2_norm) ** 2)))
        corr = np.corrcoef(p1_norm, p2_norm)[0, 1]
        shape_sim = (corr + 1) / 2
        range_ratio = min(np.ptp(p1), np.ptp(p2)) / (max(np.ptp(p1), np.ptp(p2)) + 1e-8)
        
        # Calculate trend similarity
        trend1 = np.polyfit(np.arange(len(p1)), p1, 1)[0]
        trend2 = np.polyfit(np.arange(len(p2)), p2, 1)[0]
        trend_sim = 1 - abs(trend1 - trend2) / (abs(trend1) + abs(trend2) + 1e-8)
        
        # Combine similarities with weights
        similarity = (
            euclidean_sim * 0.3 +
            shape_sim * 0.3 +
            range_ratio * 0.2 +
            trend_sim * 0.2
        )
        
        # Penalize length differences
        len_penalty = min(len(pattern1), len(pattern2)) / max(len(pattern1), len(pattern2))
        similarity *= len_penalty
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error in calculate_pattern_similarity: {str(e)}")
        traceback.print_exc()
        return 0.0

def analyze_trend(values):
    """Analyze trend in a sequence of values"""
    try:
        if len(values) < 2:
            return 'stable'
            
        # Calculate changes
        changes = np.diff(values)
        
        # Calculate trend metrics
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        # Determine trend direction and strength
        if abs(mean_change) < std_change:
            return 'stable'
        elif mean_change > 0:
            return 'increasing'
        else:
            return 'decreasing'
            
    except Exception as e:
        print(f"Error in analyze_trend: {str(e)}")
        return 'stable'

if __name__ == "__main__":
    main()
