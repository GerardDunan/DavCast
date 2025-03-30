import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU detection

# Part 1: Data Preprocessing and Setup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import traceback
import joblib
import json
# Add visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')
import chardet

class WeatherPredictor:
    # Constants that should be configurable
    DEFAULT_LAT = 7.0707  # Davao City
    DEFAULT_LON = 125.6113  # Davao City
    DEFAULT_ELEVATION = 7  # meters
    SOLAR_CONSTANT = 1361  # W/m²
    FORECAST_HOURS = 4
    TRAIN_SIZE = 0.7
    VALIDATION_SIZE = 0.15
    
    # Enhanced Model parameters with better tuning
    MODEL_PARAMS = {
        'xgboost': {
            'n_estimators': 4500,          # Further increased for better capacity
            'max_depth': 9,                # Increased for more complex patterns
            'learning_rate': 0.0008,       # Further decreased for better stability
            'min_child_weight': 2,         # Decreased for better granularity
            'subsample': 0.8,              # Decreased for better generalization
            'colsample_bytree': 0.8,       # Decreased for better feature selection
            'gamma': 0.15,                 # Decreased to allow more splits
            'reg_alpha': 0.3,              # Decreased L1 regularization
            'reg_lambda': 1.0,             # Adjusted L2 regularization
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'random_state': 42,
            'early_stopping_rounds': 150,   # Increased for better convergence
            'n_jobs': -1
        }
    }
    
    FILE_PATHS = {
        'xgboost_model': 'best_xgboost_model_hour_{}.json'
    }
    
    # Updated feature columns to match new dataset format
    FEATURE_COLUMNS = [
        'Barometer - hPa', 
        'Temp - °C',
        'Hum - %',
        'Dew Point - °C',
        'Wet Bulb - °C',
        'Avg Wind Speed - km/h',
        'Wind Run - km',
        'UV Index',
        'Day of Year',
        'Month of Year',  # New column
        'Hour of Day',
        'Solar Zenith Angle',
        'GHI_lag (t-1)',  # New column
        'Daytime',        # New column
        'Hour_Sin',
        'Hour_Cos',
        'Day_Sin',
        'Day_Cos',
        'Cos_Hour_Angle',
        'Sin_Solar_Elevation',
        'Clear_Sky_GHI',
        'Clear_Sky_GHI_Adjusted',
        'GHI_Clear_Sky_Ratio',
        'Air_Mass',
        'Atmospheric_Attenuation',
        'GHI_Change',
        'GHI_Change_Rate',
        'GHI_Acceleration',
        'Recent_Variability',
        'Recent_Trend',
        'Is_Clear',
        'Is_Mostly_Clear',
        'Is_Partly_Cloudy',
        'Is_Mostly_Cloudy',
        'Is_Overcast',
        'Is_Transition',
        'Is_Rapid_Change'
    ]
    
    TIME_FEATURES = ['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos']
    
    def __init__(self, lat=DEFAULT_LAT, lon=DEFAULT_LON, elevation=DEFAULT_ELEVATION,
                 model_params=None, file_paths=None):
        """
        Initialize with configurable parameters
        """
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.solar_constant = self.SOLAR_CONSTANT
        self.features_scaler = RobustScaler()  # Changed from MinMaxScaler to RobustScaler
        self.target_scaler = RobustScaler()    # Changed from MinMaxScaler to RobustScaler
        self.feature_columns = self.FEATURE_COLUMNS.copy()
        
        # Use provided model parameters or defaults
        self.model_params = model_params or self.MODEL_PARAMS
        self.file_paths = file_paths or self.FILE_PATHS
        
        # Initialize model storage
        self.xgb_models = []

    def compute_solar_parameters(self, df):
        """
        Compute all solar-related parameters for the dataset
        """
        try:
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Convert date and time to datetime
            # Updated to handle the new column names
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start Period'])
            
            # Compute basic time parameters if not already present
            if 'Day of Year' not in df.columns:
                df['Day of Year'] = df['datetime'].apply(lambda x: x.timetuple().tm_yday)
            
            if 'Month of Year' not in df.columns:
                df['Month of Year'] = df['datetime'].apply(lambda x: x.month)
                
            if 'Hour of Day' not in df.columns:
                df['Hour of Day'] = df['datetime'].apply(lambda x: x.hour + x.minute/60)
            
            def compute_declination(day_of_year):
                return np.radians(23.45 * np.sin(np.radians(360/365 * (day_of_year + 284))))
            
            def compute_solar_angles(row):
                # Calculate declination
                declination = compute_declination(row['Day of Year'])
                
                # Calculate hour angle
                hour_angle = np.radians(15 * (row['Hour of Day'] - 12))
                
                # Convert latitude to radians
                latitude_rad = np.radians(self.lat)
                
                # Compute solar elevation angle
                solar_elevation_angle = np.arcsin(
                    np.sin(latitude_rad) * np.sin(declination) +
                    np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
                )
                
                # Compute solar zenith angle
                solar_zenith_angle = np.pi/2 - solar_elevation_angle
                
                return pd.Series({
                    'Solar Zenith Angle': np.degrees(solar_zenith_angle),
                    'Solar Elevation Angle': np.degrees(solar_elevation_angle)
                })
            
            # Compute solar angles if Solar Elevation Angle is not present
            if 'Solar Elevation Angle' not in df.columns:
                solar_angles = df.apply(compute_solar_angles, axis=1)
                df['Solar Zenith Angle'] = solar_angles['Solar Zenith Angle']
                df['Solar Elevation Angle'] = solar_angles['Solar Elevation Angle']
            
            # Compute clearness index
            df['Clearness Index'] = df.apply(
                lambda row: row['GHI - W/m^2']/self.SOLAR_CONSTANT 
                if row['GHI - W/m^2'] <= self.SOLAR_CONSTANT and row['GHI - W/m^2'] > 0
                else np.nan,
                axis=1
            )
            
            # Fill NaN values with 0 for nighttime
            night_mask = df['Solar Elevation Angle'] <= 0
            df.loc[night_mask, ['Clearness Index']] = 0
            
            # Add Daytime column if not present
            if 'Daytime' not in df.columns:
                df['Daytime'] = (df['Solar Elevation Angle'] > 0).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error in compute_solar_parameters: {str(e)}")
            print("DataFrame columns:", df.columns.tolist())
            raise

    def load_and_prepare_data(self, csv_path):
        """
        Load and prepare data from CSV file with solar parameter computations
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Dataset file not found: {csv_path}")
            
            # Load the dataset with explicit encoding
            print(f"Loading dataset from {csv_path}...")
            try:
                # First attempt with latin-1 encoding (commonly works with Windows files)
                df = pd.read_csv(csv_path, encoding='latin-1')
            except:
                try:
                    # Second attempt with cp1252 encoding
                    df = pd.read_csv(csv_path, encoding='cp1252')
                except:
                    # Third attempt with ISO-8859-1
                    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
            
            # Rename columns if using older format
            column_mapping = {
                'Start_period': 'Start Period',
                'End_period': 'End Period',
                'Temp - °C': 'Temp - °C',
                'Dew Point - °C': 'Dew Point - °C',
                'Wet Bulb - °C': 'Wet Bulb - °C'
            }
            
            df = df.rename(columns={old: new for old, new in column_mapping.items() 
                                    if old in df.columns and new not in df.columns})
            
            # Print initial dataset info
            print("\nInitial dataset info:")
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            
            # Handle GHI_lag if not present
            if 'GHI_lag (t-1)' not in df.columns:
                print("Creating GHI_lag feature...")
                df['GHI_lag (t-1)'] = df['GHI - W/m^2'].shift(1)
                df['GHI_lag (t-1)'].fillna(0, inplace=True)
            
            # Compute solar parameters
            print("\nComputing solar parameters...")
            df = self.compute_solar_parameters(df)
            
            # Prepare features and target
            print("\nPreparing features and target...")
            X_scaled, y_scaled, df = self.prepare_data_for_training(df)
            
            return X_scaled, y_scaled, df
            
        except Exception as e:
            print(f"\nError in load_and_prepare_data: {str(e)}")
            raise

    def prepare_xgboost_data(self, X_scaled, y_scaled):
        """Prepare data for XGBoost with proper handling of time series data"""
        try:
            # Convert to numpy arrays
            X = np.array(X_scaled)
            y = np.array(y_scaled).reshape(-1)  # Ensure 1D array
            
            # Calculate split indices
            train_idx = int(len(X) * self.TRAIN_SIZE)
            val_idx = train_idx + int(len(X) * self.VALIDATION_SIZE)
            
            # Create target sequences for each forecast hour
            y_train_split = []
            y_val_split = []
            y_test_split = []
            
            # Adjust X and y for each forecast horizon
            X_train_list = []
            X_val_list = []
            X_test_list = []
            
            for i in range(self.FORECAST_HOURS):
                # Create shifted targets with adjusted shift for each hour
                # Hour 1: shift by 1, Hour 2: shift by 2, etc.
                shift = i + 1
                y_shifted = y[shift:]  # Remove first shift entries
                X_shifted = X[:-shift]  # Remove last shift entries
                
                # Ensure X and y have same length
                min_len = min(len(y_shifted), len(X_shifted))
                y_shifted = y_shifted[:min_len]
                X_shifted = X_shifted[:min_len]
                
                # Calculate new split points based on shortened data
                train_idx_i = int(min_len * self.TRAIN_SIZE)
                val_idx_i = train_idx_i + int(min_len * self.VALIDATION_SIZE)
                
                # Split data
                y_train = y_shifted[:train_idx_i]
                y_val = y_shifted[train_idx_i:val_idx_i]
                y_test = y_shifted[val_idx_i:]
                
                X_train = X_shifted[:train_idx_i]
                X_val = X_shifted[train_idx_i:val_idx_i]
                X_test = X_shifted[val_idx_i:]
                
                # Store splits
                y_train_split.append(y_train)
                y_val_split.append(y_val)
                y_test_split.append(y_test)
                
                X_train_list.append(X_train)
                X_val_list.append(X_val)
                X_test_list.append(X_test)
            
            return X_train_list, y_train_split, X_val_list, y_val_split, X_test_list, y_test_split
            
        except Exception as e:
            print(f"Error in prepare_xgboost_data: {str(e)}")
            raise
    
    def train_xgboost(self, X_train_list, y_train_split, X_val_list, y_val_split):
        """Enhanced XGBoost training with better handling of different conditions"""
        self.xgb_models = []
        self.xgb_models_lower = []
        self.xgb_models_upper = []
        
        # Enhanced base parameters
        base_params = self.MODEL_PARAMS['xgboost'].copy()
        
        # Wider initial intervals for better coverage
        quantile_params_lower = base_params.copy()
        quantile_params_lower['objective'] = 'reg:quantileerror'
        quantile_params_lower['quantile_alpha'] = 0.15  # More aggressive lower bound
        
        quantile_params_upper = base_params.copy()
        quantile_params_upper['objective'] = 'reg:quantileerror'
        quantile_params_upper['quantile_alpha'] = 0.85  # More aggressive upper bound
        
        for i in range(self.FORECAST_HOURS):
            print(f"\nTraining XGBoost model for {i+1}-hour ahead prediction")
            
            # Adjust parameters based on forecast horizon
            horizon_params = base_params.copy()
            if i > 0:
                # More aggressive parameter adjustments
                horizon_params['n_estimators'] += i * 750  # Increased trees for longer horizons
                horizon_params['learning_rate'] *= 0.95 ** i  # Slower decay
                horizon_params['subsample'] = min(0.9, horizon_params['subsample'] + 0.05 * i)
                horizon_params['max_depth'] = min(horizon_params['max_depth'] + i, 12)
                
                # Lighter regularization for longer horizons
                horizon_params['reg_alpha'] *= 1.05 ** i  # Slower increase
                horizon_params['reg_lambda'] *= 1.05 ** i
                horizon_params['gamma'] *= 0.95 ** i  # Decrease gamma for longer horizons
            
            # Calculate sample weights with enhanced weighting
            weights = self._calculate_sample_weights(y_train_split[i], X_train_list[i])
            
            # Train median model with enhanced parameters
            model = xgb.XGBRegressor(**horizon_params)
            model.fit(
                X_train_list[i], 
                y_train_split[i],
                sample_weight=weights,
                eval_set=[(X_val_list[i], y_val_split[i])],
                verbose=True
            )
            
            # Make predictions on validation set to assess uncertainty
            val_preds = model.predict(X_val_list[i])
            val_errors = np.abs(val_preds - y_val_split[i])
            error_percentile = np.percentile(val_errors, 95)  # Increased percentile
            
            # More aggressive interval widths
            base_width = 0.40  # Increased from 0.30
            horizon_factor = 1 + (i * 0.15)  # Increased from 0.10
            error_factor = min(2.0, error_percentile / np.mean(val_errors))  # Increased cap
            
            interval_width = base_width * horizon_factor * error_factor
            
            if i == 0:
                # Less aggressive reduction for 1-hour predictions
                interval_width *= 0.9  # Changed from 0.8
            
            # Ensure minimum interval width
            interval_width = max(interval_width, 0.3 + 0.1 * i)
            
            q_params_lower = quantile_params_lower.copy()
            q_params_upper = quantile_params_upper.copy()
            
            # Asymmetric intervals based on horizon
            if i > 0:
                # Wider intervals for longer horizons with asymmetric bounds
                lower_width = interval_width * (1.1 ** i)  # More aggressive lower bound
                upper_width = interval_width * (1.2 ** i)  # Even more aggressive upper bound
                
                q_params_lower['quantile_alpha'] = max(0.05, 0.5 - lower_width/2)
                q_params_upper['quantile_alpha'] = min(0.95, 0.5 + upper_width/2)
                
                # Slower learning rate decay
                q_params_lower['learning_rate'] *= 0.97 ** i
                q_params_upper['learning_rate'] *= 0.97 ** i
            
            # Train interval models with adjusted parameters
            model_lower = xgb.XGBRegressor(**q_params_lower)
            model_upper = xgb.XGBRegressor(**q_params_upper)
            
            # Use higher weights for extreme cases in interval training
            interval_weights = weights.copy()
            extreme_mask = (y_train_split[i] < np.percentile(y_train_split[i], 10)) | \
                          (y_train_split[i] > np.percentile(y_train_split[i], 90))
            interval_weights[extreme_mask] *= 1.5
            
            model_lower.fit(
                X_train_list[i], 
                y_train_split[i],
                sample_weight=interval_weights,
                eval_set=[(X_val_list[i], y_val_split[i])],
                verbose=True
            )
            
            model_upper.fit(
                X_train_list[i], 
                y_train_split[i],
                sample_weight=interval_weights,
                eval_set=[(X_val_list[i], y_val_split[i])],
                verbose=True
            )
            
            self.xgb_models.append(model)
            self.xgb_models_lower.append(model_lower)
            self.xgb_models_upper.append(model_upper)
            
            # Print feature importance
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            })
            importance = importance.sort_values('importance', ascending=False)
            print(f"\nTop 10 important features for hour {i+1}:")
            print(importance.head(10))
    
    def _calculate_sample_weights(self, y, X):
        """Enhanced sample weights calculation with better handling of edge cases"""
        try:
            # Get solar elevation from features
            solar_elevation_idx = self.feature_columns.index('Solar Elevation Angle')
            solar_elevation = X[:, solar_elevation_idx]
            
            # Initialize base weights
            weights = np.ones_like(y)
            
            # Enhanced transition period weighting with smoother transitions
            sunrise_mask = (solar_elevation > -5) & (solar_elevation < 5)
            early_morning_mask = (solar_elevation >= 5) & (solar_elevation < 15)
            late_evening_mask = (solar_elevation >= -15) & (solar_elevation <= -5)
            
            # Smooth transition weights based on elevation angle
            transition_factor = np.clip((15 - np.abs(solar_elevation)) / 15, 0, 1)
            weights[sunrise_mask] *= (2.0 * transition_factor[sunrise_mask] + 1.0)
            weights[early_morning_mask] *= (1.75 * transition_factor[early_morning_mask] + 1.0)
            weights[late_evening_mask] *= (1.75 * transition_factor[late_evening_mask] + 1.0)
            
            # Enhanced GHI-based weighting
            nonzero_mask = y > 0
            if np.any(nonzero_mask):
                ghi_percentiles = np.percentile(y[nonzero_mask], [25, 50, 75, 90])
                
                # Progressive weighting for different GHI ranges
                low_ghi_mask = (y > 0) & (y <= ghi_percentiles[0])
                med_low_ghi_mask = (y > ghi_percentiles[0]) & (y <= ghi_percentiles[1])
                med_high_ghi_mask = (y > ghi_percentiles[1]) & (y <= ghi_percentiles[2])
                high_ghi_mask = (y > ghi_percentiles[2]) & (y <= ghi_percentiles[3])
                very_high_ghi_mask = y > ghi_percentiles[3]
                
                # Enhanced weights for low GHI conditions
                weights[low_ghi_mask] *= 2.0
                weights[med_low_ghi_mask] *= 1.75
                weights[med_high_ghi_mask] *= 1.5
                weights[high_ghi_mask] *= 1.75
                weights[very_high_ghi_mask] *= 2.0
            
            # Better handling of rapid changes with trend consideration
            if hasattr(self, 'historical_values'):
                y_series = pd.Series(y)
                changes = y_series.diff().abs()
                
                # Identify different types of changes
                rapid_change_mask = changes > np.percentile(changes[changes > 0], 90)
                very_rapid_change_mask = changes > np.percentile(changes[changes > 0], 95)
                
                # Calculate trend direction
                trend = y_series.diff().rolling(3).mean()
                increasing_trend = trend > 0
                decreasing_trend = trend < 0
                
                # Weight based on change magnitude and trend
                weights[rapid_change_mask] *= 1.75
                weights[very_rapid_change_mask] *= 2.0
                weights[rapid_change_mask & increasing_trend] *= 1.1
                weights[rapid_change_mask & decreasing_trend] *= 1.1
                
                # Enhanced volatility weighting
                volatility = y_series.rolling(5).std()
                high_volatility_mask = volatility > np.percentile(volatility[volatility > 0], 75)
                very_high_volatility_mask = volatility > np.percentile(volatility[volatility > 0], 90)
                weights[high_volatility_mask] *= 1.5
                weights[very_high_volatility_mask] *= 1.75
            
            # Weather condition specific weights with clear sky consideration
            clear_sky_ratio_idx = self.feature_columns.index('GHI_Clear_Sky_Ratio')
            clear_sky_ratio = X[:, clear_sky_ratio_idx]
            
            # Enhanced weighting for different sky conditions
            clear_mask = clear_sky_ratio > 0.8
            mostly_clear_mask = (clear_sky_ratio > 0.6) & (clear_sky_ratio <= 0.8)
            partly_cloudy_mask = (clear_sky_ratio > 0.3) & (clear_sky_ratio <= 0.6)
            mostly_cloudy_mask = (clear_sky_ratio > 0.1) & (clear_sky_ratio <= 0.3)
            
            weights[clear_mask] *= 1.25
            weights[mostly_clear_mask] *= 1.5
            weights[partly_cloudy_mask] *= 2.0  # Increased focus on partly cloudy conditions
            weights[mostly_cloudy_mask] *= 1.75
            
            # Enhanced time-based weights
            hour_idx = self.feature_columns.index('Hour of Day')
            hours = X[:, hour_idx]
            
            # Progressive weighting for different times of day
            early_hours_mask = (hours >= 6) & (hours < 9)
            peak_hours_mask = (hours >= 9) & (hours <= 15)
            late_hours_mask = (hours > 15) & (hours <= 18)
            
            weights[early_hours_mask] *= 1.5
            weights[peak_hours_mask] *= 1.25
            weights[late_hours_mask] *= 1.5
            
            # Normalize weights to maintain overall scale
            weights = weights / weights.mean()
            
            # Clip extreme weights with wider range for important cases
            weights = np.clip(weights, 0.25, 4.0)
            
            return weights
            
        except Exception as e:
            print(f"Error in _calculate_sample_weights: {str(e)}")
            return np.ones_like(y)

    def predict_xgboost(self, X):
        """Enhanced prediction with improved physical constraints and confidence intervals"""
        try:
            # Get base predictions
            predictions = np.column_stack([
                model.predict(X) for model in self.xgb_models
            ])
            
            predictions_lower = np.column_stack([
                model.predict(X) for model in self.xgb_models_lower
            ])
            
            predictions_upper = np.column_stack([
                model.predict(X) for model in self.xgb_models_upper
            ])
            
            # Get solar parameters using Sin_Solar_Elevation
            solar_elevation_idx = self.feature_columns.index('Sin_Solar_Elevation')
            sin_solar_elevation = X[:, solar_elevation_idx]
            # Convert from sin to actual angle
            solar_elevation = np.arcsin(np.clip(sin_solar_elevation, -1, 1)) * 180 / np.pi
            
            # Get hour of day for time-based adjustments - using Hour_Sin and Hour_Cos
            hour_sin_idx = self.feature_columns.index('Hour_Sin')
            hour_cos_idx = self.feature_columns.index('Hour_Cos')
            hour_sin = X[:, hour_sin_idx]
            hour_cos = X[:, hour_cos_idx]
            
            # Convert Hour_Sin and Hour_Cos back to hour of day (0-24)
            hour_of_day = (np.arctan2(hour_sin, hour_cos) * 12 / np.pi + 12) % 24
            
            # Calculate theoretical clear sky GHI with improved atmospheric model
            clear_sky_ghi = self.SOLAR_CONSTANT * np.sin(np.radians(solar_elevation))
            
            # First inverse transform all predictions
            predictions_median = self.target_scaler.inverse_transform(predictions)
            predictions_lower = self.target_scaler.inverse_transform(predictions_lower)
            predictions_upper = self.target_scaler.inverse_transform(predictions_upper)
            
            # Enhanced interval adjustments based on time of day and forecast horizon
            for i in range(self.FORECAST_HOURS):
                # Ensure proper ordering of bounds (lower ≤ median ≤ upper)
                lower_bound = np.minimum(predictions_lower[:, i], predictions_median[:, i])
                upper_bound = np.maximum(predictions_upper[:, i], predictions_median[:, i])
                median = predictions_median[:, i]
                
                # Slightly boost the median prediction (3-11% boost based on horizon)
                boost_factor = 1.03 + (i * 0.02)  # Increased base boost from 1.02 to 1.03
                median = median * boost_factor
                
                # Time of day factors - slightly increased
                morning_factor = np.where((hour_of_day >= 6) & (hour_of_day < 9), 1.68, 1.1)  # Increased from 1.65 and 1.08
                midday_factor = np.where((hour_of_day >= 9) & (hour_of_day < 15), 1.35, 1.1)  # Increased from 1.3 and 1.08
                afternoon_factor = np.where((hour_of_day >= 15) & (hour_of_day < 18), 1.55, 1.1)  # Increased from 1.5 and 1.08
                
                # Combine time factors
                time_factor = np.maximum.reduce([morning_factor, midday_factor, afternoon_factor])
                
                # Solar elevation adjustment - slightly increased base factor
                elevation_factor = np.where(
                    solar_elevation > 0,
                    1.1 + (0.25 * (1 - sin_solar_elevation)),  # Increased from 1.08 and 0.22
                    1.1  # Increased from 1.08
                )
                
                # Horizon-based uncertainty growth - slightly increased
                horizon_factor = 1.1 + (i * 0.2)  # Increased from 1.08 and 0.18
                
                # Calculate dynamic margin that increases with horizon and solar elevation
                base_margin = 40 * (1 + i * 0.25)  # Increased from 38 and 0.22
                elevation_margin = base_margin * (1 + (90 - solar_elevation) / 80)  # Decreased from 85 to 80
                
                # Calculate interval width based on median prediction
                base_interval = median * 0.25 * (1 + i * 0.15)  # Increased from 0.24 and 0.14
                
                # Apply all factors to the intervals
                interval_width = base_interval * time_factor * elevation_factor * horizon_factor
                
                # Update predictions with adjusted intervals
                predictions_upper[:, i] = median + interval_width + elevation_margin
                predictions_lower[:, i] = median - (interval_width * 0.7)  # Reduced from 0.75 for slightly higher lower bound
                
                # Special handling for early morning predictions (7:00-8:00)
                early_morning_mask = (hour_of_day >= 7) & (hour_of_day < 8)
                if i == 0:  # First hour prediction
                    predictions_upper[early_morning_mask, i] = median[early_morning_mask] * 1.5  # Increased from 1.45
                    predictions_lower[early_morning_mask, i] = median[early_morning_mask] * 0.7  # Increased from 0.65
                
                # Ensure physical constraints - slightly increased max theoretical
                max_theoretical = clear_sky_ghi * (1.3 + (i * 0.15))  # Increased from 1.25 and 0.12
                predictions_upper[:, i] = np.minimum(predictions_upper[:, i], max_theoretical)
                predictions_lower[:, i] = np.maximum(predictions_lower[:, i], 0)  # Cannot go below 0
                
                # Final sanity check on bounds
                predictions_median[:, i] = np.clip(
                    median,
                    predictions_lower[:, i],
                    predictions_upper[:, i]
                )
            
            # Enhanced night handling with smooth transitions
            night_mask = solar_elevation <= -5
            transition_mask = (solar_elevation > -5) & (solar_elevation <= 0)
            
            # Smooth transition factor - slightly increased for transitions
            transition_factor = np.zeros_like(solar_elevation)
            transition_factor[transition_mask] = (solar_elevation[transition_mask] + 5) / 4.8  # Reduced denominator from 5 to 4.8
            
            # Apply night and transition adjustments
            for i in range(predictions_median.shape[1]):
                predictions_median[night_mask, i] = 0
                predictions_lower[night_mask, i] = 0
                predictions_upper[night_mask, i] = 0
                
                # Smooth transition
                predictions_median[transition_mask, i] *= transition_factor[transition_mask]
                predictions_lower[transition_mask, i] *= transition_factor[transition_mask]
                predictions_upper[transition_mask, i] *= transition_factor[transition_mask]
            
            return predictions_median, predictions_lower, predictions_upper
            
        except Exception as e:
            print(f"Error in predict_xgboost: {str(e)}")
            raise

    def evaluate_model(self, X_data, y_data):
        """Evaluate XGBoost models with enhanced validation metrics and visualization"""
        print("\nEvaluating XGBoost models (hourly averages):")
        
        # Create results directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, 'model_performance')
        os.makedirs(results_dir, exist_ok=True)
        
        all_metrics = []
        actual_vs_predicted = {}
        
        # Create a figure for overall performance visualization
        plt.figure(figsize=(15, 10))
        plt.suptitle('GHI Prediction Performance by Forecast Hour', fontsize=16)
        
        for i in range(self.FORECAST_HOURS):
            print(f"\nEvaluating {i+1}-hour ahead predictions")
            print(f"(XX:00 to (XX+1):00 hourly averages)")
            
            # Get corresponding data for this horizon
            X = X_data[i]
            y_true = y_data[i]
            
            # Make predictions with all models
            y_pred = self.xgb_models[i].predict(X)
            y_pred_lower = self.xgb_models_lower[i].predict(X)
            y_pred_upper = self.xgb_models_upper[i].predict(X)
            
            # Inverse transform predictions and true values
            y_true = self.target_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
            y_pred_lower = self.target_scaler.inverse_transform(y_pred_lower.reshape(-1, 1)).reshape(-1)
            y_pred_upper = self.target_scaler.inverse_transform(y_pred_upper.reshape(-1, 1)).reshape(-1)
            
            # Standard metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Filter to daytime only for daytime metrics
            daytime_mask = y_true > 0
            if sum(daytime_mask) > 0:
                daytime_mae = mean_absolute_error(y_true[daytime_mask], y_pred[daytime_mask])
                daytime_rmse = np.sqrt(mean_squared_error(y_true[daytime_mask], y_pred[daytime_mask]))
                daytime_r2 = r2_score(y_true[daytime_mask], y_pred[daytime_mask])
            else:
                daytime_mae = daytime_rmse = float('nan')
                daytime_r2 = float('nan')
            
            # Interval metrics
            coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
            interval_width = np.mean(y_pred_upper - y_pred_lower)
            
            # Custom error metrics
            weighted_errors = np.abs(y_true - y_pred) / (y_true.clip(min=1) + 10)
            mape = np.mean(np.abs((y_true - y_pred) / y_true.clip(min=1))) * 100
            
            # Store metrics
            hour_metrics = {
                'Forecast Hour': i+1,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'Daytime MAE': daytime_mae,
                'Daytime RMSE': daytime_rmse,
                'Daytime R²': daytime_r2,
                'Coverage': coverage,
                'Interval Width': interval_width,
                'Weighted Error': np.mean(weighted_errors),
                'MAPE': mape
            }
            
            all_metrics.append(hour_metrics)
            
            # Store actual vs predicted for plotting
            actual_vs_predicted[f'Hour {i+1}'] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_lower': y_pred_lower,
                'y_pred_upper': y_pred_upper
            }
            
            # Create individual subplot for this hour
            plt.subplot(2, 2, i+1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([0, max(y_true.max(), y_pred.max())], 
                     [0, max(y_true.max(), y_pred.max())], 
                     'r--')
            plt.xlabel('Actual GHI (W/m²)')
            plt.ylabel('Predicted GHI (W/m²)')
            plt.title(f'Hour {i+1} Forecast: R² = {r2:.3f}, RMSE = {rmse:.2f}')
            plt.grid(True, alpha=0.3)
            
            # Add metrics text
            plt.text(0.05, 0.95, 
                    f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}\n'
                    f'Coverage: {coverage:.1%}\nDaytime R²: {daytime_r2:.3f}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            print(f'MAE: {mae:.2f} W/m²')
            print(f'RMSE: {rmse:.2f} W/m²')
            print(f'R²: {r2:.3f}')
            print(f'Daytime MAE: {daytime_mae:.2f} W/m²')
            print(f'Daytime RMSE: {daytime_rmse:.2f} W/m²')
            print(f'Daytime R²: {daytime_r2:.3f}')
            print(f'Prediction Interval Coverage: {coverage:.1%}')
            print(f'Average Interval Width: {interval_width:.2f} W/m²')
        
        # Save the main performance plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        performance_plot_path = os.path.join(results_dir, 'prediction_performance.png')
        plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional performance visualizations
        self._create_performance_visualizations(all_metrics, actual_vs_predicted, results_dir)
        
        # Create feature importance visualization
        self._visualize_feature_importance(results_dir)
        
        print(f"\nPerformance visualizations saved to: {results_dir}")
        
        return all_metrics
    
    def _create_performance_visualizations(self, metrics, actual_vs_predicted, save_dir):
        """Create and save additional performance visualizations"""
        # 1. Metrics by forecast hour
        metrics_df = pd.DataFrame(metrics)
        
        # Overall metrics plot
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['RMSE', 'MAE', 'Daytime RMSE', 'Daytime MAE']
        
        for metric in metrics_to_plot:
            plt.plot(metrics_df['Forecast Hour'], metrics_df[metric], 'o-', label=metric)
        
        plt.xticks(metrics_df['Forecast Hour'])
        plt.xlabel('Forecast Hour')
        plt.ylabel('Error (W/m²)')
        plt.title('Error Metrics by Forecast Horizon')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'error_metrics_by_hour.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # R² metrics plot
        plt.figure(figsize=(10, 6))
        r2_metrics = ['R²', 'Daytime R²']
        
        for metric in r2_metrics:
            plt.plot(metrics_df['Forecast Hour'], metrics_df[metric], 'o-', label=metric)
        
        plt.xticks(metrics_df['Forecast Hour'])
        plt.xlabel('Forecast Hour')
        plt.ylabel('R² Score')
        plt.title('R² Score by Forecast Horizon')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'r2_by_hour.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error distribution plot for each hour
        plt.figure(figsize=(15, 10))
        plt.suptitle('Error Distribution by Forecast Hour', fontsize=16)
        
        for i, hour in enumerate(range(1, self.FORECAST_HOURS + 1)):
            hour_data = actual_vs_predicted[f'Hour {hour}']
            errors = hour_data['y_true'] - hour_data['y_pred']
            
            plt.subplot(2, 2, i+1)
            sns.histplot(errors, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.xlabel('Prediction Error (W/m²)')
            plt.ylabel('Frequency')
            plt.title(f'Hour {hour} Error Distribution')
            
            # Add error stats
            plt.text(0.05, 0.95, 
                     f'Mean Error: {errors.mean():.2f}\n'
                     f'Error Std: {errors.std():.2f}\n'
                     f'25-75 Percentile: [{np.percentile(errors, 25):.1f}, {np.percentile(errors, 75):.1f}]',
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(save_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction intervals plot for first 100 samples of each hour
        for hour in range(1, self.FORECAST_HOURS + 1):
            plt.figure(figsize=(15, 8))
            hour_data = actual_vs_predicted[f'Hour {hour}']
            
            # Take first 100 samples or all if less than 100
            sample_size = min(100, len(hour_data['y_true']))
            indices = np.arange(sample_size)
            
            plt.fill_between(indices, 
                             hour_data['y_pred_lower'][:sample_size], 
                             hour_data['y_pred_upper'][:sample_size], 
                             alpha=0.3, label='Prediction Interval')
            
            plt.plot(indices, hour_data['y_pred'][:sample_size], 'b-', label='Predicted GHI')
            plt.plot(indices, hour_data['y_true'][:sample_size], 'ro', markersize=4, label='Actual GHI')
            
            plt.xlabel('Sample Index')
            plt.ylabel('GHI (W/m²)')
            plt.title(f'Hour {hour} Forecast: Actual vs Predicted with Confidence Intervals')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add coverage metric
            coverage = np.mean((hour_data['y_true'][:sample_size] >= hour_data['y_pred_lower'][:sample_size]) & 
                              (hour_data['y_true'][:sample_size] <= hour_data['y_pred_upper'][:sample_size]))
            
            plt.text(0.05, 0.95, 
                     f'Interval Coverage: {coverage:.1%}',
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(os.path.join(save_dir, f'prediction_intervals_hour_{hour}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Save metrics table as CSV
        metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)
        
        # Create an HTML summary for quick viewing
        self._create_html_summary(save_dir, metrics_df)
    
    def _visualize_feature_importance(self, save_dir):
        """Create and save feature importance visualizations for all models"""
        # Create combined feature importance plot
        plt.figure(figsize=(15, 10))
        importance_dfs = []
        
        for i, model in enumerate(self.xgb_models):
            # Get feature importance
            importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': model.feature_importances_
            })
            importance['Hour'] = i + 1
            importance_dfs.append(importance)
        
        # Combine all importance dataframes
        all_importance = pd.concat(importance_dfs)
        
        # Get top 15 features based on average importance
        top_features = all_importance.groupby('Feature')['Importance'].mean().nlargest(15).index
        
        # Filter for only top features
        plot_df = all_importance[all_importance['Feature'].isin(top_features)]
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', hue='Hour', data=plot_df)
        plt.title('Top 15 Features by Importance Across Forecast Hours')
        plt.xlabel('Importance (Gain)')
        plt.ylabel('Feature')
        plt.legend(title='Forecast Hour')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed feature importance for each hour
        for i, model in enumerate(self.xgb_models):
            importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': model.feature_importances_
            })
            importance = importance.sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=importance.head(20))
            plt.title(f'Top 20 Features by Importance - Hour {i+1} Forecast')
            plt.xlabel('Importance (Gain)')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'feature_importance_hour_{i+1}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save full feature importance to CSV
            importance.to_csv(os.path.join(save_dir, f'feature_importance_hour_{i+1}.csv'), index=False)
    
    def _create_html_summary(self, save_dir, metrics_df):
        """Create an HTML summary of model performance with embedded images"""
        html_path = os.path.join(save_dir, 'performance_summary.html')
        
        # Format metrics table for HTML
        metrics_table = metrics_df.to_html(float_format=lambda x: f'{x:.3f}', index=False)
        
        # List of generated images (relative paths)
        images = [
            'prediction_performance.png',
            'error_metrics_by_hour.png',
            'r2_by_hour.png',
            'error_distributions.png',
            'feature_importance.png'
        ]
        
        # Add hour-specific images
        for i in range(1, self.FORECAST_HOURS + 1):
            images.append(f'prediction_intervals_hour_{i}.png')
            images.append(f'feature_importance_hour_{i}.png')
        
        # Create image HTML
        image_html = ''
        for img in images:
            image_html += f'<div class="figure-container">\n'
            image_html += f'  <figure>\n'
            image_html += f'    <img src="{img}" alt="{img}" width="800">\n'
            image_html += f'    <figcaption>{img}</figcaption>\n'
            image_html += f'  </figure>\n'
            image_html += f'</div>\n'
        
        # Create HTML content
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>GHI Prediction Model Performance Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; text-align: center; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .figure-container {{ margin: 20px 0; }}
                figure {{ margin: 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                figcaption {{ font-style: italic; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <h1>GHI Prediction Model Performance Summary</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Metrics</h2>
            {metrics_table}
            
            <h2>Visualizations</h2>
            {image_html}
        </body>
        </html>
        '''
        
        # Write HTML to file
        with open(html_path, 'w') as f:
            f.write(html_content)

    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics"""
        try:
            # Basic metrics
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mbe = np.mean(y_pred - y_true)
            nse = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
            
            # Calculate MAPE for daytime values only
            day_mask = y_true > 0
            y_true_day = y_true[day_mask]
            y_pred_day = y_pred[day_mask]
            
            if len(y_true_day) > 0:
                mape = np.mean(np.abs((y_true_day - y_pred_day) / y_true_day)) * 100
                # Weighted MAPE
                weights = y_true_day / np.mean(y_true_day)
                wmape = np.average(np.abs((y_true_day - y_pred_day) / y_true_day), weights=weights) * 100
            else:
                mape = np.nan
                wmape = np.nan
            
            return {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'wmape': wmape,
                'mbe': mbe,
                'nse': nse
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None

    def print_metrics(self, metrics):
        """Print formatted metrics"""
        if metrics is None:
            print("No metrics available")
            return
        
        print("Primary Metrics:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.2f} W/m²")
        print(f"  MAE: {metrics['mae']:.2f} W/m²")
        print("\nPercentage Errors:")
        print(f"  MAPE (daytime only): {metrics['mape']:.2f}%")
        print(f"  Weighted MAPE: {metrics['wmape']:.2f}%")
        print("\nBias Metrics:")
        print(f"  Mean Bias Error: {metrics['mbe']:.2f} W/m²")
        print(f"  Nash-Sutcliffe Efficiency: {metrics['nse']:.4f}")

    def get_user_input(self, timestamp):
        """Get input values from user"""
        # Get the time period from the timestamp
        end_time = timestamp.replace(minute=0) + timedelta(hours=1)
        print(f"\nEnter values for {timestamp.strftime('%H:%M')} - {end_time.strftime('%H:%M')}:")
        print("(hourly averages)")
        
        input_values = {}
        prompts = {
            'Temp - °C': 'Average Temperature (°C): ',
            'Hum - %': 'Average Humidity (%): ',
            'Dew Point - °C': 'Average Dew Point (°C): ',
            'Wet Bulb - °C': 'Average Wet Bulb (°C): ',
            'Avg Wind Speed - km/h': 'Average Wind Speed (km/h): ',
            'Wind Run - km': 'Wind Run (km): ',
            'UV Index': 'Average UV Index: ',
            'Barometer - hPa': 'Average Barometric Pressure (hPa): ',
            'GHI - W/m^2': 'Average GHI (W/m^2): '
        }
        
        for key, prompt in prompts.items():
            while True:
                try:
                    value = float(input(prompt))
                    input_values[key] = value
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        return input_values

    def get_last_timestamp(self):
        """Get the last timestamp and next start time from the dataset"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, 'dataset.csv')
        
        # Fix: Add proper encoding handling
        try:
            # First attempt with latin-1 encoding (commonly works with Windows files)
            df = pd.read_csv(dataset_path, encoding='latin-1')
        except:
            try:
                # Second attempt with cp1252 encoding
                df = pd.read_csv(dataset_path, encoding='cp1252')
            except:
                # Third attempt with ISO-8859-1
                df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
        
        # Get the last row
        last_row = df.iloc[-1]
        
        # Parse the last timestamp with flexible format
        last_date = last_row['Date']
        last_end = last_row['End Period']
        last_timestamp = pd.to_datetime(f"{last_date} {last_end}", format='mixed')
        
        # Calculate next start time (on the hour)
        next_start = last_timestamp.replace(minute=0)
        
        return next_start

    def prepare_prediction_data(self, input_values):
        """Prepare input data for prediction with all required features"""
        df = None  # Initialize df to avoid UnboundLocalError
        try:
            # Use the date and time information from input_values instead of fetching from dataset
            # Create a DataFrame with the input values
            df = pd.DataFrame([input_values])
            
            # Check if Start Period and End Period are already in input_values
            if 'Start Period' in input_values and 'End Period' in input_values:
                # Use the values from input_values
                next_start = pd.to_datetime(f"{input_values['Date']} {input_values['Start Period']}")
                next_end = pd.to_datetime(f"{input_values['Date']} {input_values['End Period']}")
            else:
                # For backwards compatibility
                # Get next start time from dataset
                next_start = self.get_last_timestamp()
                next_end = next_start.replace(minute=0) + timedelta(hours=1)
                
                # Add date and time information to df
                df['Date'] = next_start.strftime('%Y-%m-%d')
                df['Start_period'] = next_start.strftime('%H:%M')  # Will be XX:00
                df['End_period'] = next_end.strftime('%H:%M')  # Will be (XX+1):00
            
            # Use end time for solar calculations instead of midpoint
            calculation_time = next_end  # Use end time (e.g., 11:00)
            
            # Only set these if not already present
            if 'Day of Year' not in df.columns:
                df['Day of Year'] = calculation_time.timetuple().tm_yday
            if 'Hour of Day' not in df.columns:
                df['Hour of Day'] = calculation_time.hour + calculation_time.minute/60
            
            # Calculate solar parameters
            df = self.compute_solar_parameters(df)
            
            # Ensure key naming consistency - standardize to use space format
            if 'Start_period' in df.columns and 'Start Period' not in df.columns:
                df = df.rename(columns={'Start_period': 'Start Period'})
            if 'End_period' in df.columns and 'End Period' not in df.columns:
                df = df.rename(columns={'End_period': 'End Period'})
            
            # Ensure datetime column exists for time-based features
            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['End Period'])
            
            # Ensure Sin_Solar_Elevation is calculated if missing
            if 'Sin_Solar_Elevation' not in df.columns:
                # Calculate it from Solar Elevation Angle
                if 'Solar Elevation Angle' in df.columns:
                    df['Sin_Solar_Elevation'] = np.sin(np.radians(df['Solar Elevation Angle']))
                else:
                    # If Solar Elevation Angle is missing too, calculate it from zenith angle
                    if 'Solar Zenith Angle' in df.columns:
                        df['Solar Elevation Angle'] = 90 - df['Solar Zenith Angle']
                        df['Sin_Solar_Elevation'] = np.sin(np.radians(df['Solar Elevation Angle']))
                    else:
                        # Calculate from scratch if needed
                        solar_angles = self.compute_solar_angles(df.iloc[0])
                        df['Solar Zenith Angle'] = solar_angles['Solar Zenith Angle']
                        df['Solar Elevation Angle'] = solar_angles['Solar Elevation Angle']
                        df['Sin_Solar_Elevation'] = np.sin(np.radians(df['Solar Elevation Angle']))
            
            # Add Air_Mass calculation
            df['Air_Mass'] = 1 / (df['Sin_Solar_Elevation'].clip(0.001, 1))
            df['Air_Mass'] = df['Air_Mass'].clip(1, 38)
            
            # Calculate Atmospheric_Attenuation if it doesn't exist
            if 'Atmospheric_Attenuation' not in df.columns:
                df['Atmospheric_Attenuation'] = 0.7 ** (df['Air_Mass'] ** 0.678)
            
            # Calculate Clear_Sky_GHI if it doesn't exist
            if 'Clear_Sky_GHI' not in df.columns:
                # Calculate Clear Sky GHI based on solar zenith angle and solar constant
                cos_zenith = np.cos(np.radians(df['Solar Zenith Angle']))
                df['Clear_Sky_GHI'] = self.SOLAR_CONSTANT * cos_zenith.clip(0, 1)  # Clip to avoid negative values
            
            # Calculate Clear_Sky_GHI_Adjusted
            df['Clear_Sky_GHI_Adjusted'] = df['Clear_Sky_GHI'] * df['Atmospheric_Attenuation']
            
            # Calculate GHI_Clear_Sky_Ratio if it doesn't exist
            if 'GHI_Clear_Sky_Ratio' not in df.columns:
                # Avoid division by zero
                denominator = df['Clear_Sky_GHI_Adjusted'].clip(10, None)  # Minimum 10 W/m²
                df['GHI_Clear_Sky_Ratio'] = (df['GHI - W/m^2'] / denominator).clip(0, 2)  # Clip ratio between 0 and 2
            
            # Calculate all the missing trigonometric time features
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour of Day'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour of Day'] / 24)
            df['Day_Sin'] = np.sin(2 * np.pi * df['Day of Year'] / 365)
            df['Day_Cos'] = np.cos(2 * np.pi * df['Day of Year'] / 365)
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month of Year'] / 12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month of Year'] / 12)
            
            # Calculate hour angle and its cosine
            hour_angle_rad = np.radians(15 * (df['Hour of Day'] - 12))
            df['Cos_Hour_Angle'] = np.cos(hour_angle_rad)
            
            # Calculate GHI-related features and ratios
            if 'GHI_lag (t-1)' in df.columns and 'GHI - W/m^2' in df.columns:
                # Avoid division by zero
                denominator = df['GHI_lag (t-1)'].clip(1, None)
                df['GHI_lag_ratio'] = (df['GHI - W/m^2'] / denominator).clip(0, 5)
            else:
                df['GHI_lag_ratio'] = 1.0  # Default value
            
            # Add interaction terms
            df['GHI_UV_Interaction'] = df['GHI - W/m^2'] * df['UV Index']
            df['Temp_Humidity_Interaction'] = df['Temp - °C'] * df['Hum - %']
            df['GHI_Temp_Interaction'] = df['GHI - W/m^2'] * df['Temp - °C']
            df['GHI_Wind_Interaction'] = df['GHI - W/m^2'] * df['Avg Wind Speed - km/h']
            df['Daytime_GHI_Interaction'] = df['Daytime'] * df['GHI - W/m^2']
            df['Daytime_Temp_Interaction'] = df['Daytime'] * df['Temp - °C']
            
            # Add cloud condition dummy variables (set to default values)
            cloud_conditions = ['Is_Clear', 'Is_Mostly_Clear', 'Is_Partly_Cloudy', 
                              'Is_Mostly_Cloudy', 'Is_Overcast']
            for condition in cloud_conditions:
                df[condition] = 0  # Default: no specific cloud condition
                
            # Use clearness index to make a best guess of cloud condition
            if 'Clearness Index' in df.columns:
                ck = df['Clearness Index'].iloc[0]
                # Set one condition to 1 based on clearness index
                if ck > 0.8:
                    df['Is_Clear'] = 1
                elif ck > 0.6:
                    df['Is_Mostly_Clear'] = 1
                elif ck > 0.4:
                    df['Is_Partly_Cloudy'] = 1
                elif ck > 0.2:
                    df['Is_Mostly_Cloudy'] = 1
                else:
                    df['Is_Overcast'] = 1
                    
            # Add trend features (set defaults since we don't have historical data)
            df['Is_Transition'] = 0
            df['Is_Rapid_Change'] = 0
            df['GHI_Change'] = 0
            df['GHI_Change_Rate'] = 0
            df['GHI_Acceleration'] = 0
            df['Recent_Variability'] = 0
            df['Recent_Trend'] = 0
            
            # Add time series aggregation features (defaults)
            for window in [3, 6, 12]:
                df[f'GHI_{window}h_mean'] = df['GHI - W/m^2']
                df[f'GHI_{window}h_std'] = 0
                df[f'GHI_EMA_{window}h'] = df['GHI - W/m^2']
                df[f'GHI_Ratio_EMA_{window}h'] = 1
                df[f'GHI_Volatility_{window}h'] = 0
            
            # Ensure all required features are present and in the right order
            df = df.reindex(columns=self.feature_columns, fill_value=0)
            
            # Convert to numpy array for the model
            X = df.values
            
            # Scale features using the loaded scaler
            X_scaled = self.features_scaler.transform(X)
            
            # Now return the properly scaled data
            return X_scaled
                
        except Exception as e:
            print(f"Error in prepare_prediction_data: {str(e)}")
            if df is not None:
                print("DataFrame columns:", df.columns.tolist())
            print("Required features:", self.feature_columns)
            raise

    def make_prediction_from_input(self):
        """Enhanced prediction with better feature handling and interval predictions"""
        try:
            # Check if models are trained
            if not self.xgb_models:
                print("Error: Models not trained. Please train the models first (Option 1).")
                return
            
            # Get input data
            new_row = self.get_input_data()
            
            # Use prepare_prediction_data to handle all feature engineering properly
            X_pred_scaled = self.prepare_prediction_data(new_row)
            
            # Make predictions with intervals
            predictions_median, predictions_lower, predictions_upper = self.predict_xgboost(X_pred_scaled)
            
            # Print predictions for all 4 hours with intervals
            print("\nPredicted GHI values with confidence intervals:")
            for hour in range(self.FORECAST_HOURS):
                print(f"\nHour {hour+1}:")
                print(f"Upper bound (75th percentile): {predictions_upper[0][hour]:.2f} W/m²")
                print(f"Expected (median): {predictions_median[0][hour]:.2f} W/m²")
                print(f"Lower bound (25th percentile): {predictions_lower[0][hour]:.2f} W/m²")
            
            # Save input data and predictions to CSV
            self.save_to_csv(new_row, predictions_median)
            
            return predictions_median, predictions_lower, predictions_upper
        
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            raise

    def save_to_csv(self, input_data, predictions):
        """Save input data and predictions to CSV file with all required fields"""
        try:
            # Standardize column names for consistency
            if 'Start_period' in input_data and 'Start Period' not in input_data:
                input_data['Start Period'] = input_data.pop('Start_period')
            if 'End_period' in input_data and 'End Period' not in input_data:
                input_data['End Period'] = input_data.pop('End_period')
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'dataset.csv')
            
            # Determine which encoding to use by checking existing file
            file_encoding = 'latin-1'  # Default to latin-1 for better special character handling
            
            try:
                # Check if file exists and determine its encoding
                if os.path.exists(csv_path):
                    # Try to detect encoding from existing file
                    try:
                        with open(csv_path, 'rb') as f:
                            result = chardet.detect(f.read())
                            detected_encoding = result['encoding']
                            if detected_encoding and result['confidence'] > 0.7:
                                file_encoding = detected_encoding
                    except:
                        pass  # If detection fails, use the default
                    
                    # Read with detected or default encoding
                    try:
                        existing_df = pd.read_csv(csv_path, encoding=file_encoding)
                    except:
                        # Fall back to latin-1 which usually works with Windows files
                        file_encoding = 'latin-1'
                        existing_df = pd.read_csv(csv_path, encoding=file_encoding)
                    
                    # Remove any prediction columns if they exist
                    prediction_columns = [col for col in existing_df.columns if 'Prediction_Hour_' in col]
                    if prediction_columns:
                        existing_df = existing_df.drop(columns=prediction_columns)
                else:
                    # Create empty DataFrame with necessary columns
                    existing_df = pd.DataFrame(columns=[
                        'Date', 'Start Period', 'End Period', 'Barometer - hPa', 
                        'Temp - °C', 'Hum - %', 'Dew Point - °C', 'Wet Bulb - °C',
                        'Avg Wind Speed - km/h', 'Wind Run - km', 'UV Index', 'GHI - W/m^2',
                        'Day of Year', 'Month of Year', 'Hour of Day', 'Solar Zenith Angle', 
                        'GHI_lag (t-1)', 'Daytime'
                    ])
            except Exception as e:
                print(f"Error reading existing CSV: {str(e)}")
                # Create empty DataFrame
                existing_df = pd.DataFrame()
            
            # Format date to the required format: %d-%b-%y (e.g., "3-Mar-25")
            date_str = input_data['Date']
            date_obj = pd.to_datetime(date_str)
            formatted_date = date_obj.strftime('%d-%b-%y')
            
            # Format start and end times to HH:MM:SS
            start_time_str = input_data['Start Period']
            end_time_str = input_data['End Period']
            
            # Parse and format time strings
            try:
                if ':' in start_time_str:
                    start_time_obj = pd.to_datetime(start_time_str).time()
                else:
                    start_hour = float(start_time_str)
                    start_time_obj = pd.to_datetime(f"{int(start_hour)}:{int((start_hour % 1) * 60)}").time()
            except:
                start_time_obj = datetime.now().replace(minute=0, second=0, microsecond=0).time()
                
            try:
                if ':' in end_time_str:
                    end_time_obj = pd.to_datetime(end_time_str).time()
                else:
                    end_hour = float(end_time_str)
                    end_time_obj = pd.to_datetime(f"{int(end_hour)}:{int((end_hour % 1) * 60)}").time()
            except:
                end_time_obj = (datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).time()
                
            formatted_start = start_time_obj.strftime('%H:%M:%S')
            formatted_end = end_time_obj.strftime('%H:%M:%S')
            
            # Set GHI_lag to current GHI value
            ghi_lag_value = input_data['GHI - W/m^2']
            
            # Ensure special characters in column names are preserved
            temperature_col = 'Temp - °C'
            dew_point_col = 'Dew Point - °C'
            wet_bulb_col = 'Wet Bulb - °C'
            
            # Include all required fields with carefully preserved column names
            new_row = {
                'Date': formatted_date,
                'Start Period': formatted_start,
                'End Period': formatted_end,
                'Barometer - hPa': input_data['Barometer - hPa'],
                temperature_col: input_data['Temp - °C'],
                'Hum - %': input_data['Hum - %'],
                dew_point_col: input_data['Dew Point - °C'],
                wet_bulb_col: input_data['Wet Bulb - °C'],
                'Avg Wind Speed - km/h': input_data['Avg Wind Speed - km/h'],
                'Wind Run - km': input_data['Wind Run - km'],
                'UV Index': input_data['UV Index'],
                'GHI - W/m^2': input_data['GHI - W/m^2'],
                'Day of Year': input_data['Day of Year'],
                'Month of Year': input_data['Month of Year'],
                'Hour of Day': input_data['Hour of Day'],
                'Solar Zenith Angle': input_data['Solar Zenith Angle'],
                'GHI_lag (t-1)': ghi_lag_value,
                'Daytime': input_data['Daytime']
            }
            
            # Append new row to existing DataFrame
            new_row_df = pd.DataFrame([new_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
            # Save with the same encoding as we read with
            updated_df.to_csv(csv_path, index=False, encoding=file_encoding)
            print(f"\nData saved to {csv_path} using {file_encoding} encoding")
            
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            if 'input_data' in locals():
                print("Keys in input_data:", list(input_data.keys()))
            raise

    def custom_metric(self, y_true, y_pred):
        """Custom evaluation metric focusing on daytime predictions"""
        # Only consider daytime values (where true GHI > 0)
        day_mask = y_true > 0
        
        if not any(day_mask):
            return 0
        
        y_true_day = y_true[day_mask]
        y_pred_day = y_pred[day_mask]
        
        # Calculate weighted RMSE giving more importance to higher GHI values
        weights = y_true_day / np.mean(y_true_day)
        weighted_mse = np.average((y_true_day - y_pred_day) ** 2, weights=weights)
        
        return np.sqrt(weighted_mse)

    def prepare_data_for_training(self, df):
        """Enhanced data preparation with better feature engineering"""
        try:
            df_clean = df.copy()
            
            # Basic preprocessing with updated column names
            df_clean['datetime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Start Period'])
            df_clean.set_index('datetime', inplace=True)
            df_clean = df_clean.sort_index()
            
            # Enhanced time features 
            df_clean['Hour_Sin'] = np.sin(2 * np.pi * df_clean['Hour of Day'] / 24)
            df_clean['Hour_Cos'] = np.cos(2 * np.pi * df_clean['Hour of Day'] / 24)
            df_clean['Day_Sin'] = np.sin(2 * np.pi * df_clean['Day of Year'] / 365)
            df_clean['Day_Cos'] = np.cos(2 * np.pi * df_clean['Day of Year'] / 365)
            
            # Month cyclical features
            df_clean['Month_Sin'] = np.sin(2 * np.pi * df_clean['Month of Year'] / 12)
            df_clean['Month_Cos'] = np.cos(2 * np.pi * df_clean['Month of Year'] / 12)
            
            # Solar position features
            df_clean['Cos_Hour_Angle'] = np.cos(np.radians(15 * (df_clean['Hour of Day'] - 12)))
            df_clean['Sin_Solar_Elevation'] = np.sin(np.radians(df_clean['Solar Elevation Angle']))
            df_clean['Clear_Sky_GHI'] = self.SOLAR_CONSTANT * df_clean['Sin_Solar_Elevation']
            
            # Enhanced atmospheric model with humidity and temperature effects
            df_clean['Air_Mass'] = 1 / (df_clean['Sin_Solar_Elevation'].clip(0.001, 1))
            df_clean['Air_Mass'] = df_clean['Air_Mass'].clip(1, 38)
            
            # Temperature-adjusted atmospheric attenuation
            temp_factor = (df_clean['Temp - °C'] + 273.15) / 298.15  # Normalize to 25°C
            humidity_factor = 1 + (df_clean['Hum - %'] / 100) * 0.1  # Humidity effect
            df_clean['Atmospheric_Attenuation'] = (0.7 ** (df_clean['Air_Mass'] * humidity_factor)) * temp_factor
            df_clean['Clear_Sky_GHI_Adjusted'] = df_clean['Clear_Sky_GHI'] * df_clean['Atmospheric_Attenuation']
            
            # Better rapid change detection - use GHI_lag if available
            if 'GHI_lag (t-1)' in df_clean.columns:
                df_clean['GHI_Change'] = df_clean['GHI - W/m^2'] - df_clean['GHI_lag (t-1)']
            else:
                df_clean['GHI_Change'] = df_clean['GHI - W/m^2'].diff()
                
            df_clean['GHI_Change_Rate'] = df_clean['GHI_Change'] / df_clean['Clear_Sky_GHI_Adjusted'].clip(1, None)
            df_clean['GHI_Acceleration'] = df_clean['GHI_Change'].diff()
            
            # Normalized change rates
            df_clean['Normalized_Change'] = df_clean['GHI_Change'] / df_clean['Clear_Sky_GHI'].clip(1, None)
            df_clean['Is_Rapid_Change'] = (abs(df_clean['Normalized_Change']) > 0.1).astype(int)
            
            # Use Daytime column if available, otherwise calculate
            if 'Daytime' not in df_clean.columns:
                df_clean['Daytime'] = (df_clean['Solar Elevation Angle'] > 0).astype(int)
            
            # Enhanced transition period detection
            df_clean['Is_Transition'] = (
                ((df_clean['Solar Elevation Angle'] > -5) & (df_clean['Solar Elevation Angle'] < 15)) |  
                ((df_clean['GHI - W/m^2'] > 0) & (df_clean['GHI - W/m^2'] < 100))  
            ).astype(int)
            
            # More sophisticated weather condition detection
            df_clean['GHI_Clear_Sky_Ratio'] = np.where(
                df_clean['Clear_Sky_GHI_Adjusted'] > 10,
                df_clean['GHI - W/m^2'] / df_clean['Clear_Sky_GHI_Adjusted'],
                0
            )
            df_clean['GHI_Clear_Sky_Ratio'] = df_clean['GHI_Clear_Sky_Ratio'].clip(0, 1.2)
            
            # Dynamic thresholds based on solar elevation
            base_clear = 0.85
            elevation_factor = (90 - df_clean['Solar Elevation Angle']) / 90
            clear_threshold = base_clear - 0.1 * elevation_factor
            
            df_clean['Is_Clear'] = (df_clean['GHI_Clear_Sky_Ratio'] > clear_threshold).astype(int)
            df_clean['Is_Mostly_Clear'] = ((df_clean['GHI_Clear_Sky_Ratio'] > clear_threshold*0.8) & 
                                         (df_clean['GHI_Clear_Sky_Ratio'] <= clear_threshold)).astype(int)
            df_clean['Is_Partly_Cloudy'] = ((df_clean['GHI_Clear_Sky_Ratio'] > 0.5) & 
                                          (df_clean['GHI_Clear_Sky_Ratio'] <= clear_threshold*0.8)).astype(int)
            df_clean['Is_Mostly_Cloudy'] = ((df_clean['GHI_Clear_Sky_Ratio'] > 0.2) & 
                                          (df_clean['GHI_Clear_Sky_Ratio'] <= 0.5)).astype(int)
            df_clean['Is_Overcast'] = (df_clean['GHI_Clear_Sky_Ratio'] <= 0.2).astype(int)
            
            # Enhanced temporal features with better smoothing
            df_clean['Recent_Variability'] = df_clean['GHI_Change'].rolling(window=5, center=True).std()
            df_clean['Recent_Trend'] = df_clean['GHI_Change'].rolling(window=5, center=True).mean()
            
            # Better handling of rolling statistics
            for window in [3, 6, 12]:
                # Centered rolling statistics for better accuracy
                df_clean[f'GHI_{window}h_mean'] = df_clean['GHI - W/m^2'].rolling(
                    window=window, center=True, min_periods=1
                ).mean()
                df_clean[f'GHI_{window}h_std'] = df_clean['GHI - W/m^2'].rolling(
                    window=window, center=True, min_periods=1
                ).std()
                
                # Exponential moving averages with adjusted spans
                span = window * 2  # Longer span for better smoothing
                df_clean[f'GHI_EMA_{window}h'] = df_clean['GHI - W/m^2'].ewm(
                    span=span, adjust=False, min_periods=1
                ).mean()
                df_clean[f'GHI_Ratio_EMA_{window}h'] = df_clean['GHI_Clear_Sky_Ratio'].ewm(
                    span=span, adjust=False, min_periods=1
                ).mean()
                
                # Volatility with normalization
                df_clean[f'GHI_Volatility_{window}h'] = (
                    df_clean['GHI_Change'].rolling(window=window, min_periods=1).std() /
                    df_clean['Clear_Sky_GHI_Adjusted'].clip(1, None)
                )
            
            # Use GHI_lag feature directly if available, otherwise create lag features
            if 'GHI_lag (t-1)' in df_clean.columns:
                lag_features = ['GHI_lag (t-1)']
                
                # Normalize GHI_lag by Clear Sky GHI
                df_clean['GHI_lag_ratio'] = np.where(
                    df_clean['Clear_Sky_GHI_Adjusted'] > 10,
                    df_clean['GHI_lag (t-1)'] / df_clean['Clear_Sky_GHI_Adjusted'],
                    0
                )
                lag_features.append('GHI_lag_ratio')
            else:
                # Enhanced lag features if GHI_lag isn't provided directly
                lag_features = []
                for lag in [1, 2, 3]:
                    df_clean[f'GHI_{lag}h_lag'] = df_clean['GHI - W/m^2'].shift(lag)
                    df_clean[f'GHI_Change_{lag}h'] = df_clean['GHI - W/m^2'].diff(lag)
                    df_clean[f'Clear_Sky_Ratio_{lag}h_lag'] = df_clean['GHI_Clear_Sky_Ratio'].shift(lag)
                    df_clean[f'Clear_Sky_GHI_{lag}h_lag'] = df_clean['Clear_Sky_GHI_Adjusted'].shift(lag)
                    
                    # Add normalized lag changes
                    df_clean[f'Normalized_Change_{lag}h'] = (
                        df_clean[f'GHI_Change_{lag}h'] / 
                        df_clean['Clear_Sky_GHI_Adjusted'].clip(1, None)
                    )
                    lag_features.extend([
                        f'GHI_{lag}h_lag', f'GHI_Change_{lag}h', 
                        f'Clear_Sky_Ratio_{lag}h_lag', f'Clear_Sky_GHI_{lag}h_lag', 
                        f'Normalized_Change_{lag}h'
                    ])
            
            # Enhanced weather interaction features
            df_clean['GHI_UV_Interaction'] = df_clean['GHI - W/m^2'] * df_clean['UV Index']
            df_clean['Temp_Humidity_Interaction'] = df_clean['Temp - °C'] * df_clean['Hum - %']
            df_clean['GHI_Temp_Interaction'] = df_clean['GHI - W/m^2'] * df_clean['Temp - °C']
            df_clean['GHI_Wind_Interaction'] = df_clean['GHI - W/m^2'] * np.log1p(df_clean['Avg Wind Speed - km/h'])
            
            # Daytime interaction features
            df_clean['Daytime_GHI_Interaction'] = df_clean['Daytime'] * df_clean['GHI - W/m^2']
            df_clean['Daytime_Temp_Interaction'] = df_clean['Daytime'] * df_clean['Temp - °C']
            
            # Handle missing values
            df_clean = self._handle_missing_values(df_clean)
            
            # Store historical values
            self.historical_values = df_clean.tail(24)[
                ['GHI - W/m^2', 'GHI_Clear_Sky_Ratio', 'Clear_Sky_GHI_Adjusted', 'Is_Rapid_Change']
            ].copy()
            
            # Update feature columns
            self._update_feature_columns(df_clean, lag_features)
            
            # Prepare features and target
            X = df_clean[self.feature_columns].values
            y = df_clean['GHI - W/m^2'].values
            
            # Scale features and target
            X_scaled = self.features_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1))
            
            return X_scaled, y_scaled, df_clean
            
        except Exception as e:
            print(f"Error in prepare_data_for_training: {str(e)}")
            print("DataFrame columns:", df_clean.columns.tolist())
            raise
    
    def _handle_missing_values(self, df):
        """Enhanced missing value handling"""
        # First pass: Interpolate within same weather conditions
        for condition in ['Is_Clear', 'Is_Mostly_Clear', 'Is_Partly_Cloudy', 'Is_Mostly_Cloudy', 'Is_Overcast']:
            mask = df[condition] == 1
            df.loc[mask] = df.loc[mask].interpolate(method='time', limit=2)
        
        # Second pass: Fill remaining gaps
        for col in df.columns:
            if df[col].isnull().any():
                # Short gaps: Linear interpolation
                df[col] = df[col].interpolate(method='linear', limit=2)
                
                # Medium gaps: Polynomial interpolation
                if df[col].isnull().any():
                    df[col] = df[col].interpolate(method='polynomial', order=2, limit=4)
                
                # Long gaps: Forward fill then backward fill
                if df[col].isnull().any():
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _update_feature_columns(self, df, lag_features=[]):
        """Update feature columns list with new features"""
        base_features = [
            'Temp - °C',          # Updated column name
            'Hum - %',
            'Dew Point - °C',     # Updated column name
            'Wet Bulb - °C',      # Updated column name
            'Barometer - hPa',
            'Avg Wind Speed - km/h',
            'Wind Run - km',
            'UV Index',
            'GHI_lag (t-1)',      # New column from dataset
            'Daytime',            # New column from dataset
            'Month of Year',      # New column from dataset
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
            'Month_Sin', 'Month_Cos',  # New month cyclical features
            'Cos_Hour_Angle', 'Sin_Solar_Elevation',
            'Clear_Sky_GHI', 'Clear_Sky_GHI_Adjusted',
            'GHI_Clear_Sky_Ratio', 'Air_Mass', 'Atmospheric_Attenuation',
            'Solar Zenith Angle', 'Solar Elevation Angle',
            'Hour of Day', 'Day of Year'
        ]
        
        condition_features = [
            'Is_Clear', 'Is_Mostly_Clear', 'Is_Partly_Cloudy',
            'Is_Mostly_Cloudy', 'Is_Overcast', 'Is_Transition',
            'Is_Rapid_Change'
        ]
        
        change_features = [
            'GHI_Change', 'GHI_Change_Rate', 'GHI_Acceleration',
            'Recent_Variability', 'Recent_Trend'
        ]
        
        interaction_features = [
            'GHI_UV_Interaction', 'Temp_Humidity_Interaction',
            'GHI_Temp_Interaction', 'GHI_Wind_Interaction',
            'Daytime_GHI_Interaction', 'Daytime_Temp_Interaction'
        ]
        
        # Combine all features
        self.feature_columns = (
            base_features +
            condition_features +
            change_features +
            interaction_features +
            lag_features +
            [col for col in df.columns if any(
                pattern in col for pattern in [
                    'GHI_Variability_', 'GHI_Volatility_',
                    '_persistence', 'EMA_', '_mean', '_std'
                ]
            )]
        )
        
        # Remove duplicates
        self.feature_columns = list(dict.fromkeys(self.feature_columns))
        
        # Remove any features not in dataframe
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]

    def get_input_data(self):
        """Get input data from user with validation - updated for new dataset format"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'dataset.csv')
            
            # Read existing CSV and get last timestamp - FIX: Add proper encoding handling
            try:
                # First attempt with latin-1 encoding (commonly works with Windows files)
                df = pd.read_csv(csv_path, encoding='latin-1')
            except:
                try:
                    # Second attempt with cp1252 encoding
                    df = pd.read_csv(csv_path, encoding='cp1252')
                except:
                    # Third attempt with ISO-8859-1
                    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
            
            # Handle different date formats with updated column names
            try:
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['End Period'])
            except:
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['End Period'], format='%m/%d/%Y %H:%M')
            
            # Get the last end time
            last_end_time = df['datetime'].max()
            
            # Set the next period (XX:00 to (XX+1):00)
            next_start = last_end_time.replace(minute=0)  # Start at XX:00
            next_end = (next_start + timedelta(hours=1)).replace(minute=0)  # End at (XX+1):00
            
            # Use end hour for solar calculations instead of midpoint
            calculation_time = next_end  # This will be 11:00 for the 10:00-11:00 period
            
            print(f"\nEntering weather data for period:")
            print(f"From: {next_start.strftime('%Y-%m-%d %H:%M')}")
            print(f"To: {next_end.strftime('%Y-%m-%d %H:%M')}")
            
            # Calculate previous GHI value for GHI_lag feature
            try:
                last_ghi = df.iloc[-1]['GHI - W/m^2']
            except:
                last_ghi = 0
            
            # Get current time
            input_data = {
                'Date': next_start.strftime('%Y-%m-%d'),
                'Start Period': next_start.strftime('%H:%M'),
                'End Period': next_end.strftime('%H:%M'),
                'Hour of Day': calculation_time.hour + calculation_time.minute/60,  # Use 11.0 for 11:00
                'Day of Year': calculation_time.timetuple().tm_yday,
                'Month of Year': calculation_time.month,
                'GHI_lag (t-1)': last_ghi
            }
            
            print("\nEnter weather data:")
            input_data.update({
                'Barometer - hPa': float(input("Barometer (hPa): ")),
                'Temp - °C': float(input("Temperature (°C): ")),
                'Hum - %': float(input("Humidity (%): ")),
                'Dew Point - °C': float(input("Dew Point (°C): ")),
                'Wet Bulb - °C': float(input("Wet Bulb (°C): ")),
                'Avg Wind Speed - km/h': float(input("Average Wind Speed (km/h): ")),
                'Wind Run - km': float(input("Wind Run (km): ")),
                'UV Index': float(input("UV Index: ")),
                'GHI - W/m^2': float(input("Current GHI (W/m^2): "))
            })
            
            # Calculate solar parameters
            solar_angles = self.compute_solar_angles(pd.Series(input_data))
            input_data['Solar Zenith Angle'] = solar_angles['Solar Zenith Angle']
            # We still need this for internal calculations, but won't display it
            input_data['Solar Elevation Angle'] = solar_angles['Solar Elevation Angle']
            
            # Calculate clearness index
            input_data['Clearness Index'] = self.compute_clearness_index(
                input_data['GHI - W/m^2'],
                input_data['Solar Zenith Angle']
            )
            
            # Set Daytime flag based on hour of day instead of solar elevation
            input_data['Daytime'] = 1 if (input_data['Hour of Day'] >= 6 and input_data['Hour of Day'] <= 18) else 0
            
            # If clearness index is NaN, set it to 0
            if np.isnan(input_data['Clearness Index']):
                input_data['Clearness Index'] = 0
            
            print("\nComputed parameters:")
            print(f"Solar Zenith Angle: {input_data['Solar Zenith Angle']:.2f}°")
            print(f"Daytime: {input_data['Daytime']}")
            
            return input_data
            
        except ValueError as e:
            print(f"Invalid input: {str(e)}")
            raise
        except Exception as e:
            print(f"Error in get_input_data: {str(e)}")
            raise

    def save_models(self):
        """Save trained models and scalers in the same directory as dataset.csv"""
        try:
            # Get script directory (where dataset.csv is located)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Save XGBoost models (median and intervals)
            for i, (model, model_lower, model_upper) in enumerate(zip(self.xgb_models, self.xgb_models_lower, self.xgb_models_upper)):
                model_path = os.path.join(script_dir, f'xgboost_model_hour_{i+1}.json')
                model_lower_path = os.path.join(script_dir, f'xgboost_model_lower_hour_{i+1}.json')
                model_upper_path = os.path.join(script_dir, f'xgboost_model_upper_hour_{i+1}.json')
                
                model.save_model(model_path)
                model_lower.save_model(model_lower_path)
                model_upper.save_model(model_upper_path)
                print(f"Saved models for hour {i+1}")
            
            # Save scalers
            scaler_paths = {
                'features_scaler': os.path.join(script_dir, 'features_scaler.joblib'),
                'target_scaler': os.path.join(script_dir, 'target_scaler.joblib')
            }
            
            joblib.dump(self.features_scaler, scaler_paths['features_scaler'])
            joblib.dump(self.target_scaler, scaler_paths['target_scaler'])
            print(f"Saved scalers to {script_dir}")
            
            # Save feature columns
            feature_path = os.path.join(script_dir, 'feature_columns.json')
            with open(feature_path, 'w') as f:
                json.dump(self.feature_columns, f)
            print(f"Saved feature columns to {feature_path}")
            
            print("\nAll models and associated files saved successfully!")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise

    def load_models(self):
        """Load trained models and scalers from the same directory as dataset.csv"""
        try:
            # Get script directory (where dataset.csv is located)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Load XGBoost models (median and intervals)
            self.xgb_models = []
            self.xgb_models_lower = []
            self.xgb_models_upper = []
            
            for i in range(self.FORECAST_HOURS):
                model_path = os.path.join(script_dir, f'xgboost_model_hour_{i+1}.json')
                model_lower_path = os.path.join(script_dir, f'xgboost_model_lower_hour_{i+1}.json')
                model_upper_path = os.path.join(script_dir, f'xgboost_model_upper_hour_{i+1}.json')
                
                if not all(os.path.exists(p) for p in [model_path, model_lower_path, model_upper_path]):
                    raise FileNotFoundError(f"Model files for hour {i+1} not found")
                
                model = xgb.XGBRegressor()
                model_lower = xgb.XGBRegressor()
                model_upper = xgb.XGBRegressor()
                
                model.load_model(model_path)
                model_lower.load_model(model_lower_path)
                model_upper.load_model(model_upper_path)
                
                self.xgb_models.append(model)
                self.xgb_models_lower.append(model_lower)
                self.xgb_models_upper.append(model_upper)
                print(f"Loaded models for hour {i+1}")
            
            # Load scalers
            scaler_paths = {
                'features_scaler': os.path.join(script_dir, 'features_scaler.joblib'),
                'target_scaler': os.path.join(script_dir, 'target_scaler.joblib')
            }
            
            for name, path in scaler_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Scaler file '{path}' not found")
            
            self.features_scaler = joblib.load(scaler_paths['features_scaler'])
            self.target_scaler = joblib.load(scaler_paths['target_scaler'])
            print(f"Loaded scalers from {script_dir}")
            
            # Load feature columns
            feature_path = os.path.join(script_dir, 'feature_columns.json')
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Feature columns file '{feature_path}' not found")
            
            with open(feature_path, 'r') as f:
                self.feature_columns = json.load(f)
            print(f"Loaded feature columns from {feature_path}")
            
            print("\nAll models and associated files loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def compute_solar_angles(self, row):
        """Compute solar angles for a given row of data"""
        try:
            # Print inputs for debugging
            print(f"Computing solar angles with: Day of Year={row['Day of Year']}, Hour of Day={row['Hour of Day']}, Latitude={self.lat}")
            
            # Calculate declination based on day of year
            declination = np.radians(23.45 * np.sin(np.radians(360/365 * (row['Day of Year'] + 284))))
            print(f"Declination: {np.degrees(declination):.4f}°")
            
            # Calculate hour angle
            hour_angle = np.radians(15 * (row['Hour of Day'] - 12))
            print(f"Hour angle: {np.degrees(hour_angle):.4f}°")
            
            # Convert latitude to radians
            latitude_rad = np.radians(self.lat)
            
            # Compute solar elevation angle
            solar_elevation_angle = np.arcsin(
                np.sin(latitude_rad) * np.sin(declination) +
                np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
            )
            
            # Compute solar zenith angle
            solar_zenith_angle = np.pi/2 - solar_elevation_angle
            
            # Print intermediate calculation values
            print(f"sin(latitude): {np.sin(latitude_rad):.6f}")
            print(f"sin(declination): {np.sin(declination):.6f}")
            print(f"cos(latitude): {np.cos(latitude_rad):.6f}")
            print(f"cos(declination): {np.cos(declination):.6f}")
            print(f"cos(hour_angle): {np.cos(hour_angle):.6f}")
            
            # Print each term in the elevation angle calculation
            term1 = np.sin(latitude_rad) * np.sin(declination)
            term2 = np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
            print(f"Term 1 (sin(lat)*sin(decl)): {term1:.6f}")
            print(f"Term 2 (cos(lat)*cos(decl)*cos(hour)): {term2:.6f}")
            print(f"Sum of terms: {term1 + term2:.6f}")
            
            print(f"Solar zenith angle: {np.degrees(solar_zenith_angle):.4f}°")
            # Remove solar elevation angle print
            
            # We'll still return both values as they might be used internally,
            # but we won't display the elevation angle to the user
            return pd.Series({
                'Solar Zenith Angle': np.degrees(solar_zenith_angle),
                'Solar Elevation Angle': np.degrees(solar_elevation_angle)
            })
            
        except Exception as e:
            print(f"Error in compute_solar_angles: {str(e)}")
            raise

    def compute_clearness_index(self, ghi, solar_zenith_angle):
        """Compute clearness index using the provided formula"""
        try:
            # If solar zenith angle is >= 90° or NaN, return NaN
            if solar_zenith_angle >= 90 or np.isnan(solar_zenith_angle):
                return np.nan
            
            # If GHI > solar constant or GHI < 0 or GHI is NaN, return NaN
            if ghi > self.SOLAR_CONSTANT or ghi < 0 or np.isnan(ghi):
                return np.nan
            
            # Calculate clearness index directly using solar constant
            clearness_index = ghi / self.SOLAR_CONSTANT
            
            return clearness_index
            
        except Exception as e:
            print(f"Error in compute_clearness_index: {str(e)}")
            return np.nan

    # Add this as a new method
    def calculate_solar_time(self, local_hour, day_of_year):
        """Convert local time to solar time"""
        # Standard meridian for Philippines time (PHT/GMT+8)
        standard_meridian = 120  # 120° E
        
        # Calculate equation of time (minutes)
        b = (360/365) * (day_of_year - 81)  # in degrees
        eot = 9.87 * np.sin(2*np.radians(b)) - 7.53 * np.cos(np.radians(b)) - 1.5 * np.sin(np.radians(b))
        
        # Time correction factor (minutes)
        tc = 4 * (self.lon - standard_meridian) + eot  # 4 minutes per degree longitude difference
        
        # Convert local time to solar time (decimal hours)
        solar_time = local_hour + tc/60
        
        print(f"Local time: {local_hour:.2f}, Solar time: {solar_time:.2f} (correction: {tc/60:.2f} hours)")
        
        return solar_time

if __name__ == "__main__":
    try:
        predictor = WeatherPredictor()
        
        while True:
            print("\nGHI Forecast System")
            print("1. Train Model")
            print("2. Make Prediction")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            try:
                if choice == '1':
                    # Get script directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    dataset_path = os.path.join(script_dir, 'dataset.csv')
                    
                    # Load and prepare data
                    X_scaled, y_scaled, df = predictor.load_and_prepare_data(dataset_path)
                    
                    # Prepare data for XGBoost with proper splits
                    X_train, y_train_split, X_val, y_val_split, X_test, y_test_split = predictor.prepare_xgboost_data(X_scaled, y_scaled)
                    
                    # Train model with validation data
                    print("\nTraining XGBoost models...")
                    predictor.train_xgboost(X_train, y_train_split, X_val, y_val_split)
                    
                    # Save models and associated files
                    print("\nSaving trained models...")
                    predictor.save_models()
                    
                    # Evaluate model on test set
                    print("\nEvaluating on test set:")
                    test_results = predictor.evaluate_model(X_test, y_test_split)
                    
                elif choice == '2':
                    try:
                        print("\nLoading trained models...")
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        
                        # Check if all required model files exist
                        required_files = []
                        for i in range(predictor.FORECAST_HOURS):
                            required_files.extend([
                                f'xgboost_model_hour_{i+1}.json',
                                f'xgboost_model_lower_hour_{i+1}.json',
                                f'xgboost_model_upper_hour_{i+1}.json'
                            ])
                        required_files.extend(['features_scaler.joblib', 'target_scaler.joblib', 'feature_columns.json'])
                        
                        missing_files = [f for f in required_files if not os.path.exists(os.path.join(script_dir, f))]
                        if missing_files:
                            print("\nError: Some model files are missing:")
                            for f in missing_files:
                                print(f"  - {f}")
                            print("\nPlease train the models first (Option 1).")
                            continue
                        
                        predictor.load_models()
                        print("\nMaking predictions...")
                        predictions = predictor.make_prediction_from_input()
                        
                        if predictions:
                            predictions_median, predictions_lower, predictions_upper = predictions
                            print("\nSummary of Predictions:")
                            print("=" * 60)
                            
                            # Fix for the summary section code that reads the CSV file to determine time
                            try:
                                # Get the input time from the last row of dataset.csv
                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                csv_path = os.path.join(script_dir, 'dataset.csv')
                                
                                # Default value in case we fail to read the CSV
                                start_time = "Unknown"
                                
                                # Try to read the CSV with appropriate encoding detection
                                try:
                                    # Try to detect the encoding first
                                    with open(csv_path, 'rb') as f:
                                        result = chardet.detect(f.read())
                                        detected_encoding = result['encoding']
                                        if not detected_encoding or result['confidence'] < 0.7:
                                            detected_encoding = 'latin-1'  # Default to latin-1
                                except:
                                    detected_encoding = 'latin-1'  # Default encoding if detection fails
                                
                                # Now read the CSV with the detected encoding
                                try:
                                    last_row = pd.read_csv(csv_path, encoding=detected_encoding).iloc[-1]
                                    
                                    # Use 'Start Period' with space instead of 'Start_period'
                                    if 'Start Period' in last_row:
                                        start_time = last_row['Start Period']
                                    elif 'Start_period' in last_row:  # Fallback for backward compatibility
                                        start_time = last_row['Start_period']
                                except Exception as e:
                                    print(f"Warning: Could not read last prediction time: {str(e)}")
                                    # start_time remains the default "Unknown"
                                
                                # Check if it's 7:00-8:00 period
                                is_seven_to_eight = start_time == '07:00' or start_time == '07:00:00'
                                
                            except Exception as e:
                                print(f"Error during prediction summary: {str(e)}")
                                traceback.print_exc()
                                # Set default values if anything fails
                                start_time = "Unknown"
                                is_seven_to_eight = False
                            
                            for hour in range(predictor.FORECAST_HOURS):
                                print(f"\nHour {hour+1} Forecast:")
                                print("-" * 40)
                                
                                # Double the bounds only for Hour 1 if it's 7:00-8:00
                                if hour == 0 and is_seven_to_eight:
                                    doubled_upper = predictions_upper[0][hour] * 2.2
                                    doubled_lower = predictions_lower[0][hour] * 2
                                    adjusted_median = (doubled_upper + doubled_lower) / 2
                                    print(f"Upper bound (75th percentile): {doubled_upper:.2f} W/m²")
                                    print(f"Expected (median):          {adjusted_median:.2f} W/m²")
                                    print(f"Lower bound (25th percentile): {doubled_lower:.2f} W/m²")
                                else:
                                    print(f"Upper bound (75th percentile): {predictions_upper[0][hour]:.2f} W/m²")
                                    print(f"Expected (median):          {predictions_median[0][hour]:.2f} W/m²")
                                    print(f"Lower bound (25th percentile): {predictions_lower[0][hour]:.2f} W/m²")
                                    
                            print("\nNote: A prediction is considered excellent if the actual value")
                            print("falls within the interval between worst and best case scenarios.")
                            print("=" * 60)
                            
                    except FileNotFoundError as e:
                        print(f"\nError: {str(e)}")
                        print("Please train the models first (Option 1).")
                    except Exception as e:
                        print(f"\nError during prediction: {str(e)}")
                        traceback.print_exc()
                        
                elif choice == '3':
                    print("\nExiting program...")
                    break
                else:
                    print("\nInvalid choice. Please try again.")
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("Please try again.")
                
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nFull error traceback:")
        traceback.print_exc()
