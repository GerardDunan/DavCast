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
warnings.filterwarnings('ignore')

class WeatherPredictor:
    # Constants that should be configurable
    DEFAULT_LAT = 7.0707  # Davao City
    DEFAULT_LON = 125.6113  # Davao City
    DEFAULT_ELEVATION = 7  # meters
    SOLAR_CONSTANT = 1361  # W/m²
    FORECAST_HOURS = 4
    TRAIN_SIZE = 0.7
    VALIDATION_SIZE = 0.15
    
    # Model parameters
    MODEL_PARAMS = {
        'xgboost': {
            'n_estimators': 3500,
            'max_depth': 7,
            'learning_rate': 0.002,
            'min_child_weight': 4,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'gamma': 0.15,
            'reg_alpha': 0.4,
            'reg_lambda': 1.2,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'random_state': 42,
            'early_stopping_rounds': 200,
            'n_jobs': -1
        }
    }
    
    FILE_PATHS = {
        'xgboost_model': 'best_xgboost_model_hour_{}.json'
    }
    
    FEATURE_COLUMNS = [
        'Barometer - hPa', 
        'Temp - Â°C',
        'Hum - %',
        'Dew Point - Â°C',
        'Wet Bulb - Â°C',
        'Avg Wind Speed - km/h',
        'Wind Run - km',
        'UV Index',
        'Day of Year',
        'Hour of Day',
        'Solar Zenith Angle',
        'Solar Elevation Angle',
        'Clearness Index'
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
        self.features_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
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
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start_period'])
            
            # Compute basic time parameters
            df['Day of Year'] = df['datetime'].apply(lambda x: x.timetuple().tm_yday)
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
            
            # Compute solar angles
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
            
            return df
            
        except Exception as e:
            print(f"Error in compute_solar_parameters: {str(e)}")
            print("DataFrame columns:", df.columns.tolist())
            raise

    def calculate_solar_zenith(self, date):
        """Calculate solar zenith angle"""
        from pvlib import solarposition
        
        # Get solar position
        solar_position = solarposition.get_solarposition(
            time=date,
            latitude=self.lat,
            longitude=self.lon,
            altitude=self.elevation
        )
        
        return solar_position.zenith.iloc[0]

    def calculate_extraterrestrial_radiation(self, date):
        """Calculate extraterrestrial radiation"""
        from pvlib import irradiance
        
        # Get solar position
        solar_position = solarposition.get_solarposition(
            time=date,
            latitude=self.lat,
            longitude=self.lon,
            altitude=self.elevation
        )
        
        # Calculate extraterrestrial radiation
        extra_radiation = irradiance.get_extra_radiation(date)
        
        # Adjust for solar zenith angle
        dni_extra = extra_radiation * np.cos(np.radians(solar_position.zenith.iloc[0]))
        
        return max(0, dni_extra)

    def load_and_prepare_data(self, csv_path):
        """
        Load and prepare data from CSV file with solar parameter computations
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Dataset file not found: {csv_path}")
            
            # Load the dataset
            print(f"Loading dataset from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Print initial dataset info
            print("\nInitial dataset info:")
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            
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
        """Train XGBoost models with improved parameters and quantile predictions"""
        try:
            self.xgb_models = []
            self.xgb_models_lower = []
            self.xgb_models_upper = []
            
            # Ensure feature_columns is initialized with FEATURE_COLUMNS
            if not hasattr(self, 'feature_columns'):
                self.feature_columns = self.FEATURE_COLUMNS.copy()
            
            # Enhanced base parameters for better accuracy
            base_params = {
                'n_estimators': 4000,
                'max_depth': 8,
                'learning_rate': 0.001,
                'min_child_weight': 3,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'gamma': 0.2,
                'reg_alpha': 0.5,
                'reg_lambda': 1.5,
                'tree_method': 'hist',
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Keep original quantile ranges
            quantile_params_lower = base_params.copy()
            quantile_params_lower['objective'] = 'reg:quantileerror'
            quantile_params_lower['quantile_alpha'] = 0.25  # Keep at 25th percentile
            
            quantile_params_upper = base_params.copy()
            quantile_params_upper['objective'] = 'reg:quantileerror'
            quantile_params_upper['quantile_alpha'] = 0.75  # Keep at 75th percentile
            
            # Additional parameters for transition periods
            transition_params_lower = quantile_params_lower.copy()
            transition_params_lower['min_child_weight'] = 2  # Allow more granular splits for transitions
            transition_params_lower['subsample'] = 0.9  # Slightly higher subsample for better stability
            
            transition_params_upper = quantile_params_upper.copy()
            transition_params_upper['min_child_weight'] = 2
            transition_params_upper['subsample'] = 0.9
            
            for i in range(self.FORECAST_HOURS):
                print(f"\nTraining XGBoost model for {i+1}-hour ahead prediction")
                
                # Adjust parameters based on forecast horizon
                horizon_params = base_params.copy()
                if i > 0:
                    # Increase trees and decrease learning rate for longer horizons
                    horizon_params['n_estimators'] += i * 500
                    horizon_params['learning_rate'] *= 0.9
                    horizon_params['subsample'] *= 0.95
                
                # Get solar elevation from features for current training data
                solar_elevation_idx = self.feature_columns.index('Solar Elevation Angle')
                solar_elevation = X_train_list[i][:, solar_elevation_idx]
                
                # Add sample weights for better handling of edge cases
                weights = self._calculate_sample_weights(y_train_split[i], X_train_list[i])
                
                # Train median model with sample weights
                model = xgb.XGBRegressor(**horizon_params)
                model.fit(
                    X_train_list[i], 
                    y_train_split[i],
                    sample_weight=weights,
                    eval_set=[(X_val_list[i], y_val_split[i])],
                    verbose=True
                )
                
                # Train lower bound model with enhanced weights for low values
                model_lower = xgb.XGBRegressor(**quantile_params_lower)
                low_value_weights = weights.copy()
                low_value_mask = y_train_split[i] < np.percentile(y_train_split[i][y_train_split[i] > 0], 25)
                low_value_weights[low_value_mask] *= 1.5  # Increase weight for low values
                
                model_lower.fit(
                    X_train_list[i], 
                    y_train_split[i],
                    sample_weight=low_value_weights,
                    eval_set=[(X_val_list[i], y_val_split[i])],
                    verbose=True
                )
                
                # Train upper bound model
                model_upper = xgb.XGBRegressor(**quantile_params_upper)
                model_upper.fit(
                    X_train_list[i], 
                    y_train_split[i],
                    sample_weight=weights,
                    eval_set=[(X_val_list[i], y_val_split[i])],
                    verbose=True
                )
                
                # Train transition-specific models
                transition_mask = (solar_elevation > -5) & (solar_elevation < 15)
                transition_weights = weights.copy()
                transition_weights[transition_mask] *= 2.0  # Double weight for transitions
                
                model_lower_transition = xgb.XGBRegressor(**transition_params_lower)
                model_upper_transition = xgb.XGBRegressor(**transition_params_upper)
                
                model_lower_transition.fit(
                    X_train_list[i],
                    y_train_split[i],
                    sample_weight=transition_weights,
                    eval_set=[(X_val_list[i], y_val_split[i])],
                    verbose=True
                )
                
                model_upper_transition.fit(
                    X_train_list[i],
                    y_train_split[i],
                    sample_weight=transition_weights,
                    eval_set=[(X_val_list[i], y_val_split[i])],
                    verbose=True
                )
                
                self.xgb_models.append(model)
                self.xgb_models_lower.append((model_lower, model_lower_transition))
                self.xgb_models_upper.append((model_upper, model_upper_transition))
                
                # Print feature importance
                importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                })
                importance = importance.sort_values('importance', ascending=False)
                print(f"\nTop 10 important features for hour {i+1}:")
                print(importance.head(10))
        
        except Exception as e:
            print(f"Error in train_xgboost: {str(e)}")
            raise

    def _calculate_sample_weights(self, y, X):
        """Calculate sample weights to focus on important cases"""
        try:
            # Get solar elevation from features - using Solar Elevation Angle directly
            solar_elevation_idx = self.feature_columns.index('Solar Elevation Angle')  # This matches FEATURE_COLUMNS
            solar_elevation = X[:, solar_elevation_idx]
            
            # Initialize base weights
            weights = np.ones_like(y)
            
            # Upweight transition periods (sunrise/sunset)
            transition_mask = (solar_elevation > -5) & (solar_elevation < 15)
            weights[transition_mask] *= 1.5
            
            # Upweight high GHI periods
            high_ghi_mask = y > np.percentile(y[y > 0], 75)
            weights[high_ghi_mask] *= 1.3
            
            # Upweight rapid changes
            if hasattr(self, 'historical_values'):
                y_series = pd.Series(y)
                changes = y_series.diff().abs()
                rapid_change_mask = changes > np.percentile(changes, 90)
                weights[rapid_change_mask] *= 1.4
            
            # Normalize weights
            weights = weights / weights.mean()
            
            return weights
            
        except Exception as e:
            print(f"Error in _calculate_sample_weights: {str(e)}")
            return np.ones_like(y)

    def predict_xgboost(self, X):
        """Enhanced prediction with improved interval handling"""
        try:
            # Get base predictions from median models (these are not tuples)
            predictions = np.column_stack([
                model.predict(X) for model in self.xgb_models
            ])
            
            # Get predictions from both standard and transition models
            predictions_lower = np.zeros_like(predictions)
            predictions_upper = np.zeros_like(predictions)
            
            # Get solar parameters
            solar_elevation_idx = self.feature_columns.index('Solar Elevation Angle')
            solar_elevation = X[:, solar_elevation_idx]
            
            for i in range(self.FORECAST_HOURS):
                # Unpack the tuples correctly
                model_lower, model_lower_transition = self.xgb_models_lower[i]
                model_upper, model_upper_transition = self.xgb_models_upper[i]
                
                # Get predictions from both models
                pred_lower_std = model_lower.predict(X)
                pred_lower_trans = model_lower_transition.predict(X)
                pred_upper_std = model_upper.predict(X)
                pred_upper_trans = model_upper_transition.predict(X)
                
                # Identify transition periods
                transition_mask = (solar_elevation > -5) & (solar_elevation < 15)
                
                # Use maximum of upper predictions and minimum of lower predictions
                predictions_lower[:, i] = np.where(
                    transition_mask,
                    np.minimum(pred_lower_std, pred_lower_trans),
                    pred_lower_std
                )
                
                predictions_upper[:, i] = np.where(
                    transition_mask,
                    np.maximum(pred_upper_std, pred_upper_trans),
                    pred_upper_std
                )
            
            # Apply physical constraints
            predictions_median = self.target_scaler.inverse_transform(predictions)
            predictions_lower = self.target_scaler.inverse_transform(predictions_lower)
            predictions_upper = self.target_scaler.inverse_transform(predictions_upper)
            
            # Calculate safety margin based on historical error patterns
            if hasattr(self, 'historical_values'):
                error_margin = np.abs(predictions_median - self.historical_values['GHI - W/m^2'].values[-1])
                # Add error margin to upper bound and subtract from lower bound
                predictions_upper = predictions_upper + error_margin[:, np.newaxis]
                predictions_lower = np.maximum(0, predictions_lower - error_margin[:, np.newaxis])
            
            # Ensure physical constraints with wider bounds
            predictions_lower = np.maximum(0, predictions_lower)  # GHI can't be negative
            max_theoretical = self.SOLAR_CONSTANT * np.sin(np.radians(solar_elevation))[:, np.newaxis] * 1.2  # Allow 20% above theoretical
            predictions_upper = np.minimum(predictions_upper, max_theoretical)
            
            # Ensure proper ordering while maintaining wider intervals
            predictions_lower = np.minimum(predictions_lower, predictions_median * 0.9)  # Allow lower bound to be 10% below median
            predictions_upper = np.maximum(predictions_upper, predictions_median * 1.15)  # Allow upper bound to be 15% above median
            
            # Handle night periods
            night_mask = solar_elevation <= -5
            transition_mask = (solar_elevation > -5) & (solar_elevation <= 0)
            
            # Smooth transition for night periods
            transition_factor = np.zeros_like(solar_elevation)
            transition_factor[transition_mask] = (solar_elevation[transition_mask] + 5) / 5
            
            # Reshape transition_factor for broadcasting
            transition_factor = transition_factor.reshape(-1, 1)
            
            # Apply night and transition adjustments
            predictions_median[night_mask] = 0
            predictions_lower[night_mask] = 0
            predictions_upper[night_mask] = 0
            
            # Apply transition factor with proper broadcasting
            if np.any(transition_mask):
                predictions_median[transition_mask] = predictions_median[transition_mask] * transition_factor[transition_mask]
                predictions_lower[transition_mask] = predictions_lower[transition_mask] * transition_factor[transition_mask]
                predictions_upper[transition_mask] = predictions_upper[transition_mask] * transition_factor[transition_mask]
            
            return predictions_median, predictions_lower, predictions_upper
            
        except Exception as e:
            print(f"Error in predict_xgboost: {str(e)}")
            raise

    def evaluate_model(self, X_data, y_data):
        """Evaluate XGBoost models with enhanced validation metrics"""
        print("\nEvaluating XGBoost models (55-minute averages):")
        
        all_metrics = []
        
        for i in range(self.FORECAST_HOURS):
            print(f"\nEvaluating {i+1}-hour ahead predictions")
            print(f"(XX:05 to (XX+1):00 averages)")
            
            # Get corresponding data for this horizon
            X = X_data[i]
            y_true = y_data[i]
            
            # Make predictions with all models
            y_pred = self.xgb_models[i].predict(X)
            
            # Correctly unpack and use the tuple models
            model_lower, model_lower_transition = self.xgb_models_lower[i]
            model_upper, model_upper_transition = self.xgb_models_upper[i]
            
            # Get predictions from both standard and transition models
            y_pred_lower = model_lower.predict(X)
            y_pred_upper = model_upper.predict(X)
            
            # Inverse transform predictions and true values
            y_true = self.target_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
            y_pred_lower = self.target_scaler.inverse_transform(y_pred_lower.reshape(-1, 1)).reshape(-1)
            y_pred_upper = self.target_scaler.inverse_transform(y_pred_upper.reshape(-1, 1)).reshape(-1)
            
            # Calculate standard metrics
            metrics = self.calculate_metrics(y_true, y_pred)
            
            # Add interval metrics
            metrics['interval_coverage'] = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
            metrics['avg_interval_width'] = np.mean(y_pred_upper - y_pred_lower)
            
            # High GHI cases (> 300 W/m²)
            high_mask = y_true > 300
            if np.any(high_mask):
                metrics['high_ghi_rmse'] = np.sqrt(mean_squared_error(y_true[high_mask], y_pred[high_mask]))
                high_coverage = (y_true[high_mask] >= y_pred_lower[high_mask]) & (y_true[high_mask] <= y_pred_upper[high_mask])
                metrics['high_ghi_coverage'] = np.mean(high_coverage)
            else:
                metrics['high_ghi_rmse'] = np.nan
                metrics['high_ghi_coverage'] = np.nan
            
            # Low GHI cases (< 10 W/m²)
            low_mask = y_true < 10
            if np.any(low_mask):
                metrics['low_ghi_rmse'] = np.sqrt(mean_squared_error(y_true[low_mask], y_pred[low_mask]))
                low_coverage = (y_true[low_mask] >= y_pred_lower[low_mask]) & (y_true[low_mask] <= y_pred_upper[low_mask])
                metrics['low_ghi_coverage'] = np.mean(low_coverage)
            else:
                metrics['low_ghi_rmse'] = np.nan
                metrics['low_ghi_coverage'] = np.nan
            
            # Rapid change metrics
            changes = np.abs(np.diff(y_true))
            rapid_change_mask = np.zeros_like(y_true, dtype=bool)
            rapid_change_mask[1:] = changes > 50  # Define rapid change as >50 W/m² between consecutive measurements
            
            if np.any(rapid_change_mask):
                metrics['rapid_change_rmse'] = np.sqrt(mean_squared_error(
                    y_true[rapid_change_mask],
                    y_pred[rapid_change_mask]
                ))
                rapid_coverage = (y_true[rapid_change_mask] >= y_pred_lower[rapid_change_mask]) & (y_true[rapid_change_mask] <= y_pred_upper[rapid_change_mask])
                metrics['rapid_change_coverage'] = np.mean(rapid_coverage)
            else:
                metrics['rapid_change_rmse'] = np.nan
                metrics['rapid_change_coverage'] = np.nan
            
            # Transition period metrics (sunrise/sunset)
            transition_mask = (y_true > 0) & (y_true < 50)
            if np.any(transition_mask):
                metrics['transition_rmse'] = np.sqrt(mean_squared_error(
                    y_true[transition_mask],
                    y_pred[transition_mask]
                ))
                transition_coverage = (y_true[transition_mask] >= y_pred_lower[transition_mask]) & (y_true[transition_mask] <= y_pred_upper[transition_mask])
                metrics['transition_coverage'] = np.mean(transition_coverage)
            else:
                metrics['transition_rmse'] = np.nan
                metrics['transition_coverage'] = np.nan
            
            all_metrics.append(metrics)
            
            print(f"\nPerformance for the next {i+1}-hour prediction:")
            self.print_metrics(metrics)
            print(f"Interval Coverage: {metrics['interval_coverage']*100:.2f}%")
            print(f"Average Interval Width: {metrics['avg_interval_width']:.2f} W/m²")
            
            # Print extreme case metrics
            print("\nExtreme Case Performance:")
            print(f"High GHI (>300 W/m²) RMSE: {metrics['high_ghi_rmse']:.2f} W/m²")
            print(f"High GHI Coverage: {metrics['high_ghi_coverage']*100:.2f}%")
            print(f"Low GHI (<10 W/m²) RMSE: {metrics['low_ghi_rmse']:.2f} W/m²")
            print(f"Low GHI Coverage: {metrics['low_ghi_coverage']*100:.2f}%")
            print(f"Rapid Change RMSE: {metrics['rapid_change_rmse']:.2f} W/m²")
            print(f"Rapid Change Coverage: {metrics['rapid_change_coverage']*100:.2f}%")
            print(f"Transition Period RMSE: {metrics['transition_rmse']:.2f} W/m²")
            print(f"Transition Period Coverage: {metrics['transition_coverage']*100:.2f}%")
        
        return all_metrics

    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics with improved MAPE handling"""
        try:
            # Basic metrics
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mbe = np.mean(y_pred - y_true)
            nse = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
            
            # Improved MAPE calculation
            # Only consider daytime values and handle low GHI better
            day_mask = y_true > 10  # Increased threshold from 0 to 10 W/m²
            y_true_day = y_true[day_mask]
            y_pred_day = y_pred[day_mask]
            
            if len(y_true_day) > 0:
                # Modified MAPE calculation with better handling of low values
                epsilon = 10  # Small constant to prevent division by very small values
                mape = np.mean(np.abs((y_true_day - y_pred_day) / (y_true_day + epsilon))) * 100
                
                # Weighted MAPE with dynamic weighting
                weights = np.log1p(y_true_day) / np.log1p(y_true_day).mean()  # Log-scale weights
                wmape = np.average(np.abs((y_true_day - y_pred_day) / (y_true_day + epsilon)), 
                                 weights=weights) * 100
                
                # Additional weighted metrics for low GHI
                low_ghi_mask = (y_true > 0) & (y_true <= 50)  # Low GHI range
                if np.any(low_ghi_mask):
                    low_ghi_weights = 1 / (y_true[low_ghi_mask] + epsilon)
                    low_ghi_weights = low_ghi_weights / low_ghi_weights.sum()
                    weighted_low_ghi_error = np.average(
                        np.abs(y_true[low_ghi_mask] - y_pred[low_ghi_mask]),
                        weights=low_ghi_weights
                    )
                else:
                    weighted_low_ghi_error = np.nan
            else:
                mape = np.nan
                wmape = np.nan
                weighted_low_ghi_error = np.nan
            
            return {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'wmape': wmape,
                'mbe': mbe,
                'nse': nse,
                'weighted_low_ghi_error': weighted_low_ghi_error
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
        print("(55-minute averages)")
        
        input_values = {}
        prompts = {
            'Temp - Â°C': 'Average Temperature (°C): ',
            'Hum - %': 'Average Humidity (%): ',
            'Dew Point - Â°C': 'Average Dew Point (°C): ',
            'Wet Bulb - Â°C': 'Average Wet Bulb (°C): ',
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
        df = pd.read_csv(dataset_path)
        
        # Get the last row
        last_row = df.iloc[-1]
        
        # Parse the last timestamp with flexible format
        last_date = last_row['Date']
        last_end = last_row['End_period']
        last_timestamp = pd.to_datetime(f"{last_date} {last_end}", format='mixed')
        
        # Calculate next start time (5 minutes after the hour)
        next_start = last_timestamp + timedelta(minutes=5)
        
        return next_start

    def prepare_prediction_data(self, input_values):
        """Prepare input data for prediction"""
        try:
            # Get next start time from dataset
            next_start = self.get_last_timestamp()
            next_end = next_start.replace(minute=0) + timedelta(hours=1)
            
            # Create a DataFrame with the input values
            df = pd.DataFrame([input_values])
            
            # Add date and time information
            df['Date'] = next_start.strftime('%Y-%m-%d')
            df['Start_period'] = next_start.strftime('%H:%M')  # Will be XX:05
            df['End_period'] = next_end.strftime('%H:%M')  # Will be (XX+1):00
            
            # Add solar parameters for the middle of the averaging period
            # Use XX:32 (middle of XX:05 to XX+1:00) for solar calculations
            calculation_time = next_start + timedelta(minutes=27)
            df['Day of Year'] = calculation_time.timetuple().tm_yday
            df['Hour of Day'] = calculation_time.hour + calculation_time.minute/60
            
            # Calculate solar parameters
            df = self.compute_solar_parameters(df)
            
            # Add Clear_Sky_GHI_Adjusted
            df['Air_Mass'] = 1 / (df['Sin_Solar_Elevation'].clip(0.001, 1))
            df['Air_Mass'] = df['Air_Mass'].clip(1, 38)
            
            # Temperature-adjusted atmospheric attenuation
            temp_factor = (df['Temp - Â°C'] + 273.15) / 298.15  # Normalize to 25°C
            humidity_factor = 1 + (df['Hum - %'] / 100) * 0.1  # Humidity effect
            df['Atmospheric_Attenuation'] = (0.7 ** (df['Air_Mass'] * humidity_factor)) * temp_factor
            df['Clear_Sky_GHI_Adjusted'] = df['Clear_Sky_GHI'] * df['Atmospheric_Attenuation']
            
            # Ensure all required features are present
            for feature in self.feature_columns:
                if feature not in df.columns:
                    print(f"Warning: Missing feature {feature}, setting to 0")
                    df[feature] = 0
            
            # Extract features in correct order
            X = df[self.feature_columns].values
            
            # Scale features
            X_scaled = self.features_scaler.transform(X)
            
            return X_scaled
            
        except Exception as e:
            print(f"Error in prepare_prediction_data: {str(e)}")
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
            
            # Create DataFrame with input
            df_pred = pd.DataFrame([new_row])
            
            # Apply same feature engineering as training
            df_pred['datetime'] = pd.to_datetime(df_pred['Date'] + ' ' + df_pred['Start_period'])
            df_pred.set_index('datetime', inplace=True)
            
            # Add time features
            df_pred['Hour_Sin'] = np.sin(2 * np.pi * df_pred['Hour of Day'] / 24)
            df_pred['Hour_Cos'] = np.cos(2 * np.pi * df_pred['Hour of Day'] / 24)
            df_pred['Day_Sin'] = np.sin(2 * np.pi * df_pred['Day of Year'] / 365)
            df_pred['Day_Cos'] = np.cos(2 * np.pi * df_pred['Day of Year'] / 365)
            
            # Solar position features
            solar_angles = df_pred.apply(self.compute_solar_angles, axis=1)
            df_pred['Solar Zenith Angle'] = solar_angles['Solar Zenith Angle']
            df_pred['Solar Elevation Angle'] = solar_angles['Solar Elevation Angle']
            df_pred['Cos_Hour_Angle'] = np.cos(np.radians(15 * (df_pred['Hour of Day'] - 12)))
            df_pred['Sin_Solar_Elevation'] = np.sin(np.radians(df_pred['Solar Elevation Angle']))
            df_pred['Clear_Sky_GHI'] = self.SOLAR_CONSTANT * df_pred['Sin_Solar_Elevation']
            
            # Add Clear_Sky_GHI_Adjusted calculation
            df_pred['Air_Mass'] = 1 / (df_pred['Sin_Solar_Elevation'].clip(0.001, 1))
            df_pred['Air_Mass'] = df_pred['Air_Mass'].clip(1, 38)
            
            # Temperature-adjusted atmospheric attenuation
            temp_factor = (df_pred['Temp - Â°C'] + 273.15) / 298.15  # Normalize to 25°C
            humidity_factor = 1 + (df_pred['Hum - %'] / 100) * 0.1  # Humidity effect
            df_pred['Atmospheric_Attenuation'] = (0.7 ** (df_pred['Air_Mass'] * humidity_factor)) * temp_factor
            df_pred['Clear_Sky_GHI_Adjusted'] = df_pred['Clear_Sky_GHI'] * df_pred['Atmospheric_Attenuation']
            
            # Weather interaction features
            df_pred['GHI_Clear_Sky_Ratio'] = np.where(
                df_pred['Clear_Sky_GHI'] > 0,
                df_pred['GHI - W/m^2'] / df_pred['Clear_Sky_GHI'],
                0
            )
            df_pred['GHI_Clear_Sky_Ratio'] = df_pred['GHI_Clear_Sky_Ratio'].clip(0, 1.2)
            
            # Weather condition indicators
            df_pred['Is_Clear'] = (df_pred['GHI_Clear_Sky_Ratio'] > 0.8).astype(int)
            df_pred['Is_Mostly_Clear'] = ((df_pred['GHI_Clear_Sky_Ratio'] > 0.6) & 
                                        (df_pred['GHI_Clear_Sky_Ratio'] <= 0.8)).astype(int)
            df_pred['Is_Partly_Cloudy'] = ((df_pred['GHI_Clear_Sky_Ratio'] > 0.4) & 
                                         (df_pred['GHI_Clear_Sky_Ratio'] <= 0.6)).astype(int)
            df_pred['Is_Mostly_Cloudy'] = ((df_pred['GHI_Clear_Sky_Ratio'] > 0.2) & 
                                         (df_pred['GHI_Clear_Sky_Ratio'] <= 0.4)).astype(int)
            df_pred['Is_Overcast'] = (df_pred['GHI_Clear_Sky_Ratio'] <= 0.2).astype(int)
            
            # Transition period detection
            df_pred['Is_Transition'] = (
                (df_pred['Solar Elevation Angle'] > -5) & 
                (df_pred['Solar Elevation Angle'] < 15)
            ).astype(int)
            
            # Change and variability features
            if hasattr(self, 'historical_values'):
                hist_ghi = pd.concat([self.historical_values['GHI - W/m^2'], df_pred['GHI - W/m^2']])
                df_pred['GHI_Change'] = hist_ghi.diff().iloc[-1]
                df_pred['Recent_Variability'] = hist_ghi.tail(5).std()
                df_pred['Recent_Trend'] = hist_ghi.tail(5).diff().mean()
                df_pred['Is_Rapid_Change'] = (abs(df_pred['GHI_Change'] / df_pred['Clear_Sky_GHI_Adjusted'].clip(1, None)) > 0.1).astype(int)
                
                # Add volatility features
                for window in [3, 6, 12]:
                    df_pred[f'GHI_Volatility_{window}h'] = (
                        hist_ghi.tail(window).diff().std() / 
                        df_pred['Clear_Sky_GHI_Adjusted'].clip(1, None)
                    ).iloc[0]
            else:
                # Default values if no historical data
                df_pred['GHI_Change'] = 0
                df_pred['Recent_Variability'] = 0
                df_pred['Recent_Trend'] = 0
                df_pred['Is_Rapid_Change'] = 0
                for window in [3, 6, 12]:
                    df_pred[f'GHI_Volatility_{window}h'] = 0
            
            # Add Clear_Sky_Humidity_Interaction
            df_pred['Clear_Sky_Humidity_Interaction'] = df_pred['Clear_Sky_GHI_Adjusted'] * humidity_factor
            
            # Add GHI rate of change features
            if hasattr(self, 'historical_values'):
                hist_ghi = pd.concat([self.historical_values['GHI - W/m^2'], df_pred['GHI - W/m^2']])
                df_pred['GHI_Change_Rate'] = hist_ghi.diff().iloc[-1] / pd.Timedelta('1H').total_seconds()
                df_pred['GHI_Acceleration'] = hist_ghi.diff().diff().iloc[-1] / pd.Timedelta('1H').total_seconds()
                
                # Add exponential moving averages
                for window in [3, 6, 12]:
                    df_pred[f'GHI_EMA_{window}h'] = hist_ghi.ewm(span=window*2, adjust=False, min_periods=1).mean().iloc[-1]
                    
                    hist_ratio = pd.concat([self.historical_values['GHI_Clear_Sky_Ratio'], df_pred['GHI_Clear_Sky_Ratio']])
                    df_pred[f'GHI_Ratio_EMA_{window}h'] = hist_ratio.ewm(span=window*2, adjust=False, min_periods=1).mean().iloc[-1]
                
                # Add lag features
                for lag in [1, 2, 3]:
                    df_pred[f'GHI_{lag}h_lag'] = self.historical_values['GHI - W/m^2'].iloc[-lag]
                    df_pred[f'Clear_Sky_Ratio_{lag}h_lag'] = self.historical_values['GHI_Clear_Sky_Ratio'].iloc[-lag]
                    df_pred[f'Clear_Sky_GHI_{lag}h_lag'] = self.historical_values['Clear_Sky_GHI_Adjusted'].iloc[-lag]
            else:
                # Default values if no historical data
                df_pred['GHI_Change_Rate'] = 0
                df_pred['GHI_Acceleration'] = 0
                
                for window in [3, 6, 12]:
                    df_pred[f'GHI_EMA_{window}h'] = df_pred['GHI - W/m^2']
                    df_pred[f'GHI_Ratio_EMA_{window}h'] = df_pred['GHI_Clear_Sky_Ratio']
                
                for lag in [1, 2, 3]:
                    df_pred[f'GHI_{lag}h_lag'] = df_pred['GHI - W/m^2']
                    df_pred[f'Clear_Sky_Ratio_{lag}h_lag'] = df_pred['GHI_Clear_Sky_Ratio']
                    df_pred[f'Clear_Sky_GHI_{lag}h_lag'] = df_pred['Clear_Sky_GHI_Adjusted']
            
            # Add interaction features
            df_pred['GHI_UV_Interaction'] = df_pred['GHI - W/m^2'] * df_pred['UV Index']
            df_pred['Temp_Humidity_Interaction'] = df_pred['Temp - Â°C'] * df_pred['Hum - %']
            df_pred['GHI_Temp_Interaction'] = df_pred['GHI - W/m^2'] * df_pred['Temp - Â°C']
            df_pred['GHI_Wind_Interaction'] = df_pred['GHI - W/m^2'] * np.log1p(df_pred['Avg Wind Speed - km/h'])
            
            # Ensure all required features are present
            missing_features = [col for col in self.feature_columns if col not in df_pred.columns]
            if missing_features:
                print("Warning: Missing features:", missing_features)
                for feature in missing_features:
                    df_pred[feature] = 0  # Add missing features with default values
            
            # Get features in correct order
            X_pred = df_pred[self.feature_columns].values
            
            # Scale features
            X_pred_scaled = self.features_scaler.transform(X_pred)
            
            # Make predictions with intervals
            predictions_median, predictions_lower, predictions_upper = self.predict_xgboost(X_pred_scaled)
            
            # Print predictions for all 4 hours with intervals
            print("\nPredicted GHI values with confidence intervals:")
            for hour in range(self.FORECAST_HOURS):
                print(f"\nHour {hour+1}:")
                print(f"Upper bound (75th percentile): {float(predictions_upper[0, hour]):.2f} W/m²")
                print(f"Expected (median): {float(predictions_median[0, hour]):.2f} W/m²")
                print(f"Lower bound (25th percentile): {float(predictions_lower[0, hour]):.2f} W/m²")
            
            # Save input data and predictions to CSV
            self.save_to_csv(new_row, predictions_median)
            
            return predictions_median, predictions_lower, predictions_upper
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            print("Available columns:", df_pred.columns.tolist())
            print("Required features:", self.feature_columns)
            raise

    def save_to_csv(self, input_data, predictions):
        """Save only input data to CSV file with proper time handling"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'dataset.csv')
            
            # Read existing CSV
            df = pd.read_csv(csv_path)
            
            # Determine the date format from existing data
            sample_date = df['Date'].iloc[0]
            date_format = '%Y-%m-%d' if '-' in sample_date else '%m/%d/%Y'
            
            # Convert input date to match the format in CSV
            input_date = datetime.strptime(input_data['Date'], '%Y-%m-%d')
            formatted_date = input_date.strftime(date_format)
            
            # Create new row with input data using the correct date format
            new_row = {
                'Date': formatted_date,
                'Start_period': input_data['Start_period'],
                'End_period': input_data['End_period'],
                'Barometer - hPa': input_data['Barometer - hPa'],
                'Temp - Â°C': input_data['Temp - Â°C'],
                'Hum - %': input_data['Hum - %'],
                'Dew Point - Â°C': input_data['Dew Point - Â°C'],
                'Wet Bulb - Â°C': input_data['Wet Bulb - Â°C'],
                'Avg Wind Speed - km/h': input_data['Avg Wind Speed - km/h'],
                'Wind Run - km': input_data['Wind Run - km'],
                'UV Index': input_data['UV Index'],
                'GHI - W/m^2': input_data['GHI - W/m^2'],
                'Solar Zenith Angle': input_data['Solar Zenith Angle'],
                'Solar Elevation Angle': input_data['Solar Elevation Angle'],
                'Clearness Index': input_data['Clearness Index'],
                'Hour of Day': input_data['Hour of Day'],
                'Day of Year': input_data['Day of Year']
            }
            
            # Create DataFrame from new row
            new_df = pd.DataFrame([new_row])
            
            # Append new row to existing DataFrame
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Sort by date and time
            # Convert dates to datetime for sorting, handling both formats
            def parse_date(date_str, time_str):
                try:
                    if '-' in date_str:
                        return pd.to_datetime(f"{date_str} {time_str}", format='%Y-%m-%d %H:%M')
                    else:
                        return pd.to_datetime(f"{date_str} {time_str}", format='%m/%d/%Y %H:%M')
                except:
                    return pd.to_datetime(f"{date_str} {time_str}")
            
            df['datetime'] = df.apply(lambda x: parse_date(x['Date'], x['Start_period']), axis=1)
            df = df.sort_values('datetime')
            df = df.drop('datetime', axis=1)
            
            # Save back to CSV
            df.to_csv(csv_path, index=False)
            print(f"\nSaved input data to {csv_path}")
            
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
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
            
            # Basic preprocessing (keep existing)
            df_clean['datetime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Start_period'])
            df_clean.set_index('datetime', inplace=True)
            df_clean = df_clean.sort_index()
            
            # Initialize feature_columns with the base FEATURE_COLUMNS first
            self.feature_columns = self.FEATURE_COLUMNS.copy()
            
            # Enhanced time features (keep existing)
            df_clean['Hour_Sin'] = np.sin(2 * np.pi * df_clean['Hour of Day'] / 24)
            df_clean['Hour_Cos'] = np.cos(2 * np.pi * df_clean['Hour of Day'] / 24)
            df_clean['Day_Sin'] = np.sin(2 * np.pi * df_clean['Day of Year'] / 365)
            df_clean['Day_Cos'] = np.cos(2 * np.pi * df_clean['Day of Year'] / 365)
            
            # Rest of the method remains the same...
            
            # IMPROVED: Better solar position features
            df_clean['Cos_Hour_Angle'] = np.cos(np.radians(15 * (df_clean['Hour of Day'] - 12)))
            df_clean['Sin_Solar_Elevation'] = np.sin(np.radians(df_clean['Solar Elevation Angle']))
            df_clean['Clear_Sky_GHI'] = self.SOLAR_CONSTANT * df_clean['Sin_Solar_Elevation']
            
            # IMPROVED: Enhanced atmospheric model with humidity and temperature effects
            df_clean['Air_Mass'] = 1 / (df_clean['Sin_Solar_Elevation'].clip(0.001, 1))
            df_clean['Air_Mass'] = df_clean['Air_Mass'].clip(1, 38)
            
            # New temperature-adjusted atmospheric attenuation
            temp_factor = (df_clean['Temp - Â°C'] + 273.15) / 298.15  # Normalize to 25°C
            humidity_factor = 1 + (df_clean['Hum - %'] / 100) * 0.1  # Humidity effect
            df_clean['Atmospheric_Attenuation'] = (0.7 ** (df_clean['Air_Mass'] * humidity_factor)) * temp_factor
            df_clean['Clear_Sky_GHI_Adjusted'] = df_clean['Clear_Sky_GHI'] * df_clean['Atmospheric_Attenuation']
            
            # IMPROVED: Better rapid change detection
            df_clean['GHI_Change'] = df_clean['GHI - W/m^2'].diff()
            df_clean['GHI_Change_Rate'] = df_clean['GHI_Change'] / df_clean['Clear_Sky_GHI_Adjusted'].clip(1, None)
            df_clean['GHI_Acceleration'] = df_clean['GHI_Change'].diff()
            
            # Normalized change rates
            df_clean['Normalized_Change'] = df_clean['GHI_Change'] / df_clean['Clear_Sky_GHI'].clip(1, None)
            df_clean['Is_Rapid_Change'] = (abs(df_clean['Normalized_Change']) > 0.1).astype(int)
            
            # IMPROVED: Enhanced transition period detection
            df_clean['Is_Transition'] = (
                ((df_clean['Solar Elevation Angle'] > -5) & (df_clean['Solar Elevation Angle'] < 15)) |  # Wider range
                ((df_clean['GHI - W/m^2'] > 0) & (df_clean['GHI - W/m^2'] < 100))  # Higher threshold
            ).astype(int)
            
            # IMPROVED: More sophisticated weather condition detection
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
            
            # IMPROVED: Enhanced temporal features with better smoothing
            df_clean['Recent_Variability'] = df_clean['GHI_Change'].rolling(window=5, center=True).std()
            df_clean['Recent_Trend'] = df_clean['GHI_Change'].rolling(window=5, center=True).mean()
            
            # IMPROVED: Better handling of rolling statistics
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
            
            # IMPROVED: Enhanced lag features
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
            
            # IMPROVED: Enhanced weather interaction features
            df_clean['GHI_UV_Interaction'] = df_clean['GHI - W/m^2'] * df_clean['UV Index']
            df_clean['Temp_Humidity_Interaction'] = df_clean['Temp - Â°C'] * df_clean['Hum - %']
            df_clean['GHI_Temp_Interaction'] = df_clean['GHI - W/m^2'] * df_clean['Temp - Â°C']
            df_clean['GHI_Wind_Interaction'] = df_clean['GHI - W/m^2'] * np.log1p(df_clean['Avg Wind Speed - km/h'])
            
            # Handle missing values (keep existing sophisticated approach)
            df_clean = self._handle_missing_values(df_clean)
            
            # Store historical values
            self.historical_values = df_clean.tail(24)[
                ['GHI - W/m^2', 'GHI_Clear_Sky_Ratio', 'Clear_Sky_GHI_Adjusted', 'Is_Rapid_Change']
            ].copy()
            
            # Update feature columns
            self._update_feature_columns(df_clean)
            
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
    
    def _update_feature_columns(self, df):
        """Update feature columns list with new features"""
        # Keep original FEATURE_COLUMNS
        original_features = self.FEATURE_COLUMNS.copy()
        
        # Additional feature groups
        base_features = [
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
            'Cos_Hour_Angle', 'Sin_Solar_Elevation',
            'Clear_Sky_GHI', 'Clear_Sky_GHI_Adjusted',
            'GHI_Clear_Sky_Ratio', 'Air_Mass', 'Atmospheric_Attenuation'
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
        
        # Combine all features while preserving original ones first
        self.feature_columns = (
            original_features +
            base_features +
            condition_features +
            change_features +
            [col for col in df.columns if any(
                pattern in col for pattern in [
                    'GHI_Variability_', 'GHI_Volatility_',
                    '_persistence', 'EMA_', '_lag', '_Interaction'
                ]
            )]
        )

    def get_input_data(self):
        """Get input data from user with validation"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'dataset.csv')
            
            # Read existing CSV and get last timestamp
            df = pd.read_csv(csv_path)
            
            # Handle different date formats
            try:
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['End_period'])
            except:
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['End_period'], format='%m/%d/%Y %H:%M')
            
            # Get the last end time
            last_end_time = df['datetime'].max()
            
            # Set the next period (XX:05 to (XX+1):00)
            next_start = last_end_time.replace(minute=5)  # Start at XX:05
            next_end = (next_start + timedelta(hours=1)).replace(minute=0)  # End at (XX+1):00
            
            print(f"\nEntering weather data for period:")
            print(f"From: {next_start.strftime('%Y-%m-%d %H:%M')}")
            print(f"To: {next_end.strftime('%Y-%m-%d %H:%M')}")
            
            # Get current time
            input_data = {
                'Date': next_start.strftime('%Y-%m-%d'),
                'Start_period': next_start.strftime('%H:%M'),
                'End_period': next_end.strftime('%H:%M'),
                'Hour of Day': next_start.hour + next_start.minute/60,
                'Day of Year': next_start.timetuple().tm_yday
            }
            
            print("\nEnter weather data:")
            input_data.update({
                'Barometer - hPa': float(input("Barometer (hPa): ")),
                'Temp - Â°C': float(input("Temperature (°C): ")),
                'Hum - %': float(input("Humidity (%): ")),
                'Dew Point - Â°C': float(input("Dew Point (°C): ")),
                'Wet Bulb - Â°C': float(input("Wet Bulb (°C): ")),
                'Avg Wind Speed - km/h': float(input("Average Wind Speed (km/h): ")),
                'Wind Run - km': float(input("Wind Run (km): ")),
                'UV Index': float(input("UV Index: ")),
                'GHI - W/m^2': float(input("Current GHI (W/m^2): "))
            })
            
            # Calculate solar parameters
            solar_angles = self.compute_solar_angles(pd.Series(input_data))
            input_data['Solar Zenith Angle'] = solar_angles['Solar Zenith Angle']
            input_data['Solar Elevation Angle'] = solar_angles['Solar Elevation Angle']
            
            # Calculate clearness index using the new method
            input_data['Clearness Index'] = self.compute_clearness_index(
                input_data['GHI - W/m^2'],
                input_data['Solar Zenith Angle']
            )
            
            # If clearness index is NaN, set it to 0
            if np.isnan(input_data['Clearness Index']):
                input_data['Clearness Index'] = 0
            
            print("\nComputed parameters:")
            print(f"Solar Zenith Angle: {input_data['Solar Zenith Angle']:.2f}°")
            print(f"Solar Elevation Angle: {input_data['Solar Elevation Angle']:.2f}°")
            print(f"Clearness Index: {input_data['Clearness Index']:.4f}")
            
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
            for i, (model, (model_lower, model_lower_transition), (model_upper, model_upper_transition)) in enumerate(zip(self.xgb_models, self.xgb_models_lower, self.xgb_models_upper)):
                # Save median model
                model_path = os.path.join(script_dir, f'xgboost_model_hour_{i+1}.json')
                model.save_model(model_path)
                
                # Save lower bound models
                model_lower_path = os.path.join(script_dir, f'xgboost_model_lower_hour_{i+1}.json')
                model_lower.save_model(model_lower_path)
                model_lower_transition_path = os.path.join(script_dir, f'xgboost_model_lower_transition_hour_{i+1}.json')
                model_lower_transition.save_model(model_lower_transition_path)
                
                # Save upper bound models
                model_upper_path = os.path.join(script_dir, f'xgboost_model_upper_hour_{i+1}.json')
                model_upper.save_model(model_upper_path)
                model_upper_transition_path = os.path.join(script_dir, f'xgboost_model_upper_transition_hour_{i+1}.json')
                model_upper_transition.save_model(model_upper_transition_path)
                
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
                hour = i + 1
                # Load paths for all models
                model_path = os.path.join(script_dir, f'xgboost_model_hour_{hour}.json')
                model_lower_path = os.path.join(script_dir, f'xgboost_model_lower_hour_{hour}.json')
                model_lower_transition_path = os.path.join(script_dir, f'xgboost_model_lower_transition_hour_{hour}.json')
                model_upper_path = os.path.join(script_dir, f'xgboost_model_upper_hour_{hour}.json')
                model_upper_transition_path = os.path.join(script_dir, f'xgboost_model_upper_transition_hour_{hour}.json')
                
                # Check if all model files exist
                model_files = [model_path, model_lower_path, model_lower_transition_path, 
                             model_upper_path, model_upper_transition_path]
                if not all(os.path.exists(p) for p in model_files):
                    raise FileNotFoundError(f"Model files for hour {hour} not found")
                
                # Load all models
                model = xgb.XGBRegressor()
                model_lower = xgb.XGBRegressor()
                model_lower_transition = xgb.XGBRegressor()
                model_upper = xgb.XGBRegressor()
                model_upper_transition = xgb.XGBRegressor()
                
                model.load_model(model_path)
                model_lower.load_model(model_lower_path)
                model_lower_transition.load_model(model_lower_transition_path)
                model_upper.load_model(model_upper_path)
                model_upper_transition.load_model(model_upper_transition_path)
                
                # Store models in the same tuple format as during training
                self.xgb_models.append(model)
                self.xgb_models_lower.append((model_lower, model_lower_transition))
                self.xgb_models_upper.append((model_upper, model_upper_transition))
                
                print(f"Loaded models for hour {hour}")
            
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
            # Calculate declination
            declination = np.radians(23.45 * np.sin(np.radians(360/365 * (row['Day of Year'] + 284))))
            
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
                            hour = i + 1
                            required_files.extend([
                                f'xgboost_model_hour_{hour}.json',
                                f'xgboost_model_lower_hour_{hour}.json',
                                f'xgboost_model_lower_transition_hour_{hour}.json',
                                f'xgboost_model_upper_hour_{hour}.json',
                                f'xgboost_model_upper_transition_hour_{hour}.json'
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
                            print("\nPredicted GHI values with confidence intervals:")
                            for hour in range(predictor.FORECAST_HOURS):
                                print(f"\nHour {hour+1}:")
                                # Extract values correctly from numpy arrays
                                upper = predictions_upper[0][hour] if isinstance(predictions_upper, np.ndarray) else predictions_upper[hour]
                                median = predictions_median[0][hour] if isinstance(predictions_median, np.ndarray) else predictions_median[hour]
                                lower = predictions_lower[0][hour] if isinstance(predictions_lower, np.ndarray) else predictions_lower[hour]
                                
                                print(f"Upper bound (75th percentile): {upper:.2f} W/m²")
                                print(f"Expected (median): {median:.2f} W/m²")
                                print(f"Lower bound (25th percentile): {lower:.2f} W/m²")
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
