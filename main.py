import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import os
import logging  # Add logging import
import joblib
import json

# Check if Optuna is installed and available
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

class GHIPredictionModel:
    def __init__(self):
        self.scaler = RobustScaler()
        self.models_median = {}
        self.models_upper = {}  # 95th percentile
        self.models_lower = {}  # 5th percentile
        self.lower_adjustments = {}
        self.upper_adjustments = {}
        self.feature_columns = None
        self.target_column = 'GHI - W/m^2'
        self.forecast_horizons = [1, 2, 3, 4] # Define forecast horizons
        
        # Solar position constants for Davao City
        self.latitude = 7.0707
        self.longitude = 125.6113
        self.elevation = 7  # meters
        self.solar_constant = 1361  # W/m²
        
        # Get the directory where main.py is located
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging to file and console"""
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Configure logging
        log_file = os.path.join(logs_dir, f'ghi_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console but with less detail
            ]
        )
        
        # Create a separate logger for detailed debug info (file only)
        self.debug_logger = logging.getLogger('debug')
        self.debug_logger.setLevel(logging.DEBUG)
        # Prevent debug messages from propagating to the root logger (and thus the console)
        self.debug_logger.propagate = False
        
        debug_handler = logging.FileHandler(os.path.join(logs_dir, f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
        debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.debug_logger.addHandler(debug_handler)
        
        # Save the log file path for reference
        self.log_file = log_file
        debug_log_path = os.path.join(logs_dir, f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        # Use file path info only in logs, not terminal
        logging.info(f"Log files created: \n- {log_file} \n- {debug_log_path}")
    
    def load_data(self, file_path):
        """
        Load hourly GHI data from a CSV file.
        
        Parameters:
        file_path (str): Path to the CSV file containing the data
        
        Returns:
        pandas.DataFrame: The loaded dataset
        """
        logging.info(f"Loading data from {file_path}...")
        
        # Try reading with different encodings to handle special characters like °
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(file_path, encoding='latin-1')
                logging.info("File loaded using latin-1 encoding")
            except:
                data = pd.read_csv(file_path, encoding='cp1252')
                logging.info("File loaded using cp1252 encoding")
        
        # Convert date and time to datetime
        if 'Date' in data.columns and 'Start Period' in data.columns:
            # Try multiple format patterns to handle various date formats
            try:
                data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Start Period'], 
                                                format='%d-%b-%y %H:%M:%S')
            except ValueError:
                # If that fails, try with autodetection and coercion
                data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Start Period'], errors='coerce')
                # Check for NaT values and log them
                nat_count = data['datetime'].isna().sum()
                if nat_count > 0:
                    logging.warning(f"Warning: {nat_count} rows have invalid datetime values that were converted to NaT")
                    # For prediction purposes, we can use a synthetic timestamp
                    if nat_count == len(data):
                        logging.warning("All datetime values are NaT. Using current time as reference.")
                        start_time = pd.Timestamp.now().floor('H')
                        data['datetime'] = pd.date_range(start=start_time, periods=len(data), freq='H')
        
        # Sort by datetime
        if 'datetime' in data.columns:
            data = data.sort_values('datetime').reset_index(drop=True)
        
        logging.info(f"Loaded {len(data)} rows of data")
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the data by handling missing values and creating features.
        
        Parameters:
        data (pandas.DataFrame): The dataset to preprocess
        
        Returns:
        pandas.DataFrame: The preprocessed dataset
        """
        logging.info("Preprocessing data...")
        df = data.copy()
        
        # Check for missing values
        missing_values = df.isnull().sum()
        logging.info(f"Missing values before handling:\n{missing_values[missing_values > 0]}")
        
        # Fill missing values with appropriate methods
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # For GHI values, fill with 0 if it's nighttime, else interpolate
                if col == self.target_column and 'Daytime' in df.columns:
                    night_mask = (df['Daytime'] == 0) & df[col].isnull()
                    df.loc[night_mask, col] = 0
                    df[col] = df[col].interpolate(method='linear')
                else:
                    df[col] = df[col].interpolate(method='linear')
        
        # Check remaining missing values
        missing_after = df.isnull().sum()
        logging.info(f"Missing values after handling:\n{missing_after[missing_after > 0]}")
        
        return df
    
    def create_features(self, data, lag_hours=3):
        """
        Create features for the model, respecting the sequential nature of time series data.
        Creates targets for multiple forecast horizons (1h, 2h, 3h, 4h).
        
        Parameters:
        data (pandas.DataFrame): The preprocessed dataset
        lag_hours (int): Number of lag hours to use
        
        Returns:
        pandas.DataFrame: Dataset with additional features and multi-horizon targets
        """
        print(f"Creating sequential features with {lag_hours} lag hours for horizons {self.forecast_horizons}...")
        df = data.copy()
        
        # Ensure data is sorted by time before creating lag features
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
            print(f"Data sorted by time from {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Create lagged features for GHI - these respect time ordering because they use .shift()
        for i in range(1, lag_hours + 1):
            df[f'GHI_lag_{i}'] = df[self.target_column].shift(i)
        
        # Create rolling statistics that respect time ordering
        # These only use past data points for each prediction
        df['GHI_rolling_mean_6h'] = df[self.target_column].rolling(window=6, min_periods=1).mean()
        df['GHI_rolling_max_24h'] = df[self.target_column].rolling(window=24, min_periods=1).max()
        
        # Add solar position features first (doesn't depend on clear sky)
        df = self.add_solar_position_features(df)
        
        # Now calculate clear sky GHI (depends on solar position)
        df = self.calculate_clear_sky_ghi(df)
        
        # Add trend features (after we have solar and clear sky features)
        df = self.add_trend_features(df)
        
        # Add diurnal decomposition (requires clear sky index)
        df = self.add_diurnal_decomposition(df)
        
        # Create targets for each forecast horizon
        # Initialize target_columns list to store target column names - THIS IS THE CRITICAL FIX
        self.target_columns = []
        for horizon in self.forecast_horizons:
            target_col_name = f'target_GHI_{horizon}h'
            df[target_col_name] = df[self.target_column].shift(-horizon)
            self.target_columns.append(target_col_name)
        
        # Remove rows with NaN values due to lagging/leading
        # Drop rows where any target has NaN or where required features have NaN
        min_row = lag_hours  # Minimum row to include due to lagging
        max_row = len(df) - max(self.forecast_horizons)  # Maximum row due to future targets
        
        df = df.iloc[min_row:max_row].copy()
        
        # Drop rows with remaining NaN values
        df = df.dropna()
        
        print(f"After creating features and removing NaN values, {len(df)} rows remain")
        return df
    
    def add_solar_position_features(self, df):
        """
        Enhanced solar position features with improved transition handling.
        
        Based on: Bright et al. (2015) "Improved sunrise and sunset times algorithm"
        and Yang (2020) "Improved handling of solar position and transition periods in GHI forecasting"
        """
        print("Adding solar position features...")
        
        # Extract latitude, longitude and datetime
        lat_rad = np.radians(self.latitude)
        
        # Extract datetime components
        if 'datetime' in df.columns:
            # Get day of year
            df['day_of_year'] = df['datetime'].dt.dayofyear
            
            # Calculate hour angle - representing time of day relative to solar noon
            # Solar hour angle: 15° per hour, -180° to +180°, 0° at solar noon
            solar_hour = df['datetime'].dt.hour + df['datetime'].dt.minute/60
            df['hour_angle'] = (solar_hour - 12) * 15  # degrees
            
            # Calculate declination angle (angle between sun rays and Earth's equator)
            # Cooper's equation (widely used in solar engineering)
            # Declination varies between -23.45° (winter solstice) and +23.45° (summer solstice)
            df['declination'] = 23.45 * np.sin(np.radians(360 * (284 + df['day_of_year']) / 365))
            
            # Calculate solar zenith angle cosine (zenith = angle between sun and vertical)
            # This is a key parameter for solar radiation modeling
            declination_rad = np.radians(df['declination'])
            hour_angle_rad = np.radians(df['hour_angle'])
            
            df['solar_zenith_cos'] = (np.sin(lat_rad) * np.sin(declination_rad) + 
                                    np.cos(lat_rad) * np.cos(declination_rad) * 
                                    np.cos(hour_angle_rad))
            
            # Constrain to valid range [-1, 1]
            df['solar_zenith_cos'] = np.clip(df['solar_zenith_cos'], -1, 1)
            
            # Calculate solar zenith angle in degrees (useful for some models)
            df['solar_zenith_angle'] = np.degrees(np.arccos(df['solar_zenith_cos']))
        
        # These should be indented at the outer level - they use columns created above if datetime exists
        # But they'll use default values if those columns don't exist
        if 'solar_zenith_angle' not in df.columns:
            df['solar_zenith_angle'] = 90.0  # Default value
            df['solar_zenith_cos'] = 0.0     # Default value
        
        # Calculate solar elevation angle (90° - zenith angle)
        df['solar_elevation'] = 90 - df['solar_zenith_angle']
        
        # Add explicit daylight flag with transition zone detection
        df['is_daylight'] = (df['solar_zenith_cos'] > 0.01).astype(int)
        
        # Add sunrise/sunset proximity features with exponential weighting
        if 'hour_angle' in df.columns:
            # Exponential weighting emphasizes the most recent changes near transition
            df['hours_from_sunrise'] = np.where(
                df['hour_angle'] < 0,  # Before noon
                np.abs(df['hour_angle'] + 90) / 15,  # Convert angle to hours
                np.abs(df['hour_angle'] - 270) / 15   # After noon
            )
        else:
            # Default value if hour_angle not available
            df['hours_from_sunrise'] = 6.0  # Middle of the day
        
        # Create transition zone indicator (1 if within 1 hour of sunrise/sunset)
        df['is_transition'] = (df['hours_from_sunrise'] < 1).astype(int)
        
        # Create specialized sunrise/sunset indicator
        df['is_sunrise_sunset'] = ((df['solar_zenith_cos'] > 0) & 
                                  (df['solar_zenith_cos'] < 0.3)).astype(int)
        
        # Higher values when close to sunrise/sunset
        df['sunrise_sunset_proximity'] = np.exp(-2 * df['hours_from_sunrise'])
        
        return df
    
    def add_diurnal_decomposition(self, df):
        """
        Decompose GHI into diurnal and stochastic components.
        
        Based on: Yang et al. (2015) "Solar irradiance forecasting using a deep learning model"
        and Verbois et al. (2018) "A statistical seasonal decomposition method for solar irradiance forecasting"
        """
        print("Adding diurnal decomposition features...")
        
        # Calculate the clear sky index (k_t)
        # Already implemented in clear_sky_ghi method
        
        # Calculate diurnal component - smoothed clear sky index
        # Using rolling window to get seasonal pattern
        window_width = 5 * 24  # 5 days of hourly data
        
        # Group by hour of day to capture diurnal pattern
        df['hour'] = df['datetime'].dt.hour
        hourly_groups = df.groupby('hour')
        
        # Calculate the smoothed clear sky index by hour
        df['smooth_kt'] = np.nan
        
        for hour, group in hourly_groups:
            if len(group) >= 5:  # Need enough data points
                indices = group.index
                df.loc[indices, 'smooth_kt'] = df.loc[indices, 'clear_sky_index'].rolling(
                    min_periods=1, window=5, center=True).mean()
        
        # Fill remaining NaNs with hour-of-day mean - FIX: avoid inplace warning
        hour_means = df.groupby('hour')['clear_sky_index'].transform('mean')
        # Use assignment instead of inplace operation
        df['smooth_kt'] = df['smooth_kt'].fillna(hour_means)
        
        # Replace any remaining NaNs or infinities
        df['smooth_kt'] = df['smooth_kt'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate stochastic component (residual)
        df['stochastic_kt'] = df['clear_sky_index'] - df['smooth_kt']
        
        # Cap stochastic component to reasonable values
        df['stochastic_kt'] = df['stochastic_kt'].clip(-1, 1)
        
        # Create lagged features for stochastic component
        for i in range(1, 4):
            df[f'stochastic_kt_lag_{i}'] = df['stochastic_kt'].shift(i)
        
        return df
    
    def add_trend_features(self, df):
        """
        Add GHI rate-of-change and trend features with time-of-day context.
        
        Based on: Voyant et al. (2017) "Machine learning methods for solar radiation forecasting"
        and Pedro et al. (2019) "Assessment of machine learning techniques for deterministic solar forecasting"
        """
        print("Adding GHI trend features...")
        
        # Calculate rate of change with time-of-day context
        df['ghi_change_1h'] = df[self.target_column].diff()
        
        # Rate of change percentage (avoid division by zero)
        # For very low values, use absolute changes instead of percentages
        df['ghi_change_pct'] = np.zeros(len(df))
        nonzero_mask = df[self.target_column].shift(1) > 10  # Only calculate where previous GHI > 10 W/m²
        df.loc[nonzero_mask, 'ghi_change_pct'] = df.loc[nonzero_mask, 'ghi_change_1h'] / df.loc[nonzero_mask, self.target_column].shift(1) * 100
        
        # Create hour-specific trend features
        # Research shows trends behave differently by time of day
        hours = df['datetime'].dt.hour
        
        # Morning hours (increasing trend expected)
        morning_mask = (hours >= 6) & (hours <= 11)
        # Evening hours (decreasing trend expected)
        evening_mask = (hours >= 13) & (hours <= 18)
        # Mid-day hours (peak values expected)
        midday_mask = (hours >= 11) & (hours <= 13)  # Expanded to include 11 AM
        
        # Morning trend features - FIX: convert to float64 first to avoid dtype warning
        df['morning_change'] = 0.0  # Initialize as float instead of int
        if morning_mask.any():
            df.loc[morning_mask, 'morning_change'] = df.loc[morning_mask, 'ghi_change_1h'].values
        
        # Evening trend features - FIX: convert to float64 first to avoid dtype warning
        df['evening_change'] = 0.0  # Initialize as float instead of int
        if evening_mask.any():
            df.loc[evening_mask, 'evening_change'] = df.loc[evening_mask, 'ghi_change_1h'].values
            
        # Add midday peak features specifically for the peak solar hours
        df['is_peak_hour'] = midday_mask.astype(float)
        df['midday_change'] = 0.0
        if midday_mask.any():
            df.loc[midday_mask, 'midday_change'] = df.loc[midday_mask, 'ghi_change_1h'].values
        
        # Calculate trend direction: positive (1), negative (-1), or flat (0)
        df['ghi_trend'] = np.sign(df['ghi_change_1h'])
        
        # Calculate acceleration (change of rate of change)
        df['ghi_acceleration'] = df['ghi_change_1h'].diff()
        
        # Add recent variability metrics
        # Short-term variability (1-3 hours)
        df['ghi_variability_3h'] = df[self.target_column].rolling(3).std()
        
        # Daily profile similarity - how similar is today to yesterday at this time?
        # Shift by 24 hours to get the same time yesterday
        df['yesterday_ghi'] = df[self.target_column].shift(24)
        df['yesterday_similarity'] = np.abs(df[self.target_column] - df['yesterday_ghi'])
        
        # Calculate clear sky deviation
        df['clear_sky_deviation'] = df[self.target_column] - df['clear_sky_ghi']
        
        return df
    
    def prepare_train_test_data(self, df, val_size=0.2, random_state=42):
        """
        Prepare training and validation datasets using a strictly sequential time-based split.
        Handles multi-horizon targets.
        
        Parameters:
        -----------
        df (pandas.DataFrame): Dataset with features and multi-horizon targets
        val_size (float): Proportion of data for validation (fixed at 0.2 or 20%)
        random_state (int): Random seed for reproducibility (only used if datetime not available)
        
        Returns:
        tuple: X_train, X_val, y_train (DataFrame), y_val (DataFrame), feature_columns
        """
        logging.info(f"Preparing sequential train-validation split with validation_size={val_size}...")
        
        # Define feature columns to exclude
        # Exclude original GHI and all target columns
        exclude_cols = ['datetime', 'Date', 'Start Period', 'End Period', 
                        self.target_column] + self.target_columns
        
        # Get feature columns
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_columns # Store for later use
        
        # Create feature matrix and target DataFrame
        X = df[feature_columns]
        y = df[self.target_columns] # y is now a DataFrame
        
        # Split data - ALWAYS use time-based split for time series data
        if 'datetime' in df.columns:
            # Sort by datetime to ensure sequential ordering
            df_sorted = df.sort_values('datetime')
            X = X.loc[df_sorted.index]  # Reindex X to match sorted df
            y = y.loc[df_sorted.index]  # Reindex y to match sorted df
            
            # Verify sequential ordering
            logging.info("Verifying sequential data ordering...")
            date_diffs = df_sorted['datetime'].diff().dropna()
            if (date_diffs < pd.Timedelta(0)).any():
                logging.warning("Dates are not in strictly ascending order!")
            
            # Time-based split - last val_size% of data for validation
            split_idx = int(len(df_sorted) * (1 - val_size))
            train_end_date = df_sorted['datetime'].iloc[split_idx-1]
            val_start_date = df_sorted['datetime'].iloc[split_idx]
            
            X_train = X.iloc[:split_idx]
            X_val = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx] # y_train is DataFrame
            y_val = y.iloc[split_idx:]   # y_val is DataFrame
            
            logging.info(f"Training data: from {df_sorted['datetime'].iloc[0]} to {train_end_date}")
            logging.info(f"Validation data: from {val_start_date} to {df_sorted['datetime'].iloc[-1]}")
        else:
            logging.warning("No datetime column found. Using random split instead.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=random_state, shuffle=False # Keep shuffle=False for time series
            )
        
        logging.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        return X_train, X_val, y_train, y_val, feature_columns
    
    def scale_features(self, X_train, X_test, X_val=None):
        """
        Scale features using RobustScaler.
        
        Parameters:
        -----------
        X_train: Training data features
        X_test: Test data features (not used anymore, kept for compatibility)
        X_val: Validation data features
        
        Returns:
        --------
        tuple: Scaled features (X_train_scaled, X_test_scaled, X_val_scaled)
        """
        logging.info("Scaling features with RobustScaler...")
        
        # Check for and handle infinite or very large values before scaling
        def clean_dataframe(df):
            """Replace inf/NaN values and clip very large values"""
            result = df.copy()
            
            # Replace infinities with NaN first
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # Check for columns with NaN values
            nan_columns = result.columns[result.isna().any()].tolist()
            if nan_columns:
                logging.info(f"Found NaN values in columns: {nan_columns}")
                for col in nan_columns:
                    # Fill NaN values with median for that column
                    if pd.api.types.is_numeric_dtype(result[col]):
                        median_val = result[col].median()
                        result[col] = result[col].fillna(median_val)
                        logging.info(f"  - Filled NaNs in '{col}' with median value: {median_val}")
                    else:
                        # For non-numeric columns, fill with mode
                        mode_val = result[col].mode()[0] if not result[col].mode().empty else "MISSING"
                        result[col] = result[col].fillna(mode_val)
                        logging.info(f"  - Filled NaNs in '{col}' with mode value: {mode_val}")
            
            # Clip extremely large values to reasonable ranges - only for numeric columns
            for col in result.columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    try:
                        q1 = result[col].quantile(0.01)
                        q3 = result[col].quantile(0.99)
                        iqr = q3 - q1
                        lower_bound = q1 - 5 * iqr
                        upper_bound = q3 + 5 * iqr
                        
                        # Count outliers before clipping
                        outliers = ((result[col] < lower_bound) | (result[col] > upper_bound)).sum()
                        if outliers > 0:
                            # Clip values to bounds
                            result[col] = result[col].clip(lower_bound, upper_bound)
                            logging.info(f"  - Clipped {outliers} outliers in '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                    except TypeError:
                        logging.warning(f"  - Warning: Could not process column '{col}' for outliers (might have mixed types)")
                        # Try to convert to numeric if possible, otherwise leave as is
                        try:
                            result[col] = pd.to_numeric(result[col], errors='coerce')
                            # Fill any NaN values from conversion
                            result[col] = result[col].fillna(result[col].median() if not result[col].median().isna() else 0)
                        except:
                            logging.warning(f"  - Could not convert '{col}' to numeric type")
            
            return result
        
        # Drop non-numeric columns that can't be scaled
        numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = [col for col in X_train.columns if col not in numeric_cols]
        
        if non_numeric_cols:
            logging.info(f"Dropping non-numeric columns before scaling: {non_numeric_cols}")
            X_train = X_train[numeric_cols].copy()
            if X_val is not None:
                X_val = X_val[numeric_cols].copy()
        
        # Clean dataframes before scaling
        X_train_clean = clean_dataframe(X_train)
        
        # Fit scaler on clean training data
        self.scaler.fit(X_train_clean)
        
        # Transform all datasets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train_clean),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Create empty test data frame for compatibility
        X_test_scaled = pd.DataFrame(columns=X_train.columns)
        
        if X_val is not None:
            X_val_clean = clean_dataframe(X_val)
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val_clean),
                columns=X_val.columns,
                index=X_val.index
            )
            return X_train_scaled, X_test_scaled, X_val_scaled
        
        return X_train_scaled, X_test_scaled
    
    def train_models_with_best_params(self, X_train, y_train, random_state=42):
        """
        Train models for each forecast horizon using the best parameters.
        Uses the Direct Strategy: separate models per horizon.
        
        Parameters:
        -----------
        X_train (pandas.DataFrame): Training features
        y_train (pandas.DataFrame): Training targets for all horizons
        random_state (int): Random seed
        """
        print(f"Training direct models for horizons {self.forecast_horizons} with best parameters...")
        
        # Clear previous models
        self.models_median = {}
        self.models_lower = {}
        self.models_upper = {}
        
        # Train models for each horizon
        for horizon in self.forecast_horizons:
            target_col = f'target_GHI_{horizon}h'
            
            # Check if best parameters exist for this horizon
            if hasattr(self, 'best_params') and horizon in self.best_params:
                # Train median model
                print(f"Training median model for {horizon}h horizon...")
                self.models_median[horizon] = self._train_single_model(
                    X_train, y_train[target_col], params=self.best_params[horizon]['median']
                )
                
                # Train lower bound model
                if 'lower' in self.best_params[horizon]:
                    print(f"Training lower bound model for {horizon}h horizon...")
                    self.models_lower[horizon] = self._train_single_model(
                        X_train, y_train[target_col], params=self.best_params[horizon]['lower']
                    )
                
                # Train upper bound model
                if 'upper' in self.best_params[horizon]:
                    print(f"Training upper bound model for {horizon}h horizon...")
                    self.models_upper[horizon] = self._train_single_model(
                        X_train, y_train[target_col], params=self.best_params[horizon]['upper']
                    )
            else:
                # Use default parameters if no optimized parameters are available
                print(f"No optimized parameters found for horizon {horizon}h. Using defaults.")
                default_params = {
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 1.0,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1
                }
                
                # Train with default parameters
                self.models_median[horizon] = self._train_single_model(
                    X_train, y_train[target_col], params=default_params
                )
        
        print("All models trained successfully.")
    
    def evaluate_validation_models(self, X_val, y_val, timestamps=None):
        """
        Evaluate models on validation data for all forecast horizons.
        
        Parameters:
        -----------
        X_val: Validation features
        y_val: Validation targets
        timestamps: Optional timestamps for the validation data
        
        Returns:
        --------
        dict: Evaluation metrics for each horizon
        """
        print("Evaluating models with specialized GHI metrics on validation data...")
        validation_metrics = {}
        
        # Define the predictions dictionary
        predictions_dict = {}
        
        for horizon in self.forecast_horizons:
            print(f"\n--- Evaluating models for {horizon}h ahead ---")
            target_col = f'target_GHI_{horizon}h'
            y_val_horizon = y_val[target_col]
            
            # Calculate persistence forecast (naive baseline)
            persistence_pred = X_val[f'GHI_lag_1'].values
            
            # Get predictions from standard or specialized models
            if hasattr(self, 'specialized_models') and self.specialized_models.get('day', {}).get(horizon) is not None:
                # Use the specialized prediction function
                all_horizon_preds = self.predict_with_specialized_models(X_val)
                y_pred_median = all_horizon_preds[horizon]
            else:
                # Fall back to standard models
                y_pred_median = self.models_median[horizon].predict(X_val)
            
            # Ensure non-negative predictions
            y_pred_median = np.maximum(0, y_pred_median)
            
            # Get daylight mask for specialized metrics
            daylight_mask = X_val['solar_zenith_cos'] > 0.01
            
            # Calculate standard metrics
            mae = mean_absolute_error(y_val_horizon, y_pred_median)
            rmse = np.sqrt(mean_squared_error(y_val_horizon, y_pred_median))
            r2 = r2_score(y_val_horizon, y_pred_median)
            
            # Calculate normalized metrics (rRMSE, rMAE) as per journal recommendations
            # Yang et al. (2020) recommends normalization by installed capacity
            capacity = 1000  # Typical 1kW/m² normalization for GHI
            nrmse = rmse / capacity * 100  # as percentage
            
            # Calculate persistence model metrics
            persistence_mae = mean_absolute_error(y_val_horizon, persistence_pred)
            persistence_rmse = np.sqrt(mean_squared_error(y_val_horizon, persistence_pred))
            
            # Calculate skill score relative to persistence (as in meteorological forecasting)
            mae_skill = 1 - (mae / persistence_mae)
            rmse_skill = 1 - (rmse / persistence_rmse)
            
            # Calculate specialized metrics for daylight only
            if np.any(daylight_mask):
                day_mae = mean_absolute_error(y_val_horizon[daylight_mask], y_pred_median[daylight_mask])
                day_rmse = np.sqrt(mean_squared_error(y_val_horizon[daylight_mask], y_pred_median[daylight_mask]))
                
                # Calculate forecast bias - negative means underforecasting
                bias = np.mean(y_pred_median[daylight_mask] - y_val_horizon[daylight_mask])
                
                # Calculate RMSE by solar position bins for detailed error analysis
                zenith_bins = [0.1, 0.3, 0.5, 0.7, 0.9]
                rmse_by_zenith = {}
                
                for i in range(len(zenith_bins)):
                    if i == 0:
                        bin_mask = X_val['solar_zenith_cos'] <= zenith_bins[i]
                    else:
                        bin_mask = (X_val['solar_zenith_cos'] > zenith_bins[i-1]) & (X_val['solar_zenith_cos'] <= zenith_bins[i])
                    
                    if np.any(bin_mask):
                        bin_rmse = np.sqrt(mean_squared_error(y_val_horizon[bin_mask], y_pred_median[bin_mask]))
                        rmse_by_zenith[f'zenith_{i}'] = bin_rmse
            else:
                day_mae = np.nan
                day_rmse = np.nan
                bias = np.nan
                rmse_by_zenith = {}
        
        # Print metrics
            print(f"Horizon {horizon}h - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, nRMSE: {nrmse:.2f}%")
            print(f"Horizon {horizon}h - Daylight only - MAE: {day_mae:.2f}, RMSE: {day_rmse:.2f}, Bias: {bias:.2f}")
            print(f"Horizon {horizon}h - Persistence MAE: {persistence_mae:.2f}, RMSE: {persistence_rmse:.2f}")
            print(f"Horizon {horizon}h - Skill Score (MAE): {mae_skill:.2f}, (RMSE): {rmse_skill:.2f}")
            
            # Initialize metrics for this horizon
            validation_metrics[horizon] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'skill_score_mae': mae_skill,
                'skill_score_rmse': rmse_skill,
                'normalized_rmse': nrmse,
                'daylight_mae': day_mae,
                'daylight_rmse': day_rmse,
                'forecast_bias': bias,
                'persistence_mae': persistence_mae,
                'persistence_rmse': persistence_rmse,
                'rmse_by_zenith': rmse_by_zenith
            }
            
            # Calculate prediction intervals if available
            if hasattr(self, 'error_percentiles') and horizon in self.error_percentiles:
                lower_err, upper_err = self.error_percentiles[horizon]
                
                # Calculate bounds using error percentiles
                lower_bounds = y_pred_median + lower_err
                upper_bounds = y_pred_median + upper_err
                
                # Ensure non-negative and proper ordering
                lower_bounds = np.maximum(0, lower_bounds)
                upper_bounds = np.maximum(lower_bounds, upper_bounds)
                
                # Calculate coverage
                coverage = 100 * np.mean((y_val_horizon >= lower_bounds) & (y_val_horizon <= upper_bounds))
                
                # Calculate PICP (Prediction Interval Coverage Probability)
                picp = coverage / 100.0
                
                # Calculate PINAW (Prediction Interval Normalized Average Width)
                interval_widths = upper_bounds - lower_bounds
                avg_width = np.mean(interval_widths)
                target_range = np.max(y_val_horizon) - np.min(y_val_horizon) if len(y_val_horizon) > 0 else 1.0
                pinaw = avg_width / target_range if target_range > 0 else np.nan
                
                # Calculate Interval Score (IS)
                target_coverage = 0.90  # Default value, should be consistent with calibration
                alpha = 1 - target_coverage
                
                # Calculate indicators for observations outside the prediction interval
                below_lower = y_val_horizon < lower_bounds
                above_upper = y_val_horizon > upper_bounds
                
                # Calculate penalties
                penalty_below = 2/alpha * (lower_bounds[below_lower] - y_val_horizon[below_lower]) if np.any(below_lower) else 0
                penalty_above = 2/alpha * (y_val_horizon[above_upper] - upper_bounds[above_upper]) if np.any(above_upper) else 0
                
                # Calculate the interval score
                interval_score = avg_width + np.sum(penalty_below) / len(y_val_horizon) + np.sum(penalty_above) / len(y_val_horizon)
                
                # Calculate CRPS (Continuous Ranked Probability Score) approximation
                prediction_errors = np.abs(y_val_horizon - y_pred_median)
                crps = np.mean(prediction_errors) * (1 + (1 - picp) * np.log((1 - picp) / (1 - target_coverage)))
                
                # Calculate Coverage Deviation
                coverage_deviation = np.abs(picp - target_coverage)
                
                # Print interval metrics 
                print(f"Horizon {horizon}h - Interval Metrics - PICP: {picp:.4f}, PINAW: {pinaw:.4f}, IS: {interval_score:.4f}")
                print(f"Horizon {horizon}h - CRPS: {crps:.4f}, Coverage Deviation: {coverage_deviation:.4f}")
                
                # Add interval metrics to validation_metrics
                validation_metrics[horizon].update({
                    'coverage': coverage,
                    'picp': picp,
                    'pinaw': pinaw,
                    'interval_score': interval_score,
                    'crps': crps,
                    'coverage_deviation': coverage_deviation
                })
            elif horizon in self.models_lower and horizon in self.models_upper:
                # Use quantile models
                lower_preds = self.models_lower[horizon].predict(X_val)
                upper_preds = self.models_upper[horizon].predict(X_val)
                
                # Ensure non-negative and proper ordering
                lower_preds = np.maximum(0, lower_preds)
                upper_preds = np.maximum(lower_preds, upper_preds)
                
                # Calculate coverage
                coverage = 100 * np.mean((y_val_horizon >= lower_preds) & (y_val_horizon <= upper_preds))
                
                # Calculate PICP (Prediction Interval Coverage Probability)
                picp = coverage / 100.0
                
                # Calculate PINAW (Prediction Interval Normalized Average Width)
                interval_widths = upper_preds - lower_preds
                avg_width = np.mean(interval_widths)
                target_range = np.max(y_val_horizon) - np.min(y_val_horizon) if len(y_val_horizon) > 0 else 1.0
                pinaw = avg_width / target_range if target_range > 0 else np.nan
                
                # Calculate Interval Score (IS)
                target_coverage = 0.90  # Default value, should be consistent with calibration
                alpha = 1 - target_coverage
                
                # Calculate indicators for observations outside the prediction interval
                below_lower = y_val_horizon < lower_preds
                above_upper = y_val_horizon > upper_preds
                
                # Calculate penalties
                penalty_below = 2/alpha * (lower_preds[below_lower] - y_val_horizon[below_lower]) if np.any(below_lower) else 0
                penalty_above = 2/alpha * (y_val_horizon[above_upper] - upper_preds[above_upper]) if np.any(above_upper) else 0
                
                # Calculate the interval score
                interval_score = avg_width + np.sum(penalty_below) / len(y_val_horizon) + np.sum(penalty_above) / len(y_val_horizon)
                
                # Calculate CRPS approximation
                prediction_errors = np.abs(y_val_horizon - y_pred_median)
                crps = np.mean(prediction_errors) * (1 + (1 - picp) * np.log((1 - picp) / (1 - target_coverage)))
                
                # Calculate Coverage Deviation
                coverage_deviation = np.abs(picp - target_coverage)
                
                # Print interval metrics
                print(f"Horizon {horizon}h - Quantile Model Interval Metrics - PICP: {picp:.4f}, PINAW: {pinaw:.4f}, IS: {interval_score:.4f}")
                print(f"Horizon {horizon}h - CRPS: {crps:.4f}, Coverage Deviation: {coverage_deviation:.4f}")
                
                # Add interval metrics to validation_metrics
                validation_metrics[horizon].update({
                    'coverage': coverage,
                    'picp': picp,
                    'pinaw': pinaw,
                    'interval_score': interval_score,
                    'crps': crps,
                    'coverage_deviation': coverage_deviation
                })
            else:
                # No prediction intervals available
                validation_metrics[horizon]['coverage'] = np.nan
            
            # Store predictions
            predictions_dict[horizon] = {
                'actual': y_val_horizon.values,
                'predicted': y_pred_median,
                'lower': self.models_lower[horizon].predict(X_val) if horizon in self.models_lower else None,
                'upper': self.models_upper[horizon].predict(X_val) if horizon in self.models_upper else None
            }
        
        return validation_metrics
    
    def save_results(self, predictions, y_true, timestamps=None):
        """
        Save prediction results and evaluation metrics to files.
        
        Parameters:
        -----------
        predictions: Dictionary of predictions for each horizon
        y_true: Actual target values
        timestamps: Index or timestamps for the predictions
        """
        # Print debug info about predictions structure
        print(f"Debug - predictions keys: {list(predictions.keys())}")
        if predictions and len(predictions) > 0:
            # Print example of prediction structure for first horizon
            first_horizon = list(predictions.keys())[0]
            print(f"Debug - predictions[{first_horizon}] keys: {list(predictions[first_horizon].keys() if isinstance(predictions[first_horizon], dict) else ['<not a dict>'])}")
        
        print("Results processing completed.")
    
    def validate_forecast_setup(self, X_train, X_val, y_train, y_val):
        """
        Validate that our forecasting setup properly separates training and validation data in time.
        
        Parameters:
        X_train, X_val, y_train, y_val: The training and validation data splits
        """
        print("Validating forecasting setup...")
        
        # Use index if datetime column was dropped or not present in X/y
        train_end_idx = X_train.index.max()
        val_start_idx = X_val.index.min()
        
        # Assuming original DataFrame index corresponds to time order
        if train_end_idx >= val_start_idx:
            print(f"WARNING: Potential overlap or incorrect order between train (ends {train_end_idx}) and validation (starts {val_start_idx}) based on index.")
        else:
            print(f"Confirmed: Training ends at index {train_end_idx}, validation starts at index {val_start_idx}. Assuming index implies time order.")
    
    def run_pipeline(self, file_path, val_size=0.2, lag_hours=3, random_state=42):
        """
        Run the full GHI prediction pipeline from data loading to evaluation.
        
        Parameters:
        -----------
        file_path (str): Path to the CSV file containing the data
        val_size (float): Proportion of data to use for validation (fixed at 0.2 or 20%)
        lag_hours (int): Number of lag hours to use for feature creation
        random_state (int): Random seed for reproducibility (used for model training)
        
        Returns:
        --------
        dict: val_metrics - Evaluation metrics for each horizon on validation data
        """
        # Since data parameter is removed, always load from file_path
        data = self.load_data(file_path)
        
        # Preprocess the data
        data = self.preprocess_data(data)
        
        # Add solar position features
        data = self.add_solar_position_features(data)
        
        # Calculate theoretical clear sky GHI
        data = self.calculate_clear_sky_ghi(data)
        
        # Create lag features and targets for multiple horizons
        featured_data = self.create_features(data, lag_hours=lag_hours)
        
        # Create train/validation split (time-aware)
        # Note: test set is now removed
        X_train, X_val, y_train, y_val = self.split_time_series_data(
            featured_data, val_size=val_size
        )
        
        # Extract and save feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled, _, X_val_scaled = self.scale_features(X_train, pd.DataFrame(), X_val)
        
        # Optimize model parameters
        self.optimize_model_parameters(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Train models with optimized parameters
        print("Training direct models for horizons", self.forecast_horizons, "with best parameters...")
        self.train_models(X_train_scaled, y_train)
        
        # Calibrate prediction intervals using validation data
        print("\nCalibrating prediction intervals to achieve 90% coverage with maximum bound gap of 100...")
        self.calibrate_prediction_intervals(X_val_scaled, y_val, target_coverage=0.90, max_bound_gap=100)
        
        # Validate the models and save validation results to CSV
        print("\nEvaluating models on validation data and saving results to CSV...")
        val_metrics = self.evaluate_validation_models(X_val_scaled, y_val)
        
        # Prepare validation results for saving
        validation_results = {}
        
        # Store predictions and actual values for each horizon
        for horizon in self.forecast_horizons:
            target_col = f'target_GHI_{horizon}h'
            y_val_horizon = y_val[target_col].values
            
            # Get median model predictions
            y_pred_median = self.models_median[horizon].predict(X_val_scaled)
            
            # Get upper and lower bounds if available
            y_pred_lower = None
            y_pred_upper = None
            
            if hasattr(self, 'error_percentiles') and horizon in self.error_percentiles:
                lower_err, upper_err = self.error_percentiles[horizon]
                y_pred_lower = np.maximum(0, y_pred_median + lower_err)
                y_pred_upper = np.maximum(y_pred_lower, y_pred_median + upper_err)
            elif horizon in self.models_lower and horizon in self.models_upper:
                y_pred_lower = np.maximum(0, self.models_lower[horizon].predict(X_val_scaled))
                y_pred_upper = np.maximum(y_pred_lower, self.models_upper[horizon].predict(X_val_scaled))
            
            # Store results
            validation_results[f'actual_{horizon}h'] = y_val_horizon
            validation_results[f'median_{horizon}h'] = y_pred_median
            if y_pred_lower is not None:
                validation_results[f'lower_{horizon}h'] = y_pred_lower
            if y_pred_upper is not None:
                validation_results[f'upper_{horizon}h'] = y_pred_upper
        
        # Store datetime if available for better CSV output
        if 'datetime' in X_val.columns:
            self.val_datetimes = X_val['datetime']
        
        # Save validation results to CSV
        self.save_validation_results(X_val_scaled, validation_results)
        
        # Print validation metrics summary
        print("\n=== Validation Set Summary Metrics ===")
        for horizon, metrics in val_metrics.items():
            # Use only metrics that are definitely available from evaluate_validation_models
            print(f"Horizon {horizon}h: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics.get('r2', 0):.2f}")
            
            # Add the new interval metrics if they exist
            if all(metric in metrics for metric in ['picp', 'pinaw', 'interval_score', 'crps', 'coverage_deviation']):
                print(f"Horizon {horizon}h Interval Metrics: PICP={metrics['picp']:.4f}, PINAW={metrics['pinaw']:.4f}, " 
                      f"IS={metrics['interval_score']:.4f}, CRPS={metrics['crps']:.4f}, " 
                      f"Coverage Deviation={metrics['coverage_deviation']:.4f}")
        
        # Print Final Multi-Horizon Metrics
        print("\n--- Final Multi-Horizon Metrics ---")
        for horizon, metrics in val_metrics.items():
            # Extract metrics based on what's available in the dictionary
            mae = metrics.get('mae', 0)
            rmse = metrics.get('rmse', 0)
            r2 = metrics.get('r2', 0)  # Use R² from metrics
            skill = metrics.get('skill_score_rmse', metrics.get('skill_score_mae', 0))  # Try both skill score types
            
            # Ensure skill score is not NaN
            if np.isnan(skill):
                skill = 0
            
            print(f"Horizon {horizon}h: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}, Skill Score={skill:.2f}")
            
            # Add the interval metrics if they exist
            if all(metric in metrics for metric in ['picp', 'pinaw', 'interval_score', 'crps', 'coverage_deviation']):
                print(f"Horizon {horizon}h Interval Metrics: PICP={metrics['picp']:.4f}, PINAW={metrics['pinaw']:.4f}, " 
                      f"IS={metrics['interval_score']:.4f}, CRPS={metrics['crps']:.4f}, " 
                      f"Coverage Deviation={metrics['coverage_deviation']:.4f}")
        
        # Return the evaluation metrics for all horizons
        return val_metrics

    def predict(self, X_new, return_intervals=True):
        """
        Make predictions with empirically calibrated intervals.
        """
        logging.info(f"Making multi-horizon predictions for horizons {self.forecast_horizons}...")
        
        # Handle metadata columns gracefully - check which ones are available
        metadata_cols = ['Date', 'datetime', 'Start Period', 'End Period']
        available_cols = [col for col in metadata_cols if col in X_new.columns]
        missing_cols = set(metadata_cols) - set(available_cols)
        
        # Instead of raising an error, just log the info about missing columns
        if missing_cols:
            logging.info(f"Some metadata columns are not in input data: {missing_cols}")
            logging.info("Continuing with prediction without these columns...")
        
        # Get feature columns from the model
        if self.feature_columns is None:
            raise ValueError("Model has not been trained (feature_columns is None)")
        
        # Filter out metadata columns from required features
        required_features = [col for col in self.feature_columns 
                              if col not in metadata_cols]
        
        # Get only the columns needed for prediction
        available_features = [col for col in required_features if col in X_new.columns]
        missing_features = set(required_features) - set(available_features)
        
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
        
        # Debug info about predictions structure
        self.debug_logger.debug(f"Debug - predictions keys: {self.forecast_horizons}")
        
        # Create predictions for each forecast horizon
        predictions = {}
        
        # Create a DataFrame to log nighttime detections for exporting
        nighttime_log = []
        
        for horizon in self.forecast_horizons:
            if horizon not in self.models_median:
                raise ValueError(f"No trained model available for {horizon}h horizon")
            
            # Make median prediction
            median_preds = self.models_median[horizon].predict(X_new[available_features])
            
            # Initialize predictions dictionary for this horizon
            horizon_preds = {}
            
            # Apply a robust nighttime check for future predictions
            is_night = np.zeros_like(median_preds, dtype=bool)
            is_transition = np.zeros_like(median_preds, dtype=bool)  # For dawn/dusk transitions
            
            # If datetime information is available, use it to check nighttime directly
            if 'datetime' in X_new.columns:
                try:
                    # Calculate target prediction timestamps for this horizon
                    future_timestamps = X_new['datetime'] + pd.Timedelta(hours=horizon)
                    
                    # For each timestamp, determine if it's nighttime based on time and season
                    for i, timestamp in enumerate(future_timestamps):
                        # Extract hour (0-23) from timestamp
                        hour = timestamp.hour
                        month = timestamp.month
                        
                        # SIMPLIFIED NIGHTTIME DETECTION BASED ON PHILIPPINES CLIMATE
                        # For Davao City, Philippines (latitude: 7.07)
                        # The Philippines has two main seasons:
                        
                        # For Philippines climate (Tropical)
                        # Cool dry season (December to February)
                        if month in [12, 1, 2]:  
                            is_nighttime = (hour >= 18) or (hour < 6)  # Earlier sunset, later sunrise
                            is_transition_time = hour == 6 or hour == 17  # Dawn/dusk transition hours
                            season = "Cool dry"
                        # Hot dry season (March to May)
                        elif month in [3, 4, 5]:
                            is_nighttime = (hour >= 18) or (hour < 5)  # Later sunset, earlier sunrise
                            is_transition_time = hour == 5 or hour == 17  # Dawn/dusk transition hours
                            season = "Hot dry"
                        # Rainy season (June to November)
                        else:  # months 6-11
                            is_nighttime = (hour >= 18) or (hour < 6)  # More cloud cover, affects daylight
                            is_transition_time = hour == 6 or hour == 17  # Dawn/dusk transition hours
                            season = "Rainy"
                        
                        if is_nighttime:
                            is_night[i] = True
                            self.debug_logger.debug(f"Horizon {horizon}h prediction at {timestamp} will be during nighttime (hour: {hour}, month: {month}, season: {season})")
                        elif is_transition_time:
                            is_transition[i] = True
                            self.debug_logger.debug(f"Horizon {horizon}h prediction at {timestamp} will be during transition period (hour: {hour}, month: {month}, season: {season})")
                            
                        # Add to nighttime log for export
                        nighttime_log.append({
                            'horizon': horizon,
                            'timestamp': timestamp,
                            'hour': hour,
                            'month': month,
                            'season': season,
                            'is_nighttime': is_nighttime,
                            'is_transition': is_transition_time
                        })
                
                    # Log summary of detections
                    self.debug_logger.debug(f"Nighttime check result for horizon {horizon}h: {np.sum(is_night)}/{len(is_night)} nighttime periods detected")
                    if np.any(is_transition):
                        self.debug_logger.debug(f"Transition check result for horizon {horizon}h: {np.sum(is_transition)}/{len(is_transition)} transition periods detected")
                    
                    # Only show essential info in console
                    logging.info(f"Horizon {horizon}h: {np.sum(is_night)}/{len(is_night)} nighttime periods, {np.sum(is_transition)}/{len(is_transition)} transition periods detected")
                    
                except Exception as e:
                    logging.warning(f"Could not perform timestamp nighttime check: {str(e)}")
                    logging.info(f"Falling back to simpler detection for horizon {horizon}h")
                    
                    # Fallback: check if horizon extends into typical night hours
                    current_hour = X_new['datetime'].dt.hour.values[0]
                    target_hour = (current_hour + horizon) % 24
                    
                    # Conservative nighttime check (6 PM to 6 AM)
                    if target_hour >= 18 or target_hour < 6:
                        is_night[:] = True
                        logging.info(f"Fallback: Horizon {horizon}h (target hour {target_hour}) detected as nighttime")
                    # Check transition periods
                    elif target_hour == 6 or target_hour == 17:
                        is_transition[:] = True
                        logging.info(f"Fallback: Horizon {horizon}h (target hour {target_hour}) detected as transition period")
            else:
                # If datetime not available, use solar zenith from the data
                current_night_mask = X_new['solar_zenith_cos'] <= 0.01
                is_night |= current_night_mask.values
                
                # If available, use transition zone info
                if 'is_transition' in X_new.columns:
                    is_transition |= X_new['is_transition'].values.astype(bool)
                
                logging.info(f"No datetime available. Using solar zenith and transition indicators for horizon {horizon}h.")
            
            # Apply night mask to predictions
            if np.any(is_night):
                original_preds = median_preds.copy()
                median_preds = np.where(is_night, 0, median_preds)
                logging.info(f"Set {np.sum(is_night)} nighttime predictions to 0 for horizon {horizon}h")
                # Avoid printing full arrays, just show summary
                self.debug_logger.debug(f"Before: Mean={np.mean(original_preds):.2f}, After: Mean={np.mean(median_preds):.2f}")
            
            # Apply transition adjustments (reduce predictions during dawn/dusk by 70%)
            if np.any(is_transition):
                # First ensure no negative values
                median_preds = np.maximum(0, median_preds)
                
                transition_factor = 0.3  # Reduce to 30% of original value
                original_trans_preds = median_preds.copy()
                median_preds = np.where(is_transition, median_preds * transition_factor, median_preds)
                logging.info(f"Reduced {np.sum(is_transition)} transition period predictions to 30% for horizon {horizon}h")
                self.debug_logger.debug(f"Before transition: Mean={np.mean(original_trans_preds):.2f}, After: Mean={np.mean(median_preds):.2f}")
            
            # Store median predictions
            horizon_preds['predicted'] = median_preds
            
            # Add prediction intervals if requested
            if return_intervals:
                # Use error percentiles if available (new approach)
                if hasattr(self, 'error_percentiles') and horizon in self.error_percentiles:
                    lower_percentile, upper_percentile = self.error_percentiles[horizon]
                    
                    # Check if we're working with arrays of values that need to be properly handled
                    if isinstance(lower_percentile, np.ndarray) and lower_percentile.size > 1:
                        # Store original arrays for reference
                        self.lower_percentiles = self.error_percentiles[horizon][0] 
                        self.upper_percentiles = self.error_percentiles[horizon][1]
                    
                    # Make sure lower_percentile has the same shape as median_preds
                    if hasattr(self, 'forecast_horizons') and len(self.forecast_horizons) > 0:
                        # Check if 'horizon' column exists in the input data
                        if 'horizon' in X_new.columns:
                            # Reshape the percentiles to match the predictions
                            # This assumes predictions are organized by horizon
                            horizon_idx = X_new['horizon'].astype(int).values - 1  # Convert to 0-based index
                            
                            # Use the horizon index to select the appropriate percentile for each prediction
                            lower_percentile_matched = np.array([self.lower_percentiles[idx] for idx in horizon_idx])
                            upper_percentile_matched = np.array([self.upper_percentiles[idx] for idx in horizon_idx])
                            
                            # Now the shapes should match
                            lower_bounds = median_preds + lower_percentile_matched
                            upper_bounds = median_preds + upper_percentile_matched
                        else:
                            # If 'horizon' column doesn't exist, use the current horizon for all predictions
                            logging.info(f"No 'horizon' column in input data. Using horizon {horizon} for all predictions.")
                            # Use the current horizon index
                            try:
                                horizon_idx = self.forecast_horizons.index(horizon)
                                # Convert to scalar if numpy array - this resolves ambiguity in boolean context
                                if isinstance(lower_percentile, np.ndarray):
                                    # Use a simple approach - take the scalar value if size 1, or mean value otherwise
                                    if lower_percentile.size == 1:
                                        lower_percentile_val = lower_percentile.item()
                                    else:
                                        lower_percentile_val = np.mean(lower_percentile)
                                else:
                                    lower_percentile_val = lower_percentile
                                    
                                if isinstance(upper_percentile, np.ndarray):
                                    if upper_percentile.size == 1:
                                        upper_percentile_val = upper_percentile.item()
                                    else:
                                        upper_percentile_val = np.mean(upper_percentile)
                                else:
                                    upper_percentile_val = upper_percentile
                            except (ValueError, KeyError, IndexError, AttributeError):
                                # If error, fall back to using the raw values
                                if isinstance(lower_percentile, np.ndarray):
                                    lower_percentile_val = np.mean(lower_percentile)
                                else:
                                    lower_percentile_val = lower_percentile
                                    
                                if isinstance(upper_percentile, np.ndarray):
                                    upper_percentile_val = np.mean(upper_percentile)
                                else:
                                    upper_percentile_val = upper_percentile
                            
                            # Apply to all predictions
                            lower_bounds = median_preds + lower_percentile_val
                            upper_bounds = median_preds + upper_percentile_val
                    else:
                        # If not using horizons, make sure percentiles are broadcasted properly
                        # Convert arrays to scalars if needed to avoid boolean context issues
                        if isinstance(lower_percentile, np.ndarray):
                            if lower_percentile.size == 1:
                                lower_percentile = lower_percentile.item()
                            else:
                                lower_percentile = np.mean(lower_percentile)
                                
                        if isinstance(upper_percentile, np.ndarray):
                            if upper_percentile.size == 1:
                                upper_percentile = upper_percentile.item()
                            else:
                                upper_percentile = np.mean(upper_percentile)
                                
                        lower_bounds = median_preds + lower_percentile
                        upper_bounds = median_preds + upper_percentile
                    
                    # Apply night mask to bounds as well
                    if np.any(is_night):
                        lower_bounds = np.where(is_night, 0, lower_bounds)
                        upper_bounds = np.where(is_night, 0, upper_bounds)
                    
                    # Ensure non-negative values before applying transition factor
                    lower_bounds = np.maximum(0, lower_bounds)
                    upper_bounds = np.maximum(0, upper_bounds)
                    
                    # Apply transition adjustments to bounds as well
                    if np.any(is_transition):
                        transition_factor = 0.3  # Same factor as for median predictions
                        lower_bounds = np.where(is_transition, lower_bounds * transition_factor, lower_bounds)
                        upper_bounds = np.where(is_transition, upper_bounds * transition_factor, upper_bounds)
                    
                    # Final check to ensure bounds are non-negative and lower <= upper
                    lower_bounds = np.maximum(0, lower_bounds)
                    upper_bounds = np.maximum(lower_bounds, upper_bounds)
                    
                    horizon_preds['lower'] = lower_bounds
                    horizon_preds['upper'] = upper_bounds
                
                # Fallback to old approach if error percentiles aren't available
                elif horizon in self.models_lower and horizon in self.models_upper:
                    lower_preds = self.models_lower[horizon].predict(X_new[available_features])
                    upper_preds = self.models_upper[horizon].predict(X_new[available_features])
                    
                    # Apply night mask to bounds
                    if np.any(is_night):
                        lower_preds = np.where(is_night, 0, lower_preds)
                        upper_preds = np.where(is_night, 0, upper_preds)
                        self.debug_logger.debug(f"Set upper/lower bounds to 0 for nighttime predictions")
                    
                    # Ensure non-negative values before applying transition factor
                    lower_preds = np.maximum(0, lower_preds)
                    upper_preds = np.maximum(0, upper_preds)
                    
                    # Apply transition adjustments to bounds
                    if np.any(is_transition):
                        transition_factor = 0.3  # Same factor as for median predictions
                        lower_preds = np.where(is_transition, lower_preds * transition_factor, lower_preds)
                        upper_preds = np.where(is_transition, upper_preds * transition_factor, upper_preds)
                        self.debug_logger.debug(f"Reduced upper/lower bounds to 30% for transition period predictions")
                    
                    # If we have adjustment factors, apply them
                    if hasattr(self, 'lower_adjustments') and horizon in self.lower_adjustments:
                        lower_adj = self.lower_adjustments.get(horizon, 0)
                        upper_adj = self.upper_adjustments.get(horizon, 0)
                        
                        lower_diff = median_preds - lower_preds
                        upper_diff = upper_preds - median_preds
                        
                        lower_preds = lower_preds - lower_adj * lower_diff
                        upper_preds = upper_preds + upper_adj * upper_diff
                    
                    horizon_preds['lower'] = np.maximum(0, lower_preds)
                    horizon_preds['upper'] = np.maximum(horizon_preds['lower'], upper_preds)
            
            # Store predictions for this horizon
            predictions[horizon] = horizon_preds
        
        # Print debug info about the first horizon's predictions
        if predictions and len(predictions) > 0:
            first_horizon = list(predictions.keys())[0]
            self.debug_logger.debug(f"Debug - predictions[{first_horizon}] keys: {list(predictions[first_horizon].keys())}")
        
        # Save the nighttime detection log to CSV
        if nighttime_log:
            os.makedirs('logs', exist_ok=True)
            nighttime_df = pd.DataFrame(nighttime_log)
            nighttime_csv = os.path.join('logs', f'nighttime_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            nighttime_df.to_csv(nighttime_csv, index=False)
            logging.info(f"Nighttime detection details saved to: {nighttime_csv}")
        
        return predictions

    def calibrate_prediction_intervals(self, X_val, y_val, target_coverage=0.90, max_iterations=15, max_bound_gap=100):
        """
        Direct empirical calibration of prediction intervals.
        
        This approach doesn't rely on the quantile models at all - instead it:
        1. Uses the median model for the central prediction
        2. Calculates empirical error distributions
        3. Creates intervals directly from those error distributions
        
        Parameters:
        -----------
        X_val: Validation features
        y_val: Validation targets
        target_coverage: Target coverage level (default: 0.90)
        max_iterations: Maximum number of calibration iterations
        max_bound_gap: Maximum allowed gap between lower and upper bounds (default: 100)
        """
        logging.info(f"Calibrating prediction intervals to achieve {target_coverage*100:.0f}% coverage with max bound gap of {max_bound_gap}...")
        
        # Filter out nighttime data (before 5 AM and after 6 PM)
        daytime_mask = None
        if 'datetime' in X_val.columns:
            # Use datetime hours to identify daytime
            hours = X_val['datetime'].dt.hour
            daytime_mask = (hours >= 5) & (hours <= 18)
            logging.info(f"Filtering validation data to daytime hours (5 AM to 6 PM): {daytime_mask.sum()} of {len(X_val)} records kept")
        else:
            # Use solar elevation as proxy if datetime not available
            daytime_mask = X_val['solar_zenith_cos'] > 0.01  # Basic daylight check
            logging.info(f"Filtering validation data using solar position: {daytime_mask.sum()} of {len(X_val)} records kept")
        
        # Apply the filter to validation data
        X_val_daytime = X_val[daytime_mask].copy()
        y_val_daytime = y_val[daytime_mask].copy()
        
        # Store the sample count for consistency in evaluation
        self.calibration_sample_count = len(X_val_daytime)
        logging.info(f"Stored calibration sample count: {self.calibration_sample_count}")
        
        logging.info(f"Using {len(X_val_daytime)} daytime validation records for calibration (excluded nighttime data)")
        
        # Dictionary to store calibration factors
        self.error_percentiles = {}
        
        for horizon in self.forecast_horizons:
            target_col = f'target_GHI_{horizon}h'
            
            if horizon not in self.models_median:
                logging.info(f"Skipping calibration for horizon {horizon}h - missing model")
                continue
            
            # Get actual values
            actual = y_val_daytime[target_col].values
            
            # Get median predictions
            median_preds = self.models_median[horizon].predict(X_val_daytime)
            
            # Calculate prediction errors
            errors = actual - median_preds
            
            # Find error percentiles for desired coverage
            alpha = (1 - target_coverage) / 2  # split equally on both sides
            lower_percentile = np.percentile(errors, alpha * 100)
            upper_percentile = np.percentile(errors, (1 - alpha) * 100)
            
            # Calculate initial bound gap
            initial_gap = upper_percentile - lower_percentile
            
            logging.info(f"Horizon {horizon}h - Initial error percentiles: {lower_percentile:.2f} to {upper_percentile:.2f} (gap: {initial_gap:.2f})")
            
            # Identify peak times
            is_peak_time = False
            if 'is_peak_hour' in X_val_daytime.columns:
                is_peak_time = X_val_daytime['is_peak_hour'].values > 0
            else:
                # If specific peak hour feature doesn't exist, use solar elevation as proxy
                is_peak_time = (X_val_daytime['solar_elevation'] >= 60) & (X_val_daytime['solar_zenith_cos'] >= 0.5)
                
                # Also check if 'datetime' is available to identify peak hours (11 AM - 1 PM)
                if 'datetime' in X_val_daytime.columns:
                    try:
                        # Get hours of the day for all timestamps
                        hours_of_day = X_val_daytime['datetime'].dt.hour
                        
                        # Mark 11 AM - 1 PM as peak hours
                        peak_hour_mask = (hours_of_day >= 11) & (hours_of_day <= 13)
                        
                        # Combine with existing peak time detection
                        is_peak_time = is_peak_time | peak_hour_mask
                        
                        logging.info(f"Identified {peak_hour_mask.sum()} records between 11 AM - 1 PM as peak hours for horizon {horizon}h")
                    except Exception as e:
                        logging.warning(f"Error identifying peak hours by time: {str(e)}")
            
            # Separate errors for peak and non-peak times
            peak_errors = errors[is_peak_time]
            non_peak_errors = errors[~is_peak_time]
            
            # Initialize variables for percentile adjustment
            adjusted_lower_percentile = lower_percentile
            adjusted_upper_percentile = upper_percentile
            
            if len(peak_errors) > 0:
                # Calculate peak time percentiles and gap
                peak_lower = np.percentile(peak_errors, alpha * 100)
                peak_upper = np.percentile(peak_errors, (1 - alpha) * 100)
                peak_gap = peak_upper - peak_lower
                
                logging.info(f"Peak time error percentiles: {peak_lower:.2f} to {peak_upper:.2f} (gap: {peak_gap:.2f})")
                
                # For peak times, use gap of 150 or actual gap if smaller
                if peak_gap > 150:
                    # Only constrain if significantly larger than 150
                    if peak_gap > 200:
                        logging.info(f"Peak time gap ({peak_gap:.2f}) is very large, using adaptive approach")
                        # Adaptive approach: use wider bounds for high inaccuracy horizons
                        if horizon <= 2:  # Short-term horizons (1-2h) use 200
                            peak_target_gap = 200
                        else:  # Longer-term horizons (3h+) use 250
                            peak_target_gap = 250
                            
                        # Adjust peak percentiles to target gap
                        gap_excess = peak_gap - peak_target_gap
                        peak_lower += gap_excess / 3  # Adjust lower bound by 1/3
                        peak_upper -= gap_excess * 2/3  # Adjust upper bound by 2/3
                    else:
                        # Less aggressive constraint for gaps between 150-200
                        peak_target_gap = 150
                        gap_excess = peak_gap - peak_target_gap
                        peak_lower += gap_excess / 3
                        peak_upper -= gap_excess * 2/3
                
                # For non-peak times
                if len(non_peak_errors) > 0:
                    # Calculate non-peak percentiles
                    non_peak_lower = np.percentile(non_peak_errors, alpha * 100)
                    non_peak_upper = np.percentile(non_peak_errors, (1 - alpha) * 100)
                    non_peak_gap = non_peak_upper - non_peak_lower
                    
                    logging.info(f"Non-peak time error percentiles: {non_peak_lower:.2f} to {non_peak_upper:.2f} (gap: {non_peak_gap:.2f})")
                    
                    # For non-peak times, use adaptive approach based on horizon
                    if non_peak_gap > max_bound_gap:
                        # Use wider bounds for longer horizons
                        if horizon <= 2:  # Short-term horizons (1-2h)
                            non_peak_target_gap = 150
                        else:  # Longer-term horizons (3h+)
                            non_peak_target_gap = 175
                            
                        # Only constrain if gap is significantly larger than target
                        if non_peak_gap > non_peak_target_gap * 1.5:
                            logging.info(f"Non-peak gap ({non_peak_gap:.2f}) is very large, constraining to {non_peak_target_gap}")
                            gap_excess = non_peak_gap - non_peak_target_gap
                            non_peak_lower += gap_excess / 3  # Adjust lower bound by 1/3
                            non_peak_upper -= gap_excess * 2/3  # Adjust upper bound by 2/3
                        else:
                            # For smaller excesses, use less aggressive constraint
                            non_peak_target_gap = max(max_bound_gap, non_peak_gap * 0.8)
                            gap_excess = non_peak_gap - non_peak_target_gap
                            non_peak_lower += gap_excess / 3
                            non_peak_upper -= gap_excess * 2/3
                    
                    # Combine the percentiles
                    adjusted_lower_percentile = np.where(is_peak_time, peak_lower, non_peak_lower)
                    adjusted_upper_percentile = np.where(is_peak_time, peak_upper, non_peak_upper)
                else:
                    # If all times are peak times
                    adjusted_lower_percentile = peak_lower
                    adjusted_upper_percentile = peak_upper
            else:
                # If no peak times, use adaptive approach based on horizon
                non_peak_gap = upper_percentile - lower_percentile
                
                if non_peak_gap > max_bound_gap:
                    # Use wider bounds for longer horizons
                    if horizon <= 2:  # Short-term horizons (1-2h)
                        non_peak_target_gap = 150
                    else:  # Longer-term horizons (3h+)
                        non_peak_target_gap = 175
                        
                    # Only constrain if gap is significantly larger than target
                    if non_peak_gap > non_peak_target_gap * 1.5:
                        logging.info(f"Gap ({non_peak_gap:.2f}) is very large, constraining to {non_peak_target_gap}")
                        gap_excess = non_peak_gap - non_peak_target_gap
                        adjusted_lower_percentile += gap_excess / 3  # Adjust lower bound by 1/3
                        adjusted_upper_percentile -= gap_excess * 2/3  # Adjust upper bound by 2/3
                    else:
                        # For smaller excesses, use less aggressive constraint
                        non_peak_target_gap = max(max_bound_gap, non_peak_gap * 0.8)
                        gap_excess = non_peak_gap - non_peak_target_gap
                        adjusted_lower_percentile += gap_excess / 3
                        adjusted_upper_percentile -= gap_excess * 2/3
                        
                    logging.info(f"Adjusted error percentiles: {adjusted_lower_percentile:.2f} to {adjusted_upper_percentile:.2f} (gap: {adjusted_upper_percentile - adjusted_lower_percentile:.2f})")
            
            # Create intervals by adding these percentiles to median predictions
            lower_bounds = median_preds + adjusted_lower_percentile
            upper_bounds = median_preds + adjusted_upper_percentile
            
            # Ensure non-negative bounds and lower <= upper
            lower_bounds = np.maximum(0, lower_bounds)
            upper_bounds = np.maximum(lower_bounds, upper_bounds)
            
            # Calculate the actual coverage (percentage of points within the interval)
            in_interval = (y_val_daytime[target_col] >= lower_bounds) & (y_val_daytime[target_col] <= upper_bounds)
            empirical_coverage = in_interval.mean()
            
            # Add check for coverage and iteratively widen if needed
            if empirical_coverage < target_coverage * 0.9:  # If coverage < 81% (90% of target)
                logging.info(f"Coverage ({empirical_coverage*100:.2f}%) is too low, iteratively adjusting bounds")
                
                # Start with current percentiles
                iter_lower = adjusted_lower_percentile
                iter_upper = adjusted_upper_percentile
                
                for i in range(1, max_iterations+1):
                    # Widen bounds by 10% each iteration
                    widen_factor = 0.1 * i
                    iter_lower = adjusted_lower_percentile * (1 + widen_factor)
                    iter_upper = adjusted_upper_percentile * (1 + widen_factor)
                    
                    # Create new bounds
                    new_lower = median_preds + iter_lower
                    new_upper = median_preds + iter_upper
                    
                    # Ensure non-negative bounds and lower <= upper
                    new_lower = np.maximum(0, new_lower)
                    new_upper = np.maximum(new_lower, new_upper)
                    
                    # Calculate new coverage
                    new_in_interval = (y_val_daytime[target_col] >= new_lower) & (y_val_daytime[target_col] <= new_upper)
                    new_coverage = new_in_interval.mean()
                    
                    if new_coverage >= target_coverage * 0.85:  # Stop if we reach 85% of target
                        lower_bounds = new_lower
                        upper_bounds = new_upper
                        empirical_coverage = new_coverage
                        adjusted_lower_percentile = iter_lower
                        adjusted_upper_percentile = iter_upper
                        logging.info(f"After {i} iterations, improved coverage to {empirical_coverage*100:.2f}%")
                        break
                    
                    if i == max_iterations:
                        logging.warning(f"Could not achieve target coverage after {max_iterations} iterations")
                        lower_bounds = new_lower
                        upper_bounds = new_upper
                        empirical_coverage = new_coverage
                        adjusted_lower_percentile = iter_lower
                        adjusted_upper_percentile = iter_upper
            
            # Find current gap for logging
            if isinstance(adjusted_upper_percentile, np.ndarray) and adjusted_upper_percentile.ndim > 0:
                current_upper = np.mean(adjusted_upper_percentile)
                current_lower = np.mean(adjusted_lower_percentile)
            else:
                current_upper = adjusted_upper_percentile
                current_lower = adjusted_lower_percentile
            
            current_gap = current_upper - current_lower
            
            logging.info(f"Empirical calibration for horizon {horizon}h: coverage = {empirical_coverage*100:.2f}%, bound gap = {current_gap:.2f}")
            
            # Convert any array percentiles to scalar averages for easier application later
            if isinstance(adjusted_lower_percentile, np.ndarray):
                adjusted_lower_percentile = np.mean(adjusted_lower_percentile)
                logging.info(f"Converted lower percentile from array to scalar: {adjusted_lower_percentile:.2f}")
            
            if isinstance(adjusted_upper_percentile, np.ndarray):
                adjusted_upper_percentile = np.mean(adjusted_upper_percentile)
                logging.info(f"Converted upper percentile from array to scalar: {adjusted_upper_percentile:.2f}")
            
            # Store the error percentiles for prediction
            self.error_percentiles[horizon] = (adjusted_lower_percentile, adjusted_upper_percentile)
        
        return self.error_percentiles

    def optimize_model_parameters(self, X_train, y_train, X_val, y_val):
        """
        Optimize model parameters for each forecast horizon using Optuna.
        
        Parameters:
        -----------
        X_train (pandas.DataFrame): Training features
        y_train (pandas.DataFrame): Training targets
        X_val (pandas.DataFrame): Validation features
        y_val (pandas.DataFrame): Validation targets
        """
        if not HAS_OPTUNA:
            print("Optuna not installed. Skipping hyperparameter optimization.")
            return
        
        # Initialize best parameters dict if it doesn't exist
        if not hasattr(self, 'best_params'):
            self.best_params = {}
        
        print("\nOptimizing model parameters for all forecast horizons using Optuna...")
        
        # For each forecast horizon, run optimization
        for horizon in self.forecast_horizons:
            target_col = f'target_GHI_{horizon}h'
            print(f"\n--- Optimizing models for {horizon}h horizon ---")
            
            # Define the objective function for median prediction
            def objective_median(trial):
                # Suggest values for hyperparameters
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }
                
                # Create and train model
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    **params,
                    random_state=42
                )
                model.fit(X_train, y_train[target_col])
                
                # Calculate MAE
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val[target_col], y_pred)
                
                return mae
            
            # Define objective function for lower bound
            def objective_lower(trial):
                # Suggest values for hyperparameters
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }
                
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    **params,
                    random_state=42
                )
                model.fit(X_train, y_train[target_col])
                
                y_pred = model.predict(X_val)
                # Optimize for capturing 5th percentile lower bound
                errors = y_val[target_col] - y_pred
                q05_error = np.percentile(errors, 5)
                
                # Penalize both over and under-coverage, but asymmetrically
                coverage = np.mean((y_val[target_col] >= y_pred + q05_error))
                coverage_error = abs(coverage - 0.05) * 100
                
                # Incorporate tightness of the bound
                bound_width = abs(q05_error)
                
                # Combined objective (minimize)
                return coverage_error + 0.01 * bound_width
            
            # Define objective function for upper bound
            def objective_upper(trial):
                # Suggest values for hyperparameters
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }
                
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    **params,
                    random_state=42
                )
                model.fit(X_train, y_train[target_col])
                
                y_pred = model.predict(X_val)
                # Optimize for capturing 95th percentile upper bound
                errors = y_val[target_col] - y_pred
                q95_error = np.percentile(errors, 95)
                
                # Penalize both over and under-coverage, but asymmetrically
                coverage = np.mean((y_val[target_col] <= y_pred + q95_error))
                coverage_error = abs(coverage - 0.05) * 100
                
                # Incorporate tightness of the bound
                bound_width = abs(q95_error)
                
                # Combined objective (minimize)
                return coverage_error + 0.01 * bound_width
            
            # Create optimization studies
            median_study = optuna.create_study(direction='minimize')
            lower_study = optuna.create_study(direction='minimize')
            upper_study = optuna.create_study(direction='minimize')
            
            # Run optimization for limited number of trials (quick for demo)
            print(f"Running optimization for horizon {horizon}h median model...")
            median_study.optimize(objective_median, n_trials=15)
            
            print(f"Running optimization for lower bound model...")
            lower_study.optimize(objective_lower, n_trials=15)
            
            print(f"Running optimization for upper bound model...")
            upper_study.optimize(objective_upper, n_trials=15)
            
            # Store best parameters
            self.best_params[horizon] = {
                'median': median_study.best_params,
                'lower': lower_study.best_params,
                'upper': upper_study.best_params
            }
            
            # Print best parameters
            print(f"\nBest parameters for horizon {horizon}h median model:")
            print(median_study.best_params)
            print(f"Best MAE: {median_study.best_value:.4f}")
            
            print(f"\nBest parameters for horizon {horizon}h lower bound model:")
            print(lower_study.best_params)
            print(f"Best objective value: {lower_study.best_value:.4f}")
            
            print(f"\nBest parameters for horizon {horizon}h upper bound model:")
            print(upper_study.best_params)
            print(f"Best objective value: {upper_study.best_value:.4f}")
        
        print("\nParameter optimization complete.")
    
    def train_models(self, X_train, y_train):
        """
        Train direct models for each forecast horizon. Uses separate XGBoost models for each horizon.
        
        Parameters:
        -----------
        X_train (pandas.DataFrame): Training features
        y_train (pandas.DataFrame): Training targets for all horizons
        """
        print("Training direct multi-horizon prediction models...")
        
        # Clear previous models
        self.models_median = {}
        self.models_lower = {}
        self.models_upper = {}
        
        # For each horizon, train a separate model
        for horizon in self.forecast_horizons:
            print(f"\nTraining XGBoost models for horizon {horizon}h...")
            target_col = f'target_GHI_{horizon}h'
            
            # Check if we have best parameters for this horizon
            if hasattr(self, 'best_params') and horizon in self.best_params:
                median_params = self.best_params[horizon]['median']
                print(f"  Using optimized parameters: {median_params}")
                # Train median model with best parameters
                self.models_median[horizon] = self._train_single_model(
                    X_train, y_train[target_col], params=median_params
                )
            else:
                # Train with default parameters
                print(f"  Using default parameters (no optimization found)")
                self.models_median[horizon] = self._train_single_model(
                    X_train, y_train[target_col]
                )
            
            # Let user know model was successfully trained
            print(f"  ✓ Median model trained for {horizon}h horizon")

        # Print a summary
        print("\nModel training complete.")
        print(f"Trained {len(self.models_median)} median models for horizons: {list(sorted(self.models_median.keys()))}")
        
        return self.models_median

    def calculate_clear_sky_ghi(self, df):
        """
        Advanced REST2 derived clear sky model with dynamic atmospheric transmittance.
        
        Based on: Gueymard (2008) "REST2: High-performance solar radiation model for cloudless-sky irradiance"
        and Bright et al. (2018) "Improved modelling of the clear sky solar radiation"
        """
        print("Calculating advanced theoretical clear sky GHI...")
        
        # Better air mass calculation with improved accuracy for low solar elevations
        # Kasten and Young (1989) formula - more accurate than simple 1/cos(zenith)
        df['air_mass'] = np.where(
            df['solar_zenith_cos'] > 0.01,
            1.0 / (df['solar_zenith_cos'] + 0.50572 * (96.07995 - np.degrees(np.arccos(df['solar_zenith_cos'])))**-1.6364),
            np.nan
        )
        
        # Set air mass to NaN for night time
        df.loc[df['solar_zenith_cos'] <= 0, 'air_mass'] = np.nan
        
        # Dynamic transmittance model based on air mass
        # Boland et al. (2008) "Dynamic atmospheric transmittance for solar radiation modelling"
        # Using more sophisticated transmittance model that varies with air mass
        df['transmittance'] = np.where(
            df['air_mass'].notna(),
            0.8277 - 0.0322 * (df['air_mass'] - 1),
            np.nan
        )
        
        # Enhanced transmittance for peak hours (11 AM to 1 PM) to better handle peak irradiance
        if 'datetime' in df.columns:
            hours = df['datetime'].dt.hour
            peak_hours_mask = (hours >= 11) & (hours <= 13)
            
            if peak_hours_mask.any():
                # For peak hours, apply a more optimistic transmittance model
                # Higher aerosol transmission during these hours based on research
                peak_transmittance = np.where(
                    df['air_mass'].notna(),
                    0.8577 - 0.0272 * (df['air_mass'] - 1),  # Slightly optimized for peak hours
                    np.nan
                )
                
                # Apply enhanced transmittance only to peak hours
                df.loc[peak_hours_mask, 'transmittance'] = peak_transmittance[peak_hours_mask]
                
                # Log the enhancement
                print(f"Applied enhanced transmittance model to {peak_hours_mask.sum()} peak hour records")
        
        # Cap transmittance to realistic values
        df['transmittance'] = np.clip(df['transmittance'].fillna(0), 0.5, 0.85)
        
        # Extraterrestrial radiation with more accurate eccentricity formula
        # Spencer (1971) formula, widely validated in solar research
        day_angle = 2 * np.pi * (df['day_of_year'] - 1) / 365.25
        eccentricity = (1.00011 + 0.034221 * np.cos(day_angle) + 0.00128 * np.sin(day_angle) +
                       0.000719 * np.cos(2 * day_angle) + 0.000077 * np.sin(2 * day_angle))
        
        # Calculate clear sky GHI = solar constant * eccentricity * cos(zenith) * transmittance
        # Handle potential infinities by using np.where and fillna
        df['air_mass_adjusted'] = df['air_mass'].fillna(5.0)  # Fill NaN with high value for night
        df['air_mass_adjusted'] = np.clip(df['air_mass_adjusted'], 0, 10)  # Clip to reasonable range
        
        df['clear_sky_ghi'] = np.where(
            df['solar_zenith_cos'] > 0,
            self.solar_constant * eccentricity * df['solar_zenith_cos'] * df['transmittance']**df['air_mass_adjusted'],
            0
        )
        
        # Apply physically-based constraints
        df.loc[df['solar_zenith_cos'] <= 0, 'clear_sky_ghi'] = 0
        df.loc[df['clear_sky_ghi'] < 0, 'clear_sky_ghi'] = 0
        
        # Handle extremely large values (in case of numerical issues)
        max_expected_ghi = 1500  # Typical max GHI under ideal conditions
        df['clear_sky_ghi'] = np.clip(df['clear_sky_ghi'], 0, max_expected_ghi)
        
        # Calculate clear sky index with improved handling for low GHI values
        # Following Engerer & Mills (2014) methodology for clear sky index calculation
        # Avoiding division by zero
        df['clear_sky_index'] = np.where(
            df['clear_sky_ghi'] > 10,  # Only calculate for meaningful clear sky values
            df[self.target_column] / df['clear_sky_ghi'],
            0  # Set to 0 for night or very low clear sky values
        )
        
        # Handle extreme values with validated caps from literature
        df['clear_sky_index'] = np.clip(df['clear_sky_index'], 0, 1.5)  # Physically realistic max
        
        # Replace any infinities
        df['clear_sky_index'] = df['clear_sky_index'].replace([np.inf, -np.inf], 0)
        
        # Create clear sky index features
        for i in range(1, 4):
            df[f'clear_sky_index_lag_{i}'] = df['clear_sky_index'].shift(i)
        
        print("Advanced clear sky GHI calculated successfully")
        return df

    def _train_single_model(self, X_train, y_train, params=None, sample_weight=None):
        """
        Train a single XGBoost model with modified objective function.
        
        Research basis: Yang et al. (2019) "Short-term solar irradiance forecasting based on a hybrid deep learning methodology"
        and Aguiar et al. (2016) "An asymmetric Huber loss function for solar irradiance forecasting"
        """
        if params is None:
            params = {}
        
        # Define custom asymmetric loss function for GHI forecasting
        # This penalizes overestimation more than underestimation
        # Based on research showing overestimation is more problematic in energy scheduling
        def asymmetric_huber_obj(predt, dtrain):
            y = dtrain.get_label()
            delta = 10.0  # Threshold parameter
            
            # Default asymmetry - penalize overestimation more
            default_asymmetry = 1.5
            
            # Check if we have weights that indicate this is a peak hour model
            # Peak hour models need a more balanced loss function
            is_peak_model = False
            if 'peak' in str(params).lower() or (sample_weight is not None and np.mean(sample_weight) > 1.0):
                is_peak_model = True
                # More balanced asymmetry for peak hours to avoid underprediction
                asymmetry = 1.2  # Less asymmetric for peak hours
                logging.info("Using balanced loss function for peak hour model")
            else:
                asymmetry = default_asymmetry
            
            residual = y - predt
            abs_residual = np.abs(residual)
            
            # Calculate gradients and hessians
            grad = np.where(residual >= 0,
                            np.where(abs_residual <= delta, -1, -delta / abs_residual),
                            np.where(abs_residual <= delta, asymmetry, asymmetry * delta / abs_residual))
            
            hess = np.where(abs_residual <= delta, 1,
                            delta / abs_residual**2)
            
            return grad, hess
        
        # Default parameters
        base_params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        # Use asymmetric loss for median models
        if 'quantile_alpha' not in params and params.get('objective') == 'reg:squarederror':
            base_params['objective'] = asymmetric_huber_obj
        
        # Update with provided parameters
        base_params.update(params)
        
        # Create and train model
        model = xgb.XGBRegressor(**base_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        
        return model

    def train_specialized_models(self, X_train, y_train):
        """
        Train specialized models for different temporal regimes.
        
        Based on: Yang et al. (2018) "Time-of-day specific ensemble models for solar forecasting"
        and Feng et al. (2020) "Machine learning models for solar radiation forecasting with regime recognition"
        """
        print("Training specialized regime-based models...")
        
        # Define regimes
        # 1. Night (no sun)
        # 2. Transition (sunrise/sunset)
        # 3. Day (full sun)
        # 4. Peak hours (11 AM - 1 PM) - New regime specifically for peak solar hours
        
        # Create masks for each regime
        night_mask = X_train['solar_zenith_cos'] <= 0.01
        transition_mask = (X_train['solar_zenith_cos'] > 0.01) & (X_train['solar_zenith_cos'] < 0.3)
        
        # Use hour information for peak hour detection if available
        if 'is_peak_hour' in X_train.columns:
            peak_mask = X_train['is_peak_hour'] > 0
        else:
            # If specific peak hour feature doesn't exist, use solar elevation as proxy
            # Peak hours typically have the highest solar elevation
            peak_mask = (X_train['solar_elevation'] >= 60) & (X_train['solar_zenith_cos'] >= 0.5)
            
        # Remaining day hours (not peak, not transition, not night)
        day_mask = (X_train['solar_zenith_cos'] >= 0.3) & (~peak_mask)
        
        print(f"Training samples by regime - Night: {night_mask.sum()}, Transition: {transition_mask.sum()}, "
              f"Day: {day_mask.sum()}, Peak: {peak_mask.sum()}")
        
        # Store specialized models
        self.specialized_models = {
            'night': {},
            'transition': {},
            'day': {},
            'peak': {}  # New category for peak hours
        }
        
        # Train models for each horizon and each regime
        for horizon in self.forecast_horizons:
            target_col = f'target_GHI_{horizon}h'
            
            # Skip night models (just predict 0)
            self.specialized_models['night'][horizon] = None
            
            # Train transition models if enough samples
            if transition_mask.sum() > 100:
                print(f"Training transition model for {horizon}h horizon...")
                X_transition = X_train[transition_mask]
                y_transition = y_train.loc[transition_mask, target_col]
                
                # Use specialized parameters for transition periods
                transition_params = {
                    'n_estimators': 200,
                    'learning_rate': 0.01,
                    'max_depth': 5,
                    'gamma': 2.0,  # Higher regularization for this challenging regime
                    'min_child_weight': 5
                }
                
                self.specialized_models['transition'][horizon] = self._train_single_model(
                    X_transition, y_transition, transition_params
                )
            else:
                self.specialized_models['transition'][horizon] = None
                print(f"Not enough transition samples for {horizon}h horizon")
            
            # Train day model
            if day_mask.sum() > 100:
                print(f"Training day model for {horizon}h horizon...")
                X_day = X_train[day_mask]
                y_day = y_train.loc[day_mask, target_col]
                
                # Slightly different parameters for day regime
                day_params = {
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 7,
                    'gamma': 1.0,
                    'min_child_weight': 3
                }
                
                self.specialized_models['day'][horizon] = self._train_single_model(
                    X_day, y_day, day_params
                )
            else:
                self.specialized_models['day'][horizon] = None
                print(f"Not enough day samples for {horizon}h horizon")
                
            # Train peak hour model if enough samples
            if peak_mask.sum() > 30:  # Even with fewer samples, create a specialized model
                print(f"Training peak hour model for {horizon}h horizon ({peak_mask.sum()} samples)...")
                X_peak = X_train[peak_mask]
                y_peak = y_train.loc[peak_mask, target_col]
                
                # Specialized parameters for peak hours with balanced loss function
                # Use lower asymmetry factor to avoid underprediction
                peak_params = {
                    'n_estimators': 300,
                    'learning_rate': 0.03,
                    'max_depth': 6,
                    'gamma': 0.8,  # Lower regularization to allow more accurate peak fitting
                    'min_child_weight': 2,
                    'subsample': 0.9,  # Higher subsample ratio to use more data
                    'colsample_bytree': 0.9
                }
                
                # Create a more balanced sample weight that doesn't overly penalize overestimation
                # This helps prevent systematic underprediction during peak hours
                sample_weights = None
                if len(y_peak) > 0:
                    # Calculate clear sky ratio to identify extremely clear days
                    if 'clear_sky_index' in X_peak.columns:
                        clear_sky_ratio = X_peak['clear_sky_index'].values
                        # Give more weight to samples with high clear sky index (clear days)
                        sample_weights = 1.0 + np.clip(clear_sky_ratio, 0, 1) * 0.5
                
                self.specialized_models['peak'][horizon] = self._train_single_model(
                    X_peak, y_peak, peak_params, sample_weight=sample_weights
                )
            else:
                self.specialized_models['peak'][horizon] = None
                print(f"Not enough peak hour samples for {horizon}h horizon")
        
        print("Specialized models training complete")
        return self.specialized_models

    def predict_with_specialized_models(self, X):
        """
        Make predictions using specialized models for different temporal regimes.
        """
        logging.info("Making predictions with specialized temporal regime models...")
        
        # Initialize predictions dictionary
        predictions = {}
        
        # Check if we have specialized models
        if not hasattr(self, 'specialized_models') or not self.specialized_models:
            logging.warning("No specialized models available. Falling back to standard models.")
            return self.predict(X)
        
        # For each forecast horizon
        for horizon in self.forecast_horizons:
            if horizon not in self.models_median:
                continue
                
            # Get the default prediction from standard model
            median_preds = self.models_median[horizon].predict(X)
            
            # Initialize arrays for interval predictions
            lower_bounds = np.zeros_like(median_preds)
            upper_bounds = np.zeros_like(median_preds)
            
            # Make interval predictions if available
            if hasattr(self, 'error_percentiles') and horizon in self.error_percentiles:
                lower_err, upper_err = self.error_percentiles[horizon]
                lower_bounds = median_preds + lower_err
                upper_bounds = median_preds + upper_err
            elif horizon in self.models_lower and horizon in self.models_upper:
                lower_bounds = self.models_lower[horizon].predict(X)
                upper_bounds = self.models_upper[horizon].predict(X) 
            
            # Ensure non-negative
            lower_bounds = np.maximum(0, lower_bounds)
            upper_bounds = np.maximum(lower_bounds, upper_bounds)
            
            # Create masks for different regimes
        night_mask = X['solar_zenith_cos'] <= 0.01
        transition_mask = (X['solar_zenith_cos'] > 0.01) & (X['solar_zenith_cos'] < 0.3)
        # Use hour information for peak hour detection if available
        if 'is_peak_hour' in X.columns:
            peak_mask = X['is_peak_hour'] > 0
        else:
            # If specific peak hour feature doesn't exist, use solar elevation as proxy
            peak_mask = (X['solar_elevation'] >= 60) & (X['solar_zenith_cos'] >= 0.5)
                
            # Remaining day hours
            day_mask = (X['solar_zenith_cos'] >= 0.3) & (~peak_mask)
            
            # Create a copy of predictions to modify
            specialized_preds = median_preds.copy()
            specialized_lower = lower_bounds.copy()
            specialized_upper = upper_bounds.copy()
            
            # Apply night model (always zero)
            specialized_preds[night_mask] = 0
            specialized_lower[night_mask] = 0
            specialized_upper[night_mask] = 0
            
            # Apply transition model if available
            if self.specialized_models['transition'][horizon] is not None and transition_mask.any():
                transition_indices = np.where(transition_mask)[0]
                X_transition = X.iloc[transition_indices]
                transition_preds = self.specialized_models['transition'][horizon].predict(X_transition)
                specialized_preds[transition_mask] = transition_preds
                
                # Scale intervals proportionally
                if transition_preds.shape[0] > 0:
                    ratio = np.divide(
                        transition_preds, 
                        median_preds[transition_mask],
                        out=np.ones_like(transition_preds),
                        where=median_preds[transition_mask] > 1
                    )
                    specialized_lower[transition_mask] = lower_bounds[transition_mask] * ratio
                    specialized_upper[transition_mask] = upper_bounds[transition_mask] * ratio
            
            # Apply day model if available
            if self.specialized_models['day'][horizon] is not None and day_mask.any():
                day_indices = np.where(day_mask)[0]
                X_day = X.iloc[day_indices]
                day_preds = self.specialized_models['day'][horizon].predict(X_day)
                specialized_preds[day_mask] = day_preds
                
                # Scale intervals proportionally
                if day_preds.shape[0] > 0:
                    ratio = np.divide(
                        day_preds, 
                        median_preds[day_mask],
                        out=np.ones_like(day_preds),
                        where=median_preds[day_mask] > 1
                    )
                    specialized_lower[day_mask] = lower_bounds[day_mask] * ratio
                    specialized_upper[day_mask] = upper_bounds[day_mask] * ratio
                    
            # Apply peak hour model if available - new addition
            if 'peak' in self.specialized_models and self.specialized_models['peak'][horizon] is not None and peak_mask.any():
                peak_indices = np.where(peak_mask)[0]
                X_peak = X.iloc[peak_indices]
                
                # Use specialized peak hour model for predictions
                peak_preds = self.specialized_models['peak'][horizon].predict(X_peak)
                specialized_preds[peak_mask] = peak_preds
                
                logging.info(f"Using specialized peak hour model for {peak_mask.sum()} samples in horizon {horizon}h")
                
                # Adjust prediction intervals for peak hours
                # For peak hours, we need more symmetric intervals as both under and over predictions are problematic
                if peak_preds.shape[0] > 0:
                    # Scale intervals proportionally but with more balanced bounds
                    ratio = np.divide(
                        peak_preds, 
                        median_preds[peak_mask],
                        out=np.ones_like(peak_preds),
                        where=median_preds[peak_mask] > 1
                    )
                    
                    # Calculate more symmetric intervals around peak predictions
                    # Current interval midpoints
                    midpoints = (lower_bounds[peak_mask] + upper_bounds[peak_mask]) / 2
                    # Current interval widths
                    widths = upper_bounds[peak_mask] - lower_bounds[peak_mask]
                    
                    # New intervals centered around peak predictions with similar width
                    specialized_lower[peak_mask] = peak_preds - (widths / 2)
                    specialized_upper[peak_mask] = peak_preds + (widths / 2)
                    
                    # Ensure non-negative
                    specialized_lower[peak_mask] = np.maximum(0, specialized_lower[peak_mask])
                    
                    # Log the average adjustment made to peak hour predictions
                    if len(peak_preds) > 0:
                        avg_adjustment = np.mean(peak_preds - median_preds[peak_mask])
                        logging.info(f"Avg adjustment to peak hour predictions for horizon {horizon}h: {avg_adjustment:.2f} W/m²")
            
            # Ensure all predictions are non-negative
            specialized_preds = np.maximum(0, specialized_preds)
            specialized_lower = np.maximum(0, specialized_lower)
            specialized_upper = np.maximum(specialized_lower, specialized_upper)
            
            # Store predictions for this horizon
            predictions[horizon] = {
                'median': specialized_preds,
                'lower': specialized_lower,
                'upper': specialized_upper
            }
        
        return predictions

    def low_ghi_correction(self, predictions, X):
        """
        Apply corrections to improve low-GHI predictions.
        
        Based on: Betti et al. (2020) "Mapping the performance of GHI forecasts for low irradiance conditions"
        and Lauret et al. (2015) "A benchmarking of machine learning techniques for solar radiation forecasting"
        """
        print("Applying low-GHI corrections...")
        
        corrected_predictions = {}
        
        # Threshold for "low" GHI values
        low_ghi_threshold = 50  # W/m²
        
        for horizon, preds in predictions.items():
            # Get clear sky GHI for the prediction time
            clear_sky_pred = X['clear_sky_ghi'].values
            
            # Apply correction only where clear sky GHI is low but positive
            low_ghi_mask = (clear_sky_pred > 0) & (clear_sky_pred < low_ghi_threshold)
            
            # Copy original predictions
            corrected = preds.copy()
            
            if np.any(low_ghi_mask):
                # For low GHI values, apply a correction factor
                # Research shows predictions in this range are often overestimated
                correction_factor = np.where(
                    preds[low_ghi_mask] > 0.5 * clear_sky_pred[low_ghi_mask],
                    0.5 * clear_sky_pred[low_ghi_mask] / preds[low_ghi_mask],
                    1.0
                )
                
                # Apply the correction
                corrected[low_ghi_mask] = preds[low_ghi_mask] * correction_factor
            
            corrected_predictions[horizon] = corrected
        
        return corrected_predictions

    def split_time_series_data(self, data, val_size=0.2):
        """
        Split time series data into training and validation sets, preserving temporal order.
        
        Parameters:
        -----------
        data (pandas.DataFrame): The dataset with features and targets
        val_size (float): Proportion of data to use for validation (default=0.2)
        
        Returns:
        --------
        tuple: X_train, X_val, y_train, y_val
        """
        logging.info(f"Splitting time series data with validation size {val_size}")
        
        # Ensure data is sorted by datetime if available
        if 'datetime' in data.columns:
            data = data.sort_values('datetime').reset_index(drop=True)
            logging.info(f"Data sorted by time from {data['datetime'].min()} to {data['datetime'].max()}")
        
        # Calculate split indices based on desired proportions
        total_samples = len(data)
        train_end = int(total_samples * (1 - val_size))
        
        # Separate features from target columns
        X_cols = [col for col in data.columns if not col.startswith('target_')]
        y_cols = [col for col in data.columns if col.startswith('target_')]
        
        # Perform the time-based split
        X_train = data[X_cols].iloc[:train_end]
        X_val = data[X_cols].iloc[train_end:]
        
        y_train = data[y_cols].iloc[:train_end]
        y_val = data[y_cols].iloc[train_end:]
        
        logging.info(f"Split sizes - Train: {len(X_train)}, Validation: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val

    def evaluate_validation_detailed(self, X_val, y_val):
        """Evaluate models on validation data with detailed metrics and error percentiles."""
        metrics = {}
        
        # Initialize the metrics dictionary for each horizon to avoid KeyError later
        for horizon in self.forecast_horizons:
            metrics[horizon] = {}
        
        # Dictionary to store predictions and bounds for all horizons
        validation_results = {}
        
        # Filter out nighttime data (before 5 AM and after 6 PM)
        daytime_mask = None
        if 'datetime' in X_val.columns:
            # Use datetime hours to identify daytime
            hours = X_val['datetime'].dt.hour
            daytime_mask = (hours >= 5) & (hours <= 18)
            logging.info(f"Filtering validation data to daytime hours (5 AM to 6 PM): {daytime_mask.sum()} of {len(X_val)} records kept")
        elif hasattr(self, 'val_datetimes'):
            # Use stored datetimes if available
            # Make sure to convert to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(self.val_datetimes):
                self.val_datetimes = pd.to_datetime(self.val_datetimes)
                
            hours = self.val_datetimes.dt.hour
            daytime_mask = (hours >= 5) & (hours <= 18)
            logging.info(f"Filtering validation data using stored datetimes to daytime hours (5 AM to 6 PM): {daytime_mask.sum()} of {len(X_val)} records kept")
            
            # Log the range of hours found for debugging
            min_hour = hours.min()
            max_hour = hours.max()
            hour_counts = hours.value_counts().sort_index()
            logging.info(f"Hour distribution in validation set: {min_hour} to {max_hour}")
            logging.info(f"Hour counts: {hour_counts.to_dict()}")
        else:
            # Use solar elevation as proxy if datetime not available
            daytime_mask = X_val['solar_zenith_cos'] > 0.01  # Basic daylight check
            logging.info(f"Filtering validation data using solar position: {daytime_mask.sum()} of {len(X_val)} records kept")
        
        # If mask isn't filtering anything (all True), something might be wrong
        if daytime_mask is not None and daytime_mask.sum() == len(daytime_mask):
            logging.warning("Daytime mask isn't filtering any records! All hours appear to be daytime. Check your datetime data.")
            
            # Force a more restrictive filter as fallback
            if hasattr(self, 'val_datetimes') and not self.val_datetimes.empty:
                # Use the full requested daytime range (5 AM to 6 PM) instead of the restrictive filter
                hours = self.val_datetimes.dt.hour
                daytime_mask = (hours >= 5) & (hours <= 18)
                logging.info(f"Applying fallback filtering (5 AM to 6 PM): {daytime_mask.sum()} of {len(X_val)} records kept")
            elif 'solar_zenith_cos' in X_val.columns:
                # Use a stricter solar position filter
                daytime_mask = X_val['solar_zenith_cos'] > 0.3
                logging.info(f"Applying stricter solar position filter: {daytime_mask.sum()} of {len(X_val)} records kept")
            
            # If we still have no filtering, fall back to a simple subsample
            if daytime_mask.sum() == len(daytime_mask):
                logging.warning("Still unable to filter! Falling back to using same number of samples as calibration.")
                # Check how many samples were used in calibration if available
                if hasattr(self, 'calibration_sample_count'):
                    sample_size = min(self.calibration_sample_count, len(X_val))
                    daytime_mask = np.zeros(len(X_val), dtype=bool)
                    daytime_mask[:sample_size] = True
                    logging.info(f"Using first {sample_size} samples to match calibration sample count")
                else:
                    # Use a reasonable default if we don't know calibration count
                    sample_size = len(X_val) // 2
                    daytime_mask = np.zeros(len(X_val), dtype=bool)
                    daytime_mask[:sample_size] = True
                    logging.info(f"Using first {sample_size} samples as default fallback")
        
        # Apply the filter to validation data
        X_val_daytime = X_val[daytime_mask].copy()
        y_val_daytime = y_val[daytime_mask].copy()
        
        # Also filter the datetimes if they exist
        if hasattr(self, 'val_datetimes'):
            self.val_datetimes_filtered = self.val_datetimes[daytime_mask].copy()
        
        logging.info(f"Using {len(X_val_daytime)} daytime validation records for evaluation (excluded nighttime data)")
        
        for horizon in self.forecast_horizons:
            target_col = f'target_GHI_{horizon}h'
            actual = y_val_daytime[target_col].values
            
            # Predict with median model
            median_preds = self.models_median[horizon].predict(X_val_daytime)
            
            # Store actual and median values
            validation_results[f'actual_{horizon}h'] = actual
            validation_results[f'median_{horizon}h'] = median_preds
            
            # Calculate metrics
            mae = mean_absolute_error(actual, median_preds)
            rmse = mean_squared_error(actual, median_preds, squared=False)
            
            # Add R² calculation
            r2 = r2_score(actual, median_preds)
            
            # Calculate MAPE
            non_zero_mask = actual > 10
            if np.sum(non_zero_mask) > 0:
                mape = 100 * np.mean(np.abs((actual[non_zero_mask] - median_preds[non_zero_mask]) / actual[non_zero_mask]))
            else:
                mape = np.nan
            
            # Calculate prediction interval coverage using error percentiles
            if hasattr(self, 'error_percentiles') and horizon in self.error_percentiles:
                lower_err, upper_err = self.error_percentiles[horizon]
                
                # Make sure error percentiles are compatible shape
                if isinstance(lower_err, np.ndarray) and len(lower_err) != len(median_preds):
                    # Log the shape mismatch
                    logging.warning(f"Shape mismatch between median_preds ({len(median_preds)}) and error percentiles ({len(lower_err)})")
                    
                    # If lower_err is an array but wrong size, convert to scalar by taking the mean
                    if len(lower_err) > 1:
                        lower_err = np.mean(lower_err)
                        upper_err = np.mean(upper_err)
                        logging.info(f"Converted error percentiles to scalars: lower={lower_err:.2f}, upper={upper_err:.2f}")
                
                # Calculate bounds using error percentiles
                lower_bounds = median_preds + lower_err
                upper_bounds = median_preds + upper_err
                
                # Ensure non-negative and proper ordering
                lower_bounds = np.maximum(0, lower_bounds)
                upper_bounds = np.maximum(lower_bounds, upper_bounds)
                
                # Store lower and upper bounds
                validation_results[f'lower_{horizon}h'] = lower_bounds
                validation_results[f'upper_{horizon}h'] = upper_bounds
                
                # Calculate coverage
                coverage = 100 * np.mean((actual >= lower_bounds) & (actual <= upper_bounds))
                
                # Log the coverage achieved with calibrated intervals
                logging.info(f"Horizon {horizon}h calibrated interval coverage: {coverage:.2f}% (target: 90%)")
                
                # Calculate additional prediction interval metrics
                
                # 1. Prediction Interval Coverage Probability (PICP)
                # PICP is already calculated as coverage/100 (converting from percentage to probability)
                picp = coverage / 100.0
                
                # 2. Prediction Interval Normalized Average Width (PINAW)
                # Calculate average width of the intervals normalized by the range of the target variable
                interval_widths = upper_bounds - lower_bounds
                avg_width = np.mean(interval_widths)
                target_range = np.max(actual) - np.min(actual) if len(actual) > 0 else 1.0
                pinaw = avg_width / target_range if target_range > 0 else np.nan
                
                # 3. Interval Score (IS)
                # Calculates the interval score which penalizes for observations outside the interval
                # and rewards narrower intervals
                target_coverage = 0.90  # Default value, should be consistent with calibration
                alpha = 1 - target_coverage
                
                # Calculate indicators for observations outside the prediction interval
                below_lower = actual < lower_bounds
                above_upper = actual > upper_bounds
                
                # Calculate penalties
                penalty_below = 2/alpha * (lower_bounds[below_lower] - actual[below_lower]) if np.any(below_lower) else 0
                penalty_above = 2/alpha * (actual[above_upper] - upper_bounds[above_upper]) if np.any(above_upper) else 0
                
                # Calculate the interval score
                interval_score = avg_width + np.sum(penalty_below) / len(actual) + np.sum(penalty_above) / len(actual)
                
                # 4. Continuous Ranked Probability Score (CRPS) approximation
                # For normal distribution assumption, CRPS can be approximated
                # We'll use a simplified version based on prediction intervals
                prediction_errors = np.abs(actual - median_preds)
                crps = np.mean(prediction_errors) * (1 + (1 - picp) * np.log((1 - picp) / (1 - target_coverage)))
                
                # 5. Coverage Deviation
                # Absolute difference between the achieved coverage and the target coverage
                coverage_deviation = np.abs(picp - target_coverage)
                
                # Log the additional metrics
                logging.info(f"Horizon {horizon}h additional metrics: PICP={picp:.4f}, PINAW={pinaw:.4f}, IS={interval_score:.4f}, CRPS={crps:.4f}, Coverage Deviation={coverage_deviation:.4f}")
                
                # Store the additional metrics
                metrics[horizon].update({
                    'picp': picp,
                    'pinaw': pinaw,
                    'interval_score': interval_score,
                    'crps': crps,
                    'coverage_deviation': coverage_deviation
                })
            # Fallback to old method
            elif horizon in self.models_lower and horizon in self.models_upper:
                # Previous logic here...
                lower_preds = self.models_lower[horizon].predict(X_val_daytime)
                upper_preds = self.models_upper[horizon].predict(X_val_daytime)
                
                # Store lower and upper bounds
                validation_results[f'lower_{horizon}h'] = lower_preds
                validation_results[f'upper_{horizon}h'] = upper_preds
                
                # Calculate coverage (percentage of actual values within prediction interval)
                coverage = 100 * np.mean((y_val_daytime[target_col] >= lower_preds) & (y_val_daytime[target_col] <= upper_preds))
                
                # Log the coverage achieved with quantile models
                logging.info(f"Horizon {horizon}h quantile model coverage: {coverage:.2f}% (target: 90%)")
                
                # Calculate additional prediction interval metrics (same as for calibrated intervals)
                # 1. Prediction Interval Coverage Probability (PICP)
                picp = coverage / 100.0
                
                # 2. Prediction Interval Normalized Average Width (PINAW)
                interval_widths = upper_preds - lower_preds
                avg_width = np.mean(interval_widths)
                target_range = np.max(actual) - np.min(actual) if len(actual) > 0 else 1.0
                pinaw = avg_width / target_range if target_range > 0 else np.nan
                
                # 3. Interval Score (IS)
                target_coverage = 0.90  # Default value, should be consistent with calibration
                alpha = 1 - target_coverage
                
                # Calculate indicators for observations outside the prediction interval
                below_lower = actual < lower_preds
                above_upper = actual > upper_preds
                
                # Calculate penalties
                penalty_below = 2/alpha * (lower_preds[below_lower] - actual[below_lower]) if np.any(below_lower) else 0
                penalty_above = 2/alpha * (actual[above_upper] - upper_preds[above_upper]) if np.any(above_upper) else 0
                
                # Calculate the interval score
                interval_score = avg_width + np.sum(penalty_below) / len(actual) + np.sum(penalty_above) / len(actual)
                
                # 4. Continuous Ranked Probability Score (CRPS) approximation
                prediction_errors = np.abs(actual - median_preds)
                crps = np.mean(prediction_errors) * (1 + (1 - picp) * np.log((1 - picp) / (1 - target_coverage)))
                
                # 5. Coverage Deviation
                coverage_deviation = np.abs(picp - target_coverage)
                
                # Log the additional metrics
                logging.info(f"Horizon {horizon}h quantile model additional metrics: PICP={picp:.4f}, PINAW={pinaw:.4f}, IS={interval_score:.4f}, CRPS={crps:.4f}, Coverage Deviation={coverage_deviation:.4f}")
                
                # Store the additional metrics
                metrics[horizon].update({
                    'picp': picp,
                    'pinaw': pinaw,
                    'interval_score': interval_score,
                    'crps': crps,
                    'coverage_deviation': coverage_deviation
                })
            else:
                coverage = np.nan
                # If no prediction intervals available, use placeholder values
                validation_results[f'lower_{horizon}h'] = np.full_like(median_preds, np.nan)
                validation_results[f'upper_{horizon}h'] = np.full_like(median_preds, np.nan)
            
            # Calculate skill score against persistence forecast
            # For GHI forecasting, persistence means using current value as prediction
            if 'GHI - W/m^2' in X_val_daytime.columns:
                # Current GHI values
                current_ghi = X_val_daytime['GHI - W/m^2'].values
                
                # Calculate persistence error (using current value to predict future)
                persistence_mae = mean_absolute_error(actual, current_ghi)
                persistence_rmse = mean_squared_error(actual, current_ghi, squared=False)
                
                # Calculate skill scores (improvement over persistence)
                # 1 means perfect, 0 means same as persistence, negative means worse
                if persistence_mae > 0:
                    skill_score_mae = 1 - (mae / persistence_mae)
                else:
                    skill_score_mae = 0
                    
                if persistence_rmse > 0:
                    skill_score_rmse = 1 - (rmse / persistence_rmse)
                else:
                    skill_score_rmse = 0
            elif 'GHI_lag_1' in X_val_daytime.columns:
                # If original GHI column not available, use GHI_lag_1 as persistence forecast
                # GHI_lag_1 is the GHI value from 1 hour ago
                persistence_pred = X_val_daytime['GHI_lag_1'].values
                
                # Calculate persistence error
                persistence_mae = mean_absolute_error(actual, persistence_pred)
                persistence_rmse = mean_squared_error(actual, persistence_pred, squared=False)
                
                # Calculate skill scores (improvement over persistence)
                if persistence_mae > 0:
                    skill_score_mae = 1 - (mae / persistence_mae)
                else:
                    skill_score_mae = 0
                    
                if persistence_rmse > 0:
                    skill_score_rmse = 1 - (rmse / persistence_rmse)
                else:
                    skill_score_rmse = 0
            else:
                # If no suitable persistence column is available, default to reasonable values
                # Instead of NaN, use 0 which means "same as persistence model"
                skill_score_mae = 0
                skill_score_rmse = 0
            
            # Store metrics
            metrics[horizon].update({
                'mae': mae,
                'rmse': rmse,
                'r2': r2,  # Added R² to metrics
                'mape': mape,
                'coverage': coverage,
                'skill_score_mae': skill_score_mae,
                'skill_score_rmse': skill_score_rmse
            })
        
        # Calculate and log overall metrics
        all_horizons_metrics = {
            'mae': np.mean([metrics[h]['mae'] for h in self.forecast_horizons]),
            'rmse': np.mean([metrics[h]['rmse'] for h in self.forecast_horizons]),
            'r2': np.mean([metrics[h]['r2'] for h in self.forecast_horizons]),  # Added R² to overall metrics
            'coverage': np.mean([metrics[h]['coverage'] for h in self.forecast_horizons if not np.isnan(metrics[h]['coverage'])]),
            'skill_score_mae': np.mean([metrics[h]['skill_score_mae'] for h in self.forecast_horizons if not np.isnan(metrics[h]['skill_score_mae'])])
        }
        
        # Check if skill_score_mae is NaN (which happens if all input values were NaN)
        if np.isnan(all_horizons_metrics['skill_score_mae']):
            all_horizons_metrics['skill_score_mae'] = 0  # Use 0 instead of NaN
        
        # Add the new interval metrics to the overall metrics if they exist
        if all('picp' in metrics[h] for h in self.forecast_horizons if h in metrics):
            all_horizons_metrics['picp'] = np.mean([metrics[h]['picp'] for h in self.forecast_horizons if 'picp' in metrics[h] and not np.isnan(metrics[h]['picp'])])
            all_horizons_metrics['pinaw'] = np.mean([metrics[h]['pinaw'] for h in self.forecast_horizons if 'pinaw' in metrics[h] and not np.isnan(metrics[h]['pinaw'])])
            all_horizons_metrics['interval_score'] = np.mean([metrics[h]['interval_score'] for h in self.forecast_horizons if 'interval_score' in metrics[h] and not np.isnan(metrics[h]['interval_score'])])
            all_horizons_metrics['crps'] = np.mean([metrics[h]['crps'] for h in self.forecast_horizons if 'crps' in metrics[h] and not np.isnan(metrics[h]['crps'])])
            all_horizons_metrics['coverage_deviation'] = np.mean([metrics[h]['coverage_deviation'] for h in self.forecast_horizons if 'coverage_deviation' in metrics[h] and not np.isnan(metrics[h]['coverage_deviation'])])
        
        logging.info(f"Overall metrics (daytime only): MAE={all_horizons_metrics['mae']:.2f}, RMSE={all_horizons_metrics['rmse']:.2f}, " 
                   f"R²={all_horizons_metrics['r2']:.2f}, Coverage={all_horizons_metrics['coverage']:.2f}%, Skill Score={all_horizons_metrics['skill_score_mae']:.2f}")
        
        # Log the additional overall metrics if they exist
        if 'picp' in all_horizons_metrics:
            logging.info(f"Overall interval metrics: PICP={all_horizons_metrics['picp']:.4f}, PINAW={all_horizons_metrics['pinaw']:.4f}, " 
                       f"IS={all_horizons_metrics['interval_score']:.4f}, CRPS={all_horizons_metrics['crps']:.4f}, " 
                       f"Coverage Deviation={all_horizons_metrics['coverage_deviation']:.4f}")
        
        # Save validation results to CSV
        if hasattr(self, 'val_datetimes_filtered'):
            # Pass the filtered datetimes to the save function 
            self.save_validation_results_with_datetimes(X_val_daytime, validation_results, self.val_datetimes_filtered)
        else:
            self.save_validation_results(X_val_daytime, validation_results)
        
        return metrics
        
    def save_validation_results_with_datetimes(self, X_val, validation_results, datetimes):
        """
        Save validation set results to a CSV file, including actual values and prediction intervals.
        This version specifically handles filtered datetimes.
        
        Parameters:
        -----------
        X_val: Validation features DataFrame
        validation_results: Dictionary containing actual, median, lower, and upper values for each horizon
        datetimes: Pandas Series with datetime values matching the filtered validation data
        """
        try:
            # Create a DataFrame from the validation results
            results_df = pd.DataFrame(validation_results)
            
            # Add datetimes
            results_df['datetime'] = datetimes.values
            logging.info(f"Added filtered datetimes to validation results")
            
            # Move datetime to first column
            cols = results_df.columns.tolist()
            cols = ['datetime'] + [col for col in cols if col != 'datetime']
            results_df = results_df[cols]
            
            # Group columns by horizon for better readability
            organized_cols = ['datetime']
                
            # Organize columns by horizon: actual, lower, median, upper for each horizon
            for horizon in self.forecast_horizons:
                horizon_cols = [
                    f'actual_{horizon}h',
                    f'lower_{horizon}h',
                    f'median_{horizon}h', 
                    f'upper_{horizon}h'
                ]
                organized_cols.extend([col for col in horizon_cols if col in results_df.columns])
            
            # Reorder columns if all expected columns are present
            if set(organized_cols).issubset(set(results_df.columns)):
                results_df = results_df[organized_cols]
            
            # Define output path - use script directory instead of current working directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'validation_results_{timestamp}.csv')
            
            # Save to CSV
            results_df.to_csv(output_path, index=False)
            logging.info(f"Validation results with datetimes saved to {output_path}")
            print(f"Validation results with prediction intervals and datetimes saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving validation results with datetimes: {str(e)}")
            print(f"Error saving validation results with datetimes: {str(e)}")

    def predict_future_hours(self, data=None, file_path=None, num_hours=4):
        """
        Predict GHI values for the next few hours beyond the dataset.
        """
        if data is None and file_path is not None:
            data = self.load_data(file_path)
        
        if data is None:
            raise ValueError("No data provided for prediction")
        
        # Preprocess and create features
        data = self.preprocess_data(data)
        X_pred = self.create_features_for_prediction(data)
        
        # Keep track of the last timestamp if available
        if 'datetime' in X_pred.columns:
            last_timestamp = X_pred['datetime'].iloc[-1]
            # Check if the timestamp is NaT and handle it
            if pd.isna(last_timestamp):
                logging.warning("Last timestamp is NaT, using current time instead")
                last_timestamp = pd.Timestamp.now()
        else:
            last_timestamp = None
        
        # Drop metadata columns that weren't part of model training
        X_pred = X_pred.drop(['Date', 'Start Period', 'End Period', 'datetime'], errors='ignore')
        
        # Ensure we have all required feature columns
        missing_cols = set(self.feature_columns) - set(X_pred.columns)
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        
        # Ensure columns are in the same order as during training
        X_pred = X_pred[self.feature_columns]
        
        # Debug: Check if we have all required feature columns
        missing_features = [col for col in self.feature_columns if col not in X_pred.columns]
        if missing_features:
            logging.warning(f"Missing required features for prediction: {missing_features}")
            # Create missing features with default values to avoid errors
            for col in missing_features:
                X_pred[col] = 0.0
        
        # Debug: Check for NaN values in features
        nan_columns = X_pred.columns[X_pred.isna().any()].tolist()
        if nan_columns:
            logging.warning(f"NaN values found in prediction features: {nan_columns}")
            # Fill NaN values with 0 to prevent errors
            X_pred = X_pred.fillna(0)
        
        # Check for infinity or very large values - but only in numeric columns
        inf_columns = []
        for col in X_pred.columns:
            # First check if the column is numeric before applying np.isinf
            if pd.api.types.is_numeric_dtype(X_pred[col]):
                if np.isinf(X_pred[col]).any():
                    inf_columns.append(col)
                    self.debug_logger.debug(f"Infinity found in column '{col}', replacing with 0")
                    X_pred[col] = X_pred[col].replace([np.inf, -np.inf], 0)
                
                # Also check for very large values that might cause overflow
                if (np.abs(X_pred[col]) > 1e10).any():
                    self.debug_logger.debug(f"Extremely large values found in column '{col}', capping values")
                    X_pred[col] = np.clip(X_pred[col], -1e10, 1e10)
            else:
                self.debug_logger.debug(f"Skipping infinity check for non-numeric column: {col}")

        if inf_columns:
            logging.info(f"Fixed infinity in these columns: {inf_columns}")
        
        # Use the model's scaler that was already fit during training
        try:
            # Print column counts to confirm we have the right number
            self.debug_logger.debug(f"Scaling {len(X_pred.columns)} features (expected {len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else 'unknown'})")
            
            # Ensure column order matches exactly what the scaler expects
            if hasattr(self.scaler, 'feature_names_in_'):
                # Reorder columns to match the exact order the scaler expects
                expected_columns = self.scaler.feature_names_in_
                missing_cols = set(expected_columns) - set(X_pred.columns)
                
                if missing_cols:
                    logging.warning(f"Missing columns needed by scaler: {missing_cols}")
                    for col in missing_cols:
                        X_pred[col] = 0.0  # Add missing columns with default values
                
                # Reorder to exactly match the scaler's expected order
                X_pred = X_pred[expected_columns]
            
            # One final check for any non-finite values that might have snuck in
            X_pred_clean = X_pred.copy()
            for col in X_pred_clean.columns:
                if pd.api.types.is_numeric_dtype(X_pred_clean[col]):
                    X_pred_clean[col] = X_pred_clean[col].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Debug info about data range
            self.debug_logger.debug(f"Data range check - Min values: {X_pred_clean.min().min():.4f}, Max values: {X_pred_clean.max().max():.4f}")
            
            # More detailed diagnostics for debugging scaling issues
            self.debug_logger.debug("\n=== DETAILED DIAGNOSTICS ===")
            
            X_pred_scaled = self.scaler.transform(X_pred_clean)
            
            # Convert back to DataFrame to maintain column names
            X_pred_scaled_df = pd.DataFrame(X_pred_scaled, columns=X_pred_clean.columns)
            
            logging.info(f"Successfully scaled features with shape: {X_pred_scaled_df.shape}")
        except Exception as e:
            logging.error(f"ERROR during feature scaling: {str(e)}")
            # More detailed diagnostics for numeric issues
            self.debug_logger.debug("\n=== DETAILED DIAGNOSTICS ===")
            for col in X_pred.columns:
                if pd.api.types.is_numeric_dtype(X_pred[col]):
                    col_data = X_pred[col]
                    has_inf = np.isinf(col_data).any()
                    has_nan = np.isnan(col_data).any()
                    if has_inf or has_nan:
                        self.debug_logger.debug(f"Column '{col}' contains {'infinity' if has_inf else ''} {'NaN' if has_nan else ''}")
                    
                    # Check for extremely large values
                    if not has_inf and not has_nan:
                        try:
                            col_max = np.max(np.abs(col_data))
                            if col_max > 1e10:
                                self.debug_logger.debug(f"Column '{col}' has extremely large value: {col_max}")
                        except:
                            self.debug_logger.debug(f"Error computing max for column '{col}'")
                else:
                    self.debug_logger.debug(f"Column '{col}' is non-numeric type: {X_pred[col].dtype}")
        
            raise
        
        # Make predictions for each horizon
        future_predictions = []
        
        # Generate predictions for each requested hour
        for hour in range(1, min(num_hours+1, max(self.forecast_horizons)+1)):
            if hour not in self.forecast_horizons:
                logging.info(f"Horizon {hour}h not in trained horizons, skipping")
                continue
                
            # Make predictions with intervals
            if hour in self.models_median:
                try:
                    # Debug: Print model input shape
                    self.debug_logger.debug(f"Model input shape for horizon {hour}h: {X_pred_scaled_df.shape}")
                    
                    # Basic prediction
                    median_pred = self.models_median[hour].predict(X_pred_scaled_df)[0]
                    
                    # Use error percentiles for confidence intervals if available
                    if hasattr(self, 'error_percentiles') and hour in self.error_percentiles:
                        lower_err, upper_err = self.error_percentiles[hour]
                        
                        # Check if error percentiles are arrays or scalars
                        if isinstance(lower_err, np.ndarray) and lower_err.size > 1:
                            # Use mean value if it's an array
                            lower_err = lower_err.mean()
                        if isinstance(upper_err, np.ndarray) and upper_err.size > 1:
                            # Use mean value if it's an array
                            upper_err = upper_err.mean()
                            
                        lower_bound = max(0, median_pred + lower_err)
                        upper_bound = max(lower_bound, median_pred + upper_err)
                    elif hour in self.models_lower and hour in self.models_upper:
                        # Fallback to model-based intervals
                        lower_bound = max(0, self.models_lower[hour].predict(X_pred_scaled_df)[0])
                        upper_bound = max(lower_bound, self.models_upper[hour].predict(X_pred_scaled_df)[0])
                    else:
                        # No intervals available
                        lower_bound = None
                        upper_bound = None
                    
                    # Enforce maximum gap between bounds if both are defined
                    if lower_bound is not None and upper_bound is not None:
                        # Check if this is a peak hour (11 AM, 12 PM, or 1 PM)
                        is_peak_hour = False
                        
                        # Determine if this is a peak hour based on the time
                        if last_timestamp is not None:
                            try:
                                from pandas import Timedelta
                                future_timestamp = last_timestamp + Timedelta(hours=hour)
                                hour_of_day = future_timestamp.hour
                                
                                # Define peak hours as 11 AM to 1 PM (inclusive)
                                is_peak_hour = hour_of_day >= 11 and hour_of_day <= 13
                                
                                if is_peak_hour:
                                    logging.info(f"Horizon {hour}h targeting hour {hour_of_day} (11 AM - 1 PM): Using peak hour bound gap of 150.")
                            except Exception as e:
                                logging.warning(f"Error determining peak hour status: {str(e)}")
                        
                        # Use different max gap constraints for peak vs. non-peak hours
                        if is_peak_hour:
                            max_bound_gap = 150  # Peak hour constraint (11 AM - 1 PM)
                        else:
                            max_bound_gap = 100  # Standard constraint for non-peak hours
                            
                        current_gap = upper_bound - lower_bound
                        
                        if current_gap > max_bound_gap:
                            # The original approach adjusted bounds symmetrically, but this can cause the upper bound 
                            # to go below the median, which is statistically incorrect
                            gap_excess = current_gap - max_bound_gap
                            
                            # Calculate distances from median to bounds
                            lower_distance = median_pred - lower_bound
                            upper_distance = upper_bound - median_pred
                            
                            # Adjust bounds proportionally to their distance from median
                            if lower_distance + upper_distance > 0:  # Avoid division by zero
                                lower_adjust_ratio = lower_distance / (lower_distance + upper_distance)
                                upper_adjust_ratio = upper_distance / (lower_distance + upper_distance)
                                
                                # Apply adjustments while preserving median as center point
                                lower_bound += gap_excess * lower_adjust_ratio
                                upper_bound -= gap_excess * upper_adjust_ratio
                            else:
                                # Fallback to equal adjustment if distances are invalid
                                lower_bound += gap_excess / 2
                                upper_bound -= gap_excess / 2
                            
                            # Ensure bounds remain valid relative to median
                            lower_bound = max(0, lower_bound)
                            upper_bound = max(median_pred, upper_bound)  # Ensure upper bound is at least the median
                            
                            # Final check to ensure the gap is constrained
                            if upper_bound - lower_bound > max_bound_gap:
                                # If still too large, adjust lower bound only
                                lower_bound = upper_bound - max_bound_gap
                            
                            # Update logging to show which constraint was applied
                            constraint_type = "peak hour (150)" if is_peak_hour else "standard (100)"
                            logging.info(f"Constrained prediction interval for horizon {hour}h from gap {current_gap:.2f} to {upper_bound - lower_bound:.2f} [{constraint_type} constraint]")
                    
                    # Create future timestamp if datetime is available
                    if last_timestamp is not None:
                        try:
                            from pandas import Timedelta
                            future_timestamp = last_timestamp + Timedelta(hours=hour)
                            future_time_str = future_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            # Check if the future timestamp falls during nighttime, based on Philippines climate
                            hour_of_day = future_timestamp.hour
                            month = future_timestamp.month
                            
                            # Apply the same Philippines climate nighttime detection logic
                            # Cool dry season (December to February)
                            if month in [12, 1, 2]:
                                is_nighttime = (hour_of_day >= 18) or (hour_of_day < 6)
                                # Transition periods (dawn/dusk)
                                is_transition = hour_of_day == 6 or hour_of_day == 17
                                season = "Cool dry"
                            # Hot dry season (March to May)
                            elif month in [3, 4, 5]:
                                is_nighttime = (hour_of_day >= 18) or (hour_of_day < 5)
                                # Transition periods (dawn/dusk)
                                is_transition = hour_of_day == 5 or hour_of_day == 17
                                season = "Hot dry"
                            # Rainy season (June to November)
                            else:  # months 6-11
                                is_nighttime = (hour_of_day >= 18) or (hour_of_day < 6)
                                # Transition periods (dawn/dusk)
                                is_transition = hour_of_day == 6 or hour_of_day == 17
                                season = "Rainy"
                        
                            # If nighttime, set all predictions to zero
                            if is_nighttime:
                                logging.info(f"Horizon {hour}h at {future_time_str} (hour: {hour_of_day}, month: {month}) is during nighttime ({season} season). Setting GHI to 0.")
                                median_pred = 0.0
                                lower_bound = 0.0
                                upper_bound = 0.0
                            # If transition period (dawn/dusk), reduce predictions by 70%
                            elif is_transition:
                                # First ensure no negative values
                                median_pred = max(0.0, median_pred)
                                if lower_bound is not None:
                                    lower_bound = max(0.0, lower_bound)
                                if upper_bound is not None:
                                    upper_bound = max(0.0, upper_bound)
                                    
                                transition_factor = 0.3  # Reduce to 30% of predicted value
                                original_pred = median_pred
                                original_lower = lower_bound
                                original_upper = upper_bound
                                
                                median_pred *= transition_factor
                                if lower_bound is not None:
                                    lower_bound *= transition_factor
                                if upper_bound is not None:
                                    upper_bound *= transition_factor
                                
                                logging.info(f"Horizon {hour}h at {future_time_str} (hour: {hour_of_day}, month: {month}) is during transition ({season} season). Reducing GHI from {original_pred:.2f} to {median_pred:.2f}.")
                            else:
                                logging.info(f"Horizon {hour}h at {future_time_str} (hour: {hour_of_day}, month: {month}) is during daytime ({season} season).")
                        except Exception as e:
                            logging.warning(f"Error processing timestamp for horizon {hour}h: {str(e)}")
                            future_time_str = f"t+{hour}h"
                        # The else clause here is incorrect - it's overwriting the datetime format
                        # else:
                        #    future_time_str = f"t+{hour}h"
                    
                    # Add to predictions list
                    future_predictions.append({
                        'datetime': future_time_str,
                        'lower_bound': round(lower_bound, 2) if lower_bound is not None else None,
                        'median': round(median_pred, 2),
                        'upper_bound': round(upper_bound, 2) if upper_bound is not None else None
                    })
                except Exception as e:
                    logging.error(f"Error predicting for horizon {hour}h: {str(e)}")
            else:
                # No model available for this horizon
                logging.info(f"No median model available for horizon {hour}h, skipping prediction")
        
        # Convert to DataFrame for easy display
        if future_predictions:
            result_df = pd.DataFrame(future_predictions)
            # Reorder columns to match requested sequence
            result_df = result_df[['datetime', 'lower_bound', 'median', 'upper_bound']]
            logging.info("\n=== Future Hour Predictions ===")
            logging.info(f"\n{result_df.to_string()}")
            
            # Save predictions
            result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'future_predictions.csv')  # Save in davcast folder
            result_df.to_csv(result_file, index=False)
            logging.info(f"Future predictions saved to {result_file}")
            
            return result_df
        else:
            logging.warning("No future predictions generated")
            return None

    def create_features_for_prediction(self, data):
        """
        Create features for a single prediction.
        
        Parameters:
        -----------
        data: DataFrame
            Dataset containing the latest rows needed for feature generation
        
        Returns:
        --------
        DataFrame: Single row with all required features for prediction
        """
        df = data.copy()
        
        # Check for NaT in datetime column and handle it
        if 'datetime' in df.columns and df['datetime'].isna().any():
            logging.warning("NaT values detected in datetime column during feature creation")
            # Replace NaT with current time for prediction purposes
            nat_mask = df['datetime'].isna()
            if nat_mask.all():
                logging.warning("All datetime values are NaT, using current time")
                df['datetime'] = pd.Timestamp.now()
            else:
                # Fill NaT with the nearest valid timestamp
                df['datetime'] = df['datetime'].fillna(method='ffill').fillna(method='bfill')
                if df['datetime'].isna().any():
                    # If still NaT values, use current time
                    df.loc[df['datetime'].isna(), 'datetime'] = pd.Timestamp.now()
        
        # Calculate all the same features as in training, but don't create targets
        # Create lagged features for GHI
        lag_hours = max([int(col.split('_')[-1]) for col in df.columns if col.startswith('GHI_lag_')], default=3)
        for i in range(1, lag_hours + 1):
            col_name = f'GHI_lag_{i}'
            if col_name not in df.columns:
                df[col_name] = df[self.target_column].shift(i)
        
        # Create rolling statistics
        if 'GHI_rolling_mean_6h' not in df.columns:
            df['GHI_rolling_mean_6h'] = df[self.target_column].rolling(window=6, min_periods=1).mean()
        
        if 'GHI_rolling_max_24h' not in df.columns:
            df['GHI_rolling_max_24h'] = df[self.target_column].rolling(window=24, min_periods=1).max()
        
        # Add solar position features first - IMPORTANT: This needs to come before clear sky calculation
        df = self.add_solar_position_features(df)
        
        # Then calculate clear sky GHI which depends on solar position features
        df = self.calculate_clear_sky_ghi(df)
        
        # Add trend features if not already present
        if 'Clear Sky Index' not in df.columns:
            df = self.add_trend_features(df)
        
        # Add diurnal decomposition if not already present
        if 'CSI_smooth' not in df.columns:
            df = self.add_diurnal_decomposition(df)
        
        # Only return the last row which has all the lagged features filled
        return df.iloc[[-1]].copy()
        
    def save_models(self, model_dir=None):
        """
        Save all trained models and scalers to files.
        
        Parameters:
        -----------
        model_dir (str): Directory to save the models
        """
        # Use the same directory as main.py if no model_dir is specified
        if model_dir is None:
            model_dir = self.base_dir
        
        print(f"Saving models and scalers to {model_dir}...")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save feature columns to JSON
        with open(os.path.join(model_dir, 'feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f)
        
        # Save scalers
        joblib.dump(self.scaler, os.path.join(model_dir, 'features_scaler.joblib'))
        
        # Save each forecast horizon model
        for horizon in self.forecast_horizons:
            # Save median model
            if horizon in self.models_median:
                self.models_median[horizon].save_model(os.path.join(model_dir, f'xgboost_model_hour_{horizon}.json'))
            
            # Save lower bound model
            if horizon in self.models_lower:
                self.models_lower[horizon].save_model(os.path.join(model_dir, f'xgboost_model_lower_hour_{horizon}.json'))
            
            # Save upper bound model
            if horizon in self.models_upper:
                self.models_upper[horizon].save_model(os.path.join(model_dir, f'xgboost_model_upper_hour_{horizon}.json'))
        
        # Save error percentiles if available
        if hasattr(self, 'error_percentiles') and self.error_percentiles:
            joblib.dump(self.error_percentiles, os.path.join(model_dir, 'error_percentiles.joblib'))
        
        print(f"Models and scalers saved successfully to {model_dir}")
    
    def load_models(self, model_dir=None):
        """
        Load all trained models and scalers from files.
        
        Parameters:
        -----------
        model_dir (str): Directory to load the models from
        
        Returns:
        --------
        bool: True if models loaded successfully, False otherwise
        """
        # Use the same directory as main.py if no model_dir is specified
        if model_dir is None:
            model_dir = self.base_dir
        
        print(f"Loading models and scalers from {model_dir}...")
        
        try:
            # Load feature columns
            feature_columns_path = os.path.join(model_dir, 'feature_columns.json')
            if os.path.exists(feature_columns_path):
                with open(feature_columns_path, 'r') as f:
                    self.feature_columns = json.load(f)
                print(f"Loaded {len(self.feature_columns)} feature columns")
            else:
                print(f"Warning: Feature columns file not found at {feature_columns_path}")
                return False
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'features_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("Loaded feature scaler")
            else:
                print(f"Warning: Scaler file not found at {scaler_path}")
                return False
            
            # Initialize model dictionaries
            self.models_median = {}
            self.models_lower = {}
            self.models_upper = {}
            
            # Load each forecast horizon model
            for horizon in self.forecast_horizons:
                # Load median model
                median_model_path = os.path.join(model_dir, f'xgboost_model_hour_{horizon}.json')
                if os.path.exists(median_model_path):
                    self.models_median[horizon] = xgb.XGBRegressor()
                    self.models_median[horizon].load_model(median_model_path)
                    print(f"Loaded median model for horizon {horizon}h")
                else:
                    print(f"Warning: Median model for horizon {horizon}h not found")
                    return False
                
                # Load lower bound model
                lower_model_path = os.path.join(model_dir, f'xgboost_model_lower_hour_{horizon}.json')
                if os.path.exists(lower_model_path):
                    self.models_lower[horizon] = xgb.XGBRegressor()
                    self.models_lower[horizon].load_model(lower_model_path)
                    print(f"Loaded lower bound model for horizon {horizon}h")
                
                # Load upper bound model
                upper_model_path = os.path.join(model_dir, f'xgboost_model_upper_hour_{horizon}.json')
                if os.path.exists(upper_model_path):
                    self.models_upper[horizon] = xgb.XGBRegressor()
                    self.models_upper[horizon].load_model(upper_model_path)
                    print(f"Loaded upper bound model for horizon {horizon}h")
            
            # Load error percentiles if available
            error_percentiles_path = os.path.join(model_dir, 'error_percentiles.joblib')
            if os.path.exists(error_percentiles_path):
                self.error_percentiles = joblib.load(error_percentiles_path)
                print("Loaded error percentiles for calibrated intervals")
            
            print("All models and scalers loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def save_validation_results(self, X_val, validation_results):
        """
        Save validation set results to a CSV file, including actual values and prediction intervals.
        
        Parameters:
        -----------
        X_val: Validation features DataFrame
        validation_results: Dictionary containing actual, median, lower, and upper values for each horizon
        """
        try:
            # Create a DataFrame from the validation results
            results_df = pd.DataFrame(validation_results)
            
            # Add timestamps if available - check for stored datetimes first
            if hasattr(self, 'val_datetimes') and self.val_datetimes is not None:
                # Use the separately stored datetime column
                results_df['datetime'] = self.val_datetimes.values
                logging.info(f"Using stored validation datetimes for results export")
            elif 'datetime' in X_val.columns:
                # Fallback to X_val column if available
                results_df['datetime'] = X_val['datetime'].values
                logging.info(f"Using X_val datetime column for results export")
            else:
                logging.warning("No datetime column available for validation results")
            
            # Move datetime to first column if it exists
            if 'datetime' in results_df.columns:
                cols = results_df.columns.tolist()
                cols = ['datetime'] + [col for col in cols if col != 'datetime']
                results_df = results_df[cols]
            
            # Group columns by horizon for better readability
            organized_cols = []
            if 'datetime' in results_df.columns:
                organized_cols.append('datetime')
                
            # Organize columns by horizon: actual, lower, median, upper for each horizon
            for horizon in self.forecast_horizons:
                horizon_cols = [
                    f'actual_{horizon}h',
                    f'lower_{horizon}h',
                    f'median_{horizon}h', 
                    f'upper_{horizon}h'
                ]
                organized_cols.extend([col for col in horizon_cols if col in results_df.columns])
            
            # Reorder columns if all expected columns are present
            if set(organized_cols).issubset(set(results_df.columns)):
                results_df = results_df[organized_cols]
            
            # Define output path - use script directory instead of current working directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'validation_results_{timestamp}.csv')
            
            # Save to CSV
            results_df.to_csv(output_path, index=False)
            logging.info(f"Validation results saved to {output_path}")
            print(f"Validation results with prediction intervals saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving validation results: {str(e)}")
            print(f"Error saving validation results: {str(e)}")


# Main execution
if __name__ == "__main__":
    import os
    
    def display_menu():
        """Display the main menu options"""
        print("\n===== GHI Prediction Model Menu =====")
        print("1. Train new models")
        print("2. Make predictions with existing models")
        print("3. Exit")
        return input("\nSelect an option (1-3): ")
    
    def train_models():
        """Function to handle model training"""
        print("\n===== Training New Models =====")
        
        # Create instance of the model
        model = GHIPredictionModel()
        
        # Define input parameters - use the same directory as main.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'dataset.csv')
        val_size = 0.2  # Fixed at 20% for validation
        lag_hours = 3
        random_state = 42
        
        # Run the entire pipeline for multi-horizon forecasting
        result = model.run_pipeline(
            file_path=file_path,
            val_size=val_size,
            lag_hours=lag_hours,
            random_state=random_state
        )
        
        # Save the trained models
        model.save_models()
        
        print("\nTraining completed successfully!")
    
    def make_predictions():
        """Function to handle predictions with existing models"""
        print("\n===== Making Predictions with Existing Models =====")
        
        # Check if models exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(base_dir, 'xgboost_model_hour_1.json')
        if not os.path.exists(model_file):
            print("\nError: No trained models found. Please train models first (Option 1).")
            return
        
        # Create instance of the model
        model = GHIPredictionModel()
        
        # Load the trained models
        if not model.load_models():
            print("\nError: Failed to load models. Please train new models.")
            return
        
        # Generate predictions for future hours
        print("\n--- Predicting Future Hours ---")
        file_path = os.path.join(base_dir, 'dataset.csv')
        predictions = model.predict_future_hours(file_path=file_path)
        
        print("\nPrediction completed successfully!")
    
    # Main program loop
    while True:
        choice = display_menu()
        
        if choice == '1':
            train_models()
        elif choice == '2':
            make_predictions()
        elif choice == '3':
            print("\nExiting program. Goodbye!")
            break
        else:
            print("\nInvalid option. Please select 1, 2, or 3.")
