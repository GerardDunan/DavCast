import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import datetime
import warnings
from datetime import datetime
import traceback
import pickle
import json
import gzip
warnings.filterwarnings('ignore')

# Constants
ADDU_LATITUDE = 7.0711
ADDU_LONGITUDE = 125.6134

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
                    ADDU_LATITUDE, 
                    ADDU_LONGITUDE, 
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

class PredictionErrorLearner:
    def __init__(self):
        self.error_history = {}
        self.adjustment_factors = {}
        self.recent_errors = []
        self.learning_rate = 0.2
        self.pattern_adjustments = {}
        
    def record_error(self, hour, predicted, actual, conditions, error_history=None):
        error = actual - predicted
        error_pct = error / actual if actual != 0 else 0
        
        if hour not in self.error_history:
            self.error_history[hour] = []
            
        self.error_history[hour].append({
            'error': error,
            'error_pct': error_pct,
            'conditions': conditions,
            'predicted': predicted,
            'actual': actual,
            'timestamp': pd.Timestamp.now()
        })
        
        self.recent_errors.append({
            'error_pct': error_pct,
            'timestamp': pd.Timestamp.now(),
            'hour': hour
        })
        
        if len(self.recent_errors) > 20:
            self.recent_errors.pop(0)
            
        self._update_adjustment_factors(hour, error_pct, conditions)
        self._update_pattern_adjustments(hour, predicted, actual, conditions)
        
    def get_adjustment(self, hour, conditions):
        key = f"{hour}"
        base_adjustment = self.adjustment_factors.get(key, 1.0)
        
        if len(self.recent_errors) >= 3:
            weights = np.array([0.5, 0.3, 0.2])
            recent_errors = [e['error_pct'] for e in self.recent_errors[-3:]]
            recent_trend = np.average(recent_errors, weights=weights)
            base_adjustment *= (1 + recent_trend * 0.3)
            
        if key in self.pattern_adjustments:
            patterns = self.pattern_adjustments[key]
            if patterns['consecutive_under'] >= 2:
                base_adjustment *= 1.2
            elif patterns['consecutive_over'] >= 2:
                base_adjustment *= 0.8
                
        return base_adjustment

    def _update_pattern_adjustments(self, hour, predicted, actual, conditions):
        key = f"{hour}"
        if key not in self.pattern_adjustments:
            self.pattern_adjustments[key] = {
                'under_predictions': 0,
                'over_predictions': 0,
                'consecutive_under': 0,
                'consecutive_over': 0
            }
            
        if actual > predicted:
            self.pattern_adjustments[key]['consecutive_under'] += 1
            self.pattern_adjustments[key]['consecutive_over'] = 0
            self.pattern_adjustments[key]['under_predictions'] += 1
        else:
            self.pattern_adjustments[key]['consecutive_over'] += 1
            self.pattern_adjustments[key]['consecutive_under'] = 0
            self.pattern_adjustments[key]['over_predictions'] += 1
            
    def _update_adjustment_factors(self, hour, error_pct, conditions):
        key = f"{hour}"
        if key not in self.adjustment_factors:
            self.adjustment_factors[key] = 1.0
            
        current_factor = self.adjustment_factors[key]
        new_factor = current_factor * (1 + error_pct * self.learning_rate)
        new_factor = max(0.4, min(2.5, new_factor))
        self.adjustment_factors[key] = current_factor * 0.6 + new_factor * 0.4

    def get_hour_adjustments(self, hour):
        """Get all adjustment factors for a specific hour"""
        key = f"{hour}"
        adjustments = {
            'base': self.adjustment_factors.get(key, 1.0),
            'pattern': 1.0,
            'trend': 1.0
        }
        
        # Add pattern-based adjustments
        if key in self.pattern_adjustments:
            patterns = self.pattern_adjustments[key]
            if patterns['consecutive_under'] >= 2:
                adjustments['pattern'] = 1.2
            elif patterns['consecutive_over'] >= 2:
                adjustments['pattern'] = 0.8
        
        # Add trend-based adjustments
        if len(self.recent_errors) >= 3:
            weights = np.array([0.5, 0.3, 0.2])
            recent_errors = [e['error_pct'] for e in self.recent_errors[-3:]]
            recent_trend = np.average(recent_errors, weights=weights)
            adjustments['trend'] = 1 + (recent_trend * 0.3)
        
        return adjustments

class AutomatedPredictor:
    def __init__(self, data_folder='predictions/'):
        # Create all necessary folders
        self.data_folder = data_folder
        self.history_folder = os.path.join(data_folder, 'history')
        self.models_folder = os.path.join(data_folder, 'models')
        self.stats_folder = os.path.join(data_folder, 'learning_stats')  # Changed to learning_stats
        
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.history_folder, exist_ok=True)
        os.makedirs(self.models_folder, exist_ok=True)
        os.makedirs(self.stats_folder, exist_ok=True)
        
        # Define file paths
        self.patterns_file = os.path.join(self.models_folder, 'learned_patterns.pkl')
        self.errors_file = os.path.join(self.history_folder, 'prediction_errors.csv')
        self.stats_file = os.path.join(self.stats_folder, 'hourly_stats.json')
        self.learning_state_file = os.path.join(self.models_folder, 'learning_state.pkl')
        
        # Initialize learning parameters
        self.hourly_patterns = {}
        self.seasonal_patterns = {}
        self.transition_patterns = {}
        self.weather_impacts = {}
        self.error_learner = PredictionErrorLearner()
        self.consecutive_errors = []
        self.prediction_history = []
        
        # Load previous state
        self.load_state()
        
        # Add model versioning
        self.best_model_file = os.path.join(self.models_folder, 'best_model.pkl.gz')
        self.model_performance = {
            'best_mae': float('inf'),
            'best_timestamp': None,
            'current_mae': float('inf'),
            'evaluation_window': 100  # Number of predictions to evaluate performance
        }
        
        # Load best model if exists
        self.load_best_model()

    def save_state(self):
        """Save all learning state and history with compression and retention policies"""
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Compress learning state
            learning_state = {
                'hourly_patterns': self._compress_patterns(self.hourly_patterns),
                'seasonal_patterns': self._compress_patterns(self.seasonal_patterns),
                'transition_patterns': self._compress_patterns(self.transition_patterns),
                'weather_impacts': self.weather_impacts,
                'consecutive_errors': self.consecutive_errors[-10:],  # Keep only last 10
                'error_learner_state': {
                    'error_history': self._compress_error_history(
                        self.error_learner.error_history),
                    'adjustment_factors': self.error_learner.adjustment_factors,
                    'pattern_adjustments': self.error_learner.pattern_adjustments
                },
                'timestamp': timestamp
            }
            
            # 2. Save compressed state
            state_file = os.path.join(self.models_folder, f'learning_state_{timestamp}.pkl.gz')
            with gzip.open(state_file, 'wb') as f:
                pickle.dump(learning_state, f)
            
            # 3. Maintain only last 7 days of history
            self._cleanup_old_files(self.models_folder, 'learning_state_', days=7)
            
            # 4. Save prediction history with aggregation
            if self.prediction_history:
                history_df = pd.DataFrame(self.prediction_history)
                # Aggregate by hour and save last 30 days
                history_df['date'] = pd.to_datetime(history_df['date'])
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=30)
                history_df = history_df[history_df['date'] >= cutoff_date]
                
                # Save compressed CSV
                history_file = os.path.join(
                    self.history_folder, 
                    f'prediction_history_{timestamp}.csv.gz'
                )
                history_df.to_csv(history_file, compression='gzip', index=False)
            
            # 5. Save hourly statistics (keep only essential metrics)
            hourly_stats = self._get_compressed_stats()
            stats_file = os.path.join(self.stats_folder, f'hourly_stats_{timestamp}.json')
            with open(stats_file, 'w') as f:
                json.dump(hourly_stats, f)
            
            # 6. Cleanup old files
            self._cleanup_old_files(self.history_folder, 'prediction_history_', days=30)
            self._cleanup_old_files(self.stats_folder, 'hourly_stats_', days=7)
            
            print(f"\nSaved compressed learning state and history at {timestamp}")
            
        except Exception as e:
            print(f"Error saving state: {str(e)}")
            traceback.print_exc()

    def _compress_patterns(self, patterns):
        """Compress pattern data by removing old entries and aggregating"""
        if isinstance(patterns, dict):
            compressed = {}
            for key, values in patterns.items():
                if isinstance(values, list):
                    # Keep only last 30 days of patterns
                    recent_values = [
                        v for v in values 
                        if (pd.Timestamp.now() - pd.Timestamp(v['date'])).days <= 30
                    ]
                    compressed[key] = recent_values[-100:]  # Keep max 100 entries
                else:
                    compressed[key] = values
            return compressed
        return patterns

    def _compress_error_history(self, error_history):
        """Compress error history by aggregating and limiting retention"""
        compressed = {}
        for hour, errors in error_history.items():
            # Keep only last 7 days of errors
            recent_errors = [
                e for e in errors 
                if (pd.Timestamp.now() - e['timestamp']).days <= 7
            ]
            compressed[hour] = recent_errors[-50:]  # Keep max 50 entries per hour
        return compressed

    def _get_compressed_stats(self):
        """Get compressed version of hourly statistics"""
        stats = {}
        for hour, patterns in self.hourly_patterns.items():
            values = [p['value'] for p in patterns[-50:]]  # Use only last 50 values
            if values:
                stats[str(hour)] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        return stats

    def _cleanup_old_files(self, folder, prefix, days=7):
        """Remove files older than specified days"""
        try:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            
            for filename in os.listdir(folder):
                if filename.startswith(prefix):
                    try:
                        # Extract timestamp from filename (format: learning_state_20241202_071928.pkl.gz)
                        timestamp_str = filename.split(prefix)[1].split('.')[0]  # Get 20241202_071928
                        # Use pd.to_datetime instead of Timestamp.strptime
                        file_date = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
                        
                        if file_date < cutoff_date:
                            file_path = os.path.join(folder, filename)
                            os.remove(file_path)
                            print(f"Removed old file: {filename}")
                    except (IndexError, ValueError) as e:
                        print(f"Skipping file with invalid timestamp format: {filename}")
                        continue
                        
        except Exception as e:
            print(f"Error in _cleanup_old_files: {str(e)}")
            traceback.print_exc()

    def load_state(self):
        """Load previous learning state and history"""
        try:
            if os.path.exists(self.learning_state_file):
                with gzip.open(self.learning_state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.hourly_patterns = state['hourly_patterns']
                    self.seasonal_patterns = state['seasonal_patterns']
                    self.transition_patterns = state['transition_patterns']
                    self.weather_impacts = state['weather_impacts']
                    self.consecutive_errors = state['consecutive_errors']
                    
                    # Restore error learner state
                    error_state = state['error_learner_state']
                    self.error_learner.error_history = error_state['error_history']
                    self.error_learner.adjustment_factors = error_state['adjustment_factors']
                    self.error_learner.pattern_adjustments = error_state['pattern_adjustments']
                
                # Load prediction history
                if os.path.exists(self.errors_file):
                    self.prediction_history = pd.read_csv(self.errors_file).to_dict('records')
                
                print("Loaded previous learning state")
            else:
                print("No previous learning state found - starting fresh")
                
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            print("Starting with fresh learning state")

    def update_with_actual(self, date, hour, actual_value):
        """Update learning with actual value and evaluate model"""
        try:
            updated = False
            # Find the most recent prediction for this hour that hasn't been updated
            for pred in reversed(self.prediction_history):
                if (pred.get('date') == date and 
                    pred.get('hour') == hour and 
                    pred.get('actual') is None):
                    # Update prediction record with actual value
                    pred['actual'] = actual_value
                    pred['error'] = actual_value - pred['predicted']
                    pred['error_percentage'] = (pred['error'] / actual_value * 100) if actual_value != 0 else 0
                    
                    # Update consecutive errors for learning
                    self.consecutive_errors.append(pred['error'])
                    if len(self.consecutive_errors) > 10:
                        self.consecutive_errors.pop(0)
                    
                    # Update error learner with context
                    self.error_learner.record_error(
                        hour, 
                        pred['predicted'], 
                        actual_value, 
                        pred['conditions'],
                        error_history=self.consecutive_errors
                    )
                    
                    # Print detailed learning update
                    print(f"\nLearning Update for Hour {hour:02d}:")
                    print(f"Date: {date}")  # Add date to output
                    print(f"Predicted: {pred['predicted']:.2f} W/m²")
                    print(f"Actual: {actual_value:.2f} W/m²")
                    print(f"Error: {pred['error']:.2f} W/m²")
                    print(f"Error percentage: {pred['error_percentage']:.1f}%")
                    print(f"New adjustment factor: {self.error_learner.get_adjustment(hour, pred['conditions']):.3f}")
                    
                    updated = True
                    break
            
            if not updated:
                print(f"No pending prediction found for date {date}, hour {hour:02d}")
            
            # Evaluate model performance after update
            self.evaluate_and_save_model()
            
            # Save state after update
            self.save_state()
            
        except Exception as e:
            print(f"Error in update_with_actual: {str(e)}")
            traceback.print_exc()

    def evaluate_and_save_model(self):
        """Evaluate current model performance and save if best"""
        try:
            if len(self.prediction_history) >= self.model_performance['evaluation_window']:
                recent_predictions = pd.DataFrame(self.prediction_history[-self.model_performance['evaluation_window']:])
                current_mae = np.mean(np.abs(recent_predictions['error']))
                self.model_performance['current_mae'] = current_mae
                
                if current_mae < self.model_performance['best_mae']:
                    print(f"\nNew best model! (MAE: {current_mae:.2f} vs previous: {self.model_performance['best_mae']:.2f})")
                    self.model_performance['best_mae'] = current_mae
                    self.model_performance['best_timestamp'] = pd.Timestamp.now()
                    
                    # Save best model state
                    best_state = {
                        'hourly_patterns': self.hourly_patterns,
                        'seasonal_patterns': self.seasonal_patterns,
                        'transition_patterns': self.transition_patterns,
                        'weather_impacts': self.weather_impacts,
                        'error_learner': self.error_learner,
                        'performance': self.model_performance,
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    with gzip.open(self.best_model_file, 'wb') as f:
                        pickle.dump(best_state, f)
                    print("Saved new best model")
                    
        except Exception as e:
            print(f"Error in evaluate_and_save_model: {str(e)}")

    def load_best_model(self):
        """Load the best performing model"""
        try:
            if os.path.exists(self.best_model_file):
                with gzip.open(self.best_model_file, 'rb') as f:
                    best_state = pickle.load(f)
                    self.hourly_patterns = best_state['hourly_patterns']
                    self.seasonal_patterns = best_state['seasonal_patterns']
                    self.transition_patterns = best_state['transition_patterns']
                    self.weather_impacts = best_state['weather_impacts']
                    self.error_learner = best_state['error_learner']
                    self.model_performance = best_state['performance']
                print(f"Loaded best model (MAE: {self.model_performance['best_mae']:.2f})")
            else:
                print("No best model found - starting fresh")
        except Exception as e:
            print(f"Error loading best model: {str(e)}")

    def learn_from_historical_data(self, data):
        """Learn patterns from all historical data"""
        print("\nLearning from historical data...")
        
        # Group data by date
        dates = data['timestamp'].dt.date.unique()
        total_dates = len(dates)
        
        for i, date in enumerate(dates):
            day_data = data[data['timestamp'].dt.date == date]
            
            # Skip incomplete days
            if len(day_data) < 24:
                continue
                
            print(f"\rProcessing historical data: {i+1}/{total_dates} days", end='')
            
            # Learn hourly patterns
            for hour in range(24):
                hour_data = day_data[day_data['timestamp'].dt.hour == hour]
                if hour_data.empty:
                    continue
                
                value = hour_data['Solar Rad - W/m^2'].iloc[0]
                weather_conditions = {
                    'temperature': hour_data['Average Temperature'].iloc[0],
                    'humidity': hour_data['Average Humidity'].iloc[0],
                    'pressure': hour_data['Average Barometer'].iloc[0],
                    'uv': hour_data['UV Index'].iloc[0]
                }
                
                # Store hourly pattern
                if hour not in self.hourly_patterns:
                    self.hourly_patterns[hour] = []
                self.hourly_patterns[hour].append({
                    'value': value,
                    'conditions': weather_conditions,
                    'date': date
                })
                
                # Learn hour-to-hour transitions
                if hour < 23:
                    next_hour = hour + 1
                    next_data = day_data[day_data['timestamp'].dt.hour == next_hour]
                    if not next_data.empty:
                        next_value = next_data['Solar Rad - W/m^2'].iloc[0]
                        transition_key = f"{hour}-{next_hour}"
                        if transition_key not in self.transition_patterns:
                            self.transition_patterns[transition_key] = []
                        self.transition_patterns[transition_key].append({
                            'from_value': value,
                            'to_value': next_value,
                            'conditions': weather_conditions,
                            'date': date
                        })
            
            # Learn seasonal patterns
            season = self._get_season(date)
            if season not in self.seasonal_patterns:
                self.seasonal_patterns[season] = []
            self.seasonal_patterns[season].append({
                'date': date,
                'pattern': day_data['Solar Rad - W/m^2'].values,
                'weather': weather_conditions
            })
        
        print("\nAnalyzing learned patterns...")
        self._analyze_patterns()
        self.save_state()

    def _analyze_patterns(self):
        """Analyze learned patterns to extract insights"""
        # Analyze hourly patterns
        hourly_stats = {}
        for hour, patterns in self.hourly_patterns.items():
            values = [p['value'] for p in patterns]
            hourly_stats[hour] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Print statistics
        print("\nLearned Pattern Statistics:")
        for hour in range(24):
            if hour in hourly_stats:
                stats = hourly_stats[hour]
                print(f"\nHour {hour:02d}:00")
                print(f"Mean: {stats['mean']:.2f} W/m²")
                print(f"Std: {stats['std']:.2f} W/m²")
                print(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}] W/m²")

    def _get_season(self, date):
        """Get season for a given date in the Philippines"""
        month = date.month
        # Wet season: June to November
        # Dry season: December to May
        if month >= 6 and month <= 11:
            return 'wet'
        else:
            return 'dry'

    def get_hourly_stats(self):
        """Calculate statistics for hourly patterns"""
        stats = {}
        for hour, patterns in self.hourly_patterns.items():
            values = [p['value'] for p in patterns]
            if values:
                stats[str(hour)] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        return stats

    def predict_next_hour(self, data, target_date, current_hour):
        """Make prediction for the next hour"""
        try:
            current_hour = int(current_hour)
            next_hour = (current_hour + 1) % 24
            
            # Get current day's data
            current_day = data[data['timestamp'].dt.date == target_date]
            if current_day.empty:
                raise ValueError("No data available for target date")
            
            # Get current value and conditions
            current_data = current_day[
                current_day['timestamp'].dt.hour == current_hour
            ]
            
            if current_data.empty:
                raise ValueError(f"No data available for hour {current_hour}")
                
            current_value = current_data['Solar Rad - W/m^2'].iloc[0]
            conditions = {
                'temperature': current_data['Average Temperature'].iloc[0],
                'humidity': current_data['Average Humidity'].iloc[0],
                'pressure': current_data['Average Barometer'].iloc[0],
                'uv': current_data['UV Index'].iloc[0]
            }
            
            # Find similar days
            similar_days = self._find_similar_days(data, target_date, current_hour, conditions)
            
            # Calculate multiple prediction components
            predictions = {
                'pattern_based': self._get_pattern_prediction(similar_days, current_value),
                'ratio_based': self._get_ratio_prediction(similar_days, current_value),
                'trend_based': self._get_trend_prediction(current_day, current_hour),
                'typical_value': self._get_typical_value(next_hour)
            }
            
            # Get clear sky radiation
            clear_sky = calculate_clear_sky_radiation(
                next_hour, ADDU_LATITUDE, ADDU_LONGITUDE, target_date
            )
            
            # Dynamic weighting based on recent performance
            weights = self._calculate_prediction_weights()
            
            # Combine predictions with weights
            prediction = (
                predictions['pattern_based'] * weights['pattern'] +
                predictions['ratio_based'] * weights['ratio'] +
                predictions['trend_based'] * weights['trend'] +
                predictions['typical_value'] * weights['typical']
            )
            
            # Apply error learning adjustment
            adjustment = self.error_learner.get_adjustment(next_hour, conditions)
            prediction = prediction * adjustment
            
            # Validate against clear sky and recent values
            prediction = self._validate_prediction(prediction, clear_sky, current_value)
            
            # Print detailed analysis
            print(f"\nPrediction Analysis for hour {next_hour:02d}:00")
            print(f"Clear sky radiation: {clear_sky:.2f} W/m²")
            print(f"Current value: {current_value:.2f} W/m²")
            print(f"Pattern-based: {predictions['pattern_based']:.2f} W/m²")
            print(f"Ratio-based: {predictions['ratio_based']:.2f} W/m²")
            print(f"Trend-based: {predictions['trend_based']:.2f} W/m²")
            print(f"Typical value: {predictions['typical_value']:.2f} W/m²")
            print(f"Applied learning adjustment: {adjustment:.3f}")
            print(f"Final prediction: {prediction:.2f} W/m²")
            
            # Store prediction for learning
            self.prediction_history.append({
                'date': target_date,
                'hour': next_hour,
                'predicted': prediction,
                'actual': None,  # Initialize as None
                'error': None,   # Initialize as None
                'error_percentage': None,  # Initialize as None
                'conditions': conditions,
                'weights': weights,
                'adjustment': adjustment
            })
            
            return prediction
            
        except Exception as e:
            print(f"Error in predict_next_hour: {str(e)}")
            traceback.print_exc()
            return None

    def _find_similar_days(self, data, target_date, current_hour, conditions):
        """Find similar days with enhanced matching criteria"""
        try:
            similar_days = []
            
            # Get historical data excluding current day
            historical_data = data[data['timestamp'].dt.date < target_date].copy()
            
            for date in historical_data['timestamp'].dt.date.unique():
                day_data = historical_data[historical_data['timestamp'].dt.date == date]
                
                if len(day_data) < 24:
                    continue
                
                # Calculate similarity scores
                weather_sim = self._calculate_weather_similarity(day_data, conditions)
                pattern_sim = self._calculate_pattern_similarity(day_data, current_hour)
                
                if weather_sim > 0.7 and pattern_sim > 0.7:  # Stricter similarity threshold
                    similar_days.append({
                        'date': date,
                        'data': day_data,
                        'similarity': (weather_sim + pattern_sim) / 2
                    })
            
            return sorted(similar_days, key=lambda x: x['similarity'], reverse=True)[:5]
            
        except Exception as e:
            print(f"Error in _find_similar_days: {str(e)}")
            return []

    def _calculate_weather_similarity(self, day_data, conditions):
        """Calculate weather condition similarity"""
        try:
            # Get average conditions for the day
            day_conditions = {
                'temperature': day_data['Average Temperature'].mean(),
                'humidity': day_data['Average Humidity'].mean(),
                'pressure': day_data['Average Barometer'].mean(),
                'uv': day_data['UV Index'].mean()
            }
            
            # Calculate similarity for each condition
            temp_sim = 1 - min(abs(day_conditions['temperature'] - conditions['temperature']) / 10, 1)
            humidity_sim = 1 - min(abs(day_conditions['humidity'] - conditions['humidity']) / 30, 1)
            pressure_sim = 1 - min(abs(day_conditions['pressure'] - conditions['pressure']) / 10, 1)
            uv_sim = 1 - min(abs(day_conditions['uv'] - conditions['uv']) / 5, 1)
            
            # Weighted combination
            similarity = (
                temp_sim * 0.3 +
                humidity_sim * 0.3 +
                pressure_sim * 0.2 +
                uv_sim * 0.2
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error in _calculate_weather_similarity: {str(e)}")
            return 0.0

    def _calculate_pattern_similarity(self, day_data, current_hour):
        """Calculate pattern similarity up to current hour"""
        try:
            pattern = day_data[
                day_data['timestamp'].dt.hour <= current_hour
            ]['Solar Rad - W/m^2'].values
            
            if len(pattern) < 2:
                return 0.0
            
            mean_val = np.mean(pattern)
            std_val = np.std(pattern)
            trend = np.polyfit(np.arange(len(pattern)), pattern, 1)[0]
            
            mean_sim = 1 - min(abs(mean_val - pattern[-1]) / (pattern[-1] + 1e-8), 1)
            std_sim = 1 - min(std_val / (mean_val + 1e-8), 1)
            trend_sim = 1 - min(abs(trend) / 100, 1)
            
            similarity = (
                mean_sim * 0.4 +
                std_sim * 0.3 +
                trend_sim * 0.3
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error in _calculate_pattern_similarity: {str(e)}")
            return 0.0

    def _get_pattern_prediction(self, similar_days, current_value):
        """Get prediction based on similar day patterns"""
        try:
            if not similar_days:
                return current_value
            
            predictions = []
            weights = []
            
            for day in similar_days:
                day_data = day['data']
                next_hour_data = day_data['Solar Rad - W/m^2'].values
                if len(next_hour_data) > 0:
                    predictions.append(next_hour_data[-1])
                    weights.append(day['similarity'])
            
            if predictions:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                return float(np.average(predictions, weights=weights))
            
            return current_value
            
        except Exception as e:
            print(f"Error in _get_pattern_prediction: {str(e)}")
            return current_value

    def _get_ratio_prediction(self, similar_days, current_value):
        """Get prediction based on similar day ratios"""
        try:
            if not similar_days:
                return current_value
            
            ratios = []
            weights = []
            
            for day in similar_days:
                day_data = day['data']
                values = day_data['Solar Rad - W/m^2'].values
                if len(values) >= 2:
                    ratio = values[-1] / (values[-2] + 1e-8)
                    ratios.append(ratio)
                    weights.append(day['similarity'])
            
            if ratios:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                avg_ratio = np.average(ratios, weights=weights)
                return float(current_value * avg_ratio)
            
            return current_value
            
        except Exception as e:
            print(f"Error in _get_ratio_prediction: {str(e)}")
            return current_value

    def _get_trend_prediction(self, current_day, current_hour):
        """Get prediction based on current day trend"""
        try:
            data = current_day[
                current_day['timestamp'].dt.hour <= current_hour
            ]['Solar Rad - W/m^2'].values
            
            if len(data) >= 2:
                x = np.arange(len(data))
                trend = np.polyfit(x, data, 1)[0]
                next_value = data[-1] + trend
                return max(0, float(next_value))
            
            return data[-1] if len(data) > 0 else 0
            
        except Exception as e:
            print(f"Error in _get_trend_prediction: {str(e)}")
            return 0

    def _get_typical_value(self, hour):
        """Get typical value for given hour"""
        try:
            if hour in self.hourly_patterns:
                values = [p['value'] for p in self.hourly_patterns[hour]]
                if values:
                    return float(np.mean(values))
            return 0
            
        except Exception as e:
            print(f"Error in _get_typical_value: {str(e)}")
            return 0

    def _calculate_prediction_weights(self):
        """Calculate dynamic weights with learning-based adjustments"""
        if not self.consecutive_errors:
            return {'pattern': 0.3, 'ratio': 0.3, 'trend': 0.2, 'typical': 0.2}
        
        # Calculate error statistics
        recent_errors = np.array(self.consecutive_errors[-5:])
        error_std = np.std(recent_errors) if len(recent_errors) > 1 else 0
        
        # Base weights based on error patterns
        if error_std < 50:  # Stable predictions
            weights = {'pattern': 0.4, 'ratio': 0.3, 'trend': 0.2, 'typical': 0.1}
        elif error_std < 100:  # Moderate variability
            weights = {'pattern': 0.3, 'ratio': 0.3, 'trend': 0.3, 'typical': 0.1}
        else:  # High variability
            weights = {'pattern': 0.2, 'ratio': 0.2, 'trend': 0.4, 'typical': 0.2}
        
        # Apply learning-based adjustments
        if self.prediction_history:
            df = pd.DataFrame(self.prediction_history)
            if len(df) >= 100:
                recent_error = df['error'].tail(100).abs().mean()
                if recent_error > 50:  # High error rate
                    weights['pattern'] *= 0.9
                    weights['trend'] *= 1.1
        
        return weights

    def _validate_prediction(self, prediction, clear_sky, current_value):
        """Validate and adjust prediction"""
        try:
            # Get current hour
            hour = pd.Timestamp.now().hour
            
            # Force zero radiation during night hours (18:00-05:00)
            if hour >= 18 or hour <= 5:  # Changed from hour < 6 to hour <= 5
                return 0.0
            
            # Early morning transition (06:00-07:59)
            if hour < 8:
                # Gradual increase from zero
                if hour == 6:
                    return min(prediction, clear_sky * 0.15)  # Max 15% of clear sky
                elif hour == 7:
                    return min(prediction, clear_sky * 0.3)  # Max 30% of clear sky
            
            # Late afternoon transition (16:00-17:59)
            if hour >= 16:
                # Gradual decrease to zero
                if hour == 17:
                    return min(prediction, clear_sky * 0.1)  # Max 10% of clear sky
                elif hour == 16:
                    return min(prediction, clear_sky * 0.25)  # Max 25% of clear sky
            
            # Normal daytime hours (08:00-15:59)
            # Ensure prediction is within reasonable bounds
            prediction = max(0, min(prediction, clear_sky * 0.85))
            
            # Limit maximum change from current value
            max_increase = current_value * 2.0  # Maximum 100% increase
            max_decrease = current_value * 0.3  # Maximum 70% decrease
            prediction = min(max(prediction, max_decrease), max_increase)
            
            return prediction
            
        except Exception as e:
            print(f"Error in _validate_prediction: {str(e)}")
            return current_value  # Return current value as fallback

    def _store_prediction(self, current_value, prediction, next_hour, conditions):
        """Store prediction for learning"""
        try:
            transition_key = f"{next_hour-1}-{next_hour}"
            if transition_key not in self.transition_patterns:
                self.transition_patterns[transition_key] = []
                
            # Store prediction with timestamp and conditions
            self.transition_patterns[transition_key].append({
                'from_value': current_value,
                'to_value': prediction,
                'conditions': conditions,
                'timestamp': pd.Timestamp.now(),
                'error': None  # Will be updated when actual value is available
            })
            
            # Also store in prediction history
            self.prediction_history.append({
                'hour': next_hour,
                'current_value': current_value,
                'predicted_value': prediction,
                'conditions': str(conditions),
                'timestamp': pd.Timestamp.now()
            })
            
        except Exception as e:
            print(f"Error in _store_prediction: {str(e)}")
            traceback.print_exc()

    def analyze_learning_performance(self, window_size=100):
        """Analyze and visualize learning performance over time"""
        try:
            if not self.prediction_history:
                print("No prediction history available for analysis")
                return
                
            # Convert history to DataFrame
            history_df = pd.DataFrame(self.prediction_history)
            history_df['date'] = pd.to_datetime(history_df['date'])
            
            # Calculate rolling metrics
            history_df['rolling_mae'] = history_df['error'].abs().rolling(window_size).mean()
            history_df['rolling_mape'] = history_df['error_percentage'].abs().rolling(window_size).mean()
            history_df['improvement'] = history_df['rolling_mae'].diff().rolling(window_size).mean()
            
            # Clear any existing plots
            plt.close('all')
            
            # Create figure with better memory management
            fig = plt.figure(figsize=(20, 15), dpi=100)  # Reduced DPI
            # Use a built-in style instead of seaborn
            plt.style.use('default')  # or try 'classic', 'bmh', 'ggplot'
            
            # Plot 1: Prediction Accuracy Over Time
            ax1 = fig.add_subplot(221)
            ax1.plot(history_df['date'], history_df['rolling_mae'], 
                    'b-', linewidth=2, label='Rolling MAE')
            ax1.set_title('Learning Progress - Error Reduction', fontsize=14, pad=20)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Mean Absolute Error (W/m²)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Error Distribution Changes
            ax2 = fig.add_subplot(222)
            recent_errors = history_df['error'].tail(window_size)
            old_errors = history_df['error'].head(window_size)
            ax2.hist([old_errors, recent_errors], 
                    label=['Initial Errors', 'Recent Errors'], 
                    alpha=0.7, bins=30, color=['red', 'green'])
            ax2.set_title('Error Distribution Improvement', fontsize=14, pad=20)
            ax2.set_xlabel('Prediction Error (W/m²)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # Plot 3: Hourly Performance Improvement
            ax3 = fig.add_subplot(223)
            hourly_improvement = history_df.groupby('hour')['error'].agg(['mean', 'std']).round(2)
            bars = ax3.bar(range(len(hourly_improvement)), 
                          hourly_improvement['mean'],
                          yerr=hourly_improvement['std'],
                          capsize=5,
                          color='skyblue',
                          error_kw={'ecolor': 'gray'})
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}',
                        ha='center', va='bottom')
            
            ax3.set_title('Hourly Prediction Performance', fontsize=14, pad=20)
            ax3.set_xlabel('Hour of Day', fontsize=12)
            ax3.set_ylabel('Mean Error (W/m²)', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(range(24))
            ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Learning Rate
            ax4 = fig.add_subplot(224)
            ax4.plot(history_df['date'], history_df['improvement'], 
                    'g-', linewidth=2)
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax4.set_title('Learning Rate (Negative is Better)', fontsize=14, pad=20)
            ax4.set_xlabel('Date', fontsize=12)
            ax4.set_ylabel('Error Change Rate', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
            
            # Adjust layout and save
            plt.tight_layout(pad=3.0)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.stats_folder, f'learning_analysis_{timestamp}.png')
            
            # Save with reduced memory usage
            plt.savefig(plot_path, bbox_inches='tight', dpi=100)
            plt.close(fig)  # Explicitly close the figure
            
            # Generate detailed learning report
            self._generate_learning_report(history_df, hourly_improvement)
            
        except Exception as e:
            print(f"Error in analyze_learning_performance: {str(e)}")
            traceback.print_exc()

    def _generate_learning_report(self, history_df, hourly_stats):
        """Generate detailed learning report"""
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.stats_folder, f'learning_report_{timestamp}.txt')
            
            with open(report_path, 'w') as f:
                f.write("=== Solar Radiation Prediction Learning Report ===\n\n")
                
                # Overall Performance Metrics
                f.write("Overall Performance:\n")
                f.write(f"Total Predictions: {len(history_df)}\n")
                f.write(f"Average Error: {history_df['error'].mean():.2f} W/m²\n")
                f.write(f"Error Standard Deviation: {history_df['error'].std():.2f} W/m²\n")
                f.write(f"Mean Absolute Percentage Error: {history_df['error_percentage'].abs().mean():.2f}%\n\n")
                
                # Recent Performance (last 100 predictions)
                recent = history_df.tail(100)
                f.write("Recent Performance (Last 100 Predictions):\n")
                f.write(f"Average Error: {recent['error'].mean():.2f} W/m²\n")
                f.write(f"Error Standard Deviation: {recent['error'].std():.2f} W/m²\n")
                f.write(f"MAPE: {recent['error_percentage'].abs().mean():.2f}%\n\n")
                
                # Hourly Analysis
                f.write("Hourly Performance:\n")
                for hour, stats in hourly_stats.iterrows():
                    f.write(f"Hour {hour:02d}:00\n")
                    f.write(f"  Mean Error: {stats['mean']:.2f} W/m²\n")
                    f.write(f"  Error Std: {stats['std']:.2f} W/m²\n")
                
                # Learning Adjustments
                f.write("\nCurrent Learning Parameters:\n")
                for hour in range(24):
                    adjustments = self.error_learner.get_hour_adjustments(hour)
                    f.write(f"\nHour {hour:02d}:00 Adjustments:\n")
                    for condition, factor in adjustments.items():
                        f.write(f"  {condition}: {factor:.3f}\n")
                
                # Recommendations
                f.write("\nSystem Recommendations:\n")
                self._generate_recommendations(history_df, f)
                
            print(f"\nLearning analysis saved to {report_path}")
            
        except Exception as e:
            print(f"Error in _generate_learning_report: {str(e)}")
            traceback.print_exc()

    def _generate_recommendations(self, history_df, f):
        """Generate system recommendations based on learning analysis"""
        try:
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns(history_df)
            
            # Write recommendations
            f.write("\nBased on error analysis:\n")
            
            # Check for systematic bias
            if abs(history_df['error'].mean()) > 50:
                bias_direction = "high" if history_df['error'].mean() < 0 else "low"
                f.write(f"- System shows systematic bias towards {bias_direction} predictions\n")
                f.write(f"  Recommendation: Adjust base prediction weights\n")
            
            # Check for hour-specific issues
            problematic_hours = error_patterns['problematic_hours']
            if problematic_hours:
                f.write("\nHours needing attention:\n")
                for hour, error_rate in problematic_hours:
                    f.write(f"- Hour {hour:02d}:00 (Error rate: {error_rate:.2f}%)\n")
                    f.write("  Recommendation: Review and adjust hourly patterns\n")
            
            # Weather condition impacts
            if error_patterns['weather_impacts']:
                f.write("\nWeather condition impacts:\n")
                for condition, impact in error_patterns['weather_impacts'].items():
                    f.write(f"- {condition}: {impact:.2f}% average error\n")
                    if abs(impact) > 20:
                        f.write("  Recommendation: Adjust weather similarity calculations\n")
            
            # Learning rate analysis
            recent_improvement = error_patterns['learning_rate']
            if recent_improvement > -1:  # Not improving fast enough
                f.write("\nLearning rate recommendations:\n")
                f.write("- Consider increasing learning rate for faster adaptation\n")
                f.write("- Review similarity thresholds for pattern matching\n")
            
        except Exception as e:
            print(f"Error in _generate_recommendations: {str(e)}")
            f.write("\nError generating recommendations\n")

    def _analyze_error_patterns(self, history_df):
        """Analyze error patterns in prediction history"""
        try:
            patterns = {
                'problematic_hours': [],
                'weather_impacts': {},
                'learning_rate': 0.0
            }
            
            # Analyze hourly performance
            hourly_errors = history_df.groupby('hour')['error_percentage'].agg(['mean', 'std'])
            for hour, stats in hourly_errors.iterrows():
                if abs(stats['mean']) > 15:  # More than 15% average error
                    patterns['problematic_hours'].append((hour, abs(stats['mean'])))
            
            # Analyze weather impacts if weather data is available
            if 'weather_condition' in history_df.columns:
                weather_impacts = history_df.groupby('weather_condition')['error_percentage'].mean()
                patterns['weather_impacts'] = weather_impacts.to_dict()
            
            # Calculate learning rate (improvement over time)
            if len(history_df) > 100:
                recent_errors = history_df['error'].tail(100).abs().mean()
                old_errors = history_df['error'].head(100).abs().mean()
                patterns['learning_rate'] = (recent_errors - old_errors) / old_errors * 100
            
            return patterns
            
        except Exception as e:
            print(f"Error in _analyze_error_patterns: {str(e)}")
            return {'problematic_hours': [], 'weather_impacts': {}, 'learning_rate': 0.0}

    def record_prediction(self, hour, predicted, actual, conditions):
        """Record prediction results for learning analysis"""
        try:
            # Initialize prediction record with None for actual value
            prediction_record = {
                'date': pd.Timestamp.now(),
                'hour': hour,
                'predicted': predicted,
                'actual': None,  # Initialize as None, will be updated later
                'error': None,   # Initialize as None, will be updated later
                'error_percentage': None,  # Initialize as None, will be updated later
                'conditions': conditions
            }
            
            # If actual value is provided immediately, update the record
            if actual is not None:
                prediction_record['actual'] = actual
                prediction_record['error'] = actual - predicted
                prediction_record['error_percentage'] = (prediction_record['error'] / actual *100) if actual !=0 else 0
            
            self.prediction_history.append(prediction_record)
            
            # Trigger learning analysis every 24 predictions
            if len(self.prediction_history) % 24 == 0:
                self.analyze_learning_performance()
                
        except Exception as e:
            print(f"Error in record_prediction: {str(e)}")

    def predict_day(self, date, features):
        """Predict solar radiation for all hours in a day"""
        try:
            predictions = []
            timestamps = []
            
            for hour in range(24):
                # Create timestamp
                timestamp = pd.Timestamp.combine(date, pd.Timestamp(f"{hour:02d}:00").time())
                
                # Get current conditions
                current_conditions = {
                    'temperature': features['Average Temperature'][hour] if hour < len(features) else None,
                    'humidity': features['Average Humidity'][hour] if hour < len(features) else None,
                    'uv': features['UV Index'][hour] if hour < len(features) else None
                }
                
                # Make prediction for this hour
                prediction = self.predict_next_hour(features[:hour+1], hour, current_conditions)
                
                predictions.append(prediction)
                timestamps.append(timestamp)
            
            # Create DataFrame with predictions
            results_df = pd.DataFrame({
                'Timestamp': timestamps,
                'Hour': [t.hour for t in timestamps],
                'Predicted Solar Radiation (W/m²)': predictions
            })
            
            # Save predictions to CSV
            results_df.to_csv(os.path.join(self.stats_folder, 'daily_predictions.csv'), index=False)
            
            # Create prediction plot
            plt.figure(figsize=(12, 6))
            
            # Ensure values are finite before plotting
            hour_data = results_df['Hour'].values
            pred_data = results_df['Predicted Solar Radiation (W/m²)'].values
            
            # Replace any non-finite values with 0
            pred_data = np.nan_to_num(pred_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Plot predictions
            plt.plot(hour_data, pred_data, 
                    marker='o', linestyle='-', linewidth=2, label='Predicted',
                    color='blue')
            
            # Plot actual values where available
            actual_mask = results_df['Actual'].notna()
            if actual_mask.any():
                actual_data = results_df.loc[actual_mask, 'Actual'].values
                # Replace any non-finite values with 0
                actual_data = np.nan_to_num(actual_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                plt.plot(results_df.loc[actual_mask, 'Hour'], 
                        actual_data,
                        marker='s', linestyle='--', linewidth=2, label='Actual',
                        color='green')
            
            plt.title('Solar Radiation Predictions vs Actual Values')
            plt.xlabel('Hour of Day')
            plt.ylabel('Solar Radiation (W/m²)')
            plt.grid(True)
            plt.xticks(range(24))
            plt.ylim(bottom=0)
            plt.legend()
            
            # Add value annotations with validation
            for idx, row in results_df.iterrows():
                pred_val = row['Predicted']
                if np.isfinite(pred_val):
                    plt.annotate(f"{pred_val:.0f}", 
                                (row['Hour'], pred_val),
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center')
                if pd.notna(row['Actual']):
                    actual_val = row['Actual']
                    if np.isfinite(actual_val):
                        plt.annotate(f"{actual_val:.0f}", 
                                    (row['Hour'], actual_val),
                                    textcoords="offset points", 
                                    xytext=(0,-15), 
                                    ha='center',
                                    color='green')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.stats_folder, 'daily_predictions.png'))
            plt.close()
            
            return results_df
            
        except Exception as e:
            print(f"Error in predict_day: {str(e)}")
            traceback.print_exc()
            return None

    def _get_dynamic_learning_rate(self, hour, error_history):
        """Calculate dynamic learning rate based on hour and recent performance"""
        try:
            # Base learning rates for different periods
            base_rates = {
                'early_morning': 0.4,  # 6-9 hours
                'peak_hours': 0.3,     # 10-14 hours
                'afternoon': 0.35,     # 15-17 hours
                'other': 0.2
            }
            
            # Determine period
            if 6 <= hour <= 9:
                base_rate = base_rates['early_morning']
            elif 10 <= hour <= 14:
                base_rate = base_rates['peak_hours']
            elif 15 <= hour <= 17:
                base_rate = base_rates['afternoon']
            else:
                base_rate = base_rates['other']
                
            # Adjust based on recent error patterns
            if error_history:
                recent_errors = error_history[-5:]  # Last 5 errors
                error_std = np.std(recent_errors)
                
                # Increase learning rate if errors are consistently high
                if error_std > 50:  # High variability
                    base_rate *= 1.5
                elif np.mean(np.abs(recent_errors)) > 75:  # Large errors
                    base_rate *= 1.3
                
            return min(base_rate, 0.5)  # Cap at 0.5 to prevent instability
            
        except Exception as e:
            print(f"Error in _get_dynamic_learning_rate: {str(e)}")
            return 0.2  # Default learning rate

    def _calculate_weather_impact(self, conditions, hour):
        """Calculate enhanced weather impact factor"""
        try:
            impact = 1.0
            
            # Enhanced weighting for peak hours (10-14)
            is_peak_hour = 10 <= hour <= 14
            
            # Temperature impact
            temp = conditions.get('temperature', 25)
            if is_peak_hour:
                temp_impact = 1 - abs(temp - 30) / 50  # Optimal temp around 30°C
                impact *= temp_impact * 1.5  # Higher weight during peak hours
            else:
                temp_impact = 1 - abs(temp - 25) / 40
                impact *= temp_impact
                
            # Humidity impact
            humidity = conditions.get('humidity', 70)
            if is_peak_hour:
                humidity_impact = 1 - (humidity / 100) * 0.8  # More sensitive to humidity
            else:
                humidity_impact = 1 - (humidity / 100) * 0.6
            impact *= humidity_impact
            
            # UV index impact
            uv = conditions.get('uv', 5)
            expected_uv = self._get_expected_uv_for_hour(hour)
            uv_ratio = min(uv / expected_uv if expected_uv > 0 else 1, 1.5)
            impact *= uv_ratio
            
            return max(0.1, min(impact, 1.5))  # Limit impact range
            
        except Exception as e:
            print(f"Error in _calculate_weather_impact: {str(e)}")
            return 1.0

    def _find_similar_patterns(self, current_pattern, hour, max_patterns=5):
        """Find similar patterns with enhanced time-based weighting"""
        try:
            similar_patterns = []
            current_date = pd.Timestamp.now().date()
            
            for date, pattern in self.hourly_patterns.get(hour, []):
                if date == current_date:
                    continue
                    
                # Calculate pattern similarity
                pattern_similarity = self._calculate_pattern_similarity(
                    current_pattern, pattern['values'])
                    
                # Calculate time-based weight
                days_diff = (current_date - date).days
                time_weight = np.exp(-days_diff / 30)  # Exponential decay
                
                # Calculate weather similarity
                weather_similarity = self._calculate_weather_similarity(
                    pattern['conditions'], 
                    self.current_conditions
                )
                
                # Combined similarity score
                total_similarity = (
                    pattern_similarity * 0.4 +
                    weather_similarity * 0.4 +
                    time_weight * 0.2
                )
                
                similar_patterns.append({
                    'date': date,
                    'pattern': pattern,
                    'similarity': total_similarity
                })
            
            # Sort and return top patterns
            similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_patterns[:max_patterns]
            
        except Exception as e:
            print(f"Error in _find_similar_patterns: {str(e)}")
            return []

    def _validate_prediction_with_outlier_detection(self, prediction, hour, current_value):
        """Validate prediction with enhanced outlier detection"""
        try:
            # Get historical statistics for this hour
            hour_stats = self._get_hour_statistics(hour)
            if not hour_stats:
                return prediction
                
            mean_value = hour_stats['mean']
            std_value = hour_stats['std']
            
            # Calculate z-score
            z_score = abs(prediction - mean_value) / (std_value + 1e-8)
            
            # Handle potential outliers
            if z_score > 3:  # More than 3 standard deviations
                print(f"Potential outlier detected: {prediction:.2f} W/m²")
                # Use weighted average with historical mean
                adjusted_prediction = (
                    prediction * 0.3 +
                    mean_value * 0.4 +
                    current_value * 0.3
                )
                print(f"Adjusted to: {adjusted_prediction:.2f} W/m²")
                return adjusted_prediction
                
            return prediction
            
        except Exception as e:
            print(f"Error in _validate_prediction_with_outlier_detection: {str(e)}")
            return prediction

def save_results(results_df, plot_path, csv_path):
    """Save results with visualization"""
    try:
        # Save to CSV
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved predictions to {csv_path}")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Ensure values are finite before plotting
        hour_data = results_df['Hour'].values
        pred_data = results_df['Predicted Solar Radiation (W/m²)'].values
        
        # Replace any non-finite values with 0
        pred_data = np.nan_to_num(pred_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Plot predictions
        plt.plot(hour_data, pred_data, 
                marker='o', linestyle='-', linewidth=2, label='Predicted',
                color='blue')
        
        # Plot actual values where available
        actual_mask = results_df['Actual'].notna()
        if actual_mask.any():
            actual_data = results_df.loc[actual_mask, 'Actual'].values
            # Replace any non-finite values with 0
            actual_data = np.nan_to_num(actual_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            plt.plot(results_df.loc[actual_mask, 'Hour'], 
                    actual_data,
                    marker='s', linestyle='--', linewidth=2, label='Actual',
                    color='green')
        
        plt.title('Solar Radiation Predictions vs Actual Values')
        plt.xlabel('Hour of Day')
        plt.ylabel('Solar Radiation (W/m²)')
        plt.grid(True)
        plt.xticks(range(24))
        plt.ylim(bottom=0)
        plt.legend()
        
        # Add value annotations with validation
        for idx, row in results_df.iterrows():
            pred_val = row['Predicted']
            if np.isfinite(pred_val):
                plt.annotate(f"{pred_val:.0f}", 
                            (row['Hour'], pred_val),
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center')
            if pd.notna(row['Actual']):
                actual_val = row['Actual']
                if np.isfinite(actual_val):
                    plt.annotate(f"{actual_val:.0f}", 
                                (row['Hour'], actual_val),
                                textcoords="offset points", 
                                xytext=(0,-15), 
                                ha='center',
                                color='green')
        
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")
        
    except Exception as e:
        print(f"Error in save_results: {str(e)}")
        traceback.print_exc()

def main():
    try:
        print("Starting automated solar radiation prediction system...")
        
        # Initialize predictor with error tracking
        predictor = AutomatedPredictor()

        # Load and preprocess data
        hourly_data, minute_data, feature_averages = preprocess_data('dataset.csv')
        if hourly_data is None or minute_data is None:
            raise ValueError("Failed to load and preprocess data")
            
        print("\nLearning from historical data...")
        predictor.learn_from_historical_data(hourly_data)
        
        # Get all unique dates
        dates = hourly_data['timestamp'].dt.date.unique()
        total_predictions = 0
        total_error = 0
        
        print("\nTesting predictions on historical data...")
        for date in dates:
            day_data = hourly_data[hourly_data['timestamp'].dt.date == date]
            
            for hour in range(23):  # Up to 23 to predict next hour
                current_data = day_data[day_data['timestamp'].dt.hour == hour]
                if current_data.empty:
                    continue
                
                # Get current conditions
                conditions = {
                    'temperature': current_data['Average Temperature'].iloc[0],
                    'humidity': current_data['Average Humidity'].iloc[0],
                    'pressure': current_data['Average Barometer'].iloc[0],
                    'uv': current_data['UV Index'].iloc[0]
                }
                
                # Make prediction for next hour
                prediction = predictor.predict_next_hour(hourly_data, date, hour)
                
                # Get actual value for next hour
                next_hour_data = day_data[day_data['timestamp'].dt.hour == hour + 1]
                if not next_hour_data.empty and prediction is not None:
                    actual = next_hour_data['Solar Rad - W/m^2'].iloc[0]
                    error = abs(actual - prediction)
                    total_error += error
                    total_predictions += 1
                    
                    # Record prediction for learning analysis
                    predictor.record_prediction(
                        hour=hour,
                        predicted=prediction,
                        actual=actual,
                        conditions=conditions
                    )
                    
                    # Update learning with actual value
                    predictor.update_with_actual(date, hour + 1, actual)  # Use hour + 1
                    
                    print(f"\nDate: {date}, Hour {hour:02d}:00 -> {hour+1:02d}:00")
                    print(f"Predicted: {prediction:.2f} W/m²")
                    print(f"Actual: {actual:.2f} W/m²")
                    print(f"Error: {error:.2f} W/m²")
        
        # Print overall performance
        if total_predictions > 0:
            avg_error = total_error / total_predictions
            print(f"\nOverall Performance:")
            print(f"Total predictions: {total_predictions}")
            print(f"Average error: {avg_error:.2f} W/m²")
        
        # Generate final learning analysis
        predictor.analyze_learning_performance()
        
        # Make predictions for current/next day
        current_date = pd.Timestamp.now().date()
        print(f"\nMaking predictions for {current_date}...")
        predictions_df = predictor.predict_day(current_date, feature_averages)
        
        # Save final state
        predictor.save_state()
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
