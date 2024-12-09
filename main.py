import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
from datetime import datetime, timedelta
import traceback
import pickle
import json
from pathlib import Path
from copy import deepcopy
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
warnings.filterwarnings('ignore')

# Constants
ADDU_LATITUDE = 7.0711
ADDU_LONGITUDE = 125.6134

def calculate_cloud_cover(temperature, humidity, pressure, solar_rad, clear_sky_rad):
    """
    Calculate cloud cover using multiple empirical models and sensor data.
    
    Based on research:
    1. Kasten & Czeplak (1980) - Solar radiation and cloud cover relationship
    2. Crawford & Duchon (1999) - Cloud cover from radiation ratio
    3. Dobson & Smith (1988) - Temperature-humidity based cloud estimation
    
    Parameters:
        temperature (float): Air temperature in Celsius
        humidity (float): Relative humidity (0-100)
        pressure (float): Atmospheric pressure in hPa
        solar_rad (float): Measured solar radiation in W/m²
        clear_sky_rad (float): Calculated clear sky radiation in W/m²
        
    Returns:
        float: Estimated cloud cover (0-1)
    """
    try:
        # Add debug prints
        print("\nCloud Cover Calculation Components:")
        print(f"Temperature: {temperature:.2f}°C")
        print(f"Humidity: {humidity:.2f}%")
        print(f"Pressure: {pressure:.2f}hPa")
        print(f"Solar Radiation: {solar_rad:.2f}W/m²")
        print(f"Clear Sky Radiation: {clear_sky_rad:.2f}W/m²")

        # 1. Radiation ratio method
        if clear_sky_rad > 10:
            rad_ratio = min(1.0, max(0.0, solar_rad / clear_sky_rad))
            cloud_cover_rad = 1 - np.sqrt(rad_ratio)
        else:
            cloud_cover_rad = 0.5
        print(f"Radiation-based cloud cover: {cloud_cover_rad:.3f}")

        # 2. Temperature-Humidity method
        rh_factor = np.clip(humidity / 100.0, 0, 1)
        temp_factor = np.clip((30 - temperature) / 30, 0, 1)
        cloud_cover_th = rh_factor * temp_factor
        print(f"Temperature-Humidity cloud cover: {cloud_cover_th:.3f}")

        # 3. Pressure variation impact
        pressure_norm = np.clip((pressure - 980) / (1030 - 980), 0, 1)
        print(f"Pressure-based factor: {pressure_norm:.3f}")
        
        # Dynamic weighting
        if clear_sky_rad > 50:
            weights = [0.7, 0.2, 0.1]
            print("Using daytime weights: [0.7, 0.2, 0.1]")
        elif clear_sky_rad > 10:
            weights = [0.5, 0.3, 0.2]
            print("Using transition weights: [0.5, 0.3, 0.2]")
        else:
            weights = [0.2, 0.6, 0.2]
            print("Using nighttime weights: [0.2, 0.6, 0.2]")

        # Calculate final cloud cover
        cloud_cover = (
            weights[0] * cloud_cover_rad +
            weights[1] * cloud_cover_th +
            weights[2] * pressure_norm
        )
        
        final_cloud_cover = float(np.clip(cloud_cover, 0, 1))
        print(f"Final calculated cloud cover: {final_cloud_cover:.3f}")
        
        return final_cloud_cover

    except Exception as e:
        print(f"Error in calculate_cloud_cover: {str(e)}")
        traceback.print_exc()
        return 0.5

class WeatherConditionAnalyzer:
    def __init__(self):
        self.condition_thresholds = {
            'temperature': {'high': 30, 'low': 20},
            'humidity': {'high': 80, 'low': 40},
            'cloud_cover': {'high': 0.8, 'low': 0.2}
        }
        self.cloud_cover_history = []
        self.max_history_size = 1000  # Prevent unlimited growth

    def analyze_conditions(self, conditions):
        """Analyze weather conditions including cloud cover"""
        try:
            impact_factors = {}
            
            # Extract and validate parameters
            temp = float(conditions.get('temperature', 25))
            humidity = float(conditions.get('humidity', 60))
            pressure = float(conditions.get('pressure', 1013.25))
            solar_rad = float(conditions.get('solar_rad', 0))
            clear_sky_rad = float(conditions.get('clear_sky_rad', 1000))
            hour = conditions.get('hour', 12)

            print("\nAnalyzing Weather Conditions:")
            print(f"Hour: {hour}")
            print(f"Temperature: {temp}°C")
            print(f"Humidity: {humidity}%")
            print(f"Pressure: {pressure}hPa")
            print(f"Solar Radiation: {solar_rad}W/m²")
            print(f"Clear Sky Radiation: {clear_sky_rad}W/m²")

            # Calculate cloud cover using the correct function
            cloud_cover = calculate_cloud_cover(
                temperature=temp,
                humidity=humidity,
                pressure=pressure,
                solar_rad=solar_rad,
                clear_sky_rad=clear_sky_rad
            )

            # Store cloud cover with timestamp and hour
            self.cloud_cover_history.append({
                'timestamp': pd.Timestamp.now(),
                'hour': hour,
                'cloud_cover': cloud_cover,
                'solar_rad': solar_rad,
                'clear_sky_rad': clear_sky_rad,
                'temperature': temp,
                'humidity': humidity,
                'pressure': pressure
            })

            # Maintain history size
            if len(self.cloud_cover_history) > self.max_history_size:
                self.cloud_cover_history = self.cloud_cover_history[-self.max_history_size:]

            # Calculate impact factors with enhanced cloud consideration
            if 6 <= hour <= 18:  # Daytime
                base_impact = cloud_cover * 0.8  # Stronger impact during day
                if 10 <= hour <= 14:  # Peak hours
                    base_impact *= 1.2
            else:
                base_impact = cloud_cover * 0.4  # Reduced impact at night

            impact_factors['cloud_cover'] = float(np.clip(base_impact, 0, 1))
            
            print(f"\nCalculated Impacts:")
            print(f"Cloud Cover: {cloud_cover:.3f}")
            print(f"Cloud Impact: {impact_factors['cloud_cover']:.3f}")

            # Calculate other impacts
            if temp > self.condition_thresholds['temperature']['high']:
                impact_factors['temperature'] = min((temp - self.condition_thresholds['temperature']['high']) / 10, 1)
            elif temp < self.condition_thresholds['temperature']['low']:
                impact_factors['temperature'] = min((self.condition_thresholds['temperature']['low'] - temp) / 10, 1)

            if humidity > self.condition_thresholds['humidity']['high']:
                impact_factors['humidity'] = min((humidity - self.condition_thresholds['humidity']['high']) / 20, 1)

            return impact_factors

        except Exception as e:
            print(f"Error in analyze_conditions: {str(e)}")
            traceback.print_exc()
            return {}

    def get_cloud_cover_trend(self, hours=3):
        """Analyze recent cloud cover trend"""
        try:
            if not self.cloud_cover_history:
                return None
                
            recent_history = self.cloud_cover_history[-hours:]
            if len(recent_history) < 2:
                return None
                
            cloud_values = [h['cloud_cover'] for h in recent_history]
            return np.polyfit(range(len(cloud_values)), cloud_values, 1)[0]
            
        except Exception as e:
            print(f"Error in get_cloud_cover_trend: {str(e)}")
            return None

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
        
        # Calculate clear sky radiation for both datasets using actual weather data
        for df in [hourly_data, minute_data]:
            df['clear_sky_radiation'] = df.apply(
                lambda row: calculate_clear_sky_radiation(
                    row['hour'] + row['minute']/60 if 'minute' in df.columns else row['hour'],
                    ADDU_LATITUDE, 
                    ADDU_LONGITUDE, 
                    row['date'],
                    temperature=row['Average Temperature'],
                    humidity=row['Average Humidity'],
                    pressure=row['Average Barometer']
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

def calculate_clear_sky_radiation(hour, latitude, longitude, date, temperature=25, humidity=60, pressure=1013.25):
    """
    Calculate theoretical clear sky radiation using the Bird Clear Sky Model (Bird & Hulstrom, 1981)
    
    Parameters:
        hour (float): Hour of day (0-24)
        latitude (float): Location latitude in degrees (-90 to 90)
        longitude (float): Location longitude in degrees (-180 to 180)
        date (datetime.date): Date of calculation
        temperature (float): Ambient temperature in Celsius
        humidity (float): Relative humidity (0-100)
        pressure (float): Atmospheric pressure in hPa
        
    Returns:
        float: Clear sky radiation in W/m²
    """
    try:
        # 1. Input validation and conversion
        hour = float(np.clip(hour, 0, 24))
        latitude = float(np.clip(latitude, -90, 90))
        longitude = float(np.clip(longitude, -180, 180))
        temperature = float(np.clip(temperature, -50, 60))
        humidity = float(np.clip(humidity, 0, 100))
        pressure = float(np.clip(pressure, 300, 1100))
        
        # Ensure proper date object
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
            
        # 2. Calculate day of year and solar position
        day_of_year = date.timetuple().tm_yday
        day_angle = 2 * np.pi * (day_of_year - 1) / 365
        
        # Solar constant (W/m²) - NASA updated value
        solar_constant = 1361.1
        
        # 3. Calculate Earth-Sun distance correction (Spencer, 1971)
        eccentricity = (1.00011 + 0.034221 * np.cos(day_angle) + 
                       0.00128 * np.sin(day_angle) + 
                       0.000719 * np.cos(2 * day_angle) + 
                       0.000077 * np.sin(2 * day_angle))
        
        # 4. Solar declination (Spencer, 1971)
        declination = (0.006918 - 0.399912 * np.cos(day_angle) + 
                      0.070257 * np.sin(day_angle) - 
                      0.006758 * np.cos(2 * day_angle) + 
                      0.000907 * np.sin(2 * day_angle) - 
                      0.002697 * np.cos(3 * day_angle) + 
                      0.001480 * np.sin(3 * day_angle))
        
        # 5. Calculate solar position
        lat_rad = np.radians(latitude)
        hour_angle = np.radians(15 * (hour - 12))  # 15 degrees per hour from solar noon
        
        # Solar zenith angle
        cos_zenith = (np.sin(lat_rad) * np.sin(declination) + 
                     np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
        cos_zenith = np.clip(cos_zenith, -1, 1)
        zenith = np.arccos(cos_zenith)
        
        # Return 0 if sun is below horizon
        if cos_zenith <= 0:
            return 0.0
        
        # 6. Calculate relative air mass (Kasten and Young, 1989)
        air_mass = 1 / (cos_zenith + 0.50572 * pow(96.07995 - np.degrees(zenith), -1.6364))
        
        # Pressure-corrected air mass
        air_mass_pressure = air_mass * (pressure / 1013.25)
        
        # 7. Calculate precipitable water content (Leckner, 1978)
        T = temperature + 273.15  # Convert to Kelvin
        e_sat = 6.11 * np.exp((17.27 * temperature) / (temperature + 237.3))
        vapor_pressure = e_sat * humidity / 100
        water = 0.14 * vapor_pressure * air_mass + 2.1
        
        # 8. Calculate atmospheric transmittances
        
        # Add ozone constant at the top of the function
        ozone = 0.35  # cm NTP (typical value for clear sky conditions)
        
        # Rayleigh scattering (Bird & Hulstrom, 1981)
        tau_r = np.exp(-0.0903 * pow(air_mass_pressure, 0.84) * 
                      (1.0 + air_mass_pressure - pow(air_mass_pressure, 1.01)))
        tau_o = 1 - 0.1611 * ozone * air_mass * (1.0 + 139.48 * ozone * air_mass) ** -0.3035 - \
                0.002715 * ozone * air_mass / (1.0 + 0.044 * ozone * air_mass + 0.0003 * (ozone * air_mass) ** 2)
        tau_w = 1 - 2.4959 * water * air_mass / \
                ((1 + 79.034 * water * air_mass) ** 0.6828 + 6.385 * water * air_mass)
        beta = 0.05  # Angstrom turbidity coefficient
        alpha = 1.3   # Wavelength exponent
        tau_a = np.exp(-beta * pow(air_mass, alpha))
        
        # 9. Calculate direct normal irradiance (DNI)
        dni = solar_constant * eccentricity * tau_r * tau_o * tau_w * (0.9 * tau_a + 0.1)
        
        # 10. Calculate diffuse radiation components
        
        # Rayleigh diffuse radiation
        dr = 0.5 * solar_constant * eccentricity * cos_zenith * (1 - tau_r) * tau_o * tau_w * tau_a
        
        # Aerosol diffuse radiation
        da = 0.5 * solar_constant * eccentricity * cos_zenith * tau_r * tau_o * tau_w * (1 - tau_a * 0.9)
        
        # Multiple reflection factor
        ground_albedo = 0.2
        sky_albedo = 0.068
        reflection_factor = 1 / (1 - ground_albedo * sky_albedo)
        
        # Total diffuse radiation
        diffuse = (dr + da) * reflection_factor
        
        # 11. Calculate direct radiation on horizontal surface
        direct = dni * cos_zenith
        
        # 12. Total clear sky radiation
        clear_sky = direct + diffuse
        
        # 13. Apply time-of-day corrections
        if hour < 5 or hour > 18:  # Night hours
            return 0
        elif hour < 8 or hour > 16:  # Early morning/late afternoon
            clear_sky *= 0.75 + 0.25 * cos_zenith  # Gradual transition
        elif 11 <= hour <= 13:  # Peak hours
            clear_sky *=0.95 + 0.05 * cos_zenith  # Small zenith angle correction
            
        # 14. Final validation
        clear_sky = float(np.clip(clear_sky, 0, 1200))  # Cap at 1200 W/m²
        
        return clear_sky
        
    except Exception as e:
        print(f"Error in calculate_clear_sky_radiation: {str(e)}")
        traceback.print_exc()
        return 0.0    
    
class PredictionErrorLearner:
    def __init__(self):
        self.error_history = {}
        self.adjustment_factors = {}
        self.hourly_errors = {hour: [] for hour in range(24)}
        self.learning_rate = 0.01  # 1% per percentage point of error
        self.min_adjustment = 0.7
        self.max_adjustment = 1.3
        self.error_window_size = 10
        self.recent_errors = []  # Add this line to initialize recent_errors
        self.pattern_adjustments = {}  # Keep this as it's used in get_adjustment
        
    def _update_adjustment_factors(self, hour, error_pct, conditions):
        """Update adjustment factors with smoother transitions"""
        key = f"{hour}"
        if key not in self.adjustment_factors:
            self.adjustment_factors[key] = {'factor': 1.0, 'errors': []}
            
        # Add error to history
        self.hourly_errors[hour].append(error_pct)
        if len(self.hourly_errors[hour]) > self.error_window_size:
            self.hourly_errors[hour].pop(0)
            
        # Calculate new adjustment based on recent errors
        recent_errors = self.hourly_errors[hour][-self.error_window_size:]
        avg_error = np.mean(recent_errors) if recent_errors else 0
        
        # Calculate new factor with controlled adjustment
        current_factor = self.adjustment_factors[key]['factor']
        new_factor = current_factor * (1 + avg_error * self.learning_rate)
        new_factor = max(self.min_adjustment, min(self.max_adjustment, new_factor))
        
        # Smooth transition (80/20 blend)
        self.adjustment_factors[key]['factor'] = current_factor * 0.8 + new_factor * 0.2
        
        print(f"\nHour {hour} adjustment update:")
        print(f"Average error: {avg_error:.2f}%")
        print(f"Previous factor: {current_factor:.3f}")
        print(f"New factor: {new_factor:.3f}")
        print(f"Smoothed factor: {self.adjustment_factors[key]['factor']:.3f}")

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
        """Update adjustment factors with more aggressive learning"""
        key = f"{hour}"
        if key not in self.adjustment_factors:
            self.adjustment_factors[key] = 1.0
            
        # Dynamic learning rate based on error magnitude
        if abs(error_pct) > 20:
            learning_rate = 0.3  # More aggressive for large errors
        else:
            learning_rate = 0.2  # Standard rate for smaller errors
            
        current_factor = self.adjustment_factors[key]
        
        # Calculate new adjustment factor
        if error_pct > 0:  # Prediction was too low
            new_factor = current_factor * (1 + error_pct * learning_rate)
        else:  # Prediction was too high
            new_factor = current_factor * (1 / (1 + abs(error_pct) * learning_rate))
            
        # Apply limits
        new_factor = max(self.min_adjustment, min(self.max_adjustment, new_factor))
        
        # Weighted average with more weight to new adjustment
        self.adjustment_factors[key] = current_factor * 0.7 + new_factor * 0.3
        
        print(f"\nAdjustment factor update for hour {hour}:")
        print(f"Previous factor: {current_factor:.3f}")
        print(f"New factor: {new_factor:.3f}")
        print(f"Final factor: {self.adjustment_factors[key]:.3f}")

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

    def _update_learning_rate(self, error_history):
        """Dynamic learning rate based on recent performance"""
        recent_errors = error_history[-24:]  # Last 24 predictions
        error_std = np.std([e['error_percentage'] for e in recent_errors])
        
        if error_std > 50:  # High variance in errors
            self.learning_rate = 0.15  # More aggressive learning
        elif error_std > 25:
            self.learning_rate = 0.1
        else:
            self.learning_rate = 0.05  # More conservative when stable

    def _update_hourly_errors(self, hour, error_percentage):
        """Update hourly error tracking with proper handling"""
        try:
            # Skip error tracking for nighttime hours
            if hour >= 18 or hour <= 5:
                return
                
            # Validate error percentage
            if error_percentage is None or not np.isfinite(error_percentage):
                return
                
            # Initialize hour if needed
            if hour not in self.hourly_errors:
                self.hourly_errors[hour] = []
                
            # Add error to history
            self.hourly_errors[hour].append(error_percentage)
            
            # Maintain window size
            if len(self.hourly_errors[hour]) > self.error_window_size:
                self.hourly_errors[hour].pop(0)
                
            # Get valid recent errors
            recent_errors = [e for e in self.hourly_errors[hour][-5:] 
                              if np.isfinite(e)]
            
            # Print only if we have valid errors
            if recent_errors:
                print(f"\nUpdated hour {hour} performance:")
                print(f"Recent errors: {[f'{e:.1f}%' for e in recent_errors]}")
                
        except Exception as e:
            print(f"Error updating hourly errors: {str(e)}")

class ErrorPatternAnalyzer:
    def __init__(self):
        self.error_patterns = {
            'hourly': {},      # Hour-specific error patterns
            'weather': {},     # Weather condition related errors
            'seasonal': {},    # Seasonal patterns
            'trend': []        # Overall error trends
        }
        self.pattern_window = 168  # 7 days * 24 hours

    def analyze_errors(self, history_df):
        try:
            # Hourly patterns
            hourly_errors = history_df.groupby('hour')['error_percentage'].agg(['mean', 'std'])
            for hour, stats in hourly_errors.iterrows():
                self.error_patterns['hourly'][hour] = {
                    'mean_error': stats['mean'],
                    'std_error': stats['std'],
                    'needs_attention': abs(stats['mean']) > 15
                }

            # Weather-related patterns
            if 'conditions' in history_df.columns:
                for condition in ['temperature', 'humidity', 'cloud_cover']:
                    condition_errors = self._analyze_condition_impact(history_df, condition)
                    self.error_patterns['weather'][condition] = condition_errors

            return self.error_patterns

        except Exception as e:
            print(f"Error in analyze_errors: {str(e)}")
            traceback.print_exc()
            return {}

    def _analyze_condition_impact(self, history_df, condition):
        try:
            conditions = pd.json_normalize(history_df['conditions'].apply(eval))
            if condition in conditions:
                bins = pd.qcut(conditions[condition], q=5)
                impact = history_df.groupby(bins)['error_percentage'].agg(['mean', 'std'])
                return impact.to_dict('index')
            return {}
        except Exception as e:
            print(f"Error analyzing condition impact: {str(e)}")
            return {}

class FallbackPredictor:
    def __init__(self):
        self.fallback_methods = {
            'moving_average': {'weight': 0.4, 'threshold': 20},
            'clear_sky': {'weight': 0.3, 'threshold': 30},
            'historical': {'weight': 0.3, 'threshold': 25}
        }
        self.fallback_history = []

    def get_fallback_prediction(self, current_value, conditions, clear_sky, historical_data):
        try:
            predictions = {}
            weights = {}
            total_weight = 0

            # Get predictions from each fallback method
            for method, config in self.fallback_methods.items():
                pred = self._get_method_prediction(
                    method, current_value, conditions, clear_sky, historical_data
                )
                if pred is not None:
                    predictions[method] = pred
                    weights[method] = config['weight']
                    total_weight += config['weight']

            # Combine predictions
            if predictions and total_weight > 0:
                fallback_prediction = sum(
                    pred * (weights[method] / total_weight)
                    for method, pred in predictions.items()
                )
                return fallback_prediction

            return current_value

        except Exception as e:
            print(f"Error in get_fallback_prediction: {str(e)}")
            return current_value

    def _get_method_prediction(self, method, current_value, conditions, clear_sky, historical_data):
        try:
            if method == 'moving_average':
                return self._calculate_moving_average(historical_data)
            elif method == 'clear_sky':
                return clear_sky * 0.85
            elif method == 'historical':
                return self._get_historical_prediction(historical_data, conditions)
            return None

        except Exception as e:
            print(f"Error in fallback method {method}: {str(e)}")
            return None

    def _calculate_moving_average(self, historical_data):
        try:
            if historical_data is not None and len(historical_data) > 0:
                recent_values = historical_data['Solar Rad - W/m^2'].tail(24).values
                if len(recent_values) > 0:
                    return float(np.mean(recent_values))
            return None
        except Exception as e:
            print(f"Error calculating moving average: {str(e)}")
            return None

    def _get_historical_prediction(self, historical_data, conditions):
        try:
            if historical_data is not None and len(historical_data) > 0:
                similar_records = self._find_similar_conditions(historical_data, conditions)
                if similar_records is not None and len(similar_records) > 0:
                    return float(np.mean(similar_records['Solar Rad - W/m^2']))
            return None
        except Exception as e:
            print(f"Error getting historical prediction: {str(e)}")
            return None

    def _find_similar_conditions(self, historical_data, conditions):
        try:
            return historical_data.tail(24)  # Simplified version
        except Exception as e:
            print(f"Error finding similar conditions: {str(e)}")
            return None

class AutomatedPredictor:
    def __init__(self, data_folder='predictions/'):
        # Initialize directories first
        self._initialize_directories()
        
        # Initialize other attributes
        self.hourly_patterns = {}
        self.seasonal_patterns = {}
        self.transition_patterns = {}
        self.weather_impacts = {}
        self.error_learner = PredictionErrorLearner()
        self.consecutive_errors = []
        self.prediction_history = []
        self.max_consecutive_errors = 10
        #self.max_history_size = 10000  # Increase history size limit
        
        # Load previous state
        self.load_state()
        
        # Add model versioning
        self.model_performance = {
            'best_mae': float('inf'),
            'best_timestamp': None,
            'current_mae': float('inf'),
            'evaluation_window': 100
        }
        
        # Load best model if exists
        self.load_best_model()
        
        # Add new components
        self.weather_analyzer = WeatherConditionAnalyzer()
        self.error_analyzer = ErrorPatternAnalyzer()
        self.fallback_predictor = FallbackPredictor()
        
        # Add method performance tracking
        self.method_performance = {
            'pattern': {'weight': 0.3, 'errors': [], 'window_size': 100},
            'ratio': {'weight': 0.3, 'errors': [], 'window_size': 100},
            'trend': {'weight': 0.2, 'errors': [], 'window_size': 100},
            'typical': {'weight': 0.2, 'errors': [], 'window_size': 100}
        }

        self.cloud_impact_weights = {
            'high_cloud': 0.7,    # Significant reduction in radiation
            'medium_cloud': 0.85,  # Moderate reduction
            'low_cloud': 0.95     # Slight reduction
        }

        # Add hourly error tracking
        self.hourly_errors = {hour: [] for hour in range(24)}
        self.error_window_size = 100

        # Add to existing initialization
        self.time_series_ensemble = TimeSeriesEnsemble()
        print("Initialized TimeSeriesEnsemble")

    def _cleanup_old_learning_states(self, max_states=5):
        """Remove old learning states when limit is reached"""
        try:
            # Get all learning state files
            learning_states = [f for f in os.listdir(self.models_folder) 
                             if f.startswith('learning_state_') and f.endswith('.pkl')]
            
            # If number of files exceeds max_states, remove oldest ones
            if len(learning_states) > max_states:
                # Sort files by timestamp (oldest first)
                learning_states.sort()
                # Remove oldest files
                files_to_remove = learning_states[:-max_states]  # Keep only the latest max_states files
                
                for file in files_to_remove:
                    file_path = os.path.join(self.models_folder, file)
                    try:
                        os.remove(file_path)
                        print(f"Removed old learning state: {file}")
                    except Exception as e:
                        print(f"Error removing file {file}: {str(e)}")
                        
        except Exception as e:
            print(f"Error in cleanup_old_learning_states: {str(e)}")
            traceback.print_exc()

    def save_state(self):
        """Save learning state and history"""
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Save learning state with enhanced model information
            learning_state = {
                'hourly_patterns': self.hourly_patterns,
                'seasonal_patterns': self.seasonal_patterns,
                'transition_patterns': self.transition_patterns,
                'weather_impacts': self.weather_impacts,
                'consecutive_errors': self.consecutive_errors[-10:],  # Keep only last 10
                'error_learner_state': {
                    'error_history': self.error_learner.error_history,
                    'adjustment_factors': self.error_learner.adjustment_factors,
                    'pattern_adjustments': self.error_learner.pattern_adjustments
                },
                # Add model performance metrics
                'model_performance': {
                    'best_mae': float(self.model_performance.get('best_mae', float('inf'))),
                    'current_mae': float(self.model_performance.get('current_mae', float('inf'))),
                    'best_timestamp': str(self.model_performance.get('best_timestamp', None)),
                    'evaluation_window': self.model_performance['evaluation_window'],
                    'total_predictions': len(self.prediction_history)
                },
                'metadata': {
                    'version': '1.0',
                    'timestamp': timestamp,
                    'created_at': str(pd.Timestamp.now()),
                    'total_predictions': len(self.prediction_history)
                },
                'ensemble_state': {
                    'hour_specific_models': dict(
                        (str(hour), {'model': pickle.dumps(model)})
                        for hour, model in self.time_series_ensemble.hour_specific_models.items()
                    )
                },
                'ensemble_models': {
                    hour: model for hour, model in self.time_series_ensemble.hour_specific_models.items()
                    if not isinstance(model, dict)  # Skip constant predictors
                }
            }
            
            # Save state using regular pickle with error handling
            state_file = os.path.join(self.models_folder, f'learning_state_{timestamp}.pkl')
            try:
                with open(state_file, 'wb') as f:
                    pickle.dump(learning_state, f)
                print(f"\nSaved learning state to: {state_file}")
                
                # Also save latest version
                latest_state_file = os.path.join(self.models_folder, 'latest_learning_state.pkl')
                with open(latest_state_file, 'wb') as f:
                    pickle.dump(learning_state, f)
                print(f"Saved latest learning state to: {latest_state_file}")
                
                # Save a JSON summary of the learning state
                summary_file = os.path.join(self.models_folder, 'learning_state_summary.json')
                summary = {
                    'timestamp': timestamp,
                    'model_performance': learning_state['model_performance'],
                    'metadata': learning_state['metadata'],
                    'file_path': state_file
                }
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=4)
                print(f"Saved learning state summary to: {summary_file}")
                
            except Exception as e:
                print(f"Error saving learning state: {str(e)}")
                traceback.print_exc()
            
            # Save prediction history if exists
            if self.prediction_history:
                try:
                    history_df = pd.DataFrame(self.prediction_history)
                    history_file = os.path.join(self.history_folder, f'prediction_history_{timestamp}.csv')
                    history_df.to_csv(history_file, index=False)
                    
                    # Also save latest version to stats folder
                    latest_history_file = os.path.join(self.stats_folder, 'latest_prediction_history.csv')
                    history_df.to_csv(latest_history_file, index=False)
                    
                    print(f"Saved prediction history to: {history_file}")
                    print(f"Saved latest history to: {latest_history_file}")
                    
                except Exception as e:
                    print(f"Error saving prediction history: {str(e)}")
                    traceback.print_exc()
            
            # Cleanup old learning states
            self._cleanup_old_learning_states()
                
        except Exception as e:
            print(f"Error in save_state: {str(e)}")
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
                print(f"Loading learning state from: {self.learning_state_file}")
                with open(self.learning_state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.hourly_patterns = state.get('hourly_patterns', {})
                    self.seasonal_patterns = state.get('seasonal_patterns', {})
                    self.transition_patterns = state.get('transition_patterns', {})
                    self.weather_impacts = state.get('weather_impacts', {})
                    self.consecutive_errors = state.get('consecutive_errors', [])
                    
                    # Restore error learner state
                    error_state = state.get('error_learner_state', {})
                    self.error_learner.error_history = error_state.get('error_history', {})
                    self.error_learner.adjustment_factors = error_state.get('adjustment_factors', {})
                    self.error_learner.pattern_adjustments = error_state.get('pattern_adjustments', {})
                
                # Load prediction history if exists
                if os.path.exists(self.errors_file):
                    print(f"Loading prediction history from: {self.errors_file}")
                    self.prediction_history = pd.read_csv(self.errors_file).to_dict('records')
                
                print("Successfully loaded previous learning state")

                if 'ensemble_state' in state:
                    ensemble_state = state['ensemble_state']
                    self.time_series_ensemble.hour_specific_models = {
                        hour: pickle.loads(model_data['model'])
                        for hour, model_data in ensemble_state['hour_specific_models'].items()
                    }
                if 'ensemble_models' in state:
                    self.time_series_ensemble.hour_specific_models.update(state['ensemble_models'])
            else:
                print("No previous learning state found - starting fresh")
                # Initialize empty state
                self.hourly_patterns = {}
                self.seasonal_patterns = {}
                self.transition_patterns = {}
                self.weather_impacts = {}
                self.consecutive_errors = []
                self.prediction_history = []
                
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            print("Starting with fresh learning state")
            # Initialize empty state on error
            self.hourly_patterns = {}
            self.seasonal_patterns = {}
            self.transition_patterns = {}
            self.weather_impacts = {}
            self.consecutive_errors = []
            self.prediction_history = []

    def update_with_actual(self, date, hour, actual_value):
        """Update prediction with actual value"""
        try:
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            updated = False
            
            # Find matching prediction
            for pred in reversed(self.prediction_history):
                if (pred['date'] == date_str and 
                    pred['hour'] == hour and 
                    pred['actual'] is None):
                    
                    # Update prediction record
                    pred['actual'] = float(actual_value)
                    pred['error'] = float(actual_value - pred['predicted'])
                    
                    # Handle error percentage calculation
                    if hour >= 18 or hour <= 5:  # Nighttime
                        pred['error_percentage'] = 0.0
                    elif actual_value == 0 and pred['predicted'] == 0:
                        pred['error_percentage'] = 0.0
                    elif actual_value == 0:
                        pred['error_percentage'] = 100.0 if pred['predicted'] > 0 else 0.0
                    else:
                        pred['error_percentage'] = (pred['error'] / actual_value) * 100
                    
                    # Update error learner
                    self.error_learner.record_error(
                        hour=hour,
                        predicted=pred['predicted'],
                        actual=actual_value,
                        conditions=pred['conditions']
                    )
                    
                    print(f"\nUpdated prediction for {date_str}, hour {hour}:")
                    print(f"Predicted: {pred['predicted']:.2f}")
                    print(f"Actual: {actual_value:.2f}")
                    print(f"Error: {pred['error']:.2f}")
                    print(f"Error %: {pred['error_percentage']:.2f}%")
                    
                    updated = True
                    break
            
            if not updated:
                print(f"No pending prediction found for date {date}, hour {hour}")
            
            return updated
            
        except Exception as e:
            print(f"Error in update_with_actual: {str(e)}")
            traceback.print_exc()
            return False

    def evaluate_and_save_model(self):
        """Evaluate current model performance and save if best"""
        try:
            if len(self.prediction_history) >= self.model_performance['evaluation_window']:
                # Convert prediction history to DataFrame if it's not already
                if isinstance(self.prediction_history, list):
                    recent_predictions = pd.DataFrame(self.prediction_history[-self.model_performance['evaluation_window']:])
                else:
                    recent_predictions = self.prediction_history[-self.model_performance['evaluation_window']:]
                
                # Calculate current performance
                current_mae = np.mean(np.abs(recent_predictions['error']))
                self.model_performance['current_mae'] = current_mae
                
                print(f"\nEvaluating model performance:")
                print(f"Current MAE: {current_mae:.2f} W/m²")
                print(f"Best MAE: {self.model_performance.get('best_mae', float('inf')):.2f} W/m²")
                
                # Always save performance metrics with explicit path handling
                metrics_path = os.path.join(self.models_folder, 'model_performance.json')
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                
                performance_metrics = {
                    'current_mae': float(current_mae),
                    'best_mae': float(self.model_performance.get('best_mae', float('inf'))),
                    'best_timestamp': str(self.model_performance.get('best_timestamp', None)),
                    'total_predictions': len(self.prediction_history),
                    'last_evaluated': str(pd.Timestamp.now()),
                    'evaluation_window': self.model_performance['evaluation_window']
                }
                
                # Save metrics with error handling
                try:
                    with open(metrics_path, 'w') as f:
                        json.dump(performance_metrics, f, indent=4)
                    print(f"Performance metrics saved to: {metrics_path}")
                except Exception as e:
                    print(f"Error saving performance metrics: {str(e)}")
                
                # Save best model if current performance is better
                if current_mae < self.model_performance.get('best_mae', float('inf')):
                    print(f"\nNew best model detected! (MAE: {current_mae:.2f} W/m² vs previous: {self.model_performance.get('best_mae', float('inf')):.2f} W/m²)")
                    self.model_performance['best_mae'] = current_mae
                    self.model_performance['best_timestamp'] = pd.Timestamp.now()
                    # Save best model state with complete information
                    best_state = {
                        'hourly_patterns': self.hourly_patterns,
                        'seasonal_patterns': self.seasonal_patterns,
                        'transition_patterns': self.transition_patterns,
                        'weather_impacts': self.weather_impacts,
                        'error_learner_state': {
                            'error_history': self.error_learner.error_history,
                            'adjustment_factors': self.error_learner.adjustment_factors,
                            'pattern_adjustments': self.error_learner.pattern_adjustments
                        },
                        'model_performance': self.model_performance,
                        'timestamp': pd.Timestamp.now(),
                        'metadata': {
                            'version': '1.0',
                            'created_at': str(pd.Timestamp.now()),
                            'evaluation_window': self.model_performance['evaluation_window'],
                            'total_predictions': len(self.prediction_history)
                        }
                    }

                    # Save best model with error handling
                    best_model_path = os.path.join(self.models_folder, 'best_model.pkl')
                    try:
                        with open(best_model_path, 'wb') as f:
                            pickle.dump(best_state, f)
                        print(f"Saved new best model to: {best_model_path}")
                        
                        # Save timestamped version
                        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        archive_path = os.path.join(self.models_folder, f'best_model_{timestamp}.pkl')
                        with open(archive_path, 'wb') as f:
                            pickle.dump(best_state, f)
                        print(f"Archived copy saved to: {archive_path}")
                        
                        # Save a JSON summary of the best model
                        summary_path = os.path.join(self.models_folder, 'best_model_summary.json')
                        summary = {
                            'timestamp': str(pd.Timestamp.now()),
                            'mae': float(current_mae),
                            'total_predictions': len(self.prediction_history),
                            'model_version': '1.0',
                            'file_path': best_model_path
                        }
                        with open(summary_path, 'w') as f:
                            json.dump(summary, f, indent=4)
                        print(f"Model summary saved to: {summary_path}")
                        
                    except Exception as e:
                        print(f"Error saving best model: {str(e)}")
                        traceback.print_exc()
                else:
                    print("Current model performance not better than best model.")
                
            else:
                print(f"\nNot enough predictions ({len(self.prediction_history)}) for model evaluation. Need {self.model_performance['evaluation_window']}.")
                
        except Exception as e:
            print(f"Error in evaluate_and_save_model: {str(e)}")
            traceback.print_exc()

    def load_best_model(self):
        """Load the best performing model if available"""
        try:
            best_model_path = os.path.join(self.models_folder, 'best_model.pkl')
            if os.path.exists(best_model_path):
                print("\nLoading best model...")
                with open(best_model_path, 'rb') as f:
                    best_state = pickle.load(f)
                    
                # Restore model state
                self.hourly_patterns = best_state['hourly_patterns']
                self.seasonal_patterns = best_state['seasonal_patterns']
                self.transition_patterns = best_state['transition_patterns']
                self.weather_impacts = best_state['weather_impacts']
                
                # Restore error learner state
                error_learner_state = best_state['error_learner_state']
                self.error_learner.error_history = error_learner_state['error_history']
                self.error_learner.adjustment_factors = error_learner_state['adjustment_factors']
                self.error_learner.pattern_adjustments = error_learner_state['pattern_adjustments']
                
                # Restore performance metrics
                self.model_performance = best_state['model_performance']
                
                print(f"Loaded best model from {best_model_path}")
                print(f"Best model MAE: {self.model_performance['best_mae']:.2f} W/m²")
                print(f"Best model timestamp: {self.model_performance['best_timestamp']}")
                
            else:
                print("No best model found - starting fresh")
                self.model_performance = {
                    'best_mae': float('inf'),
                    'best_timestamp': None,
                    'current_mae': float('inf'),
                    'evaluation_window': 100
                }
                
        except Exception as e:
            print(f"Error loading best model: {str(e)}")
            traceback.print_exc()

    def learn_from_historical_data(self, data):
        """Learn patterns from historical data"""
        print("\nTraining ensemble models...")
        
        # Train hour-specific ensemble models
        for hour in range(24):
            hour_data = data[data['timestamp'].dt.hour == hour]
            if len(hour_data) < 24:  # Need minimum data
                print(f"Insufficient data for hour {hour:02d} (samples: {len(hour_data)})")
                continue
            
            print(f"\nTraining model for hour {hour:02d} with {len(hour_data)} samples")
            
            # Get target values
            targets = hour_data['Solar Rad - W/m^2']
            
            # Train ensemble model
            self.time_series_ensemble.fit(hour_data, targets, hour)
        
        # Print training status
        self.time_series_ensemble.get_training_status()

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

    def _get_historical_average(self, conditions):
        """Calculate historical average based on similar conditions"""
        try:
            hour = conditions.get('hour', 0)
            
            # Get historical data for this hour
            if not self.prediction_history:
                return None
                
            history_df = pd.DataFrame(self.prediction_history)
            
            # Convert date strings to datetime if needed
            if 'date' in history_df.columns:
                history_df['date'] = pd.to_datetime(history_df['date'])
                history_df['days_old'] = (pd.Timestamp.now() - history_df['date']).dt.days
            else:
                print("Warning: No date information in prediction history")
                return None
                
            hour_data = history_df[history_df['hour'] == hour]
            
            if hour_data.empty:
                return None
            
            # Calculate weighted average based on recency
            hour_data['weight'] = np.exp(-hour_data['days_old'] / 30)  # Exponential decay
            
            # Use 'actual' values where available, otherwise use 'predicted'
            if 'actual' in hour_data.columns:
                values = hour_data['actual']
            else:
                values = hour_data['predicted']
                
            weighted_avg = (values * hour_data['weight']).sum() / hour_data['weight'].sum()
            
            print(f"Historical average for hour {hour}: {weighted_avg:.2f} W/m²")
            return float(weighted_avg)
            
        except Exception as e:
            print(f"Error in _get_historical_average: {str(e)}")
            traceback.print_exc()
            return None

    def predict_next_hour(self, data, target_date, current_hour):
        """Enhanced prediction with validation"""
        try:
            # Input validation
            if not isinstance(current_hour, (int, float)) or not 0 <= current_hour <= 23:
                raise ValueError(f"Invalid hour: {current_hour}")
            
            next_hour = (current_hour + 1) % 24
            
            # Get conditions with validation
            conditions = self._get_current_conditions(data, target_date, current_hour)
            if not conditions:
                print("Failed to get current conditions")
                return None
            
            # Get predictions with validation
            predictions = self._get_all_predictions(data, target_date, current_hour, conditions)
            if not predictions:
                print("Failed to get predictions")
                return None
            
            # Calculate adjustments
            weather_adjustment = self._calculate_weather_adjustment(conditions, next_hour)
            transition_adjustment = self._calculate_transition_adjustments(next_hour, conditions)
            
            # Get ensemble prediction
            final_prediction = self._get_ensemble_prediction(predictions, next_hour, conditions)
            
            # Apply adjustments
            final_prediction *= weather_adjustment * transition_adjustment
            
            # Validate final prediction
            final_prediction = self._validate_prediction(final_prediction, next_hour, conditions)
            
            # Get actual value if available
            next_data = data[
                (data['timestamp'].dt.date == target_date) & 
                (data['timestamp'].dt.hour == next_hour)
            ]
            actual_value = float(next_data['Solar Rad - W/m^2'].iloc[0]) if not next_data.empty else None
            
            # Store prediction with actual value if available
            if final_prediction is not None:
                success = self._store_prediction_with_history(
                    target_date, 
                    next_hour, 
                    final_prediction, 
                    actual_value,  # Include actual value if available
                    conditions
                )
                if not success:
                    print("Failed to store prediction")
            
            return final_prediction
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return None

    def _get_main_prediction(self, current_value, conditions, clear_sky):
        """Get main model prediction with enhanced transition handling"""
        try:
            hour = conditions.get('hour', 0)
            
            # Force 0 for nighttime hours
            if hour >= 18 or hour <= 5:
                return 0.0, None, 1.0

            # Get similar days for pattern and ratio predictions
            similar_days = self._find_similar_days(
                data=self.prediction_history,
                target_date=pd.Timestamp.now().date(),
                current_hour=conditions.get('hour', 0),
                conditions=conditions
            )
            
            # Calculate base predictions with fallback values
            pattern_pred = self._get_pattern_prediction(similar_days, current_value) or current_value
            ratio_pred = self._get_ratio_prediction(similar_days, current_value) or current_value
            
            # Get trend prediction using recent history
            if len(self.prediction_history) >= 2:
                recent_data = pd.DataFrame(self.prediction_history[-24:])
                trend_pred = self._get_trend_prediction(recent_data, recent_data['hour'].iloc[-1])
            else:
                trend_pred = current_value

            # Get cloud cover and adjust clear sky
            cloud_cover = self.weather_analyzer.analyze_conditions(conditions).get('cloud_cover', 0)
            adjusted_clear_sky = clear_sky * (1 - (cloud_cover * 0.7))
            
            # Get typical value for this hour
            typical_pred = self._get_typical_value(hour) or current_value
            historical_avg = self._get_historical_average(conditions) or current_value

            # Print debug information
            print(f"\nPrediction components:")
            print(f"Current value: {current_value:.2f}")
            print(f"Pattern prediction: {pattern_pred:.2f}")
            print(f"Ratio prediction: {ratio_pred:.2f}")
            print(f"Trend prediction: {trend_pred:.2f}")
            print(f"Typical prediction: {typical_pred:.2f}")
            print(f"Adjusted clear sky: {adjusted_clear_sky:.2f}")
            print(f"Historical average: {historical_avg:.2f}")
            
            # Get dynamic weights based on recent performance
            weights = self._calculate_prediction_weights()
            print(f"Base weights: {weights}")
            
            # Ensure all components are valid numbers
            components = {
                'pattern': pattern_pred,
                'ratio': ratio_pred,
                'trend': trend_pred,
                'typical': typical_pred
            }
            
            # Filter out None values and adjust weights
            valid_components = {k: v for k, v in components.items() if v is not None}
            if not valid_components:
                return current_value, weights, 1.0
                
            # Normalize remaining weights
            total_weight = sum(weights[k] for k in valid_components.keys())
            if total_weight > 0:
                normalized_weights = {k: weights[k]/total_weight for k in valid_components.keys()}
            else:
                normalized_weights = {k: 1.0/len(valid_components) for k in valid_components.keys()}

            # First combine model-based predictions
            model_prediction = sum(valid_components[k] * normalized_weights[k] for k in valid_components.keys())

            # Then combine with current value and clear sky components
            final_prediction = (
                model_prediction * 0.4 +
                current_value * 0.4 +
                adjusted_clear_sky * 0.2
            )
            
            # Apply learning adjustment
            adjustment = self.error_learner.get_adjustment(hour, conditions)
            final_prediction *= adjustment
            
            print(f"Model prediction: {model_prediction:.2f}")
            print(f"Final prediction: {final_prediction:.2f}")
            print(f"Adjustment factor: {adjustment:.3f}")
            
            # Validate prediction
            if not np.isfinite(final_prediction) or final_prediction < 0:
                print("Invalid prediction value, using fallback")
                final_prediction = current_value  # Fallback to current value
            
            return final_prediction, weights, adjustment

        except Exception as e:
            print(f"Error in _get_main_prediction: {str(e)}")
            traceback.print_exc()
            return current_value, None, 1.0

    def _get_ensemble_prediction(self, predictions, hour, conditions):
        """Get ensemble prediction with normalized weights"""
        try:
            # Get ensemble model prediction
            ensemble_prediction = None
            if hour in self.time_series_ensemble.hour_specific_models:
                # Prepare features for ensemble
                features = {
                    'temperature': conditions['temperature'],
                    'humidity': conditions['humidity'],
                    'pressure': conditions['pressure'],
                    'uv': conditions['uv'],
                    'clear_sky_rad': conditions['clear_sky_rad'],
                    'solar_rad': conditions['solar_rad']
                }
                
                ensemble_prediction = self.time_series_ensemble.predict(features, hour)
                if ensemble_prediction is not None:
                    predictions['ensemble'] = float(ensemble_prediction)
                    print(f"ML Ensemble prediction: {ensemble_prediction:.2f}")
            
            # Base weights must sum to 1.0
            weights = {
                'main_model': 0.35,
                'moving_avg': 0.25,
                'pattern': 0.20,
                'clear_sky': 0.20
            }
            
            # If ensemble prediction available, redistribute weights
            if ensemble_prediction is not None:
                weights = {
                    'ensemble': 0.30,
                    'main_model': 0.25,
                    'moving_avg': 0.20,
                    'pattern': 0.15,
                    'clear_sky': 0.10
                }
            
            # Ensure weights sum to 1.0
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            # Calculate weighted prediction
            final_prediction = 0.0
            valid_weight = 0.0
            
            for method, pred in predictions.items():
                if method in weights and pred is not None and np.isfinite(pred):
                    weight = weights[method]
                    final_prediction += pred * weight
                    valid_weight += weight
            
            if valid_weight > 0:
                final_prediction /= valid_weight
            
            print(f"\nFinal ensemble prediction: {final_prediction:.2f}")
            print("Weights used:")
            for method, weight in weights.items():
                if method in predictions:
                    print(f"{method}: {weight:.2f}")
            
            return float(final_prediction)
            
        except Exception as e:
            print(f"Error in ensemble prediction: {str(e)}")
            traceback.print_exc()
            return float(predictions.get('main_model', 0.0))

    def _calculate_prediction_confidence(self, prediction, current_value, clear_sky):
        """Calculate confidence score for a prediction"""
        try:
            confidence = 1.0
            
            # Check for reasonable change ratio
            change_ratio = prediction / current_value if current_value > 0 else 0
            if change_ratio < 0.4 or change_ratio > 2.5:
                confidence *= 0.5
            
            # Check against clear sky model
            clear_sky_ratio = prediction / clear_sky if clear_sky > 0 else 0
            if clear_sky_ratio > 0.95:  # Unlikely to exceed 95% of clear sky
                confidence *= 0.7
            
            # Check recent prediction accuracy if available
            if len(self.prediction_history) >= 24:
                recent = pd.DataFrame(self.prediction_history[-24:])
                if 'error_percentage' in recent.columns:
                    avg_error = recent['error_percentage'].abs().mean()
                    confidence *= max(0.5, 1 - (avg_error / 100))
            
            return confidence
            
        except Exception as e:
            print(f"Error in _calculate_prediction_confidence: {str(e)}")
            return 0.5  # Return moderate confidence on error

    def _save_prediction_history(self):
        """Save prediction history to CSV with enhanced saving"""
        try:
            if self.prediction_history:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                
                # Save to history folder
                history_file = os.path.join(self.history_folder, f'prediction_history_{timestamp}.csv')
                history_df = pd.DataFrame(self.prediction_history)
                history_df.to_csv(history_file, index=False)
                
                # Save to data folder
                data_file = os.path.join(self.data_folder, 'latest_predictions.csv')
                history_df.to_csv(data_file, index=False)
                
                # Save detailed data to reports folder
                report_file = os.path.join(self.reports_folder, f'detailed_predictions_{timestamp}.csv')
                
                # Add additional analysis columns
                history_df['timestamp'] = pd.to_datetime(history_df['date'])
                history_df['hour_of_day'] = history_df['hour']
                history_df['prediction_accuracy'] = 100 - abs(history_df['error_percentage'])
                
                # Save detailed report
                history_df.to_csv(report_file, index=False)
                
                # Generate and save summary report
                summary_file = os.path.join(self.reports_folder, f'prediction_summary_{timestamp}.txt')
                with open(summary_file, 'w') as f:
                    f.write("=== Prediction Summary ===\n\n")
                    f.write(f"Generated: {timestamp}\n")
                    f.write(f"Total Predictions: {len(history_df)}\n")
                    f.write(f"Date Range: {history_df['timestamp'].min()} to {history_df['timestamp'].max()}\n\n")
                    
                    # Add hourly statistics
                    f.write("Hourly Performance:\n")
                    hourly_stats = history_df.groupby('hour').agg({
                        'error': ['mean', 'std'],
                        'error_percentage': 'mean',
                        'prediction_accuracy': 'mean'
                    }).round(2)
                    
                    for hour in range(24):
                        if hour in hourly_stats.index:
                            stats = hourly_stats.loc[hour]
                            f.write(f"\nHour {hour:02d}:00\n")
                            f.write(f"  Mean Error: {stats[('error', 'mean')]:.2f} W/m²\n")
                            f.write(f"  Error Std: {stats[('error', 'std')]:.2f} W/m²\n")
                            f.write(f"  Mean Error %: {stats[('error_percentage', 'mean')]:.2f}%\n")
                            f.write(f"  Accuracy: {stats[('prediction_accuracy', 'mean')]:.2f}%\n")
                
                print(f"Saved prediction summary to: {summary_file}")
                
        except Exception as e:
            print(f"Error saving prediction history: {str(e)}")
            traceback.print_exc()

    def _find_similar_days(self, data, target_date, current_hour, conditions):
        """Find similar days with enhanced matching criteria"""
        try:
            similar_days = []
            
            # Handle different input data types
            if isinstance(data, pd.DataFrame):
                # If input is DataFrame, use timestamp column directly
                historical_data = data[data['timestamp'].dt.date < target_date].copy()
                dates = historical_data['timestamp'].dt.date.unique()
            else:
                # If input is prediction history list
                historical_data = pd.DataFrame(data)
                if 'date' in historical_data.columns:
                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                    dates = historical_data['date'].dt.date.unique()
                else:
                    print("No date information found in historical data")
                    return []
            
            for date in dates:
                if isinstance(data, pd.DataFrame):
                    day_data = historical_data[historical_data['timestamp'].dt.date == date]
                else:
                    day_data = historical_data[historical_data['date'].dt.date == date]
                
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
            traceback.print_exc()
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
            # Get radiation data based on available columns
            if 'Solar Rad - W/m^2' in day_data.columns:
                values = day_data['Solar Rad - W/m^2']
            elif 'predicted' in day_data.columns:
                values = day_data['predicted']
            else:
                print("No suitable radiation data found")
                return 0.0
            
            # Get hour data based on available columns
            if 'hour' in day_data.columns:
                hour_mask = day_data['hour'] <= current_hour
            elif 'timestamp' in day_data.columns:
                hour_mask = day_data['timestamp'].dt.hour <= current_hour
            else:
                print("No suitable hour data found")
                return 0.0
            
            pattern = values[hour_mask].values
            
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
                std_sim *0.3 +
                trend_sim *0.3
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error in _calculate_pattern_similarity: {str(e)}")
            traceback.print_exc()
            return 0.0

    def _get_pattern_prediction(self, current_value, hour):
        """Get pattern-based prediction with validation"""
        try:
            if hour >= 18 or hour <= 5:
                return 0.0
                
            if not isinstance(current_value, (int, float)) or not np.isfinite(current_value):
                return 0.0
                
            # Get historical patterns for this hour
            hour_patterns = self.hourly_patterns.get(hour, [])
            if not hour_patterns:
                return current_value
                
            # Filter valid patterns
            valid_patterns = [p['value'] for p in hour_patterns 
                             if isinstance(p['value'], (int, float)) and np.isfinite(p['value'])]
            
            if not valid_patterns:
                return current_value
                
            return float(np.mean(valid_patterns))
            
        except Exception as e:
            print(f"Error in pattern prediction: {str(e)}")
            return current_value

    def _get_ratio_prediction(self, similar_days, current_value):
        """Get prediction based on similar day ratios"""
        try:
            if similar_days is None or len(similar_days) == 0:
                return current_value
            
            ratios = []
            weights = []
            
            for day in similar_days:
                if isinstance(day, dict) and 'data' in day and 'similarity' in day:
                    day_data = day['data']
                    if not day_data.empty:
                        values = day_data['Solar Rad - W/m^2'].values
                        if len(values) >= 2:
                            ratio = values[-1] / (values[-2] + 1e-8)  # Add small value to prevent division by zero
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
            if current_day is None or current_hour is None:
                return 0
                
            # Check if current_day is a DataFrame and has the required columns
            if not isinstance(current_day, pd.DataFrame):
                return 0
            
            # Get hour column from the data
            if 'hour' in current_day.columns:
                hour_mask = current_day['hour'] <= current_hour
            else:
                # Try to extract hour from index or other timestamp column
                for col in ['timestamp', 'date', 'time']:
                    if col in current_day.columns:
                        current_day['hour'] = pd.to_datetime(current_day[col]).dt.hour
                        hour_mask = current_day['hour'] <= current_hour
                        break
                else:
                    print("No suitable timestamp column found")
                    return 0
            
            if not hour_mask.any():
                return 0
                
            # Get solar radiation data
            if 'Solar Rad - W/m^2' in current_day.columns:
                data = current_day.loc[hour_mask, 'Solar Rad - W/m^2'].values
            elif 'predicted' in current_day.columns:
                data = current_day.loc[hour_mask, 'predicted'].values
            else:
                print("No suitable radiation data column found")
                return 0
            
            if len(data) >= 2:
                x = np.arange(len(data))
                trend = np.polyfit(x, data, 1)[0]
                next_value = data[-1] + trend
                return max(0, float(next_value))
            
            return float(data[-1]) if len(data) > 0 else 0
                
        except Exception as e:
            print(f"Error in _get_trend_prediction: {str(e)}")
            traceback.print_exc()
            return 0

    def _get_typical_value(self, hour):
        """Get typical value for given hour"""
        try:
            if hour is None:
                return 0
            
            if hour in self.hourly_patterns:
                values = [p['value'] for p in self.hourly_patterns[hour]]
                if values:
                    return float(np.mean(values))
            return 0
            
        except Exception as e:
            print(f"Error in _get_typical_value: {str(e)}")
            return 0

    def _calculate_prediction_weights(self):
        """Get current weights from method performance"""
        return {method: float(perf['weight'])  # Convert np.float64 to Python float
                for method, perf in self.method_performance.items()}

    def _update_method_performance(self, predictions, actual):
        """Update performance metrics for each prediction method"""
        try:
            for method, perf in self.method_performance.items():
                if method in predictions:
                    error = float(abs(predictions[method] - actual))  # Convert to float
                    perf['errors'].append(error)
                    if len(perf['errors']) > perf['window_size']:
                        perf['errors'].pop(0)
                        
            # Update weights based on inverse error
            total_inverse_error = 0.0  # Use Python float
            new_weights = {}
            
            for method, perf in self.method_performance.items():
                if perf['errors']:
                    avg_error = float(np.mean(perf['errors'][-10:]))  # Convert to float
                    inverse_error = float(1 / (avg_error + 1e-8))  # Convert to float
                    total_inverse_error += inverse_error
                    new_weights[method] = inverse_error
            
            # Normalize and smooth weights
            if total_inverse_error > 0:
                for method in self.method_performance:
                    if method in new_weights:
                        normalized_weight = float(new_weights[method] / total_inverse_error)
                        current_weight = float(self.method_performance[method]['weight'])
                        # 70/30 blend of old and new weights
                        self.method_performance[method]['weight'] = float(
                            current_weight * 0.7 + normalized_weight * 0.3
                        )
                        
            print("\nUpdated method weights:")
            for method, perf in self.method_performance.items():
                print(f"{method}: {perf['weight']:.3f}")
                
            # Add ensemble performance tracking
            if 'ensemble' in predictions:
                ensemble_error = abs(predictions['ensemble'] - actual)
                if 'ensemble' not in self.method_performance:
                    self.method_performance['ensemble'] = {
                        'weight': 0.3,
                        'errors': [],
                        'window_size': 100
                    }
                self.method_performance['ensemble']['errors'].append(ensemble_error)
                
                # Adjust weights based on recent performance
                self._adjust_ensemble_weight()
    
        except Exception as e:
            print(f"Error updating method performance: {str(e)}")
            traceback.print_exc()

    def _adjust_ensemble_weight(self):
        """Dynamically adjust ensemble weight based on performance"""
        try:
            ensemble_errors = self.method_performance['ensemble']['errors'][-100:]
            other_methods_errors = []
            for method in ['pattern', 'ratio', 'trend', 'typical']:
                if method in self.method_performance:
                    other_methods_errors.extend(self.method_performance[method]['errors'][-100:])
            
            if ensemble_errors and other_methods_errors:
                ensemble_mae = np.mean(ensemble_errors)
                others_mae = np.mean(other_methods_errors)
                
                # Adjust weight based on relative performance
                if ensemble_mae < others_mae:
                    self.method_performance['ensemble']['weight'] = min(0.4, 
                        self.method_performance['ensemble']['weight'] * 1.1)
                else:
                    self.method_performance['ensemble']['weight'] = max(0.2,
                        self.method_performance['ensemble']['weight'] * 0.9)
        
        except Exception as e:
            print(f"Error adjusting ensemble weight: {str(e)}")

    def _validate_prediction(self, prediction, hour, conditions):
        """Validate prediction values"""
        try:
            if prediction is None:
                return None
                
            if not np.isfinite(prediction):
                return None
                
            # Get reasonable bounds based on hour and conditions
            max_expected = self._get_max_expected(hour, conditions)
            
            # Clip prediction to reasonable range
            prediction = np.clip(prediction, 0, max_expected)
            
            return float(prediction)
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return None

    def _store_prediction_with_history(self, date, hour, predicted, actual, conditions):
        """Store prediction with proper history tracking"""
        try:
            # Create prediction record
            prediction_record = {
                'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
                'hour': hour,
                'predicted': float(predicted) if predicted is not None else None,
                'actual': float(actual) if actual is not None else None,
                'conditions': conditions,
                'timestamp': pd.Timestamp.now()
            }
            
            # Calculate error metrics if actual value is available
            if actual is not None and predicted is not None:
                prediction_record['error'] = float(actual - predicted)
                if actual != 0:
                    prediction_record['error_percentage'] = (prediction_record['error'] / actual) * 100
                else:
                    prediction_record['error_percentage'] = 0.0 if predicted == 0 else 100.0
            
            # Store prediction without size limit
            self.prediction_history.append(prediction_record)
            
            print(f"\nStored prediction for {date}, hour {hour}:")
            print(f"Predicted: {predicted:.2f}")
            print(f"Actual: {actual if actual is not None else 'Not yet available'}")
            print(f"Total predictions stored: {len(self.prediction_history)}")
            
            return True
            
        except Exception as e:
            print(f"Error storing prediction: {str(e)}")
            traceback.print_exc()
            return False

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
            
            # Ensure stats folder exists
            os.makedirs(self.stats_folder, exist_ok=True)
            
            # Clear any existing plots
            plt.close('all')
            
            # Create figure with better memory management
            fig = plt.figure(figsize=(20, 15), dpi=100)
            plt.style.use('default')
            
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
            plot_path = os.path.join(os.path.abspath(self.stats_folder), f'learning_analysis_{timestamp}.png')
            
            # Save with reduced memory usage and print the path
            plt.savefig(plot_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            print(f"\nLearning analysis plot saved to: {plot_path}")
            
            # Generate detailed learning report with absolute path
            report_path = os.path.join(os.path.abspath(self.stats_folder), f'learning_report_{timestamp}.txt')
            self._generate_learning_report(history_df, hourly_improvement, report_path)
            print(f"Learning report saved to: {report_path}")
            
        except Exception as e:
            print(f"Error in analyze_learning_performance: {str(e)}")
            traceback.print_exc()

    def _generate_learning_report(self, history_df, hourly_stats, report_path):
        """Generate detailed learning report"""
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
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
                prediction_record['error_percentage'] = (prediction_record['error'] / actual * 100 if actual != 0 else 0)
            
            self._store_prediction(prediction_record)
            
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
        """Calculate dynamic learning rate with improved transitions"""
        try:
            # Enhanced base rates for different periods
            base_rates = {
                'dawn': 0.5,      # 5-6 hours (higher for better dawn adaptation)
                'morning': 0.4,   # 7-9 hours
                'peak': 0.3,      # 10-14 hours
                'afternoon': 0.45, # 15-17 hours (increased for better afternoon)
                'other': 0.2
            }
            
            # Determine period with enhanced transition handling
            if hour == 5 or hour == 6:
                base_rate = base_rates['dawn']
            elif 7 <= hour <= 9:
                base_rate = base_rates['morning']
            elif 10 <= hour <= 14:
                base_rate = base_rates['peak']
            elif 15 <= hour <= 17:
                base_rate = base_rates['afternoon']
            else:
                base_rate = base_rates['other']
                
            # Adjust based on recent error patterns
            if error_history:
                recent_errors = error_history[-5:]
                error_std = np.std(recent_errors)
                mean_error = np.mean(recent_errors)
                
                # Increase learning rate for consistent under-predictions
                if mean_error < -50:  # Consistent under-prediction
                    base_rate *= 1.5
                elif mean_error > 50:  # Consistent over-prediction
                    base_rate *= 1.3
                    
                # Adjust for high variability
                if error_std > 100:  # Very high variability
                    base_rate *= 1.4
                    
            return min(base_rate, 0.6)  # Increased max rate for faster adaptation
            
        except Exception as e:
            print(f"Error in _get_dynamic_learning_rate: {str(e)}")
            return 0.3  # Increased default rate

    def _calculate_weather_impact(self, conditions):
        """Improved weather impact calculation based on correlations"""
        impact = 1.0
        
        # Pressure impact (strongest correlation: 0.413)
        pressure = conditions['pressure']
        if pressure <= 1005.858:  # Highly negative impact zone
            impact *= 0.7
        elif pressure >= 1008.264:  # Positive impact zone
            impact *= 1.2
        
        # Temperature compensation (correlation: -0.269)
        temp = conditions['temperature']
        if temp > 30.038:  # High temperature zone with large errors
            impact *= 0.85
        elif temp < 27.388:  # More accurate predictions
            impact *= 1.1
        
        # UV-based adjustment (helps with daytime accuracy)
        uv = conditions['uv']
        if uv > 5.0:  # High UV indicates clear sky
            impact = min(impact * 1.1, 1.2)  # Cap the increase
        
        return impact

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

    def _get_moving_average(self, data, target_date, current_hour):
        """Calculate moving average with better handling of zero values"""
        try:
            next_hour = (current_hour + 1) % 24
            if next_hour >= 18 or next_hour <= 5:
                return 0.0

            historical_data = data[data['timestamp'].dt.date <= target_date].copy()
            current_mask = (historical_data['timestamp'].dt.date == target_date) & \
                          (historical_data['timestamp'].dt.hour <= current_hour)
            historical_data = historical_data[~current_mask]
            
            if historical_data.empty:
                return 0.0
            
            # Filter out zero values for better averages
            valid_data = historical_data[historical_data['Solar Rad - W/m^2'] > 0]
            if valid_data.empty:
                return 0.0
            
            values = valid_data['Solar Rad - W/m^2'].values
            
            # Calculate averages only from non-zero values
            day_avg = np.mean(values[-24:]) if len(values) >= 24 else np.mean(values)
            
            hour_data = valid_data[valid_data['timestamp'].dt.hour == current_hour]
            hour_avg = hour_data['Solar Rad - W/m^2'].mean() if not hour_data.empty else day_avg
            
            recent_avg = np.mean(values[-3:]) if len(values) >= 3 else np.mean(values)
            
            # Print components for debugging
            print("\nMoving Average Components:")
            print(f"Hour average: {hour_avg:.2f}")
            print(f"Day average: {day_avg:.2f}")
            print(f"Recent average: {recent_avg:.2f}")
            
            # Weighted combination
            moving_avg = (
                hour_avg * 0.5 +
                day_avg * 0.3 +
                recent_avg * 0.2
            )
            
            print(f"Combined moving average: {moving_avg:.2f}")
            return float(moving_avg)
            
        except Exception as e:
            print(f"Error in moving average calculation: {str(e)}")
            traceback.print_exc()
            return 0.0

    def _get_hour_statistics(self, hour):
        """Get statistical data for a specific hour"""
        try:
            if not self.prediction_history:
                return None
                
            # Convert prediction history to DataFrame
            history_df = pd.DataFrame(self.prediction_history)
            
            # Filter data for the specified hour
            hour_data = history_df[history_df['hour'] == hour]
            
            if hour_data.empty:
                return None
                
            # Calculate statistics
            stats = {
                'mean': float(hour_data['predicted'].mean()),
                'std': float(hour_data['predicted'].std()),
                'min': float(hour_data['predicted'].min()),
                'max': float(hour_data['predicted'].max()),
                'count': len(hour_data)
            }
            
            # Add recent performance if available
            recent_data = hour_data.tail(24)  # Last 24 predictions for this hour
            if not recent_data.empty:
                stats['recent_mean'] = float(recent_data['predicted'].mean())
                stats['recent_std'] = float(recent_data['predicted'].std())
            
            return stats
            
        except Exception as e:
            print(f"Error in _get_hour_statistics: {str(e)}")
            traceback.print_exc()
            return None

    def generate_detailed_learning_report(self, report_period='all'):
        """Generate comprehensive learning report with enhanced saving"""
        try:
            # Add validation for report_period
            valid_periods = ['all', 'day', 'week', 'month']
            if report_period not in valid_periods:
                raise ValueError(f"Invalid report period. Must be one of {valid_periods}")

            # Add check for minimum data requirements
            if len(self.prediction_history) < 24:
                print("Warning: Limited data available for analysis (less than 24 predictions)")

            if not self.prediction_history:
                print("No prediction history available for analysis")
                return

            # Convert history to DataFrame
            history_df = pd.DataFrame(self.prediction_history)
            history_df['date'] = pd.to_datetime(history_df['date'])

            # Filter data based on report period
            if report_period != 'all':
                cutoff_date = pd.Timestamp.now()
                if report_period == 'day':
                    cutoff_date -= timedelta(days=1)
                elif report_period == 'week':
                    cutoff_date -= timedelta(days=7)
                elif report_period == 'month':
                    cutoff_date -= timedelta(days=30)
                history_df = history_df[history_df['date'] >= cutoff_date]

            # Create report directory with timestamp and period
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            report_name = f'learning_report_{report_period}_{timestamp}'
            report_dir = Path(self.reports_folder) / report_name
            report_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nCreating report directory: {report_dir}")

            # Save data files
            history_df = pd.DataFrame(self.prediction_history)
            
            # Save raw data
            data_file = report_dir / 'prediction_data.csv'
            history_df.to_csv(data_file, index=False)
            print(f"Saved prediction data to: {data_file}")
            
            # Save processed data with additional metrics
            processed_df = history_df.copy()
            processed_df['timestamp'] = pd.to_datetime(processed_df['date'])
            processed_df['hour_of_day'] = processed_df['hour']
            processed_df['prediction_accuracy'] = 100 - abs(processed_df['error_percentage'])
            
            processed_file = report_dir / 'processed_data.csv'
            processed_df.to_csv(processed_file, index=False)
            print(f"Saved processed data to: {processed_file}")

            # Generate main report file
            report_path = report_dir / 'detailed_report.txt'
            print(f"Generating report file: {report_path}")

            # Open file with UTF-8 encoding
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Detailed Learning Report ({report_period}) ===\n")
                f.write(f"Generated: {pd.Timestamp.now()}\n\n")
                
                # Overall Statistics
                f.write("1. Overall Performance Metrics\n")
                f.write("============================\n")
                self._write_overall_metrics(f, history_df)

                # Hourly Performance
                f.write("\n2. Hourly Performance Analysis\n")
                f.write("============================\n")
                self._write_hourly_analysis(f, history_df)

                # Weather Impact Analysis
                f.write("\n3. Weather Impact Analysis\n")
                f.write("========================\n")
                self._write_weather_analysis(f, history_df)

                # Learning Progress
                f.write("\n4. Learning Progress Analysis\n")
                f.write("==========================\n")
                self._write_learning_progress(f, history_df)

                # Error Pattern Analysis
                f.write("\n5. Error Pattern Analysis\n")
                f.write("=======================\n")
                self._write_error_patterns(f, history_df)

            # Generate JSON metrics file
            metrics_path = report_dir / 'performance_metrics.json'
            self._save_performance_metrics(metrics_path, history_df)

            # Generate visualizations
            self._generate_performance_plots(report_dir, history_df)

            print(f"\nDetailed learning report generated at: {report_dir}")

            # Print confirmation of saved files
            print(f"\nReport files saved:")
            print(f"Main report: {report_path}")
            print(f"Metrics: {report_dir / 'performance_metrics.json'}")
            print(f"Plots: {report_dir}")

            # Create a ZIP archive of the report directory
            import shutil
            zip_path = Path(self.reports_folder) / f'{report_name}.zip'
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', report_dir)
            print(f"\nCreated ZIP archive of report: {zip_path}")

            return report_dir

        except Exception as e:
            print(f"Error generating detailed report: {str(e)}")
            traceback.print_exc()

    def _write_overall_metrics(self, f, history_df):
        """Write overall performance metrics to report"""
        try:
            # Convert date strings to datetime
            history_df['date'] = pd.to_datetime(history_df['date'])
            
            # Basic metrics using full history
            metrics = {
                'Total Predictions': len(history_df),
                'Mean Absolute Error': history_df['error'].abs().mean(),
                'RMSE': np.sqrt((history_df['error'] ** 2).mean()),
                'Mean Absolute Percentage Error': history_df['error_percentage'].abs().mean(),
                'Bias (Mean Error)': history_df['error'].mean(),
                'Error Standard Deviation': history_df['error'].std()
            }

            # Write metrics with proper formatting
            f.write("1. Overall Performance Metrics\n")
            f.write("============================\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.2f}\n")

            # Add prediction accuracy bands
            accuracy_bands = {
                '< 5%': (history_df['error_percentage'].abs() < 5).mean() * 100,
                '5-10%': ((history_df['error_percentage'].abs() >= 5) & 
                          (history_df['error_percentage'].abs() < 10)).mean() * 100,
                '10-20%': ((history_df['error_percentage'].abs() >= 10) & 
                           (history_df['error_percentage'].abs() < 20)).mean() * 100,
                '> 20%': (history_df['error_percentage'].abs() >= 20).mean() * 100
            }

            f.write("\nPrediction Accuracy Bands:\n")
            for band, percentage in accuracy_bands.items():
                f.write(f"  {band}: {percentage:.1f}%\n")

            # Add date range information with proper datetime handling
            f.write("\nPrediction Period:\n")
            start_date = history_df['date'].min()
            end_date = history_df['date'].max()
            total_days = (end_date - start_date).days + 1
            
            f.write(f"Start Date: {start_date.strftime('%Y-%m-%d')}\n")
            f.write(f"End Date: {end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Total Days: {total_days}\n")

        except Exception as e:
            print(f"Error calculating overall metrics: {str(e)}")
            traceback.print_exc()
            f.write(f"\nError calculating overall metrics: {str(e)}\n")

    def _write_hourly_analysis(self, f, history_df):
        """Write detailed hourly performance analysis"""
        try:
            hourly_stats = history_df.groupby('hour').agg({
                'error': ['mean', 'std', 'count'],
                'error_percentage': ['mean', 'std'],
                'predicted': ['mean', 'min', 'max'],
                'actual': ['mean', 'min', 'max']
            }).round(2)

            f.write("\nHourly Performance Breakdown:\n")
            for hour in range(24):
                if hour in hourly_stats.index:
                    stats = hourly_stats.loc[hour]
                    f.write(f"\nHour {hour:02d}:00\n")
                    f.write(f"  Predictions: {stats[('error', 'count')]:.0f}\n")
                    f.write(f"  Mean Error: {stats[('error', 'mean')]:.2f} W/m²\n")
                    f.write(f"  Error Std: {stats[('error', 'std')]:.2f} W/m²\n")
                    f.write(f"  Mean % Error: {stats[('error_percentage', 'mean')]:.2f}%\n")
                    f.write(f"  Prediction Range: [{stats[('predicted', 'min')]:.0f}, {stats[('predicted', 'max')]:.0f}] W/m²\n")
                    f.write(f"  Actual Range: [{stats[('actual', 'min')]:.0f}, {stats[('actual', 'max')]:.0f}] W/m²\n")

        except Exception as e:
            f.write(f"\nError in hourly analysis: {str(e)}\n")

    def _write_weather_analysis(self, f, history_df):
        """Analyze and write weather condition impacts with improved correlation analysis"""
        try:
            # Convert conditions from string to dict if needed
            if 'conditions' not in history_df.columns:
                f.write("\nNo weather conditions data available for analysis\n")
                return

            # Convert conditions to DataFrame, handling both string and dict formats
            conditions_df = pd.DataFrame([
                eval(cond) if isinstance(cond, str) else cond 
                for cond in history_df['conditions']
            ])
            
            # Ensure numeric values for correlation analysis
            for param in ['temperature', 'humidity', 'pressure', 'uv']:
                if param in conditions_df.columns:
                    conditions_df[param] = pd.to_numeric(conditions_df[param], errors='coerce')
            
            # Analyze each weather parameter
            for param in ['temperature', 'humidity', 'pressure', 'uv']:
                if param in conditions_df.columns:
                    f.write(f"\n{param.capitalize()} Impact Analysis:\n")
                    
                    # Handle parameters with potential duplicate values
                    unique_values = conditions_df[param].nunique()
                    if unique_values < 5:
                        # Use value_counts for parameters with few unique values
                        impact_analysis = history_df.groupby(conditions_df[param])['error_percentage'].agg([
                            'mean', 'std', 'count'
                        ]).round(2)
                    else:
                        try:
                            # Create bins with duplicate handling
                            bins = pd.qcut(conditions_df[param], q=5, duplicates='drop')
                            impact_analysis = history_df.groupby(bins)['error_percentage'].agg([
                                'mean', 'std', 'count'
                            ]).round(2)
                        except ValueError:
                            # Fallback to custom bins if qcut fails
                            min_val = conditions_df[param].min()
                            max_val = conditions_df[param].max()
                            custom_bins = np.linspace(min_val, max_val, 6)
                            bins = pd.cut(conditions_df[param], bins=custom_bins)
                            impact_analysis = history_df.groupby(bins)['error_percentage'].agg([
                                'mean', 'std', 'count'
                            ]).round(2)
                        
                        for bin_range, stats in impact_analysis.iterrows():
                            f.write(f"\n  Range {bin_range}:\n")
                            f.write(f"    Count: {stats['count']:.0f}\n")
                            f.write(f"    Mean Error %: {stats['mean']:.2f}%\n")
                            f.write(f"    Error Std %: {stats['std']:.2f}%\n")

            # Improved correlation analysis
            f.write("\nWeather Parameter Correlations:\n")
            for param in ['temperature', 'humidity', 'pressure', 'uv']:
                if param in conditions_df.columns:
                    # Remove any NaN values for correlation calculation
                    valid_mask = ~(conditions_df[param].isna() | history_df['error_percentage'].isna())
                    if valid_mask.any():
                        param_values = conditions_df[param][valid_mask].values
                        error_values = history_df['error_percentage'][valid_mask].values
                        if len(param_values) > 1 and len(error_values) > 1:
                            correlation = np.corrcoef(param_values, error_values)[0,1]
                            f.write(f"{param.capitalize()} correlation with error: {correlation:.3f}\n")
                        else:
                            f.write(f"{param.capitalize()} correlation with error: insufficient data points\n")
                    else:
                        f.write(f"{param.capitalize()} correlation with error: insufficient valid data\n")

            # Add additional weather impact summary
            f.write("\nWeather Impact Summary:\n")
            for param in ['temperature', 'humidity', 'pressure', 'uv']:
                if param in conditions_df.columns:
                    param_data = conditions_df[param]
                    error_data = history_df['error_percentage']
                    
                    # Calculate average error for high and low values
                    median_value = param_data.median()
                    high_mask = param_data > median_value
                    low_mask = param_data <= median_value
                    
                    high_error = error_data[high_mask].mean()
                    low_error = error_data[low_mask].mean()
                    
                    f.write(f"\n{param.capitalize()}:\n")
                    f.write(f"  High values (>{median_value:.1f}): {high_error:.2f}% mean error\n")
                    f.write(f"  Low values (<={median_value:.1f}): {low_error:.2f}% mean error\n")
                    f.write(f"  Impact difference: {abs(high_error - low_error):.2f}%\n")

        except Exception as e:
            f.write(f"\nError in weather analysis: {str(e)}\n")
            traceback.print_exc()

    def _write_learning_progress(self, f, history_df):
        """Analyze and write learning progress over time with proper window handling"""
        try:
            # Calculate rolling metrics with data length validation
            window_sizes = [24, 72, 168]  # 1 day, 3 days, 1 week
            
            f.write("\nLearning Progress Analysis:\n")
            for window in window_sizes:
                # Check if we have enough data for this window size
                if len(history_df) > window:
                    rolling_mae = history_df['error'].abs().rolling(window, min_periods=1).mean()
                    rolling_mape = history_df['error_percentage'].abs().rolling(window, min_periods=1).mean()
                    
                    f.write(f"\n{window}-hour Rolling Window:\n")
                    
                    # Get initial values (using first available value after window)
                    initial_idx = min(window, len(rolling_mae) - 1)
                    initial_mae = rolling_mae.iloc[initial_idx]
                    initial_mape = rolling_mape.iloc[initial_idx]
                    
                    # Get final values
                    final_mae = rolling_mae.iloc[-1]
                    final_mape = rolling_mape.iloc[-1]
                    
                    f.write(f"  Initial MAE: {initial_mae:.2f} W/m²\n")
                    f.write(f"  Final MAE: {final_mae:.2f} W/m²\n")
                    f.write(f"  Initial MAPE: {initial_mape:.2f}%\n")
                    f.write(f"  Final MAPE: {final_mape:.2f}%\n")
                    
                    # Calculate improvement
                    if initial_mae != 0:
                        mae_improvement = ((initial_mae - final_mae) / initial_mae * 100)
                        f.write(f"  MAE Improvement: {mae_improvement:.1f}%\n")
                else:
                    f.write(f"\n{window}-hour Rolling Window: Insufficient data (need >{window} points)\n")

        except Exception as e:
            f.write(f"\nError in learning progress analysis: {str(e)}\n")
            traceback.print_exc()

    def _write_error_patterns(self, f, history_df):
        """Analyze and write error patterns"""
        try:
            f.write("\nError Pattern Analysis:\n")

            # Analyze consecutive errors
            history_df['error_direction'] = np.sign(history_df['error'])
            consecutive_errors = (history_df['error_direction'] != 
                                history_df['error_direction'].shift()).cumsum()
            error_runs = history_df.groupby(consecutive_errors)['error_direction'].agg(['count', 'first'])
            
            f.write("\nConsecutive Error Patterns:\n")
            f.write("  Over-predictions (>3 consecutive):\n")
            over_runs = error_runs[
                (error_runs['first'] < 0) & (error_runs['count'] > 3)]['count'].value_counts()
            for length, count in over_runs.items():
                f.write(f"    {length} consecutive: {count} occurrences\n")
                
            f.write("\n  Under-predictions (>3 consecutive):\n")
            under_runs = error_runs[
                (error_runs['first'] > 0) & (error_runs['count'] > 3)]['count'].value_counts()
            for length, count in under_runs.items():
                f.write(f"    {length} consecutive: {count} occurrences\n")

        except Exception as e:
            f.write(f"\nError in error pattern analysis: {str(e)}\n")

    def _save_performance_metrics(self, metrics_path, history_df):
        try:
            # Create hourly stats dictionary first
            hourly_stats = {}
            for hour, hour_data in history_df.groupby('hour'):
                hourly_stats[str(hour)] = {
                    'mae': float(hour_data['error'].abs().mean()),
                    'mape': float(hour_data['error_percentage'].abs().mean()),
                    'count': int(len(hour_data))
            }
            
            # Then use it in the main metrics dictionary
            metrics = {
                'metadata': {
                    'generated_at': str(pd.Timestamp.now()),
                    'version': '1.0',
                    'data_points': len(history_df),
                    'date_range': {
                        'start': str(history_df['date'].min()),
                        'end': str(history_df['date'].max())
                    }
                },
                'overall': {
                    'total_predictions': len(history_df),
                    'mae': float(history_df['error'].abs().mean()),
                    'rmse': float(np.sqrt((history_df['error'] ** 2).mean())),
                    'mape': float(history_df['error_percentage'].abs().mean()),
                    'bias': float(history_df['error'].mean())
                },
                'hourly': hourly_stats,
                'learning_progress': {
                    'initial_mae': float(history_df['error'].abs().iloc[:24].mean()),
                    'final_mae': float(history_df['error'].abs().iloc[-24:].mean()),
                    'improvement_percentage': float(
                        (history_df['error'].abs().iloc[:24].mean() - 
                         history_df['error'].abs().iloc[-24:].mean()) / 
                        history_df['error'].abs().iloc[:24].mean() * 100)
                    }
            }

            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            print(f"Error saving performance metrics: {str(e)}")

    def _generate_performance_plots(self, report_dir, history_df):
        """Generate comprehensive performance visualization plots with robust data validation"""
        try:
            plot_files = []
            
            # Deep copy to avoid modifying original data
            plot_df = history_df.copy()
            
            # Ensure date column is datetime
            if 'date' in plot_df.columns:
                plot_df['date'] = pd.to_datetime(plot_df['date'])
            
            # Clean and validate all numeric columns
            numeric_columns = ['error', 'error_percentage', 'predicted', 'actual']
            for col in numeric_columns:
                if col in plot_df.columns:
                    # Replace infinite values
                    plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
                    # Fill NaN values with column median
                    median_value = plot_df[col].median()
                    plot_df[col] = plot_df[col].fillna(median_value)
                    # Clip extreme values to reasonable ranges
                    if 'percentage' in col:
                        plot_df[col] = plot_df[col].clip(-100, 100)
                    else:
                        q1, q99 = plot_df[col].quantile([0.01, 0.99])
                        plot_df[col] = plot_df[col].clip(q1, q99)
            
            # 1. Error Distribution Plot
            plt.figure(figsize=(10, 6))
            valid_errors = plot_df['error_percentage'].dropna()
            if len(valid_errors) > 0:
                sns.histplot(data=valid_errors, bins=50)
                plt.title('Error Distribution')
                plt.xlabel('Error Percentage')
                plt.savefig(report_dir / 'error_distribution.png')
            plt.close()
            
            # 2. Learning Progress Plot
            plt.figure(figsize=(10, 6))
            plot_df['rolling_mae'] = plot_df['error'].abs().rolling(24, min_periods=1).mean()
            valid_indices = np.arange(len(plot_df))
            valid_mae = plot_df['rolling_mae'].dropna()
            if len(valid_mae) > 0:
                plt.plot(valid_indices[-len(valid_mae):], valid_mae)
                plt.title('Learning Progress (24-hour Rolling MAE)')
                plt.ylabel('Mean Absolute Error (W/m²)')
                plt.xlabel('Prediction Number')
                plt.savefig(report_dir / 'learning_progress.png')
            plt.close()
            
            # 3. Hourly Performance Heatmap
            plt.figure(figsize=(12, 8))
            try:
                hourly_errors = plot_df.pivot_table(
                    values='error_percentage',
                    index=plot_df.index // 24,
                    columns='hour',
                    aggfunc='mean'
                )
                if not hourly_errors.empty:
                    sns.heatmap(hourly_errors, cmap='RdYlBu_r', center=0)
                    plt.title('Hourly Error Patterns')
                    plt.xlabel('Hour of Day')
                    plt.ylabel('Day Number')
                    plt.savefig(report_dir / 'hourly_heatmap.png')
            except Exception as e:
                print(f"Error generating heatmap: {str(e)}")
            plt.close()
            
            # 4. Prediction vs Actual Scatter Plot
            plt.figure(figsize=(10, 10))
            valid_mask = (
                plot_df['actual'].notna() & 
                plot_df['predicted'].notna() & 
                np.isfinite(plot_df['actual']) & 
                np.isfinite(plot_df['predicted']))
            if valid_mask.any():
                valid_data = plot_df[valid_mask]
                plt.scatter(
                    valid_data['actual'],
                    valid_data['predicted'],
                    alpha=0.5,
                    s=20  # Smaller point size
                )
                # Add identity line
                min_val = min(valid_data['actual'].min(), valid_data['predicted'].min())
                max_val = max(valid_data['actual'].max(), valid_data['predicted'].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                plt.title('Predicted vs Actual Values')
                plt.xlabel('Actual (W/m²)')
                plt.ylabel('Predicted (W/m²)')
                plt.legend()
                plt.savefig(report_dir / 'prediction_scatter.png')
            plt.close()
            
            print(f"\nGenerated plots saved to: {report_dir}")
            
        except Exception as e:
            print(f"Error generating performance plots: {str(e)}")
            traceback.print_exc()

    def _initialize_directories(self):
        """Initialize all required directories and file paths for saving reports and data"""
        try:
            # Create base directories with absolute paths
            self.base_dir = os.path.abspath(os.path.join(os.getcwd(), 'predictions'))
            self.data_folder = os.path.join(self.base_dir, 'data')
            self.stats_folder = os.path.join(self.base_dir, 'stats')
            self.reports_folder = os.path.join(self.base_dir, 'reports')
            self.models_folder = os.path.join(self.base_dir, 'models')
            self.history_folder = os.path.join(self.base_dir, 'history')  # Add history folder

            # Initialize file paths
            self.learning_state_file = os.path.join(self.models_folder, 'learning_state.pkl')
            self.patterns_file = os.path.join(self.models_folder, 'learned_patterns.pkl')
            self.errors_file = os.path.join(self.stats_folder, 'prediction_errors.csv')
            self.stats_file = os.path.join(self.stats_folder, 'hourly_stats.json')

            # Create all directories
            for directory in [self.base_dir, self.data_folder, self.stats_folder, 
                             self.reports_folder, self.models_folder, self.history_folder]:  # Add history_folder
                os.makedirs(directory, exist_ok=True)
                print(f"Directory created/verified: {directory}")

            print("\nInitialized file paths:")
            print(f"Learning state: {self.learning_state_file}")
            print(f"Patterns file: {self.patterns_file}")
            print(f"Errors file: {self.errors_file}")
            print(f"Stats file: {self.stats_file}")
            print(f"History folder: {self.history_folder}")  # Add print statement

        except Exception as e:
            print(f"Error initializing directories: {str(e)}")
            traceback.print_exc()

    def _update_weather_impacts(self, conditions):
        """Enhanced weather impact calculation"""
        impacts = {}
        
        # Pressure has strongest correlation (0.502)
        pressure_impact = self._calculate_pressure_impact(conditions['pressure'])
        impacts['pressure'] = pressure_impact * 0.4  # Increased weight
        
        # Humidity has significant impact (129.57% difference)
        humidity_impact = self._calculate_humidity_impact(conditions['humidity'])
        impacts['humidity'] = humidity_impact * 0.3
        
        # Temperature and UV impacts
        impacts['temperature'] = self._calculate_temperature_impact(conditions['temperature']) * 0.2
        impacts['uv'] = self._calculate_uv_impact(conditions['uv']) * 0.1
        
        return impacts

    def _adjust_prediction(self, initial_prediction, hour, conditions):
        """Add pattern-based correction"""
        if self.error_learner.pattern_adjustments.get(str(hour), {}).get('consecutive_under', 0) > 3:
            # Compensate for persistent under-prediction
            adjustment_factor = 1.15
        elif self.error_learner.pattern_adjustments.get(str(hour), {}).get('consecutive_over', 0) > 3:
            # Compensate for persistent over-prediction
            adjustment_factor = 0.85
        else:
            adjustment_factor = 1.0
            
        return initial_prediction * adjustment_factor

    def _adjust_for_error_patterns(self, hour, prediction):
        """Enhanced error pattern correction"""
        consecutive_unders = self.error_learner.pattern_adjustments.get(str(hour), {}).get('consecutive_under', 0)
        consecutive_overs = self.error_learner.pattern_adjustments.get(str(hour), {}).get('consecutive_over', 0)
        
        if consecutive_unders > 8:  # Long streak of under-predictions
            adjustment = 1.25
        elif consecutive_unders > 4:
            adjustment = 1.15
        elif consecutive_overs > 4:
            adjustment = 0.85
        else:
            adjustment = 1.0
            
            # Add time-of-day specific adjustments
        if 6 <= hour <= 8:  # Morning hours (severe under-prediction)
            adjustment *= 1.2
        elif 15 <= hour <= 17:  # Evening hours (severe over-prediction)
            adjustment *= 0.8
        
        return prediction * adjustment

    def _generate_learning_analysis_plot(self, save_path):
        try:
            # Create figure with specific DPI
            plt.figure(figsize=(15, 10), dpi=100)
            
            # Get clean data
            history_df = pd.DataFrame(self.prediction_history)
            if history_df.empty:
                return
                
            # Convert error column to numeric and clean
            history_df['error'] = pd.to_numeric(history_df['error'], errors='coerce')
            history_df = history_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['error'])
            
            if len(history_df) < 2:
                return
                
            # Calculate rolling metrics with validation
            for window, color, label in zip([24, 72, 168], ['blue', 'green', 'red'], 
                                          ['24h', '72h', '168h']):
                mae = history_df['error'].abs().rolling(window=window, min_periods=1).mean()
                valid_mask = mae.notna() & np.isfinite(mae)
                
                if valid_mask.any():
                    x_vals = np.arange(len(mae))[valid_mask]
                    y_vals = mae[valid_mask].values
                    
                    # Remove outliers
                    q1, q99 = np.percentile(y_vals, [1, 99])
                    y_vals = np.clip(y_vals, q1, q99)
                    
                    plt.plot(x_vals, y_vals, color=color, label=f'{label} MAE', alpha=0.7)
            
            plt.title('Learning Progress')
            plt.xlabel('Prediction Number')
            plt.ylabel('Mean Absolute Error')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Plot error: {str(e)}")
            plt.close()

    def _calculate_transition_adjustments(self, hour, conditions):
        """Enhanced transition period handling"""
        try:
            # Dawn transition (5-8 AM)
            if 5 <= hour <= 8:
                base = {
                    5: 1.2,  # Gentle start
                    6: 1.5,  # Steeper increase
                    7: 1.8,  # Peak adjustment
                    8: 1.4   # Tapering off
                }
                adjustment = base.get(hour, 1.0)
                
                # Modify based on conditions
                if conditions['pressure'] > 1007.6:  # High pressure improves accuracy
                    adjustment *= 1.2
                if conditions['humidity'] > 75:  # High humidity needs more boost
                    adjustment *= 1.15
                    
            # Dusk transition (15-17)
            elif 15 <= hour <= 17:
                base = {
                    15: 0.85,  # Gentle decrease
                    16: 0.70,  # Steeper decrease
                    17: 0.50   # Sharp reduction
                }
                adjustment = base.get(hour, 1.0)
                
                # Modify based on conditions
                if conditions['humidity'] > 70:  # High humidity needs more reduction
                    adjustment *= 0.85
                    
            else:
                adjustment = 1.0
                
            return adjustment
            
        except Exception as e:
            print(f"Error in transition adjustment: {str(e)}")
            return 1.0

    def _calculate_error_percentage(self, actual, predicted):
        """Calculate error percentage with proper handling of zero values"""
        try:
            if actual == 0 and predicted == 0:
                return 0.0
            elif actual == 0:
                return 100.0 if predicted > 0 else 0.0
            else:
                return ((actual - predicted) / actual) * 100
        except Exception as e:
            print(f"Error calculating percentage: {str(e)}")
            return float('nan')

    def _update_method_weights(self, predictions, actual):
        """Update method weights with robust error handling"""
        try:
            if actual is None or not np.isfinite(actual):
                return
                
            errors = {}
            for method, pred in predictions.items():
                if pred is not None and np.isfinite(pred):
                    error = abs(actual - pred)
                    errors[method] = error
                else:
                    errors[method] = float('inf')
                    
            # Calculate weights inversely proportional to errors
            total_inverse_error = sum(1/e if e > 0 else 1 for e in errors.values())
            
            if total_inverse_error > 0:
                weights = {
                    method: (1/error)/total_inverse_error if error > 0 else 1/len(errors)
                    for method, error in errors.items()
                }
            else:
                # Equal weights if all errors are zero
                weights = {method: 1/len(errors) for method in errors}
                
            return weights
            
        except Exception as e:
            print(f"Error updating weights: {str(e)}")
            return None

    def _calculate_weather_adjustment(self, conditions, hour):
        """Enhanced weather impact calculation"""
        try:
            # Base adjustments
            pressure_impact = self._get_pressure_impact(conditions['pressure'])
            humidity_impact = self._get_humidity_impact(conditions['humidity'], hour)
            temp_impact = self._get_temperature_impact(conditions['temperature'], hour)
            
            # Weight impacts based on time of day
            if 6 <= hour <= 8:  # Morning
                weights = {'pressure': 0.5, 'humidity': 0.3, 'temperature': 0.2}
            elif 11 <= hour <= 13:  # Peak hours
                weights = {'pressure': 0.3, 'humidity': 0.4, 'temperature': 0.3}
            elif 15 <= hour <= 17:  # Evening
                weights = {'pressure': 0.2, 'humidity': 0.5, 'temperature': 0.3}
            else:
                weights = {'pressure': 0.4, 'humidity': 0.4, 'temperature': 0.2}
                
            adjustment = (
                pressure_impact * weights['pressure'] +
                humidity_impact * weights['humidity'] +
                temp_impact * weights['temperature']
            )
            
            return np.clip(adjustment, 0.5, 1.5)
            
        except Exception as e:
            print(f"Error in weather adjustment: {str(e)}")
            return 1.0

    def _get_current_conditions(self, data, target_date, current_hour):
        """Get current conditions with validation"""
        try:
            # Get current day's data
            current_day = data[data['timestamp'].dt.date == target_date]
            if current_day.empty:
                print(f"No data available for date {target_date}")
                return None
                
            # Get current hour data
            current_data = current_day[current_day['timestamp'].dt.hour == current_hour]
            if current_data.empty:
                print(f"No data available for hour {current_hour}")
                return None
                
            # Extract conditions
            conditions = {
                'hour': int(current_hour),
                'temperature': float(current_data['Average Temperature'].iloc[0]),
                'humidity': float(current_data['Average Humidity'].iloc[0]),
                'pressure': float(current_data['Average Barometer'].iloc[0]),
                'uv': float(current_data['UV Index'].iloc[0]),
                'solar_rad': float(current_data['Solar Rad - W/m^2'].iloc[0]),
                'clear_sky_rad': float(calculate_clear_sky_radiation(
                    hour=current_hour,
                    latitude=ADDU_LATITUDE,
                    longitude=ADDU_LONGITUDE,
                    date=target_date,
                    temperature=current_data['Average Temperature'].iloc[0],
                    humidity=current_data['Average Humidity'].iloc[0],
                    pressure=current_data['Average Barometer'].iloc[0]
                ))
            }
            
            return conditions
            
        except Exception as e:
            print(f"Error getting current conditions: {str(e)}")
            traceback.print_exc()
            return None

    # Add this method inside the AutomatedPredictor class
    def _get_all_predictions(self, data, target_date, current_hour, conditions):
        """Get predictions from all available methods"""
        try:
            predictions = {}
            
            # Get current value
            current_data = data[
                (data['timestamp'].dt.date == target_date) & 
                (data['timestamp'].dt.hour == current_hour)
            ]  # Add closing bracket here

            if current_data.empty:
                print("No current data available")
                return None
                
            current_value = float(current_data['Solar Rad - W/m^2'].iloc[0])
            
            # Calculate clear sky radiation for next hour
            next_hour = (current_hour + 1) % 24
            clear_sky = calculate_clear_sky_radiation(
                hour=next_hour,
                latitude=ADDU_LATITUDE,
                longitude=ADDU_LONGITUDE,
                date=target_date,
                temperature=conditions['temperature'],
                humidity=conditions['humidity'],
                pressure=conditions['pressure']
            )
            
            # Get main model prediction
            main_prediction, weights, adjustment = self._get_main_prediction(
                current_value, conditions, clear_sky
            )
            predictions['main_model'] = main_prediction
            
            # Get moving average prediction
            moving_avg = self._get_moving_average(data, target_date, current_hour)
            predictions['moving_avg'] = moving_avg
            
            # Get pattern-based prediction
            pattern_pred = self._get_pattern_prediction(current_value, next_hour)
            predictions['pattern'] = pattern_pred
            
            # Add clear sky prediction
            predictions['clear_sky'] = clear_sky * 0.85  # Typical clear sky factor
            
            # Validate predictions
            for method, pred in predictions.items():
                if pred is None or not np.isfinite(pred):
                    print(f"Invalid prediction from {method}")
                    predictions[method] = current_value
            
            print("\nPrediction components:")
            for method, pred in predictions.items():
                print(f"{method}: {pred:.2f}")
                
            return predictions
            
        except Exception as e:
            print(f"Error in _get_all_predictions: {str(e)}")
            traceback.print_exc()
            return None

    # Add these methods inside the AutomatedPredictor class

    def _get_pressure_impact(self, pressure):
        """Calculate impact of pressure on solar radiation"""
        try:
            # Pressure thresholds based on analysis
            if pressure <= 1005.0:  # Low pressure system
                return 0.8
            elif pressure >= 1015.0:  # High pressure system
                return 1.2
            else:
                # Linear interpolation between thresholds
                return 0.8 + (pressure - 1005.0) * 0.04 / 10.0
                
        except Exception as e:
            print(f"Error in pressure impact calculation: {str(e)}")
            return 1.0

    def _get_humidity_impact(self, humidity, hour):
        """Calculate impact of humidity on solar radiation"""
        try:
            # Different impacts based on time of day
            if 6 <= hour <= 9:  # Morning hours
                if humidity > 85:
                    return 0.7
                elif humidity > 70:
                    return 0.85
                else:
                    return 1.0
            elif 10 <= hour <= 14:  # Peak hours
                if humidity > 80:
                    return 0.8
                elif humidity > 65:
                    return 0.9
                else:
                    return 1.0
            else:  # Other hours
                if humidity > 75:
                    return 0.85
                else:
                    return 1.0
                
        except Exception as e:
            print(f"Error in humidity impact calculation: {str(e)}")
            return 1.0

    def _get_temperature_impact(self, temperature, hour):
        """Calculate impact of temperature on solar radiation"""
        try:
            # Temperature thresholds vary by time of day
            if 10 <= hour <= 14:  # Peak hours
                if temperature > 32:
                    return 0.9  # High temperatures can indicate cloud cover
                elif temperature < 25:
                    return 0.85  # Unusually cool suggests clouds
                else:
                    return 1.0
            else:  # Non-peak hours
                if temperature > 30:
                    return 0.95
                elif temperature < 23:
                    return 0.9
                else:
                    return 1.0
                
        except Exception as e:
            print(f"Error in temperature impact calculation: {str(e)}")
            return 1.0

    # Add this method to the AutomatedPredictor class
    def _store_prediction_with_history(self, date, hour, predicted, actual, conditions):
        """Store prediction with proper history tracking"""
        try:
            # Create prediction record
            prediction_record = {
                'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
                'hour': hour,
                'predicted': float(predicted) if predicted is not None else None,
                'actual': float(actual) if actual is not None else None,
                'conditions': conditions,
                'timestamp': pd.Timestamp.now()
            }
            
            # Calculate error metrics if actual value is available
            if actual is not None and predicted is not None:
                prediction_record['error'] = float(actual - predicted)
                if actual != 0:
                    prediction_record['error_percentage'] = (prediction_record['error'] / actual) * 100
                else:
                    prediction_record['error_percentage'] = 0.0 if predicted == 0 else 100.0
            
            # Store prediction
            self.prediction_history.append(prediction_record)
            
            # Keep history size manageable
            if len(self.prediction_history) > 1000:  # Keep last 1000 predictions
                self.prediction_history = self.prediction_history[-1000:]
                
            print(f"\nStored prediction for {date}, hour {hour}:")
            print(f"Predicted: {predicted:.2f}")
            print(f"Actual: {actual if actual is not None else 'Not yet available'}")
            
            return True
            
        except Exception as e:
            print(f"Error storing prediction: {str(e)}")
            traceback.print_exc()
            return False

    # Modify the predict_next_hour method to use the new storage function
    def predict_next_hour(self, data, target_date, current_hour):
        """Enhanced prediction with validation"""
        try:
            # Input validation
            if not isinstance(current_hour, (int, float)) or not 0 <= current_hour <= 23:
                raise ValueError(f"Invalid hour: {current_hour}")
            
            next_hour = (current_hour + 1) % 24
            
            # Get conditions with validation
            conditions = self._get_current_conditions(data, target_date, current_hour)
            if not conditions:
                print("Failed to get current conditions")
                return None
            
            # Get predictions with validation
            predictions = self._get_all_predictions(data, target_date, current_hour, conditions)
            if not predictions:
                print("Failed to get predictions")
                return None
            
            # Calculate adjustments
            weather_adjustment = self._calculate_weather_adjustment(conditions, next_hour)
            transition_adjustment = self._calculate_transition_adjustments(next_hour, conditions)
            
            # Get ensemble prediction
            final_prediction = self._get_ensemble_prediction(predictions, next_hour, conditions)
            
            # Apply adjustments
            final_prediction *= weather_adjustment * transition_adjustment
            
            # Validate final prediction
            final_prediction = self._validate_prediction(final_prediction, next_hour, conditions)
            
            # Store prediction with proper history tracking
            if final_prediction is not None:
                success = self._store_prediction_with_history(
                    target_date, 
                    next_hour, 
                    final_prediction, 
                    None,  # Actual value will be updated later
                    conditions
                )
                if not success:
                    print("Failed to store prediction")
            
            return final_prediction
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return None

    # Add this method to the AutomatedPredictor class
    def _get_max_expected(self, hour, conditions):
        """Get maximum expected solar radiation with more conservative limits"""
        try:
            # Calculate clear sky radiation
            clear_sky = calculate_clear_sky_radiation(
                hour=hour,
                latitude=ADDU_LATITUDE,
                longitude=ADDU_LONGITUDE,
                date=pd.Timestamp.now().date(),
                temperature=conditions['temperature'],
                humidity=conditions['humidity'],
                pressure=conditions['pressure']
            )
            
            # Get historical maximum with validation
            hour_stats = self._get_hour_statistics(hour)
            historical_max = hour_stats['max'] if hour_stats and 'max' in hour_stats else clear_sky
            
            # More conservative limits based on time of day
            if hour <= 5 or hour >= 19:  # Night hours
                return 0.0
            elif 6 <= hour <= 8:  # Dawn
                return min(clear_sky * 0.6, historical_max * 1.1)  # More conservative dawn limits
            elif 16 <= hour <= 18:  # Dusk
                return min(clear_sky * 0.4, historical_max * 1.0)  # More conservative dusk limits
            else:  # Daytime
                # Consider weather conditions
                cloud_impact = self.weather_analyzer.analyze_conditions(conditions).get('cloud_cover', 0)
                weather_factor = 1.0 - (cloud_impact * 0.8)  # Increased cloud impact
                
                # More conservative maximum
                max_expected = min(
                    clear_sky * weather_factor,  # No additional allowance above clear sky
                    historical_max * 1.1  # Only 10% above historical maximum
                )
                
                print(f"\nMaximum expected calculation for hour {hour}:")
                print(f"Clear sky radiation: {clear_sky:.2f} W/m²")
                print(f"Historical maximum: {historical_max:.2f} W/m²")
                print(f"Weather factor: {weather_factor:.2f}")
                print(f"Final max expected: {max_expected:.2f} W/m²")
                
                return float(max_expected)
                
        except Exception as e:
            print(f"Error in _get_max_expected: {str(e)}")
            traceback.print_exc()
            return 1000.0  # More conservative fallback maximum

    def _validate_prediction(self, prediction, hour, conditions):
        """Validate prediction values with enhanced checks"""
        try:
            if prediction is None:
                return None
                
            if not np.isfinite(prediction):
                print("Non-finite prediction value")
                return None
                
            # Get reasonable bounds
            max_expected = self._get_max_expected(hour, conditions)
            
            # Basic range validation
            if prediction < 0:
                print(f"Negative prediction ({prediction:.2f}) corrected to 0")
                return 0.0
                
            if prediction > max_expected:
                print(f"Prediction ({prediction:.2f}) exceeded maximum expected ({max_expected:.2f})")
                return max_expected
                
            # Validate against current conditions
            current_value = conditions.get('solar_rad', 0)
            if current_value > 0:
                change_ratio = prediction / current_value
                if change_ratio > 3:  # More than triple
                    print(f"Excessive increase ({change_ratio:.1f}x) from current value")
                    return min(current_value * 2, prediction)  # Limit to double
                    
            return float(prediction)
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            traceback.print_exc()
            return None

class TimeSeriesEnsemble:
    def __init__(self):
        # Base models with adjusted parameters
        self.lgb = LGBMRegressor(
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=20,
            feature_fraction=0.8,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1
        )
        
        self.xgb = XGBRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbosity=0
        )
        
        self.ensemble = StackingRegressor(
            estimators=[
                ('lgb', self.lgb),
                ('xgb', self.xgb)
            ],
            final_estimator=LGBMRegressor(
                n_estimators=25,
                learning_rate=0.1,
                num_leaves=7,
                min_child_samples=20,
                verbose=-1
            ),
            n_jobs=-1
        )
        
        self.hour_specific_models = {}
        self.feature_columns = None  # Store feature columns used in training
        self.model_status = {hour: {'trained': False, 'samples': 0} for hour in range(24)}
        print("Initialized TimeSeriesEnsemble with base models")

    def prepare_features(self, data, hour):
        """Prepare features with consistent column ordering"""
        try:
            # Define standard feature columns
            feature_cols = [
                'hour_sin', 'hour_cos', 
                'temperature', 'humidity', 'pressure', 'uv',
                'clear_sky_radiation', 'current_radiation'
            ]
            
            # Create features DataFrame
            if isinstance(data, pd.DataFrame):
                features = pd.DataFrame()
                features['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
                features['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
                features['temperature'] = data['Average Temperature']
                features['humidity'] = data['Average Humidity']
                features['pressure'] = data['Average Barometer']
                features['uv'] = data['UV Index']
                features['clear_sky_radiation'] = data['clear_sky_radiation']
                features['current_radiation'] = data['Solar Rad - W/m^2']
            else:
                # Single prediction case
                features = pd.DataFrame({
                    'hour_sin': [np.sin(2 * np.pi * hour/24)],
                    'hour_cos': [np.cos(2 * np.pi * hour/24)],
                    'temperature': [data.get('temperature', 25)],
                    'humidity': [data.get('humidity', 60)],
                    'pressure': [data.get('pressure', 1013.25)],
                    'uv': [data.get('uv', 0)],
                    'clear_sky_radiation': [data.get('clear_sky_rad', 0)],
                    'current_radiation': [data.get('solar_rad', 0)]
                })
            
            # Ensure consistent column order
            features = features[feature_cols]
            
            # Store feature columns if not already set
            if self.feature_columns is None:
                self.feature_columns = feature_cols
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            traceback.print_exc()
            return None

    def fit(self, X, y, hour):
        """Fit hour-specific model with consistent features"""
        try:
            if len(X) < 24:  # Need minimum data
                print(f"Insufficient data for hour {hour:02d}")
                return
                
            # Prepare features with consistent columns
            features = self.prepare_features(X, hour)
            if features is None:
                return
                
            # Create new model instance for this hour
            self.hour_specific_models[hour] = deepcopy(self.ensemble)
            
            # Train the model
            self.hour_specific_models[hour].fit(features, y)
            self.model_status[hour]['trained'] = True
            self.model_status[hour]['samples'] = len(y)
            print(f"Trained model for hour {hour:02d} with {len(y)} samples")
            
        except Exception as e:
            print(f"Error training model for hour {hour:02d}: {str(e)}")
            traceback.print_exc()

    def predict(self, X, hour):
        """Make prediction with proper feature preparation"""
        try:
            if hour not in self.hour_specific_models:
                print(f"No model available for hour {hour:02d}")
                return None
                
            # Prepare features with consistent columns
            features = self.prepare_features(X, hour)
            if features is None:
                return None
                
            # Ensure features match training columns
            if set(features.columns) != set(self.feature_columns or []):
                print(f"Feature mismatch for hour {hour:02d}")
                return None
                
            # Make prediction
            prediction = self.hour_specific_models[hour].predict(features)
            return float(prediction[0])
            
        except Exception as e:
            print(f"Error in prediction for hour {hour:02d}: {str(e)}")
            traceback.print_exc()
            return None

    def get_training_status(self):
        """Get training status summary"""
        trained_hours = sum(1 for h in self.model_status if self.model_status[h]['trained'])
        print("\nEnsemble Training Status:")
        print(f"Total trained models: {trained_hours}/24 hours")
        print("\nHourly Status:")
        for hour in range(24):
            status = self.model_status[hour]
            if status['trained']:
                print(f"Hour {hour:02d}: Trained with {status['samples']} samples")
            else:
                print(f"Hour {hour:02d}: Not trained")

def main():
    try:
        print("Starting automated solar radiation prediction system...")
        
        # Initialize predictor with error tracking
        predictor = AutomatedPredictor()
        print("\nInitialized directory structure:")
        print(f"Base directory: {predictor.base_dir}")
        print(f"Data folder: {predictor.data_folder}")
        print(f"Stats folder: {predictor.stats_folder}")
        print(f"Reports folder: {predictor.reports_folder}")
        print(f"Models folder: {predictor.models_folder}")
        
        # Load and preprocess data
        hourly_data, minute_data, feature_averages = preprocess_data('dataset.csv')
        if hourly_data is None or minute_data is None:
            raise ValueError("Failed to load and preprocess data")
        
        print(f"\nLoaded data shape: {hourly_data.shape}")
        print(f"Date range: {hourly_data['timestamp'].min()} to {hourly_data['timestamp'].max()}")
        
        print("\nLearning from historical data...")
        predictor.learn_from_historical_data(hourly_data)
        
        # Get all unique dates
        dates = sorted(hourly_data['timestamp'].dt.date.unique())
        print(f"\nProcessing {len(dates)} unique dates")
        
        total_predictions = 0
        total_error = 0
        
        print("\nTesting predictions on historical data...")
        batch_size = 24  # Process in 24-hour batches
        current_batch = []
        
        for date_idx, date in enumerate(dates):
            print(f"\nProcessing date {date} ({date_idx + 1}/{len(dates)})")
            day_data = hourly_data[hourly_data['timestamp'].dt.date == date]
            
            for hour in range(23):  # Process up to hour 23
                current_data = day_data[day_data['timestamp'].dt.hour == hour]
                next_data = day_data[day_data['timestamp'].dt.hour == hour + 1]
                
                if current_data.empty or next_data.empty:
                    print(f"Skipping hour {hour} - insufficient data")
                    continue
                
                # Make prediction for next hour
                prediction = predictor.predict_next_hour(hourly_data, date, hour)
                
                # Get actual value for next hour
                if not next_data.empty:
                    actual = float(next_data['Solar Rad - W/m^2'].iloc[0])
                    next_hour = hour + 1
                    
                    print(f"\nPrediction for hour {next_hour}:")
                    print(f"Predicted: {prediction:.2f}")
                    print(f"Actual: {actual:.2f}")
                    
                    # Update prediction with actual value
                    if prediction is not None:
                        predictor.update_with_actual(date, next_hour, actual)
                        
                        # Calculate error
                        error = abs(actual - prediction)
                        total_error += error
                        total_predictions += 1
                        
                        # Calculate error percentage with proper handling of zero values
                        if actual == 0:
                            error_percentage = 0.0 if prediction == 0 else 100.0
                        else:
                            error_percentage = (error / actual) * 100
                        
                        # Add to current batch with proper timestamps
                        current_batch.append({
                            'date': date,
                            'hour': next_hour,
                            'timestamp': next_data['timestamp'].iloc[0],
                            'predicted': prediction,
                            'actual': actual,
                            'error': error,
                            'error_percentage': error_percentage
                        })
                        
                        print(f"Error: {error:.2f}")
                        print(f"Error %: {error_percentage:.2f}%")
                else:
                    print(f"No actual data available for next hour {hour + 1}")
                
                # Save state and evaluate periodically
                if total_predictions % 24 == 0:
                    predictor.save_state()
                    
                    if len(current_batch) >= batch_size:
                        # Save batch with timestamp information
                        batch_df = pd.DataFrame(current_batch)
                        batch_df['timestamp'] = pd.to_datetime(batch_df['timestamp'])
                        batch_file = os.path.join(
                            predictor.data_folder,
                            f'prediction_batch_{date}_{hour:02d}.csv'
                        )
                        batch_df.to_csv(batch_file, index=False)
                        current_batch = []
                    
                    predictor.evaluate_and_save_model()
            
            # Generate reports periodically
            if (date_idx + 1) % 7 == 0:
                predictor.generate_detailed_learning_report('week')
        
        # Final evaluation and reporting
        if total_predictions > 0:
            avg_error = total_error / total_predictions
            print(f"\nFinal Performance Metrics:")
            print(f"Total predictions: {total_predictions}")
            print(f"Average error: {avg_error:.2f} W/m²")
            
            predictor.analyze_learning_performance()
            for period in ['day', 'week', 'month', 'all']:
                predictor.generate_detailed_learning_report(period)
                
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
