import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, SelectFromModel, RFE, RFECV, VarianceThreshold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
from datetime import datetime
import joblib
import os
import logging
import json
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import pickle
import copy
plt.style.use('default')

class GHIPredictionSystem:
    def __init__(self, data_path, target_error=0.05):
        self.data_path = data_path
        self.target_error = target_error
        self.best_model = None
        self.best_error = float('inf')
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.setup_logging()
        
    def setup_logging(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        logging.basicConfig(
            filename=f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def engineer_features(self, df):
        """Engineer features for the input dataframe"""
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Set the index if not already set
        if 'period_end' in df.columns:
            df['period_end'] = pd.to_datetime(df['period_end'])
            df.set_index('period_end', inplace=True)
        
        # Extract time components
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        
        # Davao-specific seasons
        df['is_dry_period'] = df.index.month.isin([1, 2, 3, 4]).astype(int)
        df['is_transitional'] = df.index.month.isin([5, 6]).astype(int)
        df['is_wet_period'] = df.index.month.isin([7, 8, 9, 10, 11, 12]).astype(int)
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Monsoon and seasonal features
        df['monsoon_influence'] = df.index.month.isin([6, 7, 8, 9, 10, 11]).astype(int)
        df['cyclone_season'] = df.index.month.isin([10, 11, 12]).astype(int)
        
        # Diurnal patterns
        df['morning_convection'] = df.index.hour.isin([9, 10, 11]).astype(int)
        df['afternoon_convection'] = df.index.hour.isin([14, 15, 16, 17]).astype(int)
        
        # Weather interaction features (if columns exist)
        if 'cloud_opacity' in df.columns and 'clearsky_ghi' in df.columns:
            df['cloud_clearsky_interaction'] = df['cloud_opacity'] * df['clearsky_ghi']
        
        if 'air_temp' in df.columns and 'relative_humidity' in df.columns:
            df['temp_humidity_interaction'] = df['air_temp'] * df['relative_humidity']
            
            # Convection potential
            df['convection_potential'] = (
                (df['air_temp'] > df['air_temp'].rolling(24).mean()) & 
                (df['relative_humidity'] > 75)
            ).astype(int)
        
        # Mountain effect (if wind direction exists)
        if 'wind_direction_10m' in df.columns:
            df['mountain_effect'] = (
                (df.index.hour.isin([14, 15, 16, 17])) & 
                (df['wind_direction_10m'].between(45, 135))
            ).astype(int)
        
        # Calculate clear sky index if possible
        if 'ghi' in df.columns and 'clearsky_ghi' in df.columns:
            df['clear_sky_index'] = df['ghi'] / df['clearsky_ghi'].where(df['clearsky_ghi'] > 0, 1)
        
        # Rolling statistics if GHI exists
        if 'ghi' in df.columns:
            df['ghi_rolling_mean'] = df['ghi'].rolling(window=3, min_periods=1).mean()
            df['ghi_rolling_std'] = df['ghi'].rolling(window=3, min_periods=1).std()
        
        return df

    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        try:
            # Load the data
            df = pd.read_csv(self.data_path)
            
            # Convert period_end to datetime and set as index
            df['period_end'] = pd.to_datetime(df['period_end'])
            df.set_index('period_end', inplace=True)
            
            # Create target variable (next hour's GHI)
            df['next_ghi'] = df['ghi'].shift(-1)
            
            # Drop the last row since it won't have a next_ghi value
            df = df.dropna(subset=['next_ghi'])
            
            # Engineer features
            df = self.engineer_features(df)
            
            # Get feature columns
            feature_columns = self.get_initial_features()
            
            # Only keep features that exist in the dataframe
            available_features = [f for f in feature_columns if f in df.columns]
            
            logging.info(f"Loaded data shape: {df.shape}")
            logging.info(f"Available features: {len(available_features)}")
            logging.info(f"Target variable: next_ghi")
            
            return df, available_features
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def get_initial_features(self):
        """Get the initial set of features for the model"""
        return [
            'ghi', 'clearsky_ghi', 'cloud_opacity', 'air_temp',
            'relative_humidity', 'wind_speed_10m', 'wind_direction_10m',
            'surface_pressure', 'hour', 'month', 'day_of_year',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'is_dry_period', 'is_transitional', 'is_wet_period',
            'monsoon_influence', 'cyclone_season',
            'morning_convection', 'afternoon_convection',
            'cloud_clearsky_interaction', 'temp_humidity_interaction',
            'convection_potential', 'mountain_effect',
            'clear_sky_index', 'ghi_rolling_mean', 'ghi_rolling_std'
        ]

    def perform_comprehensive_feature_selection(self, X, y):
        """Comprehensive feature selection using multiple methods"""
        # Check if saved feature selection results exist
        feature_selection_path = 'models/feature_selection_results.json'
        if os.path.exists(feature_selection_path):
            logging.info("Loading saved feature selection results...")
            with open(feature_selection_path, 'r') as f:
                saved_results = json.load(f)
                self.selected_features = saved_results['selected_features']
                self.feature_importance = saved_results['feature_importance']
                logging.info(f"Loaded {len(self.selected_features)} pre-selected features")
                return self.selected_features
        
        logging.info("Performing comprehensive feature selection...")
        
        # Scale the features for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        selected_features_dict = {}
        
        # 1. Variance Threshold (fast)
        logging.info("Running Variance Threshold selection...")
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X_scaled)
        variance_features = X.columns[selector.get_support()].tolist()
        selected_features_dict['variance'] = variance_features
        
        # 2. Mutual Information (moderate speed)
        logging.info("Running Mutual Information selection...")
        mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(15, X.shape[1]))
        mi_selector.fit(X_scaled, y)
        mi_scores = mi_selector.scores_
        mi_features = X.columns[mi_selector.get_support()].tolist()
        selected_features_dict['mutual_info'] = mi_features
        
        # 3. F-regression (fast)
        logging.info("Running F-regression selection...")
        f_selector = SelectKBest(score_func=f_regression, k=min(15, X.shape[1]))
        f_selector.fit(X_scaled, y)
        f_scores = f_selector.scores_
        f_features = X.columns[f_selector.get_support()].tolist()
        selected_features_dict['f_regression'] = f_features
        
        # 4. Lasso with adjusted parameters (moderate speed)
        logging.info("Running Lasso selection...")
        lasso = LassoCV(
            cv=3,  # Reduced from 5
            random_state=42,
            max_iter=1000,
            tol=1e-2,
            n_jobs=-1,
            selection='random'
        )
        lasso.fit(X_scaled, y)
        lasso_selector = SelectFromModel(lasso, prefit=True, max_features=15)
        lasso_features = X.columns[lasso_selector.get_support()].tolist()
        selected_features_dict['lasso'] = lasso_features
        
        # 5. Random Forest Importance (faster configuration)
        logging.info("Running Random Forest selection...")
        rf = RandomForestRegressor(
            n_estimators=50,  # Reduced from 100
            random_state=42,
            n_jobs=-1,
            max_depth=8,
            max_features='sqrt'
        )
        rf.fit(X_scaled, y)
        rf_selector = SelectFromModel(rf, prefit=True, max_features=15)
        rf_features = X.columns[rf_selector.get_support()].tolist()
        selected_features_dict['random_forest'] = rf_features
        
        # Skip RFECV as it's too slow and replace with a simpler approach
        logging.info("Running simplified feature ranking...")
        # Combine scores from different methods
        feature_scores = {}
        for feature in X.columns:
            score = (
                mi_scores[X.columns.get_loc(feature)] +
                f_scores[X.columns.get_loc(feature)] +
                abs(lasso.coef_[X.columns.get_loc(feature)]) +
                rf.feature_importances_[X.columns.get_loc(feature)]
            )
            feature_scores[feature] = score
        
        # Select top features based on combined scores
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        selected_features_dict['combined_ranking'] = [f[0] for f in top_features]
        
        # Combine results and select features that appear in multiple methods
        all_selected_features = []
        for features in selected_features_dict.values():
            all_selected_features.extend(features)
        
        # Count occurrences of each feature
        feature_counts = {}
        for feature in all_selected_features:
            feature_counts[feature] = all_selected_features.count(feature)
        
        # Select features that appear in at least 2 methods (reduced from 3)
        final_features = [feature for feature, count in feature_counts.items() 
                         if count >= 2]
        
        # Ensure we have at least 8 features
        if len(final_features) < 8:
            remaining_features = [f for f, _ in top_features if f not in final_features]
            final_features.extend(remaining_features[:8 - len(final_features)])
        
        # Log feature selection results
        logging.info("Feature selection results:")
        for method, features in selected_features_dict.items():
            logging.info(f"{method}: {len(features)} features")
        logging.info(f"Final selected features ({len(final_features)}): {final_features}")
        
        # Store feature importance scores
        self.feature_importance = {feature: feature_scores[feature] for feature in final_features}
        
        # Save feature selection results
        results_to_save = {
            'selected_features': final_features,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method_results': {
                method: features 
                for method, features in selected_features_dict.items()
            }
        }
        
        with open(feature_selection_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        logging.info(f"Saved feature selection results to {feature_selection_path}")
        
        return final_features

    def create_optimized_ensemble(self, iteration, n_jobs):
        """Create an optimized ensemble model"""
        # Optimize base parameters
        base_estimators = 100
        base_learning_rate = 0.1
        
        # Efficient parameter adjustment
        n_estimators = base_estimators + (25 * iteration)  # Reduced from 50 to 25
        learning_rate = base_learning_rate / (1 + 0.05 * iteration)  # Slower decay
        max_depth = min(6 + iteration // 3, 12)  # More controlled depth increase
        
        xgb_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': 1 + iteration // 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',  # Faster histogram-based algorithm
            'n_jobs': n_jobs,
            'random_state': 42 + iteration
        }
        
        lgb_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': 31 + iteration,
            'n_jobs': n_jobs,
            'random_state': 42 + iteration,
            'force_row_wise': True  # More efficient computation
        }
        
        cat_params = {
            'iterations': n_estimators,
            'learning_rate': learning_rate,
            'depth': max_depth,
            'thread_count': n_jobs,
            'random_state': 42 + iteration,
            'verbose': False,
            'bootstrap_type': 'Bernoulli'  # More efficient bootstrapping
        }
        
        estimators = [
            ('xgb', xgb.XGBRegressor(**xgb_params)),
            ('lgb', LGBMRegressor(**lgb_params)),
            ('cat', CatBoostRegressor(**cat_params))
        ]
        
        return VotingRegressor(estimators=estimators)

    def train_and_optimize(self):
        """Enhanced training loop with adaptive learning"""
        try:
            # Load and preprocess data
            logging.info("Loading and preprocessing data...")
            df, feature_columns = self.load_and_preprocess_data()
            
            # Store data and features
            self.data = df
            self.feature_columns = feature_columns
            self.target_column = 'next_ghi'
            
            # Prepare features and target
            X = self.data[self.feature_columns]
            y = self.data[self.target_column]
            
            # Store original index
            self.original_index = X.index
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns
            )
            
            # Split data - 70-30 split
            train_idx, val_idx = train_test_split(
                np.arange(len(X_scaled)), 
                test_size=0.3,
                random_state=42
            )
            
            # Split using indices
            X_train = X_scaled.iloc[train_idx]
            X_val = X_scaled.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            
            # Store validation indices for later use
            self.val_idx = val_idx
            
            # Initialize variables for training
            iteration = 0
            max_iterations = 20
            best_error = float('inf')
            best_models = None
            
            # Training loop
            while iteration < max_iterations:
                logging.info(f"\nIteration {iteration + 1}/{max_iterations}")
                
                # Train models
                models = self.train_weighted_models(X_train, y_train, np.ones(len(y_train)), iteration)
                
                # Create ensemble
                ensemble = self.create_weighted_ensemble(models, X_val, y_val)
                
                # Calculate metrics
                metrics = self.calculate_detailed_metrics(ensemble, X_val, y_val)
                
                # Check if this is the best performance
                current_error = metrics['Overall Performance']['Overall']['Mean Error']
                if current_error < best_error:
                    best_error = current_error
                    best_models = copy.deepcopy(models)
                    self.best_model = ensemble  # Store the best ensemble model
                    self.selected_features = self.feature_columns  # Store selected features
                    
                    # Save both the archive and current model
                    self.save_best_models(models, ensemble, metrics, iteration)
                    success = self.save_trained_model()
                    if not success:
                        logging.error("Failed to save current model")
                    
                    logging.info(f"New best performance! Mean Error: {current_error:.2f}")
                
                # Update learning strategy and check convergence
                if self.check_convergence_criteria(metrics):
                    logging.info("Convergence criteria met. Stopping training.")
                    break
                
                iteration += 1
            
            # Verify final model was saved
            if not os.path.exists('models/current_model_ensemble.pkl'):
                logging.error("Final model was not saved correctly")
                
            return best_models
            
        except Exception as e:
            logging.error(f"Error in training: {str(e)}")
            raise

    def save_best_models(self, models, ensemble, metrics, iteration):
        """Save the best performing models and metrics"""
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('models', f'best_models_{timestamp}_iter_{iteration}')
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save individual models
            for model_type, model in models.items():
                with open(os.path.join(save_dir, f'{model_type}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            
            # Save ensemble model
            with open(os.path.join(save_dir, 'ensemble.pkl'), 'wb') as f:
                pickle.dump(ensemble, f)
            
            # Save scaler
            with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save metrics and configuration
            config = {
                'timestamp': timestamp,
                'iteration': iteration,
                'final_error': metrics['Overall Performance']['Overall']['Mean Error'],
                'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else {},
                'selected_features': self.selected_features if hasattr(self, 'selected_features') else []
            }
            
            with open(os.path.join(save_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)
            
            logging.info(f"Successfully saved best models and metrics to {save_dir}")
            
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            raise

    def calculate_temporal_weights(self, df):
        """Calculate sample weights based on temporal importance"""
        current_year = df.index.max().year
        years_diff = (df.index.year - df.index.min().year + 1)
        
        # Base temporal weights
        temporal_weights = 1 + np.log1p(years_diff)
        
        # Seasonal pattern weights
        season_weights = np.sin(2 * np.pi * df.index.dayofyear / 365) + 2
        
        # Weather pattern weights
        clear_sky_ratio = df['ghi'] / df['clearsky_ghi'].where(df['clearsky_ghi'] > 0, 1)
        weather_weights = 1 + abs(1 - clear_sky_ratio)
        
        # Combine weights
        combined_weights = temporal_weights * season_weights * weather_weights
        
        # Normalize weights
        return combined_weights / combined_weights.mean()

    def analyze_prediction_errors(self, ensemble, X_val, y_val, 
                                seasonal_errors, weather_errors, time_errors):
        """Analyze prediction errors in detail"""
        y_pred = ensemble.predict(X_val)
        errors = np.abs(y_val - y_pred)
        
        # Seasonal analysis
        seasons = {
            'dry_period': X_val.index.month.isin([1, 2, 3, 4]),
            'transitional': X_val.index.month.isin([5, 6]),
            'wet_period': X_val.index.month.isin([7, 8, 9, 10, 11, 12])
        }
        
        for season, mask in seasons.items():
            seasonal_errors[season].extend(errors[mask])
        
        # Weather condition analysis
        weather_conditions = {
            'clear_sky': X_val['cloud_opacity'] < 0.2,
            'partly_cloudy': (X_val['cloud_opacity'] >= 0.2) & (X_val['cloud_opacity'] < 0.8),
            'overcast': X_val['cloud_opacity'] >= 0.8
        }
        
        for condition, mask in weather_conditions.items():
            weather_errors[condition].extend(errors[mask])
        
        # Time of day analysis
        time_periods = {
            'morning': (X_val.index.hour >= 6) & (X_val.index.hour < 12),
            'afternoon': (X_val.index.hour >= 12) & (X_val.index.hour < 18),
            'evening': (X_val.index.hour >= 18) | (X_val.index.hour < 6)
        }
        
        for period, mask in time_periods.items():
            time_errors[period].extend(errors[mask])

    def update_learning_strategy(self, error_analysis):
        """Update learning strategy based on error patterns"""
        logging.info("Updating learning strategy...")
        
        # Check if we need to increase model complexity
        if self.check_model_complexity_needs():
            self.increase_model_complexity()
        
        # Adjust for seasonal patterns
        if error_analysis.get('seasonal_bias', False):
            self.adjust_seasonal_weights()
        
        # Adjust for weather patterns
        if error_analysis.get('weather_bias', False):
            self.adjust_weather_weights()
        
        # Adjust for time of day patterns
        if error_analysis.get('time_bias', False):
            self.adjust_time_weights()
        
        # Update learning rate based on performance
        if hasattr(self, 'performance_history') and len(self.performance_history) > 0:
            recent_performance = self.performance_history[-1].get('validation_error', float('inf'))
            if recent_performance > self.target_error_threshold:
                # Decrease learning rate if error is high
                for model_name in self.model_params:
                    current_lr = self.model_params[model_name].get('learning_rate', 0.1)
                    self.model_params[model_name]['learning_rate'] = current_lr * 0.9
        
        logging.info("Learning strategy updated")

    def create_xgboost_model(self, iteration):
        return xgb.XGBRegressor(
            n_estimators=100 + (25 * iteration),
            learning_rate=0.1 / (1 + 0.05 * iteration),
            max_depth=min(6 + iteration // 3, 12),
            random_state=42 + iteration,
            n_jobs=-1
        )

    def create_lightgbm_model(self, iteration):
        return LGBMRegressor(
            n_estimators=100 + (25 * iteration),
            learning_rate=0.1 / (1 + 0.05 * iteration),
            num_leaves=31 + iteration,
            random_state=42 + iteration,
            n_jobs=-1
        )

    def create_catboost_model(self, iteration):
        return CatBoostRegressor(
            iterations=100 + (25 * iteration),
            learning_rate=0.1 / (1 + 0.05 * iteration),
            depth=min(6 + iteration // 3, 12),
            random_state=42 + iteration,
            verbose=False,
            thread_count=-1
        )

    def adjust_learning_strategy(self, iteration):
        """Dynamic learning strategy adjustment"""
        if iteration % 5 == 0:
            # Adjust model parameters based on iteration
            self.model_params = {
                'n_estimators': 100 + (50 * iteration),
                'learning_rate': max(0.01, 0.1 / (1 + 0.1 * iteration)),
                'max_depth': min(6 + iteration // 3, 15),
                'subsample': max(0.6, 0.9 - 0.02 * iteration),
                'colsample_bytree': max(0.6, 0.9 - 0.02 * iteration)
            }
            
            # Adjust training parameters
            self.batch_size = min(1000, 100 * (iteration + 1))

    def increase_model_complexity(self):
        """Efficiently increase model complexity"""
        if not hasattr(self, 'complexity_level'):
            self.complexity_level = 0
        self.complexity_level += 1
        
        # Adjust parameters based on complexity level
        self.current_params = {
            'n_estimators': 100 + (25 * self.complexity_level),
            'max_depth': min(6 + self.complexity_level, 12)
        }

    def optimize_features(self):
        """Optimize feature selection"""
        if hasattr(self, 'feature_importance'):
            # Use feature importance to select top features
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:8]
            self.selected_features = [f[0] for f in top_features]

    def reset_learning_strategy(self):
        """Reset learning strategy when maximum iterations reached"""
        logging.info("Resetting learning strategy...")
        
        # Reset scalers and feature selectors
        self.scaler = StandardScaler()
        
        # Optionally force new feature selection
        feature_selection_path = 'models/feature_selection_results.json'
        if os.path.exists(feature_selection_path):
            os.rename(feature_selection_path, 
                     f'models/feature_selection_results_old_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            logging.info("Archived old feature selection results")
        
        # Perform new feature selection
        self.perform_comprehensive_feature_selection(self.X, self.y)

    def custom_mape(self, y_true, y_pred):
        """Calculate MAPE while handling zero values"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero = y_true != 0
        return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

    def save_model(self, iteration, mape, rmse):
        # Save model
        model_path = f'models/ghi_model_iteration_{iteration}.joblib'
        joblib.dump(self.best_model, model_path)
        
        # Save model metadata
        metadata = {
            'iteration': iteration,
            'mape': float(mape),
            'rmse': float(rmse),
            'selected_features': self.selected_features,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f'models/model_metadata_{iteration}.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Model saved: {model_path}")

    def predict(self, input_data):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Prepare input data
        input_scaled = self.scaler.transform(input_data[self.selected_features])
        
        # Make prediction
        prediction = self.best_model.predict(input_scaled)
        return prediction

    def save_final_report(self, df, y_pred, training_history):
        """Save comprehensive final report and predictions"""
        logging.info("Generating final report and saving predictions...")
        
        # Create reports directory if it doesn't exist
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save predictions to CSV
        predictions_df = df.copy()
        predictions_df['predicted_next_ghi'] = y_pred
        predictions_df['prediction_error'] = abs(predictions_df['next_ghi'] - predictions_df['predicted_next_ghi'])
        predictions_df['prediction_error_percent'] = (predictions_df['prediction_error'] / predictions_df['next_ghi']) * 100
        
        predictions_df.to_csv(f'reports/predictions_{timestamp}.csv', index=False)
        
        # 2. Generate performance metrics
        performance_metrics = {
            'overall_mape': float(self.best_error),
            'rmse': float(np.sqrt(mean_squared_error(predictions_df['next_ghi'], predictions_df['predicted_next_ghi']))),
            'mean_error': float(predictions_df['prediction_error'].mean()),
            'max_error': float(predictions_df['prediction_error'].max()),
            'min_error': float(predictions_df['prediction_error'].min()),
            'std_error': float(predictions_df['prediction_error'].std()),
            'predictions_within_5_percent': float((predictions_df['prediction_error_percent'] <= 5).mean() * 100),
            'total_training_iterations': len(training_history),
            'training_duration_minutes': (datetime.now() - self.training_start_time).total_seconds() / 60
        }
        
        # 3. Generate hourly performance analysis
        hourly_performance = predictions_df.groupby('hour').agg({
            'prediction_error_percent': ['mean', 'std', 'count']
        }).round(2)
        
        # 4. Generate monthly performance analysis
        monthly_performance = predictions_df.groupby('month').agg({
            'prediction_error_percent': ['mean', 'std', 'count']
        }).round(2)
        
        # 5. Create comprehensive report
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': timestamp,
            'performance_metrics': performance_metrics,
            'feature_importance': self.feature_importance,
            'selected_features': self.selected_features,
            'training_history': training_history,
            'hourly_performance': hourly_performance.to_dict(),
            'monthly_performance': monthly_performance.to_dict(),
            'model_parameters': self.best_model.get_params(),
        }
        
        # Save report as JSON
        with open(f'reports/final_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # 6. Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>GHI Prediction System Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>GHI Prediction System Report</h1>
            <h2>Performance Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Overall MAPE</td><td>{performance_metrics['overall_mape']:.2f}%</td></tr>
                <tr><td>RMSE</td><td>{performance_metrics['rmse']:.2f}</td></tr>
                <tr><td>Predictions within 5%</td><td>{performance_metrics['predictions_within_5_percent']:.2f}%</td></tr>
                <tr><td>Training Duration</td><td>{performance_metrics['training_duration_minutes']:.2f} minutes</td></tr>
            </table>
            
            <h2>Feature Importance</h2>
            <table>
                <tr><th>Feature</th><th>Importance Score</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))}
            </table>
            
            <h2>Training History</h2>
            <table>
                <tr><th>Iteration</th><th>MAPE</th><th>RMSE</th></tr>
                {''.join(f"<tr><td>{i+1}</td><td>{hist['mape']:.2f}%</td><td>{hist['rmse']:.2f}</td></tr>" for i, hist in enumerate(training_history))}
            </table>
        </body>
        </html>
        """
        
        with open(f'reports/final_report_{timestamp}.html', 'w') as f:
            f.write(html_report)
        
        logging.info(f"Final report saved: reports/final_report_{timestamp}.html")
        logging.info(f"Predictions saved: reports/predictions_{timestamp}.csv")
        
        # Print summary to console
        print("\n=== Final Model Performance ===")
        print(f"Overall MAPE: {performance_metrics['overall_mape']:.2f}%")
        print(f"RMSE: {performance_metrics['rmse']:.2f}")
        print(f"Predictions within 5%: {performance_metrics['predictions_within_5_percent']:.2f}%")
        print(f"Training Duration: {performance_metrics['training_duration_minutes']:.2f} minutes")
        print("\nReports have been saved in the 'reports' directory.")
        
        # Generate and save performance figures
        self.save_performance_figures(df, y_pred, training_history)
        
        # Add figures to HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_report += f"""
            <h2>Performance Visualizations</h2>
            <div>
                <h3>Training History</h3>
                <img src="figures/training_history_{timestamp}.png" style="max-width:100%">
                
                <h3>Actual vs Predicted Values</h3>
                <img src="figures/actual_vs_predicted_{timestamp}.png" style="max-width:100%">
                
                <h3>Error Distribution</h3>
                <img src="figures/error_distribution_{timestamp}.png" style="max-width:100%">
                
                <h3>Feature Importance</h3>
                <img src="figures/feature_importance_{timestamp}.png" style="max-width:100%">
                
                <h3>Hourly Performance</h3>
                <img src="figures/hourly_performance_{timestamp}.png" style="max-width:100%">
                
                <h3>Monthly Performance</h3>
                <img src="figures/monthly_performance_{timestamp}.png" style="max-width:100%">
                
                <h3>Error Heatmap</h3>
                <img src="figures/error_heatmap_{timestamp}.png" style="max-width:100%">
            </div>
        """
        
        # ... (rest of existing report generation code) ...

    def save_performance_figures(self, df, y_pred, training_history):
        """Generate and save performance visualization figures"""
        logging.info("Generating performance figures...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figures_dir = 'reports/figures'
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        # Set style for all plots
        plt.style.use('default')
        
        # 1. Training History Plot
        plt.figure(figsize=(12, 6))
        epochs = [h['iteration'] for h in training_history]
        mapes = [h['mape'] for h in training_history]
        rmses = [h['rmse'] for h in training_history]
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, mapes, 'b-', label='MAPE')
        plt.title('Training MAPE History')
        plt.xlabel('Iteration')
        plt.ylabel('MAPE (%)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, rmses, 'r-', label='RMSE')
        plt.title('Training RMSE History')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/training_history_{timestamp}.png')
        plt.close()
        
        # 2. Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['next_ghi'], y_pred, alpha=0.5)
        plt.plot([df['next_ghi'].min(), df['next_ghi'].max()], 
                 [df['next_ghi'].min(), df['next_ghi'].max()], 
                 'r--', label='Perfect Prediction')
        plt.xlabel('Actual GHI')
        plt.ylabel('Predicted GHI')
        plt.title('Actual vs Predicted GHI Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{figures_dir}/actual_vs_predicted_{timestamp}.png')
        plt.close()
        
        # 3. Error Distribution Plot
        plt.figure(figsize=(10, 6))
        error_percent = (y_pred - df['next_ghi']) / df['next_ghi'] * 100
        plt.hist(error_percent, bins=50, alpha=0.75)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Percentage Error')
        plt.ylabel('Count')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig(f'{figures_dir}/error_distribution_{timestamp}.png')
        plt.close()
        
        # 4. Feature Importance Plot
        plt.figure(figsize=(12, 6))
        feature_imp = pd.Series(self.feature_importance).sort_values(ascending=True)
        feature_imp.plot(kind='barh')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/feature_importance_{timestamp}.png')
        plt.close()
        
        # 5. Hourly Performance Plot
        hourly_errors = df.groupby('hour')['prediction_error_percent'].mean()
        plt.figure(figsize=(12, 6))
        hourly_errors.plot(kind='bar')
        plt.title('Average Prediction Error by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Error (%)')
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/hourly_performance_{timestamp}.png')
        plt.close()
        
        # 6. Monthly Performance Plot
        monthly_errors = df.groupby('month')['prediction_error_percent'].mean()
        plt.figure(figsize=(12, 6))
        monthly_errors.plot(kind='bar')
        plt.title('Average Prediction Error by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Error (%)')
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/monthly_performance_{timestamp}.png')
        plt.close()
        
        # 7. Error Heatmap by Hour and Month
        pivot_table = df.pivot_table(
            values='prediction_error_percent',
            index='hour',
            columns='month',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        plt.imshow(pivot_table, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Error %')
        plt.title('Prediction Error (%) by Hour and Month')
        plt.xlabel('Month')
        plt.ylabel('Hour')
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                plt.text(j, i, f'{pivot_table.iloc[i, j]:.1f}', 
                        ha='center', va='center')
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/error_heatmap_{timestamp}.png')
        plt.close()
        
        logging.info(f"Performance figures saved in {figures_dir}")

    def get_location_specific_features(self):
        """Add Davao City-specific features"""
        return {
            'latitude': 7.0711,  # Davao City latitude (near equator)
            'longitude': 125.6134,  # Davao City longitude
            'elevation': 22.0,  # meters above sea level for Davao City
            'timezone': 'Asia/Manila',  # UTC+8
            'climate_type': 'tropical_rainforest',  # Köppen climate classification: Af
            'annual_rainfall': 1830,  # mm/year (average)
            'avg_temperature': 27.5,  # °C (annual average)
            'avg_humidity': 80  # % (annual average)
        }

    def create_advanced_features(self, df):
        """Enhanced feature engineering for Davao's tropical climate"""
        # Davao-specific seasons (more nuanced than simple wet/dry)
        # Relatively dry: January to April
        # Transitional: May to June
        # Wet: July to December
        df['is_dry_period'] = df.index.month.isin([1, 2, 3, 4]).astype(int)
        df['is_transitional'] = df.index.month.isin([5, 6]).astype(int)
        df['is_wet_period'] = df.index.month.isin([7, 8, 9, 10, 11, 12]).astype(int)
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Monsoon effects (Davao is less affected by monsoons due to location)
        df['monsoon_influence'] = df.index.month.isin([6, 7, 8, 9, 10, 11]).astype(int)
        
        # Tropical cyclone exposure (Davao is relatively sheltered)
        df['cyclone_season'] = df.index.month.isin([10, 11, 12]).astype(int)
        
        # Diurnal patterns (typical for equatorial region)
        df['morning_convection'] = df.index.hour.isin([9, 10, 11]).astype(int)
        df['afternoon_convection'] = df.index.hour.isin([14, 15, 16, 17]).astype(int)
        
        # Solar position calculations (very important near equator)
        df['solar_elevation'] = self.calculate_solar_elevation(
            df.index, 
            latitude=7.0711, 
            longitude=125.6134
        )
        df['day_length'] = self.calculate_day_length(df.index, latitude=7.0711)
        
        # Enhanced humidity features (crucial in Davao's climate)
        df['absolute_humidity'] = self.calculate_absolute_humidity(
            df['air_temp'], 
            df['relative_humidity']
        )
        
        # Weather interaction features
        df['cloud_clearsky_interaction'] = df['cloud_opacity'] * df['clearsky_ghi']
        df['temp_humidity_interaction'] = df['air_temp'] * df['relative_humidity']
        
        # Local convection indicators
        df['convection_potential'] = (
            (df['air_temp'] > df['air_temp'].rolling(24).mean()) & 
            (df['relative_humidity'] > 75)
        ).astype(int)
        
        # Terrain influence (Davao's geography)
        df['mountain_effect'] = (
            (df.index.hour.isin([14, 15, 16, 17])) & 
            (df['wind_direction_10m'].between(45, 135))
        ).astype(int)
        
    def calculate_solar_elevation(self, dates, latitude, longitude):
        """
        Calculate solar elevation angle for Davao's specific location
        More accurate than generic calculations
        """
        # Time equation parameters
        day_of_year = dates.dayofyear
        
        # Solar declination for latitude near equator
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
        
        # Calculate day length in hours
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)
        
        cos_hour_angle = -np.tan(lat_rad) * np.tan(decl_rad)
        cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
        hour_angle = np.arccos(cos_hour_angle)
        
        day_length = 2 * np.degrees(hour_angle) / 15  # Convert to hours
        return day_length

    def calculate_absolute_humidity(self, temperature, relative_humidity):
        """
        Calculate absolute humidity for tropical conditions
        More relevant for Davao's climate
        """
        # Constants for tropical conditions
        A = 17.27
        B = 237.7  # °C
        
        # Calculate saturation vapor pressure
        temp_celsius = temperature
        alpha = ((A * temp_celsius) / (B + temp_celsius)) + np.log(relative_humidity/100.0)
        
        # Calculate absolute humidity (g/m³)
        abs_humidity = 216.7 * (relative_humidity/100.0 * 6.112 * 
                           np.exp(A * temp_celsius/(B + temp_celsius)) / 
                           (273.15 + temp_celsius))
        
        return abs_humidity

    def calculate_day_length(self, dates, latitude):
        """Calculate day length for given dates and latitude"""
        # Convert dates to day of year
        day_of_year = dates.dayofyear
        
        # Calculate solar declination
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
        
        # Calculate day length in hours
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)
        
        cos_hour_angle = -np.tan(lat_rad) * np.tan(decl_rad)
        cos_hour_angle = np.clip(cos_hour_angle, -1, 1)  # Ensure valid range
        hour_angle = np.arccos(cos_hour_angle)
        
        # Convert to hours (2 * hour_angle / 15 degrees per hour)
        day_length = 2 * np.degrees(hour_angle) / 15
        
        return day_length

    def calculate_solar_elevation(self, dates, latitude, longitude):
        """Calculate solar elevation angle for Davao's specific location"""
        # Time equation parameters
        day_of_year = dates.dayofyear
        
        # Solar declination for latitude near equator
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
        
        # Hour angle calculation for Davao
        local_solar_time = dates.hour + dates.minute/60
        hour_angle = (local_solar_time - 12) * 15
        
        # Convert to radians
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        
        # Calculate solar elevation
        sin_elevation = (np.sin(lat_rad) * np.sin(decl_rad) + 
                        np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
        elevation = np.degrees(np.arcsin(sin_elevation))
        
        return elevation

    def calculate_absolute_humidity(self, temperature, relative_humidity):
        """Calculate absolute humidity for tropical conditions"""
        # Constants for tropical conditions
        A = 17.27
        B = 237.7  # °C
        
        # Calculate saturation vapor pressure
        temp_celsius = temperature
        alpha = ((A * temp_celsius) / (B + temp_celsius)) + np.log(relative_humidity/100.0)
        
        # Calculate absolute humidity (g/m³)
        abs_humidity = 216.7 * (relative_humidity/100.0 * 6.112 * 
                           np.exp(A * temp_celsius/(B + temp_celsius)) / 
                           (273.15 + temp_celsius))
        
        return abs_humidity

    def adjust_learning_strategy(self, iteration):
        """Adjust learning strategy for Davao's specific climate patterns"""
        current_month = datetime.now().month
        current_hour = datetime.now().hour
        
        # Base adjustments
        if current_month in [1, 2, 3, 4]:  # Relatively dry period
            self.model_params.update({
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_weight': 3
            })
        elif current_month in [5, 6]:  # Transitional period
            self.model_params.update({
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 4
            })
        else:  # Wet period
            self.model_params.update({
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 5
            })
        
        # Diurnal adjustments
        if current_hour in [14, 15, 16, 17]:  # Afternoon convection period
            self.model_params.update({
                'max_depth': min(6, self.model_params['max_depth']),
                'learning_rate': self.model_params['learning_rate'] * 0.8
            })

    def pretrain_on_historical_data(self, X_historical, y_historical, w_historical):
        """Pre-train models on historical data"""
        logging.info("Pre-training on historical data...")
        
        # Scale features
        X_historical_scaled = self.scaler.fit_transform(X_historical)
        
        # Initialize base models for pre-training
        base_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                n_jobs=-1,
                random_state=42
            ),
            'lgb': LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                n_jobs=-1,
                random_state=42
            ),
            'cat': CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                thread_count=-1,
                random_state=42,
                verbose=False
            )
        }
        
        # Pre-train each model
        for name, model in base_models.items():
            logging.info(f"Pre-training {name} model...")
            try:
                if name == 'cat':
                    model.fit(
                        X_historical_scaled, 
                        y_historical,
                        sample_weight=w_historical,
                        verbose=False
                    )
                else:
                    model.fit(
                        X_historical_scaled, 
                        y_historical,
                        sample_weight=w_historical
                    )
                
                # Calculate and log performance metrics
                y_pred = model.predict(X_historical_scaled)
                mape = mean_absolute_percentage_error(y_historical, y_pred)
                rmse = np.sqrt(mean_squared_error(y_historical, y_pred))
                logging.info(f"{name} pre-training metrics - MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
                
            except Exception as e:
                logging.error(f"Error pre-training {name} model: {str(e)}")
        
        # Store pre-trained models
        self.pretrained_models = base_models
        logging.info("Pre-training completed")

    def train_weighted_models(self, X_train_scaled, y_train, w_train, iteration):
        """Train models with sample weights"""
        models = {}
        
        # Create new model instances with updated parameters
        base_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=100 + (25 * iteration),
                learning_rate=0.1 / (1 + 0.05 * iteration),
                max_depth=min(6 + iteration // 3, 12),
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                n_jobs=-1,
                random_state=42 + iteration
            ),
            'lgb': LGBMRegressor(
                n_estimators=100 + (25 * iteration),
                learning_rate=0.1 / (1 + 0.05 * iteration),
                num_leaves=31 + iteration,
                n_jobs=-1,
                random_state=42 + iteration,
                force_row_wise=True  # Add this to avoid the warning
            ),
            'cat': CatBoostRegressor(
                iterations=100 + (25 * iteration),
                learning_rate=0.1 / (1 + 0.05 * iteration),
                depth=min(6 + iteration // 3, 12),
                thread_count=-1,
                random_state=42 + iteration,
                verbose=False
            )
        }
        
        # Train models with weights
        for name, model in base_models.items():
            try:
                if name == 'cat':
                    model.fit(X_train_scaled, y_train, 
                             sample_weight=w_train, verbose=False)
                else:
                    model.fit(X_train_scaled, y_train, 
                             sample_weight=w_train)
                models[name] = model
                
                # Log training metrics
                y_pred = model.predict(X_train_scaled)
                mape = mean_absolute_percentage_error(y_train, y_pred)
                rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                logging.info(f"{name} training metrics - MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name} model: {str(e)}")
        
        return models

    def create_weighted_ensemble(self, models, X_val_scaled, y_val):
        """Create a weighted ensemble from the trained models"""
        logging.info("Creating weighted ensemble...")
        
        # Calculate weights based on validation performance
        model_weights = {}
        predictions = {}
        
        for name, model in models.items():
            try:
                # Get predictions
                y_pred = model.predict(X_val_scaled)
                predictions[name] = y_pred
                
                # Calculate metrics
                mape = mean_absolute_percentage_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                logging.info(f"{name} validation metrics - MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
                
                # Calculate weight (inverse of error)
                weight = 1 / (mape + 0.1)  # Add small constant to avoid division by zero
                model_weights[name] = weight
                
            except Exception as e:
                logging.error(f"Error evaluating {name} model: {str(e)}")
                model_weights[name] = 0
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        if total_weight > 0:
            model_weights = {name: w/total_weight for name, w in model_weights.items()}
        else:
            # If all weights are 0, use equal weights
            model_weights = {name: 1/len(models) for name in models}
        
        logging.info(f"Initial model weights: {model_weights}")
        
        # Create VotingRegressor with weighted models
        estimators = [(name, model) for name, model in models.items()]
        weights = [model_weights[name] for name, _ in estimators]
        
        ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights
        )
        
        # Fit ensemble on validation data
        ensemble.fit(X_val_scaled, y_val)
        
        # Calculate ensemble performance
        y_pred_ensemble = ensemble.predict(X_val_scaled)
        ensemble_mape = mean_absolute_percentage_error(y_val, y_pred_ensemble)
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
        
        logging.info(f"Ensemble metrics - MAPE: {ensemble_mape:.4f}, RMSE: {ensemble_rmse:.4f}")
        
        return ensemble

    def calculate_adaptive_thresholds(self, seasonal_errors, weather_errors, time_errors):
        """Calculate adaptive error thresholds based on different conditions"""
        logging.info("Calculating adaptive thresholds...")
        
        thresholds = {}
        
        # Calculate seasonal thresholds
        seasonal_thresholds = {}
        for season, errors in seasonal_errors.items():
            if errors:  # Check if we have errors for this season
                # Use percentile-based thresholds
                seasonal_thresholds[season] = {
                    'base': np.median(errors),
                    'strict': np.percentile(errors, 25),
                    'relaxed': np.percentile(errors, 75)
                }
        thresholds['seasonal'] = seasonal_thresholds
        
        # Calculate weather condition thresholds
        weather_thresholds = {}
        for condition, errors in weather_errors.items():
            if errors:
                # Adjust thresholds based on weather conditions
                base_threshold = np.median(errors)
                if condition == 'clear_sky':
                    # Stricter thresholds for clear sky conditions
                    weather_thresholds[condition] = {
                        'base': base_threshold * 0.9,
                        'strict': base_threshold * 0.7,
                        'relaxed': base_threshold * 1.2
                    }
                elif condition == 'partly_cloudy':
                    # Moderate thresholds for partly cloudy conditions
                    weather_thresholds[condition] = {
                        'base': base_threshold,
                        'strict': base_threshold * 0.8,
                        'relaxed': base_threshold * 1.3
                    }
                else:  # overcast
                    # More relaxed thresholds for overcast conditions
                    weather_thresholds[condition] = {
                        'base': base_threshold * 1.1,
                        'strict': base_threshold * 0.9,
                        'relaxed': base_threshold * 1.4
                    }
        thresholds['weather'] = weather_thresholds
        
        # Calculate time of day thresholds
        time_thresholds = {}
        for period, errors in time_errors.items():
            if errors:
                base_threshold = np.median(errors)
                if period == 'morning':
                    # Account for morning transition
                    time_thresholds[period] = {
                        'base': base_threshold * 1.1,
                        'strict': base_threshold * 0.9,
                        'relaxed': base_threshold * 1.3
                    }
                elif period == 'afternoon':
                    # Account for afternoon convection
                    time_thresholds[period] = {
                        'base': base_threshold * 1.2,
                        'strict': base_threshold * 0.95,
                        'relaxed': base_threshold * 1.4
                    }
                else:  # evening
                    # More relaxed thresholds for evening
                    time_thresholds[period] = {
                        'base': base_threshold * 1.15,
                        'strict': base_threshold * 0.9,
                        'relaxed': base_threshold * 1.35
                    }
        thresholds['time'] = time_thresholds
        
        # Log threshold summaries
        logging.info("Calculated adaptive thresholds:")
        for category, category_thresholds in thresholds.items():
            logging.info(f"\n{category.capitalize()} thresholds:")
            for condition, values in category_thresholds.items():
                logging.info(f"  {condition}: {values}")
        
        return thresholds

    def update_model_weights(self, ensemble, error_analysis):
        """Update model weights based on error analysis"""
        if not hasattr(self, 'model_weights_history'):
            self.model_weights_history = []
        
        # Get current weights from the ensemble
        current_weights = np.array(ensemble.weights) if ensemble.weights is not None else \
                         np.ones(len(ensemble.estimators)) / len(ensemble.estimators)
        
        # Calculate adjustment factors based on error analysis
        adjustments = np.ones_like(current_weights)
        
        # Apply adjustments based on error patterns
        for i, estimator in enumerate(ensemble.estimators):
            # Get model name from the estimator type
            name = type(estimator).__name__
            
            # Seasonal adjustment
            if error_analysis.get('seasonal_bias', False):
                adjustments[i] *= 0.95
            
            # Weather condition adjustment
            if error_analysis.get('weather_bias', False):
                adjustments[i] *= 0.9
            
            # Time of day adjustment
            if error_analysis.get('time_bias', False):
                adjustments[i] *= 0.95
        
        # Update weights
        new_weights = current_weights * adjustments
        new_weights = new_weights / new_weights.sum()  # Normalize
        
        # Store weights history
        self.model_weights_history.append({
            'weights': new_weights,
            'adjustments': adjustments
        })
        
        # Update ensemble weights
        ensemble.weights = list(new_weights)  # Convert to list for VotingRegressor
        
        # Log the updates
        weight_dict = dict(zip([type(est).__name__ for est in ensemble.estimators], new_weights))
        logging.info(f"Updated model weights: {weight_dict}")
        
        return ensemble

    def log_detailed_metrics(self, metrics):
        """Log detailed performance metrics"""
        logging.info("\n=== Detailed Performance Metrics ===")
        
        # Seasonal performance
        logging.info("\nSeasonal Performance:")
        for season, values in metrics['Seasonal Performance'].items():
            logging.info(f"\n{season}:")
            for metric, value in values.items():
                logging.info(f"  {metric}: {value:.2f}")
        
        # Weather condition performance
        logging.info("\nWeather Condition Performance:")
        for condition, values in metrics['Weather Condition Performance'].items():
            logging.info(f"\n{condition}:")
            for metric, value in values.items():
                logging.info(f"  {metric}: {value:.2f}")
        
        # Time of day performance
        logging.info("\nTime of Day Performance:")
        for period, values in metrics['Time of Day Performance'].items():
            logging.info(f"\n{period}:")
            for metric, value in values.items():
                logging.info(f"  {metric}: {value:.2f}")
        
        # Overall performance
        logging.info("\nOverall Performance:")
        for metric, value in metrics['Overall Performance']['Overall'].items():
            logging.info(f"  {metric}: {value:.2f}")
        
        # Save metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f'metrics_{timestamp}.txt'
        
        with open(metrics_file, 'w') as f:
            f.write("=== Detailed Performance Metrics ===\n")
            
            # Write seasonal performance
            f.write("\nSeasonal Performance:\n")
            for season, values in metrics['Seasonal Performance'].items():
                f.write(f"\n{season}:\n")
                for metric, value in values.items():
                    f.write(f"  {metric}: {value:.2f}\n")
            
            # Write weather condition performance
            f.write("\nWeather Condition Performance:\n")
            for condition, values in metrics['Weather Condition Performance'].items():
                f.write(f"\n{condition}:\n")
                for metric, value in values.items():
                    f.write(f"  {metric}: {value:.2f}\n")
            
            # Write time of day performance
            f.write("\nTime of Day Performance:\n")
            for period, values in metrics['Time of Day Performance'].items():
                f.write(f"\n{period}:\n")
                for metric, value in values.items():
                    f.write(f"  {metric}: {value:.2f}\n")
            
            # Write overall performance
            f.write("\nOverall Performance:\n")
            for metric, value in metrics['Overall Performance']['Overall'].items():
                f.write(f"  {metric}: {value:.2f}\n")
        
        logging.info(f"\nDetailed metrics saved to {metrics_file}")

    def check_model_complexity_needs(self):
        """Check if model complexity needs to be increased"""
        if not hasattr(self, 'performance_history'):
            return False
        
        # Get recent performance metrics
        recent_metrics = self.performance_history[-3:]  # Last 3 iterations
        
        if len(recent_metrics) < 3:
            return False
        
        # Check for plateauing performance
        errors = [m.get('validation_error', float('inf')) for m in recent_metrics]
        
        # Calculate if performance is plateauing
        is_plateauing = all(
            abs(errors[i] - errors[i-1]) < 0.01 * errors[i-1]  # Less than 1% improvement
            for i in range(1, len(errors)))
        
        # Check if error is still above acceptable threshold
        error_too_high = errors[-1] > self.target_error_threshold
        
        return is_plateauing and error_too_high

    def increase_model_complexity(self):
        """Increase model complexity based on performance analysis"""
        logging.info("Increasing model complexity...")
        
        # Update XGBoost parameters
        self.model_params['xgb'].update({
            'max_depth': min(self.model_params['xgb']['max_depth'] + 1, 12),
            'n_estimators': self.model_params['xgb']['n_estimators'] + 50,
            'min_child_weight': max(self.model_params['xgb']['min_child_weight'] - 1, 1)
        })
        
        # Update LightGBM parameters
        self.model_params['lgb'].update({
            'num_leaves': min(self.model_params['lgb']['num_leaves'] * 2, 127),
            'n_estimators': self.model_params['lgb']['n_estimators'] + 50,
            'min_child_samples': max(self.model_params['lgb']['min_child_samples'] - 2, 5)
        })
        
        # Update CatBoost parameters
        self.model_params['cat'].update({
            'depth': min(self.model_params['cat']['depth'] + 1, 10),
            'iterations': self.model_params['cat']['iterations'] + 50,
            'min_data_in_leaf': max(self.model_params['cat']['min_data_in_leaf'] - 1, 1)
        })
        
        logging.info("Updated model parameters:")
        for model_name, params in self.model_params.items():
            logging.info(f"{model_name}: {params}")

    def initialize_model_params(self):
        """Initialize model parameters"""
        self.model_params = {
            'xgb': {
                'max_depth': 6,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist',
                'n_jobs': -1,
                'random_state': 42
            },
            'lgb': {
                'num_leaves': 31,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'min_child_samples': 20,
                'n_jobs': -1,
                'random_state': 42,
                'force_col_wise': True
            },
            'cat': {
                'depth': 6,
                'iterations': 100,
                'learning_rate': 0.1,
                'min_data_in_leaf': 20,
                'thread_count': -1,
                'random_state': 42,
                'verbose': False
            }
        }
        
        # Initialize target error threshold
        self.target_error_threshold = 0.15  # 15% MAPE
        
        # Initialize performance history
        self.performance_history = []

    def check_convergence_criteria(self, metrics):
        """Check if training should stop based on convergence criteria"""
        if not hasattr(self, 'performance_history') or len(self.performance_history) < 3:
            return False
        
        # Get recent performance metrics
        recent_metrics = self.performance_history[-3:]
        errors = [m.get('validation_error', float('inf')) for m in recent_metrics]
        
        # Check for convergence conditions
        error_threshold_met = errors[-1] <= self.target_error_threshold
        
        # Calculate relative improvement
        improvements = [
            abs(errors[i] - errors[i-1]) / errors[i-1]
            for i in range(1, len(errors))
        ]
        
        # Check if improvements are diminishing
        diminishing_returns = all(imp < 0.01 for imp in improvements)  # Less than 1% improvement
        
        # Check iteration count
        max_iterations_reached = len(self.performance_history) >= self.max_iterations
        
        # Log convergence status
        logging.info("\nConvergence Check:")
        logging.info(f"  Error threshold met: {error_threshold_met}")
        logging.info(f"  Diminishing returns: {diminishing_returns}")
        logging.info(f"  Max iterations reached: {max_iterations_reached}")
        logging.info(f"  Recent errors: {errors}")
        logging.info(f"  Recent improvements: {improvements}")
        
        # Additional checks for Davao-specific conditions
        current_month = datetime.now().month
        current_hour = datetime.now().hour
        
        # More stringent convergence during clear weather seasons
        if current_month in [1, 2, 3, 4]:  # Dry season
            error_threshold_met = errors[-1] <= self.target_error_threshold * 0.9
        
        # More relaxed convergence during wet season
        elif current_month in [6, 7, 8, 9, 10, 11]:
            error_threshold_met = errors[-1] <= self.target_error_threshold * 1.1
        
        # Adjust for time of day
        if current_hour in [14, 15, 16, 17]:  # Afternoon convection period
            error_threshold_met = errors[-1] <= self.target_error_threshold * 1.2
        
        # Save convergence status
        convergence_status = {
            'iteration': len(self.performance_history),
            'final_error': errors[-1],
            'error_threshold_met': error_threshold_met,
            'diminishing_returns': diminishing_returns,
            'max_iterations_reached': max_iterations_reached,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save convergence status to file
        with open('convergence_history.txt', 'a') as f:
            f.write(f"\n{json.dumps(convergence_status)}")
        
        # Return True if any stopping condition is met
        return error_threshold_met or (diminishing_returns and len(self.performance_history) > 5) or max_iterations_reached

    def initialize_convergence_params(self):
        """Initialize convergence parameters"""
        self.max_iterations = 20
        self.target_error_threshold = 0.15  # 15% MAPE
        self.min_improvement_threshold = 0.01  # 1% minimum improvement
        self.convergence_window = 3  # Number of iterations to check for convergence
        
        # Initialize performance history if not exists
        if not hasattr(self, 'performance_history'):
            self.performance_history = []

    def load_best_model(self, model_path):
        """Load the best saved model"""
        try:
            # Load configuration
            with open(f'{model_path}_config.json', 'r') as f:
                config = json.load(f)
            logging.info(f"Loaded configuration with error: {config['final_error']:.4f}")
            
            # Load individual models
            models = {}
            for model_type in ['xgb', 'lgb', 'cat']:
                with open(f'{model_path}_{model_type}.pkl', 'rb') as f:
                    models[model_type] = pickle.load(f)
            
            # Load ensemble
            with open(f'{model_path}_ensemble.pkl', 'rb') as f:
                ensemble = pickle.load(f)
            
            # Load scaler
            with open(f'{model_path}_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            logging.info("Successfully loaded best model and components")
            return models, ensemble
            
        except Exception as e:
            logging.error(f"Error loading best model: {str(e)}")
            return None, None

    def analyze_and_adjust_model(self, metrics, models, ensemble):
        """Analyze metrics and make targeted improvements"""
        logging.info("Analyzing metrics and making adjustments...")
        
        # Extract key performance indicators
        time_performance = {
            'morning': metrics['Time of Day Performance']['Morning']['Mean Error'],
            'afternoon': metrics['Time of Day Performance']['Afternoon']['Mean Error'],
            'evening': metrics['Time of Day Performance']['Evening']['Mean Error']
        }
        
        weather_performance = {
            'clear_sky': metrics['Weather Condition Performance']['Clear_sky']['Mean Error'],
            'partly_cloudy': metrics['Weather Condition Performance']['Partly_cloudy']['Mean Error'],
            'overcast': metrics['Weather Condition Performance']['Overcast']['Mean Error']
        }
        
        # Identify areas needing improvement
        adjustments = {}
        
        # Adjust for time of day performance
        if time_performance['afternoon'] > 250:  # High afternoon errors
            adjustments['afternoon'] = {
                'learning_rate': self.model_params['xgb']['learning_rate'] * 0.9,
                'max_depth': min(self.model_params['xgb']['max_depth'] + 1, 12),
                'min_child_weight': max(self.model_params['xgb']['min_child_weight'] - 1, 1)
            }
        
        # Adjust for weather conditions
        if weather_performance['clear_sky'] > 270:  # High clear sky errors
            adjustments['clear_sky'] = {
                'subsample': min(self.model_params['xgb']['subsample'] + 0.1, 1.0),
                'colsample_bytree': min(self.model_params['xgb']['colsample_bytree'] + 0.1, 1.0)
            }
        
        # Apply adjustments to models
        for condition, params in adjustments.items():
            logging.info(f"Applying adjustments for {condition}: {params}")
            self.update_model_parameters(models, params, condition)
        
        return models, ensemble

    def update_model_parameters(self, models, new_params, condition):
        """Update model parameters based on performance analysis"""
        for model_name, model in models.items():
            if model_name == 'xgb':
                # XGBoost specific adjustments
                if hasattr(model, 'get_params'):
                    current_params = model.get_params()
                    for param, value in new_params.items():
                        if param in current_params:
                            current_params[param] = value
                    model.set_params(**current_params)
            
            elif model_name == 'lgb':
                # LightGBM specific adjustments
                if condition == 'afternoon':
                    model.set_params(
                        num_leaves=min(model.get_params()['num_leaves'] + 4, 127),
                        min_child_samples=max(model.get_params()['min_child_samples'] - 2, 5))
            
            elif model_name == 'cat':
                # CatBoost specific adjustments
                if condition == 'clear_sky':
                    model.set_params(
                        depth=min(model.get_params()['depth'] + 1, 10),
                        min_data_in_leaf=max(model.get_params()['min_data_in_leaf'] - 1, 1))

    def calculate_detailed_metrics(self, ensemble, X_val, y_val):
        """Calculate detailed performance metrics"""
        logging.info("Calculating detailed metrics...")
        
        # Get predictions
        y_pred = ensemble.predict(X_val)
        
        # Initialize metrics dictionary
        metrics = {
            'Seasonal Performance': {},
            'Weather Condition Performance': {},
            'Time of Day Performance': {},
            'Overall Performance': {}
        }
        
        # Get time components using validation indices
        time_index = self.original_index[self.val_idx]
        hour = pd.DatetimeIndex(time_index).hour
        month = pd.DatetimeIndex(time_index).month
        
        # Convert y_val and y_pred to numpy arrays for calculations
        y_val_np = y_val.to_numpy()
        y_pred_np = np.array(y_pred)
        
        # Seasonal analysis
        seasons = {
            'Dry_period': [1, 2, 3, 4],
            'Transitional': [5, 6],
            'Wet_period': [7, 8, 9, 10, 11, 12]
        }
        
        for season, months in seasons.items():
            mask = np.isin(month, months)
            errors = np.abs(y_val_np[mask] - y_pred_np[mask])
            if len(errors) > 0:
                metrics['Seasonal Performance'][season] = {
                    'Mean Error': float(np.mean(errors)),
                    'Std Dev': float(np.std(errors)),
                    'Median Error': float(np.median(errors)),
                    '90th Percentile': float(np.percentile(errors, 90))
                }
        
        # Weather condition analysis
        X_val_np = X_val.to_numpy()
        cloud_opacity = X_val['cloud_opacity'].to_numpy()
        
        weather_conditions = {
            'Clear_sky': cloud_opacity < 0.2,
            'Partly_cloudy': (cloud_opacity >= 0.2) & (cloud_opacity < 0.8),
            'Overcast': cloud_opacity >= 0.8
        }
        
        for condition, mask in weather_conditions.items():
            errors = np.abs(y_val_np[mask] - y_pred_np[mask])
            if len(errors) > 0:
                metrics['Weather Condition Performance'][condition] = {
                    'Mean Error': float(np.mean(errors)),
                    'Std Dev': float(np.std(errors)),
                    'Median Error': float(np.median(errors)),
                    '90th Percentile': float(np.percentile(errors, 90))
                }
        
        # Time of day analysis
        time_periods = {
            'Morning': (hour >= 6) & (hour < 12),
            'Afternoon': (hour >= 12) & (hour < 18),
            'Evening': (hour >= 18) | (hour < 6)
        }
        
        for period, mask in time_periods.items():
            errors = np.abs(y_val_np[mask] - y_pred_np[mask])
            if len(errors) > 0:
                metrics['Time of Day Performance'][period] = {
                    'Mean Error': float(np.mean(errors)),
                    'Std Dev': float(np.std(errors)),
                    'Median Error': float(np.median(errors)),
                    '90th Percentile': float(np.percentile(errors, 90))
                }
        
        # Overall performance
        overall_errors = np.abs(y_val_np - y_pred_np)
        metrics['Overall Performance'] = {
            'Overall': {  # Added nested dictionary for consistency
                'Mean Error': float(np.mean(overall_errors)),
                'Std Dev': float(np.std(overall_errors)),
                'Median Error': float(np.median(overall_errors)),
                '90th Percentile': float(np.percentile(overall_errors, 90))
            }
        }
        
        # Save metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f'metrics_{timestamp}.txt'
        
        with open(metrics_file, 'w') as f:
            f.write("=== Detailed Performance Metrics ===\n\n")
            
            for category, subcategories in metrics.items():
                f.write(f"{category}:\n\n")
                for subcategory, metrics_dict in subcategories.items():
                    f.write(f"{subcategory}:\n")
                    for metric_name, metric_value in metrics_dict.items():
                        f.write(f"  {metric_name}: {metric_value:.2f}\n")
                    f.write("\n")
        
        logging.info(f"Detailed metrics saved to {metrics_file}")
        return metrics

    def save_trained_model(self):
        """Save the trained model and necessary components"""
        try:
            # Get the current working directory and create full path
            current_dir = os.getcwd()
            model_path = os.path.join(current_dir, 'models')
            
            # Create models directory if it doesn't exist
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            # Verify we have a model to save
            if not hasattr(self, 'best_model') or self.best_model is None:
                logging.error("No model to save")
                return False
            
            # Save the ensemble model
            ensemble_path = os.path.join(model_path, 'ensemble.pkl')
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            # Save the scaler
            scaler_path = os.path.join(model_path, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Ensure we have selected features
            if not hasattr(self, 'selected_features'):
                self.selected_features = self.feature_columns
            
            # Save feature list and other necessary components
            model_info = {
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else {},
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            info_path = os.path.join(model_path, 'info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f)
                
            logging.info(f"Model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            return False

    def load_trained_model(self):
        """Load the trained model and necessary components"""
        print("\nDEBUG - Starting model loading process...")
        
        try:
            # Get paths
            current_dir = os.getcwd()
            model_path = os.path.join(current_dir, 'models')
            
            print(f"\nCurrent working directory: {current_dir}")
            print(f"Looking for models in: {model_path}")
            
            # Check if models directory exists
            if not os.path.exists(model_path):
                print(f"\nERROR: Models directory not found at {model_path}")
                return False
                
            # List contents of models directory
            print("\nContents of models directory:")
            try:
                files = os.listdir(model_path)
                print(files)
            except Exception as e:
                print(f"Error listing directory contents: {str(e)}")
            
            # Check for required files
            required_files = {
                'ensemble': os.path.join(model_path, 'ensemble.pkl'),
                'scaler': os.path.join(model_path, 'scaler.pkl'),
                'info': os.path.join(model_path, 'info.json')
            }
            
            print("\nChecking for required files:")
            for file_type, file_path in required_files.items():
                exists = os.path.exists(file_path)
                print(f"{file_type}: {file_path}")
                print(f"Exists: {exists}")
                
                if not exists:
                    print(f"ERROR: Missing required file: {file_path}")
                    return False
            
            print("\nAttempting to load model files...")
            
            # Load ensemble
            try:
                with open(required_files['ensemble'], 'rb') as f:
                    self.best_model = pickle.load(f)
                    print("Successfully loaded ensemble model")
            except Exception as e:
                print(f"Error loading ensemble model: {str(e)}")
                return False
            
            # Load scaler
            try:
                with open(required_files['scaler'], 'rb') as f:
                    self.scaler = pickle.load(f)
                    print("Successfully loaded scaler")
            except Exception as e:
                print(f"Error loading scaler: {str(e)}")
                return False
            
            # Load info
            try:
                with open(required_files['info'], 'r') as f:
                    model_info = json.load(f)
                    self.selected_features = model_info['selected_features']
                    self.feature_importance = model_info.get('feature_importance', {})
                    print("Successfully loaded model info")
            except Exception as e:
                print(f"Error loading model info: {str(e)}")
                return False
            
            print("\nAll model components loaded successfully!")
            return True
            
        except Exception as e:
            print(f"\nUnexpected error in load_trained_model: {str(e)}")
            return False

    def predict_next_hour(self, use_last_data=True):
        """Predict GHI for the next hour using last available data"""
        try:
            print("\nDEBUG - Starting prediction process...")
            
            # Load model if not loaded
            if not hasattr(self, 'best_model') or self.best_model is None:
                load_success = self.load_trained_model()
                if not load_success:
                    raise ValueError("No trained model found. Please train a model first.")
            
            if use_last_data:
                print("\nLoading last hour's data from raw.csv...")
                try:
                    # Load the raw data
                    df = pd.read_csv('raw.csv')
                    df['period_end'] = pd.to_datetime(df['period_end'])
                    
                    # Get the last row of data
                    last_data = df.iloc[-1].to_dict()
                    
                    print(f"\nUsing data from: {last_data['period_end']}")
                    print("Last hour's values:")
                    print(f"GHI: {last_data['ghi']:.2f}")
                    print(f"Clearsky GHI: {last_data['clearsky_ghi']:.2f}")
                    print(f"Cloud Opacity: {last_data['cloud_opacity']:.2f}")
                    print(f"Air Temperature: {last_data['air_temp']:.2f}°C")
                    
                    # Create input DataFrame
                    X = pd.DataFrame([last_data])
                    
                    # Engineer features
                    X = self.engineer_features(X)
                    
                    # Select features
                    X_selected = X[self.selected_features]
                    
                    # Scale features
                    X_scaled = self.scaler.transform(X_selected)
                    
                    # Make prediction
                    prediction = self.best_model.predict(X_scaled)[0]
                    
                    print(f"\nPredicted GHI for next hour: {prediction:.2f}")
                    
                    # Calculate and print prediction time
                    last_time = pd.to_datetime(last_data['period_end'])
                    next_hour = last_time + pd.Timedelta(hours=1)
                    print(f"Prediction for: {next_hour}")
                    
                    return prediction, next_hour
                    
                except Exception as e:
                    print(f"Error processing last hour's data: {str(e)}")
                    raise
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

def main():
    predictor = GHIPredictionSystem('raw.csv', target_error=0.05)
    
    while True:
        print("\n=== GHI Prediction System Menu ===")
        print("1. Train New Model")
        print("2. Predict Next Hour GHI")
        print("3. Exit")
        print("===============================")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            print("\nInitiating model training...")
            try:
                predictor.train_and_optimize()
                predictor.save_trained_model()
                print("Model training completed successfully!")
                print("The trained model has been saved.")
            except Exception as e:
                print(f"Error during training: {str(e)}")
                
        elif choice == '2':
            print("\nPredicting next hour GHI...")
            try:
                prediction, next_hour = predictor.predict_next_hour(use_last_data=True)
                print("\nPrediction Summary:")
                print(f"Time: {next_hour}")
                print(f"Predicted GHI: {prediction:.2f}")
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                
        elif choice == '3':
            print("\nExiting program...")
            break
            
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
