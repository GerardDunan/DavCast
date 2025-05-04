import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import os
import joblib
from datetime import datetime

# Function to calculate prediction interval metrics
def calculate_pi_metrics(actual, lower, upper, target_coverage=0.95):
    """
    Calculate prediction interval metrics.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    lower : array-like
        Lower bound of prediction interval
    upper : array-like
        Upper bound of prediction interval
    target_coverage : float, default=0.95
        Target coverage probability
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Calculate prediction interval width (PIW)
    piw = np.mean(upper - lower)
    
    # Calculate prediction interval coverage probability (PICP)
    picp = np.mean((actual >= lower) & (actual <= upper))
    
    # Calculate mean absolute coverage error (MACE)
    mace = abs(picp - target_coverage)
    
    # Calculate coverage width-based criterion (CWC)
    # Using the formula: CWC = PINAW[1 + φ(PICP)·exp(-η(PICP - PINC))]
    # where φ(PICP) = 1 if PICP < PINC, 0 otherwise
    
    # Calculate PINAW (Prediction Interval Normalized Average Width)
    range_actual = np.max(actual) - np.min(actual)
    pinaw = piw / range_actual if range_actual != 0 else piw
    
    # Set parameters
    eta = 10  # Trade-off parameter
    pinc = 0.95  # Target coverage (95% confidence interval)
    
    # Calculate φ(PICP)
    phi = 1.0 if picp < pinc else 0.0
    
    # Calculate CWC
    cwc = pinaw * (1 + phi * np.exp(-eta * (picp - pinc)))
    
    # Calculate number of samples outside interval
    n_outside = np.sum((actual < lower) | (actual > upper))
    
    # Calculate mean interval score (MIS)
    alpha = 1 - target_coverage
    mis = np.mean(upper - lower + (2/alpha) * (lower - actual) * (actual < lower) + (2/alpha) * (actual - upper) * (actual > upper))
    
    return {
        'PIW': piw,
        'PINAW': pinaw,
        'PICP': picp,
        'MACE': mace,
        'CWC': cwc,
        'n_outside': n_outside,
        'MIS': mis
    }

# Function to calculate hourly metrics for each horizon
def calculate_hourly_metrics(df, horizons, alpha=0.05):
    """
    Calculate hourly metrics for each horizon
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with predictions and true values
    horizons : list
        List of horizons to calculate metrics for
    alpha : float, optional
        Significance level, by default 0.05 (for 95% confidence)
        
    Returns:
    --------
    dict
        Dictionary with hourly metrics for each horizon
    """
    hourly_metrics = {}
    
    for horizon in horizons:
        horizon_metrics = {}
        
        # Group by hour
        df_hourly = df.groupby('Hour of Day')
        
        for hour, hour_df in df_hourly:
            true_vals = hour_df[f'GHI_t+{horizon}']
            pred_vals = hour_df[f'pred_t+{horizon}']
            lower_vals = hour_df[f'lower_t+{horizon}']
            upper_vals = hour_df[f'upper_t+{horizon}']
            
            metrics = calculate_pi_metrics(true_vals, lower_vals, upper_vals)
            horizon_metrics[hour] = metrics
        
        hourly_metrics[horizon] = horizon_metrics
    
    return hourly_metrics

# Function to create horizon targets
def create_horizon_targets(df, horizons, target_col='GHI - W/m^2'):
    """
    Create targets for each horizon
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    horizons : list
        List of horizons to create targets for
    target_col : str, optional
        Target column, by default 'GHI - W/m^2'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with targets for each horizon
    """
    df_copy = df.copy()
    
    for h in horizons:
        df_copy[f'GHI_t+{h}'] = df_copy[target_col].shift(-h)
    
    # Drop rows with NaN values (at the end, where we don't have targets)
    df_copy = df_copy.dropna(subset=[f'GHI_t+{h}' for h in horizons])
    
    return df_copy

# Function to save trained models and scaler
def save_models(models, scaler, feature_cols, horizons, output_dir='models'):
    """
    Save trained models and scaler to disk.
    
    Parameters:
    -----------
    models : dict
        Dictionary containing trained models
    scaler : RobustScaler
        Fitted scaler
    feature_cols : list
        Feature column names
    horizons : list
        List of forecast horizons
    output_dir : str, optional
        Directory to save models to, by default 'models'
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save time-stamped model version
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    version_dir = os.path.join(output_dir, f"version_{timestamp}")
    os.makedirs(version_dir, exist_ok=True)
    
    # Save feature columns
    with open(os.path.join(version_dir, 'feature_columns.txt'), 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    # Save horizons
    with open(os.path.join(version_dir, 'horizons.txt'), 'w') as f:
        for h in horizons:
            f.write(f"{h}\n")
    
    # Save scaler
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save each model using both XGBoost native format and joblib
    for model_name, model in models.items():
        # Save using XGBoost's native method
        model_path = os.path.join(version_dir, f"model_{model_name}.json")
        model.save_model(model_path)
        print(f"Model {model_name} saved to {model_path} (XGBoost format)")
        
        # Also save using joblib for scikit-learn compatibility
        joblib_path = os.path.join(version_dir, f"model_{model_name}.joblib")
        joblib.dump(model, joblib_path)
        print(f"Model {model_name} saved to {joblib_path} (joblib format)")
    
    # Create a readme file with model information
    with open(os.path.join(version_dir, 'README.txt'), 'w') as f:
        f.write(f"XGBoost Probabilistic Forecasting Models\n")
        f.write(f"=======================================\n\n")
        f.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Models included:\n")
        for h in horizons:
            f.write(f"- Horizon t+{h}: median, lower (2.5%), upper (97.5%)\n")
        f.write(f"\nFeatures used: {len(feature_cols)}\n")
        f.write(f"Total models: {len(models)}\n")
        f.write(f"\nFile formats:\n")
        f.write(f"- .json files: XGBoost native format\n")
        f.write(f"- .joblib files: scikit-learn compatible format\n")
    
    # Save a pointer to the latest version
    with open(os.path.join(output_dir, 'latest_version.txt'), 'w') as f:
        f.write(version_dir)
    
    print(f"\nAll models and metadata saved to {version_dir}")
    return version_dir

# Main script
def main():
    print("Loading dataset...")
    df = pd.read_csv('dav/dataset.csv')
    
    # Filter only daytime data
    df = df[df['Daytime'] == 1]
    
    # Define horizons
    horizons = [1, 2, 3, 4]
    
    # Create horizon targets
    df = create_horizon_targets(df, horizons)
    
    # Define features
    feature_cols = [
        'Barometer - hPa', 'Temp - °C', 'Hum - %', 'Dew Point - °C',
        'Wet Bulb - °C', 'Avg Wind Speed - km/h', 'Rain - mm',
        'High Rain Rate - mm/h', 'GHI - W/m^2', 'UV Index', 'Wind Run - km',
        'Month of Year', 'Hour of Day', 'Solar Zenith Angle', 'GHI_lag (t-1)'
    ]
    
    # Split data sequentially: 80% train, 20% validation
    train_size = int(0.8 * len(df))
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:]
    
    # Prepare scaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(df_train[feature_cols])
    X_val = scaler.transform(df_val[feature_cols])
    
    # Prediction results
    results = df_val.copy()
    
    # Store Optuna studies for each horizon
    studies = {}
    
    # Store metrics for each horizon
    metrics_dfs = {}
    
    # Store models for feature importance
    models = {}
    
    # Define fixed hyperparameters for each horizon
    fixed_hyperparameters = {
        1: {
            'learning_rate': 0.299886,
            'max_depth': 3,
            'min_child_weight': 5,
            'subsample': 0.873891,
            'colsample_bytree': 0.833698703,
            'gamma': 0.039496299,
            'reg_alpha': 4.684330146,
            'reg_lambda': 1.515566684,
            'random_state': 42
        },
        2: {
            'learning_rate': 0.092545,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.987069,
            'colsample_bytree': 0.731380785,
            'gamma': 0.008642597,
            'reg_alpha': 4.810797539,
            'reg_lambda': 0.045691497,
            'random_state': 42
        },
        3: {
            'learning_rate': 0.291604,
            'max_depth': 5,
            'min_child_weight': 8,
            'subsample': 0.907255,
            'colsample_bytree': 0.679264793,
            'gamma': 0.030007779,
            'reg_alpha': 0.943774354,
            'reg_lambda': 4.443179255,
            'random_state': 42
        },
        4: {
            'learning_rate': 0.020517,
            'max_depth': 5,
            'min_child_weight': 7,
            'subsample': 0.980632,
            'colsample_bytree': 0.909183243,
            'gamma': 0.17127282,
            'reg_alpha': 4.705228623,
            'reg_lambda': 0.02276723,
            'random_state': 42
        }
    }

    # Hyperparameter optimization and model training for each horizon
    for horizon in horizons:
        print(f"\nTraining model for horizon t+{horizon}...")
        target_col = f'GHI_t+{horizon}'
        y_train = df_train[target_col].values
        y_val = df_val[target_col].values
        
        # Use fixed hyperparameters instead of Optuna optimization
        base_params = fixed_hyperparameters[horizon].copy()
        print(f"Using fixed hyperparameters for horizon t+{horizon}: {base_params}")
        
        # Create parameter sets for each model
        lower_params = base_params.copy()
        lower_params['objective'] = 'reg:quantileerror'
        lower_params['quantile_alpha'] = 0.025
        
        median_params = base_params.copy()
        median_params['objective'] = 'reg:squarederror'
        
        upper_params = base_params.copy()
        upper_params['objective'] = 'reg:quantileerror'
        upper_params['quantile_alpha'] = 0.975
        
        # Create and train the models
        lower_model = xgb.XGBRegressor(**lower_params)
        median_model = xgb.XGBRegressor(**median_params)
        upper_model = xgb.XGBRegressor(**upper_params)
        
        # Fit models
        lower_model.fit(X_train, y_train)
        median_model.fit(X_train, y_train)
        upper_model.fit(X_train, y_train)
        
        # Store models for feature importance
        models[f'{horizon}_lower'] = lower_model
        models[f'{horizon}_median'] = median_model 
        models[f'{horizon}_upper'] = upper_model
        
        # Make predictions
        results[f'pred_t+{horizon}'] = median_model.predict(X_val)
        results[f'lower_t+{horizon}'] = lower_model.predict(X_val)
        results[f'upper_t+{horizon}'] = upper_model.predict(X_val)
        
        # Calculate metrics
        metrics = calculate_pi_metrics(y_val, 
                                      results[f'lower_t+{horizon}'], 
                                      results[f'upper_t+{horizon}'])
        
        # Store metrics in a DataFrame for later use
        metrics_df = pd.DataFrame({
            'metric': list(metrics.keys()),
            'value': list(metrics.values())
        })
        metrics_dfs[horizon] = metrics_df
        
        print(f"\nValidation metrics for horizon t+{horizon}:")
        print(f"PICP: {metrics['PICP']:.4f} (target: 0.95)")
        print(f"PINAW: {metrics['PINAW']:.4f} W/m²")
        print(f"MACE: {metrics['MACE']:.4f} W/m²")
        print(f"CWC: {metrics['CWC']:.4f}")
        print(f"n_outside: {metrics['n_outside']}")
        print(f"MIS: {metrics['MIS']:.4f} W/m²")
        
        # Feature importance
        feature_importance = median_model.feature_importances_
        feature_names = feature_cols
        
        print(f"\nFeature importance for horizon t+{horizon}:")
        for feature, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
    
    # Calculate hourly metrics
    print("\nCalculating hourly metrics...")
    hourly_metrics = calculate_hourly_metrics(results, horizons)
    
    # Print hourly metrics for each horizon
    
    # Prepare data for combined hourly CSV
    combined_hourly_data = []
    
    for horizon in horizons:
        print(f"\n===============================================")
        print(f"HOURLY PROBABILISTIC METRICS FOR HORIZON t+{horizon}")
        print(f"===============================================")
        print("Time Period  |   PICP   |    PIW    |    PINAW    |   Winkler   |    CWC    |    MAE    |")
        print("-------------|----------|-----------|-------------|-------------|-----------|-----------|")
        
        for hour in range(6, 19):  # Daytime hours (6:00 to 18:00)
            metrics = hourly_metrics[horizon].get(hour, None)
            if metrics is not None:
                start_time = f"{hour:02d}:00:00"
                
                # Calculate MAE for this hour
                hour_true = results[f'GHI_t+{horizon}'][results['Hour of Day'] == hour]
                hour_pred = results[f'pred_t+{horizon}'][results['Hour of Day'] == hour]
                hour_mae = mean_absolute_error(hour_true, hour_pred)
                
                print(f"{hour:02d}:00:00   | {metrics['PICP']:.3f}  | {metrics['PIW']:.2f}    | {metrics['PINAW']:.4f}    | {metrics['MIS']:.2f}      | {metrics['CWC']:.2f}    | {hour_mae:.2f}    |")
                
                # Add data to combined hourly data
                combined_hourly_data.append({
                    'Hour': hour,
                    'Time Period': start_time,
                    'Horizon': f"t+{horizon}",
                    'PICP': metrics['PICP'],
                    'PIW': metrics['PIW'],
                    'PINAW': metrics['PINAW'],
                    'Winkler': metrics['MIS'],
                    'CWC': metrics['CWC'],
                    'MAE': hour_mae,
                    'n_outside': metrics['n_outside'],
                    'MACE': metrics['MACE']
                })
        
        # Calculate overall metrics
        overall_metrics = calculate_pi_metrics(
            results[f'GHI_t+{horizon}'], 
            results[f'lower_t+{horizon}'], 
            results[f'upper_t+{horizon}']
        )
        
        # Calculate overall MAE
        overall_mae = mean_absolute_error(results[f'GHI_t+{horizon}'], results[f'pred_t+{horizon}'])
        
        print("-------------|----------|-----------|-------------|-------------|-----------|-----------|")
        print(f"Overall      | {overall_metrics['PICP']:.3f}  | {overall_metrics['PIW']:.2f}    | {overall_metrics['PINAW']:.4f}    | {overall_metrics['MIS']:.2f}      | {overall_metrics['CWC']:.2f}    | {overall_mae:.2f}    |")
        print("==========================================================================================")
        
        # Add overall metrics to combined data
        combined_hourly_data.append({
            'Hour': 'Overall',
            'Time Period': 'Overall',
            'Horizon': f"t+{horizon}",
            'PICP': overall_metrics['PICP'],
            'PIW': overall_metrics['PIW'],
            'PINAW': overall_metrics['PINAW'],
            'Winkler': overall_metrics['MIS'],
            'CWC': overall_metrics['CWC'],
            'MAE': overall_mae,
            'n_outside': overall_metrics['n_outside'],
            'MACE': overall_metrics['MACE']
        })
    
    # Create and save combined hourly metrics DataFrame
    combined_hourly_df = pd.DataFrame(combined_hourly_data)
    
    # Sort the DataFrame by Hour (putting 'Overall' at the end) and Horizon
    combined_hourly_df['Hour_sort'] = pd.Categorical(
        combined_hourly_df['Hour'],
        categories=list(range(6, 19)) + ['Overall'],
        ordered=True
    )
    combined_hourly_df = combined_hourly_df.sort_values(['Hour_sort', 'Horizon'])
    combined_hourly_df = combined_hourly_df.drop('Hour_sort', axis=1)
    
    # Save to CSV
    os.makedirs('results', exist_ok=True)
    combined_hourly_df.to_csv('results/hourly_metrics_all_horizons.csv', index=False)
    print(f"\nCombined hourly metrics for all horizons saved to results/hourly_metrics_all_horizons.csv")
    
    # Create directory for results if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create subdirectory for hyperparameters if it doesn't exist
    hyperparams_dir = os.path.join(results_dir, "hyperparameters")
    if not os.path.exists(hyperparams_dir):
        os.makedirs(hyperparams_dir)

    # Save best hyperparameters to CSV
    best_hyperparams_data = {}
    for horizon in horizons:
        best_hyperparams_data[f'horizon_{horizon}'] = {
            'learning_rate': fixed_hyperparameters[horizon]['learning_rate'],
            'max_depth': fixed_hyperparameters[horizon]['max_depth'],
            'min_child_weight': fixed_hyperparameters[horizon]['min_child_weight'],
            'subsample': fixed_hyperparameters[horizon]['subsample'],
            'colsample_bytree': fixed_hyperparameters[horizon]['colsample_bytree'],
            'gamma': fixed_hyperparameters[horizon]['gamma'],
            'reg_alpha': fixed_hyperparameters[horizon]['reg_alpha'],
            'reg_lambda': fixed_hyperparameters[horizon]['reg_lambda'],
            'best_objective_value': 0  # Fixed hyperparameters, no Optuna optimization
        }

    # Convert nested dictionary to DataFrame
    hyperparams_df = pd.DataFrame.from_dict({(h): best_hyperparams_data[h] 
                                             for h in best_hyperparams_data.keys()}, 
                                             orient='index')
    hyperparams_df.index.name = 'horizon'
    hyperparams_df.reset_index(inplace=True)

    # Save to CSV
    hyperparams_csv_path = os.path.join(hyperparams_dir, f'best_hyperparameters_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    hyperparams_df.to_csv(hyperparams_csv_path, index=False)
    print(f"Best hyperparameters saved to {hyperparams_csv_path}")

    # Save combined performance metrics to CSV
    performance_metrics_csv_path = os.path.join(results_dir, f'performance_metrics_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    dfs = []
    for h, metrics_df in metrics_dfs.items():
        metrics_df['horizon'] = h
        dfs.append(metrics_df)
    combined_metrics_df = pd.concat(dfs)
    combined_metrics_df.to_csv(performance_metrics_csv_path, index=False)
    print(f"Performance metrics saved to {performance_metrics_csv_path}")

    # Calculate and save normalized feature importance for each model
    feature_importance_dir = os.path.join(results_dir, "feature_importance")
    if not os.path.exists(feature_importance_dir):
        os.makedirs(feature_importance_dir)

    feature_importances = {}
    # Dictionary to store overall feature importance
    overall_feature_importances = {'median': {}, 'lower': {}, 'upper': {}}
    feature_counts = {'median': {}, 'lower': {}, 'upper': {}}
    
    # Create mapping from feature index to feature name
    feature_map = {f'f{i}': name for i, name in enumerate(feature_cols)}
    
    for horizon in horizons:
        # Get feature importance for each model
        for q in ['median', 'lower', 'upper']:
            model_key = f'{horizon}_{q}'
            model = models[model_key]
            feature_importance = model.get_booster().get_score(importance_type='gain')
            
            # Normalize feature importance
            total_importance = sum(feature_importance.values())
            normalized_importance = {k: v/total_importance for k, v in feature_importance.items()}
            
            # Map feature IDs to actual feature names
            mapped_importance = {feature_map.get(feature_id, feature_id): importance 
                                for feature_id, importance in normalized_importance.items()}
            
            # Store in dictionary
            if horizon not in feature_importances:
                feature_importances[horizon] = {}
            feature_importances[horizon][q] = mapped_importance
            
            # Add to overall feature importance
            for feature, importance in mapped_importance.items():
                if feature not in overall_feature_importances[q]:
                    overall_feature_importances[q][feature] = importance
                    feature_counts[q][feature] = 1
                else:
                    overall_feature_importances[q][feature] += importance
                    feature_counts[q][feature] += 1

    # Calculate average feature importance across all horizons
    for q in ['median', 'lower', 'upper']:
        for feature in overall_feature_importances[q]:
            overall_feature_importances[q][feature] /= feature_counts[q][feature]

    # Convert to DataFrame and save
    for horizon, importance_dict in feature_importances.items():
        # Convert to DataFrame
        importance_df = pd.DataFrame(importance_dict)
        importance_df.index.name = 'feature'
        importance_df.reset_index(inplace=True)
        
        # Save to CSV
        importance_path = os.path.join(feature_importance_dir, f'feature_importance_h{horizon}.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance for horizon {horizon} saved to {importance_path}")
    
    # Save overall feature importance
    overall_importance_df = pd.DataFrame(overall_feature_importances)
    overall_importance_df.index.name = 'feature'
    overall_importance_df.reset_index(inplace=True)
    overall_importance_path = os.path.join(feature_importance_dir, 'feature_importance_overall.csv')
    overall_importance_df.to_csv(overall_importance_path, index=False)
    print(f"Overall feature importance across all horizons saved to {overall_importance_path}")
    
    # Save models and scaler
    print("\nSaving models and scaler to disk...")
    models_dir = save_models(models, scaler, feature_cols, horizons)
    print(f"Models saved successfully! You can load them for future predictions.")
    
    # Create a prediction function example file
    prediction_example_path = os.path.join(models_dir, 'prediction_example.py')
    with open(prediction_example_path, 'w') as f:
        f.write("""import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os

def load_models(model_dir, use_joblib=False):
    \"\"\"
    Load models and scaler from the specified directory
    
    Parameters:
    -----------
    model_dir : str
        Directory containing the saved models
    use_joblib : bool, optional
        Whether to load models from .joblib files instead of .json files, by default False
        
    Returns:
    --------
    tuple
        (models, scaler, feature_cols, horizons)
    \"\"\"
    # Load scaler
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    
    # Load feature columns
    with open(os.path.join(model_dir, 'feature_columns.txt'), 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    # Load horizons
    with open(os.path.join(model_dir, 'horizons.txt'), 'r') as f:
        horizons = [int(line.strip()) for line in f.readlines()]
    
    # Load models
    models = {}
    for horizon in horizons:
        for model_type in ['lower', 'median', 'upper']:
            model_name = f"{horizon}_{model_type}"
            
            if use_joblib:
                # Load model using joblib
                model_path = os.path.join(model_dir, f"model_{model_name}.joblib")
                model = joblib.load(model_path)
                print(f"Loaded model {model_name} from joblib format")
            else:
                # Load model using XGBoost native format
                model_path = os.path.join(model_dir, f"model_{model_name}.json")
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                print(f"Loaded model {model_name} from XGBoost format")
                
            models[model_name] = model
    
    return models, scaler, feature_cols, horizons

def predict(df, models, scaler, feature_cols, horizons):
    \"\"\"
    Make probabilistic predictions for the input dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with features
    models : dict
        Dictionary of loaded models
    scaler : object
        Fitted scaler
    feature_cols : list
        List of feature column names
    horizons : list
        List of forecast horizons
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with predictions for each horizon
    \"\"\"
    # Make a copy of the input dataframe
    results = df.copy()
    
    # Scale features
    X = scaler.transform(df[feature_cols])
    
    # Make predictions for each horizon
    for horizon in horizons:
        lower_model = models[f'{horizon}_lower']
        median_model = models[f'{horizon}_median']
        upper_model = models[f'{horizon}_upper']
        
        results[f'pred_t+{horizon}'] = median_model.predict(X)
        results[f'lower_t+{horizon}'] = lower_model.predict(X)
        results[f'upper_t+{horizon}'] = upper_model.predict(X)
    
    return results

# Example usage
if __name__ == "__main__":
    # Path to model directory (use the directory with your saved models)
    model_dir = "."  # Replace with your model directory path
    
    # Choose whether to load models using joblib or XGBoost format
    use_joblib = False  # Set to True to use joblib format instead
    
    # Load models
    models, scaler, feature_cols, horizons = load_models(model_dir, use_joblib)
    
    # Example: Load new data for prediction
    # new_data = pd.read_csv('new_data.csv')
    
    # Example: Create sample data for testing
    new_data = pd.DataFrame({
        'Barometer - hPa': [1010.5],
        'Temp - °C': [25.3],
        'Hum - %': [75.2],
        'Dew Point - °C': [20.1],
        'Wet Bulb - °C': [22.3],
        'Avg Wind Speed - km/h': [8.5],
        'Rain - mm': [0.0],
        'High Rain Rate - mm/h': [0.0],
        'GHI - W/m^2': [650.5],
        'UV Index': [6.7],
        'Wind Run - km': [15.3],
        'Month of Year': [7],
        'Hour of Day': [12],
        'Solar Zenith Angle': [25.6],
        'GHI_lag (t-1)': [645.2]
    })
    
    # Make predictions
    predictions = predict(new_data, models, scaler, feature_cols, horizons)
    
    # Print results
    for horizon in horizons:
        print(f"Horizon t+{horizon}:")
        print(f"  Prediction: {predictions[f'pred_t+{horizon}'].values[0]:.2f} W/m²")
        print(f"  95% CI: [{predictions[f'lower_t+{horizon}'].values[0]:.2f}, {predictions[f'upper_t+{horizon}'].values[0]:.2f}] W/m²")
        print()
""")
    print(f"Prediction example script created at {prediction_example_path}")

if __name__ == "__main__":
    main()
