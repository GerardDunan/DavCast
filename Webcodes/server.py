from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import sys
import os
import joblib
import logging
import traceback
import json
import subprocess
import hmac
from functools import wraps
from datetime import datetime
import pytz
sys.path.append("project/src")
from database.ghi_history import GHIDatabase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS to allow all origins for development
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize the database
ghi_db = GHIDatabase()

# Add environment variable for update API key
UPDATE_API_KEY = os.getenv('UPDATE_API_KEY', 'change_this_to_a_secure_random_key')

# Function to verify authentication
def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        # Check if Authorization header exists
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logger.warning("Update API called without Authorization header")
            return jsonify({"error": "Authorization required"}), 401
            
        # Header should be in format: Bearer <api_key>
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            logger.warning("Malformed Authorization header")
            return jsonify({"error": "Invalid authorization format"}), 401
            
        provided_key = parts[1]
        
        # Compare in constant time to prevent timing attacks
        if not hmac.compare_digest(provided_key, UPDATE_API_KEY):
            logger.warning("Invalid API key provided")
            return jsonify({"error": "Invalid API key"}), 401
            
        return view_function(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['GET'])
def get_predictions():
    try:
        davcast_dir = "/root/davcast1.7/project/DavCastAverage"
        logger.debug(f"Checking directory: {davcast_dir}")
        
        if not os.path.exists(davcast_dir):
            logger.error(f"Directory not found: {davcast_dir}")
            return jsonify([]), 200  # Return empty array instead of error
            
        # Look for CSV files instead of JSON
        prediction_files = [f for f in os.listdir(davcast_dir) if f.startswith('prediction_') and f.endswith('.csv')]
        logger.debug(f"Found prediction files: {prediction_files}")
        
        if not prediction_files:
            logger.warning("No prediction files found")
            return jsonify([]), 200
        
        # Sort by filename using datetime pattern in the filename prediction_YYYYMMDD-HHMM.csv
        # For better debugging, list all prediction files found
        logger.debug("Available prediction files:")
        for f in prediction_files:
            logger.debug(f"  - {f}")
            
        # Sort by most recent file based on the date pattern in the filename
        def extract_datetime(filename):
            try:
                # Extract date from filename pattern like prediction_20250218-2300.csv
                date_part = filename.replace('prediction_', '').replace('.csv', '')
                # Create a sortable string - higher = newer
                return date_part
            except:
                return ""
        
        sorted_files = sorted(prediction_files, key=extract_datetime, reverse=True)
        logger.debug(f"Sorted prediction files (newest first): {sorted_files}")
        
        if not sorted_files:
            return jsonify([]), 200
            
        latest_file = sorted_files[0]
        logger.debug(f"Using latest prediction file: {latest_file}")
        
        # Read CSV file
        predictions_path = os.path.join(davcast_dir, latest_file)
        df = pd.read_csv(predictions_path)
        
        # Show what's in the file
        logger.debug(f"CSV columns: {df.columns.tolist()}")
        logger.debug(f"Preview of prediction data: {df.head().to_dict('records')}")
        
        # Convert DataFrame to list of dictionaries
        predictions = []
        for _, row in df.iterrows():
            prediction = {
                'timestamp': row['timestamp'] if 'timestamp' in row else str(row.name),
                'lower_bound': float(row['lower_bound']) if 'lower_bound' in row else 0.0,
                'median': float(row['median']) if 'median' in row else 0.0,
                'upper_bound': float(row['upper_bound']) if 'upper_bound' in row else 0.0
            }
            predictions.append(prediction)
        
        logger.debug(f"Returning {len(predictions)} predictions")
        return jsonify(predictions)
        
    except Exception as e:
        logger.error(f"Error in get_predictions: {str(e)}")
        logger.error(traceback.format_exc())  # Log the full traceback
        return jsonify([]), 200  # Return empty array instead of error

@app.route('/add-weather', methods=['POST'])
def add_weather():
    try:
        data = request.json
        logger.debug(f"Received data: {data}")
        
        # Read the existing dataset
        dataset_path = "/root/weatherlink/dataset.csv"
        df = pd.read_csv(dataset_path)
        
        # Create new row with the received data
        new_row = pd.DataFrame([data])
        
        # Append the new row
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save the updated dataset
        df.to_csv(dataset_path, index=False)
        
        # Update the GHI history database with the new data
        ghi_db.update_historical_data(dataset_path)
        logger.debug("Updated GHI history database after adding new weather data")
        
        return jsonify({"message": "Weather data added successfully"})
        
    except Exception as e:
        logger.error(f"Error adding weather data: {str(e)}")
        logger.error(f"Full error details: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/latest-predictions', methods=['GET'])
def get_latest_predictions():
    try:
        # Find most recent prediction file
        prediction_dir = "/root/davcast1.7/project/DavCastAverage"
        prediction_files = sorted(
            [f for f in os.listdir(prediction_dir) if f.startswith("latest_predictions")],
            reverse=True
        )
        
        if not prediction_files:
            return jsonify({'error': 'No predictions found'}), 404
            
        latest_file = os.path.join(prediction_dir, prediction_files[0])
        data = pd.read_csv(latest_file)
        
        return jsonify(data.to_dict(orient='records'))
        
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/weather-data', methods=['GET'])
def get_weather_data():
    try:
        # Read the CSV file
        df = pd.read_csv('/root/weatherlink/dataset.csv')
        
        # Convert the data to a list of dictionaries
        weather_data = []
        for _, row in df.iterrows():
            entry = {
                'Date': row['Date'],
                'Start Period': row['Start Period'],
                'End Period': row['End Period'],
                'Barometer - hPa': float(row['Barometer - hPa']),
                'Temp - °C': float(row['Temp - °C']),
                'Hum - %': float(row['Hum - %']),
                'Dew Point - °C': float(row['Dew Point - °C']),
                'Wet Bulb - °C': float(row['Wet Bulb - °C']),
                'Avg Wind Speed - km/h': float(row['Avg Wind Speed - km/h']),
                'Rain - mm': float(row['Rain - mm']),
                'High Rain Rate - mm/h': float(row['High Rain Rate - mm/h']),
                'GHI - W/m^2': float(row['GHI - W/m^2']),
                'UV Index': float(row['UV Index']),
                'Wind Run - km': float(row['Wind Run - km']),
                'Month of Year': float(row['Month of Year']),
                'Hour of Day': float(row['Hour of Day']),
                'Solar Zenith Angle': float(row['Solar Zenith Angle']),
                'GHI_lag (t-1)': float(row['GHI_lag (t-1)'])
            }
            weather_data.append(entry)
        
        # Sort by date and time to ensure latest data is last
        weather_data.sort(key=lambda x: (x['Date'], x['Start Period']))
        
        return jsonify(weather_data)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/full-dataset', methods=['GET'])
def get_full_dataset():
    try:
        model = request.args.get('model', 'average')
        file_path = "/root/weatherlink/dataset.csv"
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify([]), 200
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Handle the new datetime format
        if 'Date' in df.columns and 'Start Period' in df.columns:
            try:
                # Convert date format from "DD-Mon-YY" to datetime
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start Period'], 
                                              format='%d-%b-%y %H:%M:%S', errors='coerce')
                logger.debug("Successfully parsed dates!")
            except Exception as e:
                logger.error(f"Error parsing datetime: {e}")
                # Try with flexible format as fallback
                try:
                    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start Period'], errors='coerce')
                    logger.debug("Parsed dates with flexible format")
                except Exception as e2:
                    logger.error(f"Failed to create datetime: {str(e2)}")
        
        # Replace NaT values with None
        df['datetime'] = df['datetime'].replace({pd.NaT: None})
        
        return app.response_class(df.to_json(orient='records', date_format='iso'), mimetype='application/json')
        
    except Exception as e:
        logger.error(f"Error in get_full_dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify([]), 200  # Return empty array instead of error

@app.route('/update-historical-data', methods=['POST'])
def update_historical_data():
    try:
        csv_path = "/root/weatherlink/dataset.csv"
        ghi_db.update_historical_data(csv_path)
        return jsonify({"message": "Historical data updated successfully"})
    except Exception as e:
        logger.error(f"Error updating historical data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/historical-ghi/<int:days_ago>', methods=['GET'])
def get_historical_ghi(days_ago):
    try:
        data = ghi_db.get_day_data(days_ago)
        logger.debug(f"Sending historical GHI data: {data}")
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting historical GHI: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/historical-dates', methods=['GET'])
def get_historical_dates():
    try:
        dates = ghi_db.get_all_dates()
        return jsonify(dates)
    except Exception as e:
        logger.error(f"Error getting historical dates: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/trigger-update', methods=['POST'])
@require_api_key
def trigger_update():
    """Trigger a data update from the web application."""
    logger.info("Data update triggered via API")
    
    try:
        # Determine if we should use email-only mode based on request parameter
        email_only = request.json.get('email_only', False)
        
        # Build the command
        cmd = ["python", "data_updater.py"]
        if email_only:
            cmd.append("--email-only")
            
        # Run the updater as a background process
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return jsonify({
            "status": "success", 
            "message": "Update process started",
            "mode": "email_only" if email_only else "full"
        })
        
    except Exception as e:
        logger.error(f"Error triggering update: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/next-hour-forecast', methods=['GET'])
def get_next_hour_forecast():
    try:
        davcast_dir = "/root/davcast1.7/project/DavCastAverage"
        logger.debug(f"Checking directory for next hour forecasts: {davcast_dir}")
        
        # Get all prediction files
        prediction_files = [f for f in os.listdir(davcast_dir) if f.startswith('prediction_') and f.endswith('.csv')]
        logger.debug(f"Found prediction files: {prediction_files}")
        
        if not prediction_files:
            logger.warning("No prediction files found")
            return jsonify([]), 200
            
        # Sort files by datetime in filename
        sorted_files = sorted(prediction_files, key=lambda x: x.replace('prediction_', '').replace('.csv', ''))
        logger.debug(f"Sorted prediction files: {sorted_files}")
        
        next_hour_predictions = []
        
        # Read each prediction file
        for file in sorted_files:
            file_path = os.path.join(davcast_dir, file)
            df = pd.read_csv(file_path)
            
            # Get the first prediction (1-hour ahead forecast)
            if not df.empty:
                first_pred = df.iloc[0]
                prediction = {
                    'timestamp': first_pred['timestamp'] if 'timestamp' in first_pred else None,
                    'lower_bound': float(first_pred['lower_bound']) if 'lower_bound' in first_pred else 0.0,
                    'median': float(first_pred['median']) if 'median' in first_pred else 0.0,
                    'upper_bound': float(first_pred['upper_bound']) if 'upper_bound' in first_pred else 0.0
                }
                next_hour_predictions.append(prediction)
        
        logger.debug(f"Returning {len(next_hour_predictions)} next hour predictions")
        return jsonify(next_hour_predictions)
        
    except Exception as e:
        logger.error(f"Error in get_next_hour_forecast: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify([]), 200

@app.route('/ghi-today', methods=['GET'])
def get_ghi_today():
    try:
        # Get current date in Manila timezone
        manila_tz = pytz.timezone('Asia/Manila')
        manila_now = datetime.now(manila_tz)
        current_date = manila_now.strftime('%d-%b-%y')  # Format to match dataset's date format
        
        # Read the dataset
        dataset_path = "/root/weatherlink/dataset.csv"
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return jsonify({"error": "Dataset not found"}), 404
            
        df = pd.read_csv(dataset_path)
        
        # Filter data for current date
        today_data = df[df['Date'] == current_date]
        
        if today_data.empty:
            logger.warning(f"No data found for date: {current_date}")
            return jsonify({"error": f"No data available for {current_date}"}), 404
            
        # Extract only the requested columns and convert to list of dictionaries
        ghi_data = []
        for _, row in today_data.iterrows():
            entry = {
                'Date': row['Date'],
                'Start Period': row['Start Period'],
                'End Period': row['End Period'],
                'GHI': float(row['GHI - W/m^2'])
            }
            ghi_data.append(entry)
            
        # Sort by Start Period
        ghi_data.sort(key=lambda x: x['Start Period'])
        
        return jsonify(ghi_data)
        
    except Exception as e:
        logger.error(f"Error in get_ghi_today: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize the database with current data
    ghi_db = GHIDatabase()
    ghi_db.update_historical_data("/root/weatherlink/dataset.csv")
    app.run(debug=True, host='0.0.0.0', port=5000) 
