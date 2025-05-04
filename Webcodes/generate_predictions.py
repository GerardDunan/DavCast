import os
import sys
import pandas as pd
import numpy as np
import logging
import traceback
import datetime
from pathlib import Path
from davcast_deploy2 import TFTPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("prediction_generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PredictionGenerator")

class PredictionGenerator:
    def __init__(self, model_path, feature_scaler_path, target_scaler_path, data_path):
        """Initialize the prediction generator with model and scaler paths."""
        self.model_path = model_path
        self.feature_scaler_path = feature_scaler_path
        self.target_scaler_path = target_scaler_path
        self.data_path = data_path
        self.current_dir = os.path.dirname(data_path)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PredictionGenerator')
        
        # Initialize predictor
        self.logger.info("=== Starting Prediction Generator ===")
        try:
            self.predictor = TFTPredictor(
                model_path=model_path,
                feature_scaler_path=feature_scaler_path,
                target_scaler_path=target_scaler_path
            )
            self.logger.info("Successfully initialized TFTPredictor")
        except Exception as e:
            self.logger.error(f"Failed to initialize TFTPredictor: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def load_latest_data(self):
        """Load the latest data from the dataset file."""
        try:
            # Look for dataset in current directory
            for file in os.listdir(self.current_dir):
                if file.endswith('.csv') and 'dataset' in file.lower():
                    dataset_path = os.path.join(self.current_dir, file)
                    break
            else:
                raise FileNotFoundError("No dataset file found in current directory")
            
            logger.info(f"Loading data from {dataset_path}")
            
            # Try multiple encodings to handle special characters
            try:
                df = pd.read_csv(dataset_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(dataset_path, encoding='latin-1')
                    logger.info("File loaded using latin-1 encoding")
                except:
                    df = pd.read_csv(dataset_path, encoding='cp1252')
                    logger.info("File loaded using cp1252 encoding")
                
            logger.info(f"Loaded {len(df)} rows from dataset")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_predictions(self, num_hours=4):
        """Generate predictions for the next few hours."""
        try:
            # Load the latest data
            data = self.load_latest_data()
            
            # Get the last row's Date and End Period
            last_date = data.iloc[-1]['Date']  # Format: "04-May-25"
            last_end_period = data.iloc[-1]['End Period']  # Format: "2:00:00"
            
            # Parse the last datetime
            try:
                # Try parsing with the new format first
                last_datetime = pd.to_datetime(f"{last_date} {last_end_period}", format='%d-%b-%y %H:%M:%S')
                logger.info(f"Last dataset datetime: {last_datetime}")
            except Exception as e:
                logger.warning(f"Failed to parse with primary format: {str(e)}")
                try:
                    # Fallback to the old format
                    last_datetime = pd.to_datetime(f"{last_date} {last_end_period}", format='%B %d, %Y %H:%M:%S')
                    logger.info(f"Last dataset datetime: {last_datetime}")
                except Exception as e2:
                    logger.error(f"Failed to create datetime: {str(e2)}")
            
            # Add datetime column if not present
            if 'datetime' not in data.columns:
                logger.info("Creating datetime column from Date and Start Period")
                try:
                    # Try the new format first
                    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Start Period'], 
                                            format='%d-%b-%y %H:%M:%S', errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not parse datetime with primary format: {str(e)}")
                    try:
                        # Try with different format
                        data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Start Period'], errors='coerce')
                    except Exception as e2:
                        logger.error(f"Failed to create datetime: {str(e2)}")
            
            # Set datetime as index if not already
            if 'datetime' in data.columns:
                data.set_index('datetime', inplace=True)
            
            # Generate predictions using TFTPredictor
            logger.info(f"Generating predictions for next {num_hours} hours")
            predictions = self.predictor.predict(data)
            
            # Convert predictions to DataFrame format
            formatted_predictions = []
            for i, (period, values) in enumerate(predictions.items()):
                # Calculate prediction datetime by adding hours to the last datetime
                prediction_datetime = last_datetime + pd.Timedelta(hours=i+1)
                
                row = {
                    'timestamp': prediction_datetime.strftime('%d/%m/%Y %H:%M'),
                    'lower_bound': values['lower'],
                    'median': values['mean'],
                    'upper_bound': values['upper'],
                    'model': 'GHI Average'
                }
                formatted_predictions.append(row)
            
            result_df = pd.DataFrame(formatted_predictions)
            logger.info(f"Successfully generated {len(result_df)} predictions")
            
            # Store the last datetime for use in save_predictions
            self._last_dataset_datetime = last_datetime
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def save_predictions(self, predictions_df):
        """Save predictions to both a timestamped file and latest_predictions.csv."""
        try:
            if not hasattr(self, '_last_dataset_datetime'):
                raise ValueError("No dataset datetime available. Please run generate_predictions first.")
            
            # Use the stored last datetime from the dataset
            last_datetime = self._last_dataset_datetime
            timestamp = last_datetime.strftime("%Y%m%d-%H%M")
            
            # Define the target directory
            target_dir = "/root/davcast1.7/project/DavCastAverage"
            
            # Create the directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Create filenames with the new directory path
            predictions_file = os.path.join(target_dir, f"prediction_{timestamp}.csv")
            latest_file = os.path.join(target_dir, "latest_predictions.csv")
            
            # Save predictions
            predictions_df.to_csv(predictions_file, index=False)
            predictions_df.to_csv(latest_file, index=False)
            
            self.logger.info(f"Predictions saved to {predictions_file}")
            self.logger.info(f"Predictions also saved to {latest_file}")
            
            return predictions_file
            
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            raise ValueError(f"Failed to save predictions: {str(e)}")

    def run(self):
        """Run the full prediction pipeline."""
        try:
            logger.info("Starting prediction generation process")
            predictions_df = self.generate_predictions(num_hours=4)
            output_file = self.save_predictions(predictions_df)
            logger.info("Prediction generation process completed successfully")
            return output_file
        except Exception as e:
            logger.error(f"Prediction generation process failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None

if __name__ == "__main__":
    try:
        # Initialize paths - using absolute paths for clarity
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "tft_model.keras")
        feature_scaler_path = os.path.join(current_dir, "feature_scaler.pkl")
        target_scaler_path = os.path.join(current_dir, "target_scaler.pkl")
        data_path = "/root/weatherlink/dataset.csv"  # Use absolute path
        
        # Ensure output directory exists
        output_dir = "/root/davcast1.7/project/DavCastAverage"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create and run prediction generator
        generator = PredictionGenerator(
            model_path=model_path,
            feature_scaler_path=feature_scaler_path,
            target_scaler_path=target_scaler_path,
            data_path=data_path
        )
        output_file = generator.run()
        
        if output_file:
            print(f"SUCCESS: Predictions generated and saved to {output_file}")
        else:
            print("ERROR: No output file was generated")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Failed to generate predictions: {str(e)}")
        print("ERROR: Failed to generate predictions. Check log for details.")
        sys.exit(1)