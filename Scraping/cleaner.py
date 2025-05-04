import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import io
import os
import sys

"""
WeatherLink CSV Cleaner

This script processes WeatherLink CSV data files for training datasets.
It calculates hourly averages and computes additional parameters needed for the model.

Usage:
    python cleaner.py --input file1.csv [file2.csv]

    Where:
        --input: One or two CSV files to process. If two files are provided,
                they will be combined after processing.

Output:
    The script will save a cleaned CSV file named 'cleaned.csv' in the same 
    directory as the script.

Example:
    python cleaner.py --input data.csv
    python cleaner.py --input january_data.csv february_data.csv
"""

# Define constants and location coordinates
LATITUDE = 7.0707
LONGITUDE = 125.6113
ELEVATION = 7  # meters
SOLAR_CONSTANT = 1361  # Updated solar constant in W/m²

def find_header_row(file_path):
    """Find the row containing column headers"""
    encoding = 'utf-8'
    try:
        # Try reading with UTF-8
        with open(file_path, 'r', encoding=encoding) as f:
            csv_content = f.readlines()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with latin-1
        encoding = 'latin-1'
        with open(file_path, 'r', encoding=encoding) as f:
            csv_content = f.readlines()
    
    # Find the row with column headers
    column_row_index = 0
    target_columns = ['Date & Time', 'Solar Rad - W/m^2']
    
    for i, line in enumerate(csv_content):
        # Check if this line contains any of the expected column names
        if any(col in line for col in target_columns):
            column_row_index = i
            break
    
    return column_row_index, encoding

def load_dataset(file_path):
    """Load dataset with header row detection"""
    column_row_index, encoding = find_header_row(file_path)
    
    try:
        df = pd.read_csv(file_path, skiprows=column_row_index, encoding=encoding)
        # Rename Solar Rad column to GHI
        if 'Solar Rad - W/m^2' in df.columns:
            df = df.rename(columns={'Solar Rad - W/m^2': 'GHI - W/m^2'})
        
        # Convert Date & Time to datetime
        df['Date & Time'] = pd.to_datetime(df['Date & Time'], dayfirst=True)
        
        return df, column_row_index
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, column_row_index

def filter_training_date_range(df):
    """Prepare data for training without date restrictions"""
    # No date filtering - use the entire dataset
    print(f"Training data starting from {df['Date & Time'].min()}")
    print(f"Total rows in dataset: {len(df)}")
    
    return df

def calculate_hourly_averages(df):
    """Calculate hourly averages for all columns except Date & Time and Wind Run"""
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Make sure we have the hour_group column
    if 'hour_group' not in df_copy.columns:
        df_copy['hour_group'] = df_copy['Date & Time'].dt.floor('H')
    
    # Columns to average (all except Date & Time and Wind Run and date)
    avg_columns = [col for col in df_copy.columns if col not in ['Date & Time', 'Wind Run - km', 'hour_group', 'date']]
    
    # Calculate averages and wind run differences
    results = []
    
    # Get all unique hour_groups
    hour_groups = sorted(df_copy['hour_group'].unique())
    
    # Find the last complete hour
    last_complete_hour = None
    for i in range(len(hour_groups) - 1):
        current_hour = hour_groups[i]
        next_hour = hour_groups[i + 1]
        
        # If the next hour is exactly 1 hour after the current one, the current hour is complete
        if next_hour == current_hour + timedelta(hours=1):
            last_complete_hour = current_hour
    
    # Filter out data after the last complete hour
    if last_complete_hour is not None:
        # Include the full interval of the last complete hour
        max_time = last_complete_hour + timedelta(hours=1)
        
        # Get the end time as a string for clearer reporting
        end_hour_str = (last_complete_hour + timedelta(hours=1)).strftime('%H:%M')
        
        print(f"Last complete hourly interval: {last_complete_hour.strftime('%Y-%m-%d %H:%M')} to {end_hour_str}")
        print(f"Including all data up to: {max_time}")
        
        df_copy = df_copy[df_copy['Date & Time'] <= max_time]
    
    for start_hour, group in df_copy.groupby('hour_group'):
        # Skip if this hour is after the last complete hour
        if last_complete_hour is not None and start_hour > last_complete_hour:
            continue
            
        end_hour = start_hour + timedelta(hours=1)
        
        # Filter data for the current hour (inclusive of both start and end)
        hour_data = df_copy[
            (df_copy['Date & Time'] >= start_hour) & 
            (df_copy['Date & Time'] <= end_hour)  # Include the next hour's 0:00 data point
        ]
        
        # Check if we have the next hour's data point (to ensure the hour is complete)
        next_hour_exists = any(df_copy['hour_group'] == end_hour)
        
        # Only process complete hours (those that have data for the next hour too)
        if len(hour_data) > 0 and next_hour_exists:
            row = {}
            
            # Get date and time components
            row['Date'] = start_hour.date()
            row['Start Period'] = start_hour.time()
            row['End Period'] = end_hour.time()
            
            # Calculate averages for other columns
            for col in avg_columns:
                row[col] = hour_data[col].mean()
            
            # Calculate wind run as difference between end and start of period
            try:
                end_value = hour_data['Wind Run - km'].iloc[-1]
                start_value = hour_data['Wind Run - km'].iloc[0]
                row['Wind Run - km'] = end_value - start_value
            except (IndexError, KeyError):
                row['Wind Run - km'] = None
                
            results.append(row)
    
    # Create new dataframe with processed data
    result_df = pd.DataFrame(results)
    return result_df

def compute_additional_parameters(data):
    """Compute additional parameters for the dataset"""
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Create new column with explicit date format
    data['Date_Explicit'] = data['Date'].dt.strftime('%B %d, %Y')
    
    # Compute date-based parameters (removing Day of Year)
    data['Month of Year'] = data['Date'].apply(compute_month_of_year)
    
    # Compute Hour of Day using End Period column
    data['End Period'] = pd.to_datetime(data['End Period'], format='%H:%M:%S').dt.time
    data['Hour of Day'] = data['End Period'].apply(lambda x: x.hour if x.hour != 0 else 24)
    
    # Compute solar zenith angle - using the date to calculate day of year just for this calculation
    data['Solar Zenith Angle'] = data.apply(
        lambda row: compute_solar_zenith_angle(compute_day_of_year(row['Date']), row['Hour of Day']),
        axis=1
    )
    
    # Add GHI lags and day period columns
    data = compute_ghi_lags(data)
    data = add_day_period_columns(data)
    
    # Delete original Date column and rename Date_Explicit to Date
    data = data.drop('Date', axis=1)
    data = data.rename(columns={'Date_Explicit': 'Date'})
    
    # Get all columns except 'Date'
    other_columns = [col for col in data.columns if col != 'Date']
    
    # Reorder columns with 'Date' first
    data = data[['Date'] + other_columns]
    
    return data

def compute_day_of_year(date_time):
    """Compute day of year for a given date - used only for solar calculations"""
    return date_time.timetuple().tm_yday

def compute_hour_of_day(date_time):
    """
    Compute hour of day based on End Period.
    For 0:00, returns 24. For all other times, returns hour + minute/60
    """
    if date_time.hour == 0 and date_time.minute == 0:
        return 24
    return date_time.hour + date_time.minute / 60

def compute_declination(day_of_year):
    """Compute solar declination angle"""
    return np.radians(23.45 * np.sin(np.radians(360 / 365 * (day_of_year + 284))))

def compute_solar_zenith_angle(day_of_year, hour_of_day):
    """Compute solar zenith angle"""
    declination = compute_declination(day_of_year)
    hour_angle = np.radians(15 * (hour_of_day - 12))  # Hour angle in radians
    latitude_rad = np.radians(LATITUDE)

    # Compute solar elevation angle
    solar_elevation_angle = np.arcsin(
        np.sin(latitude_rad) * np.sin(declination) +
        np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )

    # Compute solar zenith angle
    solar_zenith_angle = np.pi / 2 - solar_elevation_angle

    return np.degrees(solar_zenith_angle)  # Solar zenith angle (°)

def compute_ghi_lags(data):
    """Compute GHI_lag (t-1)."""
    data['GHI_lag (t-1)'] = data['GHI - W/m^2']
    return data

def add_day_period_columns(data):
    """Add Daytime column based on Hour of Day."""
    data['Daytime'] = np.where((data['Hour of Day'] >= 6) & (data['Hour of Day'] <= 18), 1, 0)
    return data

def compute_month_of_year(date_time):
    """
    Compute month of year from timestamp with quarter-month precision.
    Returns: month as float (e.g., Jan = 1.0, 1.25, 1.5, 1.75; Feb = 2.0, 2.25, 2.5, 2.75)
    """
    month = date_time.month
    day = date_time.day
    
    # Determine which quarter of the month
    if day <= 7:
        quarter = 0.0
    elif day <= 14:
        quarter = 0.25
    elif day <= 21:
        quarter = 0.5
    else:
        quarter = 0.75
        
    return month + quarter

def determine_season(month):
    """
    Determine the season based on the month number.
    Args:
        month (float): Month number with quarter precision (e.g., 1.0, 1.25, etc.)
    Returns:
        int: Season category (1: Cool Dry, 2: Hot Dry, 3: Rainy)
    """
    base_month = int(month)  # Get the base month number without the quarter
    
    if base_month in [12, 1, 2]:
        return 1  # Cool Dry
    elif base_month in [3, 4, 5]:
        return 2  # Hot Dry
    else:  # months 6-11
        return 3  # Rainy

def fix_column_names(df):
    """
    Fix encoding issues with column names, particularly the degree symbol (°)
    which might appear as 'Â°' in some encodings.
    """
    renamed_columns = {}
    for col in df.columns:
        new_col = col
        
        # Fix any incorrectly encoded degree symbols
        if 'Â°' in col:
            new_col = col.replace('Â°', '°')
            renamed_columns[col] = new_col
        elif 'Â' in col and '°' in col:
            new_col = col.replace('Â', '')
            renamed_columns[col] = new_col
        
        # Check for standard temperature columns that might be missing the degree symbol
        elif any(col.startswith(prefix) for prefix in ['Temp - ', 'Wet Bulb - ', 'Dew Point - ']):
            if ' C' in col and '°C' not in col:
                # Replace " C" with " °C"
                new_col = col.replace(' C', ' °C')
                renamed_columns[col] = new_col
            elif 'C' in col and '°' not in col:
                # Find where the 'C' is and insert the degree symbol before it
                c_index = col.find('C')
                if c_index > 0:
                    new_col = col[:c_index] + '°' + col[c_index:]
                    renamed_columns[col] = new_col
    
    # Apply the renaming if any changes were needed
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
        print(f"Fixed encoding in {len(renamed_columns)} column names:")
        for old, new in renamed_columns.items():
            print(f"  - '{old}' → '{new}'")
    
    return df

def check_complete_days(df):
    """
    Check which days have complete data (all hours from 0:00 to 23:00)
    Returns a list of dates that have all 24 hour groups
    """
    # Get the date part from the timestamp
    df['date'] = df['hour_group'].dt.date
    
    # Count how many unique hours each date has
    date_hour_counts = {}
    for date, group in df.groupby('date'):
        hours = group['hour_group'].dt.hour.unique()
        date_hour_counts[date] = len(hours)
    
    # Find complete days (those with all 24 hours)
    complete_days = [date for date, count in date_hour_counts.items() if count == 24]
    
    # Return the list of complete days
    return complete_days

def main():
    """Main function to process datasets"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process WeatherLink CSV data files and generate a cleaned dataset for training.',
        epilog='Example: python cleaner.py --input data.csv'
    )
    parser.add_argument(
        '--input', 
        required=True, 
        nargs='+', 
        help='Path to one or two input CSV files. If two files are provided, they will be combined.'
    )
    args = parser.parse_args()
    
    # Validate input files
    files = args.input
    if not files:
        print("No input files provided. Exiting...")
        sys.exit(1)
    
    # Check if files exist
    for file_path in files:
        if not os.path.isfile(file_path):
            print(f"Error: File {file_path} does not exist.")
            sys.exit(1)
    
    # Process each file
    dataframes = []
    for file_path in files:
        print(f"Processing file: {os.path.basename(file_path)}")
        df, header_rows = load_dataset(file_path)
        
        if df is not None:
            if header_rows > 0:
                print(f"Removed {header_rows} header row(s) from {os.path.basename(file_path)}")
            
            # Filter data for training
            df = filter_training_date_range(df)
                
            if df is None:
                continue
            
            dataframes.append(df)
    
    # If no valid dataframes, exit
    if not dataframes:
        print("No valid data to process. Exiting...")
        return
    
    # Combine datasets if there are two
    if len(dataframes) == 2:
        print("Combining two datasets and removing duplicates...")
        # Concatenate and sort by date/time
        combined_df = pd.concat(dataframes).sort_values('Date & Time')
        # Remove duplicates based on Date & Time
        combined_df = combined_df.drop_duplicates(subset=['Date & Time'])
        df = combined_df
    else:
        df = dataframes[0]
    
    # Calculate hourly averages
    print("Calculating hourly averages...")
    hourly_df = calculate_hourly_averages(df)
    
    # Compute additional parameters
    print("Computing additional parameters...")
    final_df = compute_additional_parameters(hourly_df)
    
    # Print column names before fixing
    print("\nColumn names before fixing:")
    for col in final_df.columns:
        if any(temp in col for temp in ['Temp -', 'Wet Bulb -', 'Dew Point -']):
            print(f"  - '{col}'")
    
    # Fix any encoding issues in column names
    final_df = fix_column_names(final_df)
    
    # Print column names after fixing
    print("\nColumn names after fixing:")
    for col in final_df.columns:
        if any(temp in col for temp in ['Temp -', 'Wet Bulb -', 'Dew Point -']):
            print(f"  - '{col}'")
    
    # Set output filename to cleaned.csv in the same directory as the script
    output_filename = "cleaned.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)
    
    try:
        # Try to save with explicit UTF-8-SIG encoding to preserve special characters
        # UTF-8-SIG (with BOM) helps preserve special characters like the degree symbol
        final_df.to_csv(output_path, index=False, mode='w', encoding='utf-8-sig')
        print(f"Final dataset saved to: {output_path}")
    except (PermissionError, OSError) as e:
        print(f"ERROR: Could not save file to {output_path}. Details: {str(e)}")
        print("Please make sure you have write permissions to the directory.")

if __name__ == "__main__":
    main() 
