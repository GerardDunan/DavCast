import pandas as pd
import os
import datetime
import re

def convert_date_format(date_str):
    """
    Convert between the two date formats used in the CSV files.
    Handles both "1-May-25" and "May 02, 2025" formats.
    Also normalizes dates like "2-May-25" and "02-May-25" to be treated as the same.
    """
    try:
        # First, normalize the date string to handle both single and double-digit days
        if "-" in date_str and len(date_str.split("-")) == 3:
            day, month, year = date_str.split("-")
            # Normalize day to ensure single-digit days are treated the same as zero-padded days
            day = str(int(day))  # Remove leading zeros if present
            normalized_date_str = f"{day}-{month}-{year}"
            
            # Convert to datetime object
            date_obj = datetime.datetime.strptime(f"{day}-{month}-{year}", "%d-%b-%y")
            
            # Return in the "May 02, 2025" format
            return date_obj.strftime("%b %d, %Y")
        
        # Check if the date is in the "May 02, 2025" format
        elif "," in date_str:
            try:
                # Convert to datetime object - use the correct format string for "May 01, 2025"
                date_obj = datetime.datetime.strptime(date_str, "%b %d, %Y")
                
                # Return in the normalized "d-May-yy" format (no leading zeros)
                day = date_obj.day
                month = date_obj.strftime("%b")
                year = date_obj.strftime("%y")
                return f"{day}-{month}-{year}"
            except ValueError:
                # If the first attempt fails, try with different format variations
                try:
                    # Try parsing with different format in case of variations
                    date_obj = datetime.datetime.strptime(date_str, "%B %d, %Y")
                    day = date_obj.day
                    month = date_obj.strftime("%b")
                    year = date_obj.strftime("%y")
                    return f"{day}-{month}-{year}"
                except ValueError:
                    # Return as is if all parsing attempts fail
                    print(f"Could not parse date '{date_str}' with any known format")
                    return date_str
        else:
            return date_str  # Return as is if format is unknown
    except Exception as e:
        print(f"Error converting date '{date_str}': {e}")
        return date_str

def normalize_time_format(time_str):
    """
    Normalize time formats to ensure "0:00:00" and "00:00:00" are treated as the same.
    Returns the normalized time in 24-hour format with consistent zero-padding.
    """
    try:
        # Remove any leading/trailing whitespace
        time_str = str(time_str).strip()
        
        # Try to parse the time string into a datetime object
        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 3:
                # Ensure all parts are numbers
                hour = int(parts[0])
                minute = int(parts[1])
                second = int(parts[2])
                # Return consistently formatted time
                return f"{hour:02d}:{minute:02d}:{second:02d}"
            elif len(parts) == 2:
                # Handle case with just hours and minutes
                hour = int(parts[0])
                minute = int(parts[1])
                return f"{hour:02d}:{minute:02d}:00"
        
        # If we can't parse it, return as is
        return time_str
    except Exception as e:
        print(f"Error normalizing time '{time_str}': {e}")
        return time_str

def create_merge_key(row):
    """
    Create a standardized merge key from date and time fields.
    Normalizes all components to ensure consistent comparison.
    """
    try:
        # Normalize date
        date = str(row['Date']).strip()
        if "-" in date and len(date.split("-")) == 3:
            day, month, year = date.split("-")
            # Normalize day to remove leading zeros
            day = str(int(day))
            date = f"{day}-{month}-{year}"
        
        # Normalize time periods
        start_time = normalize_time_format(row['Start Period'])
        end_time = normalize_time_format(row['End Period'])
        
        # Return standardized key
        return f"{date}_{start_time}_{end_time}"
    except Exception as e:
        print(f"Error creating merge key: {e}")
        # Default fallback - use original values
        return f"{row['Date']}_{row['Start Period']}_{row['End Period']}"

def merge_csv_files(dataset_path, cleaned_path, output_path=None):
    """
    Merge cleaned.csv into dataset.csv avoiding duplications by checking
    Date, Start Period, and End Period.
    
    Args:
        dataset_path: Path to the original dataset.csv
        cleaned_path: Path to the cleaned.csv with new data
        output_path: Path to save the merged CSV (defaults to overwriting dataset.csv)
    
    Returns:
        Path to the merged file
    """
    print(f"Loading dataset from: {dataset_path}")
    print(f"Loading cleaned data from: {cleaned_path}")
    
    # If no output path is specified, overwrite the original dataset
    if output_path is None:
        output_path = dataset_path
    
    # Read the CSV files with UTF-8-SIG encoding to handle special characters like degree symbol
    try:
        # Try reading with UTF-8-SIG first (which is how cleaner.py writes the files)
        try:
            dataset_df = pd.read_csv(dataset_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            # If UTF-8-SIG fails, try UTF-8, then latin-1 as fallbacks
            try:
                dataset_df = pd.read_csv(dataset_path, encoding='utf-8')
            except UnicodeDecodeError:
                dataset_df = pd.read_csv(dataset_path, encoding='latin-1')
        
        # Read cleaned.csv with UTF-8-SIG since we know cleaner.py writes it that way
        cleaned_df = pd.read_csv(cleaned_path, encoding='utf-8-sig')
        
        original_columns = list(dataset_df.columns)  # Save original columns for later
        print(f"Dataset shape before processing: {dataset_df.shape}")
        print(f"Cleaned data shape: {cleaned_df.shape}")
        
        # First, normalize time formats in both dataframes to handle "0:00:00" vs "00:00:00"
        dataset_df['Start Period'] = dataset_df['Start Period'].apply(normalize_time_format)
        dataset_df['End Period'] = dataset_df['End Period'].apply(normalize_time_format)
        
        # Clean date strings in cleaned_df first (convert to d-Mon-yy format)
        # Handle the quotation marks that might be around date strings
        cleaned_df['Date'] = cleaned_df['Date'].apply(lambda x: str(x).strip('"\''))
        
        # Convert dates in cleaned_df to match the format in dataset_df
        cleaned_df['Date'] = cleaned_df['Date'].apply(convert_date_format)
        
        # Normalize time formats in cleaned_df
        cleaned_df['Start Period'] = cleaned_df['Start Period'].apply(normalize_time_format)
        cleaned_df['End Period'] = cleaned_df['End Period'].apply(normalize_time_format)
        
        # Create standardized merge keys for proper comparison
        dataset_df['Merge_Key'] = dataset_df.apply(create_merge_key, axis=1)
        cleaned_df['Merge_Key'] = cleaned_df.apply(create_merge_key, axis=1)
        
        # Remove duplicates from dataset_df first (using normalized keys)
        duplicate_count = dataset_df.duplicated(subset=['Merge_Key']).sum()
        if duplicate_count > 0:
            print(f"Found {duplicate_count} duplicates in the original dataset. Removing them...")
            dataset_df = dataset_df.drop_duplicates(subset=['Merge_Key'])
            print(f"Dataset shape after removing duplicates: {dataset_df.shape}")
        
        # Identify records in cleaned_df that are not in dataset_df
        existing_keys = set(dataset_df['Merge_Key'])
        new_records = cleaned_df[~cleaned_df['Merge_Key'].isin(existing_keys)]
        
        print(f"Found {len(new_records)} new records to add")
        
        if len(new_records) == 0:
            print("No new records to add. Dataset remains unchanged.")
            # Make sure to remove the Merge_Key column before returning
            if 'Merge_Key' in dataset_df.columns and 'Merge_Key' not in original_columns:
                dataset_df = dataset_df.drop('Merge_Key', axis=1)
            # Save with UTF-8-SIG encoding to preserve special characters
            dataset_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            return output_path
        
        # Drop the temporary merge key before concatenating to avoid it appearing in the output
        if 'Merge_Key' in new_records.columns:
            new_records = new_records.drop('Merge_Key', axis=1)
        if 'Merge_Key' in dataset_df.columns:
            dataset_df = dataset_df.drop('Merge_Key', axis=1)
        
        # Concatenate the dataframes
        merged_df = pd.concat([dataset_df, new_records], ignore_index=True)
        
        # Re-normalize times after merging to ensure consistency
        merged_df['Start Period'] = merged_df['Start Period'].apply(normalize_time_format)
        merged_df['End Period'] = merged_df['End Period'].apply(normalize_time_format)
        
        # Sort by date and time
        try:
            merged_df['SortDate'] = pd.to_datetime(merged_df['Date'], format='%d-%b-%y', errors='coerce')
            merged_df['SortTime'] = pd.to_datetime(merged_df['Start Period'], format='%H:%M:%S', errors='coerce')
            merged_df = merged_df.sort_values(by=['SortDate', 'SortTime'])
            merged_df = merged_df.drop(['SortDate', 'SortTime'], axis=1)
        except Exception as e:
            print(f"Warning: Error sorting data by date and time: {e}")
            print("Proceeding without sorting...")
        
        # Final check for duplicates based on normalized versions of date and time fields
        merged_df['Temp_Key'] = merged_df.apply(create_merge_key, axis=1)
        final_duplicate_count = merged_df.duplicated(subset=['Temp_Key']).sum()
        
        if final_duplicate_count > 0:
            print(f"Found {final_duplicate_count} duplicates in the final merged dataset. Removing them...")
            merged_df = merged_df.drop_duplicates(subset=['Temp_Key'])
        
        # Remove the temporary key
        if 'Temp_Key' in merged_df.columns:
            merged_df = merged_df.drop('Temp_Key', axis=1)
        
        # Ensure we only have the original columns in the final output
        if set(merged_df.columns) != set(original_columns):
            # Keep only original columns and in the same order
            merged_df = merged_df[original_columns]
        
        # Save the merged dataframe with UTF-8-SIG encoding to preserve special characters
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Merged data saved to: {output_path}")
        print(f"Final dataset shape: {merged_df.shape}")
        
        return output_path
        
    except Exception as e:
        print(f"Error merging CSV files: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge cleaned.csv into dataset.csv avoiding duplications')
    parser.add_argument('--dataset', type=str, default='dataset.csv',
                        help='Path to the dataset.csv file (default: dataset.csv)')
    parser.add_argument('--cleaned', type=str, default='cleaned.csv',
                        help='Path to the cleaned.csv file (default: cleaned.csv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the merged CSV (default: overwrite dataset.csv)')
    
    args = parser.parse_args()
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve relative paths
    dataset_path = os.path.join(script_dir, args.dataset)
    cleaned_path = os.path.join(script_dir, args.cleaned)
    output_path = os.path.join(script_dir, args.output) if args.output else None
    
    merge_csv_files(dataset_path, cleaned_path, output_path) 
