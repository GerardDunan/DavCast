import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

class GHIDatabase:
    def __init__(self):
        self.db_path = "project/src/database/ghi_history.db"
        self.ensure_db_directory()
        self.init_database()

    def ensure_db_directory(self):
        directory = os.path.dirname(self.db_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ghi_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            time TEXT,
            ghi_value REAL
        )
        ''')
        
        conn.commit()
        conn.close()

    def update_historical_data(self, csv_path):
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                print(f"Trying to read CSV with {encoding} encoding...")
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"Successfully read CSV with {encoding} encoding!")
                break
            except UnicodeDecodeError:
                print(f"Failed to read with {encoding} encoding, trying next...")
                continue
        
        if df is None:
            raise ValueError("Could not read the CSV file with any of the attempted encodings")
        
        # Print column names for debugging
        print("Available columns:", df.columns.tolist())
        
        # Convert Date and Start Period to datetime
        try:
            # First convert the date to datetime object
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start Period'])
            # Format the date string in the required format for database storage
            df['Date'] = df['datetime'].dt.strftime('%d-%b-%y')
            print("Successfully parsed dates!")
        except KeyError as e:
            print("Column error. Available columns are:", df.columns.tolist())
            raise e
        except ValueError as e:
            print("Date parsing error. Sample date value:", df['Date'].iloc[0])
            raise e
        
        # Filter for times between 5 AM and 6 PM (updated to include 5 AM)
        df['hour'] = df['datetime'].dt.hour
        df = df[(df['hour'] >= 5) & (df['hour'] <= 18)]
        
        # Sort by datetime and get the latest date
        df = df.sort_values('datetime', ascending=True)
        latest_date = df['datetime'].iloc[-1].date()
        
        # Exclude the latest date and get previous 6 days
        target_dates = [(latest_date - timedelta(days=x+1)) for x in range(6)]
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM ghi_history')
        
        # Insert data for each day
        for date in target_dates:
            date_str = date.strftime('%d-%b-%y')
            day_data = df[df['Date'] == date_str]
            
            for _, row in day_data.iterrows():
                cursor.execute('''
                INSERT INTO ghi_history (date, time, ghi_value)
                VALUES (?, ?, ?)
                ''', (date_str, row['Start Period'], row['GHI - W/m^2']))
        
        conn.commit()
        conn.close()

    def get_day_data(self, days_ago):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all dates in the database
        cursor.execute('SELECT DISTINCT date FROM ghi_history ORDER BY date DESC')
        dates = cursor.fetchall()
        
        if days_ago >= len(dates):
            return None
        
        target_date = dates[days_ago][0]
        
        # Convert date string to datetime for formatting
        date_obj = datetime.strptime(target_date, '%d-%b-%y')
        display_date = date_obj.strftime('%B %d, %Y')
        
        # Debug print
        print(f"Fetching historical data for date: {target_date}")
        
        # Get data for the target date with times from 6 AM to 6 PM
        # Modified query to handle single-digit hour formats
        cursor.execute('''
        SELECT time, ghi_value 
        FROM ghi_history 
        WHERE date = ? 
        ORDER BY 
            CAST(substr(time, 1, INSTR(time, ':') - 1) AS INTEGER),
            time
        ''', (target_date,))
        
        results = cursor.fetchall()
        
        # Debug print the results
        times = [row[0] for row in results]
        values = [row[1] for row in results]
        print(f"Retrieved {len(times)} time slots from database:")
        for i, (t, v) in enumerate(zip(times, values)):
            hour = int(t.split(':')[0])
            print(f"  {i+1}. Time: {t}, Hour: {hour}, Value: {v}")
        
        # Format times to ensure sorting works correctly
        # Create a complete map of hours from 6 to 18
        hours_map = {}
        for t, v in zip(times, values):
            hour = int(t.split(':')[0])
            hours_map[hour] = v
        
        # Ensure we have all hours from 5 to 18 (updated to include 5 AM)
        formatted_times = []
        formatted_values = []
        for hour in range(5, 19):
            # Use actual value if available, otherwise use 0
            value = hours_map.get(hour, 0)
            # Format time with leading zeros for proper sorting
            formatted_times.append(f"{hour:02d}:00:00")
            formatted_values.append(value)
            
        print(f"After formatting, hours: {formatted_times}")
        print(f"After formatting, values: {formatted_values}")
        
        conn.close()
        
        return {
            'times': formatted_times,
            'values': formatted_values,
            'display_date': display_date
        }

    def get_all_dates(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT date FROM ghi_history ORDER BY date DESC')
        dates = cursor.fetchall()
        conn.close()
        
        # Convert dates to formatted strings
        formatted_dates = []
        for date_tuple in dates:
            date_obj = datetime.strptime(date_tuple[0], '%d-%b-%y')
            formatted_date = date_obj.strftime('%B %d, %Y')
            formatted_dates.append(formatted_date)
        
        return formatted_dates 