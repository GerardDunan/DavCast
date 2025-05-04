import { useState, useEffect } from 'react';
import { WeatherData } from '../types';

interface Props {
  onBack: () => void;
  onDataUpdate: () => Promise<void>;
  onAddWeather: () => void;
}

// Weather icons for the table headers
const WEATHER_ICONS = {
  temperature: '🌡️',
  humidity: '💧',
  dewPoint: '💨',
  wetBulb: '🌊',
  windSpeed: '🌪️',
  windRun: '🍃',
  uvIndex: '☀️',
  ghi: '☀️',
  date: '📅',
  time: '⏱️'
};

const AdminPage = ({ onBack, onDataUpdate, onAddWeather }: Props) => {
  const [averageData, setAverageData] = useState<WeatherData[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch average dataset
        const averageResponse = await fetch('http://146.190.121.70:5000/full-dataset?model=average');
        if (averageResponse.ok) {
          const averageData = await averageResponse.json();
          // Reverse the order to show latest first
          setAverageData([...averageData].reverse());
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  // Format date from DD-MMM-YY to DD/MM/YYYY
  const formatDate = (dateStr: string | number | undefined) => {
    if (dateStr === undefined) return '';
    
    // Convert to string if it's a number
    const dateString = typeof dateStr === 'number' ? dateStr.toString() : dateStr;
    
    // Handle different date formats
    const dateParts = dateString.split(/[-\/]/); // Split by dash or slash
    
    if (dateParts.length < 3) return dateString; // Return original if format not recognized
    
    // Check if the format is DD-MMM-YY (e.g., 07-Mar-24)
    if (dateParts[1].length === 3) {
      const day = dateParts[0];
      const monthMap: {[key: string]: string} = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
      };
      const month = monthMap[dateParts[1]] || '01';
      const year = '20' + dateParts[2]; // Assuming 20xx for simplicity
      
      return `${day}/${month}/${year}`;
    }
    
    // If it's already in MM/DD/YYYY format, rearrange to DD/MM/YYYY
    if (dateParts[0].length <= 2 && dateParts[1].length <= 2) {
      return `${dateParts[1]}/${dateParts[0]}/${dateParts[2]}`;
    }
    
    return dateString;
  };

  // Format time from HH:MM:SS to HH:MM
  const formatTime = (timeStr: string | number | undefined) => {
    if (timeStr === undefined) return '';
    
    // Convert to string if it's a number
    const timeString = typeof timeStr === 'number' ? timeStr.toString() : timeStr;
    
    const timeParts = timeString.split(':');
    if (timeParts.length >= 2) {
      return `${timeParts[0]}:${timeParts[1]}`;
    }
    
    return timeString;
  };

  // Helper function to safely process field values
  const processWeatherValue = (item: WeatherData, field: string) => {
    // First, check for exact field matches based on the console output
    const exactFieldMappings: { [key: string]: string } = {
      'Temp - °C': 'Temp - °C',
      'Dew Point - °C': 'Dew Point - °C',
      'Wet Bulb - °C': 'Wet Bulb - °C',
      'Hum - %': 'Hum - %',
      'Avg Wind Speed - km/h': 'Avg Wind Speed - km/h',
      'Wind Run - km': 'Wind Run - km',
      'UV Index': 'UV Index',
      'GHI - W/m^2': 'GHI - W/m^2'
    };

    // Get the exact field name from the mapping
    const exactField = exactFieldMappings[field];
    
    // Try to get the value using the exact field name
    if (exactField && item[exactField] !== undefined) {
      const value = Number(item[exactField]);
      if (!isNaN(value)) {
        return value.toFixed(1);
      }
    }

    // If exact field not found, try the original field
    if (item[field] !== undefined) {
      const value = Number(item[field]);
      if (!isNaN(value)) {
        return value.toFixed(1);
      }
    }

    // Try alternative field names without special characters
    const plainField = field.replace(/[°]/g, '').replace(/ /g, '');
    for (const key of Object.keys(item)) {
      const plainKey = key.replace(/[°]/g, '').replace(/ /g, '');
      if (plainKey === plainField) {
        const value = Number(item[key]);
        if (!isNaN(value)) {
          return value.toFixed(1);
        }
      }
    }

    // If still no value found, look for specific field patterns from the console output
    const fieldPatterns: { [key: string]: RegExp } = {
      'Temp - °C': /Temp.*C/i,
      'Dew Point - °C': /Dew.*Point.*C/i,
      'Wet Bulb - °C': /Wet.*Bulb.*C/i
    };

    const pattern = fieldPatterns[field];
    if (pattern) {
      for (const key of Object.keys(item)) {
        if (pattern.test(key)) {
          const value = Number(item[key]);
          if (!isNaN(value)) {
            return value.toFixed(1);
          }
        }
      }
    }

    // If no valid value found, return NaN
    return 'NaN';
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-dark-bg to-dark-card p-4">
      <div className="container mx-auto">
        <div className="flex justify-between items-center mb-6">
          <button
            onClick={onBack}
            className="text-white hover:text-primary-yellow"
          >
            ← Back
          </button>
          <h1 className="text-2xl font-bold text-white">Admin Dashboard</h1>
          <button
            onClick={onAddWeather}
            className="bg-primary-yellow text-dark-bg px-4 py-2 rounded hover:bg-opacity-80"
          >
            Add Weather Data
          </button>
        </div>

        {/* Data Table with fixed header and scrollable body */}
        <div className="bg-dark-card rounded-xl p-6 shadow-lg border border-opacity-10 border-white">
          <h2 className="text-xl font-bold text-white mb-4">Weather Data</h2>
          <div className="max-h-[70vh] overflow-auto">
            <table className="w-full text-gray-300">
              <thead className="sticky top-0 bg-primary-yellow text-dark-bg">
                <tr>
                  {/* Adjusted column widths with flex classes */}
                  <th className="px-2 py-2 text-center w-24">{WEATHER_ICONS.date}<br/>Date</th>
                  <th className="px-2 py-2 text-center w-20">{WEATHER_ICONS.time}<br/>Start<br/>Period</th>
                  <th className="px-2 py-2 text-center w-20">{WEATHER_ICONS.time}<br/>End<br/>Period</th>
                  <th className="px-2 py-2 text-center">{WEATHER_ICONS.temperature}<br/>Temperature<br/>(°C)</th>
                  <th className="px-2 py-2 text-center">{WEATHER_ICONS.humidity}<br/>Humidity<br/>(%)</th>
                  <th className="px-2 py-2 text-center">{WEATHER_ICONS.dewPoint}<br/>Dew<br/>Point (°C)</th>
                  <th className="px-2 py-2 text-center">{WEATHER_ICONS.wetBulb}<br/>Wet<br/>Bulb (°C)</th>
                  <th className="px-2 py-2 text-center">{WEATHER_ICONS.windSpeed}<br/>Wind Speed<br/>(km/h)</th>
                  <th className="px-2 py-2 text-center">{WEATHER_ICONS.windRun}<br/>Wind<br/>Run (km)</th>
                  <th className="px-2 py-2 text-center">{WEATHER_ICONS.uvIndex}<br/>UV<br/>Index</th>
                  <th className="px-2 py-2 text-center">{WEATHER_ICONS.ghi}<br/>Average GHI<br/>(W/m²)</th>
                </tr>
              </thead>
              <tbody>
                {averageData.map((item, index) => (
                  <tr key={index} className="border-b border-opacity-10 border-white hover:bg-opacity-10 hover:bg-yellow-500">
                    <td className="px-2 py-2 text-center">{formatDate(item.Date)}</td>
                    <td className="px-2 py-2 text-center">{formatTime(item['Start Period'])}</td>
                    <td className="px-2 py-2 text-center">{formatTime(item['End Period'])}</td>
                    <td className="px-2 py-2 text-center">{processWeatherValue(item, 'Temp - °C')}</td>
                    <td className="px-2 py-2 text-center">{processWeatherValue(item, 'Hum - %')}</td>
                    <td className="px-2 py-2 text-center">{processWeatherValue(item, 'Dew Point - °C')}</td>
                    <td className="px-2 py-2 text-center">{processWeatherValue(item, 'Wet Bulb - °C')}</td>
                    <td className="px-2 py-2 text-center">{processWeatherValue(item, 'Avg Wind Speed - km/h')}</td>
                    <td className="px-2 py-2 text-center">{processWeatherValue(item, 'Wind Run - km')}</td>
                    <td className="px-2 py-2 text-center">{processWeatherValue(item, 'UV Index')}</td>
                    <td className="px-2 py-2 text-center">{processWeatherValue(item, 'GHI - W/m^2')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Additional weather data display */}
        {averageData.length > 0 && (
          <div className="grid grid-cols-2 gap-4 mt-6 pt-6 border-t border-opacity-10 border-white">
            {/* This section is commented out as it's currently not used */}
          </div>
        )}
      </div>
    </div>
  );
};

export default AdminPage;