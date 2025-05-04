import { useState, useEffect } from 'react';
import { format } from 'date-fns';
import Papa from 'papaparse';
import CalendarView from './components/CalendarView';
import DayDetailsView from './components/DayDetailsView';
import TimeSlotTable from './components/TimeSlotTable';
import AdminPage from './components/AdminPage';
import LoginPage from './components/LoginPage';
import AddWeatherPage from './components/AddWeatherPage';
import SolarPowerGeneration from './components/SolarPowerGeneration';
import PowerDispatchOptimization from './components/PowerDispatchOptimization';
import ApplianceCalculator from './components/ApplianceCalculator';
import { WeatherData, User, PredictionData } from './types';
import axios from 'axios';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
} from 'recharts';

// Update the View type to be a string literal union
type View = 'main' | 'calendar' | 'details' | 'login' | 'admin' | 'add-weather' | 'solar-power' | 'power-dispatch' | 'appliance-calculator';

const WEATHER_ICONS = {
  temperature: '🌡️',
  humidity: '💧',
  dewPoint: '💨',
  wetBulb: '🌊',
  windSpeed: '🌪️',
  uvIndex: '☀️'
};

// Update the HistoricalGHI interface
interface HistoricalGHI {
  times: string[];
  values: number[];
  display_date: string;
}

// Add WeatherParam interface before the App function
interface WeatherParam {
  key: string;
  icon: string;
  label: string;
  tooltip: string;
  field: string;
  fallbackField?: string;
  defaultValue?: number;
  unit: string;
}

// Function to prepare chart data with debugging
const prepareChartData = (weatherData: WeatherData[], predictions: PredictionData[]) => {
  if (!weatherData.length) return [];
  
  // Debug the date format in actual data
  console.log("Weather data first few rows:", weatherData.slice(0, 3));
  console.log("Weather data last few rows:", weatherData.slice(-3));
  
  // Get the target date (current date) - find it from the latest entry
  const latestEntry = weatherData[weatherData.length - 1];
  const targetDate = latestEntry.Date;
  console.log("Target date from latest entry:", targetDate);
  
  // Find all unique dates in the dataset for debugging
  const uniqueDates = [...new Set(weatherData.map(d => d.Date))];
  console.log("All unique dates in dataset:", uniqueDates);
  
  // Check column names to see what's available
  const sampleEntry = weatherData[0];
  console.log("Sample entry column names:", Object.keys(sampleEntry));
  
  // Determine the correct start period column name
  const startPeriodColumn = Object.keys(sampleEntry).find(key => 
    key === 'Start Period' || key === 'Start_period' || key.includes('Start')
  );
  console.log("Using start period column:", startPeriodColumn);
  
  // Filter weather data for the target date
  const actualDayData = weatherData.filter(entry => {
    // Parse the date string from entry.Date (format: "Month DD, YYYY")
    const entryDate = new Date(entry.Date);
    const targetDateObj = new Date(targetDate);
    const hour = parseInt(entry['Start Period'].split(':')[0]);
    
    return (
      entryDate.getFullYear() === targetDateObj.getFullYear() &&
      entryDate.getMonth() === targetDateObj.getMonth() &&
      entryDate.getDate() === targetDateObj.getDate() &&
      hour >= 5 && hour <= 18 // Only include hours between 5 AM and 6 PM
    );
  });
  
  console.log("Actual day data for GHI values:", actualDayData);
  
  // Process predictions and only include daytime hours (5 AM to 6 PM)
  // Log the raw predictions to debug
  console.log("Raw predictions:", predictions);
  
  // Create a lookup map for predictions
  const predictionMap: Record<number, PredictionData> = {};
  
  // Convert timestamp strings to Date objects for easier filtering
  predictions.forEach(p => {
    if (!p.timestamp) {
      console.log("Prediction missing timestamp:", p);
      return;
    }
    
    // Parse the timestamp in DD/MM/YYYY HH:mm format (this is the end period)
    const [datePart, timePart] = p.timestamp.split(' ');
    const [day, month, year] = datePart.split('/');
    const [hour, minute] = timePart.split(':');
    
    // Since timestamp is end period, subtract 1 hour to get start period
    const predTime = new Date(parseInt(year), parseInt(month) - 1, parseInt(day), parseInt(hour), parseInt(minute));
    const startHour = (parseInt(hour) - 1 + 24) % 24; // Subtract 1 hour and handle midnight case
    
    // Only include predictions for hours between 5 AM and 6 PM
    if (startHour >= 5 && startHour <= 18) {
      predictionMap[startHour] = {
        ...p,
        timestamp: `${datePart} ${startHour.toString().padStart(2, '0')}:${minute}` // Update timestamp to start period
      };
      console.log("Added prediction for hour:", startHour, "with values:", {
        lower: p.lower_bound,
        median: p.median,
        upper: p.upper_bound
      });
    }
  });
  
  // Build hourly data for daytime hours (5 AM to 6 PM)
  const hourData = Array.from({ length: 13 }, (_, i) => i + 5).map(hour => {
    const timeLabel = `${hour % 12 || 12} ${hour < 12 ? 'AM' : 'PM'} - ${(hour + 1) % 12 || 12} ${(hour + 1) < 12 ? 'AM' : 'PM'}`;

    const baseData = {
      hour,
      time: new Date(new Date().setHours(hour, 0, 0, 0)).getTime(),
      timeLabel
    };

    // Add actual data from target date if available for this hour
    const actualData = actualDayData.find(d => parseInt(d['Start Period'].split(':')[0]) === hour);
    
    // Get prediction for this hour from our map
    const prediction = predictionMap[hour];

    // Initialize values
    let actualValue = undefined;
    let forecastRange = undefined;
    let medianValue = undefined;
    let isActualDataPoint = false;
    let connectionLine = undefined;
    
    // Set actual value if we have actual data
    if (actualData) {
      actualValue = parseFloat(String(actualData['GHI - W/m^2'])) || 0;
      isActualDataPoint = true;
      console.log(`Setting actual value for hour ${hour}: ${actualValue}`);
    }
    
    // Set forecast values if we have prediction data
    if (prediction) {
      forecastRange = [prediction.lower_bound, prediction.upper_bound];
      medianValue = prediction.median;
      console.log(`Setting forecast values for hour ${hour}: median=${medianValue}, range=[${forecastRange}]`);
    }
    
    return {
      ...baseData,
      actual: actualValue,
      forecastRange: forecastRange,
      median: medianValue,
      isActualDataPoint,
      connectionLine: undefined // Will be set in post-processing
    };
  });

  // Post-process the data to create connection line
  let lastActualDataPoint = null;
  let firstForecastHourIndex = -1;
  
  // Find the last actual data point and the first forecast hour index
  for (let i = 0; i < hourData.length; i++) {
    if (hourData[i].actual !== undefined) {
      lastActualDataPoint = {
        hour: hourData[i].hour,
        value: hourData[i].actual,
        index: i
      };
    }
    
    // If this is the first hour with a forecast but no actual data, mark it
    if (hourData[i].median !== undefined && hourData[i].actual === undefined && firstForecastHourIndex === -1) {
      firstForecastHourIndex = i;
    }
  }
  
  // If we found a transition point, create the connection line
  if (lastActualDataPoint && firstForecastHourIndex !== -1) {
    console.log(`Creating connection line between actual data at hour ${lastActualDataPoint.hour} and forecast at hour ${hourData[firstForecastHourIndex].hour}`);
    
    // Set connection line values for the transition points
    if (hourData[lastActualDataPoint.index]) {
      hourData[lastActualDataPoint.index].connectionLine = lastActualDataPoint.value;
    }
    
    if (hourData[firstForecastHourIndex] && hourData[firstForecastHourIndex].median !== undefined) {
      hourData[firstForecastHourIndex].connectionLine = hourData[firstForecastHourIndex].median;
    }
  }

  console.log("Final chart data:", hourData);
  return hourData;
};

function App() {
  // Update the state declaration
  const [view, setView] = useState<View>('main');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [selectedTime, setSelectedTime] = useState('Now');
  const [weatherData, setWeatherData] = useState<WeatherData[]>([]);
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [nextHourPredictions, setNextHourPredictions] = useState<PredictionData[]>([]);
  const [powerRating, setPowerRating] = useState<string>('');
  const [panelCount, setPanelCount] = useState<string>('');
  const [performanceRatio, setPerformanceRatio] = useState<string>('0.75');
  const [currentSolarGeneration, setCurrentSolarGeneration] = useState<number>(0);
  const [generationData, setGenerationData] = useState<number[]>([]);
  const [slideDirection, setSlideDirection] = useState<'left' | 'right' | null>(null);
  const [selectedDay, setSelectedDay] = useState<number>(0);
  const [historicalGHI, setHistoricalGHI] = useState<HistoricalGHI | null>(null);
  const [historicalDates, setHistoricalDates] = useState<string[]>([]);
  const [scrollPosition, setScrollPosition] = useState(0);

  // Generate calendar days data
  const generateDays = () => {
    const today = new Date();
    return Array.from({ length: 7 }, (_, i) => {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      return {
        id: format(date, 'yyyy-MM-dd'),
        day: format(date, 'EEEE'),
        date: format(date, 'dd MMM'),
        active: i === 0
      };
    });
  };

  const days = generateDays();

  // Fetch predictions based on selected model
  useEffect(() => {
    const fetchData = async () => {
      try {
        // First fetch predictions
        const predictionsResponse = await fetch('http://146.190.121.70:5000/predict');
        if (!predictionsResponse.ok) {
          throw new Error('Failed to fetch predictions');
        }
        const predictionsData = await predictionsResponse.json();
        console.log('Predictions data:', predictionsData);
        
        if (Array.isArray(predictionsData) && predictionsData.length > 0) {
          setPredictions(predictionsData);
        } else {
          console.warn('Empty or invalid predictions received');
        }

        // Fetch next hour predictions
        const nextHourResponse = await fetch('http://146.190.121.70:5000/next-hour');
        if (!nextHourResponse.ok) {
          throw new Error('Failed to fetch next hour predictions');
        }
        const nextHourData = await nextHourResponse.json();
        console.log('Next hour predictions:', nextHourData);
        
        if (Array.isArray(nextHourData) && nextHourData.length > 0) {
          setNextHourPredictions(nextHourData);
        } else {
          console.warn('Empty or invalid next hour predictions received');
        }

        // Then fetch weather data
        const weatherResponse = await fetch('http://146.190.121.70:5000/weather-data');
        if (!weatherResponse.ok) {
          throw new Error('Failed to fetch weather data');
        }
        const weatherData = await weatherResponse.json();
        if (weatherData && Array.isArray(weatherData) && weatherData.length > 0) {
          console.log('Weather data:', weatherData);
          setWeatherData(weatherData);
        } else {
          console.warn('Empty or invalid weather data received');
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []); // Empty dependency array means this runs once on mount

  // Get current prediction
  const currentPrediction = predictions.length > 0 ? predictions[0] : null;

  const handleLogin = (credentials: User) => {
    if (credentials.username === 'davcast' && credentials.password === 'sea_aws') {
      setView('admin');
    } else {
      alert('Invalid credentials');
    }
  };

  const handleTimeSelect = (time: string) => {
    setSelectedTime(time);
    setView('details');
  };

  const refreshData = async () => {
    try {
      // First fetch predictions
      let predictionsData = [];
      const predictionsResponse = await fetch('http://146.190.121.70:5000/predict');
      if (predictionsResponse.ok) {
        predictionsData = await predictionsResponse.json();
        console.log('New predictions data:', predictionsData);
        
        if (Array.isArray(predictionsData) && predictionsData.length > 0) {
          setPredictions(predictionsData);
        } else {
          console.warn('Empty or invalid predictions received');
        }
      } else {
        console.error('Failed to fetch predictions:', await predictionsResponse.text());
      }

      // Fetch the weather data
      const weatherResponse = await fetch('http://146.190.121.70:5000/weather-data');
      if (weatherResponse.ok) {
        const weatherData = await weatherResponse.json();
        // If we have a successful response, use it
        if (weatherData && Array.isArray(weatherData) && weatherData.length > 0) {
          console.log('Latest weather data:', weatherData);
          setWeatherData(weatherData);
        } else {
          console.warn('Empty or invalid weather data received');
        }
      } else {
        console.error('Failed to fetch weather data:', await weatherResponse.text());
      }

    } catch (error) {
      console.error('Error refreshing data:', error);
    }
  };

  // Add this useEffect to fetch data when component mounts
  useEffect(() => {
    fetchDataWithInterval();
  }, []);

  // Function to fetch data with error handling
  const fetchDataWithInterval = () => {
    refreshData().catch(err => {
      console.error("Error refreshing data:", err);
    });
  };

  const handleNavigation = (newView: View) => {
    setView(newView);
  };

  // Add scroll event listener
  useEffect(() => {
    const handleScroll = () => {
      setScrollPosition(window.scrollY);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Update the TopBar component with scroll opacity and hover effects
  const TopBar = ({ setView, isSidebarOpen, setIsSidebarOpen }: { 
    setView: (view: View) => void;
    isSidebarOpen: boolean;
    setIsSidebarOpen: (open: boolean) => void;
  }) => {
    // Opacity calculation for background
    const opacity = Math.max(0.6, 0.9 - (scrollPosition / 500));
    
    return (
      <div 
        className="p-4 fixed top-0 left-0 right-0 z-10 transition-all duration-300"
        style={{ 
          backgroundColor: `rgba(26, 26, 26, ${opacity})`,
          backdropFilter: `blur(${opacity * 8}px)`,
          borderBottom: '1px solid rgba(255, 184, 0, 0.1)'
        }}
      >
        <div className="container mx-auto flex justify-between items-center">
          {/* Menu button on left with hover effect */}
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
            className="text-white p-2 rounded-full transition-all duration-200 hover:text-primary-yellow"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          
          {/* DavCast logo on right with hover effect */}
          <button 
            onClick={() => setView('main')} 
            className="text-2xl font-bold text-primary-yellow rounded-lg px-3 py-1 transition-all duration-200 hover:bg-black hover:bg-opacity-30"
          >
            DavCast
          </button>
        </div>
      </div>
    );
  };

  // Update the SideBar component to match dark theme
  const SideBar = ({ setView, setIsSidebarOpen }: {
    setView: (view: View) => void;
    setIsSidebarOpen: (open: boolean) => void;
  }) => (
    <div className="fixed inset-0 bg-black bg-opacity-80 z-20">
      <div className="absolute left-0 top-0 bottom-0 w-64 bg-dark-card p-4 border-r border-opacity-10 border-primary-yellow">
        <button 
          onClick={() => setIsSidebarOpen(false)}
          className="absolute top-4 right-4 text-gray-400 hover:text-primary-yellow transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <nav className="mt-8">
          <ul className="space-y-4">
            {[
              { title: 'Home', view: 'main' as const },
              { title: 'Solar Power Generation', view: 'solar-power' as const },
              { title: 'Admin Login', view: 'login' as const },
            ].map(({ title, view: targetView }) => (
              <li key={title}>
                <button 
                  onClick={() => { setView(targetView); setIsSidebarOpen(false); }}
                  className={`w-full text-left px-4 py-2 text-white rounded transition-colors duration-200 
                      ${targetView === 'solar-power' 
                      ? 'hover:bg-primary-yellow hover:text-dark-bg' 
                      : 'hover:text-primary-yellow'}`}
                >
                  {title}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </div>
  );

  // Update the slide buttons to match dark theme
  const SlideButton = ({ onClick }: { onClick: () => void }) => (
    <button
      onClick={onClick}
      className="fixed right-4 top-1/2 transform -translate-y-1/2 bg-dark-card bg-opacity-80 rounded-full p-3 hover:border-primary-yellow hover:border transition-all duration-300 z-20"
    >
      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white hover:text-primary-yellow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </button>
  );

  // Update the left slide button
  const SlideButtonLeft = ({ onClick }: { onClick: () => void }) => (
    <button
      onClick={onClick}
      className="fixed left-4 top-1/2 transform -translate-y-1/2 bg-dark-card bg-opacity-80 rounded-full p-3 hover:border-primary-yellow hover:border transition-all duration-300 z-20"
    >
      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white hover:text-primary-yellow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
      </svg>
    </button>
  );

  // Update the right slide button
  const SlideButtonRight = ({ onClick }: { onClick: () => void }) => (
    <button
      onClick={onClick}
      className="fixed right-4 top-1/2 transform -translate-y-1/2 bg-dark-card bg-opacity-80 rounded-full p-3 hover:border-primary-yellow hover:border transition-all duration-300 z-20"
    >
      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white hover:text-primary-yellow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </button>
  );

  // Add validation function
  const isConfigValid = (powerRating: string, panelCount: string, performanceRatio: string) => {
    return (
      powerRating && 
      panelCount && 
      performanceRatio && 
      parseFloat(powerRating) > 0 && 
      parseFloat(panelCount) > 0 && 
      parseFloat(performanceRatio) > 0
    );
  };

  const getDayDate = (daysAgo: number) => {
    const date = new Date();
    date.setDate(date.getDate() - daysAgo);
    return date.toLocaleDateString('en-US', {
      month: 'numeric',
      day: 'numeric',
      year: 'numeric'
    });
  };

  // Update the ScrollButton styling and container styles
  const ScrollButton = ({ direction, onClick }: { direction: 'left' | 'right'; onClick: () => void }) => (
    <button
      onClick={onClick}
      className={`absolute ${direction === 'left' ? 'left-2' : 'right-2'} top-1/2 transform -translate-y-1/2 
        bg-dark-card hover:border hover:border-primary-yellow rounded-full p-3 shadow-lg z-10 text-white hover:text-primary-yellow transition-all duration-200`}
    >
      {direction === 'left' ? '◀' : '▶'}
    </button>
  );

  // Add this function to format time consistently
  const formatTimeRange = (timestamp: string) => {
    try {
      if (!timestamp) return '';
      
      // Handle different date formats
      let date;
      if (timestamp.includes('/')) {
        // Format: "DD/MM/YYYY HH:mm"
        const [datePart, timePart] = timestamp.split(' ');
        const [day, month, year] = datePart.split('/');
        const [hour, minute] = timePart.split(':');
        date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day), parseInt(hour), parseInt(minute));
      } else {
        date = new Date(timestamp);
      }
      
      if (isNaN(date.getTime())) {
        console.error('Invalid date:', timestamp);
        return '';
      }
      
      const hour = date.getHours();
      const startTime = new Date(date).toLocaleString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      });
      const endTime = new Date(new Date(date).setHours(hour + 1)).toLocaleString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      });
      return `${startTime} - ${endTime}`;
    } catch (error) {
      console.error('Error formatting time:', error);
      return '';
    }
  };

  // Prepare data for the chart
  const chartData = weatherData.length > 0 && predictions.length > 0 
    ? prepareChartData(weatherData, predictions)
    : [];

  // Custom tooltip for the chart
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      // Check if this is a "bridge" data point where we artificially added actual value
      // to connect lines (when both median and actual exist, but actual matches median exactly)
      const isBridgePoint = data.median !== undefined && 
                            data.actual !== undefined && 
                            data.actual === data.median &&
                            !data.isActualDataPoint;
      
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-md shadow-md">
          <p className="font-bold">{data.timeLabel}</p>
          
          {data.forecastRange && (
            <div className="mt-2">
              <p className="font-semibold text-amber-600">Interval Forecast:</p>
              <p>{`${data.forecastRange[0].toFixed(2)} - ${data.forecastRange[1].toFixed(2)} W/m²`}</p>
            </div>
          )}
          
          {data.median !== undefined && (
            <p className="text-purple-600">{`Point Estimate ${data.median.toFixed(2)} W/m²`}</p>
          )}
          
          {data.actual !== undefined && !isBridgePoint && (
            <p className="text-orange-500">{`Actual: ${data.actual.toFixed(2)} W/m²`}</p>
          )}
        </div>
      );
    }
    return null;
  };

  // Update the weather parameters section
  const weatherParams: WeatherParam[] = [
    {
      key: 'temperature',
      icon: WEATHER_ICONS.temperature,
      label: 'Temperature',
      tooltip: 'Temperature affects solar panel efficiency and is crucial for GHI forecasting. Higher temperatures typically reduce panel efficiency.',
      field: 'Temp - °C',
      unit: '°C'
    },
    {
      key: 'humidity',
      icon: WEATHER_ICONS.humidity,
      label: 'Humidity',
      tooltip: 'Humidity levels impact solar radiation transmission through the atmosphere. High humidity can scatter and absorb solar radiation.',
      field: 'Hum - %',
      unit: '%'
    },
    {
      key: 'dewPoint',
      icon: WEATHER_ICONS.dewPoint,
      label: 'Dew Point',
      tooltip: 'Dew point indicates atmospheric moisture content and cloud formation potential. Lower dew points often correlate with clearer skies.',
      field: 'Dew Point - °C',
      unit: '°C'
    },
    {
      key: 'wetBulb',
      icon: WEATHER_ICONS.wetBulb,
      label: 'Wet Bulb',
      tooltip: 'Wet bulb temperature combines temperature and humidity effects, helping predict atmospheric conditions.',
      field: 'Wet Bulb - °C',
      unit: '°C'
    },
    {
      key: 'windSpeed',
      icon: WEATHER_ICONS.windSpeed,
      label: 'Wind Speed',
      tooltip: 'Wind speed affects cloud movement and atmospheric particle distribution, influencing GHI fluctuations.',
      field: 'Avg Wind Speed - km/h',
      unit: 'km/h'
    },
    {
      key: 'uvIndex',
      icon: WEATHER_ICONS.uvIndex,
      label: 'UV Index',
      tooltip: 'UV Index directly correlates with solar radiation intensity. Higher UV indices typically indicate stronger solar radiation.',
      field: 'UV Index',
      unit: ''
    }
  ];

  // Update the processWeatherValue function
  const processWeatherValue = (data: WeatherData | null, field: string): number => {
    if (!data || !data[field]) return 0;
    
    const value = data[field];
    const numValue = typeof value === 'string' ? parseFloat(value) : Number(value);
    
    if (isNaN(numValue)) return 0;
    return Math.round(numValue * 100) / 100; // Round to 2 decimal places
  };

  // Update the weather parameters display section in renderMainView
  {weatherData.length > 0 && (
    <div className="grid grid-cols-2 gap-4 mt-6 pt-6 border-t border-gray-200">
      {weatherParams.map((param) => {
        const latestData = weatherData[weatherData.length - 1];
        const value = processWeatherValue(latestData, param.field);
        const displayValue = value.toFixed(2) + param.unit;

        return (
          <div key={param.key} className="text-center text-gray-800 group relative">
            <div className="text-sm flex items-center justify-center gap-2">
              {param.icon} {param.label}
            </div>
            <div className="font-bold text-amber-600">
              {displayValue}
            </div>
            <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute z-10 p-2 -bottom-28 left-1/2 transform -translate-x-1/2 w-48 bg-amber-500 text-white text-xs rounded-lg shadow-lg">
              {param.tooltip}
            </div>
          </div>
        );
      })}
    </div>
  )}

  // Update the SimpleHistoricalChart component
  const SimpleHistoricalChart = ({ data }: { data: HistoricalGHI | null }) => {
    if (!data || !data.values || !data.times || data.values.length === 0) {
      return <div className="text-center text-gray-500 p-4">No historical data available</div>;
    }

    console.log("Raw historical data times:", data.times);
    
    // Create array of time-value pairs for sorting
    const timeValuePairs = data.times.map((timeStr, index) => {
      const [hoursStr] = timeStr.split(':');
      const hours = parseInt(hoursStr);
      return {
        originalTime: timeStr,
        hour: hours,
        value: data.values[index]
      };
    });

    // Sort by hour
    timeValuePairs.sort((a, b) => a.hour - b.hour);

    // Filter for daytime hours (5 AM to 6 PM)
    const daytimeData = timeValuePairs.filter(pair => pair.hour >= 5 && pair.hour <= 18);
    
    const chartData = daytimeData.map(({ hour, value }) => {
      // Set proper time labels with AM/PM
      let formattedTime;
      if (hour < 12) {
        formattedTime = `${hour} AM - ${hour + 1} ${hour + 1 === 12 ? 'PM' : 'AM'}`;
      } else if (hour === 12) {
        formattedTime = `12 PM - 1 PM`;
      } else if (hour < 24) {
        const displayHour = hour % 12;
        const nextHour = (hour + 1) % 12 || 12;
        const nextAmPm = hour + 1 === 24 ? 'AM' : 'PM';
        formattedTime = `${displayHour} PM - ${nextHour} ${nextAmPm}`;
      } else {
        formattedTime = `${hour % 12 || 12} AM - ${(hour + 1) % 12 || 12} AM`;
      }
      
      console.log(`Hour ${hour} formatted as: ${formattedTime}`);
      
      return {
        time: formattedTime,
        value: value
      };
    });

    console.log("Final chart data:", chartData);

    return (
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart
          data={chartData}
          margin={{ right: 30 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="time" 
            label={{ value: 'Time', position: 'insideBottom', offset: -35 }}
            tick={{ fontSize: 11 }}
            height={60}
            interval={0}
            angle={45}
            textAnchor="start"
          />
          <YAxis 
            label={{ value: 'GHI (W/m²)', angle: -90, position: 'insideLeft' }} 
          />
          <Tooltip />
          <Legend />
          <Area 
            type="monotone" 
            dataKey="value" 
            fill="#ffc658" 
            stroke="#ffa726" 
            name="GHI" 
          />
          {/* Connection Line */}
          <Line
            type="monotone"
            dataKey="connectionLine"
            stroke="#FFFFFF"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name="Transition"
            connectNulls={true}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    );
  };

  // Render main view with predictions
  const renderMainView = () => {
    // Add loading check at the start
    if (!predictions.length || !weatherData.length) {
      return (
        <div className="container mx-auto px-4 pt-20">
          <div className="bg-white bg-opacity-90 rounded-xl p-6 shadow-lg text-center">
            <div className="text-gray-600">Loading data...</div>
          </div>
        </div>
      );
    }

    return (
      <>
        <div className="container mx-auto px-4 pt-20">
          <div className="flex gap-6 mb-8">
            {/* Left Panel - Latest Time, GHI Values, and Weather Parameters */}
            <div className="w-1/3 bg-white bg-opacity-90 rounded-xl p-6 shadow-lg">
              {/* Current Time in PHT */}
              <div className="text-2xl font-bold text-gray-800 mb-4 text-center">
                {new Date().toLocaleString('en-US', {
                  timeZone: 'Asia/Manila',
                  weekday: 'long',
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                })}
                <br />
                {new Date().toLocaleString('en-US', {
                  timeZone: 'Asia/Manila',
                  hour: 'numeric',
                  minute: '2-digit',
                  hour12: true
                })}
              </div>

              {/* GHI Forecast Description */}
              <div className="text-sm text-gray-500 mb-4 text-center">
                Single Point Forecast for The Next Hour:
              </div>

              {/* Single Point Forecast Display */}
              <div className="bg-amber-50 p-4 rounded-lg mb-4">
                <div className="text-center group relative border-2 border-amber-600 p-2 rounded-md">
                  
                  <div className="text-3xl font-bold text-amber-600">
                    {predictions[0]?.median.toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-500">W/m²</div>
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute z-10 p-2 -bottom-24 left-1/2 transform -translate-x-1/2 w-48 bg-amber-500 text-white text-xs rounded-lg shadow-lg">
                    This is the most likely amount of sunlight energy expected for the next hour, representing our best single estimate.
                  </div>
                </div>
              </div>

              {/* Interval Range Forecast Description */}
              <div className="text-sm text-gray-500 mb-4 text-center">
                Interval Range Forecast for The Next Hour:
              </div>

              {/* Updated GHI Bounds Display */}
              <div className="bg-amber-50 p-4 rounded-lg mb-6">
                <div className="flex justify-between items-center gap-4">
                  <div className="text-center flex-1 group relative border-2 border-amber-600 p-2 rounded-md">
                    <div className="text-sm text-gray-600 mb-1">Minimum</div>
                    <div className="text-sm text-gray-600 mb-1">Average GHI</div>
                    <div className="text-3xl font-bold text-amber-600">
                      {predictions[0]?.lower_bound.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">W/m²</div>
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute z-10 p-2 -bottom-24 left-1/2 transform -translate-x-1/2 w-48 bg-amber-500 text-white text-xs rounded-lg shadow-lg">
                      This is the lowest amount of sunlight energy expected for a given period. It helps you estimate the minimum solar power your system will generate.
                    </div>
                  </div>
                  <div className="h-12 w-px bg-amber-200"></div>
                  <div className="text-center flex-1 group relative border-2 border-amber-600 p-2 rounded-md">
                    <div className="text-sm text-gray-600 mb-1">Maximum Average GHI</div>
                    <div className="text-3xl font-bold text-amber-600">
                      {predictions[0]?.upper_bound.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">W/m²</div>
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute z-10 p-2 -bottom-24 left-1/2 transform -translate-x-1/2 w-48 bg-amber-500 text-white text-xs rounded-lg shadow-lg">
                      This is the highest amount of sunlight energy expected for a given period. It helps you estimate the maximum solar power your system can generate.
                    </div>
                  </div>
                </div>
              </div>

              {/* Weather Parameters with Tooltips */}
              {weatherData.length > 0 && (
                <div className="grid grid-cols-2 gap-4 mt-6 pt-6 border-t border-gray-200">
                  {weatherParams.map((param) => {
                    const latestData = weatherData[weatherData.length - 1];
                    const value = processWeatherValue(latestData, param.field);
                    const displayValue = value.toFixed(2) + param.unit;

                    return (
                      <div key={param.key} className="text-center text-gray-800 group relative">
                        <div className="text-sm flex items-center justify-center gap-2">
                          {param.icon} {param.label}
                        </div>
                        <div className="font-bold text-amber-600">
                          {displayValue}
                        </div>
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute z-10 p-2 -bottom-28 left-1/2 transform -translate-x-1/2 w-48 bg-amber-500 text-white text-xs rounded-lg shadow-lg">
                          {param.tooltip}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Right Panel - Updated with Recharts */}
            <div className="w-2/3 bg-white bg-opacity-90 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Average Global Horizontal Irradiance Forecast</h2>
              <div style={{ width: '100%', height: '400px', marginBottom: '20px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={chartData}
                    margin={{ right: 70 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis 
                      dataKey="timeLabel"
                      tick={{ fontSize: 10 }}
                      height={70}
                      interval={0}
                      angle={45}
                      textAnchor="start"
                      label={{ 
                        value: 'Time', 
                        position: 'insideBottom',
                        offset: -60,
                        style: { fontSize: '10px' }
                      }}
                    />
                    <YAxis 
                      label={{ 
                        value: 'W/m²', 
                        angle: -90, 
                        position: 'insideLeft',
                        style: { fontSize: '10px' }
                      }}
                      tickFormatter={(value) => value.toFixed(0)}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend 
                      verticalAlign="top"
                      align="right"
                      wrapperStyle={{
                        paddingBottom: '10px',
                        fontSize: '11px'
                      }}
                    />
                    
                    {/* Forecast Range as Area */}
                    <Area
                      dataKey="forecastRange"
                      fill="#FFB74D"
                      fillOpacity={0.2}
                      stroke="none"
                      name="Interval Forecast"
                      connectNulls={true}
                    />
                    
                    {/* Median as Line */}
                    <Line
                      type="monotone"
                      dataKey="median"
                      stroke="#9C27B0"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      activeDot={{ r: 6 }}
                      name="Single Point Forecast"
                      connectNulls={true}
                      isAnimationActive={false}
                    />
                    
                    {/* Actual GHI as Line */}
                    <Line
                      type="monotone"
                      dataKey="actual"
                      stroke="#2196F3"
                      strokeWidth={2}
                      dot={{ r: 5 }}
                      activeDot={{ r: 7 }}
                      name="Actual GHI"
                      connectNulls={true}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div className="relative mt-4">
                <div className="flex items-center">
                  <ScrollButton direction="left" onClick={() => {
                    const container = document.getElementById('ghi-values-container');
                    if (container) {
                      container.scrollBy({ left: -200, behavior: 'smooth' });
                    }
                  }} />
                  
                  <div 
                    id="ghi-values-container"
                    className="flex overflow-x-auto scrollbar-none mx-12 px-2 py-1 gap-4"
                    style={{ 
                      scrollBehavior: 'smooth',
                      msOverflowStyle: 'none',
                      scrollbarWidth: 'none',
                      maxWidth: '100%',
                      WebkitOverflowScrolling: 'touch'
                    }}
                  >
                    {Array.from({ length: 13 }, (_, i) => i + 5).map((hour) => {
                      // Get the start period column name from first entry
                      const sampleEntry = weatherData[0];
                      const startPeriodColumn = Object.keys(sampleEntry).find(key => 
                        key === 'Start_period' || key === 'Start Period' || key.includes('Start')
                      ) || 'Start_period';
                      
                      // Get the target date from the latest entry
                      const latestEntry = weatherData[weatherData.length - 1];
                      const targetDate = latestEntry.Date;
                      
                      // Find actual data for this hour
                      const actualData = weatherData
                        .filter(d => {
                          if (!d.Date) return false;
                          
                          const dateMatch = d.Date === targetDate;
                          const startPeriod = d[startPeriodColumn] || d['Start Period'];
                          if (!startPeriod) return false;
                          
                          const hourStr = startPeriod.toString().split(':')[0];
                          const entryHour = parseInt(hourStr);
                          
                          return dateMatch && entryHour === hour;
                        })[0];

                      // Find regular prediction for this hour
                      const prediction = predictions.find(p => {
                        if (!p.timestamp) return false;
                        
                        const [datePart, timePart] = p.timestamp.split(' ');
                        const [hourStr] = timePart.split(':');
                        
                        const endHour = parseInt(hourStr);
                        const startHour = (endHour - 1 + 24) % 24;
                        
                        return startHour === hour;
                      });

                      // Find next hour prediction if we have actual data for this hour
                      const nextHourPrediction = actualData ? nextHourPredictions.find(p => {
                        if (!p.timestamp) return false;
                        
                        const [datePart, timePart] = p.timestamp.split(' ');
                        const [hourStr] = timePart.split(':');
                        
                        const endHour = parseInt(hourStr);
                        const startHour = (endHour - 1 + 24) % 24;
                        
                        return startHour === hour;
                      }) : null;

                      return (
                        <div 
                          key={hour} 
                          className={`flex-shrink-0 w-40 text-center p-3 rounded-lg border-2 ${
                            actualData 
                              ? 'border-blue-400 bg-blue-600 bg-opacity-20' 
                              : prediction 
                                ? 'border-primary-yellow bg-primary-yellow bg-opacity-10'
                                : 'border-gray-600 bg-dark-card'
                          } hover:border-opacity-100 transition-all duration-200`}
                        >
                          <div className="text-sm font-medium text-white">
                            {`${hour % 12 || 12}${hour < 12 ? 'am' : 'pm'} - ${(hour + 1) % 12 || 12}${(hour + 1) < 12 ? 'am' : 'pm'}`}
                          </div>
                          {actualData ? (
                            <div>
                              <div className="font-semibold mt-1 text-blue-400">
                                {Number(actualData['GHI - W/m^2']).toFixed(2)} W/m²
                              </div>
                              {nextHourPrediction && (
                                <div className="text-xs mt-1 text-primary-yellow font-semibold">
                                  Single Prediction: {Number(nextHourPrediction.median).toFixed(2)} W/m²
                                </div>
                              )}
                            </div>
                          ) : prediction ? (
                            <div className="space-y-1">
                              <div className="font-semibold text-primary-yellow">
                                {Number(prediction.lower_bound).toFixed(2)} - {Number(prediction.upper_bound).toFixed(2)} W/m²
                              </div>
                            </div>
                          ) : (
                            <div className="font-semibold mt-1 text-gray-400">--</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                  
                  <ScrollButton direction="right" onClick={() => {
                    const container = document.getElementById('ghi-values-container');
                    if (container) {
                      container.scrollBy({ left: 200, behavior: 'smooth' });
                    }
                  }} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Days of The Week Section */}
        <div className="bg-dark-card rounded-xl p-6 shadow-lg border border-opacity-10 border-primary-yellow">
          <h2 className="text-xl font-bold text-white mb-4">Past Week Average GHI Values</h2>
          <div className="flex space-x-2 mb-4">
            {historicalDates.slice(0, 6).map((date, index) => (
              <button
                key={index}
                onClick={() => setSelectedDay(index)}
                className={`px-4 py-2 rounded transition-all duration-200 ${
                  selectedDay === index
                    ? 'bg-primary-yellow text-dark-bg border border-primary-yellow' 
                    : 'bg-dark-bg text-white border border-gray-600 hover:border-primary-yellow'
                }`}
              >
                {date}
              </button>
            ))}
          </div>
          {historicalGHI && (
            <div className="h-[300px]">
              <SimpleHistoricalChart data={historicalGHI} />
            </div>
          )}
        </div>
      </>
    );
  };

  // Helper function to get prediction for specific hour
  const getPredictionForHour = (hour: number) => {
    return predictions.find(p => {
      const predictionHour = new Date(p.timestamp).getHours();
      return predictionHour === hour;
    });
  };

  // Replace all view checks with type-safe checks
  const viewMap: Record<View, boolean> = {
    'main': view === 'main',
    'calendar': view === 'calendar',
    'details': view === 'details',
    'login': view === 'login',
    'admin': view === 'admin',
    'add-weather': view === 'add-weather',
    'solar-power': view === 'solar-power',
    'power-dispatch': view === 'power-dispatch',
    'appliance-calculator': view === 'appliance-calculator'
  };

  // Add this function to fetch historical data
  const fetchHistoricalGHI = async (daysAgo: number) => {
    try {
      console.log("Fetching historical data for days ago:", daysAgo);
      const response = await axios.get(`http://146.190.121.70:5000/historical-ghi/${daysAgo}`);
      
      if (response.data && response.data.times && response.data.values) {
        // Log the data for debugging
        console.log("Historical GHI data received from server:", response.data);
        console.log("Raw times array:", JSON.stringify(response.data.times));
        
        // Skip the formatting of times for debugging purposes to see if that's causing issues
        // Just use the times directly from the server for now
        const rawData = {
          ...response.data
        };
        
        console.log("Setting historical GHI with raw data", rawData);
        setHistoricalGHI(rawData);
      } else {
        console.error("Received invalid historical data format");
        setHistoricalGHI(null);
      }
    } catch (error) {
      console.error('Error fetching historical GHI:', error);
      setHistoricalGHI(null);
    }
  };

  // Add this useEffect to fetch data when selected day changes
  useEffect(() => {
    fetchHistoricalGHI(selectedDay);
  }, [selectedDay]);

  // Modify the fetchHistoricalDates function to fetch all dates first
  const fetchHistoricalDates = async () => {
    try {
      const response = await axios.get('http://146.190.121.70:5000/historical-dates');
      // Sort dates chronologically (oldest to newest)
      const sortedDates = response.data.sort((a: string, b: string) => {
        const dateA = new Date(a);
        const dateB = new Date(b);
        return dateA.getTime() - dateB.getTime();
      });
      setHistoricalDates(sortedDates);
    } catch (error) {
      console.error('Error fetching historical dates:', error);
    }
  };

  // Add useEffect to fetch dates when component mounts
  useEffect(() => {
    fetchHistoricalDates();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-amber-300 to-amber-400">
      <TopBar setView={setView} isSidebarOpen={isSidebarOpen} setIsSidebarOpen={setIsSidebarOpen} />
      {isSidebarOpen && <SideBar setView={setView} setIsSidebarOpen={setIsSidebarOpen} />}

      {viewMap['calendar'] && (
        <CalendarView 
          onBack={() => setView('main')} 
          days={days}
          data={predictions} 
          onTimeSelect={handleTimeSelect}
        />
      )}

      {viewMap['details'] && (
        <DayDetailsView 
          onBack={() => setView('main')} 
          selectedTime={selectedTime} 
          data={weatherData}
        />
      )}

      {viewMap['login'] && (
        <LoginPage onLogin={handleLogin} onBack={() => setView('main')} />
      )}

      {viewMap['admin'] && (
        <AdminPage 
          onBack={() => setView('main')} 
          onDataUpdate={refreshData}
          onAddWeather={() => setView('add-weather')}
        />
      )}

      {viewMap['add-weather'] && (
        <AddWeatherPage 
          onBack={() => setView('admin')} 
          onDataUpdate={refreshData}
        />
      )}

      {viewMap['solar-power'] && (
        <div className="container mx-auto px-4 pt-20">
          <SlideButtonRight 
            onClick={() => {
              if (isConfigValid(powerRating, panelCount, performanceRatio)) {
                setView('power-dispatch');
              } else {
                alert('Please select a solar panel and enter the number of panels before proceeding.');
              }
            }} 
          />
          <SolarPowerGeneration 
            predictions={predictions}
            powerRating={powerRating}
            panelCount={panelCount}
            performanceRatio={performanceRatio}
            onPowerRatingChange={setPowerRating}
            onPanelCountChange={setPanelCount}
            onPerformanceRatioChange={setPerformanceRatio}
            onCurrentSolarGeneration={setCurrentSolarGeneration}
            onGenerationDataChange={setGenerationData}
          />
        </div>
      )}

      {viewMap['power-dispatch'] && (
        <div className="container mx-auto px-4 pt-20">
          <SlideButtonLeft 
            onClick={() => setView('solar-power')} 
          />
          <SlideButtonRight 
            onClick={() => {
              if (isConfigValid(powerRating, panelCount, performanceRatio)) {
                setView('appliance-calculator');
              } else {
                alert('Please select a solar panel and enter the number of panels before proceeding.');
                setView('solar-power');
              }
            }} 
          />
          <PowerDispatchOptimization 
            predictions={predictions}
            currentGeneration={currentSolarGeneration}
            generationData={generationData}
            powerRating={powerRating}
            panelCount={panelCount}
          />
        </div>
      )}

      {viewMap['appliance-calculator'] && (
        <div className="container mx-auto px-4 pt-20">
          <SlideButtonLeft 
            onClick={() => {
              if (isConfigValid(powerRating, panelCount, performanceRatio)) {
                setView('power-dispatch');
              } else {
                alert('Please select a solar panel and enter the number of panels before proceeding.');
                setView('solar-power');
              }
            }} 
          />
          <ApplianceCalculator 
            onBack={() => handleNavigation('main')}
            predictions={predictions}
            powerRating={powerRating}
            panelCount={panelCount}
            performanceRatio={performanceRatio}
          />
        </div>
      )}

      {viewMap['main'] && (
        <div className="container mx-auto px-4 pt-20">
          {renderMainView()}
        </div>
      )}
    </div>
  );
}

export default App;