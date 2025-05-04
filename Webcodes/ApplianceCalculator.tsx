import React, { useState, useRef } from 'react';
import { Pie, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions
} from 'chart.js';
import { PredictionData } from '../types';
import { styled } from '@mui/material/styles';
import { Slider } from '@mui/material';
import { useSlider } from '../context/SliderContext';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Add these constants at the top of the file, after the imports
const INVERTER_EFFICIENCY = 0.95;
const TEMPERATURE_LOSS = 0.15;
const SOILING_LOSS = 0.03;
const WIRING_LOSS = 0.02;

interface Props {
  onBack: () => void;
  predictions: PredictionData[];
  powerRating: string;
  panelCount: string;
  performanceRatio: string;
}

interface Appliance {
  id: string;
  name: string;
  powerRating: number;
  hourlyEnergy: number;
  icon: string;
  color: string;
}

interface SelectedAppliance extends Appliance {
  quantity: number;
}

const availableAppliances: Appliance[] = [
  { id: 'led', name: 'LED Bulb', powerRating: 10, hourlyEnergy: 0.01, icon: '💡', color: 'rgb(255, 205, 86)' },
  { id: 'fan', name: 'Ceiling Fan', powerRating: 70, hourlyEnergy: 0.07, icon: '🌀', color: 'rgb(54, 162, 235)' },
  { id: 'ac', name: 'AC Unit', powerRating: 2000, hourlyEnergy: 2, icon: '❄️', color: 'rgb(75, 192, 192)' },
  { id: 'fridge', name: 'Refrigerator', powerRating: 150, hourlyEnergy: 0.15, icon: '🧊', color: 'rgb(153, 102, 255)' },
  { id: 'tv', name: 'LED TV', powerRating: 100, hourlyEnergy: 0.1, icon: '📺', color: 'rgb(255, 99, 132)' },
  { id: 'microwave', name: 'Microwave', powerRating: 1000, hourlyEnergy: 1, icon: '🔲', color: 'rgb(255, 159, 64)' },
  { id: 'router', name: 'Wi-Fi Router', powerRating: 10, hourlyEnergy: 0.01, icon: '📡', color: 'rgb(100, 200, 150)' },
  { id: 'charger', name: 'Phone Charger', powerRating: 20, hourlyEnergy: 0.02, icon: '🔌', color: 'rgb(200, 150, 100)' }
];

const PredictionSlider = styled(Slider)(({ theme }) => ({
  color: '#f59e0b',
  height: 8,
  '& .MuiSlider-track': {
    border: 'none',
    height: 8,
  },
  '& .MuiSlider-thumb': {
    height: 24,
    width: 24,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    '&:focus, &:hover, &.Mui-active': {
      boxShadow: '0 0 0 8px rgba(245, 158, 11, 0.16)',
    },
    '&:before': {
      display: 'none',
    },
  },
  '& .MuiSlider-markLabel': {
    color: 'white',
  },
}));

const ApplianceCalculator: React.FC<Props> = ({ 
  onBack,
  predictions, 
  powerRating,
  panelCount,
  performanceRatio 
}) => {
  const { predictionPercentage, setPredictionPercentage } = useSlider();
  const [selectedAppliances, setSelectedAppliances] = useState<SelectedAppliance[]>([]);
  const [tempQuantity, setTempQuantity] = useState<string>('1');
  const [selectedApplianceId, setSelectedApplianceId] = useState<string>('');
  const chartRef = useRef(null);

  // Calculate energy for next hour (kWh)
  const calculateNextHourEnergy = (prediction: PredictionData) => {
    if (!powerRating || !panelCount) return 0;
    
    const power = parseFloat(powerRating);
    const panels = parseFloat(panelCount);
    const range = prediction.upper_bound - prediction.lower_bound;
    const ghi = prediction.lower_bound + (range * (predictionPercentage / 100));
    
    return ((ghi / 1000) * power * panels * INVERTER_EFFICIENCY * 
           (1 - TEMPERATURE_LOSS) * (1 - SOILING_LOSS) * (1 - WIRING_LOSS)) / 1000;
  };

  const handleAddAppliance = () => {
    if (!selectedApplianceId) return;
    
    const appliance = availableAppliances.find(a => a.id === selectedApplianceId);
    if (!appliance) return;

    const quantity = parseInt(tempQuantity) || 1;
    setSelectedAppliances(prev => [...prev, { ...appliance, quantity }]);
    setTempQuantity('1');
    setSelectedApplianceId('');
  };

  const handleRemoveAppliance = (index: number) => {
    setSelectedAppliances(prev => prev.filter((_, i) => i !== index));
  };

  // Calculate total consumption for each appliance
  const calculateApplianceConsumption = () => {
    return selectedAppliances.map(appliance => ({
      name: appliance.name,
      consumption: appliance.hourlyEnergy * appliance.quantity,
      color: appliance.color
    }));
  };

  // Add a helper function for time formatting
  const formatTime = (timestamp: string) => {
    try {
      if (!timestamp) return '';
      
      // Parse the timestamp string (format: "DD/MM/YYYY HH:mm")
      const [datePart, timePart] = timestamp.split(' ');
      const [day, month, year] = datePart.split('/');
      const [hour] = timePart.split(':');
      
      // Since timestamp is end period, subtract 1 hour to get start period
      const endHour = parseInt(hour);
      const startHour = endHour - 1;
      
      // Format the start hour
      const startPeriod = startHour >= 12 ? 'pm' : 'am';
      const formattedStartHour = startHour === 0 ? 12 : startHour > 12 ? startHour - 12 : startHour;
      
      // Format the end hour
      const endPeriod = endHour >= 12 ? 'pm' : 'am';
      const formattedEndHour = endHour === 0 ? 12 : endHour > 12 ? endHour - 12 : endHour;
      
      return `${formattedStartHour}${startPeriod} - ${formattedEndHour}${endPeriod}`;
    } catch (error) {
      console.error('Error formatting time:', error);
      return 'Invalid Date';
    }
  };

  // Get solar generation data for the deviation chart
  const getSolarGenerationData = () => {
    return predictions.map(p => {
      try {
        const time = formatTime(p.timestamp);
        return {
          time,
          generation: calculateNextHourEnergy(p)
        };
      } catch (error) {
        console.error('Error processing prediction:', error);
        return {
          time: '',
          generation: 0
        };
      }
    });
  };

  // Prepare data for pie chart
  const pieChartData = {
    labels: calculateApplianceConsumption().map(a => `${a.name} (${a.consumption.toFixed(2)} kWh)`),
    datasets: [{
      data: calculateApplianceConsumption().map(a => a.consumption),
      backgroundColor: calculateApplianceConsumption().map(a => a.color),
    }]
  };

  // Calculate total hourly consumption
  const getTotalHourlyConsumption = () => {
    return selectedAppliances.reduce((sum, app) => sum + app.hourlyEnergy * app.quantity, 0);
  };

  // Prepare data for the deviation chart
  const deviationChartData = {
    labels: predictions.map(p => formatTime(p.timestamp)),
    datasets: [
      {
        label: 'Solar Generation (kWh)',
        data: getSolarGenerationData().map(d => d.generation),
        borderColor: '#2ecc71',
        borderWidth: 2,
        tension: 0.4,
        fill: {
          target: '+1',
          above: 'rgba(46, 204, 113, 0.3)',
          below: 'rgba(231, 76, 60, 0.3)'
        }
      },
      {
        label: 'Energy Consumption (kWh)',
        data: predictions.map(() => getTotalHourlyConsumption()),
        borderColor: '#e74c3c',
        borderWidth: 2,
        borderDash: [5, 5],
        tension: 0.4,
        fill: false
      }
    ]
  };

  const deviationChartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Energy (kWh)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time'
        },
        ticks: {
          font: {
            size: 11
          },
          maxRotation: 0,
          minRotation: 0
        }
      }
    },
    plugins: {
      legend: {
        position: 'top' as const
      },
      title: {
        display: true,
        text: 'Generation vs Consumption'
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} kWh`;
          }
        }
      }
    },
    interaction: {
      mode: 'index' as const,
      intersect: false
    }
  };

  // Update the chart container style
  const graphStyle = {
    minHeight: "10rem",
    maxWidth: "540px",
    width: "100%",
    border: "1px solid #C4C4C4",
    borderRadius: "0.375rem",
    padding: "0.5rem"
  };

  // Get current GHI value
  const getCurrentGHI = () => {
    if (!predictions.length) return 0;
    const prediction = predictions[0];
    const range = prediction.upper_bound - prediction.lower_bound;
    return prediction.lower_bound + (range * (predictionPercentage / 100));
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-amber-300 to-amber-400">
      <div className="bg-dark-card bg-opacity-90 p-4 fixed top-0 left-0 right-0 z-10 border-b border-opacity-10 border-primary-yellow">
        <div className="container mx-auto flex justify-between items-center">
          <button onClick={onBack} className="text-white hover:text-primary-yellow transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <h1 className="text-2xl font-bold text-primary-yellow">Electrical Devices Calculator</h1>
          <div className="w-6" />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-6 p-6 pt-20">
        {/* Left side - POS style interface */}
        <div className="bg-dark-card rounded-xl p-6 shadow-lg border border-opacity-10 border-primary-yellow">
          <h2 className="text-xl font-bold text-white mb-4">Select Devices</h2>
          
          {/* Appliance selection dropdown */}
          <div className="flex gap-2 mb-6">
            <select
              value={selectedApplianceId}
              onChange={(e) => setSelectedApplianceId(e.target.value)}
              className="flex-grow p-2 border rounded focus:outline-none focus:border-primary-yellow bg-dark-bg text-white"
            >
              <option value="">Select an electrical device</option>
              {availableAppliances.map(appliance => (
                <option key={appliance.id} value={appliance.id}>
                  {appliance.icon} {appliance.name} ({appliance.powerRating}W)
                </option>
              ))}
            </select>
            <input
              type="number"
              min="1"
              value={tempQuantity}
              onChange={(e) => setTempQuantity(e.target.value)}
              className="w-20 p-2 border rounded bg-dark-bg text-white focus:outline-none focus:border-primary-yellow"
              placeholder="Qty"
            />
            <button
              onClick={handleAddAppliance}
              disabled={!selectedApplianceId}
              className="bg-dark-bg text-white px-4 py-2 rounded hover:bg-primary-yellow hover:text-dark-bg disabled:bg-gray-800 disabled:text-gray-500 disabled:hover:bg-gray-800 disabled:hover:text-gray-500 transition-all duration-200 border border-primary-yellow"
            >
              Add
            </button>
          </div>

          {/* Selected appliances with more space */}
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-white mb-3">Selected Devices</h3>
            <div className="grid grid-cols-4 gap-4 max-h-[calc(100vh-400px)] overflow-y-auto">
              {selectedAppliances.map((appliance, index) => (
                <div key={index} className="bg-black bg-opacity-40 p-3 rounded-lg relative border border-opacity-30 border-primary-yellow hover:border-opacity-100 transition-all duration-200">
                  <button
                    onClick={() => handleRemoveAppliance(index)}
                    className="absolute top-1 right-1 text-gray-400 hover:text-red-500 hover:bg-black hover:bg-opacity-30 rounded-full w-6 h-6 flex items-center justify-center transition-all duration-200"
                  >
                    ×
                  </button>
                  <div className="text-center">
                    <span className="text-3xl">{appliance.icon}</span>
                    <div className="mt-2 text-sm font-semibold text-white">{appliance.name}</div>
                    <div className="text-xs text-gray-400">Qty: {appliance.quantity}</div>
                    <div className="text-xs text-primary-yellow">{(appliance.hourlyEnergy * appliance.quantity).toFixed(2)} kWh</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Total consumption display */}
            <div className="mt-4 p-4 bg-black bg-opacity-40 rounded-lg border border-opacity-30 border-primary-yellow">
              <h4 className="font-semibold text-white">Total Energy Consumption</h4>
              <p className="text-xl font-bold text-primary-yellow">
                {selectedAppliances.reduce((sum, app) => sum + app.hourlyEnergy * app.quantity, 0).toFixed(2)} kWh
              </p>
            </div>

            {/* Move Prediction Range Slider here */}
            <div className="mt-4 p-4 bg-black bg-opacity-40 rounded-lg border border-opacity-30 border-primary-yellow">
              <h4 className="font-semibold text-white mb-3">Average GHI Forecast Interval for the Next Hour</h4>
              <div className="px-2">
                <PredictionSlider
                  value={predictionPercentage}
                  onChange={(_, value) => setPredictionPercentage(value as number)}
                  valueLabelDisplay="off"
                  step={1}
                  marks={[
                    { value: 0, label: 'Min' },
                    { value: 100, label: 'Max' }
                  ]}
                />
              </div>
            </div>

            {/* Simplified Energy Status Display */}
            <div className="mt-4 p-4 bg-black bg-opacity-40 rounded-lg border border-opacity-30 border-primary-yellow">
              <h3 className="text-white font-semibold mb-2">Energy Generated For The Next Hour:</h3>
              <div className="text-2xl font-bold text-cyan-400">
                {calculateNextHourEnergy(predictions[0]).toFixed(3)} kWh
              </div>
              <div className="pt-4 border-t border-opacity-10 border-white mt-2">
                <h4 className="text-sm font-semibold text-gray-300 mb-2">System Parameters:</h4>
                <ul className="text-sm space-y-1">
                  <li className="flex justify-between">
                    <span className="text-gray-400">Average GHI:</span>
                    <span className="font-medium text-white">{getCurrentGHI().toFixed(2)} W/m²</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-gray-400">Panel Power:</span>
                    <span className="font-medium text-white">{powerRating} W</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-gray-400">Number of Panels:</span>
                    <span className="font-medium text-white">{panelCount}</span>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-gray-400">Panel Efficiency:</span>
                    <span className="font-medium text-white">20.51%</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Right side - Charts */}
        <div className="space-y-6">
          {/* Pie Chart - only show if there are appliances */}
          {selectedAppliances.length > 0 && (
            <div className="bg-dark-card rounded-xl p-6 shadow-lg border border-opacity-10 border-primary-yellow">
              <h2 className="text-xl font-bold text-white mb-4">Electrical Devices Energy Consumption</h2>
              <div style={{ height: '250px' }}>
                <Pie data={pieChartData} options={{ maintainAspectRatio: false }} />
              </div>
            </div>
          )}

          {/* Deviation Chart - only show if there are appliances */}
          {selectedAppliances.length > 0 && (
            <div className="bg-dark-card rounded-xl p-6 shadow-lg border border-opacity-10 border-primary-yellow">
              <h2 className="text-xl font-bold text-white mb-4">Generation vs Consumption</h2>
              <div style={{ height: '300px' }}>
                <Line 
                  ref={chartRef}
                  data={deviationChartData} 
                  options={deviationChartOptions}
                />
              </div>
            </div>
          )}

          {/* Show message when no appliances */}
          {selectedAppliances.length === 0 && (
            <div className="bg-dark-card rounded-xl p-6 shadow-lg text-center text-gray-400 border border-opacity-10 border-primary-yellow">
              <p>Add devices to see energy consumption analysis</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ApplianceCalculator; 