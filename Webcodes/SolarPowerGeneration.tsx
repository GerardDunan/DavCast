import { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import Slider from '@mui/material/Slider';
import { styled } from '@mui/material/styles';
import Tooltip from '@mui/material/Tooltip';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Legend
} from 'chart.js';
import { PredictionData } from '../types';
import { useSlider } from '../context/SliderContext';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Legend
);

// Custom styled slider
const PredictionSlider = styled(Slider)(({ theme }) => ({
  color: 'var(--primary-yellow)',
  height: 8,
  '& .MuiSlider-track': {
    border: 'none',
    height: 8,
  },
  '& .MuiSlider-thumb': {
    height: 24,
    width: 24,
    backgroundColor: 'var(--dark-card)',
    border: '2px solid var(--primary-yellow)',
    '&:focus, &:hover, &.Mui-active': {
      boxShadow: '0 0 0 8px rgba(255, 184, 0, 0.16)',
    },
    '&:before': {
      display: 'none',
    },
  },
  '& .MuiSlider-markLabel': {
    color: 'white',
  },
}));

interface Props {
  predictions: PredictionData[];
  powerRating: string;
  panelCount: string;
  performanceRatio: string;
  onPowerRatingChange: (value: string) => void;
  onPanelCountChange: (value: string) => void;
  onPerformanceRatioChange: (value: string) => void;
  onCurrentSolarGeneration: (value: number) => void;
  onGenerationDataChange: (data: number[]) => void;
}

interface SolarPanel {
  brand: string;
  model: string;
  power: number;
  efficiency: number;
}

const SOLAR_PANELS: SolarPanel[] = [
  { brand: "Jinko Solar", model: "Tiger Neo N-Type 425W", power: 425, efficiency: 21.78 },
  { brand: "Jinko Solar", model: "JKM400M-54HL4-B", power: 400, efficiency: 20.51 },
  { brand: "Longi Solar", model: "Hi-MO 5m 415W", power: 415, efficiency: 20.90 },
  { brand: "Longi Solar", model: "Hi-MO 6 430W", power: 430, efficiency: 21.30 },
  { brand: "Trina Solar", model: "Vertex S+ 440W", power: 440, efficiency: 21.50 },
  { brand: "Trina Solar", model: "Vertex DE19R.08 450W", power: 450, efficiency: 21.90 },
  { brand: "Canadian Solar", model: "HiKu7 420W", power: 420, efficiency: 20.20 },
  { brand: "Canadian Solar", model: "BiHiKu 430W", power: 430, efficiency: 20.67 },
  { brand: "JA Solar", model: "JAM72S30 410W", power: 410, efficiency: 20.50 },
  { brand: "JA Solar", model: "Deep Blue 4.0 405W", power: 405, efficiency: 20.25 }
];

// Add constants for losses
const INVERTER_EFFICIENCY = 0.95;
const TEMPERATURE_LOSS = 0.15;
const SOILING_LOSS = 0.03;
const WIRING_LOSS = 0.02;

// Add a custom tooltip component
const ValueTooltip = ({ children, title }: { children: React.ReactNode; title: string }) => (
  <Tooltip title={title} placement="top">
    <span className="cursor-help">{children}</span>
  </Tooltip>
);

const SolarPowerGeneration: React.FC<Props> = ({ 
  predictions, 
  powerRating,
  panelCount,
  performanceRatio,
  onPowerRatingChange,
  onPanelCountChange,
  onPerformanceRatioChange,
  onCurrentSolarGeneration,
  onGenerationDataChange
}) => {
  const { predictionPercentage, setPredictionPercentage } = useSlider();

  // Check if required inputs are provided
  const hasRequiredInputs = powerRating && panelCount && parseInt(powerRating) > 0 && parseInt(panelCount) > 0;

  // Calculate power generation with prediction percentage
  const calculateSolarPower = (prediction: PredictionData) => {
    if (!hasRequiredInputs) return 0;
    
    const power = parseFloat(powerRating);
    const panels = parseFloat(panelCount);
    const range = prediction.upper_bound - prediction.lower_bound;
    const ghi = prediction.lower_bound + (range * (predictionPercentage / 100));
    
    return ((ghi / 1000) * power * panels * INVERTER_EFFICIENCY * 
           (1 - TEMPERATURE_LOSS) * (1 - SOILING_LOSS) * (1 - WIRING_LOSS)) / 1000;
  };

  const formatTimeRange = (timestamp: string) => {
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

  const chartData = {
    labels: predictions.map(p => formatTimeRange(p.timestamp)),
    datasets: [
      {
        label: 'Power Generation (kW)',
        data: predictions.map(p => {
          const power = calculateSolarPower(p);
          return Number(power.toFixed(3));
        }),
        borderColor: '#00CED1',
        borderWidth: 2,
        tension: 0.4,
        fill: {
          target: 'origin',
          above: 'rgba(0, 206, 209, 0.3)'
        }
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Solar Power Generation Forecast'
      }
    },
    scales: {
      y: {
        type: 'linear' as const,
        beginAtZero: true,
        title: {
          display: true,
          text: 'Power (kW)'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          callback: function(tickValue: number | string) {
            return typeof tickValue === 'number' ? tickValue.toFixed(3) : tickValue;
          }
        }
      },
      x: {
        type: 'category' as const,
        title: {
          display: true,
          text: 'Time'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
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
    interaction: {
      intersect: false,
      mode: 'index' as const
    }
  };

  // Get current GHI value
  const getCurrentGHI = () => {
    if (!hasRequiredInputs || !predictions.length) return 0;
    const prediction = predictions[0];
    const range = prediction.upper_bound - prediction.lower_bound;
    return prediction.lower_bound + (range * (predictionPercentage / 100));
  };

  // Update chart data when prediction percentage changes
  useEffect(() => {
    if (hasRequiredInputs && predictions.length > 0) {
      const currentPower = calculateSolarPower(predictions[0]);
      onCurrentSolarGeneration(currentPower);
      
      const generationData = predictions.map(p => {
        const power = calculateSolarPower(p);
        return Number(power.toFixed(3));
      });
      onGenerationDataChange(generationData);
    } else {
      onCurrentSolarGeneration(0);
      onGenerationDataChange([]);
    }
  }, [predictionPercentage, powerRating, panelCount, predictions]);

  // Get selected panel efficiency
  const getSelectedPanelEfficiency = () => {
    if (!hasRequiredInputs) return 0;
    const selectedPanel = SOLAR_PANELS.find(p => p.power === parseInt(powerRating));
    return selectedPanel?.efficiency || 0;
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-center text-white mb-8">
        Set-up your PV System
      </h1>

      <div className="flex gap-6">
        <div className="w-1/3 bg-dark-card rounded-xl p-6 shadow-lg border border-opacity-10 border-white">
          <h2 className="text-xl font-bold text-white mb-4">Solar System Size</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-gray-400 mb-2">Solar Panel:</label>
              <select
                value={powerRating}
                onChange={(e) => onPowerRatingChange(e.target.value)}
                className="w-full p-2 rounded border border-opacity-20 border-white bg-dark-bg text-white focus:outline-none focus:border-primary-yellow"
              >
                <option value="">Select a solar panel</option>
                {SOLAR_PANELS.map((panel) => (
                  <option key={panel.model} value={panel.power}>
                    {`${panel.brand} ${panel.model} (${panel.power}W)`}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-gray-400 mb-2">Number of Solar Panels:</label>
              <input
                type="number"
                min="1"
                value={panelCount}
                onChange={(e) => onPanelCountChange(e.target.value)}
                className="w-full p-2 rounded border border-opacity-20 border-white bg-dark-bg text-white focus:outline-none focus:border-primary-yellow"
                placeholder="Enter number of panels"
              />
            </div>

            {/* Only show prediction slider and results if inputs are provided */}
            {hasRequiredInputs && (
              <>
                <div className="mt-6">
                  <label className="block text-gray-700 mb-2">Average GHI Forecast Interval for the Next Hour:</label>
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

                <div className="mt-6 p-4 bg-opacity-10 bg-yellow-500 rounded-lg space-y-4 border border-opacity-10 border-white">
                  <div>
                    <h3 className="text-white font-semibold mb-2">Power Generated For The Next Hour:</h3>
                    <p className="text-2xl font-bold text-primary-yellow">
                      {predictions.length > 0 ? calculateSolarPower(predictions[0]).toFixed(3) : '0.000'} kW
                    </p>
                  </div>

                  <div className="pt-4 border-t border-opacity-10 border-white">
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
                        <span className="font-medium text-white">{getSelectedPanelEfficiency()}%</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="w-2/3 space-y-6">
          {/* Only show graph and calculations if inputs are provided */}
          {hasRequiredInputs ? (
            <>
              <div className="bg-white bg-opacity-90 rounded-xl p-6" style={{ height: '400px' }}>
                <Line options={options} data={chartData} />
              </div>

              <div className="bg-white bg-opacity-90 rounded-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">Breakdown of Calculation:</h3>
                <div className="flex justify-between gap-4">
                  {[0, 1, 2, 3].map((index) => {
                    const prediction = predictions[index];
                    const hour = index + 1;
                    const ghi = prediction.lower_bound + 
                      ((prediction.upper_bound - prediction.lower_bound) * (predictionPercentage / 100));
                    const power = calculateSolarPower(prediction);

                    return (
                      <div 
                        key={index}
                        onClick={() => {
                          const modal = document.getElementById(`calculation-modal-${index}`);
                          if (modal) {
                            modal.style.display = 'block';
                          }
                        }}
                        className="flex-1 w-48 text-center p-3 rounded-lg border-2 border-amber-400 bg-amber-50 hover:bg-amber-100 transition-all duration-200 cursor-pointer relative"
                      >
                        <div>
                          <p className="text-yellow-500 text-lg font-bold mb-1">
                            Average GHI: {ghi.toFixed(2)} W/m²
                          </p>
                          <p className="text-cyan-600 text-base font-bold">
                            {hour === 1 ? 'Next' : `${hour}-Hour Ahead`}<br/>
                            {power.toFixed(3)} kW
                          </p>
                        </div>

                        {/* Modal for detailed calculation */}
                        <div 
                          id={`calculation-modal-${index}`}
                          className="hidden fixed inset-0 bg-black bg-opacity-80 z-50"
                          onClick={(e) => {
                            if (e.target === e.currentTarget) {
                              const modal = document.getElementById(`calculation-modal-${index}`);
                              if (modal) {
                                modal.style.display = 'none';
                              }
                            }
                          }}
                        >
                          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-dark-card rounded-xl p-6 max-w-2xl w-full shadow-2xl border border-opacity-10 border-primary-yellow" onClick={(e) => e.stopPropagation()}>
                            <button 
                              onClick={(e) => {
                                e.stopPropagation();
                                const modal = document.getElementById(`calculation-modal-${index}`);
                                if (modal) {
                                  modal.style.display = 'none';
                                }
                              }}
                              className="absolute top-4 right-4 text-gray-500 hover:text-primary-yellow transition-colors"
                            >
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            </button>
                            
                            <h3 className="text-xl font-bold text-white mb-4">
                              {hour === 1 ? 'Next Hour' : `${hour}-Hour Ahead`} Power Generation Calculation
                            </h3>
                            
                            <div className="grid grid-cols-2 gap-4">
                              <div className="bg-black bg-opacity-40 p-4 rounded-lg border border-opacity-10 border-primary-yellow">
                                <p className="text-lg font-semibold text-primary-yellow mb-2">Input Parameters:</p>
                                <ul className="space-y-1 text-white">
                                  <li>• <span className="text-primary-yellow font-medium">Average GHI:</span> <span className="text-primary-yellow">{ghi.toFixed(2)} W/m²</span></li>
                                  <li>• <span className="text-cyan-400 font-medium">Panel Power:</span> {powerRating} W</li>
                                  <li>• <span className="text-green-400 font-medium">Number of Panels:</span> {panelCount}</li>
                                  <li>• <span className="text-purple-400 font-medium">Inverter Efficiency:</span> {(INVERTER_EFFICIENCY * 100).toFixed(1)}%</li>
                                </ul>
                              </div>
                              
                              <div className="bg-black bg-opacity-40 p-4 rounded-lg border border-opacity-10 border-primary-yellow">
                                <p className="text-lg font-semibold text-primary-yellow mb-2">System Losses:</p>
                                <ul className="space-y-1 text-white">
                                  <li>• <span className="text-red-400 font-medium">Temperature Loss:</span> {(TEMPERATURE_LOSS * 100).toFixed(1)}%</li>
                                  <li>• <span className="text-orange-400 font-medium">Soiling Loss:</span> {(SOILING_LOSS * 100).toFixed(1)}%</li>
                                  <li>• <span className="text-blue-400 font-medium">Wiring Loss:</span> {(WIRING_LOSS * 100).toFixed(1)}%</li>
                                </ul>
                              </div>
                            </div>

                            <div className="mt-4 bg-black bg-opacity-40 p-4 rounded-lg border border-opacity-10 border-primary-yellow">
                              <p className="text-lg font-semibold text-primary-yellow mb-2">Calculation Steps:</p>
                              <div className="grid grid-cols-2 gap-4 text-sm text-white">
                                <div>
                                  <p className="font-medium text-cyan-400">1. Convert Average GHI to kW:</p>
                                  <p className="ml-4"><span className="text-primary-yellow">{ghi}</span> ÷ 1000 = <span className="text-primary-yellow">{(ghi/1000).toFixed(3)} kW/m²</span></p>
                                </div>
                                <div>
                                  <p className="font-medium text-cyan-400">2. Multiply by panel capacity:</p>
                                  <p className="ml-4"><span className="text-primary-yellow">{(ghi/1000).toFixed(3)}</span> × (<span className="text-cyan-400">{powerRating}</span> × <span className="text-green-400">{panelCount}</span>)</p>
                                  <p className="ml-4">= <span className="text-green-400">{((ghi/1000) * parseFloat(powerRating) * parseFloat(panelCount)).toFixed(3)} kW</span></p>
                                </div>
                              </div>
                              <div className="mt-2">
                                <p className="font-medium text-cyan-400">3. Apply efficiency and losses:</p>
                                <div className="grid grid-cols-4 gap-2 mt-1 ml-4">
                                  <div className="text-center">
                                    <p className="text-xs text-purple-400">Inverter</p>
                                    <p>× {INVERTER_EFFICIENCY}</p>
                                  </div>
                                  <div className="text-center">
                                    <p className="text-xs text-red-400">Temperature</p>
                                    <p>× {(1 - TEMPERATURE_LOSS).toFixed(2)}</p>
                                  </div>
                                  <div className="text-center">
                                    <p className="text-xs text-orange-400">Soiling</p>
                                    <p>× {(1 - SOILING_LOSS).toFixed(2)}</p>
                                  </div>
                                  <div className="text-center">
                                    <p className="text-xs text-blue-400">Wiring</p>
                                    <p>× {(1 - WIRING_LOSS).toFixed(2)}</p>
                                  </div>
                                </div>
                              </div>
                            </div>

                            <div className="mt-4 bg-black bg-opacity-40 p-4 rounded-lg text-center border border-opacity-10 border-primary-yellow">
                              <p className="text-lg font-semibold text-primary-yellow">Final Result:</p>
                              <p className="text-2xl font-bold text-cyan-400">{power.toFixed(3)} kW</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          ) : (
            <div className="bg-white bg-opacity-90 rounded-xl p-6 flex items-center justify-center" style={{ height: '400px' }}>
              <p className="text-gray-500 text-center">
                Please select a solar panel and enter the number of panels<br />
                to see power generation forecast and calculations
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SolarPowerGeneration; 