import { useState, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { PredictionData } from '../types';
import styled from '@emotion/styled';
import { Slider } from '@mui/material';
import { useSlider } from '../context/SliderContext';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface Props {
  predictions: PredictionData[];
  currentGeneration: number;
  generationData?: number[];
  powerRating: string;
  panelCount: string;
  INVERTER_EFFICIENCY?: number;
  TEMPERATURE_LOSS?: number;
  SOILING_LOSS?: number;
  WIRING_LOSS?: number;
}

// Add these constants
const INVERTER_EFFICIENCY = 0.95;
const TEMPERATURE_LOSS = 0.15;
const SOILING_LOSS = 0.03;
const WIRING_LOSS = 0.02;

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

const PowerDispatchOptimization = ({ predictions, currentGeneration, generationData, powerRating, panelCount }: Props) => {
  const [loadDemand, setLoadDemand] = useState<string>('');
  const { predictionPercentage, setPredictionPercentage } = useSlider();
  const chartRef = useRef(null);

  const loadDemandValue = parseFloat(loadDemand) || 0;

  // Updated calculation to match SolarPowerGeneration
  const calculateAdjustedGeneration = (prediction: PredictionData) => {
    if (!powerRating || !panelCount) return 0;
    
    const power = parseFloat(powerRating);
    const panels = parseFloat(panelCount);
    const range = prediction.upper_bound - prediction.lower_bound;
    const ghi = prediction.lower_bound + (range * (predictionPercentage / 100));
    
    return ((ghi / 1000) * power * panels * INVERTER_EFFICIENCY * 
           (1 - TEMPERATURE_LOSS) * (1 - SOILING_LOSS) * (1 - WIRING_LOSS)) / 1000;
  };

  // Calculate current dispatch and excess based on the adjusted generation
  const currentGenValue = calculateAdjustedGeneration(predictions[0]);
  const currentDispatch = loadDemandValue > 0 ? Math.max(0, loadDemandValue - currentGenValue) : 0;
  const currentExcess = loadDemandValue > 0 ? Math.max(0, currentGenValue - loadDemandValue) : currentGenValue;

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
        label: 'Solar Generation',
        data: predictions.map(p => calculateAdjustedGeneration(p)),
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
        label: 'Load Demand',
        data: predictions.map(() => loadDemandValue),
        borderColor: '#e74c3c',
        borderWidth: 2,
        borderDash: [5, 5],
        tension: 0.4,
        fill: false
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          font: {
            size: 11
          },
          maxRotation: 0,
          minRotation: 0
        }
      },
      y: {
        type: 'linear' as const,
        beginAtZero: true,
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: function(tickValue: string | number) {
            return `${Number(tickValue).toFixed(1)} kW`;
          }
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: (context: any) => {
            const value = context.parsed.y;
            const label = context.dataset.label;
            if (label === 'Solar Generation') {
              const deviation = value - loadDemandValue;
              const status = deviation >= 0 ? 'Excess' : 'Required Dispatch';
              return [
                `${label}: ${value.toFixed(2)} kW`,
                `${status}: ${Math.abs(deviation).toFixed(2)} kW`
              ];
            }
            return `${label}: ${value.toFixed(2)} kW`;
          }
        }
      }
    },
    interaction: {
      intersect: false,
      mode: 'index' as const
    }
  };

  // Add function to get energy status text
  const getEnergyStatus = (solarGen: number, loadDemand: number) => {
    const difference = solarGen - loadDemand;
    return difference >= 0 ? 'Excess Energy' : 'Required Dispatch';
  };

  // Add function to get current GHI value
  const getCurrentGHI = () => {
    if (!predictions.length) return 0;
    const prediction = predictions[0];
    const range = prediction.upper_bound - prediction.lower_bound;
    return prediction.lower_bound + (range * (predictionPercentage / 100));
  };

  // Get selected panel efficiency
  const getSelectedPanelEfficiency = () => {
    if (!powerRating) return 0;
    // For simplicity, just return a fixed value matching the image
    return 20.51;
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Top section - Controls and Graph side by side */}
      <div className="flex gap-6">
        {/* Left side - Power Dispatch Control */}
        <div className="w-1/3 bg-dark-card rounded-xl p-6 shadow-lg border border-opacity-10 border-primary-yellow" style={{ minHeight: '420px' }}>
          <h2 className="text-xl font-bold text-white mb-4">Power Dispatch Control</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-gray-400 mb-2">Load Demand (kW):</label>
              <input
                type="number"
                min="0"
                value={loadDemand}
                onChange={(e) => setLoadDemand(e.target.value)}
                className="w-full p-2 rounded border border-opacity-20 border-white bg-dark-bg text-white focus:outline-none focus:border-primary-yellow"
                placeholder="Enter load demand in kW"
              />
            </div>

            <div>
              <label className="block text-gray-400 mb-2">Average GHI Forecast Interval for the Next Hour:</label>
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

            <div className="p-4 bg-black bg-opacity-40 rounded-lg border border-opacity-30 border-primary-yellow mt-6">
              <h3 className="text-white font-semibold mb-2">Next Hour Power Status:</h3>
              <p className="mb-2 text-white">
                Average GHI: <span className="text-primary-yellow font-medium">{getCurrentGHI().toFixed(2)} W/m²</span>
              </p>
              <p className="mb-2 text-white">
                Solar Generation: <span className="text-cyan-400 font-medium">{currentGenValue.toFixed(3)} kW</span>
              </p>
              {loadDemandValue > 0 ? (
                currentDispatch > 0 ? (
                  <p className="font-bold text-red-400">
                    Required Dispatch: {currentDispatch.toFixed(3)} kW
                  </p>
                ) : (
                  <p className="font-bold text-green-400">
                    Excess Solar: {currentExcess.toFixed(3)} kW
                  </p>
                )
              ) : (
                <p className="font-bold text-green-400">
                  Excess Solar: {currentGenValue.toFixed(3)} kW
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Right side - Graph */}
        <div className="w-2/3 bg-dark-card rounded-xl p-6 border border-opacity-10 border-primary-yellow">
          {loadDemand && parseFloat(loadDemand) > 0 ? (
            <div style={{ height: '350px' }}>
              <Line ref={chartRef} options={{
                ...options,
                scales: {
                  ...options.scales,
                  x: {
                    ...options.scales.x,
                    grid: {
                      ...options.scales.x.grid,
                      color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                      ...options.scales.x.ticks,
                      color: '#E5E5E5'
                    }
                  },
                  y: {
                    ...options.scales.y,
                    grid: {
                      ...options.scales.y.grid,
                      color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                      ...options.scales.y.ticks,
                      color: '#E5E5E5'
                    }
                  }
                },
                plugins: {
                  ...options.plugins,
                  legend: {
                    ...options.plugins.legend,
                    labels: {
                      color: '#E5E5E5'
                    }
                  }
                }
              }} data={chartData} />
            </div>
          ) : (
            <div className="h-[350px] flex items-center justify-center text-gray-400">
              <p>Enter load demand to see power dispatch analysis</p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom section - Breakdown of Calculation full width */}
      {loadDemand && parseFloat(loadDemand) > 0 && (
        <div className="w-full bg-dark-card rounded-xl p-6 border border-opacity-10 border-primary-yellow">
          <h3 className="text-lg font-bold text-white mb-4 text-center">Breakdown of Calculation:</h3>
          <div className="flex justify-between gap-4">
            {[0, 1, 2, 3].map((index) => {
              const prediction = predictions[index];
              const hour = index + 1;
              const solarGen = calculateAdjustedGeneration(prediction);
              const difference = solarGen - loadDemandValue;
              const status = getEnergyStatus(solarGen, loadDemandValue);
              const ghi = (() => {
                const range = prediction.upper_bound - prediction.lower_bound;
                return (prediction.lower_bound + (range * (predictionPercentage / 100))).toFixed(2);
              })();

              return (
                <div 
                  key={index}
                  onClick={() => {
                    const modal = document.getElementById(`dispatch-modal-${index}`);
                    if (modal) {
                      modal.style.display = 'block';
                    }
                  }}
                  className="flex-1 text-center p-3 rounded-lg border-2 border-opacity-40 border-primary-yellow hover:border-opacity-100 bg-black bg-opacity-40 hover:bg-opacity-60 transition-all duration-200 cursor-pointer relative"
                >
                  <div>
                    <p className="text-primary-yellow text-lg font-bold mb-1">
                      Average GHI: {ghi} W/m²
                    </p>
                    <p className="text-cyan-400 text-base font-bold">
                      {hour === 1 ? 'Next' : `${hour}-Hour Ahead`}<br/>
                      {Math.abs(difference).toFixed(3)} kW {status}
                    </p>
                  </div>

                  {/* Modal for detailed calculation */}
                  <div 
                    id={`dispatch-modal-${index}`}
                    className="hidden fixed inset-0 bg-black bg-opacity-80 z-50"
                    onClick={(e) => {
                      if (e.target === e.currentTarget) {
                        const modal = document.getElementById(`dispatch-modal-${index}`);
                        if (modal) {
                          modal.style.display = 'none';
                        }
                      }
                    }}
                  >
                    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-dark-card rounded-xl p-6 max-w-lg w-full shadow-2xl border border-opacity-10 border-primary-yellow">
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          const modal = document.getElementById(`dispatch-modal-${index}`);
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
                        {hour === 1 ? 'Next Hour' : `${hour}-Hour Ahead`} Power Dispatch Details
                      </h3>
                      
                      <div className="space-y-4">
                        <div className="bg-black bg-opacity-40 p-4 rounded-lg border border-opacity-10 border-primary-yellow">
                          <p className="text-lg font-semibold text-primary-yellow mb-2">Solar Generation:</p>
                          <ul className="space-y-2 text-white">
                            <li>• <span className="text-primary-yellow font-medium">Average GHI:</span> <span className="text-primary-yellow">{ghi} W/m²</span></li>
                            <li>• <span className="text-cyan-400 font-medium">Panel Power:</span> {powerRating} W</li>
                            <li>• <span className="text-green-400 font-medium">Number of Panels:</span> {panelCount}</li>
                            <li>• <span className="text-cyan-400 font-medium">Generated Power:</span> <span className="text-cyan-400">{solarGen.toFixed(3)} kW</span></li>
                          </ul>
                        </div>
                        
                        <div className="bg-black bg-opacity-40 p-4 rounded-lg border border-opacity-10 border-primary-yellow">
                          <p className="text-lg font-semibold text-primary-yellow mb-2">Load Analysis:</p>
                          <ul className="space-y-2 text-white">
                            <li>• <span className="text-orange-400 font-medium">Load Demand:</span> <span className="text-orange-400">{loadDemandValue.toFixed(3)} kW</span></li>
                            <li>• <span className="text-cyan-400 font-medium">Solar Generation:</span> <span className="text-cyan-400">{solarGen.toFixed(3)} kW</span></li>
                            <li>• <span className="text-purple-400 font-medium">Difference:</span> <span className="text-purple-400">{difference.toFixed(3)} kW</span></li>
                          </ul>
                        </div>
                        
                        <div className="bg-black bg-opacity-40 p-4 rounded-lg border border-opacity-10 border-primary-yellow">
                          <p className="text-lg font-semibold text-primary-yellow mb-2">Power Status:</p>
                          <div className="space-y-2">
                            <p className={`text-2xl font-bold ${difference >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {Math.abs(difference).toFixed(3)} kW {status}
                            </p>
                            <p className="text-sm text-white mt-2">
                              {difference >= 0 
                                ? "Excess energy can be stored or fed back to the grid"
                                : "Additional power needs to be supplied from the grid or storage"}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default PowerDispatchOptimization; 