import React, { useState } from 'react';
import { Play, Pause, Square, AlertTriangle } from 'lucide-react';

interface ThresholdConfig {
  parameter: string;
  min: number;
  max: number;
  current: number;
  unit: string;
  critical: boolean;
}

const ControlPage: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [thresholds, setThresholds] = useState<ThresholdConfig[]>([
    { parameter: 'WOB', min: 0, max: 50, current: 32.5, unit: 'klbs', critical: false },
    { parameter: 'RPM', min: 0, max: 200, current: 120, unit: 'rpm', critical: false },
    { parameter: 'Torque', min: 0, max: 40, current: 28.3, unit: 'kft-lbs', critical: false },
    { parameter: 'Pressure', min: 0, max: 5000, current: 3200, unit: 'psi', critical: false },
    { parameter: 'Flow Rate', min: 0, max: 1000, current: 650, unit: 'gpm', critical: false },
    { parameter: 'Temperature', min: 0, max: 300, current: 185, unit: '°F', critical: true },
  ]);

  const handleStart = () => {
    setIsRunning(true);
  };

  const handlePause = () => {
    setIsRunning(false);
  };

  const handleStop = () => {
    setIsRunning(false);
  };

  const updateThreshold = (index: number, field: 'min' | 'max', value: number) => {
    const newThresholds = [...thresholds];
    newThresholds[index][field] = value;
    setThresholds(newThresholds);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-cyan-400 mb-2">CONTROL PANEL</h1>
        <p className="text-gray-400">Drilling operation control and threshold management</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control Buttons Section */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
            <h2 className="text-xl font-bold text-cyan-400 mb-6">RUN CONTROLS</h2>

            {/* Status Display */}
            <div className={`mb-6 p-4 rounded-lg text-center ${
              isRunning ? 'bg-green-900/30 border border-green-500' : 'bg-gray-700/50 border border-gray-600'
            }`}>
              <div className="text-sm text-gray-400 mb-1">Status</div>
              <div className={`text-2xl font-bold ${isRunning ? 'text-green-400' : 'text-gray-400'}`}>
                {isRunning ? 'RUNNING' : 'STOPPED'}
              </div>
              {isRunning && (
                <div className="mt-2">
                  <div className="w-3 h-3 bg-green-400 rounded-full mx-auto animate-pulse"></div>
                </div>
              )}
            </div>

            {/* Control Buttons */}
            <div className="space-y-3">
              <button
                onClick={handleStart}
                disabled={isRunning}
                className={`w-full py-4 px-6 rounded-lg font-bold text-lg flex items-center justify-center gap-3 transition-all ${
                  isRunning
                    ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-green-500/50'
                }`}
              >
                <Play className="w-6 h-6" />
                START
              </button>

              <button
                onClick={handlePause}
                disabled={!isRunning}
                className={`w-full py-4 px-6 rounded-lg font-bold text-lg flex items-center justify-center gap-3 transition-all ${
                  !isRunning
                    ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-yellow-600 hover:bg-yellow-700 text-white shadow-lg hover:shadow-yellow-500/50'
                }`}
              >
                <Pause className="w-6 h-6" />
                PAUSE
              </button>

              <button
                onClick={handleStop}
                disabled={!isRunning}
                className={`w-full py-4 px-6 rounded-lg font-bold text-lg flex items-center justify-center gap-3 transition-all ${
                  !isRunning
                    ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-red-600 hover:bg-red-700 text-white shadow-lg hover:shadow-red-500/50'
                }`}
              >
                <Square className="w-6 h-6" />
                STOP
              </button>
            </div>

            {/* Emergency Stop */}
            <div className="mt-6 pt-6 border-t border-gray-700">
              <button className="w-full py-4 px-6 rounded-lg font-bold text-lg bg-red-700 hover:bg-red-800 text-white border-2 border-red-500 shadow-lg hover:shadow-red-500/50 transition-all flex items-center justify-center gap-3">
                <AlertTriangle className="w-6 h-6" />
                EMERGENCY STOP
              </button>
            </div>

            {/* Runtime Info */}
            <div className="mt-6 space-y-2 text-sm">
              <div className="flex justify-between bg-gray-700 px-3 py-2 rounded">
                <span className="text-gray-400">Runtime:</span>
                <span className="text-cyan-400 font-mono">02:35:42</span>
              </div>
              <div className="flex justify-between bg-gray-700 px-3 py-2 rounded">
                <span className="text-gray-400">Depth:</span>
                <span className="text-cyan-400 font-mono">8,245 ft</span>
              </div>
              <div className="flex justify-between bg-gray-700 px-3 py-2 rounded">
                <span className="text-gray-400">ROP:</span>
                <span className="text-cyan-400 font-mono">125.3 ft/hr</span>
              </div>
            </div>
          </div>
        </div>

        {/* Thresholds Section */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
            <h2 className="text-xl font-bold text-cyan-400 mb-6">THRESHOLD SETTINGS</h2>

            <div className="space-y-4">
              {thresholds.map((threshold, index) => {
                const percentage = ((threshold.current - threshold.min) / (threshold.max - threshold.min)) * 100;
                const isWarning = percentage > 80 || percentage < 20;

                return (
                  <div key={index} className="bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <h3 className="text-lg font-bold text-white">{threshold.parameter}</h3>
                        {threshold.critical && (
                          <span className="px-2 py-1 bg-red-900/50 border border-red-500 rounded text-xs text-red-400 font-semibold">
                            CRITICAL
                          </span>
                        )}
                        {isWarning && !threshold.critical && (
                          <span className="px-2 py-1 bg-yellow-900/50 border border-yellow-500 rounded text-xs text-yellow-400 font-semibold">
                            WARNING
                          </span>
                        )}
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-mono text-cyan-400">
                          {threshold.current.toFixed(1)}
                        </div>
                        <div className="text-xs text-gray-400">{threshold.unit}</div>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="relative h-8 bg-gray-800 rounded-lg overflow-hidden mb-3">
                      <div
                        className={`absolute h-full transition-all duration-300 ${
                          isWarning ? 'bg-yellow-500' : 'bg-cyan-500'
                        }`}
                        style={{ width: `${percentage}%` }}
                      ></div>
                      <div className="absolute inset-0 flex items-center justify-center text-xs font-mono text-white font-bold">
                        {percentage.toFixed(1)}%
                      </div>
                    </div>

                    {/* Min/Max Controls */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Min Threshold</label>
                        <div className="flex items-center gap-2">
                          <input
                            type="number"
                            value={threshold.min}
                            onChange={(e) => updateThreshold(index, 'min', parseFloat(e.target.value))}
                            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white font-mono text-sm focus:outline-none focus:border-cyan-500"
                          />
                          <span className="text-xs text-gray-400 whitespace-nowrap">{threshold.unit}</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Max Threshold</label>
                        <div className="flex items-center gap-2">
                          <input
                            type="number"
                            value={threshold.max}
                            onChange={(e) => updateThreshold(index, 'max', parseFloat(e.target.value))}
                            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white font-mono text-sm focus:outline-none focus:border-cyan-500"
                          />
                          <span className="text-xs text-gray-400 whitespace-nowrap">{threshold.unit}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Action Buttons */}
            <div className="mt-6 flex gap-3">
              <button className="flex-1 px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white font-semibold rounded-lg transition-all">
                Apply Changes
              </button>
              <button className="flex-1 px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-lg transition-all">
                Reset to Defaults
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPage;

