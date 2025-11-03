import { useState, useEffect } from 'react'
import { CircularGauge, LinearGauge } from '@/components/Gauges'

interface DrillMetrics {
  rpm: number
  torque: number
  pressure: number
  wob: number
  rop: number
  temperature: number
}

export default function RPMPage() {
  const [metrics, setMetrics] = useState<DrillMetrics>({
    rpm: 120,
    torque: 28.5,
    pressure: 3200,
    wob: 32.5,
    rop: 125.3,
    temperature: 185
  })

  const [isLive, setIsLive] = useState(false)

  useEffect(() => {
    if (isLive) {
      const interval = setInterval(() => {
        setMetrics({
          rpm: 100 + Math.random() * 80,
          torque: 20 + Math.random() * 20,
          pressure: 2800 + Math.random() * 800,
          wob: 25 + Math.random() * 20,
          rop: 100 + Math.random() * 50,
          temperature: 170 + Math.random() * 30
        })
      }, 1000)

      return () => clearInterval(interval)
    }
  }, [isLive])

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-cyan-400 mb-2">RPM & DRILLING PARAMETERS</h1>
        <p className="text-gray-400">Real-time monitoring of drilling performance metrics</p>
      </div>

      {/* Control Bar */}
      <div className="bg-gray-800 rounded-lg p-4 mb-6 border border-cyan-500/30">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setIsLive(!isLive)}
            className={`px-6 py-2 rounded-lg font-semibold transition-all ${
              isLive
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isLive ? '⏸ PAUSE' : '▶ LIVE DATA'}
          </button>

          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 ${isLive ? 'text-green-400' : 'text-gray-500'}`}>
              <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`}></div>
              <span className="font-mono text-sm">{isLive ? 'LIVE' : 'PAUSED'}</span>
            </div>

            <div className="bg-gray-700 px-4 py-2 rounded">
              <span className="text-gray-400">Update Rate:</span>
              <span className="ml-2 text-cyan-400 font-mono">1 Hz</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Gauges Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
        {/* RPM Gauge */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <CircularGauge
            value={metrics.rpm}
            min={0}
            max={200}
            label="RPM"
            unit="rpm"
            color="#06B6D4"
          />
          <div className="mt-4 text-center">
            <div className="text-sm text-gray-400">Rotary Speed</div>
            <div className="text-3xl font-mono text-cyan-400 mt-1">
              {metrics.rpm.toFixed(1)}
            </div>
            <div className="text-sm text-gray-500">rpm</div>
          </div>
        </div>

        {/* Torque Gauge */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <CircularGauge
            value={metrics.torque}
            min={0}
            max={40}
            label="TORQUE"
            unit="kft-lbs"
            color="#10B981"
          />
          <div className="mt-4 text-center">
            <div className="text-sm text-gray-400">Rotary Torque</div>
            <div className="text-3xl font-mono text-green-400 mt-1">
              {metrics.torque.toFixed(1)}
            </div>
            <div className="text-sm text-gray-500">kft-lbs</div>
          </div>
        </div>

        {/* Pressure Gauge */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <CircularGauge
            value={metrics.pressure}
            min={0}
            max={5000}
            label="PRESSURE"
            unit="psi"
            color="#F59E0B"
          />
          <div className="mt-4 text-center">
            <div className="text-sm text-gray-400">Pump Pressure</div>
            <div className="text-3xl font-mono text-yellow-400 mt-1">
              {metrics.pressure.toFixed(0)}
            </div>
            <div className="text-sm text-gray-500">psi</div>
          </div>
        </div>
      </div>

      {/* Linear Gauges */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* WOB Linear Gauge */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <h3 className="text-lg font-bold text-cyan-400 mb-4">WEIGHT ON BIT (WOB)</h3>
          <LinearGauge
            value={metrics.wob}
            min={0}
            max={50}
            label="WOB"
            unit="klbs"
            color="#8B5CF6"
          />
          <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
            <div className="bg-gray-700 rounded p-2 text-center">
              <div className="text-gray-400">Current</div>
              <div className="text-cyan-400 font-mono">{metrics.wob.toFixed(1)} klbs</div>
            </div>
            <div className="bg-gray-700 rounded p-2 text-center">
              <div className="text-gray-400">Target</div>
              <div className="text-green-400 font-mono">35.0 klbs</div>
            </div>
            <div className="bg-gray-700 rounded p-2 text-center">
              <div className="text-gray-400">Max</div>
              <div className="text-red-400 font-mono">50.0 klbs</div>
            </div>
          </div>
        </div>

        {/* ROP Linear Gauge */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <h3 className="text-lg font-bold text-cyan-400 mb-4">RATE OF PENETRATION (ROP)</h3>
          <LinearGauge
            value={metrics.rop}
            min={0}
            max={200}
            label="ROP"
            unit="ft/hr"
            color="#EC4899"
          />
          <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
            <div className="bg-gray-700 rounded p-2 text-center">
              <div className="text-gray-400">Current</div>
              <div className="text-cyan-400 font-mono">{metrics.rop.toFixed(1)} ft/hr</div>
            </div>
            <div className="bg-gray-700 rounded p-2 text-center">
              <div className="text-gray-400">Average</div>
              <div className="text-green-400 font-mono">118.5 ft/hr</div>
            </div>
            <div className="bg-gray-700 rounded p-2 text-center">
              <div className="text-gray-400">Peak</div>
              <div className="text-yellow-400 font-mono">145.2 ft/hr</div>
            </div>
          </div>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Temperature */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <h3 className="text-lg font-bold text-cyan-400 mb-4">TEMPERATURE</h3>
          <div className="relative">
            <div className="h-48 bg-gray-700 rounded-lg relative overflow-hidden">
              <div
                className="absolute bottom-0 w-full bg-gradient-to-t from-red-600 to-yellow-500 transition-all duration-300"
                style={{ height: `${(metrics.temperature / 300) * 100}%` }}
              ></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center bg-gray-900/80 px-4 py-2 rounded">
                  <div className="text-3xl font-mono text-white font-bold">
                    {metrics.temperature.toFixed(1)}°F
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 flex justify-between text-sm">
            <div className="text-gray-400">Min: 32°F</div>
            <div className="text-red-400">Max: 300°F</div>
          </div>
        </div>

        {/* Performance Indicators */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <h3 className="text-lg font-bold text-cyan-400 mb-4">PERFORMANCE INDICATORS</h3>
          <div className="space-y-3">
            <div className="bg-gray-700 rounded p-3">
              <div className="flex justify-between items-center mb-1">
                <span className="text-gray-400 text-sm">Mechanical Efficiency</span>
                <span className="text-green-400 font-mono">87%</span>
              </div>
              <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
                <div className="h-full bg-green-500" style={{ width: '87%' }}></div>
              </div>
            </div>

            <div className="bg-gray-700 rounded p-3">
              <div className="flex justify-between items-center mb-1">
                <span className="text-gray-400 text-sm">Hydraulic Efficiency</span>
                <span className="text-cyan-400 font-mono">92%</span>
              </div>
              <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-500" style={{ width: '92%' }}></div>
              </div>
            </div>

            <div className="bg-gray-700 rounded p-3">
              <div className="flex justify-between items-center mb-1">
                <span className="text-gray-400 text-sm">MSE (Mechanical Specific Energy)</span>
                <span className="text-yellow-400 font-mono">345 ksi</span>
              </div>
              <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
                <div className="h-full bg-yellow-500" style={{ width: '65%' }}></div>
              </div>
            </div>

            <div className="bg-gray-700 rounded p-3">
              <div className="flex justify-between items-center mb-1">
                <span className="text-gray-400 text-sm">Drilling Optimization</span>
                <span className="text-purple-400 font-mono">78%</span>
              </div>
              <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
                <div className="h-full bg-purple-500" style={{ width: '78%' }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Alarms & Warnings */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <h3 className="text-lg font-bold text-cyan-400 mb-4">ALARMS & WARNINGS</h3>
          <div className="space-y-2">
            <div className="bg-green-900/30 border border-green-500 rounded p-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-green-400 text-sm font-semibold">All Systems Normal</span>
              </div>
            </div>

            <div className="bg-gray-700 rounded p-3 opacity-50">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                <span className="text-gray-400 text-sm">No Active Alarms</span>
              </div>
            </div>

            <div className="bg-gray-700 rounded p-3 opacity-50">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                <span className="text-gray-400 text-sm">No Warnings</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="text-xs text-gray-400">Last Check</div>
            <div className="text-sm text-cyan-400 font-mono">
              {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
