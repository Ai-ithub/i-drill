import { useState, useEffect } from 'react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

// Mock data for noise signal
const generateNoiseSignal = (points: number = 100) => {
  return Array.from({ length: points }, (_, i) => ({
    time: i,
    signal: Math.sin(i * 0.1) + Math.random() * 0.5 - 0.25,
    noise: Math.random() * 0.3 - 0.15
  }))
}

// Mock data for histogram
const generateHistogram = () => {
  const bins = 20
  return Array.from({ length: bins }, (_, i) => ({
    bin: ((i - bins/2) * 0.2).toFixed(1),
    frequency: Math.floor(Math.random() * 50 + 10)
  }))
}

export default function SensorPage() {
  const [signalData, setSignalData] = useState(generateNoiseSignal())
  const [histogramData, setHistogramData] = useState(generateHistogram())
  const [isRecording, setIsRecording] = useState(false)

  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        setSignalData(generateNoiseSignal())
        setHistogramData(generateHistogram())
      }, 1000)

      return () => clearInterval(interval)
    }
  }, [isRecording])

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-cyan-400 mb-2">SENSOR MONITORING</h1>
        <p className="text-gray-400">Real-time noise signal analysis and distribution</p>
      </div>

      {/* Control Panel */}
      <div className="bg-gray-800 rounded-lg p-4 mb-6 border border-cyan-500/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsRecording(!isRecording)}
              className={`px-6 py-2 rounded-lg font-semibold transition-all ${
                isRecording
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isRecording ? '⏸ PAUSE' : '▶ START'}
            </button>
            <div className={`flex items-center gap-2 ${isRecording ? 'text-green-400' : 'text-gray-500'}`}>
              <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`}></div>
              <span className="font-mono text-sm">{isRecording ? 'RECORDING' : 'STOPPED'}</span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div className="bg-gray-700 px-4 py-2 rounded">
              <span className="text-gray-400">Sample Rate:</span>
              <span className="ml-2 text-cyan-400 font-mono">1000 Hz</span>
            </div>
            <div className="bg-gray-700 px-4 py-2 rounded">
              <span className="text-gray-400">Buffer:</span>
              <span className="ml-2 text-cyan-400 font-mono">100 pts</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Noise Signal Chart */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <div className="mb-4">
            <h2 className="text-xl font-bold text-cyan-400 mb-1">NOISE SIGNAL</h2>
            <p className="text-gray-400 text-sm">Real-time sensor data with noise</p>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={signalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="time" 
                stroke="#9CA3AF"
                label={{ value: 'Time (ms)', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
              />
              <YAxis 
                stroke="#9CA3AF"
                label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #06B6D4' }}
                labelStyle={{ color: '#06B6D4' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="signal" 
                stroke="#06B6D4" 
                dot={false}
                strokeWidth={2}
                name="Signal"
              />
              <Line 
                type="monotone" 
                dataKey="noise" 
                stroke="#EF4444" 
                dot={false}
                strokeWidth={1}
                name="Noise"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Histogram */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <div className="mb-4">
            <h2 className="text-xl font-bold text-cyan-400 mb-1">DISTRIBUTION HISTOGRAM</h2>
            <p className="text-gray-400 text-sm">Signal amplitude distribution</p>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={histogramData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="bin" 
                stroke="#9CA3AF"
                label={{ value: 'Amplitude Bins', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
              />
              <YAxis 
                stroke="#9CA3AF"
                label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #06B6D4' }}
                labelStyle={{ color: '#06B6D4' }}
              />
              <Bar 
                dataKey="frequency" 
                fill="#06B6D4"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Statistics Panel */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <h2 className="text-xl font-bold text-cyan-400 mb-4">STATISTICS</h2>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-700 rounded p-4">
              <div className="text-gray-400 text-sm mb-1">Mean Value</div>
              <div className="text-2xl font-mono text-cyan-400">
                {(signalData.reduce((sum, d) => sum + d.signal, 0) / signalData.length).toFixed(3)}
              </div>
            </div>

            <div className="bg-gray-700 rounded p-4">
              <div className="text-gray-400 text-sm mb-1">Std Deviation</div>
              <div className="text-2xl font-mono text-cyan-400">0.287</div>
            </div>

            <div className="bg-gray-700 rounded p-4">
              <div className="text-gray-400 text-sm mb-1">Peak-to-Peak</div>
              <div className="text-2xl font-mono text-yellow-400">1.845</div>
            </div>

            <div className="bg-gray-700 rounded p-4">
              <div className="text-gray-400 text-sm mb-1">RMS Value</div>
              <div className="text-2xl font-mono text-cyan-400">0.512</div>
            </div>

            <div className="bg-gray-700 rounded p-4">
              <div className="text-gray-400 text-sm mb-1">SNR</div>
              <div className="text-2xl font-mono text-green-400">24.3 dB</div>
            </div>

            <div className="bg-gray-700 rounded p-4">
              <div className="text-gray-400 text-sm mb-1">Noise Floor</div>
              <div className="text-2xl font-mono text-cyan-400">-45 dB</div>
            </div>
          </div>
        </div>

        {/* FFT Spectrum */}
        <div className="bg-gray-800 rounded-lg p-6 border border-cyan-500/30">
          <div className="mb-4">
            <h2 className="text-xl font-bold text-cyan-400 mb-1">FREQUENCY SPECTRUM</h2>
            <p className="text-gray-400 text-sm">FFT analysis</p>
          </div>

          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={Array.from({ length: 50 }, (_, i) => ({
              freq: i * 10,
              magnitude: Math.exp(-i * 0.1) * (50 + Math.random() * 20)
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="freq" 
                stroke="#9CA3AF"
                label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
              />
              <YAxis 
                stroke="#9CA3AF"
                label={{ value: 'Magnitude (dB)', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #06B6D4' }}
              />
              <Line 
                type="monotone" 
                dataKey="magnitude" 
                stroke="#10B981" 
                dot={false}
                strokeWidth={2}
                fill="url(#colorGradient)"
              />
              <defs>
                <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                </linearGradient>
              </defs>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

