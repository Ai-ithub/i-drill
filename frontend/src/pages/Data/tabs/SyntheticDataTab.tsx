import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Play, Pause, RefreshCw, Download, Settings } from 'lucide-react'

interface SyntheticDataPoint {
  timestamp: string
  depth: number
  wob: number
  rpm: number
  torque: number
  rop: number
  mud_pressure: number
  mud_flow: number
  temperature: number
  vibration: number
}

const METRICS = [
  { key: 'depth', label: 'Depth', unit: 'm', color: '#3b82f6', min: 0, max: 5000 },
  { key: 'wob', label: 'Weight on Bit', unit: 'kN', color: '#10b981', min: 0, max: 500 },
  { key: 'rpm', label: 'Rotary Speed', unit: 'RPM', color: '#eab308', min: 0, max: 300 },
  { key: 'torque', label: 'Torque', unit: 'N·m', color: '#f97316', min: 0, max: 50000 },
  { key: 'rop', label: 'Rate of Penetration', unit: 'm/h', color: '#a855f7', min: 0, max: 50 },
  { key: 'mud_pressure', label: 'Mud Pressure', unit: 'bar', color: '#ef4444', min: 0, max: 500 },
  { key: 'mud_flow', label: 'Mud Flow', unit: 'L/min', color: '#06b6d4', min: 0, max: 2000 },
  { key: 'temperature', label: 'Temperature', unit: '°C', color: '#ec4899', min: 20, max: 150 },
  { key: 'vibration', label: 'Vibration', unit: 'g', color: '#8b5cf6', min: 0, max: 10 },
]

// Generate synthetic data with realistic patterns
function generateSyntheticData(baseTime: Date, index: number, metrics: typeof METRICS): SyntheticDataPoint {
  const time = new Date(baseTime.getTime() + index * 1000) // 1 second intervals
  const t = index / 100 // Time factor for oscillations

  // Generate realistic drilling patterns
  const depth = 1000 + index * 0.5 + Math.sin(t * 0.1) * 10
  const wob = 200 + Math.sin(t * 0.2) * 50 + Math.random() * 20
  const rpm = 150 + Math.sin(t * 0.15) * 30 + Math.random() * 10
  const torque = 25000 + Math.sin(t * 0.25) * 5000 + Math.random() * 2000
  const rop = 20 + Math.sin(t * 0.3) * 5 + Math.random() * 2
  const mud_pressure = 200 + Math.sin(t * 0.18) * 30 + Math.random() * 10
  const mud_flow = 1000 + Math.sin(t * 0.12) * 200 + Math.random() * 50
  const temperature = 60 + Math.sin(t * 0.08) * 10 + Math.random() * 5
  const vibration = 2 + Math.sin(t * 0.4) * 1 + Math.random() * 0.5

  return {
    timestamp: time.toISOString(),
    depth: Math.max(0, Math.min(5000, depth)),
    wob: Math.max(0, Math.min(500, wob)),
    rpm: Math.max(0, Math.min(300, rpm)),
    torque: Math.max(0, Math.min(50000, torque)),
    rop: Math.max(0, Math.min(50, rop)),
    mud_pressure: Math.max(0, Math.min(500, mud_pressure)),
    mud_flow: Math.max(0, Math.min(2000, mud_flow)),
    temperature: Math.max(20, Math.min(150, temperature)),
    vibration: Math.max(0, Math.min(10, vibration)),
  }
}

export default function SyntheticDataTab() {
  const [isPlaying, setIsPlaying] = useState(true)
  const [dataPoints, setDataPoints] = useState<SyntheticDataPoint[]>([])
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['depth', 'wob', 'rpm', 'rop'])
  const [maxDataPoints] = useState(200)
  const [updateInterval, setUpdateInterval] = useState(1000) // milliseconds
  const [baseTime] = useState(new Date())

  // Generate initial data
  useEffect(() => {
    const initialData: SyntheticDataPoint[] = []
    for (let i = 0; i < maxDataPoints; i++) {
      initialData.push(generateSyntheticData(baseTime, i, METRICS))
    }
    setDataPoints(initialData)
  }, [maxDataPoints, baseTime])

  // Update data at intervals
  useEffect(() => {
    if (!isPlaying) return

    const interval = setInterval(() => {
      setDataPoints((prev) => {
        const lastPoint = prev[prev.length - 1]
        const lastIndex = prev.length
        const newPoint = generateSyntheticData(
          lastPoint ? new Date(lastPoint.timestamp) : baseTime,
          lastIndex,
          METRICS
        )

        const updated = [...prev, newPoint]
        return updated.slice(-maxDataPoints)
      })
    }, updateInterval)

    return () => clearInterval(interval)
  }, [isPlaying, updateInterval, maxDataPoints, baseTime])

  const latestData = dataPoints.length > 0 ? dataPoints[dataPoints.length - 1] : null

  const statsCards = METRICS.filter((m) => selectedMetrics.includes(m.key)).map((metric) => ({
    label: metric.label,
    value: latestData ? (latestData[metric.key as keyof SyntheticDataPoint] as number).toFixed(2) : '--',
    unit: metric.unit,
    color: metric.color,
  }))

  const handleMetricToggle = (metric: string) => {
    setSelectedMetrics((prev) =>
      prev.includes(metric) ? prev.filter((m) => m !== metric) : [...prev, metric]
    )
  }

  const handleReset = () => {
    const newBaseTime = new Date()
    const initialData: SyntheticDataPoint[] = []
    for (let i = 0; i < maxDataPoints; i++) {
      initialData.push(generateSyntheticData(newBaseTime, i, METRICS))
    }
    setDataPoints(initialData)
  }

  const handleExport = () => {
    if (!dataPoints.length) {
      alert('No data available for export')
      return
    }

    const csv = [
      ['Timestamp', ...METRICS.map((m) => m.label)].join(','),
      ...dataPoints.map((item) =>
        [
          item.timestamp,
          ...METRICS.map((m) => item[m.key as keyof SyntheticDataPoint] || ''),
        ].join(',')
      ),
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `synthetic_data_${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">Synthetic Data Generator</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Generate and visualize synthetic drilling data for testing and simulation
        </p>
      </div>

      {/* Controls */}
      <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-600 text-white font-medium transition-colors"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>

            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-medium transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Reset
            </button>

            <button
              onClick={handleExport}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-medium transition-colors"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm text-slate-700 dark:text-slate-300">Update Interval (ms):</label>
            <input
              type="number"
              min={100}
              max={10000}
              step={100}
              value={updateInterval}
              onChange={(e) => setUpdateInterval(Number(e.target.value))}
              className="w-24 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm focus:border-cyan-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Metric Selection */}
        <div className="mt-4 space-y-2">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
            Select Metrics to Display
          </label>
          <div className="flex flex-wrap gap-2">
            {METRICS.map((metric) => (
              <button
                key={metric.key}
                type="button"
                onClick={() => handleMetricToggle(metric.key)}
                className={`
                  px-4 py-2 rounded-lg text-sm font-medium transition-colors
                  ${
                    selectedMetrics.includes(metric.key)
                      ? 'bg-cyan-500 text-white'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }
                `}
              >
                {metric.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {statsCards.map((stat, index) => (
          <div
            key={index}
            className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-4 shadow-sm"
          >
            <div className="flex items-center justify-between mb-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: stat.color }}
              />
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{stat.label}</div>
            <div className="flex items-baseline gap-1">
              <div className="text-lg font-bold text-slate-900 dark:text-white">{stat.value}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400">{stat.unit}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {selectedMetrics.map((metricKey) => {
          const metric = METRICS.find((m) => m.key === metricKey)
          if (!metric) return null

          return (
            <div
              key={metricKey}
              className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm"
            >
              <h3 className="text-lg font-semibold mb-4">{metric.label}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dataPoints}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
                  <XAxis
                    dataKey="timestamp"
                    stroke="#64748b"
                    tick={{ fill: '#64748b', fontSize: 12 }}
                    tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                  />
                  <YAxis
                    stroke="#64748b"
                    tick={{ fill: '#64748b', fontSize: 12 }}
                    label={{ value: metric.unit, angle: -90, position: 'insideLeft' }}
                    domain={[metric.min, metric.max]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: number) => [`${value?.toFixed(2)} ${metric.unit}`, metric.label]}
                  />
                  <Line
                    type="monotone"
                    dataKey={metricKey}
                    stroke={metric.color}
                    strokeWidth={2}
                    dot={false}
                    name={metric.label}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )
        })}
      </div>

      {/* Combined Chart */}
      {selectedMetrics.length > 1 && (
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Combined View</h3>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dataPoints} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: number, name: string) => {
                    const metric = METRICS.find((m) => m.key === name)
                    return [`${value?.toFixed(2) || 0} ${metric?.unit || ''}`, metric?.label || name]
                  }}
                />
                <Legend />
                {selectedMetrics.map((metricKey) => {
                  const metric = METRICS.find((m) => m.key === metricKey)
                  if (!metric) return null
                  return (
                    <Line
                      key={metricKey}
                      type="monotone"
                      dataKey={metricKey}
                      stroke={metric.color}
                      strokeWidth={2}
                      dot={false}
                      name={metric.label}
                    />
                  )
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Data Info */}
      <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-4">
        <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
          <Settings className="w-4 h-4" />
          <span>
            Generating synthetic data with realistic drilling patterns. Data points: {dataPoints.length} | 
            Update rate: {updateInterval}ms | Status: {isPlaying ? 'Active' : 'Paused'}
          </span>
        </div>
      </div>
    </div>
  )
}


