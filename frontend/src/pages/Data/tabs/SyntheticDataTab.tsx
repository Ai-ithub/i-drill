import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Play, Pause, RefreshCw, Download, Settings, Radio, Database, Gauge, Thermometer } from 'lucide-react'

interface SyntheticDataPoint {
  timestamp: string
  // PLC/SCADA
  depth: number
  wob: number
  rpm: number
  torque: number
  rop: number
  mud_pressure: number
  mud_flow: number
  mud_temperature: number
  mud_density: number
  mud_viscosity: number
  hook_load: number
  standpipe_pressure: number
  casing_pressure: number
  annulus_pressure: number
  pump_pressure: number
  bit_temperature: number
  motor_temperature: number
  vibration_level: number
  power_consumption: number
  // LWD/MWD
  gamma_ray: number
  resistivity: number
  density: number
  porosity: number
  neutron_porosity: number
  sonic: number
  caliper: number
  temperature_lwd: number
}

// PLC/SCADA Parameters
const PLC_METRICS = [
  { key: 'depth', label: 'Depth', unit: 'ft', color: '#3b82f6', min: 0, max: 10000 },
  { key: 'wob', label: 'WOB', unit: 'lbs', color: '#10b981', min: 0, max: 50000 },
  { key: 'rpm', label: 'RPM', unit: 'rpm', color: '#eab308', min: 0, max: 300 },
  { key: 'torque', label: 'Torque', unit: 'ft-lbs', color: '#f97316', min: 0, max: 50000 },
  { key: 'rop', label: 'ROP', unit: 'ft/hr', color: '#a855f7', min: 0, max: 100 },
  { key: 'mud_pressure', label: 'Mud Pressure', unit: 'psi', color: '#ef4444', min: 0, max: 5000 },
  { key: 'mud_flow', label: 'Mud Flow', unit: 'gpm', color: '#06b6d4', min: 0, max: 2000 },
  { key: 'mud_temperature', label: 'Mud Temperature', unit: '°C', color: '#ec4899', min: 20, max: 150 },
  { key: 'mud_density', label: 'Mud Density', unit: 'ppg', color: '#8b5cf6', min: 8, max: 20 },
  { key: 'mud_viscosity', label: 'Mud Viscosity', unit: 'cP', color: '#6366f1', min: 1, max: 100 },
  { key: 'hook_load', label: 'Hook Load', unit: 'lbs', color: '#3b82f6', min: 100000, max: 500000 },
  { key: 'standpipe_pressure', label: 'Standpipe Pressure', unit: 'psi', color: '#0ea5e9', min: 1000, max: 5000 },
  { key: 'casing_pressure', label: 'Casing Pressure', unit: 'psi', color: '#14b8a6', min: 0, max: 3000 },
  { key: 'annulus_pressure', label: 'Annulus Pressure', unit: 'psi', color: '#10b981', min: 0, max: 3000 },
  { key: 'pump_pressure', label: 'Pump Pressure', unit: 'psi', color: '#8b5cf6', min: 1000, max: 5000 },
  { key: 'bit_temperature', label: 'Bit Temperature', unit: '°C', color: '#ef4444', min: 50, max: 200 },
  { key: 'motor_temperature', label: 'Motor Temperature', unit: '°C', color: '#f59e0b', min: 40, max: 120 },
  { key: 'vibration_level', label: 'Vibration', unit: 'g', color: '#f43f5e', min: 0, max: 10 },
  { key: 'power_consumption', label: 'Power Consumption', unit: 'kW', color: '#eab308', min: 0, max: 5000 },
]

// LWD/MWD Parameters
const LWD_METRICS = [
  { key: 'gamma_ray', label: 'Gamma Ray', unit: 'API', color: '#fbbf24', min: 20, max: 150 },
  { key: 'resistivity', label: 'Resistivity', unit: 'ohm-m', color: '#34d399', min: 0.2, max: 200 },
  { key: 'density', label: 'Density', unit: 'g/cc', color: '#60a5fa', min: 1.5, max: 3.0 },
  { key: 'porosity', label: 'Porosity', unit: '%', color: '#a78bfa', min: 5, max: 25 },
  { key: 'neutron_porosity', label: 'Neutron Porosity', unit: '%', color: '#f472b6', min: 5, max: 30 },
  { key: 'sonic', label: 'Sonic', unit: 'μs/ft', color: '#22d3ee', min: 40, max: 200 },
  { key: 'caliper', label: 'Caliper', unit: 'in', color: '#fb923c', min: 6, max: 12 },
  { key: 'temperature_lwd', label: 'LWD Temperature', unit: '°C', color: '#f87171', min: 50, max: 200 },
]

const ALL_METRICS = [...PLC_METRICS, ...LWD_METRICS]

// Generate synthetic data with realistic patterns
function generateSyntheticData(baseTime: Date, index: number): SyntheticDataPoint {
  const time = new Date(baseTime.getTime() + index * 1000) // 1 second intervals
  const t = index / 100 // Time factor for oscillations

  // PLC/SCADA data generation
  const depth = 1000 + index * 0.5 + Math.sin(t * 0.1) * 10
  const wob = 20000 + Math.sin(t * 0.2) * 5000 + Math.random() * 2000
  const rpm = 150 + Math.sin(t * 0.15) * 30 + Math.random() * 10
  const torque = 25000 + Math.sin(t * 0.25) * 5000 + Math.random() * 2000
  const rop = 20 + Math.sin(t * 0.3) * 5 + Math.random() * 2
  const mud_pressure = 2000 + Math.sin(t * 0.18) * 300 + Math.random() * 100
  const mud_flow = 1000 + Math.sin(t * 0.12) * 200 + Math.random() * 50
  const mud_temperature = 60 + Math.sin(t * 0.08) * 10 + Math.random() * 5
  const mud_density = 12 + Math.sin(t * 0.05) * 1 + Math.random() * 0.5
  const mud_viscosity = 30 + Math.sin(t * 0.1) * 10 + Math.random() * 5
  const hook_load = 200000 + wob + Math.sin(t * 0.1) * 20000 + Math.random() * 10000
  const standpipe_pressure = mud_pressure + Math.random() * 200
  const casing_pressure = 500 + Math.sin(t * 0.15) * 100 + Math.random() * 50
  const annulus_pressure = casing_pressure + Math.random() * 50
  const pump_pressure = standpipe_pressure + Math.random() * 300
  const bit_temperature = 100 + Math.sin(t * 0.2) * 20 + Math.random() * 10
  const motor_temperature = 70 + Math.sin(t * 0.15) * 15 + Math.random() * 5
  const vibration_level = 2 + Math.sin(t * 0.4) * 1 + Math.random() * 0.5
  const power_consumption = 2000 + Math.sin(t * 0.2) * 500 + Math.random() * 200

  // LWD/MWD data generation
  const gamma_ray = 50 + Math.sin(t * 0.3) * 30 + Math.random() * 10
  const resistivity = 10 + Math.sin(t * 0.25) * 5 + Math.random() * 2
  const density = 2.2 + Math.sin(t * 0.2) * 0.3 + Math.random() * 0.1
  const porosity = 15 + Math.sin(t * 0.35) * 5 + Math.random() * 2
  const neutron_porosity = porosity + Math.random() * 3
  const sonic = 80 + Math.sin(t * 0.3) * 20 + Math.random() * 5
  const caliper = 8.5 + Math.sin(t * 0.1) * 0.5 + Math.random() * 0.2
  const temperature_lwd = bit_temperature + Math.random() * 10

  return {
    timestamp: time.toISOString(),
    // PLC/SCADA
    depth: Math.max(0, Math.min(10000, depth)),
    wob: Math.max(0, Math.min(50000, wob)),
    rpm: Math.max(0, Math.min(300, rpm)),
    torque: Math.max(0, Math.min(50000, torque)),
    rop: Math.max(0, Math.min(100, rop)),
    mud_pressure: Math.max(0, Math.min(5000, mud_pressure)),
    mud_flow: Math.max(0, Math.min(2000, mud_flow)),
    mud_temperature: Math.max(20, Math.min(150, mud_temperature)),
    mud_density: Math.max(8, Math.min(20, mud_density)),
    mud_viscosity: Math.max(1, Math.min(100, mud_viscosity)),
    hook_load: Math.max(100000, Math.min(500000, hook_load)),
    standpipe_pressure: Math.max(1000, Math.min(5000, standpipe_pressure)),
    casing_pressure: Math.max(0, Math.min(3000, casing_pressure)),
    annulus_pressure: Math.max(0, Math.min(3000, annulus_pressure)),
    pump_pressure: Math.max(1000, Math.min(5000, pump_pressure)),
    bit_temperature: Math.max(50, Math.min(200, bit_temperature)),
    motor_temperature: Math.max(40, Math.min(120, motor_temperature)),
    vibration_level: Math.max(0, Math.min(10, vibration_level)),
    power_consumption: Math.max(0, Math.min(5000, power_consumption)),
    // LWD/MWD
    gamma_ray: Math.max(20, Math.min(150, gamma_ray)),
    resistivity: Math.max(0.2, Math.min(200, resistivity)),
    density: Math.max(1.5, Math.min(3.0, density)),
    porosity: Math.max(5, Math.min(25, porosity)),
    neutron_porosity: Math.max(5, Math.min(30, neutron_porosity)),
    sonic: Math.max(40, Math.min(200, sonic)),
    caliper: Math.max(6, Math.min(12, caliper)),
    temperature_lwd: Math.max(50, Math.min(200, temperature_lwd)),
  }
}

export default function SyntheticDataTab() {
  const [isPlaying, setIsPlaying] = useState(true)
  const [dataPoints, setDataPoints] = useState<SyntheticDataPoint[]>([])
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['depth', 'wob', 'rpm', 'rop', 'gamma_ray', 'resistivity'])
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'plc' | 'lwd'>('all')
  const [maxDataPoints] = useState(200)
  const [updateInterval, setUpdateInterval] = useState(1000) // milliseconds
  const [baseTime] = useState(new Date())

  // Generate initial data
  useEffect(() => {
    const initialData: SyntheticDataPoint[] = []
    for (let i = 0; i < maxDataPoints; i++) {
      initialData.push(generateSyntheticData(baseTime, i))
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
          lastIndex
        )

        const updated = [...prev, newPoint]
        return updated.slice(-maxDataPoints)
      })
    }, updateInterval)

    return () => clearInterval(interval)
  }, [isPlaying, updateInterval, maxDataPoints, baseTime])

  const latestData = dataPoints.length > 0 ? dataPoints[dataPoints.length - 1] : null

  const displayedMetrics = useMemo(() => {
    if (selectedCategory === 'all') return ALL_METRICS
    if (selectedCategory === 'plc') return PLC_METRICS
    return LWD_METRICS
  }, [selectedCategory])

  const statsCards = displayedMetrics
    .filter((m) => selectedMetrics.includes(m.key))
    .slice(0, 8)
    .map((metric) => ({
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
      initialData.push(generateSyntheticData(newBaseTime, i))
    }
    setDataPoints(initialData)
  }

  const handleExport = () => {
    if (!dataPoints.length) {
      alert('No data available for export')
      return
    }

    const csv = [
      ['Timestamp', ...ALL_METRICS.map((m) => m.label)].join(','),
      ...dataPoints.map((item) =>
        [
          item.timestamp,
          ...ALL_METRICS.map((m) => item[m.key as keyof SyntheticDataPoint] || ''),
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
          Generate and visualize synthetic data from all LWD/MWD sensors and PLC/SCADA systems for testing and simulation
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

        {/* Category Filter */}
        <div className="mt-4 space-y-2">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
            Sensor Category
          </label>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setSelectedCategory('all')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedCategory === 'all'
                  ? 'bg-cyan-500 text-white'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}
            >
              All Sensors
            </button>
            <button
              type="button"
              onClick={() => setSelectedCategory('plc')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedCategory === 'plc'
                  ? 'bg-cyan-500 text-white'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}
            >
              PLC/SCADA
            </button>
            <button
              type="button"
              onClick={() => setSelectedCategory('lwd')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedCategory === 'lwd'
                  ? 'bg-cyan-500 text-white'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}
            >
              LWD/MWD
            </button>
          </div>
        </div>

        {/* Metric Selection */}
        <div className="mt-4 space-y-2">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
            Select Metrics to Display
          </label>
          <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto">
            {displayedMetrics.map((metric) => (
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
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8 gap-4">
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
          const metric = ALL_METRICS.find((m) => m.key === metricKey)
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
                    const metric = ALL_METRICS.find((m) => m.key === name)
                    return [`${value?.toFixed(2) || 0} ${metric?.unit || ''}`, metric?.label || name]
                  }}
                />
                <Legend />
                {selectedMetrics.map((metricKey) => {
                  const metric = ALL_METRICS.find((m) => m.key === metricKey)
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
            Update rate: {updateInterval}ms | Status: {isPlaying ? 'Active' : 'Paused'} | 
            Sensors: {ALL_METRICS.length} (PLC/SCADA: {PLC_METRICS.length}, LWD/MWD: {LWD_METRICS.length})
          </span>
        </div>
      </div>
    </div>
  )
}
