import { useState, useMemo, FormEvent } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { sensorDataApi } from '@/services/api'
import { Calendar, Filter, Download, Loader2, Radio, Database, Gauge, Thermometer } from 'lucide-react'

// PLC/SCADA Parameters
const PLC_METRICS = [
  { key: 'depth', label: 'Depth', unit: 'ft', color: '#3b82f6', icon: 'TrendingUp' },
  { key: 'wob', label: 'WOB', unit: 'lbs', color: '#10b981', icon: 'Activity' },
  { key: 'rpm', label: 'RPM', unit: 'rpm', color: '#eab308', icon: 'Zap' },
  { key: 'torque', label: 'Torque', unit: 'ft-lbs', color: '#f97316', icon: 'Gauge' },
  { key: 'rop', label: 'ROP', unit: 'ft/hr', color: '#a855f7', icon: 'Activity' },
  { key: 'mud_pressure', label: 'Mud Pressure', unit: 'psi', color: '#ef4444', icon: 'Gauge' },
  { key: 'mud_flow', label: 'Mud Flow', unit: 'gpm', color: '#06b6d4', icon: 'Activity' },
  { key: 'mud_temperature', label: 'Mud Temperature', unit: '°C', color: '#ec4899', icon: 'Thermometer' },
  { key: 'mud_density', label: 'Mud Density', unit: 'ppg', color: '#8b5cf6', icon: 'Database' },
  { key: 'mud_viscosity', label: 'Mud Viscosity', unit: 'cP', color: '#6366f1', icon: 'Gauge' },
  { key: 'hook_load', label: 'Hook Load', unit: 'lbs', color: '#3b82f6', icon: 'Activity' },
  { key: 'standpipe_pressure', label: 'Standpipe Pressure', unit: 'psi', color: '#0ea5e9', icon: 'Gauge' },
  { key: 'casing_pressure', label: 'Casing Pressure', unit: 'psi', color: '#14b8a6', icon: 'Gauge' },
  { key: 'annulus_pressure', label: 'Annulus Pressure', unit: 'psi', color: '#10b981', icon: 'Gauge' },
  { key: 'pump_pressure', label: 'Pump Pressure', unit: 'psi', color: '#8b5cf6', icon: 'Gauge' },
  { key: 'bit_temperature', label: 'Bit Temperature', unit: '°C', color: '#ef4444', icon: 'Thermometer' },
  { key: 'motor_temperature', label: 'Motor Temperature', unit: '°C', color: '#f59e0b', icon: 'Thermometer' },
  { key: 'vibration_level', label: 'Vibration', unit: 'g', color: '#f43f5e', icon: 'Activity' },
  { key: 'power_consumption', label: 'Power Consumption', unit: 'kW', color: '#eab308', icon: 'Zap' },
  { key: 'pump_status', label: 'Pump Status', unit: '', color: '#6366f1', icon: 'Activity' },
  { key: 'compressor_status', label: 'Compressor Status', unit: '', color: '#8b5cf6', icon: 'Activity' },
]

// LWD/MWD Parameters
const LWD_METRICS = [
  { key: 'gamma_ray', label: 'Gamma Ray', unit: 'API', color: '#fbbf24', icon: 'Radio' },
  { key: 'resistivity', label: 'Resistivity', unit: 'ohm-m', color: '#34d399', icon: 'Radio' },
  { key: 'density', label: 'Density', unit: 'g/cc', color: '#60a5fa', icon: 'Database' },
  { key: 'porosity', label: 'Porosity', unit: '%', color: '#a78bfa', icon: 'Database' },
  { key: 'neutron_porosity', label: 'Neutron Porosity', unit: '%', color: '#f472b6', icon: 'Radio' },
  { key: 'sonic', label: 'Sonic', unit: 'μs/ft', color: '#22d3ee', icon: 'Radio' },
  { key: 'caliper', label: 'Caliper', unit: 'in', color: '#fb923c', icon: 'Gauge' },
  { key: 'temperature_lwd', label: 'LWD Temperature', unit: '°C', color: '#f87171', icon: 'Thermometer' },
]

const ALL_METRICS = [...PLC_METRICS, ...LWD_METRICS]

export default function HistoricalDataTab() {
  const now = useMemo(() => new Date(), [])
  const defaultEnd = new Date(now)
  const defaultStart = new Date(now.getTime() - 24 * 60 * 60 * 1000) // 24 hours ago

  const [rigId, setRigId] = useState('RIG_01')
  const [startTime, setStartTime] = useState(defaultStart.toISOString().slice(0, 16))
  const [endTime, setEndTime] = useState(defaultEnd.toISOString().slice(0, 16))
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['depth', 'wob', 'rpm', 'rop', 'gamma_ray', 'resistivity'])
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'plc' | 'lwd'>('all')
  const [queryParams, setQueryParams] = useState({
    rig_id: rigId,
    start_time: defaultStart.toISOString(),
    end_time: defaultEnd.toISOString(),
    limit: 1000,
  })

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    setQueryParams({
      rig_id: rigId,
      start_time: new Date(startTime).toISOString(),
      end_time: new Date(endTime).toISOString(),
      limit: 1000,
    })
  }

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['historical-data', queryParams],
    queryFn: () =>
      sensorDataApi
        .getHistorical({
          rig_id: queryParams.rig_id,
          start_time: queryParams.start_time,
          end_time: queryParams.end_time,
          limit: queryParams.limit,
        })
        .then((res) => res.data),
    placeholderData: (previousData) => previousData,
    refetchInterval: false,
  })

  const chartData = useMemo(() => {
    if (!data?.data || !Array.isArray(data.data)) return []
    return data.data.map((item: any) => ({
      timestamp: item.timestamp,
      // PLC/SCADA
      depth: item.depth || 0,
      wob: item.wob || 0,
      rpm: item.rpm || 0,
      torque: item.torque || 0,
      rop: item.rop || 0,
      mud_pressure: item.mud_pressure || 0,
      mud_flow: item.mud_flow || item.mud_flow_rate || 0,
      mud_temperature: item.mud_temperature || item.temperature || 0,
      mud_density: item.mud_density || 0,
      mud_viscosity: item.mud_viscosity || 0,
      hook_load: item.hook_load || 0,
      standpipe_pressure: item.standpipe_pressure || 0,
      casing_pressure: item.casing_pressure || 0,
      annulus_pressure: item.annulus_pressure || 0,
      pump_pressure: item.pump_pressure || 0,
      bit_temperature: item.bit_temperature || 0,
      motor_temperature: item.motor_temperature || 0,
      vibration_level: item.vibration_level || item.vibration || 0,
      power_consumption: item.power_consumption || 0,
      pump_status: item.pump_status || 0,
      compressor_status: item.compressor_status || 0,
      // LWD/MWD
      gamma_ray: item.gamma_ray || 0,
      resistivity: item.resistivity || 0,
      density: item.density || 0,
      porosity: item.porosity || 0,
      neutron_porosity: item.neutron_porosity || 0,
      sonic: item.sonic || 0,
      caliper: item.caliper || 0,
      temperature_lwd: item.temperature_lwd || 0,
    }))
  }, [data])

  const displayedMetrics = useMemo(() => {
    if (selectedCategory === 'all') return ALL_METRICS
    if (selectedCategory === 'plc') return PLC_METRICS
    return LWD_METRICS
  }, [selectedCategory])

  const handleMetricToggle = (metric: string) => {
    setSelectedMetrics((prev) =>
      prev.includes(metric) ? prev.filter((m) => m !== metric) : [...prev, metric]
    )
  }

  const handleExport = () => {
    if (!chartData.length) {
      alert('No data available for export')
      return
    }

    const csv = [
      ['Timestamp', ...ALL_METRICS.map((m) => m.label)].join(','),
      ...chartData.map((item) =>
        [
          item.timestamp,
          ...ALL_METRICS.map((m) => item[m.key as keyof typeof item] || ''),
        ].join(',')
      ),
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `historical_data_${rigId}_${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">Historical Data Analysis</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Analyze historical data from all LWD/MWD sensors and PLC/SCADA systems
        </p>
      </div>

      {/* Filters */}
      <form
        onSubmit={handleSubmit}
        className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-4"
      >
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
              Rig ID
            </label>
            <input
              type="text"
              value={rigId}
              onChange={(e) => setRigId(e.target.value)}
              className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm focus:border-cyan-500 focus:outline-none"
              placeholder="RIG_01"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
              Start Time
            </label>
            <input
              type="datetime-local"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
              className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
              End Time
            </label>
            <input
              type="datetime-local"
              value={endTime}
              onChange={(e) => setEndTime(e.target.value)}
              className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="flex items-end">
            <button
              type="submit"
              disabled={isLoading}
              className="w-full rounded-lg bg-cyan-500 hover:bg-cyan-600 text-white font-medium px-4 py-2 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <Filter className="w-4 h-4" />
                  Fetch Data
                </>
              )}
            </button>
          </div>
        </div>

        {/* Category Filter */}
        <div className="space-y-2">
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
        <div className="space-y-2">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
            Select Metrics
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
      </form>

      {/* Chart */}
      <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Historical Trends</h3>
          <button
            onClick={handleExport}
            disabled={!chartData.length}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>

        {isError && (
          <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400">
            Error loading data: {(error as Error)?.message || 'Unknown error'}
          </div>
        )}

        {isLoading && chartData.length === 0 && (
          <div className="flex items-center justify-center h-96">
            <Loader2 className="w-8 h-8 animate-spin text-cyan-500" />
          </div>
        )}

        {!isLoading && chartData.length === 0 && (
          <div className="flex items-center justify-center h-96 text-slate-500 dark:text-slate-400">
            No data available for the selected time range
          </div>
        )}

        {chartData.length > 0 && (
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
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
        )}

        {/* Data Summary */}
        {chartData.length > 0 && (
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-4">
            {displayedMetrics.filter(m => selectedMetrics.includes(m.key)).map((metric) => {
              const values = chartData.map((item) => item[metric.key as keyof typeof item] as number).filter((v) => v !== undefined && v !== null && v !== 0)
              const avg = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0
              const max = values.length > 0 ? Math.max(...values) : 0
              const min = values.length > 0 ? Math.min(...values) : 0

              return (
                <div
                  key={metric.key}
                  className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-3"
                >
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{metric.label}</div>
                  <div className="text-sm font-semibold text-slate-900 dark:text-white">
                    Avg: {avg.toFixed(2)} {metric.unit}
                  </div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Range: {min.toFixed(1)} - {max.toFixed(1)} {metric.unit}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
