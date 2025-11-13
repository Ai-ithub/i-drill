import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useQuery } from '@tanstack/react-query'
import { sensorDataApi } from '@/services/api'
import { Activity, TrendingUp, Zap, AlertCircle, WifiOff, Wifi, Gauge, Thermometer, Radio, Database } from 'lucide-react'

interface SensorDataPoint {
  timestamp: string
  // PLC/SCADA Parameters
  depth: number
  wob: number
  rpm: number
  torque: number
  rop: number
  mud_flow: number
  mud_pressure: number
  mud_temperature: number
  mud_density?: number
  mud_viscosity?: number
  mud_ph?: number
  hook_load?: number
  block_position?: number
  standpipe_pressure?: number
  casing_pressure?: number
  annulus_pressure?: number
  pump_pressure?: number
  flow_in?: number
  flow_out?: number
  pump_status?: number
  compressor_status?: number
  power_consumption?: number
  vibration_level?: number
  bit_temperature?: number
  motor_temperature?: number
  surface_temperature?: number
  internal_temperature?: number
  // LWD/MWD Parameters
  gamma_ray?: number
  resistivity?: number
  density?: number
  porosity?: number
  neutron_porosity?: number
  sonic?: number
  caliper?: number
  temperature_lwd?: number
  vibration_lwd?: number
}

export default function RealTimeDataTab() {
  const [rigId, setRigId] = useState('RIG_01')
  const [dataPoints, setDataPoints] = useState<SensorDataPoint[]>([])
  const [maxDataPoints] = useState(100)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting')
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'plc' | 'lwd'>('all')

  // WebSocket connection for real-time data
  const wsUrl = `ws://localhost:8001/api/v1/sensor-data/ws/${rigId}`
  const { data: wsData, isConnected } = useWebSocket(wsUrl)

  // Fallback: Poll API if WebSocket is not available
  const { data: apiData } = useQuery({
    queryKey: ['realtime-data', rigId],
    queryFn: () => sensorDataApi.getRealtime(rigId, 50).then((res) => res.data),
    refetchInterval: isConnected ? false : 2000,
    enabled: !isConnected,
  })

  // Update connection status
  useEffect(() => {
    setConnectionStatus(isConnected ? 'connected' : 'disconnected')
  }, [isConnected])

  // Process WebSocket data
  useEffect(() => {
    if (wsData && wsData.message_type === 'sensor_data') {
      const newData: SensorDataPoint = {
        timestamp: wsData.data.timestamp || new Date().toISOString(),
        // PLC/SCADA
        depth: wsData.data.depth || 0,
        wob: wsData.data.wob || 0,
        rpm: wsData.data.rpm || 0,
        torque: wsData.data.torque || 0,
        rop: wsData.data.rop || 0,
        mud_flow: wsData.data.mud_flow || wsData.data.mud_flow_rate || 0,
        mud_pressure: wsData.data.mud_pressure || 0,
        mud_temperature: wsData.data.mud_temperature || wsData.data.temperature || 0,
        mud_density: wsData.data.mud_density || 0,
        mud_viscosity: wsData.data.mud_viscosity || 0,
        mud_ph: wsData.data.mud_ph || 0,
        hook_load: wsData.data.hook_load || 0,
        block_position: wsData.data.block_position || 0,
        standpipe_pressure: wsData.data.standpipe_pressure || 0,
        casing_pressure: wsData.data.casing_pressure || 0,
        annulus_pressure: wsData.data.annulus_pressure || 0,
        pump_pressure: wsData.data.pump_pressure || 0,
        flow_in: wsData.data.flow_in || 0,
        flow_out: wsData.data.flow_out || 0,
        pump_status: wsData.data.pump_status || 0,
        compressor_status: wsData.data.compressor_status || 0,
        power_consumption: wsData.data.power_consumption || 0,
        vibration_level: wsData.data.vibration_level || wsData.data.vibration || 0,
        bit_temperature: wsData.data.bit_temperature || 0,
        motor_temperature: wsData.data.motor_temperature || 0,
        surface_temperature: wsData.data.surface_temperature || 0,
        internal_temperature: wsData.data.internal_temperature || 0,
        // LWD/MWD
        gamma_ray: wsData.data.gamma_ray || 0,
        resistivity: wsData.data.resistivity || 0,
        density: wsData.data.density || 0,
        porosity: wsData.data.porosity || 0,
        neutron_porosity: wsData.data.neutron_porosity || 0,
        sonic: wsData.data.sonic || 0,
        caliper: wsData.data.caliper || 0,
        temperature_lwd: wsData.data.temperature_lwd || 0,
        vibration_lwd: wsData.data.vibration_lwd || 0,
      }

      setDataPoints((prev) => {
        const updated = [...prev, newData]
        return updated.slice(-maxDataPoints)
      })
    }
  }, [wsData, maxDataPoints])

  // Process API data (fallback)
  useEffect(() => {
    if (apiData?.data && Array.isArray(apiData.data) && !isConnected) {
      const processed = apiData.data.map((item: any) => ({
        timestamp: item.timestamp || new Date().toISOString(),
        depth: item.depth || 0,
        wob: item.wob || 0,
        rpm: item.rpm || 0,
        torque: item.torque || 0,
        rop: item.rop || 0,
        mud_flow: item.mud_flow || item.mud_flow_rate || 0,
        mud_pressure: item.mud_pressure || 0,
        mud_temperature: item.mud_temperature || item.temperature || 0,
        mud_density: item.mud_density || 0,
        mud_viscosity: item.mud_viscosity || 0,
        mud_ph: item.mud_ph || 0,
        hook_load: item.hook_load || 0,
        block_position: item.block_position || 0,
        standpipe_pressure: item.standpipe_pressure || 0,
        casing_pressure: item.casing_pressure || 0,
        annulus_pressure: item.annulus_pressure || 0,
        pump_pressure: item.pump_pressure || 0,
        flow_in: item.flow_in || 0,
        flow_out: item.flow_out || 0,
        pump_status: item.pump_status || 0,
        compressor_status: item.compressor_status || 0,
        power_consumption: item.power_consumption || 0,
        vibration_level: item.vibration_level || item.vibration || 0,
        bit_temperature: item.bit_temperature || 0,
        motor_temperature: item.motor_temperature || 0,
        surface_temperature: item.surface_temperature || 0,
        internal_temperature: item.internal_temperature || 0,
        gamma_ray: item.gamma_ray || 0,
        resistivity: item.resistivity || 0,
        density: item.density || 0,
        porosity: item.porosity || 0,
        neutron_porosity: item.neutron_porosity || 0,
        sonic: item.sonic || 0,
        caliper: item.caliper || 0,
        temperature_lwd: item.temperature_lwd || 0,
        vibration_lwd: item.vibration_lwd || 0,
      }))

      setDataPoints((prev) => {
        const combined = [...prev, ...processed]
        const unique = combined.filter(
          (item, index, self) => index === self.findIndex((t) => t.timestamp === item.timestamp)
        )
        return unique.slice(-maxDataPoints)
      })
    }
  }, [apiData, isConnected, maxDataPoints])

  const latestData = dataPoints.length > 0 ? dataPoints[dataPoints.length - 1] : null

  // PLC/SCADA Parameters
  const plcParameters = useMemo(() => [
    { key: 'depth', label: 'Depth', unit: 'ft', icon: TrendingUp, color: 'bg-blue-500', chartColor: '#3b82f6', category: 'plc' },
    { key: 'wob', label: 'WOB', unit: 'lbs', icon: Activity, color: 'bg-green-500', chartColor: '#10b981', category: 'plc' },
    { key: 'rpm', label: 'RPM', unit: 'rpm', icon: Zap, color: 'bg-yellow-500', chartColor: '#eab308', category: 'plc' },
    { key: 'rop', label: 'ROP', unit: 'ft/hr', icon: Activity, color: 'bg-purple-500', chartColor: '#a855f7', category: 'plc' },
    { key: 'torque', label: 'Torque', unit: 'ft-lbs', icon: Gauge, color: 'bg-orange-500', chartColor: '#f97316', category: 'plc' },
    { key: 'mud_pressure', label: 'Mud Pressure', unit: 'psi', icon: Gauge, color: 'bg-red-500', chartColor: '#ef4444', category: 'plc' },
    { key: 'mud_flow', label: 'Mud Flow', unit: 'gpm', icon: Activity, color: 'bg-cyan-500', chartColor: '#06b6d4', category: 'plc' },
    { key: 'mud_temperature', label: 'Mud Temperature', unit: '°C', icon: Thermometer, color: 'bg-pink-500', chartColor: '#ec4899', category: 'plc' },
    { key: 'hook_load', label: 'Hook Load', unit: 'lbs', icon: Activity, color: 'bg-indigo-500', chartColor: '#6366f1', category: 'plc' },
    { key: 'standpipe_pressure', label: 'Standpipe Pressure', unit: 'psi', icon: Gauge, color: 'bg-blue-600', chartColor: '#2563eb', category: 'plc' },
    { key: 'casing_pressure', label: 'Casing Pressure', unit: 'psi', icon: Gauge, color: 'bg-teal-500', chartColor: '#14b8a6', category: 'plc' },
    { key: 'annulus_pressure', label: 'Annulus Pressure', unit: 'psi', icon: Gauge, color: 'bg-emerald-500', chartColor: '#10b981', category: 'plc' },
    { key: 'pump_pressure', label: 'Pump Pressure', unit: 'psi', icon: Gauge, color: 'bg-violet-500', chartColor: '#8b5cf6', category: 'plc' },
    { key: 'bit_temperature', label: 'Bit Temperature', unit: '°C', icon: Thermometer, color: 'bg-red-600', chartColor: '#dc2626', category: 'plc' },
    { key: 'motor_temperature', label: 'Motor Temperature', unit: '°C', icon: Thermometer, color: 'bg-amber-600', chartColor: '#d97706', category: 'plc' },
    { key: 'vibration_level', label: 'Vibration', unit: 'g', icon: Activity, color: 'bg-rose-500', chartColor: '#f43f5e', category: 'plc' },
    { key: 'power_consumption', label: 'Power Consumption', unit: 'kW', icon: Zap, color: 'bg-yellow-600', chartColor: '#ca8a04', category: 'plc' },
  ], [])

  // LWD/MWD Parameters
  const lwdParameters = useMemo(() => [
    { key: 'gamma_ray', label: 'Gamma Ray', unit: 'API', icon: Radio, color: 'bg-yellow-400', chartColor: '#facc15', category: 'lwd' },
    { key: 'resistivity', label: 'Resistivity', unit: 'ohm-m', icon: Radio, color: 'bg-green-400', chartColor: '#4ade80', category: 'lwd' },
    { key: 'density', label: 'Density', unit: 'g/cc', icon: Database, color: 'bg-blue-400', chartColor: '#60a5fa', category: 'lwd' },
    { key: 'porosity', label: 'Porosity', unit: '%', icon: Database, color: 'bg-purple-400', chartColor: '#a78bfa', category: 'lwd' },
    { key: 'neutron_porosity', label: 'Neutron Porosity', unit: '%', icon: Radio, color: 'bg-pink-400', chartColor: '#f472b6', category: 'lwd' },
    { key: 'sonic', label: 'Sonic', unit: 'μs/ft', icon: Radio, color: 'bg-cyan-400', chartColor: '#22d3ee', category: 'lwd' },
    { key: 'caliper', label: 'Caliper', unit: 'in', icon: Gauge, color: 'bg-orange-400', chartColor: '#fb923c', category: 'lwd' },
    { key: 'temperature_lwd', label: 'LWD Temperature', unit: '°C', icon: Thermometer, color: 'bg-red-400', chartColor: '#f87171', category: 'lwd' },
  ], [])

  const allParameters = [...plcParameters, ...lwdParameters]
  const displayedParameters = useMemo(() => {
    if (selectedCategory === 'all') return allParameters
    if (selectedCategory === 'plc') return plcParameters
    return lwdParameters
  }, [selectedCategory, allParameters, plcParameters, lwdParameters])

  const statsCards = displayedParameters.slice(0, 8).map((param) => ({
    ...param,
    value: latestData ? (latestData[param.key as keyof SensorDataPoint] as number || 0).toFixed(2) : '--',
    Icon: param.icon,
  }))

  return (
    <div className="space-y-6">
      {/* Header with Connection Status */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">Real-Time Data Monitoring</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Live monitoring of all LWD/MWD sensors and PLC/SCADA data
          </p>
        </div>

        <div className="flex items-center gap-4">
          <select
            value={rigId}
            onChange={(e) => {
              setRigId(e.target.value)
              setDataPoints([])
            }}
            className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-4 py-2 text-sm focus:border-cyan-500 focus:outline-none"
          >
            <option value="RIG_01">Rig 01</option>
            <option value="RIG_02">Rig 02</option>
          </select>

          <div className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900">
            {connectionStatus === 'connected' ? (
              <Wifi className="w-5 h-5 text-green-500" />
            ) : (
              <WifiOff className="w-5 h-5 text-red-500" />
            )}
            <span className="text-sm font-medium">
              {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Category Filter */}
      <div className="flex items-center gap-2">
        <button
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

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8 gap-4">
        {statsCards.map((stat, index) => {
          const Icon = stat.Icon
          return (
            <div
              key={index}
              className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-4 shadow-sm"
            >
              <div className="flex items-center justify-between mb-3">
                <div className={`${stat.color} p-2 rounded-lg`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{stat.label}</div>
              <div className="flex items-baseline gap-1">
                <div className="text-xl font-bold text-slate-900 dark:text-white">{stat.value}</div>
                {latestData && <div className="text-xs text-slate-500 dark:text-slate-400">{stat.unit}</div>}
              </div>
            </div>
          )
        })}
      </div>

      {/* All Parameters Table */}
      {latestData && (
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">All Sensor Values</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
            {allParameters.map((param) => {
              const value = latestData[param.key as keyof SensorDataPoint] as number
              if (value === undefined || value === null) return null
              return (
                <div key={param.key} className="rounded-lg border border-slate-200 dark:border-slate-700 px-3 py-2 bg-slate-50 dark:bg-slate-800/40">
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{param.label}</div>
                  <div className="text-sm font-semibold text-slate-900 dark:text-white">
                    {value.toFixed(2)} {param.unit}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Charts Grid - PLC/SCADA */}
      {selectedCategory === 'all' || selectedCategory === 'plc' ? (
        <div>
          <h3 className="text-lg font-semibold mb-4">PLC/SCADA Parameters</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {plcParameters.slice(0, 8).map((param) => {
              const Icon = param.icon
              return (
                <div key={param.key} className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Icon className="w-5 h-5" />
                    {param.label}
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={dataPoints}>
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
                        formatter={(value: number) => [`${value?.toFixed(2)} ${param.unit}`, param.label]}
                      />
                      <Line
                        type="monotone"
                        dataKey={param.key}
                        stroke={param.chartColor}
                        strokeWidth={2}
                        dot={false}
                        name={`${param.label} (${param.unit})`}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )
            })}
          </div>
        </div>
      ) : null}

      {/* Charts Grid - LWD/MWD */}
      {selectedCategory === 'all' || selectedCategory === 'lwd' ? (
        <div>
          <h3 className="text-lg font-semibold mb-4">LWD/MWD Parameters</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {lwdParameters.map((param) => {
              const Icon = param.icon
              return (
                <div key={param.key} className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Icon className="w-5 h-5" />
                    {param.label}
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={dataPoints}>
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
                        formatter={(value: number) => [`${value?.toFixed(2)} ${param.unit}`, param.label]}
                      />
                      <Line
                        type="monotone"
                        dataKey={param.key}
                        stroke={param.chartColor}
                        strokeWidth={2}
                        dot={false}
                        name={`${param.label} (${param.unit})`}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )
            })}
          </div>
        </div>
      ) : null}

      {/* No Data Message */}
      {dataPoints.length === 0 && (
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-12 text-center">
          <AlertCircle className="w-12 h-12 text-slate-400 mx-auto mb-4" />
          <p className="text-slate-500 dark:text-slate-400">
            {connectionStatus === 'connected' ? 'Waiting for real-time data...' : 'Connecting to data source...'}
          </p>
        </div>
      )}
    </div>
  )
}
