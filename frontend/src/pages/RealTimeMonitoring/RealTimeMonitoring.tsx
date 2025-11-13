import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Zap, TrendingUp, AlertCircle, Thermometer, Gauge, Droplets, Wind, Eye } from 'lucide-react'
import { useWebSocket } from '@/hooks/useWebSocket'

interface SensorData {
  timestamp: string
  depth: number
  wob: number
  rpm: number
  torque: number
  rop: number
  mud_flow: number
  mud_pressure: number
  mud_temperature?: number
  bit_temperature?: number
  motor_temperature?: number
  surface_temperature?: number
  internal_temperature?: number
  pump_pressure?: number
  annulus_pressure?: number
  standpipe_pressure?: number
  casing_pressure?: number
  flow_in?: number
  flow_out?: number
  hook_load?: number
  block_position?: number
  status: string
}

export default function RealTimeMonitoring() {
  const [rigId, setRigId] = useState('RIG_01')
  const [dataPoints, setDataPoints] = useState<SensorData[]>([])
  const [maxDataPoints] = useState(100) // Keep last 100 points for better visualization
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting')

  // WebSocket connection
  const { data: wsData, isConnected } = useWebSocket(
    `ws://localhost:8001/api/v1/sensor-data/ws/${rigId}`
  )

  // Update connection status
  useEffect(() => {
    setConnectionStatus(isConnected ? 'connected' : 'disconnected')
  }, [isConnected])

  // Process incoming WebSocket data
  useEffect(() => {
    if (wsData && wsData.message_type === 'sensor_data') {
      const newData: SensorData = {
        timestamp: wsData.data.timestamp || new Date().toISOString(),
        depth: wsData.data.depth || 0,
        wob: wsData.data.wob || 0,
        rpm: wsData.data.rpm || 0,
        torque: wsData.data.torque || 0,
        rop: wsData.data.rop || 0,
        mud_flow: wsData.data.mud_flow || 0,
        mud_pressure: wsData.data.mud_pressure || 0,
        mud_temperature: wsData.data.mud_temperature || wsData.data.temperature || 0,
        bit_temperature: wsData.data.bit_temperature || 0,
        motor_temperature: wsData.data.motor_temperature || 0,
        surface_temperature: wsData.data.surface_temperature || 0,
        internal_temperature: wsData.data.internal_temperature || 0,
        pump_pressure: wsData.data.pump_pressure || 0,
        annulus_pressure: wsData.data.annulus_pressure || 0,
        standpipe_pressure: wsData.data.standpipe_pressure || 0,
        casing_pressure: wsData.data.casing_pressure || 0,
        flow_in: wsData.data.flow_in || 0,
        flow_out: wsData.data.flow_out || 0,
        hook_load: wsData.data.hook_load || 0,
        block_position: wsData.data.block_position || 0,
        status: wsData.data.status || 'normal'
      }

      setDataPoints(prev => {
        const updated = [...prev, newData]
        // Keep only last N points
        return updated.slice(-maxDataPoints)
      })
    }
  }, [wsData, maxDataPoints])

  // Get latest data point
  const latestData = dataPoints.length > 0 ? dataPoints[dataPoints.length - 1] : null

  // Main stats cards
  const mainStatsCards = [
    {
      label: 'Current Depth',
      value: latestData ? `${latestData.depth.toFixed(1)} ft` : '--',
      icon: TrendingUp,
      color: 'bg-blue-500',
    },
    {
      label: 'WOB',
      value: latestData ? `${latestData.wob.toFixed(0)} lbs` : '--',
      icon: Activity,
      color: 'bg-green-500',
    },
    {
      label: 'RPM',
      value: latestData ? `${latestData.rpm.toFixed(0)}` : '--',
      icon: Zap,
      color: 'bg-yellow-500',
    },
    {
      label: 'ROP',
      value: latestData ? `${latestData.rop.toFixed(1)} ft/hr` : '--',
      icon: Activity,
      color: 'bg-purple-500',
    },
  ]

  // Temperature stats
  const temperatureStats = useMemo(() => {
    if (!latestData) return []
    return [
      {
        label: 'Mud Temperature',
        value: latestData.mud_temperature ? `${latestData.mud_temperature.toFixed(1)} °C` : '--',
        icon: Thermometer,
        color: 'bg-red-500',
      },
      {
        label: 'Bit Temperature',
        value: latestData.bit_temperature ? `${latestData.bit_temperature.toFixed(1)} °C` : '--',
        icon: Thermometer,
        color: 'bg-orange-500',
      },
      {
        label: 'Motor Temperature',
        value: latestData.motor_temperature ? `${latestData.motor_temperature.toFixed(1)} °C` : '--',
        icon: Thermometer,
        color: 'bg-amber-500',
      },
      {
        label: 'Surface Temperature',
        value: latestData.surface_temperature ? `${latestData.surface_temperature.toFixed(1)} °C` : '--',
        icon: Thermometer,
        color: 'bg-pink-500',
      },
    ].filter(stat => stat.value !== '--')
  }, [latestData])

  // Pressure stats
  const pressureStats = useMemo(() => {
    if (!latestData) return []
    return [
      {
        label: 'Mud Pressure',
        value: latestData.mud_pressure ? `${latestData.mud_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-blue-500',
      },
      {
        label: 'Pump Pressure',
        value: latestData.pump_pressure ? `${latestData.pump_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-indigo-500',
      },
      {
        label: 'Standpipe Pressure',
        value: latestData.standpipe_pressure ? `${latestData.standpipe_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-cyan-500',
      },
      {
        label: 'Casing Pressure',
        value: latestData.casing_pressure ? `${latestData.casing_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-teal-500',
      },
      {
        label: 'Annulus Pressure',
        value: latestData.annulus_pressure ? `${latestData.annulus_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-emerald-500',
      },
    ].filter(stat => stat.value !== '--')
  }, [latestData])

  // Flow stats
  const flowStats = useMemo(() => {
    if (!latestData) return []
    return [
      {
        label: 'Mud Flow',
        value: latestData.mud_flow ? `${latestData.mud_flow.toFixed(1)} gpm` : '--',
        icon: Droplets,
        color: 'bg-blue-500',
      },
      {
        label: 'Flow In',
        value: latestData.flow_in ? `${latestData.flow_in.toFixed(1)} gpm` : '--',
        icon: Wind,
        color: 'bg-green-500',
      },
      {
        label: 'Flow Out',
        value: latestData.flow_out ? `${latestData.flow_out.toFixed(1)} gpm` : '--',
        icon: Wind,
        color: 'bg-red-500',
      },
    ].filter(stat => stat.value !== '--')
  }, [latestData])

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
            <Eye className="w-8 h-8 text-cyan-500" />
            RTM - Real Time Monitoring
          </h1>
          <p className="text-slate-500 dark:text-slate-300">
            Real-time monitoring of all sensors, temperatures, pressures, and drilling parameters
          </p>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-3">
          <select
            value={rigId}
            onChange={(e) => setRigId(e.target.value)}
            className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-white px-4 py-2 rounded-lg"
          >
            <option value="RIG_01">Rig 01</option>
            <option value="RIG_02">Rig 02</option>
          </select>

          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                connectionStatus === 'connected'
                  ? 'bg-green-500 animate-pulse'
                  : connectionStatus === 'connecting'
                  ? 'bg-yellow-500 animate-pulse'
                  : 'bg-red-500'
              }`}
            />
            <span className="text-sm text-slate-600 dark:text-slate-300">
              {connectionStatus === 'connected'
                ? 'Connected'
                : connectionStatus === 'connecting'
                ? 'Connecting...'
                : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Main Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {mainStatsCards.map((stat, index) => {
          const Icon = stat.icon
          return (
            <div key={index} className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <div className={`${stat.color} p-3 rounded-xl`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{stat.value}</div>
            </div>
          )
        })}
      </div>

      {/* Temperature Stats */}
      {temperatureStats.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Thermometer className="w-5 h-5 text-red-500" />
            Temperature Monitoring
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {temperatureStats.map((stat, index) => {
              const Icon = stat.icon
              return (
                <div key={index} className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`${stat.color} p-3 rounded-xl`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
                  <div className="text-2xl font-bold text-slate-900 dark:text-white">{stat.value}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Pressure Stats */}
      {pressureStats.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Gauge className="w-5 h-5 text-blue-500" />
            Pressure Monitoring
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            {pressureStats.map((stat, index) => {
              const Icon = stat.icon
              return (
                <div key={index} className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`${stat.color} p-3 rounded-xl`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
                  <div className="text-2xl font-bold text-slate-900 dark:text-white">{stat.value}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Flow Stats */}
      {flowStats.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Droplets className="w-5 h-5 text-cyan-500" />
            Flow Monitoring
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {flowStats.map((stat, index) => {
              const Icon = stat.icon
              return (
                <div key={index} className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`${stat.color} p-3 rounded-xl`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
                  <div className="text-2xl font-bold text-slate-900 dark:text-white">{stat.value}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Charts Section */}
      <div className="space-y-6">
        <h2 className="text-xl font-semibold">Real-Time Charts</h2>

        {/* Temperature Charts */}
        {temperatureStats.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {latestData?.mud_temperature !== undefined && (
              <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Mud Temperature</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={dataPoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="timestamp"
                      stroke="#64748b"
                      tick={{ fill: '#64748b', fontSize: 12 }}
                      tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                    />
                    <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#f1f5f9' }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="mud_temperature"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      name="Mud Temp (°C)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {latestData?.bit_temperature !== undefined && (
              <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Bit Temperature</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={dataPoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="timestamp"
                      stroke="#64748b"
                      tick={{ fill: '#64748b', fontSize: 12 }}
                      tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                    />
                    <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#f1f5f9' }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="bit_temperature"
                      stroke="#f97316"
                      strokeWidth={2}
                      dot={false}
                      name="Bit Temp (°C)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {latestData?.motor_temperature !== undefined && (
              <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Motor Temperature</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={dataPoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="timestamp"
                      stroke="#64748b"
                      tick={{ fill: '#64748b', fontSize: 12 }}
                      tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                    />
                    <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#f1f5f9' }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="motor_temperature"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={false}
                      name="Motor Temp (°C)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* Pressure Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
            <h3 className="text-lg font-semibold mb-4">Mud Pressure</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={dataPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="mud_pressure"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  name="Mud Pressure (psi)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {latestData?.pump_pressure !== undefined && (
            <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
              <h3 className="text-lg font-semibold mb-4">Pump Pressure</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dataPoints}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="timestamp"
                    stroke="#64748b"
                    tick={{ fill: '#64748b', fontSize: 12 }}
                    tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                  />
                  <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#f1f5f9' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="pump_pressure"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={false}
                    name="Pump Pressure (psi)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Main Drilling Parameters Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
            <h3 className="text-lg font-semibold mb-4">Weight on Bit (WOB)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={dataPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="wob"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                  name="WOB (lbs)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
            <h3 className="text-lg font-semibold mb-4">Rotary Speed (RPM)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={dataPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="rpm"
                  stroke="#eab308"
                  strokeWidth={2}
                  dot={false}
                  name="RPM"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
            <h3 className="text-lg font-semibold mb-4">Rate of Penetration (ROP)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={dataPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="rop"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={false}
                  name="ROP (ft/hr)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
            <h3 className="text-lg font-semibold mb-4">Torque</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={dataPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f1f5f9' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="torque"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  name="Torque (ft-lbs)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Current Status Table */}
      {latestData && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Current Status - All Parameters</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Depth</div>
              <div className="text-white dark:text-white font-semibold">{latestData.depth.toFixed(1)} ft</div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">WOB</div>
              <div className="text-white dark:text-white font-semibold">{latestData.wob.toFixed(0)} lbs</div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">RPM</div>
              <div className="text-white dark:text-white font-semibold">{latestData.rpm.toFixed(0)}</div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Torque</div>
              <div className="text-white dark:text-white font-semibold">{latestData.torque.toFixed(0)} ft-lbs</div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">ROP</div>
              <div className="text-white dark:text-white font-semibold">{latestData.rop.toFixed(1)} ft/hr</div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Mud Flow</div>
              <div className="text-white dark:text-white font-semibold">{latestData.mud_flow.toFixed(1)} gpm</div>
            </div>
            {latestData.mud_temperature !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Mud Temp</div>
                <div className="text-white dark:text-white font-semibold">{latestData.mud_temperature.toFixed(1)} °C</div>
              </div>
            )}
            {latestData.bit_temperature !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Bit Temp</div>
                <div className="text-white dark:text-white font-semibold">{latestData.bit_temperature.toFixed(1)} °C</div>
              </div>
            )}
            {latestData.motor_temperature !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Motor Temp</div>
                <div className="text-white dark:text-white font-semibold">{latestData.motor_temperature.toFixed(1)} °C</div>
              </div>
            )}
            {latestData.pump_pressure !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Pump Pressure</div>
                <div className="text-white dark:text-white font-semibold">{latestData.pump_pressure.toFixed(1)} psi</div>
              </div>
            )}
            {latestData.standpipe_pressure !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Standpipe Pressure</div>
                <div className="text-white dark:text-white font-semibold">{latestData.standpipe_pressure.toFixed(1)} psi</div>
              </div>
            )}
            {latestData.casing_pressure !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Casing Pressure</div>
                <div className="text-white dark:text-white font-semibold">{latestData.casing_pressure.toFixed(1)} psi</div>
              </div>
            )}
            {latestData.annulus_pressure !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Annulus Pressure</div>
                <div className="text-white dark:text-white font-semibold">{latestData.annulus_pressure.toFixed(1)} psi</div>
              </div>
            )}
            {latestData.flow_in !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Flow In</div>
                <div className="text-white dark:text-white font-semibold">{latestData.flow_in.toFixed(1)} gpm</div>
              </div>
            )}
            {latestData.flow_out !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Flow Out</div>
                <div className="text-white dark:text-white font-semibold">{latestData.flow_out.toFixed(1)} gpm</div>
              </div>
            )}
            {latestData.hook_load !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Hook Load</div>
                <div className="text-white dark:text-white font-semibold">{latestData.hook_load.toFixed(0)} lbs</div>
              </div>
            )}
            {latestData.block_position !== undefined && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Block Position</div>
                <div className="text-white dark:text-white font-semibold">{latestData.block_position.toFixed(1)} ft</div>
              </div>
            )}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-50 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Status</div>
              <div className={`font-semibold ${
                latestData.status === 'normal' ? 'text-green-500' : 'text-red-500'
              }`}>
                {latestData.status}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* No Data Message */}
      {dataPoints.length === 0 && connectionStatus === 'connected' && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-12 text-center">
          <AlertCircle className="w-12 h-12 text-slate-400 mx-auto mb-4" />
          <p className="text-slate-500 dark:text-slate-400">Waiting for real-time data...</p>
        </div>
      )}

      {/* Disconnected Message */}
      {connectionStatus === 'disconnected' && (
        <div className="rounded-2xl bg-red-50 dark:bg-red-900/20 border border-red-500 rounded-lg p-6 text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600 dark:text-red-400">Connection to server lost. Please try again.</p>
        </div>
      )}
    </div>
  )
}
