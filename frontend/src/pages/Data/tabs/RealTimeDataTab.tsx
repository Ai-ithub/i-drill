import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useQuery } from '@tanstack/react-query'
import { sensorDataApi } from '@/services/api'
import { Activity, TrendingUp, Zap, AlertCircle, WifiOff, Wifi } from 'lucide-react'

interface SensorDataPoint {
  timestamp: string
  depth: number
  wob: number
  rpm: number
  torque: number
  rop: number
  mud_flow: number
  mud_pressure: number
  temperature: number
  vibration: number
}

export default function RealTimeDataTab() {
  const [rigId, setRigId] = useState('RIG_01')
  const [dataPoints, setDataPoints] = useState<SensorDataPoint[]>([])
  const [maxDataPoints] = useState(100)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting')

  // WebSocket connection for real-time data
  const wsUrl = `ws://localhost:8001/api/v1/sensor-data/ws/${rigId}`
  const { data: wsData, isConnected } = useWebSocket(wsUrl)

  // Fallback: Poll API if WebSocket is not available
  const { data: apiData } = useQuery({
    queryKey: ['realtime-data', rigId],
    queryFn: () => sensorDataApi.getRealtime(rigId, 50).then((res) => res.data),
    refetchInterval: isConnected ? false : 2000, // Poll every 2s if WS not connected
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
        depth: wsData.data.depth || 0,
        wob: wsData.data.wob || 0,
        rpm: wsData.data.rpm || 0,
        torque: wsData.data.torque || 0,
        rop: wsData.data.rop || 0,
        mud_flow: wsData.data.mud_flow || 0,
        mud_pressure: wsData.data.mud_pressure || 0,
        temperature: wsData.data.temperature || 0,
        vibration: wsData.data.vibration || 0,
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
        mud_flow: item.mud_flow || 0,
        mud_pressure: item.mud_pressure || 0,
        temperature: item.temperature || 0,
        vibration: item.vibration || 0,
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

  const statsCards = [
    {
      label: 'Current Depth',
      value: latestData ? `${latestData.depth.toFixed(1)} m` : '--',
      unit: 'm',
      icon: TrendingUp,
      color: 'bg-blue-500',
    },
    {
      label: 'Weight on Bit',
      value: latestData ? `${latestData.wob.toFixed(0)}` : '--',
      unit: 'kN',
      icon: Activity,
      color: 'bg-green-500',
    },
    {
      label: 'Rotary Speed',
      value: latestData ? `${latestData.rpm.toFixed(0)}` : '--',
      unit: 'RPM',
      icon: Zap,
      color: 'bg-yellow-500',
    },
    {
      label: 'Rate of Penetration',
      value: latestData ? `${latestData.rop.toFixed(2)}` : '--',
      unit: 'm/h',
      icon: Activity,
      color: 'bg-purple-500',
    },
    {
      label: 'Torque',
      value: latestData ? `${latestData.torque.toFixed(0)}` : '--',
      unit: 'N·m',
      icon: Activity,
      color: 'bg-orange-500',
    },
    {
      label: 'Mud Pressure',
      value: latestData ? `${latestData.mud_pressure.toFixed(1)}` : '--',
      unit: 'bar',
      icon: Activity,
      color: 'bg-red-500',
    },
  ]

  return (
    <div className="space-y-6">
      {/* Header with Connection Status */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">Real-Time Data Monitoring</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Live sensor data from drilling operations
          </p>
        </div>

        <div className="flex items-center gap-4">
          <select
            value={rigId}
            onChange={(e) => {
              setRigId(e.target.value)
              setDataPoints([]) // Clear data when switching rigs
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

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {statsCards.map((stat, index) => {
          const Icon = stat.icon
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

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Depth Chart */}
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Depth</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} label={{ value: 'm', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`${value?.toFixed(2)} m`, 'Depth']}
              />
              <Line
                type="monotone"
                dataKey="depth"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="Depth (m)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* WOB Chart */}
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Weight on Bit (WOB)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} label={{ value: 'kN', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`${value?.toFixed(1)} kN`, 'WOB']}
              />
              <Line
                type="monotone"
                dataKey="wob"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="WOB (kN)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* RPM Chart */}
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Rotary Speed (RPM)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} label={{ value: 'RPM', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`${value?.toFixed(0)} RPM`, 'RPM']}
              />
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

        {/* ROP Chart */}
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Rate of Penetration (ROP)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} label={{ value: 'm/h', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`${value?.toFixed(2)} m/h`, 'ROP']}
              />
              <Line
                type="monotone"
                dataKey="rop"
                stroke="#a855f7"
                strokeWidth={2}
                dot={false}
                name="ROP (m/h)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Torque Chart */}
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Torque</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} label={{ value: 'N·m', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`${value?.toFixed(0)} N·m`, 'Torque']}
              />
              <Line
                type="monotone"
                dataKey="torque"
                stroke="#f97316"
                strokeWidth={2}
                dot={false}
                name="Torque (N·m)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Mud Pressure Chart */}
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Mud Pressure</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" className="dark:stroke-slate-700" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} label={{ value: 'bar', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`${value?.toFixed(1)} bar`, 'Pressure']}
              />
              <Line
                type="monotone"
                dataKey="mud_pressure"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
                name="Pressure (bar)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

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

