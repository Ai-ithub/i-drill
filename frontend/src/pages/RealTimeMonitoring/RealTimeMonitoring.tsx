import { useState, useEffect, useCallback } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Zap, TrendingUp, AlertCircle } from 'lucide-react'
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
  status: string
}

export default function RealTimeMonitoring() {
  const [rigId, setRigId] = useState('RIG_01')
  const [dataPoints, setDataPoints] = useState<SensorData[]>([])
  const [maxDataPoints] = useState(50) // Keep last 50 points
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
        timestamp: wsData.data.timestamp,
        depth: wsData.data.depth,
        wob: wsData.data.wob,
        rpm: wsData.data.rpm,
        torque: wsData.data.torque,
        rop: wsData.data.rop,
        mud_flow: wsData.data.mud_flow,
        mud_pressure: wsData.data.mud_pressure,
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

  const statsCards = [
    {
      label: 'عمق فعلی',
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">مانیتورینگ لحظه‌ای</h1>
          <p className="text-slate-400">نمایش داده‌های real-time دکل حفاری</p>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-3">
          <select
            value={rigId}
            onChange={(e) => setRigId(e.target.value)}
            className="bg-slate-700 text-white px-4 py-2 rounded-lg border border-slate-600"
          >
            <option value="RIG_01">دکل 01</option>
            <option value="RIG_02">دکل 02</option>
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
            <span className="text-sm text-slate-300">
              {connectionStatus === 'connected'
                ? 'متصل'
                : connectionStatus === 'connecting'
                ? 'در حال اتصال...'
                : 'قطع شده'}
            </span>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statsCards.map((stat, index) => {
          const Icon = stat.icon
          return (
            <div key={index} className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <div className={`${stat.color} p-3 rounded-lg`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="text-slate-400 text-sm mb-1">{stat.label}</div>
              <div className="text-2xl font-bold text-white">{stat.value}</div>
            </div>
          )
        })}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* WOB Chart */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-semibold text-white mb-4">Weight on Bit (WOB)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('fa-IR')}
              />
              <YAxis stroke="#9CA3AF" tick={{ fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }}
                labelStyle={{ color: '#F1F5F9' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="wob"
                stroke="#10B981"
                strokeWidth={2}
                dot={false}
                name="WOB (lbs)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* RPM Chart */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-semibold text-white mb-4">Rotary Speed (RPM)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('fa-IR')}
              />
              <YAxis stroke="#9CA3AF" tick={{ fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }}
                labelStyle={{ color: '#F1F5F9' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="rpm"
                stroke="#EAB308"
                strokeWidth={2}
                dot={false}
                name="RPM"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* ROP Chart */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-semibold text-white mb-4">Rate of Penetration (ROP)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('fa-IR')}
              />
              <YAxis stroke="#9CA3AF" tick={{ fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }}
                labelStyle={{ color: '#F1F5F9' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="rop"
                stroke="#A855F7"
                strokeWidth={2}
                dot={false}
                name="ROP (ft/hr)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Mud Pressure Chart */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-semibold text-white mb-4">Mud Pressure</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString('fa-IR')}
              />
              <YAxis stroke="#9CA3AF" tick={{ fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }}
                labelStyle={{ color: '#F1F5F9' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="mud_pressure"
                stroke="#3B82F6"
                strokeWidth={2}
                dot={false}
                name="Pressure (psi)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Current Status */}
      {latestData && (
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-semibold text-white mb-4">وضعیت فعلی</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <div>
              <div className="text-slate-400 text-sm">Depth</div>
              <div className="text-white font-semibold">{latestData.depth.toFixed(1)} ft</div>
            </div>
            <div>
              <div className="text-slate-400 text-sm">WOB</div>
              <div className="text-white font-semibold">{latestData.wob.toFixed(0)} lbs</div>
            </div>
            <div>
              <div className="text-slate-400 text-sm">RPM</div>
              <div className="text-white font-semibold">{latestData.rpm.toFixed(0)}</div>
            </div>
            <div>
              <div className="text-slate-400 text-sm">Torque</div>
              <div className="text-white font-semibold">{latestData.torque.toFixed(0)} ft-lbs</div>
            </div>
            <div>
              <div className="text-slate-400 text-sm">ROP</div>
              <div className="text-white font-semibold">{latestData.rop.toFixed(1)} ft/hr</div>
            </div>
            <div>
              <div className="text-slate-400 text-sm">Status</div>
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
        <div className="bg-slate-800 rounded-lg p-12 border border-slate-700 text-center">
          <AlertCircle className="w-12 h-12 text-slate-500 mx-auto mb-4" />
          <p className="text-slate-400">در انتظار داده‌های real-time...</p>
        </div>
      )}

      {/* Disconnected Message */}
      {connectionStatus === 'disconnected' && (
        <div className="bg-red-900/20 border border-red-500 rounded-lg p-6 text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-400">اتصال به سرور قطع شده است. لطفاً دوباره تلاش کنید.</p>
        </div>
      )}
    </div>
  )
}
