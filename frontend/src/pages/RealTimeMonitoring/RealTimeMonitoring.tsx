import { useState } from 'react'
import { useWebSocket } from '@/services/websocket'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { SensorDataPoint } from '@/types'
import { Activity, Wifi, WifiOff } from 'lucide-react'

export default function RealTimeMonitoring() {
  const [rigId, setRigId] = useState('RIG_01')
  const [dataHistory, setDataHistory] = useState<SensorDataPoint[]>([])
  const maxHistorySize = 100

  const { isConnected, lastMessage, error, connect, disconnect } = useWebSocket({
    rigId,
    autoConnect: false,
    onMessage: (message) => {
      if (message.message_type === 'sensor_data') {
        const sensorData = message.data as SensorDataPoint
        setDataHistory((prev) => {
          const newHistory = [sensorData, ...prev].slice(0, maxHistorySize)
          return newHistory
        })
      }
    },
  })

  const chartData = dataHistory.map((point) => ({
    time: new Date(point.timestamp).toLocaleTimeString('fa-IR'),
    depth: point.depth,
    wob: point.wob,
    rpm: point.rpm,
    torque: point.torque,
    rop: point.rop,
  }))

  const latestData = dataHistory[0]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">مانیتورینگ Real-time</h1>
          <p className="text-slate-400">داده‌های زنده از سنسورها</p>
        </div>
        <div className="flex items-center gap-4">
          <input
            type="text"
            value={rigId}
            onChange={(e) => setRigId(e.target.value)}
            placeholder="Rig ID"
            className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
          />
          {isConnected ? (
            <button
              onClick={disconnect}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
            >
              <WifiOff className="w-4 h-4" />
              قطع اتصال
            </button>
          ) : (
            <button
              onClick={connect}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            >
              <Wifi className="w-4 h-4" />
              اتصال
            </button>
          )}
          <div
            className={`flex items-center gap-2 px-3 py-1 rounded-full ${
              isConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}
          >
            {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
            <span className="text-sm">{isConnected ? 'متصل' : 'قطع شده'}</span>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/20 border border-red-500 text-red-400 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {latestData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="عمق" value={latestData.depth?.toFixed(2) || 'N/A'} unit="متر" />
          <MetricCard label="WOB" value={latestData.wob?.toFixed(2) || 'N/A'} unit="kN" />
          <MetricCard label="RPM" value={latestData.rpm?.toFixed(2) || 'N/A'} unit="rpm" />
          <MetricCard label="Torque" value={latestData.torque?.toFixed(2) || 'N/A'} unit="N.m" />
          <MetricCard label="ROP" value={latestData.rop?.toFixed(2) || 'N/A'} unit="m/h" />
          <MetricCard
            label="Power"
            value={latestData.power_consumption?.toFixed(2) || 'N/A'}
            unit="kW"
          />
          <MetricCard
            label="Bit Temp"
            value={latestData.bit_temperature?.toFixed(2) || 'N/A'}
            unit="°C"
          />
          <MetricCard
            label="Vibration"
            value={latestData.vibration_level?.toFixed(2) || 'N/A'}
            unit="g"
          />
        </div>
      )}

      {chartData.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-semibold text-white mb-4">روند Real-time</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="time" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }} />
              <Legend />
              <Line type="monotone" dataKey="depth" stroke="#3b82f6" strokeWidth={2} />
              <Line type="monotone" dataKey="wob" stroke="#10b981" strokeWidth={2} />
              <Line type="monotone" dataKey="rpm" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
      <div className="text-slate-400 text-sm mb-1">{label}</div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-white">{value}</span>
        <span className="text-sm text-slate-400">{unit}</span>
      </div>
    </div>
  )
}

