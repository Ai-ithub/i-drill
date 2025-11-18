import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Zap, TrendingUp, AlertCircle, Thermometer, Gauge, Droplets, Wind, Eye } from 'lucide-react'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useAuth } from '@/context/AuthContext'

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

// Normal ranges for sensors
interface SensorRange {
  min: number
  max: number
  unit: string
}

const SENSOR_RANGES: Record<string, SensorRange> = {
  wob: { min: 40000, max: 50000, unit: 'lbs' },
  rpm: { min: 120, max: 180, unit: 'rpm' },
  torque: { min: 7000, max: 9000, unit: 'ft-lbs' },
  rop: { min: 20, max: 30, unit: 'ft/hr' },
  mud_flow: { min: 600, max: 700, unit: 'gpm' },
  mud_pressure: { min: 2500, max: 3000, unit: 'psi' },
  mud_temperature: { min: 70, max: 85, unit: '°C' },
  bit_temperature: { min: 100, max: 120, unit: '°C' },
  motor_temperature: { min: 80, max: 100, unit: '°C' },
  surface_temperature: { min: 20, max: 35, unit: '°C' },
  internal_temperature: { min: 90, max: 110, unit: '°C' },
  pump_pressure: { min: 3000, max: 3500, unit: 'psi' },
  annulus_pressure: { min: 2300, max: 2700, unit: 'psi' },
  standpipe_pressure: { min: 2800, max: 3200, unit: 'psi' },
  casing_pressure: { min: 2200, max: 2600, unit: 'psi' },
  flow_in: { min: 600, max: 700, unit: 'gpm' },
  flow_out: { min: 590, max: 690, unit: 'gpm' },
  hook_load: { min: 120000, max: 135000, unit: 'lbs' },
}

// Get anomaly status for a sensor
const getAnomalyStatus = (sensorKey: string, value: number | undefined): 'normal' | 'warning' | 'critical' => {
  if (value === undefined || value === null) return 'normal'
  const range = SENSOR_RANGES[sensorKey]
  if (!range) return 'normal'
  
  const deviation = Math.abs(value - (range.min + range.max) / 2) / ((range.max - range.min) / 2)
  
  if (deviation > 0.3) return 'critical' // More than 30% deviation
  if (deviation > 0.15 || value < range.min || value > range.max) return 'warning' // More than 15% deviation or outside range
  return 'normal'
}

export default function RealTimeMonitoring() {
  const [rigId, setRigId] = useState('RIG_01')
  const [dataPoints, setDataPoints] = useState<SensorData[]>([])
  const [maxDataPoints] = useState(100) // Keep last 100 points for better visualization
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting')
  const [useMockData, setUseMockData] = useState(false)

  // WebSocket connection with authentication
  const wsBaseUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8001/api/v1'
  const { token } = useAuth() // Get token for WebSocket authentication (fallback to cookie)
  const { data: wsData, isConnected } = useWebSocket(
    `${wsBaseUrl}/sensor-data/ws/${rigId}`,
    { token } // Token will be sent as query parameter, but cookies (httpOnly) are preferred
  )

  // Update connection status
  useEffect(() => {
    if (isConnected) {
      setConnectionStatus('connected')
      setUseMockData(false)
    } else {
      setConnectionStatus('disconnected')
      // Use mock data if WebSocket is not connected
      if (!isConnected) {
        setUseMockData(true)
      }
    }
  }, [isConnected])

  // Generate mock sensor data with some anomalies
  const generateMockData = (baseDepth = 5000, baseWOB = 45000, baseRPM = 150): SensorData => {
    const now = new Date()
    const variation = 0.05 // 5% variation for realistic data
    const randomVariation = () => 1 + (Math.random() - 0.5) * 2 * variation
    
    // Generate occasional anomalies (about 20% chance)
    const shouldHaveAnomaly = Math.random() < 0.2
    const anomalyMultiplier = shouldHaveAnomaly ? (1 + (Math.random() * 0.5)) : 1 // 1.0 to 1.5x for anomalies
    
    // Specific anomaly for some sensors occasionally
    const bitTempAnomaly = Math.random() < 0.15 // 15% chance of high bit temp
    const motorTempAnomaly = Math.random() < 0.12 // 12% chance of high motor temp
    const pumpPressureAnomaly = Math.random() < 0.18 // 18% chance of high pump pressure
    const flowAnomaly = Math.random() < 0.1 // 10% chance of flow anomaly

    return {
      timestamp: now.toISOString(),
      depth: baseDepth + Math.sin(Date.now() / 10000) * 10, // Gradually increasing with variation
      wob: Math.max(35000, Math.min(55000, baseWOB * randomVariation() * (shouldHaveAnomaly && Math.random() < 0.5 ? anomalyMultiplier : 1))),
      rpm: Math.max(100, Math.min(200, baseRPM * randomVariation() * (shouldHaveAnomaly && Math.random() < 0.5 ? anomalyMultiplier : 1))),
      torque: Math.max(6000, Math.min(10000, 8000 * randomVariation() * (shouldHaveAnomaly && Math.random() < 0.5 ? anomalyMultiplier : 1))),
      rop: Math.max(15, Math.min(35, 25.5 * randomVariation())),
      mud_flow: flowAnomaly ? (Math.random() < 0.5 ? 550 : 750) : 650 * randomVariation(), // Sometimes low or high
      mud_pressure: 2800 * randomVariation(),
      mud_temperature: bitTempAnomaly ? 90 + Math.random() * 5 : 75 + Math.random() * 5, // Sometimes high
      bit_temperature: bitTempAnomaly ? 125 + Math.random() * 10 : 110 + Math.random() * 10, // 125-135°C when anomaly
      motor_temperature: motorTempAnomaly ? 105 + Math.random() * 10 : 85 + Math.random() * 10, // 105-115°C when anomaly
      surface_temperature: 25 + Math.random() * 5, // 25-30°C
      internal_temperature: motorTempAnomaly ? 110 + Math.random() * 10 : 95 + Math.random() * 10, // Higher when motor temp anomaly
      pump_pressure: pumpPressureAnomaly ? 3600 + Math.random() * 200 : 3200 * randomVariation(), // 3600-3800 when anomaly
      annulus_pressure: 2500 * randomVariation(),
      standpipe_pressure: 3000 * randomVariation(),
      casing_pressure: 2400 * randomVariation(),
      flow_in: flowAnomaly ? (Math.random() < 0.5 ? 550 : 750) : 650 * randomVariation(),
      flow_out: flowAnomaly ? (Math.random() < 0.5 ? 540 : 740) : 645 * randomVariation(),
      hook_load: 125000 + Math.random() * 5000, // 125k-130k lbs
      block_position: 500 + Math.sin(Date.now() / 5000) * 5, // Variable position
      status: shouldHaveAnomaly ? 'anomaly' : 'normal'
    }
  }

  // Mock data generator interval
  useEffect(() => {
    if (!useMockData) return

    // Initialize with some mock data points
    const initialData: SensorData[] = Array.from({ length: 10 }, (_, i) => {
      const baseDepth = 5000 - (10 - i) * 0.1
      return generateMockData(baseDepth)
    })
    setDataPoints(initialData)

    // Generate new mock data every 2 seconds
    const interval = setInterval(() => {
      setDataPoints(prev => {
        const newData = generateMockData()
        const updated = [...prev, newData]
        return updated.slice(-maxDataPoints)
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [useMockData, maxDataPoints])

  // Process incoming WebSocket data
  useEffect(() => {
    if (wsData && wsData.message_type === 'sensor_data') {
      setUseMockData(false) // Switch to real data when available
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

  // Get anomaly list for latest data
  const anomalyList = useMemo(() => {
    if (!latestData) return []
    const anomalies: Array<{ sensor: string; value: number; unit: string; status: 'warning' | 'critical' }> = []
    
    const sensors = [
      { key: 'wob', value: latestData.wob },
      { key: 'rpm', value: latestData.rpm },
      { key: 'torque', value: latestData.torque },
      { key: 'rop', value: latestData.rop },
      { key: 'mud_flow', value: latestData.mud_flow },
      { key: 'mud_pressure', value: latestData.mud_pressure },
      { key: 'mud_temperature', value: latestData.mud_temperature },
      { key: 'bit_temperature', value: latestData.bit_temperature },
      { key: 'motor_temperature', value: latestData.motor_temperature },
      { key: 'surface_temperature', value: latestData.surface_temperature },
      { key: 'internal_temperature', value: latestData.internal_temperature },
      { key: 'pump_pressure', value: latestData.pump_pressure },
      { key: 'annulus_pressure', value: latestData.annulus_pressure },
      { key: 'standpipe_pressure', value: latestData.standpipe_pressure },
      { key: 'casing_pressure', value: latestData.casing_pressure },
      { key: 'flow_in', value: latestData.flow_in },
      { key: 'flow_out', value: latestData.flow_out },
      { key: 'hook_load', value: latestData.hook_load },
    ]

    sensors.forEach(({ key, value }) => {
      if (value !== undefined && value !== null) {
        const status = getAnomalyStatus(key, value)
        if (status !== 'normal') {
          const range = SENSOR_RANGES[key]
          anomalies.push({
            sensor: key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
            value,
            unit: range?.unit || '',
            status,
          })
        }
      }
    })

    return anomalies.sort((a, b) => (b.status === 'critical' ? 1 : 0) - (a.status === 'critical' ? 1 : 0))
  }, [latestData])

  // Main stats cards with anomaly detection
  const mainStatsCards = [
    {
      label: 'Current Depth',
      value: latestData ? `${latestData.depth.toFixed(1)} ft` : '--',
      icon: TrendingUp,
      color: 'bg-blue-500',
      anomalyStatus: 'normal' as const,
    },
    {
      label: 'WOB',
      value: latestData ? `${latestData.wob.toFixed(0)} lbs` : '--',
      icon: Activity,
      color: 'bg-green-500',
      anomalyStatus: latestData ? getAnomalyStatus('wob', latestData.wob) : 'normal' as const,
    },
    {
      label: 'RPM',
      value: latestData ? `${latestData.rpm.toFixed(0)}` : '--',
      icon: Zap,
      color: 'bg-yellow-500',
      anomalyStatus: latestData ? getAnomalyStatus('rpm', latestData.rpm) : 'normal' as const,
    },
    {
      label: 'ROP',
      value: latestData ? `${latestData.rop.toFixed(1)} ft/hr` : '--',
      icon: Activity,
      color: 'bg-purple-500',
      anomalyStatus: latestData ? getAnomalyStatus('rop', latestData.rop) : 'normal' as const,
    },
  ]

  // Temperature stats with anomaly detection
  const temperatureStats = useMemo(() => {
    if (!latestData) return []
    return [
      {
        label: 'Mud Temperature',
        value: latestData.mud_temperature ? `${latestData.mud_temperature.toFixed(1)} °C` : '--',
        icon: Thermometer,
        color: 'bg-red-500',
        anomalyStatus: latestData.mud_temperature ? getAnomalyStatus('mud_temperature', latestData.mud_temperature) : 'normal' as const,
      },
      {
        label: 'Bit Temperature',
        value: latestData.bit_temperature ? `${latestData.bit_temperature.toFixed(1)} °C` : '--',
        icon: Thermometer,
        color: 'bg-orange-500',
        anomalyStatus: latestData.bit_temperature ? getAnomalyStatus('bit_temperature', latestData.bit_temperature) : 'normal' as const,
      },
      {
        label: 'Motor Temperature',
        value: latestData.motor_temperature ? `${latestData.motor_temperature.toFixed(1)} °C` : '--',
        icon: Thermometer,
        color: 'bg-amber-500',
        anomalyStatus: latestData.motor_temperature ? getAnomalyStatus('motor_temperature', latestData.motor_temperature) : 'normal' as const,
      },
      {
        label: 'Surface Temperature',
        value: latestData.surface_temperature ? `${latestData.surface_temperature.toFixed(1)} °C` : '--',
        icon: Thermometer,
        color: 'bg-pink-500',
        anomalyStatus: latestData.surface_temperature ? getAnomalyStatus('surface_temperature', latestData.surface_temperature) : 'normal' as const,
      },
    ].filter(stat => stat.value !== '--')
  }, [latestData])

  // Pressure stats with anomaly detection
  const pressureStats = useMemo(() => {
    if (!latestData) return []
    return [
      {
        label: 'Mud Pressure',
        value: latestData.mud_pressure ? `${latestData.mud_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-blue-500',
        anomalyStatus: latestData.mud_pressure ? getAnomalyStatus('mud_pressure', latestData.mud_pressure) : 'normal' as const,
      },
      {
        label: 'Pump Pressure',
        value: latestData.pump_pressure ? `${latestData.pump_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-indigo-500',
        anomalyStatus: latestData.pump_pressure ? getAnomalyStatus('pump_pressure', latestData.pump_pressure) : 'normal' as const,
      },
      {
        label: 'Standpipe Pressure',
        value: latestData.standpipe_pressure ? `${latestData.standpipe_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-cyan-500',
        anomalyStatus: latestData.standpipe_pressure ? getAnomalyStatus('standpipe_pressure', latestData.standpipe_pressure) : 'normal' as const,
      },
      {
        label: 'Casing Pressure',
        value: latestData.casing_pressure ? `${latestData.casing_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-teal-500',
        anomalyStatus: latestData.casing_pressure ? getAnomalyStatus('casing_pressure', latestData.casing_pressure) : 'normal' as const,
      },
      {
        label: 'Annulus Pressure',
        value: latestData.annulus_pressure ? `${latestData.annulus_pressure.toFixed(1)} psi` : '--',
        icon: Gauge,
        color: 'bg-emerald-500',
        anomalyStatus: latestData.annulus_pressure ? getAnomalyStatus('annulus_pressure', latestData.annulus_pressure) : 'normal' as const,
      },
    ].filter(stat => stat.value !== '--')
  }, [latestData])

  // Flow stats with anomaly detection
  const flowStats = useMemo(() => {
    if (!latestData) return []
    return [
      {
        label: 'Mud Flow',
        value: latestData.mud_flow ? `${latestData.mud_flow.toFixed(1)} gpm` : '--',
        icon: Droplets,
        color: 'bg-blue-500',
        anomalyStatus: latestData.mud_flow ? getAnomalyStatus('mud_flow', latestData.mud_flow) : 'normal' as const,
      },
      {
        label: 'Flow In',
        value: latestData.flow_in ? `${latestData.flow_in.toFixed(1)} gpm` : '--',
        icon: Wind,
        color: 'bg-green-500',
        anomalyStatus: latestData.flow_in ? getAnomalyStatus('flow_in', latestData.flow_in) : 'normal' as const,
      },
      {
        label: 'Flow Out',
        value: latestData.flow_out ? `${latestData.flow_out.toFixed(1)} gpm` : '--',
        icon: Wind,
        color: 'bg-red-500',
        anomalyStatus: latestData.flow_out ? getAnomalyStatus('flow_out', latestData.flow_out) : 'normal' as const,
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
                : useMockData
                ? 'Demo Mode'
                : 'Disconnected'}
            </span>
            {useMockData && (
              <span className="text-xs text-slate-400 dark:text-slate-500 ml-2">(Mock Data)</span>
            )}
          </div>
        </div>
      </div>

      {/* Anomaly Alert Banner */}
      {anomalyList.length > 0 && (
        <div className={`rounded-2xl border-2 p-6 shadow-lg ${
          anomalyList.some(a => a.status === 'critical')
            ? 'bg-red-50 dark:bg-red-900/20 border-red-500'
            : 'bg-amber-50 dark:bg-amber-900/20 border-amber-500'
        }`}>
          <div className="flex items-center gap-3 mb-4">
            <AlertCircle className={`w-6 h-6 ${
              anomalyList.some(a => a.status === 'critical') ? 'text-red-500' : 'text-amber-500'
            }`} />
            <h3 className={`text-lg font-semibold ${
              anomalyList.some(a => a.status === 'critical') ? 'text-red-700 dark:text-red-300' : 'text-amber-700 dark:text-amber-300'
            }`}>
              {anomalyList.some(a => a.status === 'critical') ? 'Critical Anomalies Detected' : 'Warning: Anomalies Detected'}
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {anomalyList.map((anomaly, index) => (
              <div
                key={index}
                className={`rounded-lg p-3 ${
                  anomaly.status === 'critical'
                    ? 'bg-red-100 dark:bg-red-900/40 border border-red-300 dark:border-red-700'
                    : 'bg-amber-100 dark:bg-amber-900/40 border border-amber-300 dark:border-amber-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-medium ${
                    anomaly.status === 'critical' ? 'text-red-900 dark:text-red-200' : 'text-amber-900 dark:text-amber-200'
                  }`}>
                    {anomaly.sensor}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    anomaly.status === 'critical'
                      ? 'bg-red-500 text-white'
                      : 'bg-amber-500 text-white'
                  }`}>
                    {anomaly.status === 'critical' ? 'CRITICAL' : 'WARNING'}
                  </span>
                </div>
                <div className={`text-lg font-bold mt-1 ${
                  anomaly.status === 'critical' ? 'text-red-700 dark:text-red-300' : 'text-amber-700 dark:text-amber-300'
                }`}>
                  {anomaly.value.toFixed(1)} {anomaly.unit}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {mainStatsCards.map((stat, index) => {
          const Icon = stat.icon
          const hasAnomaly = stat.anomalyStatus !== 'normal'
          const isCritical = stat.anomalyStatus === 'critical'
          return (
            <div
              key={index}
              className={`rounded-2xl bg-white dark:bg-slate-900 border-2 p-6 shadow-sm ${
                isCritical
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                  : hasAnomaly
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20'
                  : 'border-slate-200 dark:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`${stat.color} p-3 rounded-xl`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                {hasAnomaly && (
                  <div className={`px-2 py-1 rounded-full text-xs font-semibold ${
                    isCritical
                      ? 'bg-red-500 text-white'
                      : 'bg-amber-500 text-white'
                  }`}>
                    {isCritical ? 'CRITICAL' : 'WARNING'}
                  </div>
                )}
              </div>
              <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
              <div className={`text-2xl font-bold ${
                isCritical
                  ? 'text-red-700 dark:text-red-300'
                  : hasAnomaly
                  ? 'text-amber-700 dark:text-amber-300'
                  : 'text-slate-900 dark:text-white'
              }`}>
                {stat.value}
              </div>
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
              const hasAnomaly = stat.anomalyStatus !== 'normal'
              const isCritical = stat.anomalyStatus === 'critical'
              return (
                <div
                  key={index}
                  className={`rounded-2xl bg-white dark:bg-slate-900 border-2 p-6 shadow-sm ${
                    isCritical
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : hasAnomaly
                      ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20'
                      : 'border-slate-200 dark:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`${stat.color} p-3 rounded-xl`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    {hasAnomaly && (
                      <div className={`px-2 py-1 rounded-full text-xs font-semibold ${
                        isCritical ? 'bg-red-500 text-white' : 'bg-amber-500 text-white'
                      }`}>
                        {isCritical ? 'CRITICAL' : 'WARNING'}
                      </div>
                    )}
                  </div>
                  <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
                  <div className={`text-2xl font-bold ${
                    isCritical
                      ? 'text-red-700 dark:text-red-300'
                      : hasAnomaly
                      ? 'text-amber-700 dark:text-amber-300'
                      : 'text-slate-900 dark:text-white'
                  }`}>
                    {stat.value}
                  </div>
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
              const hasAnomaly = stat.anomalyStatus !== 'normal'
              const isCritical = stat.anomalyStatus === 'critical'
              return (
                <div
                  key={index}
                  className={`rounded-2xl bg-white dark:bg-slate-900 border-2 p-6 shadow-sm ${
                    isCritical
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : hasAnomaly
                      ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20'
                      : 'border-slate-200 dark:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`${stat.color} p-3 rounded-xl`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    {hasAnomaly && (
                      <div className={`px-2 py-1 rounded-full text-xs font-semibold ${
                        isCritical ? 'bg-red-500 text-white' : 'bg-amber-500 text-white'
                      }`}>
                        {isCritical ? 'CRITICAL' : 'WARNING'}
                      </div>
                    )}
                  </div>
                  <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
                  <div className={`text-2xl font-bold ${
                    isCritical
                      ? 'text-red-700 dark:text-red-300'
                      : hasAnomaly
                      ? 'text-amber-700 dark:text-amber-300'
                      : 'text-slate-900 dark:text-white'
                  }`}>
                    {stat.value}
                  </div>
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
              const hasAnomaly = stat.anomalyStatus !== 'normal'
              const isCritical = stat.anomalyStatus === 'critical'
              return (
                <div
                  key={index}
                  className={`rounded-2xl bg-white dark:bg-slate-900 border-2 p-6 shadow-sm ${
                    isCritical
                      ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                      : hasAnomaly
                      ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/20'
                      : 'border-slate-200 dark:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`${stat.color} p-3 rounded-xl`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    {hasAnomaly && (
                      <div className={`px-2 py-1 rounded-full text-xs font-semibold ${
                        isCritical ? 'bg-red-500 text-white' : 'bg-amber-500 text-white'
                      }`}>
                        {isCritical ? 'CRITICAL' : 'WARNING'}
                      </div>
                    )}
                  </div>
                  <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
                  <div className={`text-2xl font-bold ${
                    isCritical
                      ? 'text-red-700 dark:text-red-300'
                      : hasAnomaly
                      ? 'text-amber-700 dark:text-amber-300'
                      : 'text-slate-900 dark:text-white'
                  }`}>
                    {stat.value}
                  </div>
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
              <div className="text-slate-900 dark:text-white font-semibold">{latestData.depth.toFixed(1)} ft</div>
            </div>
            <div className={`rounded-xl border-2 px-4 py-3 ${
              getAnomalyStatus('wob', latestData.wob) === 'critical'
                ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                : getAnomalyStatus('wob', latestData.wob) === 'warning'
                ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
            }`}>
              <div className="flex items-center justify-between">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">WOB</div>
                {getAnomalyStatus('wob', latestData.wob) !== 'normal' && (
                  <AlertCircle className={`w-3 h-3 ${
                    getAnomalyStatus('wob', latestData.wob) === 'critical' ? 'text-red-500' : 'text-amber-500'
                  }`} />
                )}
              </div>
              <div className={`font-semibold ${
                getAnomalyStatus('wob', latestData.wob) === 'critical'
                  ? 'text-red-700 dark:text-red-300'
                  : getAnomalyStatus('wob', latestData.wob) === 'warning'
                  ? 'text-amber-700 dark:text-amber-300'
                  : 'text-slate-900 dark:text-white'
              }`}>{latestData.wob.toFixed(0)} lbs</div>
            </div>
            <div className={`rounded-xl border-2 px-4 py-3 ${
              getAnomalyStatus('rpm', latestData.rpm) === 'critical'
                ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                : getAnomalyStatus('rpm', latestData.rpm) === 'warning'
                ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
            }`}>
              <div className="flex items-center justify-between">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">RPM</div>
                {getAnomalyStatus('rpm', latestData.rpm) !== 'normal' && (
                  <AlertCircle className={`w-3 h-3 ${
                    getAnomalyStatus('rpm', latestData.rpm) === 'critical' ? 'text-red-500' : 'text-amber-500'
                  }`} />
                )}
              </div>
              <div className={`font-semibold ${
                getAnomalyStatus('rpm', latestData.rpm) === 'critical'
                  ? 'text-red-700 dark:text-red-300'
                  : getAnomalyStatus('rpm', latestData.rpm) === 'warning'
                  ? 'text-amber-700 dark:text-amber-300'
                  : 'text-slate-900 dark:text-white'
              }`}>{latestData.rpm.toFixed(0)}</div>
            </div>
            <div className={`rounded-xl border-2 px-4 py-3 ${
              getAnomalyStatus('torque', latestData.torque) === 'critical'
                ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                : getAnomalyStatus('torque', latestData.torque) === 'warning'
                ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
            }`}>
              <div className="flex items-center justify-between">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Torque</div>
                {getAnomalyStatus('torque', latestData.torque) !== 'normal' && (
                  <AlertCircle className={`w-3 h-3 ${
                    getAnomalyStatus('torque', latestData.torque) === 'critical' ? 'text-red-500' : 'text-amber-500'
                  }`} />
                )}
              </div>
              <div className={`font-semibold ${
                getAnomalyStatus('torque', latestData.torque) === 'critical'
                  ? 'text-red-700 dark:text-red-300'
                  : getAnomalyStatus('torque', latestData.torque) === 'warning'
                  ? 'text-amber-700 dark:text-amber-300'
                  : 'text-slate-900 dark:text-white'
              }`}>{latestData.torque.toFixed(0)} ft-lbs</div>
            </div>
            <div className={`rounded-xl border-2 px-4 py-3 ${
              getAnomalyStatus('rop', latestData.rop) === 'critical'
                ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                : getAnomalyStatus('rop', latestData.rop) === 'warning'
                ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
            }`}>
              <div className="flex items-center justify-between">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">ROP</div>
                {getAnomalyStatus('rop', latestData.rop) !== 'normal' && (
                  <AlertCircle className={`w-3 h-3 ${
                    getAnomalyStatus('rop', latestData.rop) === 'critical' ? 'text-red-500' : 'text-amber-500'
                  }`} />
                )}
              </div>
              <div className={`font-semibold ${
                getAnomalyStatus('rop', latestData.rop) === 'critical'
                  ? 'text-red-700 dark:text-red-300'
                  : getAnomalyStatus('rop', latestData.rop) === 'warning'
                  ? 'text-amber-700 dark:text-amber-300'
                  : 'text-slate-900 dark:text-white'
              }`}>{latestData.rop.toFixed(1)} ft/hr</div>
            </div>
            <div className={`rounded-xl border-2 px-4 py-3 ${
              getAnomalyStatus('mud_flow', latestData.mud_flow) === 'critical'
                ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                : getAnomalyStatus('mud_flow', latestData.mud_flow) === 'warning'
                ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
            }`}>
              <div className="flex items-center justify-between">
                <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Mud Flow</div>
                {getAnomalyStatus('mud_flow', latestData.mud_flow) !== 'normal' && (
                  <AlertCircle className={`w-3 h-3 ${
                    getAnomalyStatus('mud_flow', latestData.mud_flow) === 'critical' ? 'text-red-500' : 'text-amber-500'
                  }`} />
                )}
              </div>
              <div className={`font-semibold ${
                getAnomalyStatus('mud_flow', latestData.mud_flow) === 'critical'
                  ? 'text-red-700 dark:text-red-300'
                  : getAnomalyStatus('mud_flow', latestData.mud_flow) === 'warning'
                  ? 'text-amber-700 dark:text-amber-300'
                  : 'text-slate-900 dark:text-white'
              }`}>{latestData.mud_flow.toFixed(1)} gpm</div>
            </div>
            {latestData.mud_temperature !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('mud_temperature', latestData.mud_temperature) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('mud_temperature', latestData.mud_temperature) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Mud Temp</div>
                  {getAnomalyStatus('mud_temperature', latestData.mud_temperature) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('mud_temperature', latestData.mud_temperature) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('mud_temperature', latestData.mud_temperature) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('mud_temperature', latestData.mud_temperature) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.mud_temperature.toFixed(1)} °C</div>
              </div>
            )}
            {latestData.bit_temperature !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('bit_temperature', latestData.bit_temperature) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('bit_temperature', latestData.bit_temperature) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Bit Temp</div>
                  {getAnomalyStatus('bit_temperature', latestData.bit_temperature) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('bit_temperature', latestData.bit_temperature) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('bit_temperature', latestData.bit_temperature) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('bit_temperature', latestData.bit_temperature) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.bit_temperature.toFixed(1)} °C</div>
              </div>
            )}
            {latestData.motor_temperature !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('motor_temperature', latestData.motor_temperature) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('motor_temperature', latestData.motor_temperature) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Motor Temp</div>
                  {getAnomalyStatus('motor_temperature', latestData.motor_temperature) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('motor_temperature', latestData.motor_temperature) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('motor_temperature', latestData.motor_temperature) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('motor_temperature', latestData.motor_temperature) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.motor_temperature.toFixed(1)} °C</div>
              </div>
            )}
            {latestData.pump_pressure !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('pump_pressure', latestData.pump_pressure) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('pump_pressure', latestData.pump_pressure) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Pump Pressure</div>
                  {getAnomalyStatus('pump_pressure', latestData.pump_pressure) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('pump_pressure', latestData.pump_pressure) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('pump_pressure', latestData.pump_pressure) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('pump_pressure', latestData.pump_pressure) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.pump_pressure.toFixed(1)} psi</div>
              </div>
            )}
            {latestData.standpipe_pressure !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('standpipe_pressure', latestData.standpipe_pressure) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('standpipe_pressure', latestData.standpipe_pressure) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Standpipe Pressure</div>
                  {getAnomalyStatus('standpipe_pressure', latestData.standpipe_pressure) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('standpipe_pressure', latestData.standpipe_pressure) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('standpipe_pressure', latestData.standpipe_pressure) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('standpipe_pressure', latestData.standpipe_pressure) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.standpipe_pressure.toFixed(1)} psi</div>
              </div>
            )}
            {latestData.casing_pressure !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('casing_pressure', latestData.casing_pressure) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('casing_pressure', latestData.casing_pressure) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Casing Pressure</div>
                  {getAnomalyStatus('casing_pressure', latestData.casing_pressure) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('casing_pressure', latestData.casing_pressure) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('casing_pressure', latestData.casing_pressure) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('casing_pressure', latestData.casing_pressure) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.casing_pressure.toFixed(1)} psi</div>
              </div>
            )}
            {latestData.annulus_pressure !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('annulus_pressure', latestData.annulus_pressure) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('annulus_pressure', latestData.annulus_pressure) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Annulus Pressure</div>
                  {getAnomalyStatus('annulus_pressure', latestData.annulus_pressure) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('annulus_pressure', latestData.annulus_pressure) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('annulus_pressure', latestData.annulus_pressure) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('annulus_pressure', latestData.annulus_pressure) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.annulus_pressure.toFixed(1)} psi</div>
              </div>
            )}
            {latestData.flow_in !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('flow_in', latestData.flow_in) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('flow_in', latestData.flow_in) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Flow In</div>
                  {getAnomalyStatus('flow_in', latestData.flow_in) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('flow_in', latestData.flow_in) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('flow_in', latestData.flow_in) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('flow_in', latestData.flow_in) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.flow_in.toFixed(1)} gpm</div>
              </div>
            )}
            {latestData.flow_out !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('flow_out', latestData.flow_out) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('flow_out', latestData.flow_out) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Flow Out</div>
                  {getAnomalyStatus('flow_out', latestData.flow_out) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('flow_out', latestData.flow_out) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('flow_out', latestData.flow_out) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('flow_out', latestData.flow_out) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.flow_out.toFixed(1)} gpm</div>
              </div>
            )}
            {latestData.hook_load !== undefined && (
              <div className={`rounded-xl border-2 px-4 py-3 ${
                getAnomalyStatus('hook_load', latestData.hook_load) === 'critical'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/40'
                  : getAnomalyStatus('hook_load', latestData.hook_load) === 'warning'
                  ? 'border-amber-500 bg-amber-50 dark:bg-amber-900/40'
                  : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40'
              }`}>
                <div className="flex items-center justify-between">
                  <div className="text-slate-500 dark:text-slate-400 text-xs mb-1">Hook Load</div>
                  {getAnomalyStatus('hook_load', latestData.hook_load) !== 'normal' && (
                    <AlertCircle className={`w-3 h-3 ${
                      getAnomalyStatus('hook_load', latestData.hook_load) === 'critical' ? 'text-red-500' : 'text-amber-500'
                    }`} />
                  )}
                </div>
                <div className={`font-semibold ${
                  getAnomalyStatus('hook_load', latestData.hook_load) === 'critical'
                    ? 'text-red-700 dark:text-red-300'
                    : getAnomalyStatus('hook_load', latestData.hook_load) === 'warning'
                    ? 'text-amber-700 dark:text-amber-300'
                    : 'text-slate-900 dark:text-white'
                }`}>{latestData.hook_load.toFixed(0)} lbs</div>
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

      {/* No Data Message - only show if truly no data */}
      {dataPoints.length === 0 && !useMockData && connectionStatus === 'connected' && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-12 text-center">
          <AlertCircle className="w-12 h-12 text-slate-400 mx-auto mb-4" />
          <p className="text-slate-500 dark:text-slate-400">Waiting for real-time data...</p>
        </div>
      )}

      {/* Disconnected Message - only show if not using mock data */}
      {connectionStatus === 'disconnected' && !useMockData && dataPoints.length === 0 && (
        <div className="rounded-2xl bg-amber-50 dark:bg-amber-900/20 border border-amber-500 rounded-lg p-6 text-center">
          <AlertCircle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
          <p className="text-amber-600 dark:text-amber-400">WebSocket connection unavailable. Using demo data for visualization.</p>
        </div>
      )}
    </div>
  )
}
