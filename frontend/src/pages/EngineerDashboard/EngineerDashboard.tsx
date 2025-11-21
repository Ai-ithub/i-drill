import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { sensorDataApi } from '@/services/api'
import { TrendingUp, Activity, Zap, BarChart3, Target, AlertTriangle } from 'lucide-react'
import { Card, Loading, ErrorDisplay } from '@/components/UI'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  AreaChart,
  Area,
} from 'recharts'

export default function EngineerDashboard() {
  const rigId = 'RIG_01'
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('24h')

  // Fetch analytics data
  const { data: analyticsData, isLoading: analyticsLoading } = useQuery({
    queryKey: ['engineer-analytics', rigId],
    queryFn: () => sensorDataApi.getAnalytics(rigId).then((res) => res.data.summary),
    refetchInterval: 60000,
    retry: 1,
  })

  // Fetch performance metrics
  const { data: performanceData, isLoading: performanceLoading } = useQuery({
    queryKey: ['engineer-performance', rigId],
    queryFn: async () => {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8001'}/api/v1/optimization/realtime/performance/${rigId}`,
        { credentials: 'include' }
      )
      if (!response.ok) throw new Error('Failed to fetch performance metrics')
      return response.json()
    },
    refetchInterval: 30000,
    retry: 1,
  })

  // Fetch optimization recommendations
  const { data: recommendationsData, isLoading: recommendationsLoading } = useQuery({
    queryKey: ['engineer-recommendations', rigId],
    queryFn: async () => {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8001'}/api/v1/optimization/realtime/recommendations/${rigId}`,
        { credentials: 'include' }
      )
      if (!response.ok) throw new Error('Failed to fetch recommendations')
      return response.json()
    },
    refetchInterval: 60000,
    retry: 1,
  })

  // Fetch historical data for charts
  const { data: historicalData, isLoading: historicalLoading } = useQuery({
    queryKey: ['engineer-historical', rigId, selectedTimeRange],
    queryFn: () => {
      const endTime = new Date()
      const startTime = new Date()
      switch (selectedTimeRange) {
        case '1h':
          startTime.setHours(startTime.getHours() - 1)
          break
        case '6h':
          startTime.setHours(startTime.getHours() - 6)
          break
        case '24h':
          startTime.setHours(startTime.getHours() - 24)
          break
        case '7d':
          startTime.setDate(startTime.getDate() - 7)
          break
      }
      return sensorDataApi.getHistorical({
        rig_id: rigId,
        start_time: startTime.toISOString(),
        end_time: endTime.toISOString(),
        limit: 100,
      }).then((res) => res.data.data || [])
    },
    refetchInterval: 30000,
    retry: 1,
  })

  // Mock data fallback
  const analytics = analyticsData || {
    current_depth: 5000,
    average_rop: 25.5,
    total_drilling_time_hours: 120,
    total_power_consumption: 500000,
    maintenance_alerts_count: 2,
  }

  const performance = performanceData?.metrics || {
    rop_efficiency: { current: 75.5, unit: '%' },
    energy_efficiency: { current: 12.3, average: 11.8, unit: 'm/kWh' },
    drilling_efficiency_index: { current: 82.1, unit: 'index' },
  }

  const recommendations = recommendationsData?.recommendations || []

  // Prepare chart data
  const chartData = (historicalData || []).map((item: any) => ({
    time: new Date(item.timestamp).toLocaleTimeString(),
    depth: item.depth || 0,
    wob: (item.wob || 0) / 1000, // Convert to tons
    rpm: item.rpm || 0,
    rop: item.rop || 0,
    torque: (item.torque || 0) / 1000, // Convert to kN·m
  }))

  const isLoading = analyticsLoading || performanceLoading || recommendationsLoading || historicalLoading

  if (isLoading && !analyticsData) {
    return (
      <div className="space-y-6">
        <Loading.SkeletonText lines={3} />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} padding="md">
              <Loading.Skeleton height={100} />
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Engineer's Dashboard</h1>
          <p className="text-slate-500 dark:text-slate-300 mt-1">
            Detailed analysis, performance metrics, and optimization recommendations
          </p>
        </div>

        {/* Time Range Selector */}
        <div className="flex gap-2">
          {(['1h', '6h', '24h', '7d'] as const).map((range) => (
            <button
              key={range}
              onClick={() => setSelectedTimeRange(range)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedTimeRange === range
                  ? 'bg-cyan-500 text-white'
                  : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-700'
              }`}
            >
              {range.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Performance Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card padding="md">
          <div className="flex items-center justify-between mb-4">
            <div className="bg-blue-500 p-3 rounded-xl">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
          </div>
          <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">ROP Efficiency</div>
          <div className="text-2xl font-bold">{performance.rop_efficiency.current.toFixed(1)}%</div>
          <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">
            Target: &gt;70%
          </div>
        </Card>

        <Card padding="md">
          <div className="flex items-center justify-between mb-4">
            <div className="bg-green-500 p-3 rounded-xl">
              <Zap className="w-6 h-6 text-white" />
            </div>
          </div>
          <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">Energy Efficiency</div>
          <div className="text-2xl font-bold">
            {performance.energy_efficiency.current.toFixed(2)} {performance.energy_efficiency.unit}
          </div>
          <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">
            Avg: {performance.energy_efficiency.average.toFixed(2)}
          </div>
        </Card>

        <Card padding="md">
          <div className="flex items-center justify-between mb-4">
            <div className="bg-purple-500 p-3 rounded-xl">
              <Activity className="w-6 h-6 text-white" />
            </div>
          </div>
          <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">Drilling Efficiency Index</div>
          <div className="text-2xl font-bold">
            {performance.drilling_efficiency_index.current.toFixed(1)}
          </div>
          <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">
            Higher is better
          </div>
        </Card>

        <Card padding="md">
          <div className="flex items-center justify-between mb-4">
            <div className="bg-amber-500 p-3 rounded-xl">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
          </div>
          <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">Current Depth</div>
          <div className="text-2xl font-bold">{analytics.current_depth?.toFixed(1) || 0} m</div>
          <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">
            Avg ROP: {analytics.average_rop?.toFixed(2) || 0} m/h
          </div>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROP Trend */}
        <Card padding="md">
          <Card.Header title="Rate of Penetration Trend" />
          <Card.Content>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-slate-300 dark:stroke-slate-700" />
                <XAxis dataKey="time" className="text-xs" />
                <YAxis className="text-xs" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="rop"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.3}
                  name="ROP (m/h)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card.Content>
        </Card>

        {/* WOB vs RPM */}
        <Card padding="md">
          <Card.Header title="WOB vs RPM" />
          <Card.Content>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-slate-300 dark:stroke-slate-700" />
                <XAxis dataKey="time" className="text-xs" />
                <YAxis yAxisId="left" className="text-xs" />
                <YAxis yAxisId="right" orientation="right" className="text-xs" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="wob"
                  stroke="#ef4444"
                  name="WOB (tons)"
                  strokeWidth={2}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="rpm"
                  stroke="#10b981"
                  name="RPM"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card.Content>
        </Card>

        {/* Torque Distribution */}
        <Card padding="md">
          <Card.Header title="Torque Distribution" />
          <Card.Content>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-slate-300 dark:stroke-slate-700" />
                <XAxis dataKey="time" className="text-xs" />
                <YAxis className="text-xs" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="torque" fill="#8b5cf6" name="Torque (kN·m)" />
              </BarChart>
            </ResponsiveContainer>
          </Card.Content>
        </Card>

        {/* Depth Progress */}
        <Card padding="md">
          <Card.Header title="Depth Progress" />
          <Card.Content>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-slate-300 dark:stroke-slate-700" />
                <XAxis dataKey="time" className="text-xs" />
                <YAxis className="text-xs" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="depth"
                  stroke="#f59e0b"
                  fill="#f59e0b"
                  fillOpacity={0.3}
                  name="Depth (m)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card.Content>
        </Card>
      </div>

      {/* Optimization Recommendations */}
      {recommendations.length > 0 && (
        <Card padding="md">
          <Card.Header
            title={
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-500" />
                Optimization Recommendations
              </div>
            }
          />
          <Card.Content>
            <div className="space-y-4">
              {recommendations.map((rec: any, index: number) => (
                <div
                  key={index}
                  className={`rounded-lg border px-4 py-3 ${
                    rec.priority === 'high'
                      ? 'border-cyan-500/50 bg-cyan-500/10'
                      : rec.priority === 'medium'
                      ? 'border-blue-500/50 bg-blue-500/10'
                      : 'border-slate-300/50 dark:border-slate-700 bg-slate-100/50 dark:bg-slate-800/50'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex-1">
                      <div className="font-semibold text-lg mb-1">{rec.title}</div>
                      <div className="text-sm text-slate-600 dark:text-slate-300 mb-2">
                        {rec.description}
                      </div>
                      {rec.suggestions && rec.suggestions.length > 0 && (
                        <ul className="list-disc list-inside space-y-1 text-sm text-slate-600 dark:text-slate-400">
                          {rec.suggestions.map((suggestion: string, i: number) => (
                            <li key={i}>{suggestion}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        rec.priority === 'high'
                          ? 'bg-cyan-500 text-white'
                          : rec.priority === 'medium'
                          ? 'bg-blue-500 text-white'
                          : 'bg-slate-500 text-white'
                      }`}
                    >
                      {rec.priority.toUpperCase()}
                    </span>
                  </div>
                  {rec.expected_improvement && (
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                      Expected: {rec.expected_improvement}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card.Content>
        </Card>
      )}

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card padding="md">
          <div className="text-slate-500 dark:text-slate-400 text-sm mb-2">Total Drilling Time</div>
          <div className="text-3xl font-bold">{analytics.total_drilling_time_hours?.toFixed(1) || 0} hours</div>
        </Card>

        <Card padding="md">
          <div className="text-slate-500 dark:text-slate-400 text-sm mb-2">Total Energy Consumption</div>
          <div className="text-3xl font-bold">
            {(analytics.total_power_consumption / 1000 || 0).toFixed(1)} MWh
          </div>
        </Card>

        <Card padding="md">
          <div className="text-slate-500 dark:text-slate-400 text-sm mb-2">Maintenance Alerts</div>
          <div className="text-3xl font-bold text-amber-500">
            {analytics.maintenance_alerts_count || 0}
          </div>
        </Card>
      </div>
    </div>
  )
}

