import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { sensorDataApi, healthApi } from '@/services/api'
import { Activity, TrendingUp, Zap, AlertTriangle, WifiOff } from 'lucide-react'
import SystemStatusBar from '@/components/System/SystemStatusBar'

export default function Dashboard() {
  const { data: analyticsData, isLoading } = useQuery({
    queryKey: ['analytics'],
    queryFn: () => sensorDataApi.getAnalytics('RIG_01').then((res) => res.data.summary),
    refetchInterval: 60000, // Refresh every minute
  })

  const { data: serviceStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => healthApi.detailed().then((res) => res.data),
    refetchInterval: 20000,
    staleTime: 15000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-slate-400">Loading...</div>
      </div>
    )
  }

  const serviceDetails = serviceStatus?.details ?? serviceStatus ?? {}
  const connectionChips = useMemo(() => {
    return [
      { key: 'kafka', label: 'Kafka', status: serviceDetails.kafka?.status ?? serviceDetails.kafka?.kafka },
      { key: 'db', label: 'PostgreSQL', status: serviceDetails.database?.status ?? serviceDetails.database?.database },
      { key: 'rl', label: 'RL Environment', status: serviceDetails.rl_environment?.status ?? serviceDetails.rl_environment?.rl_environment },
      { key: 'mlflow', label: 'MLflow', status: serviceDetails.mlflow?.status ?? serviceDetails.mlflow?.mlflow },
    ].filter(Boolean)
  }, [serviceDetails])

  const degradedAlerts = useMemo(() => {
    const criticalStates = new Set(['unhealthy', 'unavailable', 'degraded'])
    return connectionChips
      .filter((chip) => criticalStates.has(String(chip.status ?? '').toLowerCase()))
      .map((chip) => ({
        title: `${chip.label}`,
        body: `Service ${chip.label} is in ${chip.status ?? 'unknown'} state. Please notify the support team.`,
      }))
  }, [connectionChips])

  const pendingAlerts = analyticsData?.maintenance_alerts_count ?? 0
  const notificationCards = [
    pendingAlerts > 0
      ? {
          title: 'Active Maintenance Alert',
          body: `${pendingAlerts} items require immediate attention.`,
        }
      : {
          title: 'Maintenance Alert',
          body: 'No alerts have been registered.',
        },
    ...degradedAlerts,
  ]

  const uniqueNotifications = useMemo(
    () =>
      notificationCards.filter(
        (card, index, self) =>
          self.findIndex((other) => other.title === card.title && other.body === card.body) === index,
      ),
    [notificationCards],
  )

  const stats = [
    {
      label: 'Current Depth',
      value: `${analyticsData?.current_depth?.toFixed(1) || 0} m`,
      icon: TrendingUp,
      color: 'bg-blue-500',
    },
    {
      label: 'Average ROP',
      value: `${analyticsData?.average_rop?.toFixed(2) || 0} m/h`,
      icon: Activity,
      color: 'bg-green-500',
    },
    {
      label: 'Total Energy Consumption',
      value: `${(analyticsData?.total_power_consumption / 1000 || 0).toFixed(1)} MWh`,
      icon: Zap,
      color: 'bg-yellow-500',
    },
    {
      label: 'Maintenance Alerts',
      value: `${analyticsData?.maintenance_alerts_count || 0}`,
      icon: AlertTriangle,
      color: 'bg-red-500',
    },
  ]

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Operations Dashboard</h1>
        <p className="text-slate-500 dark:text-slate-300">Overview of drilling operations status and service health</p>
      </div>

      <SystemStatusBar />

      <div className="flex flex-wrap items-center gap-2">
        {connectionChips.map((chip) => {
          const status = String(chip.status ?? 'unknown').toLowerCase()
          const tone =
            status === 'healthy' || status === 'available'
              ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/40'
              : status === 'degraded'
              ? 'bg-amber-500/10 text-amber-500 border border-amber-500/40'
              : 'bg-rose-500/10 text-rose-400 border border-rose-400/40'
          return (
            <span key={chip.key} className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${tone}`}>
              {status === 'unavailable' || status === 'unhealthy' ? <WifiOff className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />} {chip.label} : {chip.status ?? 'unknown'}
            </span>
          )
        })}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          return (
            <div
              key={index}
              className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`${stat.color} p-3 rounded-xl`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="text-slate-500 text-sm mb-1">{stat.label}</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{stat.value}</div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-4 lg:col-span-2">
          <h2 className="text-xl font-semibold">Operations Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-100/60 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400">Total Drilling Time</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {analyticsData?.total_drilling_time_hours?.toFixed(1) || 0} hours
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-100/60 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400">Last Update</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {analyticsData?.last_updated
                  ? new Date(analyticsData.last_updated).toLocaleString('en-US')
                  : 'unknown'}
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-3">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" /> Recent Notifications
          </h2>
          {uniqueNotifications.map((card, index) => (
            <div key={index} className="rounded-xl border border-amber-500/50 bg-amber-500/10 px-4 py-3 text-sm text-amber-900 dark:text-amber-200">
              <div className="font-semibold mb-1">{card.title}</div>
              <div>{card.body}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

