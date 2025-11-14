import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { sensorDataApi, healthApi } from '@/services/api'
import { Activity, TrendingUp, Zap, AlertTriangle, WifiOff } from 'lucide-react'
import SystemStatusBar from '@/components/System/SystemStatusBar'
import { Loading, ErrorDisplay, Card, EmptyState } from '@/components/UI'

export default function Dashboard() {
  // Mock data for demonstration when backend is not available
  const mockAnalyticsData = {
    current_depth: 5000,
    average_rop: 25.5,
    total_drilling_time_hours: 120,
    total_power_consumption: 500000,
    maintenance_alerts_count: 2,
    last_updated: new Date().toISOString(),
  }

  const mockServiceStatus = {
    details: {
      kafka: {
        status: 'healthy',
        kafka: 'healthy',
        message: 'Kafka is healthy',
      },
      database: {
        status: 'healthy',
        database: 'healthy',
        message: 'PostgreSQL is healthy',
      },
      rl_environment: {
        status: 'available',
        rl_environment: 'available',
        message: 'RL environment is available',
      },
      mlflow: {
        status: 'available',
        mlflow: 'available',
        message: 'MLflow is available',
      },
    },
  }

  const { data: analyticsData, isLoading: analyticsLoading, error: analyticsError } = useQuery({
    queryKey: ['analytics', 'RIG_01'],
    queryFn: () => sensorDataApi.getAnalytics('RIG_01').then((res) => res.data.summary),
    refetchInterval: 60000, // Refresh every minute
    retry: 1, // Reduced retry to show mock data faster
    retryDelay: 1000,
    staleTime: 30000,
    refetchOnWindowFocus: false,
  })

  const { data: serviceStatus, isLoading: serviceLoading, error: healthError } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => healthApi.detailed().then((res) => res.data),
    refetchInterval: 20000,
    staleTime: 15000,
    retry: 1, // Reduced retry to show mock data faster
    retryDelay: 1000,
    refetchOnWindowFocus: false,
  })

  // Use actual data or fallback to mock data
  const effectiveAnalyticsData = analyticsData || mockAnalyticsData
  const effectiveServiceStatus = serviceStatus || mockServiceStatus
  
  const serviceDetails = effectiveServiceStatus?.details ?? effectiveServiceStatus ?? {}
  
  // Only show loading if we don't have mock data fallback yet
  const isLoading = (analyticsLoading && !analyticsError && !mockAnalyticsData) || (serviceLoading && !healthError && !mockServiceStatus)
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

  const pendingAlerts = effectiveAnalyticsData?.maintenance_alerts_count ?? 0
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
      value: `${effectiveAnalyticsData?.current_depth?.toFixed(1) || 0} m`,
      icon: TrendingUp,
      color: 'bg-blue-500',
    },
    {
      label: 'Average ROP',
      value: `${effectiveAnalyticsData?.average_rop?.toFixed(2) || 0} m/h`,
      icon: Activity,
      color: 'bg-green-500',
    },
    {
      label: 'Total Energy Consumption',
      value: `${(effectiveAnalyticsData?.total_power_consumption / 1000 || 0).toFixed(1)} MWh`,
      icon: Zap,
      color: 'bg-yellow-500',
    },
    {
      label: 'Maintenance Alerts',
      value: `${effectiveAnalyticsData?.maintenance_alerts_count || 0}`,
      icon: AlertTriangle,
      color: 'bg-red-500',
    },
  ]

  // Overview tab content component
  const OverviewContent = () => (
    <>
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
            <Card key={index} variant="default" padding="md">
              <div className="flex items-center justify-between mb-4">
                <div className={`${stat.color} p-3 rounded-xl`}>
                  <Icon className="w-6 h-6 text-white" aria-hidden="true" />
                </div>
              </div>
              <div className="text-slate-500 dark:text-slate-400 text-sm mb-1">{stat.label}</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{stat.value}</div>
            </Card>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card variant="default" padding="md" className="lg:col-span-2">
          <Card.Header title="Operations Summary" />
          <Card.Content>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-100/60 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400">Total Drilling Time</div>
                <div className="text-lg font-semibold text-slate-900 dark:text-white">
                  {effectiveAnalyticsData?.total_drilling_time_hours?.toFixed(1) || 0} hours
                </div>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-100/60 dark:bg-slate-800/40">
                <div className="text-slate-500 dark:text-slate-400">Last Update</div>
                <div className="text-lg font-semibold text-slate-900 dark:text-white">
                  {effectiveAnalyticsData?.last_updated
                    ? new Date(effectiveAnalyticsData.last_updated).toLocaleString('en-US')
                    : new Date().toLocaleString('en-US')}
                </div>
                {!analyticsData && (
                  <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">(Demo Data)</div>
                )}
              </div>
            </div>
          </Card.Content>
        </Card>

        <Card variant="default" padding="md">
          <Card.Header
            title={
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-500" aria-hidden="true" />
                Recent Notifications
              </div>
            }
          />
          <Card.Content>
            {uniqueNotifications.length > 0 ? (
              <div className="space-y-3">
                {uniqueNotifications.map((card, index) => (
                  <div
                    key={index}
                    className="rounded-xl border border-amber-500/50 bg-amber-500/10 px-4 py-3 text-sm text-amber-900 dark:text-amber-200"
                  >
                    <div className="font-semibold mb-1">{card.title}</div>
                    <div>{card.body}</div>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState
                variant="default"
                title="No notifications"
                description="You're all caught up! No new notifications at this time."
              />
            )}
          </Card.Content>
        </Card>
      </div>
    </>
  )

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Operations Dashboard</h1>
        <p className="text-slate-500 dark:text-slate-300">Overview of drilling operations status and service health</p>
      </div>

      {/* Dashboard Content */}
      <div className="mt-6">
        {isLoading ? (
          <div className="space-y-6">
            <Loading.SkeletonText lines={3} />
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {Array.from({ length: 4 }).map((_, i) => (
                <Card key={i} padding="md">
                  <Loading.Skeleton height={60} />
                </Card>
              ))}
            </div>
          </div>
        ) : (
          <OverviewContent />
        )}
      </div>
    </div>
  )
}

