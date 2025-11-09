import { useQuery } from 'react-query'
import { sensorDataApi } from '@/services/api'
import { Activity, TrendingUp, Zap, AlertTriangle } from 'lucide-react'
import SystemStatusBar from '@/components/System/SystemStatusBar'

export default function Dashboard() {
  const { data: analyticsData, isLoading } = useQuery(
    'analytics',
    () => sensorDataApi.getAnalytics('RIG_01').then((res) => res.data.summary),
    {
      refetchInterval: 60000, // Refresh every minute
    }
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-slate-400">در حال بارگذاری...</div>
      </div>
    )
  }

  const pendingAlerts = analyticsData?.maintenance_alerts_count ?? 0
  const notificationCards = [
    pendingAlerts > 0
      ? {
          title: 'هشدار نگهداشت فعال',
          body: `${pendingAlerts} مورد نیازمند رسیدگی فوری است.`,
        }
      : {
          title: 'هشدار نگهداشت',
          body: 'هیچ هشداری ثبت نشده است.',
        },
  ]

  const stats = [
    {
      label: 'عمق فعلی',
      value: `${analyticsData?.current_depth?.toFixed(1) || 0} متر`,
      icon: TrendingUp,
      color: 'bg-blue-500',
    },
    {
      label: 'میانگین ROP',
      value: `${analyticsData?.average_rop?.toFixed(2) || 0} m/h`,
      icon: Activity,
      color: 'bg-green-500',
    },
    {
      label: 'مصرف انرژی کل',
      value: `${(analyticsData?.total_power_consumption / 1000 || 0).toFixed(1)} MWh`,
      icon: Zap,
      color: 'bg-yellow-500',
    },
    {
      label: 'هشدارهای تعمیر',
      value: `${analyticsData?.maintenance_alerts_count || 0}`,
      icon: AlertTriangle,
      color: 'bg-red-500',
    },
  ]

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">داشبورد عملیات</h1>
        <p className="text-slate-500 dark:text-slate-300">نمای کلی از وضعیت عملیات حفاری و سلامت سرویس‌ها</p>
      </div>

      <SystemStatusBar />

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
          <h2 className="text-xl font-semibold">خلاصه عملیات</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-100/60 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400">زمان حفاری کل</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {analyticsData?.total_drilling_time_hours?.toFixed(1) || 0} ساعت
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 px-4 py-3 bg-slate-100/60 dark:bg-slate-800/40">
              <div className="text-slate-500 dark:text-slate-400">آخرین بروزرسانی</div>
              <div className="text-lg font-semibold text-slate-900 dark:text-white">
                {analyticsData?.last_updated
                  ? new Date(analyticsData.last_updated).toLocaleString('fa-IR')
                  : 'نامشخص'}
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-3">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" /> اعلان‌های اخیر
          </h2>
          {notificationCards.map((card, index) => (
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

