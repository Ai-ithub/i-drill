import { useQuery } from 'react-query'
import { sensorDataApi } from '@/services/api'
import { Activity, TrendingUp, Zap, AlertTriangle } from 'lucide-react'

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
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">داشبورد</h1>
        <p className="text-slate-400">نمای کلی از وضعیت عملیات حفاری</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          return (
            <div
              key={index}
              className="bg-slate-800 rounded-lg p-6 border border-slate-700"
            >
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-semibold text-white mb-4">خلاصه عملیات</h2>
          <div className="space-y-3">
            <div className="flex justify-between text-slate-300">
              <span>زمان حفاری کل:</span>
              <span className="font-semibold">
                {analyticsData?.total_drilling_time_hours?.toFixed(1) || 0} ساعت
              </span>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>آخرین بروزرسانی:</span>
              <span className="font-semibold">
                {analyticsData?.last_updated
                  ? new Date(analyticsData.last_updated).toLocaleString('fa-IR')
                  : 'نامشخص'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

