import { useQuery } from 'react-query'
import { maintenanceApi } from '@/services/api'
import { BellRing } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'

export default function NotificationBadge() {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const { data: alerts, isLoading } = useQuery(
    'maintenance-alerts-header',
    () => maintenanceApi.getAlerts().then((res) => res.data),
    {
      refetchInterval: 30000,
      staleTime: 15000,
    },
  )

  const alertCount = Array.isArray(alerts) ? alerts.filter((alert: any) => !alert.resolved).length : 0

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  const activeAlerts = Array.isArray(alerts) ? alerts.filter((alert: any) => !alert.resolved).slice(0, 5) : []

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
      >
        <BellRing className="w-4 h-4 text-amber-500" />
        {alertCount > 0 && (
          <span className="absolute -top-1 -right-1 bg-red-500 text-white text-[10px] font-bold rounded-full w-5 h-5 flex items-center justify-center">
            {alertCount > 9 ? '9+' : alertCount}
          </span>
        )}
        <span className="hidden md:inline">
          {isLoading ? '...' : alertCount > 0 ? `${alertCount} new alert${alertCount !== 1 ? 's' : ''}` : 'No alerts'}
        </span>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
          <div className="p-3 border-b border-slate-200 dark:border-slate-700">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Notifications</h3>
          </div>
          <div className="divide-y divide-slate-200 dark:divide-slate-700">
            {isLoading ? (
              <div className="p-4 text-sm text-slate-500 dark:text-slate-400 text-center">Loading...</div>
            ) : activeAlerts.length === 0 ? (
              <div className="p-4 text-sm text-slate-500 dark:text-slate-400 text-center">No active alerts</div>
            ) : (
              activeAlerts.map((alert: any) => (
                <div key={alert.id} className="p-3 hover:bg-slate-50 dark:hover:bg-slate-800">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1">
                      <div className="text-xs font-semibold text-slate-900 dark:text-white">{alert.rig_id}</div>
                      <div className="text-xs text-slate-600 dark:text-slate-300 mt-1">{alert.component}</div>
                      <div className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">{alert.message}</div>
                    </div>
                    <span
                      className={`px-2 py-1 text-[10px] font-semibold rounded-full ${
                        alert.severity === 'high'
                          ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                          : alert.severity === 'medium'
                          ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                          : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                      }`}
                    >
                      {alert.severity}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-2">
                    {alert.created_at ? new Date(alert.created_at).toLocaleString('en-US') : ''}
                  </div>
                </div>
              ))
            )}
          </div>
          {activeAlerts.length > 0 && (
            <div className="p-2 border-t border-slate-200 dark:border-slate-700">
              <a
                href="/maintenance"
                className="block text-center text-xs text-cyan-600 dark:text-cyan-400 hover:underline"
                onClick={() => setIsOpen(false)}
              >
                View all alerts
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

