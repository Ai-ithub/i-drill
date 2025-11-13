import { FormEvent, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { maintenanceApi } from '@/services/api'
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts'
import { AlertTriangle, Clock, CheckCircle2, Calendar, Wrench, Loader2, TrendingUp, Activity } from 'lucide-react'

const toInputValue = (date: Date) => {
  const pad = (value: number) => value.toString().padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(
    date.getHours()
  )}:${pad(date.getMinutes())}`
}

export default function Maintenance() {
  const queryClient = useQueryClient()
  const now = useMemo(() => new Date(), [])

  const [rigFilter, setRigFilter] = useState('RIG_01')
  const [alertForm, setAlertForm] = useState({
    rig_id: 'RIG_01',
    component: 'pump',
    alert_type: 'overheat',
    severity: 'critical',
    message: '',
  })

  const [scheduleForm, setScheduleForm] = useState({
    rig_id: 'RIG_01',
    component: 'pump',
    maintenance_type: 'inspection',
    scheduled_date: toInputValue(new Date(now.getTime() + 24 * 60 * 60 * 1000)),
    estimated_duration_hours: 4,
    priority: 'high',
    status: 'scheduled',
    assigned_to: '',
    notes: '',
  })

  const alertsQuery = useQuery({
    queryKey: ['maintenance-alerts', rigFilter],
    queryFn: () => maintenanceApi.getAlerts(rigFilter || undefined).then((res) => res.data),
    placeholderData: (previousData) => previousData,
  })

  const scheduleQuery = useQuery({
    queryKey: ['maintenance-schedule', rigFilter],
    queryFn: () => maintenanceApi.getSchedule(rigFilter || undefined).then((res) => res.data),
    placeholderData: (previousData) => previousData,
  })

  const createAlertMutation = useMutation({
    mutationFn: (payload: typeof alertForm) => maintenanceApi.createAlert(payload).then((res) => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['maintenance-alerts', rigFilter] })
      setAlertForm((prev) => ({ ...prev, message: '' }))
    },
  })

  const createScheduleMutation = useMutation({
    mutationFn: (payload: typeof scheduleForm) =>
      maintenanceApi.createSchedule({
        ...payload,
        scheduled_date: new Date(payload.scheduled_date).toISOString(),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['maintenance-schedule', rigFilter] })
      setScheduleForm((prev) => ({
        ...prev,
        notes: '',
        assigned_to: '',
      }))
    },
  })

  const deleteScheduleMutation = useMutation({
    mutationFn: (scheduleId: string) => maintenanceApi.deleteSchedule(scheduleId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['maintenance-schedule', rigFilter] }),
  })

  const handleCreateAlert = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    createAlertMutation.mutate(alertForm)
  }

  const handleCreateSchedule = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    createScheduleMutation.mutate(scheduleForm)
  }

  const alerts = alertsQuery.data ?? []
  const schedules = scheduleQuery.data ?? []

  // Calculate statistics
  const stats = useMemo(() => {
    const criticalAlerts = alerts.filter((a: any) => a.severity === 'critical').length
    const highAlerts = alerts.filter((a: any) => a.severity === 'high').length
    const scheduledCount = schedules.filter((s: any) => s.status === 'scheduled').length
    const inProgressCount = schedules.filter((s: any) => s.status === 'in_progress').length
    const completedCount = schedules.filter((s: any) => s.status === 'completed').length
    const totalDuration = schedules.reduce((sum: number, s: any) => sum + (s.estimated_duration_hours || 0), 0)

    return {
      totalAlerts: alerts.length,
      criticalAlerts,
      highAlerts,
      scheduledCount,
      inProgressCount,
      completedCount,
      totalDuration,
    }
  }, [alerts, schedules])

  // Prepare chart data for alerts by severity
  const alertSeverityData = useMemo(() => {
    const severityCounts = {
      critical: alerts.filter((a: any) => a.severity === 'critical').length,
      high: alerts.filter((a: any) => a.severity === 'high').length,
      medium: alerts.filter((a: any) => a.severity === 'medium').length,
      low: alerts.filter((a: any) => a.severity === 'low').length,
    }
    return Object.entries(severityCounts).map(([key, value]) => ({
      name: key.toUpperCase(),
      value,
    }))
  }, [alerts])

  // Prepare chart data for schedule status
  const scheduleStatusData = useMemo(() => {
    const statusCounts = {
      scheduled: schedules.filter((s: any) => s.status === 'scheduled').length,
      in_progress: schedules.filter((s: any) => s.status === 'in_progress').length,
      completed: schedules.filter((s: any) => s.status === 'completed').length,
      cancelled: schedules.filter((s: any) => s.status === 'cancelled').length,
    }
    return Object.entries(statusCounts).map(([key, value]) => ({
      name: key.replace('_', ' ').toUpperCase(),
      value,
    }))
  }, [schedules])

  // Prepare chart data for alerts over time (last 7 days)
  const alertsOverTimeData = useMemo(() => {
    const last7Days = Array.from({ length: 7 }, (_, i) => {
      const date = new Date()
      date.setDate(date.getDate() - (6 - i))
      return date.toISOString().split('T')[0]
    })

    return last7Days.map((date) => {
      const dayAlerts = alerts.filter((a: any) => {
        if (!a.created_at) return false
        return new Date(a.created_at).toISOString().split('T')[0] === date
      })
      return {
        date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        alerts: dayAlerts.length,
      }
    })
  }, [alerts])

  const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#8b5cf6']

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2">Maintenance</h1>
          <p className="text-slate-500 dark:text-slate-300">
            Monitor alerts, schedule preventive maintenance, and manage execution plans
          </p>
        </div>
        <div className="space-y-2 md:space-y-0 md:flex md:items-center md:gap-3">
          <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Filter by Rig</label>
          <input
            value={rigFilter}
            onChange={(e) => setRigFilter(e.target.value)}
            className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
            placeholder="RIG_01"
          />
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7 gap-4">
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div className="text-xs text-slate-500 dark:text-slate-400">Total Alerts</div>
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-white">{stats.totalAlerts}</div>
        </div>
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <div className="text-xs text-slate-500 dark:text-slate-400">Critical</div>
          </div>
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.criticalAlerts}</div>
        </div>
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <div className="text-xs text-slate-500 dark:text-slate-400">High Priority</div>
          </div>
          <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">{stats.highAlerts}</div>
        </div>
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <Calendar className="w-5 h-5 text-blue-500" />
            <div className="text-xs text-slate-500 dark:text-slate-400">Scheduled</div>
          </div>
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.scheduledCount}</div>
        </div>
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-5 h-5 text-cyan-500" />
            <div className="text-xs text-slate-500 dark:text-slate-400">In Progress</div>
          </div>
          <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">{stats.inProgressCount}</div>
        </div>
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle2 className="w-5 h-5 text-green-500" />
            <div className="text-xs text-slate-500 dark:text-slate-400">Completed</div>
          </div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.completedCount}</div>
        </div>
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <Clock className="w-5 h-5 text-purple-500" />
            <div className="text-xs text-slate-500 dark:text-slate-400">Total Hours</div>
          </div>
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{stats.totalDuration}h</div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alerts by Severity */}
        {alertSeverityData.some((d) => d.value > 0) && (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Alerts by Severity</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={alertSeverityData.filter((d) => d.value > 0)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {alertSeverityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Schedule Status */}
        {scheduleStatusData.some((d) => d.value > 0) && (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Schedule Status</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={scheduleStatusData.filter((d) => d.value > 0)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#64748b' }} stroke="#64748b" />
                  <YAxis tick={{ fontSize: 12, fill: '#64748b' }} stroke="#64748b" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar dataKey="value" fill="#06b6d4" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Alerts Over Time */}
        {alertsOverTimeData.some((d) => d.alerts > 0) && (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Alerts Over Time (7 Days)</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={alertsOverTimeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
                  <XAxis dataKey="date" tick={{ fontSize: 12, fill: '#64748b' }} stroke="#64748b" />
                  <YAxis tick={{ fontSize: 12, fill: '#64748b' }} stroke="#64748b" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                  />
                  <Line type="monotone" dataKey="alerts" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="space-y-6 xl:col-span-2">
          <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Active Alerts</h2>
              {alertsQuery.isFetching && (
                <span className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Updating...
                </span>
              )}
            </div>

            {alertsQuery.isError ? (
              <div className="rounded-xl border border-red-500/40 bg-red-50 dark:bg-red-900/20 px-4 py-3 text-sm text-red-700 dark:text-red-300">
                Error fetching alerts
              </div>
            ) : alerts.length === 0 ? (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-4 py-8 text-center text-slate-500 dark:text-slate-400">
                No active alerts to display.
              </div>
            ) : (
              <div className="space-y-4">
                {alerts.map((alert: any) => (
                  <div
                    key={alert.id}
                    className="rounded-xl border border-red-500/30 bg-red-50 dark:bg-red-500/5 p-6 space-y-3"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Rig</div>
                        <div className="text-lg font-semibold text-slate-900 dark:text-white">{alert.rig_id}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Component</div>
                        <div className="text-lg font-semibold text-slate-900 dark:text-white">{alert.component}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Severity</div>
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                          alert.severity === 'critical'
                            ? 'bg-red-100 dark:bg-red-500/10 border border-red-300 dark:border-red-500/40 text-red-700 dark:text-red-300'
                            : alert.severity === 'high'
                            ? 'bg-amber-100 dark:bg-amber-500/10 border border-amber-300 dark:border-amber-500/40 text-amber-700 dark:text-amber-300'
                            : 'bg-blue-100 dark:bg-blue-500/10 border border-blue-300 dark:border-blue-500/40 text-blue-700 dark:text-blue-300'
                        }`}>
                          {alert.severity}
                        </span>
                      </div>
                    </div>
                    <p className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed">{alert.message}</p>
                    <div className="text-xs text-slate-500 dark:text-slate-400 flex justify-between pt-2 border-t border-slate-200 dark:border-slate-700">
                      <span className="flex items-center gap-2">
                        <Clock className="w-3 h-3" />
                        Created:{' '}
                        {alert.created_at
                          ? new Date(alert.created_at).toLocaleString('en-US')
                          : '-'}
                      </span>
                      <span>{alert.alert_type}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Maintenance Schedule</h2>
              {scheduleQuery.isFetching && (
                <span className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Updating...
                </span>
              )}
            </div>

            {scheduleQuery.isError ? (
              <div className="rounded-xl border border-red-500/40 bg-red-50 dark:bg-red-900/20 px-4 py-3 text-sm text-red-700 dark:text-red-300">
                Error fetching maintenance schedule
              </div>
            ) : schedules.length === 0 ? (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-4 py-8 text-center text-slate-500 dark:text-slate-400">
                No schedule has been registered.
              </div>
            ) : (
              <div className="space-y-4">
                {schedules.map((schedule: any) => (
                  <div
                    key={schedule.id}
                    className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-6 space-y-3"
                  >
                    <div className="flex flex-wrap justify-between gap-4">
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Rig</div>
                        <div className="text-lg font-semibold text-slate-900 dark:text-white">{schedule.rig_id}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Component</div>
                        <div className="text-lg font-semibold text-slate-900 dark:text-white">{schedule.component}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Status</div>
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                          schedule.status === 'scheduled'
                            ? 'bg-blue-100 dark:bg-blue-500/10 border border-blue-300 dark:border-blue-500/40 text-blue-700 dark:text-blue-300'
                            : schedule.status === 'in_progress'
                            ? 'bg-cyan-100 dark:bg-cyan-500/10 border border-cyan-300 dark:border-cyan-500/40 text-cyan-700 dark:text-cyan-300'
                            : schedule.status === 'completed'
                            ? 'bg-green-100 dark:bg-green-500/10 border border-green-300 dark:border-green-500/40 text-green-700 dark:text-green-300'
                            : 'bg-slate-100 dark:bg-slate-500/10 border border-slate-300 dark:border-slate-500/40 text-slate-700 dark:text-slate-300'
                        }`}>
                          {schedule.status}
                        </span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm text-slate-700 dark:text-slate-200">
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Maintenance Type</div>
                        <div className="font-medium">{schedule.maintenance_type}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Priority</div>
                        <div className="font-medium">{schedule.priority}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Scheduled Date</div>
                        <div className="font-medium">
                          {schedule.scheduled_date
                            ? new Date(schedule.scheduled_date).toLocaleString('en-US')
                            : '-'}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Duration</div>
                        <div className="font-medium">{schedule.estimated_duration_hours} hours</div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-slate-700">
                      <div className="text-xs text-slate-500 dark:text-slate-400">
                        Assigned to: {schedule.assigned_to || 'unknown'}
                      </div>
                      <button
                        onClick={() => deleteScheduleMutation.mutate(String(schedule.id))}
                        className="text-xs text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 font-medium transition"
                        disabled={deleteScheduleMutation.isLoading}
                      >
                        {deleteScheduleMutation.isLoading ? 'Deleting...' : 'Delete Schedule'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>

        <div className="space-y-6">
          <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 space-y-4 shadow-sm">
            <div>
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Create New Alert</h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Create an alert to notify the maintenance team and enable quick planning
              </p>
            </div>
            <form className="space-y-4" onSubmit={handleCreateAlert}>
              <div className="grid grid-cols-1 gap-3">
                <input
                  value={alertForm.rig_id}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, rig_id: e.target.value }))}
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Rig ID"
                />
                <input
                  value={alertForm.component}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, component: e.target.value }))}
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Component"
                />
                <input
                  value={alertForm.alert_type}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, alert_type: e.target.value }))}
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Alert Type"
                />
                <select
                  value={alertForm.severity}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, severity: e.target.value }))}
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                >
                  <option value="low">LOW</option>
                  <option value="medium">MEDIUM</option>
                  <option value="high">HIGH</option>
                  <option value="critical">CRITICAL</option>
                </select>
                <textarea
                  rows={3}
                  value={alertForm.message}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, message: e.target.value }))}
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Description"
                />
              </div>
              <button
                type="submit"
                className="w-full h-10 rounded-md bg-red-500 hover:bg-red-600 text-white font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                disabled={createAlertMutation.isLoading}
              >
                {createAlertMutation.isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4" />
                    Create Alert
                  </>
                )}
              </button>
              {createAlertMutation.isError && (
                <div className="rounded-xl border border-red-500/40 bg-red-50 dark:bg-red-900/20 px-3 py-2 text-xs text-red-700 dark:text-red-300">
                  Error creating alert
                </div>
              )}
              {createAlertMutation.isSuccess && (
                <div className="rounded-xl border border-emerald-500/40 bg-emerald-50 dark:bg-emerald-900/20 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">
                  Alert created successfully
                </div>
              )}
            </form>
          </section>

          <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-6 space-y-4 shadow-sm">
            <div>
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Schedule Maintenance</h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Create a new schedule for the preventive maintenance team
              </p>
            </div>
            <form className="space-y-4" onSubmit={handleCreateSchedule}>
              <div className="grid grid-cols-1 gap-3">
                <input
                  value={scheduleForm.rig_id}
                  onChange={(e) => setScheduleForm((prev) => ({ ...prev, rig_id: e.target.value }))}
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Rig ID"
                />
                <input
                  value={scheduleForm.component}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, component: e.target.value }))
                  }
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Component"
                />
                <input
                  value={scheduleForm.maintenance_type}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, maintenance_type: e.target.value }))
                  }
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Maintenance Type"
                />
                <input
                  type="datetime-local"
                  value={scheduleForm.scheduled_date}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, scheduled_date: e.target.value }))
                  }
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                />
                <input
                  type="number"
                  min={1}
                  step={0.5}
                  value={scheduleForm.estimated_duration_hours}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({
                      ...prev,
                      estimated_duration_hours: Number(e.target.value),
                    }))
                  }
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Duration (hours)"
                />
                <select
                  value={scheduleForm.priority}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, priority: e.target.value }))
                  }
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                >
                  <option value="low">LOW</option>
                  <option value="medium">MEDIUM</option>
                  <option value="high">HIGH</option>
                </select>
                <select
                  value={scheduleForm.status}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, status: e.target.value }))
                  }
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                >
                  <option value="scheduled">SCHEDULED</option>
                  <option value="in_progress">IN PROGRESS</option>
                  <option value="completed">COMPLETED</option>
                  <option value="cancelled">CANCELLED</option>
                </select>
                <input
                  value={scheduleForm.assigned_to}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, assigned_to: e.target.value }))
                  }
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Assigned To"
                />
                <textarea
                  rows={3}
                  value={scheduleForm.notes}
                  onChange={(e) => setScheduleForm((prev) => ({ ...prev, notes: e.target.value }))}
                  className="rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Description"
                />
              </div>
              <button
                type="submit"
                className="w-full h-10 rounded-md bg-emerald-500 hover:bg-emerald-600 text-white font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                disabled={createScheduleMutation.isLoading}
              >
                {createScheduleMutation.isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Calendar className="w-4 h-4" />
                    Create Schedule
                  </>
                )}
              </button>
              {createScheduleMutation.isError && (
                <div className="rounded-xl border border-red-500/40 bg-red-50 dark:bg-red-900/20 px-3 py-2 text-xs text-red-700 dark:text-red-300">
                  Error creating schedule
                </div>
              )}
              {createScheduleMutation.isSuccess && (
                <div className="rounded-xl border border-emerald-500/40 bg-emerald-50 dark:bg-emerald-900/20 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">
                  Schedule created successfully
                </div>
              )}
            </form>
          </section>
        </div>
      </div>
    </div>
  )
}

