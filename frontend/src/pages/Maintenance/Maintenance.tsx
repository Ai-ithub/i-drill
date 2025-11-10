import { FormEvent, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { maintenanceApi } from '@/services/api'

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

  const alertsQuery = useQuery(
    ['maintenance-alerts', rigFilter],
    () => maintenanceApi.getAlerts(rigFilter || undefined).then((res) => res.data),
    {
      keepPreviousData: true,
    }
  )

  const scheduleQuery = useQuery(
    ['maintenance-schedule', rigFilter],
    () => maintenanceApi.getSchedule(rigFilter || undefined).then((res) => res.data),
    {
      keepPreviousData: true,
    }
  )

  const createAlertMutation = useMutation(
    (payload: typeof alertForm) => maintenanceApi.createAlert(payload).then((res) => res.data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['maintenance-alerts', rigFilter])
        setAlertForm((prev) => ({ ...prev, message: '' }))
      },
    }
  )

  const createScheduleMutation = useMutation(
    (payload: typeof scheduleForm) =>
      maintenanceApi.createSchedule({
        ...payload,
        scheduled_date: new Date(payload.scheduled_date).toISOString(),
      }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['maintenance-schedule', rigFilter])
        setScheduleForm((prev) => ({
          ...prev,
          notes: '',
          assigned_to: '',
        }))
      },
    }
  )

  const deleteScheduleMutation = useMutation(
    (scheduleId: string) => maintenanceApi.deleteSchedule(scheduleId),
    {
      onSuccess: () => queryClient.invalidateQueries(['maintenance-schedule', rigFilter]),
    }
  )

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

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
    <div>
      <h1 className="text-3xl font-bold text-white mb-2">Maintenance</h1>
          <p className="text-slate-400">
            Monitor alerts, schedule preventive maintenance, and manage execution plans
          </p>
        </div>
        <div className="space-y-2 md:space-y-0 md:flex md:items-center md:gap-3">
          <label className="text-sm text-slate-400">Filter by Rig</label>
          <input
            value={rigFilter}
            onChange={(e) => setRigFilter(e.target.value)}
            className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
            placeholder="RIG_01"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="space-y-6 xl:col-span-2">
          <section className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">Active Alerts</h2>
              {alertsQuery.isFetching && <span className="text-xs text-slate-400">Updating...</span>}
            </div>

            {alertsQuery.isError ? (
              <div className="rounded-md border border-red-500/40 bg-red-900/20 px-4 py-3 text-sm text-red-300">
                Error fetching alerts
              </div>
            ) : alerts.length === 0 ? (
              <div className="rounded-md border border-slate-700 bg-slate-900/40 px-4 py-8 text-center text-slate-300">
                No active alerts to display.
              </div>
            ) : (
              <div className="space-y-4">
                {alerts.map((alert: any) => (
                  <div
                    key={alert.id}
                    className="rounded-md border border-red-500/30 bg-red-500/5 p-4 space-y-3"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <div>
                        <div className="text-xs text-red-300">Rig</div>
                        <div className="text-lg font-semibold text-white">{alert.rig_id}</div>
                      </div>
                      <div>
                        <div className="text-xs text-red-300">Component</div>
                        <div className="text-lg font-semibold text-white">{alert.component}</div>
                      </div>
                      <div>
                        <div className="text-xs text-red-300">Severity</div>
                        <span className="px-3 py-1 rounded-full text-xs bg-red-500/10 border border-red-500/40 text-red-200">
                          {alert.severity}
                        </span>
                      </div>
                    </div>
                    <p className="text-sm text-slate-200 leading-relaxed">{alert.message}</p>
                    <div className="text-xs text-slate-400 flex justify-between">
                      <span>
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

          <section className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">Maintenance Schedule</h2>
              {scheduleQuery.isFetching && <span className="text-xs text-slate-400">Updating...</span>}
            </div>

            {scheduleQuery.isError ? (
              <div className="rounded-md border border-red-500/40 bg-red-900/20 px-4 py-3 text-sm text-red-300">
                Error fetching maintenance schedule
              </div>
            ) : schedules.length === 0 ? (
              <div className="rounded-md border border-slate-700 bg-slate-900/40 px-4 py-8 text-center text-slate-300">
                No schedule has been registered.
              </div>
            ) : (
              <div className="space-y-4">
                {schedules.map((schedule: any) => (
                  <div
                    key={schedule.id}
                    className="rounded-md border border-slate-600 bg-slate-900/40 p-4 space-y-3"
                  >
                    <div className="flex flex-wrap justify-between gap-4">
                      <div>
                        <div className="text-xs text-slate-400">Rig</div>
                        <div className="text-lg font-semibold text-white">{schedule.rig_id}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">Component</div>
                        <div className="text-lg font-semibold text-white">{schedule.component}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">Status</div>
                        <span className="px-3 py-1 rounded-full text-xs bg-cyan-500/10 border border-cyan-500/40 text-cyan-200">
                          {schedule.status}
                        </span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm text-slate-200">
                      <div>
                        <div className="text-xs text-slate-400">Maintenance Type</div>
                        <div>{schedule.maintenance_type}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">Priority</div>
                        <div>{schedule.priority}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">Scheduled Date</div>
                        <div>
                          {schedule.scheduled_date
                            ? new Date(schedule.scheduled_date).toLocaleString('en-US')
                            : '-'}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">Duration</div>
                        <div>{schedule.estimated_duration_hours} hours</div>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="text-xs text-slate-400">
                        Assigned to: {schedule.assigned_to || 'unknown'}
                      </div>
                      <button
                        onClick={() => deleteScheduleMutation.mutate(String(schedule.id))}
                        className="text-xs text-red-300 hover:text-red-200"
                        disabled={deleteScheduleMutation.isLoading}
                      >
                        Delete Schedule
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>

        <div className="space-y-6">
          <section className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-white">Create New Alert</h2>
              <p className="text-xs text-slate-400">
                Create an alert to notify the maintenance team and enable quick planning
              </p>
            </div>
            <form className="space-y-4" onSubmit={handleCreateAlert}>
              <div className="grid grid-cols-1 gap-3">
                <input
                  value={alertForm.rig_id}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, rig_id: e.target.value }))}
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Rig ID"
                />
                <input
                  value={alertForm.component}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, component: e.target.value }))}
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Component"
                />
                <input
                  value={alertForm.alert_type}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, alert_type: e.target.value }))}
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Alert Type"
                />
                <select
                  value={alertForm.severity}
                  onChange={(e) => setAlertForm((prev) => ({ ...prev, severity: e.target.value }))}
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
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
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Description"
                />
              </div>
              <button
                type="submit"
                className="w-full h-10 rounded-md bg-red-500 hover:bg-red-400 text-white font-semibold transition disabled:opacity-60"
                disabled={createAlertMutation.isLoading}
              >
                {createAlertMutation.isLoading ? 'Creating...' : 'Create Alert'}
              </button>
              {createAlertMutation.isError && (
                <div className="rounded-md border border-red-500/40 bg-red-900/20 px-3 py-2 text-xs text-red-200">
                  Error creating alert
                </div>
              )}
              {createAlertMutation.isSuccess && (
                <div className="rounded-md border border-emerald-500/40 bg-emerald-900/20 px-3 py-2 text-xs text-emerald-200">
                  Alert created successfully
                </div>
              )}
            </form>
          </section>

          <section className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4">
            <div>
              <h2 className="text-lg font-semibold text-white">Schedule Maintenance</h2>
              <p className="text-xs text-slate-400">
                Create a new schedule for the preventive maintenance team
              </p>
            </div>
            <form className="space-y-4" onSubmit={handleCreateSchedule}>
              <div className="grid grid-cols-1 gap-3">
                <input
                  value={scheduleForm.rig_id}
                  onChange={(e) => setScheduleForm((prev) => ({ ...prev, rig_id: e.target.value }))}
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Rig ID"
                />
                <input
                  value={scheduleForm.component}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, component: e.target.value }))
                  }
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Component"
                />
                <input
                  value={scheduleForm.maintenance_type}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, maintenance_type: e.target.value }))
                  }
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Maintenance Type"
                />
                <input
                  type="datetime-local"
                  value={scheduleForm.scheduled_date}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, scheduled_date: e.target.value }))
                  }
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
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
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Duration (hours)"
                />
                <select
                  value={scheduleForm.priority}
                  onChange={(e) =>
                    setScheduleForm((prev) => ({ ...prev, priority: e.target.value }))
                  }
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
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
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
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
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Assigned To"
                />
                <textarea
                  rows={3}
                  value={scheduleForm.notes}
                  onChange={(e) => setScheduleForm((prev) => ({ ...prev, notes: e.target.value }))}
                  className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                  placeholder="Description"
                />
              </div>
              <button
                type="submit"
                className="w-full h-10 rounded-md bg-emerald-500 hover:bg-emerald-400 text-slate-900 font-semibold transition disabled:opacity-60"
                disabled={createScheduleMutation.isLoading}
              >
                {createScheduleMutation.isLoading ? 'Creating...' : 'Create Schedule'}
              </button>
              {createScheduleMutation.isError && (
                <div className="rounded-md border border-red-500/40 bg-red-900/20 px-3 py-2 text-xs text-red-200">
                  Error creating schedule
                </div>
              )}
              {createScheduleMutation.isSuccess && (
                <div className="rounded-md border border-emerald-500/40 bg-emerald-900/20 px-3 py-2 text-xs text-emerald-200">
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

