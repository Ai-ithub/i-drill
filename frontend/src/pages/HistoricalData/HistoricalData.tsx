import { FormEvent, useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { sensorDataApi } from '@/services/api'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from 'recharts'
import { Download, Filter, Loader2 } from 'lucide-react'

const AVAILABLE_METRICS = [
  { key: 'wob', label: 'WOB' },
  { key: 'rpm', label: 'RPM' },
  { key: 'torque', label: 'Torque' },
  { key: 'rop', label: 'ROP' },
  { key: 'mud_pressure', label: 'Pressure' },
]

const COLOR_SCALE = ['#06b6d4', '#22c55e', '#a855f7', '#f97316', '#ef4444']

const toInputValue = (date: Date) => {
  const pad = (value: number) => value.toString().padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(
    date.getHours()
  )}:${pad(date.getMinutes())}`
}

const millis = (hours: number) => hours * 60 * 60 * 1000

export default function HistoricalData() {
  const now = useMemo(() => new Date(), [])
  const defaultEnd = toInputValue(now)
  const defaultStart = toInputValue(new Date(now.getTime() - millis(6)))

  const [formState, setFormState] = useState({
    rigId: 'RIG_01',
    start: defaultStart,
    end: defaultEnd,
    limit: 200,
    parameters: '',
  })
  const [validationError, setValidationError] = useState<string | null>(null)
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['wob', 'rpm'])
  const [statusFilter, setStatusFilter] = useState<'all' | 'normal' | 'warning'>('all')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [timeBucket, setTimeBucket] = useState(300)
  const [valueFilters, setValueFilters] = useState({
    depthMin: '',
    depthMax: '',
    wobMin: '',
    wobMax: '',
  })
  const [queryParams, setQueryParams] = useState(() => ({
    rig_id: 'RIG_01',
    start_time: new Date(defaultStart).toISOString(),
    end_time: new Date(defaultEnd).toISOString(),
    limit: 200,
    parameters: '',
  }))

  useEffect(() => {
    setFormState((prev) => ({
      ...prev,
      parameters: selectedMetrics.join(','),
    }))
  }, [selectedMetrics])

  const historicalQuery = useQuery({
    queryKey: ['historical-data', queryParams],
    queryFn: () =>
      sensorDataApi
        .getHistorical({
          rig_id: queryParams.rig_id,
          start_time: queryParams.start_time,
          end_time: queryParams.end_time,
          parameters: queryParams.parameters ? queryParams.parameters : undefined,
          limit: queryParams.limit,
        })
        .then((res) => res.data),
    placeholderData: (previousData) => previousData,
  })

  const aggregatedQuery = useQuery({
    queryKey: ['historical-aggregated', queryParams, timeBucket],
    queryFn: () =>
      sensorDataApi
        .getAggregated(
          queryParams.rig_id || 'RIG_01',
          timeBucket,
          queryParams.start_time,
          queryParams.end_time
        )
        .then((res) => res.data),
    enabled: showAdvanced,
    placeholderData: (previousData) => previousData,
  })

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const startDate = new Date(formState.start)
    const endDate = new Date(formState.end)

    if (startDate >= endDate) {
      setValidationError('Invalid time range. Start time must be before end time.')
      return
    }

    setValidationError(null)
    setQueryParams({
      rig_id: formState.rigId.trim(),
      start_time: startDate.toISOString(),
      end_time: endDate.toISOString(),
      limit: Number(formState.limit) || 200,
      parameters: formState.parameters.trim(),
    })
  }

  const records = historicalQuery.data?.data ?? []

  const displayRecords = useMemo(() => {
    if (statusFilter === 'all') {
      return records.filter((item: any) => {
        const depth = Number(item.depth ?? 0)
        const wob = Number(item.wob ?? 0)
        if (valueFilters.depthMin && depth < Number(valueFilters.depthMin)) {
          return false
        }
        if (valueFilters.depthMax && depth > Number(valueFilters.depthMax)) {
          return false
        }
        if (valueFilters.wobMin && wob < Number(valueFilters.wobMin)) {
          return false
        }
        if (valueFilters.wobMax && wob > Number(valueFilters.wobMax)) {
          return false
        }
        return true
      })
    }
    return records.filter((item: any) => {
      const status = String(item.status ?? '').toLowerCase()
      const statusMatch = statusFilter === 'normal' ? status === 'normal' : status && status !== 'normal'
      if (!statusMatch) {
        return false
      }
      const depth = Number(item.depth ?? 0)
      const wob = Number(item.wob ?? 0)
      if (valueFilters.depthMin && depth < Number(valueFilters.depthMin)) {
        return false
      }
      if (valueFilters.depthMax && depth > Number(valueFilters.depthMax)) {
        return false
      }
      if (valueFilters.wobMin && wob < Number(valueFilters.wobMin)) {
        return false
      }
      if (valueFilters.wobMax && wob > Number(valueFilters.wobMax)) {
        return false
      }
      return true
    })
  }, [records, statusFilter, valueFilters])

  const summary = useMemo(() => {
    if (!displayRecords.length) {
      return null
    }

    const totals = displayRecords.reduce(
      (acc: any, item: any) => {
        acc.depth += Number(item.depth ?? 0)
        acc.wob += Number(item.wob ?? 0)
        acc.rpm += Number(item.rpm ?? 0)
        acc.rop += Number(item.rop ?? 0)
        return acc
      },
      { depth: 0, wob: 0, rpm: 0, rop: 0 }
    )

    return {
      count: displayRecords.length,
      avgDepth: totals.depth / displayRecords.length,
      avgWob: totals.wob / displayRecords.length,
      avgRpm: totals.rpm / displayRecords.length,
      avgRop: totals.rop / displayRecords.length,
    }
  }, [displayRecords])

  const chartData = useMemo(() => {
    if (aggregatedQuery.data?.data?.length) {
      return aggregatedQuery.data.data.map((item: any) => ({
        time: item.timestamp,
        wob: item.avg_wob,
        rpm: item.avg_rpm,
        torque: item.avg_torque,
        rop: item.avg_rop,
        mud_pressure: item.avg_mud_pressure,
      }))
    }
    return displayRecords.slice(0, 200).map((item: any) => ({
      time: item.timestamp,
      wob: item.wob,
      rpm: item.rpm,
      torque: item.torque,
      rop: item.rop,
      mud_pressure: item.mud_pressure,
    }))
  }, [aggregatedQuery.data, displayRecords])

  const handleMetricToggle = (metric: string) => {
    setSelectedMetrics((prev) =>
      prev.includes(metric) ? prev.filter((item) => item !== metric) : [...prev, metric]
    )
  }

  const handleRangeChange = (key: keyof typeof valueFilters, value: string) => {
    setValueFilters((prev) => ({ ...prev, [key]: value }))
  }

  const handleExport = (type: 'csv' | 'json') => {
    if (!displayRecords.length) {
      alert('No data available for export')
      return
    }

    if (type === 'json') {
      const blob = new Blob([JSON.stringify(displayRecords, null, 2)], {
        type: 'application/json;charset=utf-8;',
      })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `historical_${queryParams.rig_id}_${Date.now()}.json`
      link.click()
      URL.revokeObjectURL(url)
      return
    }

    const headers = Object.keys(displayRecords[0])
    const csv = [headers.join(',')]
    for (const record of displayRecords) {
      const row = headers
        .map((key) => {
          const value = record[key]
          if (value === null || value === undefined) return ''
          if (typeof value === 'string' && value.includes(',')) {
            return `"${value.replace(/"/g, '""')}"`
          }
          return value
        })
        .join(',')
      csv.push(row)
    }
    const blob = new Blob([csv.join('\n')], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `historical_${queryParams.rig_id}_${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  const handleAggregatedExport = () => {
    const aggregates = aggregatedQuery.data?.data
    if (!aggregates?.length) {
      alert('No aggregated data available for export')
      return
    }
    const headers = Object.keys(aggregates[0])
    const csv = [headers.join(',')]
    for (const record of aggregates) {
      csv.push(
        headers
          .map((key) => {
            const value = record[key]
            if (value === null || value === undefined) return ''
            if (typeof value === 'string' && value.includes(',')) {
              return `"${value.replace(/"/g, '""')}"`
            }
            return value
          })
          .join(','),
      )
    }
    const blob = new Blob([csv.join('\n')], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `historical_aggregated_${queryParams.rig_id}_${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">
      <div>
        <h1 className="text-3xl font-bold mb-2">Historical Data</h1>
        <p className="text-slate-500 dark:text-slate-300">
          Search and analyze historical sensor data for trend analysis and reporting
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 space-y-4 shadow-sm"
      >
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Rig ID</label>
            <input
              value={formState.rigId}
              onChange={(e) => setFormState((prev) => ({ ...prev, rigId: e.target.value }))}
              className="w-full rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
              placeholder="RIG_01"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">From Time</label>
            <input
              type="datetime-local"
              value={formState.start}
              onChange={(e) => setFormState((prev) => ({ ...prev, start: e.target.value }))}
              className="w-full rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">To Time</label>
            <input
              type="datetime-local"
              value={formState.end}
              onChange={(e) => setFormState((prev) => ({ ...prev, end: e.target.value }))}
              className="w-full rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Max Records</label>
            <input
              type="number"
              min={1}
              max={10000}
              value={formState.limit}
              onChange={(e) =>
                setFormState((prev) => ({ ...prev, limit: Number(e.target.value) }))
              }
              className="w-full rounded-md bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-3 py-2 text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="space-y-2 lg:col-span-2">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Parameters</label>
            <div className="flex flex-wrap gap-2">
              {AVAILABLE_METRICS.map((metric) => (
                <button
                  key={metric.key}
                  type="button"
                  onClick={() => handleMetricToggle(metric.key)}
                  className={`px-4 py-2 text-sm rounded-lg border transition font-medium ${
                    selectedMetrics.includes(metric.key)
                      ? 'bg-cyan-500 text-white border-cyan-500'
                      : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:border-cyan-400/60 hover:bg-slate-50 dark:hover:bg-slate-700'
                  }`}
                >
                  {metric.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-end">
            <button
              type="submit"
              className="w-full h-10 rounded-md bg-cyan-500 hover:bg-cyan-600 text-white font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              disabled={historicalQuery.isFetching}
            >
              {historicalQuery.isFetching ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Fetching...
                </>
              ) : (
                <>
                  <Filter className="w-4 h-4" />
                  Fetch Data
                </>
              )}
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
          <button
            type="button"
            onClick={() => setShowAdvanced((prev) => !prev)}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-2 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition"
          >
            <Filter className="w-4 h-4" /> Advanced Settings
          </button>
          <div className="flex items-center gap-2">
            <label className="text-slate-700 dark:text-slate-300">Record Status:</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)}
              className="rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-1.5 text-sm text-slate-900 dark:text-white focus:border-cyan-500 focus:outline-none"
            >
              <option value="all">All</option>
              <option value="normal">Normal</option>
              <option value="warning">Warning</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => handleExport('csv')}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition"
            >
              <Download className="w-4 h-4" /> Export CSV
            </button>
            <button
              type="button"
              onClick={() => handleExport('json')}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition"
            >
              <Download className="w-4 h-4" /> Export JSON
            </button>
            {showAdvanced && (
              <button
                type="button"
                onClick={handleAggregatedExport}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition"
              >
                <Download className="w-4 h-4" /> Export Aggregated
              </button>
            )}
          </div>
        </div>

        {showAdvanced && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-500 dark:text-slate-300">
            <div className="space-y-1">
              <label className="block">Aggregation Interval (seconds)</label>
              <input
                type="number"
                min={60}
                max={3600}
                value={timeBucket}
                onChange={(e) => setTimeBucket(Number(e.target.value))}
                className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1"
              />
            </div>
            <div className="space-y-1">
              <label className="block">Custom Parameters</label>
              <input
                value={formState.parameters}
                onChange={(e) => setFormState((prev) => ({ ...prev, parameters: e.target.value }))}
                placeholder="depth,wob,rpm"
                className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1"
              />
              <p className="text-[10px] text-slate-400">Use the main query if you need specific columns.</p>
            </div>
            <div className="space-y-1">
              <label className="block">Range Filter</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="number"
                  placeholder="Min Depth"
                  value={valueFilters.depthMin}
                  onChange={(e) => handleRangeChange('depthMin', e.target.value)}
                  className="rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1"
                />
                <input
                  type="number"
                  placeholder="Max Depth"
                  value={valueFilters.depthMax}
                  onChange={(e) => handleRangeChange('depthMax', e.target.value)}
                  className="rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1"
                />
                <input
                  type="number"
                  placeholder="Min WOB"
                  value={valueFilters.wobMin}
                  onChange={(e) => handleRangeChange('wobMin', e.target.value)}
                  className="rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1"
                />
                <input
                  type="number"
                  placeholder="Max WOB"
                  value={valueFilters.wobMax}
                  onChange={(e) => handleRangeChange('wobMax', e.target.value)}
                  className="rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1"
                />
              </div>
              <p className="text-[10px] text-slate-400">Filters are applied before displaying the table and export.</p>
            </div>
          </div>
        )}

        {validationError && (
          <div className="rounded-md border border-red-500/40 bg-red-900/20 px-4 py-2 text-sm text-red-300">
            {validationError}
          </div>
        )}
      </form>

      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 space-y-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Results</h2>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              {historicalQuery.isFetching ? 'Loading...' : `Record Count: ${displayRecords.length}`}
            </p>
          </div>
          {historicalQuery.isFetching && <Loader2 className="w-5 h-5 animate-spin text-cyan-500" />}
        </div>

        {summary && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4 mb-6">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-4">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Record Count</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{summary.count}</div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-4">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Average Depth</div>
              <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">
                {summary.avgDepth.toFixed(1)} ft
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-4">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Average WOB</div>
              <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">
                {summary.avgWob.toFixed(0)} lbs
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-4">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Average RPM</div>
              <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">
                {summary.avgRpm.toFixed(0)}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-4">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Average ROP</div>
              <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">
                {summary.avgRop.toFixed(2)} ft/hr
              </div>
            </div>
          </div>
        )}

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Comparison Chart</h3>
            {aggregatedQuery.isFetching && (
              <span className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
                <Loader2 className="w-3 h-3 animate-spin" />
                Calculating aggregated statistics...
              </span>
            )}
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
                <XAxis 
                  dataKey="time" 
                  tick={{ fontSize: 12, fill: '#64748b' }}
                  stroke="#64748b"
                  tickFormatter={(value) => new Date(value).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                />
                <YAxis 
                  tick={{ fontSize: 12, fill: '#64748b' }}
                  stroke="#64748b"
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number, name) => [value?.toFixed?.(2) ?? value, name]} 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                />
                <Legend />
                {selectedMetrics.map((metric, index) => (
                  <Line
                    key={metric}
                    type="monotone"
                    dataKey={metric}
                    stroke={COLOR_SCALE[index % COLOR_SCALE.length]}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {historicalQuery.isError ? (
          <div className="rounded-xl border border-red-500/40 bg-red-50 dark:bg-red-900/20 px-4 py-4 text-red-700 dark:text-red-300">
            Error fetching data: {(historicalQuery.error as Error)?.message ?? 'unknown'}
          </div>
        ) : displayRecords.length === 0 && !historicalQuery.isFetching ? (
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-4 py-10 text-center text-slate-500 dark:text-slate-400">
            No data found for the selected time range.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-700 text-left">
              <thead>
                <tr className="text-slate-700 dark:text-slate-300 text-sm font-semibold uppercase bg-slate-50 dark:bg-slate-800">
                  <th className="px-4 py-3">Time</th>
                  <th className="px-4 py-3">Depth</th>
                  <th className="px-4 py-3">WOB</th>
                  <th className="px-4 py-3">RPM</th>
                  <th className="px-4 py-3">Torque</th>
                  <th className="px-4 py-3">ROP</th>
                  <th className="px-4 py-3">Pressure</th>
                  <th className="px-4 py-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700 text-sm text-slate-900 dark:text-slate-200 bg-white dark:bg-slate-900">
                {displayRecords.map((row: any) => (
                  <tr key={`${row.id}-${row.timestamp}`} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                    <td className="px-4 py-2 font-mono">
                      {row.timestamp ? new Date(row.timestamp).toLocaleString('en-US') : '-'}
                    </td>
                    <td className="px-4 py-2 font-mono">{row.depth?.toFixed?.(2) ?? row.depth ?? '-'}</td>
                    <td className="px-4 py-2 font-mono">{row.wob?.toFixed?.(1) ?? row.wob ?? '-'}</td>
                    <td className="px-4 py-2 font-mono">{row.rpm?.toFixed?.(0) ?? row.rpm ?? '-'}</td>
                    <td className="px-4 py-2 font-mono">{row.torque?.toFixed?.(0) ?? row.torque ?? '-'}</td>
                    <td className="px-4 py-2 font-mono">{row.rop?.toFixed?.(2) ?? row.rop ?? '-'}</td>
                    <td className="px-4 py-2 font-mono">
                      {row.mud_pressure?.toFixed?.(0) ?? row.mud_pressure ?? '-'}
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-medium ${
                          row.status === 'normal'
                            ? 'bg-green-100 dark:bg-green-500/10 text-green-700 dark:text-green-400 border border-green-300 dark:border-green-500/40'
                            : 'bg-amber-100 dark:bg-amber-500/10 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-500/40'
                        }`}
                      >
                        {row.status ?? 'unknown'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

