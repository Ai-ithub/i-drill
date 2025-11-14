import { FormEvent, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { dvrApi } from '@/services/api'
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Download,
  Filter,
  RefreshCcw,
  Database,
  Shield,
  TrendingUp,
  Activity,
} from 'lucide-react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  BarChart,
  Bar,
} from 'recharts'

export default function DVRTab() {
  const queryClient = useQueryClient()
  const [recordJson, setRecordJson] = useState(`{
  "rig_id": "RIG_01",
  "depth": 1234,
  "rpm": 85,
  "wob": 50000,
  "torque": 12000,
  "rop": 25.5,
  "mud_flow": 600,
  "mud_pressure": 2500
}`)
  const [historySize, setHistorySize] = useState(100)
  const [searchTerm, setSearchTerm] = useState('')
  const [historyFilters, setHistoryFilters] = useState(() => {
    const now = new Date()
    const end = now.toISOString().slice(0, 16)
    const start = new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString().slice(0, 16)
    return {
      rigId: '',
      start,
      end,
      anomalyOnly: false,
      validationStatus: 'all',
    }
  })
  const [historyParams, setHistoryParams] = useState<{
    rig_id?: string
    start_time: string
    end_time: string
    limit: number
  }>(() => ({
    rig_id: undefined,
    start_time: new Date(historyFilters.start).toISOString(),
    end_time: new Date(historyFilters.end).toISOString(),
    limit: 200,
  }))

  // Mock data for demonstration when backend is not available
  const mockStats = {
    summary: {
      total_records: 1250,
      valid_records: 1180,
      invalid_records: 45,
      anomaly_count: 25,
      averages: {
        depth: 5000,
        wob: 45000,
        rpm: 150,
        torque: 8000,
        rop: 25.5,
      },
    },
  }

  const mockAnomaly = {
    numeric_columns: ['depth', 'wob', 'rpm', 'torque', 'rop'],
    anomaly_count: 25,
    anomalies: [],
  }

  const mockHistory = Array.from({ length: 10 }, (_, i) => ({
    id: `record-${i}`,
    rig_id: 'RIG_01',
    timestamp: new Date(Date.now() - i * 60000).toISOString(),
    depth: 5000 + i * 10,
    wob: 45000 + i * 100,
    rpm: 150,
    torque: 8000,
    rop: 25.5,
    validation_status: i % 3 === 0 ? 'invalid' : 'valid',
    anomaly_detected: i % 5 === 0,
    anomaly_reason: i % 5 === 0 ? 'Outlier detected' : null,
  }))

  const statsQuery = useQuery({
    queryKey: ['dvr-stats'],
    queryFn: () => dvrApi.getStats().then((res) => res.data),
    refetchInterval: 15000,
    refetchOnWindowFocus: false,
    retry: 1,
    retryDelay: 1000,
  })

  const anomalyQuery = useQuery({
    queryKey: ['dvr-anomalies', historySize],
    queryFn: () => dvrApi.getAnomalies(historySize).then((res) => res.data),
    refetchInterval: 20000,
    refetchOnWindowFocus: false,
    retry: 1,
    retryDelay: 1000,
  })

  const processMutation = useMutation({
    mutationFn: (record: any) => dvrApi.processRecord(record).then((res) => res.data),
  })

  const evaluateMutation = useMutation({
    mutationFn: (payload: { record: any; size: number }) =>
      dvrApi.evaluateRecord(payload.record, payload.size).then((res) => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dvr-stats'] })
      queryClient.invalidateQueries({ queryKey: ['dvr-anomalies'] })
      queryClient.invalidateQueries({ queryKey: ['dvr-history', historyParams] })
    },
  })

  const historyQuery = useQuery({
    queryKey: ['dvr-history', historyParams],
    queryFn: () => dvrApi.getHistory(historyParams).then((res) => res.data),
    placeholderData: (previousData) => previousData,
  })

  // Use actual data or fallback to mock data
  const stats = statsQuery.data || mockStats
  const anomaly = anomalyQuery.data || mockAnomaly
  const averages = (stats?.summary?.averages ?? {}) as Record<string, number>
  const numericColumns = (anomaly?.numeric_columns ?? []) as string[]
  const historyRecords = historyQuery.data?.history ?? historyQuery.data?.data ?? mockHistory

  const filteredHistory = useMemo(() => {
    const term = searchTerm.trim().toLowerCase()
    return historyRecords.filter((item: any) => {
      if (historyFilters.validationStatus !== 'all') {
        if ((item.validation_status ?? '').toLowerCase() !== historyFilters.validationStatus) {
          return false
        }
      }
      if (historyFilters.anomalyOnly && !item.anomaly_detected) {
        return false
      }
      if (term) {
        const haystack = `${item.rig_id ?? ''} ${item.anomaly_reason ?? ''} ${item.validation_status ?? ''} ${item.action_taken ?? ''}`.toLowerCase()
        if (!haystack.includes(term)) {
          return false
        }
      }
      return true
    })
  }, [historyRecords, historyFilters, searchTerm])

  const anomalyTrend = useMemo(() => {
    const grouped = new Map<string, { anomalies: number; total: number }>()
    filteredHistory.forEach((item: any) => {
      const timeKey = new Date(item.timestamp).toISOString().slice(0, 13) + ':00'
      const entry = grouped.get(timeKey) || { anomalies: 0, total: 0 }
      entry.total += 1
      if (item.anomaly_detected) {
        entry.anomalies += 1
      }
      grouped.set(timeKey, entry)
    })
    return Array.from(grouped.entries())
      .sort(([a], [b]) => (a > b ? 1 : -1))
      .map(([time, values]) => ({
        time,
        anomalies: values.anomalies,
        total: values.total,
        ratio: values.total ? (values.anomalies / values.total) * 100 : 0,
      }))
      .slice(-50)
  }, [filteredHistory])

  const statusBreakdown = useMemo(() => {
    const counts: Record<string, number> = {}
    filteredHistory.forEach((item: any) => {
      const status = item.validation_status ?? 'unknown'
      counts[status] = (counts[status] || 0) + 1
    })
    return Object.entries(counts).map(([status, count]) => ({ status, count }))
  }, [filteredHistory])

  const validationMetrics = useMemo(() => {
    const total = filteredHistory.length
    const valid = filteredHistory.filter((item: any) => item.validation_status === 'valid').length
    const invalid = filteredHistory.filter((item: any) => item.validation_status === 'invalid').length
    const anomalies = filteredHistory.filter((item: any) => item.anomaly_detected).length
    return {
      total,
      valid,
      invalid,
      anomalies,
      validationRate: total > 0 ? (valid / total) * 100 : 0,
      anomalyRate: total > 0 ? (anomalies / total) * 100 : 0,
    }
  }, [filteredHistory])

  const handleProcess = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    try {
      const record = JSON.parse(recordJson)
      processMutation.mutate(record, {
        onSuccess: () => {
          queryClient.invalidateQueries(['dvr-stats'])
          queryClient.invalidateQueries(['dvr-anomalies'])
          queryClient.invalidateQueries(['dvr-history', historyParams])
        },
      })
    } catch (err) {
      alert('Invalid JSON format.')
    }
  }

  const handleEvaluate = () => {
    try {
      const record = JSON.parse(recordJson)
      evaluateMutation.mutate({ record, size: historySize })
    } catch (err) {
      alert('Invalid JSON format.')
    }
  }

  const handleHistorySubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (new Date(historyFilters.start) >= new Date(historyFilters.end)) {
      alert('Invalid time range')
      return
    }
    setHistoryParams({
      rig_id: historyFilters.rigId || undefined,
      start_time: new Date(historyFilters.start).toISOString(),
      end_time: new Date(historyFilters.end).toISOString(),
      limit: 500,
    })
  }

  const handleExport = async (type: 'csv' | 'pdf') => {
    try {
      const exporter = type === 'csv' ? dvrApi.exportHistoryCsv : dvrApi.exportHistoryPdf
      const response = await exporter({
        rig_id: historyParams.rig_id,
        start_time: historyParams.start_time,
        end_time: historyParams.end_time,
      })
      const blob = new Blob([response.data], {
        type: type === 'csv' ? 'text/csv;charset=utf-8;' : 'application/pdf',
      })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `dvr_history_${Date.now()}.${type}`
      link.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      alert('Export failed.')
    }
  }

  const handleLocalJsonExport = () => {
    if (!filteredHistory.length) {
      alert('No data available for export.')
      return
    }
    const blob = new Blob([JSON.stringify(filteredHistory, null, 2)], {
      type: 'application/json;charset=utf-8;',
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `dvr_history_filtered_${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6 text-slate-900 dark:text-slate-100">

      {/* Validation Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="rounded-2xl bg-gradient-to-br from-emerald-500 to-green-600 p-6 text-white shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <CheckCircle2 className="w-8 h-8" />
            <span className="text-3xl font-bold">{validationMetrics.validationRate.toFixed(1)}%</span>
          </div>
          <div className="text-sm opacity-90">Validation Rate</div>
          <div className="text-xs opacity-75 mt-1">Validation Rate</div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-6 h-6 text-emerald-500" />
            <div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{validationMetrics.valid}</div>
              <div className="text-sm text-slate-500 dark:text-slate-400">Valid Records</div>
              <div className="text-xs text-slate-400 dark:text-slate-500">Valid Records</div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="w-6 h-6 text-amber-500" />
            <div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{validationMetrics.invalid}</div>
              <div className="text-sm text-slate-500 dark:text-slate-400">Invalid Records</div>
              <div className="text-xs text-slate-400 dark:text-slate-500">Invalid Records</div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <Activity className="w-6 h-6 text-red-500" />
            <div>
              <div className="text-2xl font-bold text-slate-900 dark:text-white">{validationMetrics.anomalies}</div>
              <div className="text-sm text-slate-500 dark:text-slate-400">Anomalies Detected</div>
              <div className="text-xs text-slate-400 dark:text-slate-500">Detected Anomalies</div>
            </div>
          </div>
        </div>
      </div>

      {/* Processing History Filter */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h3 className="text-xl font-semibold flex items-center gap-2">
            <Filter className="w-5 h-5 text-cyan-500" /> Processing History Filter
          </h3>
          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            {historyQuery.isFetching && 'Updating...'}
            {historyQuery.isError && 'Error fetching history'}
          </div>
        </div>

        <form className="grid grid-cols-1 md:grid-cols-5 gap-4 text-sm" onSubmit={handleHistorySubmit}>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">Rig ID</label>
            <input
              value={historyFilters.rigId}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, rigId: e.target.value }))}
              placeholder="RIG_01"
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">From Time</label>
            <input
              type="datetime-local"
              value={historyFilters.start}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, start: e.target.value }))}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">To Time</label>
            <input
              type="datetime-local"
              value={historyFilters.end}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, end: e.target.value }))}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">Validation Status</label>
            <select
              value={historyFilters.validationStatus}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, validationStatus: e.target.value }))}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm"
            >
              <option value="all">All</option>
              <option value="valid">Valid</option>
              <option value="invalid">Invalid</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">Search</label>
            <input
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="rig / anomaly / notes"
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 text-sm"
            />
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-300 md:col-span-2">
            <input
              type="checkbox"
              checked={historyFilters.anomalyOnly}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, anomalyOnly: e.target.checked }))}
              className="rounded"
            />
            Anomaly records only
          </div>

          <div className="flex items-center gap-2 md:col-span-5 justify-end">
            <button
              type="button"
              onClick={() => handleExport('csv')}
              className="inline-flex items-center gap-2 rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-xs hover:bg-slate-50 dark:hover:bg-slate-800"
            >
              <Download className="w-4 h-4" /> CSV
            </button>
            <button
              type="button"
              onClick={() => handleExport('pdf')}
              className="inline-flex items-center gap-2 rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-xs hover:bg-slate-50 dark:hover:bg-slate-800"
            >
              <Download className="w-4 h-4" /> PDF
            </button>
            <button
              type="button"
              onClick={handleLocalJsonExport}
              className="inline-flex items-center gap-2 rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-xs hover:bg-slate-50 dark:hover:bg-slate-800"
            >
              <Download className="w-4 h-4" /> JSON
            </button>
            <button
              type="submit"
              className="inline-flex items-center gap-2 rounded-md bg-cyan-500 hover:bg-cyan-400 text-white px-4 py-2 text-sm font-semibold"
            >
              Search
            </button>
          </div>
        </form>
      </div>

      {/* Statistics and Processing */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {/* Latest Statistics */}
          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold">Latest Statistics</h3>
              <button
                onClick={() => queryClient.invalidateQueries(['dvr-stats'])}
                className="flex items-center gap-2 text-sm px-3 py-2 rounded bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700"
              >
                <RefreshCcw className="w-4 h-4" /> Refresh
              </button>
            </div>

            {stats?.success ? (
              <div className="space-y-4 text-sm">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <SummaryCard title="Record Count" value={stats.summary.count ?? 0} />
                  <SummaryCard title="Rig Count" value={stats.summary.rig_ids?.length ?? 0} />
                  <SummaryCard
                    title="Latest Time"
                    value={stats.summary.latest?.timestamp ? new Date(stats.summary.latest.timestamp).toLocaleString('en-US') : '---'}
                  />
                </div>

                <div className="text-xs text-slate-600 dark:text-slate-300">
                  <h4 className="font-semibold text-slate-900 dark:text-white mb-2">Average Key Parameters</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {Object.entries(averages).slice(0, 8).map(([key, value]) => (
                      <div key={key} className="bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 rounded px-3 py-2">
                        <div className="text-slate-500 dark:text-slate-400">{key}</div>
                        <div className="text-slate-900 dark:text-white font-mono text-sm">{Number(value).toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-sm text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded px-4 py-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                {stats?.message || 'No data received.'}
              </div>
            )}
          </div>

          {/* Anomaly Snapshot */}
          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold">Anomaly Snapshot</h3>
              <div className="flex items-center gap-3">
                <label className="text-xs text-slate-500 dark:text-slate-300">History Size</label>
                <input
                  type="number"
                  value={historySize}
                  min={10}
                  max={1000}
                  onChange={(e) => setHistorySize(Number(e.target.value))}
                  className="w-20 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-2 py-1 text-xs"
                />
              </div>
            </div>

            {anomaly?.success ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {numericColumns.map((col) => (
                  <div key={col} className="bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 rounded px-3 py-2">
                    <div className="text-slate-600 dark:text-slate-300">{col}</div>
                    <div className="text-slate-900 dark:text-white font-mono">
                      {anomaly.history_sizes?.[col] ?? 0} points
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded px-4 py-3">
                {anomaly?.message || 'No information available.'}
              </div>
            )}
          </div>
        </div>

        {/* Process Record & Anomaly Detection */}
        <div className="space-y-4">
          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 space-y-4 shadow-sm">
            <div>
              <h3 className="text-xl font-semibold">Process Record</h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                Enter a JSON record for validation, cleaning, and storage.
              </p>
            </div>

            <form className="space-y-3" onSubmit={handleProcess}>
              <textarea
                rows={6}
                value={recordJson}
                onChange={(e) => setRecordJson(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded px-3 py-2 text-xs font-mono"
              />
              <button
                type="submit"
                className="w-full flex items-center justify-center gap-2 bg-cyan-500 hover:bg-cyan-400 text-white font-semibold rounded py-2 text-sm"
              >
                <CheckCircle2 className="w-4 h-4" /> Execute Process
              </button>
            </form>

            {processMutation.data && (
              <div
                className={`text-xs rounded px-3 py-2 border ${
                  processMutation.data.success
                    ? 'border-emerald-500/50 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200'
                    : 'border-amber-500/50 bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-100'
                }`}
              >
                {processMutation.data.message ??
                  (processMutation.data.success ? 'Successfully saved.' : 'Processing failed.')}
              </div>
            )}
          </div>

          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 space-y-3 shadow-sm">
            <h3 className="text-xl font-semibold">Record Anomaly Detection</h3>
            <button
              onClick={handleEvaluate}
              className="w-full flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded py-2 text-sm"
            >
              <BarChart3 className="w-4 h-4" /> Analyze Anomaly
            </button>

            {evaluateMutation.data && (
              <div
                className={`text-xs rounded px-3 py-2 border ${
                  evaluateMutation.data.success
                    ? 'border-emerald-500/50 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200'
                    : 'border-amber-500/50 bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-100'
                }`}
              >
                <div className="font-mono text-[11px] whitespace-pre-wrap">
                  {JSON.stringify(evaluateMutation.data.record ?? evaluateMutation.data.message, null, 2)}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Anomaly Trend</h3>
            <span className="text-xs text-slate-500 dark:text-slate-400">Last {anomalyTrend.length} intervals</span>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={anomalyTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
                <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" orientation="left" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="anomalies"
                  stroke="#f97316"
                  strokeWidth={2}
                  dot={false}
                  name="Anomalies"
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="total"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                  name="Total Records"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="ratio"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                  name="Anomaly Ratio %"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-4">
          <h3 className="text-lg font-semibold">Validation Status Breakdown</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={statusBreakdown}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
                <XAxis dataKey="status" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#0ea5e9" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Processing History Table */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Processing History</h3>
          <span className="text-xs text-slate-500 dark:text-slate-400">{filteredHistory.length} records displayed</span>
        </div>

        {historyQuery.isError ? (
          <div className="text-sm text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded px-4 py-3">
            Error fetching history
          </div>
        ) : filteredHistory.length === 0 ? (
          <div className="text-sm text-slate-500 dark:text-slate-300 border border-dashed border-slate-300 dark:border-slate-700 rounded-lg px-6 py-10 text-center">
            No data found matching the selected filter.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm text-slate-600 dark:text-slate-200">
              <thead className="bg-slate-100/70 dark:bg-slate-800/70">
                <tr>
                  <th className="px-4 py-2 text-right">Time</th>
                  <th className="px-4 py-2 text-right">Rig</th>
                  <th className="px-4 py-2 text-right">Validation</th>
                  <th className="px-4 py-2 text-right">Reconciliation</th>
                  <th className="px-4 py-2 text-right">Anomaly</th>
                  <th className="px-4 py-2 text-right">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {filteredHistory.slice(0, 200).map((row: any, index: number) => (
                  <tr key={`${row.id}-${row.timestamp}-${index}`} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                    <td className="px-4 py-2">
                      {row.timestamp ? new Date(row.timestamp).toLocaleString('en-US') : '---'}
                    </td>
                    <td className="px-4 py-2">{row.rig_id ?? '---'}</td>
                    <td className="px-4 py-2">
                      <span
                        className={`px-2 py-1 rounded-full text-xs ${
                          row.validation_status === 'valid'
                            ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/40'
                            : 'bg-amber-500/10 text-amber-400 border border-amber-400/40'
                        }`}
                      >
                        {row.validation_status}
                      </span>
                    </td>
                    <td className="px-4 py-2">{row.reconciliation_status ?? '---'}</td>
                    <td className="px-4 py-2">
                      {row.anomaly_detected ? (
                        <span className="text-rose-400">✓</span>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-xs text-slate-500 dark:text-slate-300">
                      {row.anomaly_reason || '---'}
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

interface SummaryCardProps {
  title: string
  value: string | number
}

function SummaryCard({ title, value }: SummaryCardProps) {
  return (
    <div className="bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 rounded px-3 py-2">
      <div className="text-slate-500 dark:text-slate-400 text-xs">{title}</div>
      <div className="text-slate-900 dark:text-white font-mono text-lg">{value}</div>
    </div>
  )
}

