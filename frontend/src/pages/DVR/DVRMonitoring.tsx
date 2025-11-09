import { FormEvent, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { dvrApi } from '@/services/api'
import { AlertTriangle, BarChart3, CheckCircle2, Download, Filter, RefreshCcw } from 'lucide-react'
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

export default function DVRMonitoring() {
  const queryClient = useQueryClient()
  const [recordJson, setRecordJson] = useState(`{
  "rig_id": "RIG_01",
  "depth": 1234,
  "rpm": 85
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

  const statsQuery = useQuery(['dvr-stats'], () => dvrApi.getStats().then((res) => res.data), {
    refetchInterval: 15000,
    refetchOnWindowFocus: false,
  })

  const anomalyQuery = useQuery(
    ['dvr-anomalies', historySize],
    () => dvrApi.getAnomalies(historySize).then((res) => res.data),
    {
      refetchInterval: 20000,
      refetchOnWindowFocus: false,
    }
  )

  const processMutation = useMutation((record: any) => dvrApi.processRecord(record).then((res) => res.data))

  const evaluateMutation = useMutation(
    (payload: { record: any; size: number }) =>
      dvrApi.evaluateRecord(payload.record, payload.size).then((res) => res.data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['dvr-stats'])
        queryClient.invalidateQueries(['dvr-anomalies'])
        queryClient.invalidateQueries(['dvr-history', historyParams])
      },
    }
  )

  const historyQuery = useQuery(
    ['dvr-history', historyParams],
    () => dvrApi.getHistory(historyParams).then((res) => res.data),
    {
      keepPreviousData: true,
    }
  )

  const stats = statsQuery.data
  const anomaly = anomalyQuery.data
  const averages = (stats?.summary?.averages ?? {}) as Record<string, number>
  const numericColumns = (anomaly?.numeric_columns ?? []) as string[]
  const historyRecords = historyQuery.data?.history ?? historyQuery.data?.data ?? []

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

  const handleProcess = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    try {
      const record = JSON.parse(recordJson)
      processMutation.mutate(record)
    } catch (err) {
      alert('فرمت JSON معتبر نیست.')
    }
  }

  const handleEvaluate = () => {
    try {
      const record = JSON.parse(recordJson)
      evaluateMutation.mutate({ record, size: historySize })
    } catch (err) {
      alert('فرمت JSON معتبر نیست.')
    }
  }

  const handleHistorySubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (new Date(historyFilters.start) >= new Date(historyFilters.end)) {
      alert('بازه زمانی معتبر نیست')
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
      alert('خروجی با خطا مواجه شد.')
    }
  }

  const handleLocalJsonExport = () => {
    if (!filteredHistory.length) {
      alert('داده‌ای برای خروجی وجود ندارد.')
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
    <div className="space-y-6 p-6 text-slate-900 dark:text-white">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">پایش DVR</h1>
        <p className="text-slate-500 dark:text-slate-300">
          مشاهده وضعیت پردازش داده‌ها، آمار رکوردها، نمودار آنومالی و تولید گزارش خروجی.
        </p>
      </header>

      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Filter className="w-5 h-5 text-cyan-500" /> فیلتر تاریخچه پردازش
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            {historyQuery.isFetching && 'در حال بروزرسانی...'}
            {historyQuery.isError && 'خطا در دریافت تاریخچه'}
          </div>
        </div>

        <form className="grid grid-cols-1 md:grid-cols-5 gap-4 text-sm" onSubmit={handleHistorySubmit}>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">شناسه دکل</label>
            <input
              value={historyFilters.rigId}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, rigId: e.target.value }))}
              placeholder="RIG_01"
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">از زمان</label>
            <input
              type="datetime-local"
              value={historyFilters.start}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, start: e.target.value }))}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">تا زمان</label>
            <input
              type="datetime-local"
              value={historyFilters.end}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, end: e.target.value }))}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">وضعیت اعتبارسنجی</label>
            <select
              value={historyFilters.validationStatus}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, validationStatus: e.target.value }))}
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2"
            >
              <option value="all">همه</option>
              <option value="valid">معتبر</option>
              <option value="invalid">نامعتبر</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="block text-xs text-slate-500 dark:text-slate-300">جستجو در توضیحات</label>
            <input
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="rig / anomaly / notes"
              className="w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2"
            />
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-300 md:col-span-2">
            <input
              type="checkbox"
              checked={historyFilters.anomalyOnly}
              onChange={(e) => setHistoryFilters((prev) => ({ ...prev, anomalyOnly: e.target.checked }))}
            />
            فقط رکوردهای آنومالی
          </div>

          <div className="flex items-center gap-2 md:col-span-5 justify-end">
            <button
              type="button"
              onClick={() => handleExport('csv')}
              className="inline-flex items-center gap-2 rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-xs"
            >
              <Download className="w-4 h-4" /> CSV
            </button>
            <button
              type="button"
              onClick={() => handleExport('pdf')}
              className="inline-flex items-center gap-2 rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-xs"
            >
              <Download className="w-4 h-4" /> PDF
            </button>
            <button
              type="button"
              onClick={handleLocalJsonExport}
              className="inline-flex items-center gap-2 rounded-md border border-slate-300 dark:border-slate-600 px-3 py-2 text-xs"
            >
              <Download className="w-4 h-4" /> JSON
            </button>
            <button
              type="submit"
              className="inline-flex items-center gap-2 rounded-md bg-cyan-500 text-slate-900 px-4 py-2 text-sm font-semibold"
            >
              جستجو
            </button>
          </div>
        </form>
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">آخرین آمار</h2>
              <button
                onClick={() => queryClient.invalidateQueries(['dvr-stats'])}
                className="flex items-center gap-2 text-sm px-3 py-2 rounded bg-slate-700 hover:bg-slate-600"
              >
                <RefreshCcw className="w-4 h-4" /> بروزرسانی
              </button>
            </div>

            {stats?.success ? (
              <div className="space-y-4 text-sm">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <SummaryCard title="تعداد رکورد" value={stats.summary.count ?? 0} />
                  <SummaryCard
                    title="تعداد دکل"
                    value={stats.summary.rig_ids?.length ?? 0}
                  />
                  <SummaryCard
                    title="آخرین زمان"
                    value={stats.summary.latest?.timestamp ?? '---'}
                  />
                </div>

                <div className="text-xs text-slate-300">
                  <h3 className="font-semibold text-slate-100 mb-2">میانگین پارامترهای کلیدی</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {Object.entries(averages).slice(0, 8).map(([key, value]) => (
                      <div key={key} className="bg-slate-900/40 border border-slate-700 rounded px-3 py-2">
                        <div className="text-slate-400">{key}</div>
                        <div className="text-white font-mono text-sm">{Number(value).toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-sm text-amber-200 bg-amber-900/20 border border-amber-600 rounded px-4 py-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                {stats?.message || 'هیچ داده‌ای دریافت نشد.'}
              </div>
            )}
          </div>

          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 shadow-sm space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">عکس فوری آنومالی</h2>
              <div className="flex items-center gap-3">
                <label className="text-xs text-slate-300">اندازه تاریخچه</label>
                <input
                  type="number"
                  value={historySize}
                  min={10}
                  max={1000}
                  onChange={(e) => setHistorySize(Number(e.target.value))}
                  className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs"
                />
              </div>
            </div>

            {anomaly?.success ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {numericColumns.map((col) => (
                  <div key={col} className="bg-slate-900/40 border border-slate-700 rounded px-3 py-2">
                    <div className="text-slate-300">{col}</div>
                    <div className="text-slate-100 font-mono">
                      {anomaly.history_sizes?.[col] ?? 0} points
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-amber-200 bg-amber-900/20 border border-amber-600 rounded px-4 py-3">
                {anomaly?.message || 'اطلاعاتی در دسترس نیست.'}
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 space-y-4 shadow-sm">
            <div>
              <h2 className="text-xl font-semibold">پردازش رکورد</h2>
              <p className="text-xs text-slate-400">
                یک رکورد JSON وارد کنید تا جریان DVR آن را بررسی، پاک‌سازی و ذخیره کند.
              </p>
            </div>

            <form className="space-y-3" onSubmit={handleProcess}>
              <textarea
                rows={6}
                value={recordJson}
                onChange={(e) => setRecordJson(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-xs font-mono"
              />
              <button
                type="submit"
                className="w-full flex items-center justify-center gap-2 bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-semibold rounded py-2 text-sm"
              >
                <CheckCircle2 className="w-4 h-4" /> اجرای فرآیند
              </button>
            </form>

            {processMutation.data && (
              <div className={`text-xs rounded px-3 py-2 border ${processMutation.data.success ? 'border-emerald-500/50 bg-emerald-900/20 text-emerald-200' : 'border-amber-500/50 bg-amber-900/20 text-amber-100'}`}>
                {processMutation.data.message ?? (processMutation.data.success ? 'با موفقیت ذخیره شد.' : 'پردازش انجام نشد.')}
              </div>
            )}
          </div>

          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 space-y-3 shadow-sm">
            <h2 className="text-xl font-semibold">تشخیص آنومالی رکورد</h2>
            <button
              onClick={handleEvaluate}
              className="w-full flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded py-2 text-sm"
            >
              <BarChart3 className="w-4 h-4" /> تحلیل آنومالی
            </button>

            {evaluateMutation.data && (
              <div className={`text-xs rounded px-3 py-2 border ${evaluateMutation.data.success ? 'border-emerald-500/50 bg-emerald-900/20 text-emerald-200' : 'border-amber-500/50 bg-amber-900/20 text-amber-100'}`}>
                <div className="font-mono text-[11px] whitespace-pre-wrap">
                  {JSON.stringify(evaluateMutation.data.record ?? evaluateMutation.data.message, null, 2)}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">روند آنومالی</h2>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              آخرین {anomalyTrend.length} بازه
            </span>
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
                <Line yAxisId="left" type="monotone" dataKey="anomalies" stroke="#f97316" strokeWidth={2} dot={false} />
                <Line yAxisId="left" type="monotone" dataKey="total" stroke="#22c55e" strokeWidth={2} dot={false} />
                <Line yAxisId="right" type="monotone" dataKey="ratio" stroke="#6366f1" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 shadow-sm space-y-4">
          <h2 className="text-lg font-semibold">ترکیب وضعیت اعتبارسنجی</h2>
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
      </section>

      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">تاریخچه پردازش</h2>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {filteredHistory.length} رکورد نمایش داده می‌شود
          </span>
        </div>

        {historyQuery.isError ? (
          <div className="text-sm text-amber-200 bg-amber-900/20 border border-amber-600 rounded px-4 py-3">
            خطا در دریافت تاریخچه
          </div>
        ) : filteredHistory.length === 0 ? (
          <div className="text-sm text-slate-500 dark:text-slate-300 border border-dashed border-slate-300 dark:border-slate-700 rounded-lg px-6 py-10 text-center">
            داده‌ای مطابق فیلتر انتخاب شده یافت نشد.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm text-slate-600 dark:text-slate-200">
              <thead className="bg-slate-100/70 dark:bg-slate-800/70">
                <tr>
                  <th className="px-4 py-2 text-right">زمان</th>
                  <th className="px-4 py-2 text-right">دکل</th>
                  <th className="px-4 py-2 text-right">اعتبارسنجی</th>
                  <th className="px-4 py-2 text-right">آشتی</th>
                  <th className="px-4 py-2 text-right">آنومالی</th>
                  <th className="px-4 py-2 text-right">شرح</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {filteredHistory.slice(0, 200).map((row: any) => (
                  <tr key={`${row.id}-${row.timestamp}`}>
                    <td className="px-4 py-2">
                      {row.timestamp ? new Date(row.timestamp).toLocaleString('fa-IR') : '---'}
                    </td>
                    <td className="px-4 py-2">{row.rig_id ?? '---'}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        row.validation_status === 'valid'
                          ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/40'
                          : 'bg-amber-500/10 text-amber-400 border border-amber-400/40'
                      }`}>
                        {row.validation_status}
                      </span>
                    </td>
                    <td className="px-4 py-2">{row.reconciliation_status ?? '---'}</td>
                    <td className="px-4 py-2">
                      {row.anomaly_detected ? (
                        <span className="text-rose-400">✓</span>
                      ) : (
                        <span className="text-slate-400">ـ</span>
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
      </section>
    </div>
  )
}

interface SummaryCardProps {
  title: string
  value: string | number
}

function SummaryCard({ title, value }: SummaryCardProps) {
  return (
    <div className="bg-slate-900/40 border border-slate-700 rounded px-3 py-2">
      <div className="text-slate-400 text-xs">{title}</div>
      <div className="text-white font-mono text-lg">{value}</div>
    </div>
  )
}
