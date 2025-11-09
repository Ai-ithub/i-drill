import { FormEvent, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { dvrApi } from '@/services/api'
import { AlertTriangle, BarChart3, CheckCircle2, RefreshCcw } from 'lucide-react'

export default function DVRMonitoring() {
  const queryClient = useQueryClient()
  const [recordJson, setRecordJson] = useState(`{
  "value": 42
}`)
  const [historySize, setHistorySize] = useState(100)

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
      },
    }
  )

  const stats = statsQuery.data
  const anomaly = anomalyQuery.data
  const averages = (stats?.summary?.averages ?? {}) as Record<string, number>
  const numericColumns = (anomaly?.numeric_columns ?? []) as string[]

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

  return (
    <div className="space-y-6 p-6 text-white">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">پایش DVR</h1>
        <p className="text-slate-400">
          مشاهده وضعیت پردازش داده‌ها، آمار رکوردها و پایش آنومالی.
        </p>
      </header>

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
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

          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
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
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4">
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

          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-3">
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
