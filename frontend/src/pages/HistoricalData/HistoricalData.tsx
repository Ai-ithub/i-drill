import { FormEvent, useMemo, useState } from 'react'
import { useQuery } from 'react-query'
import { sensorDataApi } from '@/services/api'

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
  const [queryParams, setQueryParams] = useState(() => ({
    rig_id: 'RIG_01',
    start_time: new Date(defaultStart).toISOString(),
    end_time: new Date(defaultEnd).toISOString(),
    limit: 200,
    parameters: '',
  }))

  const historicalQuery = useQuery(
    ['historical-data', queryParams],
    () =>
      sensorDataApi
        .getHistorical({
          rig_id: queryParams.rig_id,
          start_time: queryParams.start_time,
          end_time: queryParams.end_time,
          parameters: queryParams.parameters ? queryParams.parameters : undefined,
          limit: queryParams.limit,
        })
        .then((res) => res.data),
    {
      keepPreviousData: true,
    }
  )

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const startDate = new Date(formState.start)
    const endDate = new Date(formState.end)

    if (startDate >= endDate) {
      setValidationError('بازه زمانی معتبر نیست. زمان شروع باید قبل از زمان پایان باشد.')
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
  const summary = useMemo(() => {
    if (!records.length) {
      return null
    }

    const totals = records.reduce(
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
      count: records.length,
      avgDepth: totals.depth / records.length,
      avgWob: totals.wob / records.length,
      avgRpm: totals.rpm / records.length,
      avgRop: totals.rop / records.length,
    }
  }, [records])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">داده‌های تاریخی</h1>
        <p className="text-slate-400">
          جستجو و بررسی داده‌های تاریخی سنسورها برای تحلیل روند و گزارش‌گیری
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4"
      >
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          <div className="space-y-2">
            <label className="block text-sm text-slate-400">شناسه دکل</label>
            <input
              value={formState.rigId}
              onChange={(e) => setFormState((prev) => ({ ...prev, rigId: e.target.value }))}
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
              placeholder="RIG_01"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm text-slate-400">از زمان</label>
            <input
              type="datetime-local"
              value={formState.start}
              onChange={(e) => setFormState((prev) => ({ ...prev, start: e.target.value }))}
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm text-slate-400">تا زمان</label>
            <input
              type="datetime-local"
              value={formState.end}
              onChange={(e) => setFormState((prev) => ({ ...prev, end: e.target.value }))}
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm text-slate-400">حداکثر رکورد</label>
            <input
              type="number"
              min={1}
              max={10000}
              value={formState.limit}
              onChange={(e) =>
                setFormState((prev) => ({ ...prev, limit: Number(e.target.value) }))
              }
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="space-y-2 lg:col-span-2">
            <label className="block text-sm text-slate-400">پارامترها (با کاما جدا کنید)</label>
            <input
              value={formState.parameters}
              onChange={(e) =>
                setFormState((prev) => ({ ...prev, parameters: e.target.value }))
              }
              placeholder="wob,rpm,torque"
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="flex items-end">
            <button
              type="submit"
              className="w-full h-10 rounded-md bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-semibold transition"
              disabled={historicalQuery.isFetching}
            >
              {historicalQuery.isFetching ? 'در حال بازیابی...' : 'بازیابی داده'}
            </button>
          </div>
        </div>

        {validationError && (
          <div className="rounded-md border border-red-500/40 bg-red-900/20 px-4 py-2 text-sm text-red-300">
            {validationError}
          </div>
        )}
      </form>

      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-semibold text-white">نتایج</h2>
            <p className="text-sm text-slate-400">
              {historicalQuery.isFetching ? 'در حال بارگذاری...' : `تعداد رکورد: ${records.length}`}
            </p>
          </div>
          {historicalQuery.isLoading && (
            <span className="text-sm text-slate-400">در حال بارگذاری اولیه...</span>
          )}
        </div>

        {summary && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4 mb-6">
            <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-4">
              <div className="text-xs text-slate-400 mb-1">تعداد رکورد</div>
              <div className="text-2xl font-mono text-white">{summary.count}</div>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-4">
              <div className="text-xs text-slate-400 mb-1">میانگین عمق</div>
              <div className="text-2xl font-mono text-cyan-400">
                {summary.avgDepth.toFixed(1)} ft
              </div>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-4">
              <div className="text-xs text-slate-400 mb-1">میانگین WOB</div>
              <div className="text-2xl font-mono text-cyan-400">
                {summary.avgWob.toFixed(0)} lbs
              </div>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-4">
              <div className="text-xs text-slate-400 mb-1">میانگین RPM</div>
              <div className="text-2xl font-mono text-cyan-400">
                {summary.avgRpm.toFixed(0)}
              </div>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-4">
              <div className="text-xs text-slate-400 mb-1">میانگین ROP</div>
              <div className="text-2xl font-mono text-cyan-400">
                {summary.avgRop.toFixed(2)} ft/hr
              </div>
            </div>
          </div>
        )}

        {historicalQuery.isError ? (
          <div className="rounded-md border border-red-500/40 bg-red-900/20 px-4 py-4 text-red-300">
            خطا در واکشی داده‌ها: {(historicalQuery.error as Error)?.message ?? 'نامشخص'}
          </div>
        ) : records.length === 0 && !historicalQuery.isFetching ? (
          <div className="rounded-md border border-slate-700 bg-slate-900/40 px-4 py-10 text-center text-slate-300">
            هیچ داده‌ای برای بازه انتخاب شده یافت نشد.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-700 text-left">
              <thead>
                <tr className="text-slate-300 text-sm uppercase">
                  <th className="px-4 py-3">زمان</th>
                  <th className="px-4 py-3">عمق</th>
                  <th className="px-4 py-3">WOB</th>
                  <th className="px-4 py-3">RPM</th>
                  <th className="px-4 py-3">Torque</th>
                  <th className="px-4 py-3">ROP</th>
                  <th className="px-4 py-3">Pressure</th>
                  <th className="px-4 py-3">وضعیت</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-sm text-slate-200 font-mono">
                {records.map((row: any) => (
                  <tr key={`${row.id}-${row.timestamp}`}>
                    <td className="px-4 py-2">
                      {row.timestamp ? new Date(row.timestamp).toLocaleString('fa-IR') : '-'}
                    </td>
                    <td className="px-4 py-2">{row.depth?.toFixed?.(2) ?? row.depth ?? '-'}</td>
                    <td className="px-4 py-2">{row.wob?.toFixed?.(1) ?? row.wob ?? '-'}</td>
                    <td className="px-4 py-2">{row.rpm?.toFixed?.(0) ?? row.rpm ?? '-'}</td>
                    <td className="px-4 py-2">{row.torque?.toFixed?.(0) ?? row.torque ?? '-'}</td>
                    <td className="px-4 py-2">{row.rop?.toFixed?.(2) ?? row.rop ?? '-'}</td>
                    <td className="px-4 py-2">
                      {row.mud_pressure?.toFixed?.(0) ?? row.mud_pressure ?? '-'}
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className={`px-2 py-1 rounded-full text-xs ${
                          row.status === 'normal'
                            ? 'bg-green-500/10 text-green-400 border border-green-500/40'
                            : 'bg-amber-500/10 text-amber-300 border border-amber-500/40'
                        }`}
                      >
                        {row.status ?? 'نامشخص'}
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

