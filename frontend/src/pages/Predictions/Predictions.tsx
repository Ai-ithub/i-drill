import { FormEvent, useState } from 'react'
import { useMutation } from 'react-query'
import { predictionsApi } from '@/services/api'

export default function Predictions() {
  const [rigId, setRigId] = useState('RIG_01')
  const [lookbackHours, setLookbackHours] = useState(24)
  const [modelType, setModelType] = useState('lstm')

  const predictionMutation = useMutation(
    (payload: { rig_id: string; lookback_hours: number; model_type: string }) =>
      predictionsApi
        .predictRULAuto(payload.rig_id, payload.lookback_hours, payload.model_type)
        .then((res) => res.data)
  )

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    predictionMutation.mutate({
      rig_id: rigId.trim(),
      lookback_hours: lookbackHours,
      model_type: modelType,
    })
  }

  const latestResult = predictionMutation.data

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">پیش‌بینی عمر باقی‌مانده (RUL)</h1>
        <p className="text-slate-400">
          اجرای مدل‌های یادگیری ماشین برای تخمین عمر باقیمانده تجهیزات بر اساس داده‌های تاریخی
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="block text-sm text-slate-400">شناسه دکل</label>
            <input
              value={rigId}
              onChange={(e) => setRigId(e.target.value)}
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
              placeholder="RIG_01"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm text-slate-400">بازه زمانی داده (ساعت)</label>
            <input
              type="number"
              min={1}
              max={168}
              value={lookbackHours}
              onChange={(e) => setLookbackHours(Number(e.target.value))}
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm text-slate-400">مدل</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
            >
              <option value="lstm">LSTM</option>
              <option value="transformer">Transformer</option>
              <option value="cnn_lstm">CNN-LSTM</option>
            </select>
          </div>
        </div>

        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <span className="text-sm text-slate-400">
            مدل انتخاب‌شده با استفاده از MLflow از آخرین نسخه موجود بارگذاری خواهد شد.
          </span>
          <button
            type="submit"
            className="w-full md:w-auto h-10 px-6 rounded-md bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-semibold transition disabled:opacity-60"
            disabled={predictionMutation.isLoading}
          >
            {predictionMutation.isLoading ? 'در حال پیش‌بینی...' : 'اجرای پیش‌بینی'}
          </button>
        </div>

        {predictionMutation.isError && (
          <div className="rounded-md border border-red-500/40 bg-red-900/20 px-4 py-3 text-sm text-red-300">
            خطا در اجرای پیش‌بینی: {String((predictionMutation.error as Error)?.message)}
          </div>
        )}
      </form>

      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">نتایج آخرین پیش‌بینی</h2>
          {predictionMutation.isLoading && (
            <span className="text-sm text-slate-400">در حال محاسبه...</span>
          )}
        </div>

        {!latestResult ? (
          <div className="rounded-md border border-slate-700 bg-slate-900/40 px-4 py-10 text-center text-slate-300">
            برای مشاهده خروجی، یک پیش‌بینی جدید اجرا کنید.
          </div>
        ) : latestResult.success === false ? (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-4 py-4 text-amber-100">
            {latestResult.message || 'مدل قادر به تولید پیش‌بینی نبود.'}
          </div>
        ) : (
          <div className="space-y-4">
            {latestResult.predictions?.map((prediction: any) => (
              <div
                key={`${prediction.rig_id}-${prediction.timestamp}`}
                className="rounded-md border border-cyan-500/30 bg-slate-900/40 p-4"
              >
                <div className="flex flex-wrap gap-4 justify-between items-center">
                  <div>
                    <div className="text-sm text-slate-400">دکل</div>
                    <div className="text-lg font-semibold text-white">{prediction.rig_id}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">کامپوننت</div>
                    <div className="text-lg font-semibold text-white">
                      {prediction.component || 'نامشخص'}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">عمر باقی‌مانده</div>
                    <div className="text-lg font-semibold text-cyan-400">
                      {prediction.predicted_rul?.toFixed?.(1) ?? prediction.predicted_rul ?? '-'}{' '}
                      ساعت
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">اطمینان</div>
                    <div className="text-lg font-semibold text-emerald-400">
                      {Math.round((prediction.confidence ?? 0) * 100)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">مدل</div>
                    <div className="text-lg font-semibold text-white">{prediction.model_used}</div>
                  </div>
                </div>
                <div className="mt-4 text-xs text-slate-400 flex justify-between">
                  <span>
                    به‌روزرسانی:{' '}
                    {prediction.timestamp
                      ? new Date(prediction.timestamp).toLocaleString('fa-IR')
                      : '-'}
                  </span>
                  {prediction.recommendation && (
                    <span className="text-amber-200">{prediction.recommendation}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

