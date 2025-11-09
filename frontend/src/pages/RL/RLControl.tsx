import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { rlApi } from '@/services/api'
import { RotateCcw, Play, RefreshCcw, Flame, Activity } from 'lucide-react'

interface RLActionForm {
  wob: number
  rpm: number
  flow_rate: number
}

const DEFAULT_ACTION: RLActionForm = {
  wob: 10000,
  rpm: 80,
  flow_rate: 0.05,
}

interface RLEnvironmentStateView {
  observation: number[]
  reward: number
  done: boolean
  info: Record<string, unknown>
  step: number
  episode: number
  warning?: string
  action?: Record<string, number>
}

export default function RLControl() {
  const queryClient = useQueryClient()
  const [action, setAction] = useState<RLActionForm>(DEFAULT_ACTION)
  const [randomReset, setRandomReset] = useState(false)

  const configQuery = useQuery(['rl-config'], () => rlApi.getConfig().then((res) => res.data), {
    refetchOnWindowFocus: false,
  })

  const stateQuery = useQuery(['rl-state'], () => rlApi.getState().then((res) => res.data), {
    refetchInterval: 5000,
    refetchOnWindowFocus: false,
  })

  const historyQuery = useQuery(['rl-history'], () => rlApi.getHistory(20).then((res) => res.data), {
    refetchInterval: 10000,
    refetchOnWindowFocus: false,
  })

  const resetMutation = useMutation((randomInit: boolean) => rlApi.reset(randomInit).then((res) => res.data), {
    onSuccess: (data) => {
      queryClient.setQueryData(['rl-state'], data)
      queryClient.invalidateQueries(['rl-history'])
    },
  })

  const stepMutation = useMutation(() => rlApi.step(action).then((res) => res.data), {
    onSuccess: (data) => {
      queryClient.setQueryData(['rl-state'], data)
      queryClient.invalidateQueries(['rl-history'])
    },
  })

  const config = configQuery.data?.config
  const rlAvailable = config?.available ?? true
  const state = stateQuery.data?.state as RLEnvironmentStateView | undefined
  const history = (historyQuery.data?.history ?? []) as RLEnvironmentStateView[]

  const observationLabels = useMemo(() => {
    if (config?.observation_labels?.length === state?.observation?.length) {
      return config.observation_labels
    }
    return state?.observation?.map((_, idx: number) => `Feature ${idx + 1}`) ?? []
  }, [config, state])

  const isBusy = resetMutation.isLoading || stepMutation.isLoading

  return (
    <div className="space-y-6 p-6 text-white">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">کنترل عامل تقویتی</h1>
        <p className="text-slate-400">
          اجرای گام‌های محیط حفاری و پایش وضعیت عامل RL. در صورت عدم دسترسی به محیط، پیام هشدار نمایش داده می‌شود.
        </p>
      </header>

      {!rlAvailable && (
        <div className="rounded-md border border-amber-500/40 bg-amber-900/20 px-4 py-3 text-sm text-amber-100">
          محیط RL در این محیط نصب نشده است. جهت استفاده، بسته‌های `gym` و `drilling_env` را نصب و سرویس را مجدداً راه‌اندازی کنید.
        </div>
      )}

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-xl font-semibold">وضعیت فعلی</h2>
                <p className="text-xs text-slate-400">آخرین بروزرسانی هر ۵ ثانیه</p>
              </div>
              <button
                onClick={() => queryClient.invalidateQueries(['rl-state'])}
                className="flex items-center gap-2 text-sm px-3 py-2 rounded bg-slate-700 hover:bg-slate-600"
              >
                <RefreshCcw className="w-4 h-4" /> بروزرسانی
              </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard title="گام" value={state?.step ?? 0} icon={<Activity className="w-5 h-5 text-cyan-400" />} />
              <StatCard title="اپیزود" value={state?.episode ?? 0} icon={<Flame className="w-5 h-5 text-orange-400" />} />
              <StatCard title="پاداش" value={(state?.reward ?? 0).toFixed(3)} icon={<Play className="w-5 h-5 text-emerald-400" />} />
              <StatCard title="پایان؟" value={state?.done ? 'بله' : 'خیر'} icon={<RotateCcw className="w-5 h-5 text-amber-400" />} />
            </div>

            <div className="mt-6 space-y-2">
              <h3 className="text-sm text-slate-300">مشاهده</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {observationLabels.map((label: string, idx: number) => (
                  <div key={idx} className="bg-slate-900/60 border border-slate-700 rounded px-3 py-2">
                    <div className="text-slate-400">{label}</div>
                    <div className="text-slate-100 font-mono text-sm">
                      {state?.observation
                        ? state.observation[idx]?.toFixed?.(4) ?? state.observation[idx]
                        : '---'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {state?.info && Object.keys(state.info).length > 0 && (
              <div className="mt-6 text-xs bg-slate-900/40 border border-slate-700 rounded px-4 py-3">
                <span className="text-slate-400">اطلاعات:</span>
                <pre className="mt-1 text-slate-200 whitespace-pre-wrap font-mono text-[11px]">
                  {JSON.stringify(state.info, null, 2)}
                </pre>
              </div>
            )}
          </div>

          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">تاریخچه اخیر</h2>
            {history.length === 0 ? (
              <div className="text-sm text-slate-400">داده‌ای برای نمایش وجود ندارد.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-700 text-left text-xs">
                  <thead className="text-slate-300">
                    <tr>
                      <th className="px-4 py-2">Step</th>
                      <th className="px-4 py-2">Reward</th>
                      <th className="px-4 py-2">Done</th>
                      <th className="px-4 py-2">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-slate-200">
                    {history.slice().reverse().map((entry: RLEnvironmentStateView, index: number) => (
                      <tr key={index}>
                        <td className="px-4 py-2 font-mono">{entry.step}</td>
                        <td className="px-4 py-2 font-mono">{entry.reward.toFixed(3)}</td>
                        <td className="px-4 py-2">{entry.done ? '✔' : '—'}</td>
                        <td className="px-4 py-2 font-mono text-[11px]">
                          {entry.action ? JSON.stringify(entry.action) : '---'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">اعمال عامل</h2>
            <div className="space-y-4">
              <ActionInput
                label="WOB"
                suffix="N"
                value={action.wob}
                min={config?.action_space?.wob?.min ?? 0}
                max={config?.action_space?.wob?.max ?? 50000}
                step={100}
                onChange={(value) => setAction((prev) => ({ ...prev, wob: value }))}
              />
              <ActionInput
                label="RPM"
                suffix="rpm"
                value={action.rpm}
                min={config?.action_space?.rpm?.min ?? 0}
                max={config?.action_space?.rpm?.max ?? 200}
                step={1}
                onChange={(value) => setAction((prev) => ({ ...prev, rpm: value }))}
              />
              <ActionInput
                label="Flow Rate"
                suffix="m³/s"
                value={action.flow_rate}
                min={config?.action_space?.flow_rate?.min ?? 0}
                max={config?.action_space?.flow_rate?.max ?? 0.1}
                step={0.005}
                decimals={3}
                onChange={(value) => setAction((prev) => ({ ...prev, flow_rate: value }))}
              />
            </div>

            <div className="mt-6 space-y-3">
              <button
                onClick={() => stepMutation.mutate()}
                disabled={!rlAvailable || isBusy}
                className="w-full flex items-center justify-center gap-3 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-semibold py-3 transition disabled:bg-slate-700 disabled:text-slate-400"
              >
                <Play className="w-4 h-4" /> اجرای گام
              </button>
              <button
                onClick={() => resetMutation.mutate(randomReset)}
                disabled={isBusy}
                className="w-full flex items-center justify-center gap-3 rounded-lg bg-slate-700 hover:bg-slate-600 text-white font-semibold py-3 transition"
              >
                <RotateCcw className="w-4 h-4" /> ریست محیط
              </button>
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={randomReset}
                  onChange={(e) => setRandomReset(e.target.checked)}
                  className="rounded border-slate-600 bg-slate-800"
                />
                مقداردهی تصادفی در هنگام ریست
              </label>
            </div>

            <div className="mt-6 text-xs text-slate-400">
              <p>پیشنهاد: برای مشاهدهٔ تأثیر اقدامات، پس از هر گام چند ثانیه صبر کنید تا وضعیت پایدار شود.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

interface StatCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
}

function StatCard({ title, value, icon }: StatCardProps) {
  return (
    <div className="bg-slate-900/60 border border-slate-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2 text-slate-400 text-xs">
        <span>{title}</span>
        {icon}
      </div>
      <div className="text-lg font-mono text-white">{value}</div>
    </div>
  )
}

interface ActionInputProps {
  label: string
  suffix: string
  value: number
  min: number
  max: number
  step: number
  decimals?: number
  onChange: (value: number) => void
}

function ActionInput({ label, suffix, value, min, max, step, decimals = 2, onChange }: ActionInputProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-slate-300">
        <span>{label}</span>
        <span>
          {value.toFixed(decimals)} {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full rounded bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
      />
    </div>
  )
}
