import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { rlApi } from '@/services/api'
import { RotateCcw, Play, RefreshCcw, Flame, Activity, Settings, Upload, Bot, AlertTriangle } from 'lucide-react'
import { ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts'

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
  policy_mode?: string
  policy_loaded?: boolean
}

interface RLPolicyStatusView {
  mode: 'manual' | 'auto'
  policy_loaded: boolean
  source?: string | null
  identifier?: string | null
  stage?: string | null
  loaded_at?: string | null
  auto_interval_seconds?: number
  message?: string | null
}

interface PolicyFeedbackState {
  tone: 'success' | 'error'
  message: string
}

export default function RLControl() {
  const queryClient = useQueryClient()
  const [action, setAction] = useState<RLActionForm>(DEFAULT_ACTION)
  const [randomReset, setRandomReset] = useState(false)
  const [policyMode, setPolicyMode] = useState<'manual' | 'auto'>('manual')
  const [policySource, setPolicySource] = useState<'mlflow' | 'file'>('mlflow')
  const [modelName, setModelName] = useState('ppo-drill-agent')
  const [modelStage, setModelStage] = useState('Production')
  const [policyFilePath, setPolicyFilePath] = useState('')
  const [autoInterval, setAutoInterval] = useState(1)
  const [policyFeedback, setPolicyFeedback] = useState<PolicyFeedbackState | null>(null)

  useEffect(() => {
    if (!policyFeedback) return
    const timer = setTimeout(() => setPolicyFeedback(null), 5000)
    return () => clearTimeout(timer)
  }, [policyFeedback])

  const configQuery = useQuery({
    queryKey: ['rl-config'],
    queryFn: () => rlApi.getConfig().then((res) => res.data),
    refetchOnWindowFocus: false,
  })

  const stateQuery = useQuery({
    queryKey: ['rl-state'],
    queryFn: () => rlApi.getState().then((res) => res.data),
    refetchInterval: 5000,
    refetchOnWindowFocus: false,
  })

  const historyQuery = useQuery({
    queryKey: ['rl-history'],
    queryFn: () => rlApi.getHistory(50).then((res) => res.data),
    refetchInterval: 10000,
    refetchOnWindowFocus: false,
  })

  const policyStatusQuery = useQuery({
    queryKey: ['rl-policy-status'],
    queryFn: () => rlApi.getPolicyStatus().then((res) => res.data),
    refetchInterval: 10000,
    refetchOnWindowFocus: false,
  })

  const resetMutation = useMutation({
    mutationFn: (randomInit: boolean) => rlApi.reset(randomInit).then((res) => res.data),
    onSuccess: (data) => {
      queryClient.setQueryData({ queryKey: ['rl-state'] }, data)
      queryClient.invalidateQueries({ queryKey: ['rl-history'] })
    },
  })

  const stepMutation = useMutation({
    mutationFn: () => rlApi.step(action).then((res) => res.data),
    onSuccess: (data) => {
      queryClient.setQueryData({ queryKey: ['rl-state'] }, data)
      queryClient.invalidateQueries({ queryKey: ['rl-history'] })
    },
  })

  const autoStepMutation = useMutation({
    mutationFn: () => rlApi.autoStep().then((res) => res.data),
    onSuccess: (data) => {
      queryClient.setQueryData({ queryKey: ['rl-state'] }, data)
      queryClient.invalidateQueries({ queryKey: ['rl-history'] })
    },
    onError: (error: any) => {
      const detail = error.response?.data?.detail ?? 'Auto execution failed'
      setPolicyFeedback({ tone: 'error', message: detail })
    },
  })

  const loadPolicyMutation = useMutation({
    mutationFn: (payload: { source: 'mlflow' | 'file'; model_name?: string; stage?: string; file_path?: string }) =>
      rlApi.loadPolicy(payload).then((res) => res.data),
    onSuccess: (data) => {
      setPolicyFeedback({ tone: 'success', message: data.message ?? 'Model loaded successfully.' })
      queryClient.invalidateQueries({ queryKey: ['rl-policy-status'] })
    },
    onError: (error: any) => {
      const detail = error.response?.data?.detail ?? 'Model loading encountered an error.'
      setPolicyFeedback({ tone: 'error', message: detail })
    },
  })

  const policyModeMutation = useMutation({
    mutationFn: (payload: { mode: 'manual' | 'auto'; auto_interval_seconds?: number }) =>
      rlApi.setPolicyMode(payload).then((res) => res.data),
    onSuccess: (data) => {
      setPolicyFeedback({ tone: 'success', message: data.message ?? 'Policy mode updated.' })
      queryClient.invalidateQueries({ queryKey: ['rl-policy-status'] })
    },
    onError: (error: any) => {
      const detail = error.response?.data?.detail ?? 'Failed to change policy mode.'
      setPolicyFeedback({ tone: 'error', message: detail })
    },
  })

  const config = configQuery.data?.config
  const rlAvailable = config?.available ?? true
  const state = stateQuery.data?.state as RLEnvironmentStateView | undefined
  const history = (historyQuery.data?.history ?? []) as RLEnvironmentStateView[]
  const policyStatus = policyStatusQuery.data?.status as RLPolicyStatusView | undefined

  useEffect(() => {
    if (!policyStatus) return
    if (policyStatus.mode && policyStatus.mode !== policyMode) {
      setPolicyMode(policyStatus.mode)
    }
    if (typeof policyStatus.auto_interval_seconds === 'number') {
      setAutoInterval(Number(policyStatus.auto_interval_seconds.toFixed(2)))
    }
  }, [policyStatus?.mode, policyStatus?.auto_interval_seconds])

  // Map Persian observation labels to English
  const mapObservationLabelToEnglish = (label: string): string => {
    const persianToEnglish: Record<string, string> = {
      'عمق': 'Depth',
      'فرسایش مته': 'Bit Wear',
      'نرخ حفاری': 'ROP',
      'گشتاور': 'Torque',
      'فشار': 'Pressure',
      'ارتعاش محوری': 'Vibration Axial',
      'ارتعاش جانبی': 'Vibration Lateral',
      'ارتعاش پیچشی': 'Vibration Torsional',
    }
    return persianToEnglish[label] || label
  }

  const observationLabels = useMemo(() => {
    if (config?.observation_labels?.length === state?.observation?.length) {
      return config.observation_labels.map(mapObservationLabelToEnglish)
    }
    return state?.observation?.map((_, idx: number) => `Observation ${idx + 1}`) ?? []
  }, [config, state])

  const chartData = useMemo(() => {
    return history.map((entry) => ({
      step: entry.step,
      reward: entry.reward,
      depth: entry.observation?.[0] ?? 0,
    }))
  }, [history])

  const isManualMode = policyMode === 'manual'
  const isBusy = resetMutation.isLoading || stepMutation.isLoading || autoStepMutation.isLoading
  const hasPolicyLoaded = policyStatus?.policy_loaded ?? false

  const latestLoadedAt = policyStatus?.loaded_at
    ? new Date(policyStatus.loaded_at).toLocaleString('en-US')
    : '—'

  const handleLoadPolicy = () => {
    if (policySource === 'mlflow') {
      loadPolicyMutation.mutate({ source: 'mlflow', model_name: modelName, stage: modelStage })
    } else {
      loadPolicyMutation.mutate({ source: 'file', file_path: policyFilePath })
    }
  }

  const handlePolicyModeChange = (mode: 'manual' | 'auto') => {
    if (mode === policyMode) return
    policyModeMutation.mutate({ mode, auto_interval_seconds: mode === 'auto' ? autoInterval : undefined })
  }

  const handleAutoIntervalChange = (value: number) => {
    if (Number.isNaN(value)) return
    setAutoInterval(Math.max(0.5, value))
  }

  return (
    <div className="space-y-6 p-6 text-white">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Reinforcement Learning Agent Control</h1>
        <p className="text-slate-400">
          Execute drilling environment steps and manage RL agent policies. Load trained models (MLflow or file),
          switch between manual/auto modes, and view reward trends.
        </p>
      </header>

      {policyFeedback && (
        <div
          className={`rounded-md border px-4 py-3 text-sm ${
            policyFeedback.tone === 'error'
              ? 'border-red-500/40 bg-red-900/20 text-red-200'
              : 'border-emerald-500/40 bg-emerald-900/20 text-emerald-100'
          }`}
        >
          {policyFeedback.message}
        </div>
      )}

      {!rlAvailable && (
        <div className="rounded-md border border-amber-500/40 bg-amber-900/20 px-4 py-3 text-sm text-amber-100 flex items-center gap-3">
          <AlertTriangle className="w-4 h-4" />
          RL environment is not installed. To use, install `gym` and `drilling_env` packages and restart the service.
        </div>
      )}

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-xl font-semibold">Current Status</h2>
                <p className="text-xs text-slate-400">Last updated every 5 seconds</p>
              </div>
              <button
                onClick={() => {
                  queryClient.invalidateQueries(['rl-state'])
                  queryClient.invalidateQueries(['rl-history'])
                }}
                className="flex items-center gap-2 text-sm px-3 py-2 rounded bg-slate-700 hover:bg-slate-600"
              >
                <RefreshCcw className={`w-4 h-4 ${stateQuery.isFetching ? 'animate-spin' : ''}`} /> Refresh
              </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard title="Step" value={state?.step ?? 0} icon={<Activity className="w-5 h-5 text-cyan-400" />} />
              <StatCard title="Episode" value={state?.episode ?? 0} icon={<Flame className="w-5 h-5 text-orange-400" />} />
              <StatCard title="Reward" value={(state?.reward ?? 0).toFixed(3)} icon={<Play className="w-5 h-5 text-emerald-400" />} />
              <StatCard title="Policy Mode" value={policyMode === 'auto' ? 'Auto' : 'Manual'} icon={<Bot className="w-5 h-5 text-fuchsia-400" />} />
            </div>

            <div className="mt-6 space-y-2">
              <h3 className="text-sm text-slate-300">Observation</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {observationLabels.map((label: string, idx: number) => (
                  <div key={idx} className="bg-slate-900/60 border border-slate-700 rounded px-3 py-2">
                    <div className="text-slate-400">{label}</div>
                    <div className="text-slate-100 font-mono text-sm">
                      {state?.observation ? state.observation[idx]?.toFixed?.(4) ?? state.observation[idx] : '---'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {state?.info && Object.keys(state.info).length > 0 && (
              <div className="mt-6 text-xs bg-slate-900/40 border border-slate-700 rounded px-4 py-3">
                <span className="text-slate-400">Info:</span>
                <pre className="mt-1 text-slate-200 whitespace-pre-wrap font-mono text-[11px]">
                  {JSON.stringify(state.info, null, 2)}
                </pre>
              </div>
            )}
          </div>

          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Reward and Depth Chart</h2>
              <p className="text-xs text-slate-400">Last {chartData.length} recorded steps</p>
            </div>
            {chartData.length === 0 ? (
              <div className="text-sm text-slate-400">No data available for chart.</div>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="rgba(148, 163, 184, 0.2)" strokeDasharray="3 3" />
                    <XAxis dataKey="step" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <YAxis yAxisId="left" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Reward', angle: -90, position: 'insideLeft', fill: '#22d3ee' }} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Depth', angle: 90, position: 'insideRight', fill: '#f97316' }} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#e2e8f0' }}
                    />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="reward" stroke="#22d3ee" strokeWidth={2} dot={false} />
                    <Line yAxisId="right" type="monotone" dataKey="depth" stroke="#f97316" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Recent History</h2>
            {history.length === 0 ? (
              <div className="text-sm text-slate-400">No data available to display.</div>
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
                    {history
                      .slice()
                      .reverse()
                      .map((entry: RLEnvironmentStateView, index: number) => (
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
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Policy Management</h2>
              <Settings className="w-5 h-5 text-cyan-400" />
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs text-slate-300">
              <div>
                <span className="block text-slate-400">Model Status</span>
                <span className={hasPolicyLoaded ? 'text-emerald-300' : 'text-amber-300'}>
                  {hasPolicyLoaded ? 'Loaded' : 'Not Available'}
                </span>
              </div>
              <div>
                <span className="block text-slate-400">Current Mode</span>
                <span className="text-sky-300">{policyMode === 'auto' ? 'Auto' : 'Manual'}</span>
              </div>
              <div>
                <span className="block text-slate-400">Source</span>
                <span>{policyStatus?.source ?? '---'}</span>
              </div>
              <div>
                <span className="block text-slate-400">Last Loaded</span>
                <span>{latestLoadedAt}</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => handlePolicyModeChange('manual')}
                disabled={policyMode === 'manual' || policyModeMutation.isLoading}
                className={`flex-1 flex items-center justify-center gap-2 rounded-lg py-2 text-sm font-semibold transition ${
                  policyMode === 'manual'
                    ? 'bg-cyan-600 text-white'
                    : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
                }`}
              >
                <Settings className="w-4 h-4" /> Manual Mode
              </button>
              <button
                onClick={() => handlePolicyModeChange('auto')}
                disabled={policyMode === 'auto' || policyModeMutation.isLoading || !hasPolicyLoaded}
                className={`flex-1 flex items-center justify-center gap-2 rounded-lg py-2 text-sm font-semibold transition ${
                  policyMode === 'auto'
                    ? 'bg-fuchsia-600 text-white'
                    : hasPolicyLoaded
                    ? 'bg-slate-700 hover:bg-slate-600 text-slate-200'
                    : 'bg-slate-900 text-slate-500 cursor-not-allowed'
                }`}
              >
                <Bot className="w-4 h-4" /> Auto Mode
              </button>
            </div>

            <div className="space-y-3">
              <label className="block text-xs text-slate-400">Auto Execution Interval (seconds)</label>
              <input
                type="number"
                min={0.5}
                step={0.25}
                value={autoInterval}
                onChange={(e) => handleAutoIntervalChange(parseFloat(e.target.value))}
                className="w-full rounded bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
              />
              <p className="text-[11px] text-slate-500">
                This value is sent to the service when auto mode is activated. Values less than 0.5 seconds are not accepted.
              </p>
            </div>

            <div className="space-y-3">
              <label className="block text-xs text-slate-400">Model Source</label>
              <select
                value={policySource}
                onChange={(e) => setPolicySource(e.target.value as 'mlflow' | 'file')}
                className="w-full rounded bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
              >
                <option value="mlflow">MLflow Registry</option>
                <option value="file">Local File (.pt)</option>
              </select>

              {policySource === 'mlflow' ? (
                <div className="grid grid-cols-1 gap-3">
                  <div>
                    <label className="block text-xs text-slate-400">Model Name</label>
                    <input
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      className="w-full rounded bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
                      placeholder="ppo-drill-agent"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400">Stage</label>
                    <input
                      value={modelStage}
                      onChange={(e) => setModelStage(e.target.value)}
                      className="w-full rounded bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
                      placeholder="Production"
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <label className="block text-xs text-slate-400">File Path</label>
                  <input
                    value={policyFilePath}
                    onChange={(e) => setPolicyFilePath(e.target.value)}
                    className="w-full rounded bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
                    placeholder="C:/models/ppo.pt"
                  />
                </div>
              )}

              <button
                onClick={handleLoadPolicy}
                disabled={loadPolicyMutation.isLoading || (policySource === 'file' && policyFilePath.trim() === '')}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-900 font-semibold py-3 transition disabled:bg-slate-700 disabled:text-slate-400"
              >
                <Upload className="w-4 h-4" />
                {loadPolicyMutation.isLoading ? 'Loading...' : 'Load Model'}
              </button>
            </div>
          </div>

          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-4">
            <h2 className="text-xl font-semibold">Apply Action</h2>
            <div className="space-y-4">
              <ActionInput
                label="WOB"
                suffix="N"
                value={action.wob}
                min={config?.action_space?.wob?.min ?? 0}
                max={config?.action_space?.wob?.max ?? 50000}
                step={100}
                onChange={(value) => setAction((prev) => ({ ...prev, wob: value }))}
                disabled={!isManualMode || !rlAvailable}
              />
              <ActionInput
                label="RPM"
                suffix="rpm"
                value={action.rpm}
                min={config?.action_space?.rpm?.min ?? 0}
                max={config?.action_space?.rpm?.max ?? 200}
                step={1}
                onChange={(value) => setAction((prev) => ({ ...prev, rpm: value }))}
                disabled={!isManualMode || !rlAvailable}
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
                disabled={!isManualMode || !rlAvailable}
              />
            </div>

            <div className="mt-6 space-y-3">
              <button
                onClick={() => stepMutation.mutate()}
                disabled={!rlAvailable || isBusy || !isManualMode}
                className="w-full flex items-center justify-center gap-3 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-semibold py-3 transition disabled:bg-slate-700 disabled:text-slate-400"
              >
                <Play className="w-4 h-4" /> Execute Manual Step
              </button>
              <button
                onClick={() => autoStepMutation.mutate()}
                disabled={!rlAvailable || policyMode !== 'auto' || autoStepMutation.isLoading || !hasPolicyLoaded}
                className="w-full flex items-center justify-center gap-3 rounded-lg bg-fuchsia-500 hover:bg-fuchsia-400 text-slate-900 font-semibold py-3 transition disabled:bg-slate-700 disabled:text-slate-400"
              >
                <Bot className="w-4 h-4" />
                {autoStepMutation.isLoading ? 'Executing auto step...' : 'Execute Auto Step'}
              </button>
              <button
                onClick={() => resetMutation.mutate(randomReset)}
                disabled={isBusy}
                className="w-full flex items-center justify-center gap-3 rounded-lg bg-slate-700 hover:bg-slate-600 text-white font-semibold py-3 transition"
              >
                <RotateCcw className="w-4 h-4" /> Reset Environment
              </button>
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={randomReset}
                  onChange={(e) => setRandomReset(e.target.checked)}
                  className="rounded border-slate-600 bg-slate-800"
                />
                Random initialization on reset
              </label>
            </div>

            <div className="mt-6 text-xs text-slate-400">
              <p>In auto mode, manual actions are disabled and the loaded model generates actions.</p>
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
  disabled?: boolean
}

function ActionInput({ label, suffix, value, min, max, step, decimals = 2, onChange, disabled = false }: ActionInputProps) {
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
        disabled={disabled}
        className={`w-full ${disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className={`w-full rounded bg-slate-900 border border-slate-700 px-3 py-2 text-sm ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      />
    </div>
  )
}
