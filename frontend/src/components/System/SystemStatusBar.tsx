import { useQuery } from '@tanstack/react-query'
import { healthApi } from '@/services/api'
import { AlertTriangle, MessageSquare } from 'lucide-react'

interface StatusBadgeProps {
  label: string
  status?: string
  description?: string
}

const STATUS_COLOR: Record<string, string> = {
  healthy: 'bg-emerald-500/15 text-emerald-500 border border-emerald-500/40',
  available: 'bg-emerald-500/15 text-emerald-500 border border-emerald-500/40',
  degraded: 'bg-amber-500/15 text-amber-400 border border-amber-400/50',
  unavailable: 'bg-rose-500/10 text-rose-400 border border-rose-400/40',
  unhealthy: 'bg-rose-500/10 text-rose-400 border border-rose-400/40',
}

function StatusBadge({ label, status, description }: StatusBadgeProps) {
  const normalized = (status ?? '').toLowerCase()
  const tone = STATUS_COLOR[normalized] ?? 'bg-slate-200/60 text-slate-600 border border-slate-300'
  return (
    <div className="flex flex-col gap-2 rounded-xl bg-white dark:bg-slate-900/60 px-4 py-4 border border-slate-200 dark:border-slate-800 shadow-sm">
      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
        <span>{label}</span>
        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${tone}`}>{status || 'unknown'}</span>
      </div>
      {description && <p className="text-xs text-slate-500 dark:text-slate-300 leading-relaxed">{description}</p>}
    </div>
  )
}

export default function SystemStatusBar() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => healthApi.detailed().then((res) => res.data),
    refetchInterval: 20000,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    staleTime: 15000,
  })

  const details = data?.details ?? data ?? {}
  const kafkaStatus = details.kafka?.kafka ?? details.kafka?.status ?? 'unknown'
  const dbStatus = details.database?.database ?? details.database?.status ?? 'unknown'
  const rlStatus = details.rl_environment?.rl_environment ?? details.rl_environment?.status ?? 'unknown'
  const mlflowStatus = details.mlflow?.mlflow ?? details.mlflow?.status ?? 'unknown'

  return (
    <div className="rounded-2xl bg-slate-100/70 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 px-4 py-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-cyan-500" />
          <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-100">Infrastructure Services Status</h3>
        </div>
        {isLoading && <span className="text-xs text-slate-400">Updating...</span>}
        {isError && (
          <span className="inline-flex items-center gap-1 text-xs text-amber-500">
            <AlertTriangle className="w-4 h-4" /> Error fetching status
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        <StatusBadge label="Kafka" status={kafkaStatus} description={details.kafka?.message} />
        <StatusBadge label="PostgreSQL" status={dbStatus} description={details.database?.message} />
        <StatusBadge label="RL Environment" status={rlStatus} description={details.rl_environment?.message} />
        <StatusBadge label="MLflow" status={mlflowStatus} description={details.mlflow?.message} />
      </div>
    </div>
  )
}

