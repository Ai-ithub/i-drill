import { useState, useCallback, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { controlApi } from '@/services/api'
import { CheckCircle2, X, Clock, User, Bot, AlertTriangle, History } from 'lucide-react'

export interface ChangeRequest {
  id: string
  type: 'optimization' | 'maintenance' | 'validation'
  component: string
  parameter: string
  currentValue: number | string
  newValue: number | string
  unit?: string
  reason: string
  priority: 'critical' | 'high' | 'medium' | 'low'
  autoExecute: boolean
  status: 'pending' | 'approved' | 'rejected' | 'executed' | 'failed'
  requestedBy: 'user' | 'ai'
  timestamp: string
  executedAt?: string
  executedBy?: 'user' | 'ai'
}

interface ChangeManagerProps {
  rigId: string
  autoExecutionEnabled: boolean
  onAutoExecutionToggle: (enabled: boolean) => void
  children?: React.ReactNode
}

export function useChangeManager(rigId: string) {
  const queryClient = useQueryClient()
  const [changes, setChanges] = useState<ChangeRequest[]>([])
  const [autoExecutionEnabled, setAutoExecutionEnabled] = useState(false)

  // Load changes from localStorage on mount
  useEffect(() => {
    const loadChanges = () => {
      const saved = localStorage.getItem(`changes-${rigId}`)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          setChanges(Array.isArray(parsed) ? parsed : [])
        } catch (e) {
          // Ignore parse errors
          console.error('Error parsing changes from localStorage:', e)
        }
      }
    }

    loadChanges()

    // Listen for storage events (when other tabs/components update localStorage)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === `changes-${rigId}` && e.newValue) {
        try {
          const parsed = JSON.parse(e.newValue)
          setChanges(Array.isArray(parsed) ? parsed : [])
        } catch (error) {
          console.error('Error parsing changes from storage event:', error)
        }
      }
    }

    window.addEventListener('storage', handleStorageChange)

    // Also listen for custom storage events (for same-tab updates)
    const handleCustomStorageChange = (e: Event) => {
      const customEvent = e as CustomEvent
      if (customEvent.detail?.key === `changes-${rigId}`) {
        loadChanges()
      }
    }

    window.addEventListener('localStorageChange', handleCustomStorageChange as EventListener)

    return () => {
      window.removeEventListener('storage', handleStorageChange)
      window.removeEventListener('localStorageChange', handleCustomStorageChange as EventListener)
    }
  }, [rigId])

  // Save changes to localStorage
  useEffect(() => {
    const key = `changes-${rigId}`
    localStorage.setItem(key, JSON.stringify(changes))
    
    // Dispatch custom event for same-tab synchronization
    window.dispatchEvent(
      new CustomEvent('localStorageChange', {
        detail: { key, value: changes }
      })
    )
  }, [changes, rigId])

  // Apply change mutation
  const applyChangeMutation = useMutation({
    mutationFn: async (change: ChangeRequest) => {
      // Call the actual API
      const response = await controlApi.applyChange({
        rig_id: rigId,
        change_type: change.type,
        component: change.component,
        parameter: change.parameter,
        value: change.newValue,
        auto_execute: change.autoExecute,
      })
      return response.data
    },
    onSuccess: (data, change) => {
      // Update change status
      setChanges((prev) =>
        prev.map((c) =>
          c.id === change.id
            ? { ...c, status: 'executed' as const, executedAt: new Date().toISOString(), executedBy: change.autoExecute ? 'ai' : 'user' }
            : c
        )
      )
      // Invalidate relevant queries
      queryClient.invalidateQueries(['analytics', rigId])
      queryClient.invalidateQueries(['realtime', rigId])
    },
    onError: (error, change) => {
      setChanges((prev) =>
        prev.map((c) => (c.id === change.id ? { ...c, status: 'failed' as const } : c))
      )
    },
  })

  // Create change request
  const createChange = useCallback(
    (change: Omit<ChangeRequest, 'id' | 'status' | 'timestamp' | 'requestedBy'>) => {
      const newChange: ChangeRequest = {
        ...change,
        id: `change-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        status: 'pending',
        timestamp: new Date().toISOString(),
        requestedBy: change.autoExecute && autoExecutionEnabled ? 'ai' : 'user',
      }

      setChanges((prev) => [...prev, newChange])

      // Auto-execute if enabled
      if (change.autoExecute && autoExecutionEnabled) {
        setTimeout(() => {
          applyChangeMutation.mutate(newChange)
        }, 1000)
      }

      return newChange.id
    },
    [autoExecutionEnabled, applyChangeMutation]
  )

  // Approve change
  const approveChange = useCallback(
    (changeId: string) => {
      setChanges((prev) => {
        const updated = prev.map((c) => (c.id === changeId ? { ...c, status: 'approved' as const } : c))
        const change = prev.find((c) => c.id === changeId)
        if (change) {
          // Apply the change after state update
          setTimeout(() => {
            applyChangeMutation.mutate({ ...change, status: 'approved' })
          }, 0)
        }
        return updated
      })
    },
    [applyChangeMutation]
  )

  // Reject change
  const rejectChange = useCallback((changeId: string) => {
    setChanges((prev) =>
      prev.map((c) => (c.id === changeId ? { ...c, status: 'rejected' as const } : c))
    )
  }, [])

  // Manual apply change
  const applyChange = useCallback(
    (changeId: string) => {
      setChanges((prev) => {
        const change = prev.find((c) => c.id === changeId)
        if (change) {
          // Apply the change
          applyChangeMutation.mutate({ ...change, autoExecute: false })
        }
        return prev
      })
    },
    [applyChangeMutation]
  )

  return {
    changes,
    autoExecutionEnabled,
    setAutoExecutionEnabled,
    createChange,
    approveChange,
    rejectChange,
    applyChange,
    isApplying: applyChangeMutation.isPending,
  }
}

export default function ChangeManager({ rigId, autoExecutionEnabled, onAutoExecutionToggle, children }: ChangeManagerProps) {
  const { changes, createChange, approveChange, rejectChange, applyChange, isApplying } = useChangeManager(rigId)

  const pendingChanges = changes.filter((c) => c.status === 'pending')
  const executedChanges = changes.filter((c) => c.status === 'executed').slice(-10)

  return (
    <div className="space-y-4">
      {/* Auto-Execution Control */}
      <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${autoExecutionEnabled ? 'bg-green-500' : 'bg-slate-200 dark:bg-slate-700'}`}>
              {autoExecutionEnabled ? <Bot className="w-5 h-5 text-white" /> : <User className="w-5 h-5 text-slate-600 dark:text-slate-300" />}
            </div>
            <div>
              <div className="font-semibold text-slate-900 dark:text-white">AI Auto-Execution</div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                {autoExecutionEnabled
                  ? 'AI can automatically apply approved changes'
                  : 'Manual approval required for all changes'}
              </div>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={autoExecutionEnabled}
              onChange={(e) => onAutoExecutionToggle(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-cyan-300 dark:peer-focus:ring-cyan-800 rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-slate-600 peer-checked:bg-cyan-500"></div>
          </label>
        </div>
      </div>

      {/* Pending Changes */}
      {pendingChanges.length > 0 && (
        <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-500" />
              <span className="font-semibold text-amber-900 dark:text-amber-200">
                {pendingChanges.length} Pending Change{pendingChanges.length > 1 ? 's' : ''}
              </span>
            </div>
          </div>
          <div className="space-y-2">
            {pendingChanges.map((change) => (
              <div key={change.id} className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-amber-200 dark:border-amber-700">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="font-semibold text-sm text-slate-900 dark:text-white">{change.component} - {change.parameter}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-300">
                      {change.currentValue} {change.unit} → {change.newValue} {change.unit}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{change.reason}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {change.requestedBy === 'ai' && (
                      <span className="px-2 py-1 rounded text-xs bg-cyan-500 text-white">AI</span>
                    )}
                    {change.requestedBy === 'user' && (
                      <span className="px-2 py-1 rounded text-xs bg-blue-500 text-white">Manual</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => approveChange(change.id)}
                    className="flex items-center gap-1 px-3 py-1.5 bg-green-500 hover:bg-green-400 text-white rounded text-xs font-semibold"
                  >
                    <CheckCircle2 className="w-3 h-3" />
                    Approve
                  </button>
                  <button
                    onClick={() => rejectChange(change.id)}
                    className="flex items-center gap-1 px-3 py-1.5 bg-red-500 hover:bg-red-400 text-white rounded text-xs font-semibold"
                  >
                    <X className="w-3 h-3" />
                    Reject
                  </button>
                  <button
                    onClick={() => applyChange(change.id)}
                    disabled={isApplying}
                    className="flex items-center gap-1 px-3 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-white rounded text-xs font-semibold disabled:opacity-50"
                  >
                    Apply Now
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Change History */}
      {executedChanges.length > 0 && (
        <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 p-4">
          <div className="flex items-center gap-2 mb-3">
            <History className="w-5 h-5 text-slate-500 dark:text-slate-400" />
            <span className="font-semibold text-slate-900 dark:text-white">Recent Changes</span>
          </div>
          <div className="space-y-2">
            {executedChanges.map((change) => (
              <div key={change.id} className="flex items-center justify-between p-2 bg-slate-50 dark:bg-slate-800 rounded-lg text-xs">
                <div className="flex items-center gap-2">
                  {change.executedBy === 'ai' ? (
                    <Bot className="w-4 h-4 text-cyan-500" />
                  ) : (
                    <User className="w-4 h-4 text-blue-500" />
                  )}
                  <span className="text-slate-700 dark:text-slate-300">
                    {change.component} - {change.parameter}: {change.currentValue} → {change.newValue} {change.unit}
                  </span>
                </div>
                <div className="text-slate-500 dark:text-slate-400">
                  {change.executedAt ? new Date(change.executedAt).toLocaleTimeString() : '--'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {children}
    </div>
  )
}

