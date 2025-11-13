import { ReactNode, useEffect, useState } from 'react'
import { X, CheckCircle2, AlertCircle, Info, AlertTriangle } from 'lucide-react'
import { createPortal } from 'react-dom'
import { cn } from '@/utils/cn'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface Toast {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number
  action?: {
    label: string
    onClick: () => void
  }
}

interface ToastProps {
  toast: Toast
  onClose: (id: string) => void
}

const ToastComponent = ({ toast, onClose }: ToastProps) => {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    setIsVisible(true)
    const timer = setTimeout(() => {
      setIsVisible(false)
      setTimeout(() => onClose(toast.id), 300)
    }, toast.duration || 5000)

    return () => clearTimeout(timer)
  }, [toast.id, toast.duration, onClose])

  const icons = {
    success: CheckCircle2,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
  }

  const styles = {
    success: 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200',
    error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200',
    warning: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800 text-amber-800 dark:text-amber-200',
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-200',
  }

  const Icon = icons[toast.type]

  return (
    <div
      className={cn(
        'flex items-start gap-3 p-4 rounded-lg border shadow-lg min-w-[300px] max-w-[500px] transition-all duration-300 transform',
        styles[toast.type],
        isVisible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'
      )}
      role="alert"
      aria-live="polite"
    >
      <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" aria-hidden="true" />
      <div className="flex-1 min-w-0">
        <div className="font-semibold text-sm">{toast.title}</div>
        {toast.message && <div className="text-sm mt-1 opacity-90">{toast.message}</div>}
        {toast.action && (
          <button
            onClick={toast.action.onClick}
            className="mt-2 text-sm font-medium underline hover:no-underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 rounded"
          >
            {toast.action.label}
          </button>
        )}
      </div>
      <button
        onClick={() => {
          setIsVisible(false)
          setTimeout(() => onClose(toast.id), 300)
        }}
        className="flex-shrink-0 p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
        aria-label="Close notification"
      >
        <X className="w-4 h-4" aria-hidden="true" />
      </button>
    </div>
  )
}

class ToastManager {
  private toasts: Toast[] = []
  private listeners: Set<(toasts: Toast[]) => void> = new Set()

  subscribe(listener: (toasts: Toast[]) => void) {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private notify() {
    this.listeners.forEach((listener) => listener([...this.toasts]))
  }

  show(toast: Omit<Toast, 'id'>) {
    const id = `toast-${Date.now()}-${Math.random()}`
    const newToast = { ...toast, id }
    this.toasts.push(newToast)
    this.notify()
    return id
  }

  success(title: string, message?: string, duration?: number) {
    return this.show({ type: 'success', title, message, duration })
  }

  error(title: string, message?: string, duration?: number) {
    return this.show({ type: 'error', title, message, duration: duration || 7000 })
  }

  warning(title: string, message?: string, duration?: number) {
    return this.show({ type: 'warning', title, message, duration })
  }

  info(title: string, message?: string, duration?: number) {
    return this.show({ type: 'info', title, message, duration })
  }

  close(id: string) {
    this.toasts = this.toasts.filter((toast) => toast.id !== id)
    this.notify()
  }

  closeAll() {
    this.toasts = []
    this.notify()
  }

  getToasts() {
    return [...this.toasts]
  }
}

export const toastManager = new ToastManager()

export const ToastContainer = () => {
  const [toasts, setToasts] = useState<Toast[]>([])

  useEffect(() => {
    return toastManager.subscribe(setToasts)
  }, [])

  if (toasts.length === 0) return null

  const container = document.getElementById('toast-container') || document.body

  return createPortal(
    <div
      className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none"
      aria-live="polite"
      aria-atomic="true"
    >
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastComponent toast={toast} onClose={(id) => toastManager.close(id)} />
        </div>
      ))}
    </div>,
    container
  )
}

// Export convenience functions
export const toast = {
  success: (title: string, message?: string, duration?: number) => toastManager.success(title, message, duration),
  error: (title: string, message?: string, duration?: number) => toastManager.error(title, message, duration),
  warning: (title: string, message?: string, duration?: number) => toastManager.warning(title, message, duration),
  info: (title: string, message?: string, duration?: number) => toastManager.info(title, message, duration),
  close: (id: string) => toastManager.close(id),
  closeAll: () => toastManager.closeAll(),
}

