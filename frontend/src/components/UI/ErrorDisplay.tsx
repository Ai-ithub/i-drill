import { ReactNode } from 'react'
import { AlertCircle, RefreshCw, Home } from 'lucide-react'
import { cn } from '@/utils/cn'
import Button from './Button'
import Card from './Card'

export interface ErrorDisplayProps {
  title?: string
  message?: string
  error?: Error | string
  onRetry?: () => void
  onGoHome?: () => void
  variant?: 'page' | 'inline' | 'card'
  className?: string
}

const ErrorDisplay = ({
  title = 'Something went wrong',
  message,
  error,
  onRetry,
  onGoHome,
  variant = 'page',
  className,
}: ErrorDisplayProps) => {
  const errorMessage = error instanceof Error ? error.message : error || message

  const content = (
    <div className={cn('flex flex-col items-center justify-center text-center', className)}>
      <div className="mb-4 text-red-500 dark:text-red-400">
        <AlertCircle className="w-12 h-12" aria-hidden="true" />
      </div>
      <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">{title}</h3>
      {errorMessage && (
        <p className="text-sm text-slate-600 dark:text-slate-400 max-w-md mb-6">{errorMessage}</p>
      )}
      <div className="flex gap-3">
        {onRetry && (
          <Button onClick={onRetry} variant="primary" leftIcon={<RefreshCw className="w-4 h-4" />}>
            Try Again
          </Button>
        )}
        {onGoHome && (
          <Button onClick={onGoHome} variant="outline" leftIcon={<Home className="w-4 h-4" />}>
            Go Home
          </Button>
        )}
      </div>
    </div>
  )

  if (variant === 'card') {
    return (
      <Card variant="outlined" className="border-red-200 dark:border-red-800">
        {content}
      </Card>
    )
  }

  if (variant === 'inline') {
    return <div className={cn('py-8', className)}>{content}</div>
  }

  return (
    <div className="min-h-[60vh] flex items-center justify-center p-4">
      {content}
    </div>
  )
}

export default ErrorDisplay

