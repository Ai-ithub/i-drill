import { ReactNode } from 'react'
import { Inbox, Search, AlertCircle, Database } from 'lucide-react'
import { cn } from '@/utils/cn'
import Button from './Button'

export interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: {
    label: string
    onClick: () => void
  }
  variant?: 'default' | 'search' | 'error' | 'data'
  className?: string
}

const EmptyState = ({
  icon,
  title,
  description,
  action,
  variant = 'default',
  className,
}: EmptyStateProps) => {
  const defaultIcons = {
    default: Inbox,
    search: Search,
    error: AlertCircle,
    data: Database,
  }

  const DefaultIcon = defaultIcons[variant]

  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center py-12 px-4 text-center',
        className
      )}
    >
      <div className="mb-4 text-slate-400 dark:text-slate-500">
        {icon || <DefaultIcon className="w-16 h-16" aria-hidden="true" />}
      </div>
      <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">{title}</h3>
      {description && (
        <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md mb-6">{description}</p>
      )}
      {action && (
        <Button onClick={action.onClick} variant="primary">
          {action.label}
        </Button>
      )}
    </div>
  )
}

export default EmptyState

