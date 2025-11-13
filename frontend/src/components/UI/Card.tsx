import { ReactNode, HTMLAttributes } from 'react'
import { cn } from '@/utils/cn'

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'elevated' | 'outlined'
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

export interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
  title?: string
  subtitle?: string
  action?: ReactNode
}

export interface CardContentProps extends HTMLAttributes<HTMLDivElement> {}

export interface CardFooterProps extends HTMLAttributes<HTMLDivElement> {}

const Card = ({ className, variant = 'default', padding = 'md', children, ...props }: CardProps) => {
  const baseStyles = 'rounded-2xl border transition-colors duration-200'
  
  const variants = {
    default: 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 shadow-sm',
    elevated: 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 shadow-lg',
    outlined: 'bg-transparent border-2 border-slate-300 dark:border-slate-600',
  }

  const paddings = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  }

  return (
    <div className={cn(baseStyles, variants[variant], paddings[padding], className)} {...props}>
      {children}
    </div>
  )
}

const CardHeader = ({ className, title, subtitle, action, children, ...props }: CardHeaderProps) => {
  return (
    <div className={cn('flex items-start justify-between mb-4', className)} {...props}>
      {(title || subtitle) && (
        <div className="flex-1">
          {title && <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-1">{title}</h3>}
          {subtitle && <p className="text-sm text-slate-500 dark:text-slate-400">{subtitle}</p>}
        </div>
      )}
      {action && <div className="ml-4">{action}</div>}
      {children}
    </div>
  )
}

const CardContent = ({ className, children, ...props }: CardContentProps) => {
  return (
    <div className={cn('', className)} {...props}>
      {children}
    </div>
  )
}

const CardFooter = ({ className, children, ...props }: CardFooterProps) => {
  return (
    <div className={cn('mt-4 pt-4 border-t border-slate-200 dark:border-slate-700', className)} {...props}>
      {children}
    </div>
  )
}

Card.Header = CardHeader
Card.Content = CardContent
Card.Footer = CardFooter

export default Card

