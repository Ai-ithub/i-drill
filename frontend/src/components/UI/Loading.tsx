import { ReactNode } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/utils/cn'

export interface LoadingProps {
  size?: 'sm' | 'md' | 'lg'
  text?: string
  fullScreen?: boolean
  className?: string
}

export interface SkeletonProps {
  className?: string
  variant?: 'text' | 'circular' | 'rectangular'
  width?: string | number
  height?: string | number
  animation?: 'pulse' | 'wave' | 'none'
}

export interface SkeletonTextProps {
  lines?: number
  className?: string
}

const Loading = ({ size = 'md', text, fullScreen = false, className }: LoadingProps) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  }

  const content = (
    <div className={cn('flex flex-col items-center justify-center gap-3', className)}>
      <Loader2 className={cn('animate-spin text-cyan-600 dark:text-cyan-400', sizes[size])} aria-hidden="true" />
      {text && <p className="text-sm text-slate-600 dark:text-slate-400">{text}</p>}
      <span className="sr-only">Loading...</span>
    </div>
  )

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 dark:bg-slate-950/80 backdrop-blur-sm">
        {content}
      </div>
    )
  }

  return content
}

const Skeleton = ({
  className,
  variant = 'rectangular',
  width,
  height,
  animation = 'pulse',
}: SkeletonProps) => {
  const baseStyles = 'bg-slate-200 dark:bg-slate-700 rounded'
  
  const variants = {
    text: 'h-4',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  }

  const animations = {
    pulse: 'animate-pulse',
    wave: 'animate-[wave_1.6s_ease-in-out_infinite]',
    none: '',
  }

  const style: React.CSSProperties = {}
  if (width) style.width = typeof width === 'number' ? `${width}px` : width
  if (height) style.height = typeof height === 'number' ? `${height}px` : height

  return (
    <div
      className={cn(baseStyles, variants[variant], animations[animation], className)}
      style={style}
      aria-hidden="true"
    >
      <span className="sr-only">Loading content...</span>
    </div>
  )
}

const SkeletonText = ({ lines = 3, className }: SkeletonTextProps) => {
  return (
    <div className={cn('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          variant="text"
          width={i === lines - 1 ? '75%' : '100%'}
          className="h-4"
        />
      ))}
    </div>
  )
}

Loading.Skeleton = Skeleton
Loading.SkeletonText = SkeletonText

export default Loading

