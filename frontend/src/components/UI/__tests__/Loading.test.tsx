import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Loading from '../Loading'

describe('Loading', () => {
  it('renders loading spinner', () => {
    render(<Loading />)
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })

  it('renders with text', () => {
    render(<Loading text="Loading data..." />)
    expect(screen.getByText('Loading data...')).toBeInTheDocument()
  })

  it('renders with different sizes', () => {
    const { rerender } = render(<Loading size="sm" />)
    expect(screen.getByRole('status')).toBeInTheDocument()

    rerender(<Loading size="md" />)
    expect(screen.getByRole('status')).toBeInTheDocument()

    rerender(<Loading size="lg" />)
    expect(screen.getByRole('status')).toBeInTheDocument()
  })

  it('renders full screen', () => {
    render(<Loading fullScreen />)
    const container = screen.getByText('Loading...').closest('div')
    expect(container).toHaveClass('fixed', 'inset-0')
  })
})

describe('Loading.Skeleton', () => {
  it('renders skeleton', () => {
    render(<Loading.Skeleton />)
    expect(screen.getByText('Loading content...')).toBeInTheDocument()
  })

  it('renders with custom width and height', () => {
    render(<Loading.Skeleton width={200} height={100} />)
    const skeleton = screen.getByText('Loading content...')
    expect(skeleton).toHaveStyle({ width: '200px', height: '100px' })
  })

  it('renders with different variants', () => {
    const { rerender } = render(<Loading.Skeleton variant="text" />)
    expect(screen.getByText('Loading content...')).toHaveClass('h-4')

    rerender(<Loading.Skeleton variant="circular" />)
    expect(screen.getByText('Loading content...')).toHaveClass('rounded-full')

    rerender(<Loading.Skeleton variant="rectangular" />)
    expect(screen.getByText('Loading content...')).toHaveClass('rounded-lg')
  })
})

describe('Loading.SkeletonText', () => {
  it('renders skeleton text with default lines', () => {
    render(<Loading.SkeletonText />)
    const skeletons = screen.getAllByText('Loading content...')
    expect(skeletons).toHaveLength(3)
  })

  it('renders skeleton text with custom lines', () => {
    render(<Loading.SkeletonText lines={5} />)
    const skeletons = screen.getAllByText('Loading content...')
    expect(skeletons).toHaveLength(5)
  })
})

