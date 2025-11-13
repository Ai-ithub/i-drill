import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ErrorDisplay from '../ErrorDisplay'

describe('ErrorDisplay', () => {
  it('renders with default title', () => {
    render(<ErrorDisplay />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
  })

  it('renders with custom title', () => {
    render(<ErrorDisplay title="Custom Error" />)
    expect(screen.getByText('Custom Error')).toBeInTheDocument()
  })

  it('renders with error message', () => {
    render(<ErrorDisplay message="Test error message" />)
    expect(screen.getByText('Test error message')).toBeInTheDocument()
  })

  it('renders with Error object', () => {
    const error = new Error('Test error')
    render(<ErrorDisplay error={error} />)
    expect(screen.getByText('Test error')).toBeInTheDocument()
  })

  it('renders with retry button', () => {
    const handleRetry = vi.fn()
    render(<ErrorDisplay onRetry={handleRetry} />)
    const button = screen.getByText('Try Again')
    expect(button).toBeInTheDocument()
    fireEvent.click(button)
    expect(handleRetry).toHaveBeenCalledTimes(1)
  })

  it('renders with go home button', () => {
    const handleGoHome = vi.fn()
    render(<ErrorDisplay onGoHome={handleGoHome} />)
    const button = screen.getByText('Go Home')
    expect(button).toBeInTheDocument()
    fireEvent.click(button)
    expect(handleGoHome).toHaveBeenCalledTimes(1)
  })

  it('renders with different variants', () => {
    const { rerender } = render(<ErrorDisplay variant="page" />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()

    rerender(<ErrorDisplay variant="inline" />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()

    rerender(<ErrorDisplay variant="card" />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
  })
})

