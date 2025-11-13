import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import EmptyState from '../EmptyState'

describe('EmptyState', () => {
  it('renders with title', () => {
    render(<EmptyState title="No data" />)
    expect(screen.getByText('No data')).toBeInTheDocument()
  })

  it('renders with description', () => {
    render(<EmptyState title="No data" description="There is no data to display" />)
    expect(screen.getByText('There is no data to display')).toBeInTheDocument()
  })

  it('renders with action button', () => {
    const handleAction = vi.fn()
    render(
      <EmptyState
        title="No data"
        action={{ label: 'Refresh', onClick: handleAction }}
      />
    )
    const button = screen.getByText('Refresh')
    expect(button).toBeInTheDocument()
    fireEvent.click(button)
    expect(handleAction).toHaveBeenCalledTimes(1)
  })

  it('renders with different variants', () => {
    const { rerender } = render(<EmptyState title="Test" variant="default" />)
    expect(screen.getByText('Test')).toBeInTheDocument()

    rerender(<EmptyState title="Test" variant="search" />)
    expect(screen.getByText('Test')).toBeInTheDocument()

    rerender(<EmptyState title="Test" variant="error" />)
    expect(screen.getByText('Test')).toBeInTheDocument()

    rerender(<EmptyState title="Test" variant="data" />)
    expect(screen.getByText('Test')).toBeInTheDocument()
  })
})

