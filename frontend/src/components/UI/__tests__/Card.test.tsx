import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Card from '../Card'

describe('Card', () => {
  it('renders with children', () => {
    render(<Card>Card content</Card>)
    expect(screen.getByText('Card content')).toBeInTheDocument()
  })

  it('renders with different variants', () => {
    const { rerender } = render(<Card variant="default">Default</Card>)
    expect(screen.getByText('Default').parentElement).toHaveClass('bg-white')

    rerender(<Card variant="elevated">Elevated</Card>)
    expect(screen.getByText('Elevated').parentElement).toHaveClass('shadow-lg')

    rerender(<Card variant="outlined">Outlined</Card>)
    expect(screen.getByText('Outlined').parentElement).toHaveClass('bg-transparent')
  })

  it('renders with different padding', () => {
    const { rerender } = render(<Card padding="none">No padding</Card>)
    expect(screen.getByText('No padding').parentElement).not.toHaveClass('p-')

    rerender(<Card padding="sm">Small padding</Card>)
    expect(screen.getByText('Small padding').parentElement).toHaveClass('p-4')

    rerender(<Card padding="md">Medium padding</Card>)
    expect(screen.getByText('Medium padding').parentElement).toHaveClass('p-6')

    rerender(<Card padding="lg">Large padding</Card>)
    expect(screen.getByText('Large padding').parentElement).toHaveClass('p-8')
  })

  it('renders Card.Header with title', () => {
    render(
      <Card>
        <Card.Header title="Card Title" />
        <Card.Content>Content</Card.Content>
      </Card>
    )
    expect(screen.getByText('Card Title')).toBeInTheDocument()
  })

  it('renders Card.Header with subtitle', () => {
    render(
      <Card>
        <Card.Header title="Title" subtitle="Subtitle" />
      </Card>
    )
    expect(screen.getByText('Subtitle')).toBeInTheDocument()
  })

  it('renders Card.Content', () => {
    render(
      <Card>
        <Card.Content>Content here</Card.Content>
      </Card>
    )
    expect(screen.getByText('Content here')).toBeInTheDocument()
  })

  it('renders Card.Footer', () => {
    render(
      <Card>
        <Card.Footer>Footer content</Card.Footer>
      </Card>
    )
    expect(screen.getByText('Footer content')).toBeInTheDocument()
  })
})

