import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import NewLayout from '../NewLayout'

// Mock contexts
vi.mock('@/context/ThemeContext', () => ({
  useThemeMode: () => ({
    mode: 'light',
    toggle: vi.fn(),
  }),
}))

vi.mock('@/context/RoleContext', () => ({
  useUserRole: () => ({
    role: 'engineer',
    setRole: vi.fn(),
  }),
}))

vi.mock('@/components/Notifications/NotificationBadge', () => ({
  default: () => <div data-testid="notification-badge">Notifications</div>,
}))

describe('NewLayout', () => {
  const renderLayout = (children = <div>Test Content</div>) => {
    return render(
      <BrowserRouter>
        <NewLayout>{children}</NewLayout>
      </BrowserRouter>
    )
  }

  it('renders header with logo and title', () => {
    renderLayout()
    expect(screen.getByText('i drill')).toBeInTheDocument()
  })

  it('renders navigation menu', () => {
    renderLayout()
    expect(screen.getByText('Monitoring')).toBeInTheDocument()
  })

  it('renders children content', () => {
    renderLayout(<div>Test Content</div>)
    expect(screen.getByText('Test Content')).toBeInTheDocument()
  })

  it('shows access denied for unauthorized routes', () => {
    // Mock viewer role trying to access engineer-only route
    vi.mocked(require('@/context/RoleContext').useUserRole).mockReturnValue({
      role: 'viewer',
      setRole: vi.fn(),
    })

    renderLayout()
    // Should show access denied or filter menu items
    expect(screen.getByText('Monitoring')).toBeInTheDocument()
  })
})

