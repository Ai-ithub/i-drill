import { render, screen } from '@testing-library/react'
import App from '@/App'
import { ThemeProvider } from '@/context/ThemeContext'
import { RoleProvider } from '@/context/RoleContext'

describe('App smoke test', () => {
  it('renders navigation menu', () => {
    render(
      <ThemeProvider>
        <RoleProvider>
          <App />
        </RoleProvider>
      </ThemeProvider>,
    )
    expect(screen.getByText(/مانیتورینگ لحظه‌ای/i)).toBeInTheDocument()
  })
})
