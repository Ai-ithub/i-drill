import { render, screen } from '@testing-library/react'
import App from '@/App'

describe('App smoke test', () => {
  it('renders navigation menu', () => {
    render(<App />)
    expect(screen.getByText(/مانیتورینگ لحظه‌ای/i)).toBeInTheDocument()
  })
})
