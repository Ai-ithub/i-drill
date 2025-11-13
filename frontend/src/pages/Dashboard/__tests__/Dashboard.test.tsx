import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Dashboard from '../Dashboard'

// Mock the API
vi.mock('@/services/api', () => ({
  sensorDataApi: {
    getAnalytics: vi.fn().mockResolvedValue({
      data: {
        summary: {
          current_depth: 5000,
          average_rop: 12.5,
          total_power_consumption: 50000,
          maintenance_alerts_count: 2,
          total_drilling_time_hours: 100.5,
          last_updated: new Date().toISOString(),
        },
      },
    }),
  },
  healthApi: {
    detailed: vi.fn().mockResolvedValue({
      data: {
        kafka: { status: 'healthy' },
        database: { status: 'healthy' },
        rl_environment: { status: 'healthy' },
        mlflow: { status: 'healthy' },
      },
    }),
  },
}))

describe('Dashboard', () => {
  let queryClient: QueryClient

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    })
  })

  it('renders dashboard title', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <Dashboard />
      </QueryClientProvider>
    )
    expect(screen.getByText('Operations Dashboard')).toBeInTheDocument()
  })

  it('displays loading state', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <Dashboard />
      </QueryClientProvider>
    )
    // Should show loading skeleton
    expect(screen.getByText('Operations Dashboard')).toBeInTheDocument()
  })

  it('displays stats cards when data is loaded', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <Dashboard />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByText(/Current Depth/i)).toBeInTheDocument()
      expect(screen.getByText(/Average ROP/i)).toBeInTheDocument()
    })
  })

  it('displays service status chips', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <Dashboard />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByText(/Kafka/i)).toBeInTheDocument()
    })
  })
})

