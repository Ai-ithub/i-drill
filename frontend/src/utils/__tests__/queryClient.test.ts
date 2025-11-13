/**
 * Unit tests for queryClient
 */
import { describe, it, expect } from 'vitest'
import { queryClient } from '../queryClient'

describe('queryClient', () => {
  it('should create QueryClient instance', () => {
    expect(queryClient).toBeDefined()
  })

  it('should have default options configured', () => {
    const defaultOptions = queryClient.getDefaultOptions()
    expect(defaultOptions.queries).toBeDefined()
  })

  it('should have refetchOnWindowFocus disabled', () => {
    const defaultOptions = queryClient.getDefaultOptions()
    expect(defaultOptions.queries?.refetchOnWindowFocus).toBe(false)
  })

  it('should have retry set to 1', () => {
    const defaultOptions = queryClient.getDefaultOptions()
    expect(defaultOptions.queries?.retry).toBe(1)
  })

  it('should have staleTime set to 30000', () => {
    const defaultOptions = queryClient.getDefaultOptions()
    expect(defaultOptions.queries?.staleTime).toBe(30000)
  })
})

