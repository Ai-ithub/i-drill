import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { toast, toastManager } from '../Toast'

describe('Toast Manager', () => {
  beforeEach(() => {
    toastManager.closeAll()
  })

  afterEach(() => {
    toastManager.closeAll()
  })

  it('shows success toast', () => {
    const id = toast.success('Success!', 'Operation completed')
    expect(id).toBeDefined()
    const toasts = toastManager.getToasts()
    expect(toasts).toHaveLength(1)
    expect(toasts[0].type).toBe('success')
    expect(toasts[0].title).toBe('Success!')
  })

  it('shows error toast', () => {
    const id = toast.error('Error!', 'Something went wrong')
    expect(id).toBeDefined()
    const toasts = toastManager.getToasts()
    expect(toasts[0].type).toBe('error')
  })

  it('shows warning toast', () => {
    const id = toast.warning('Warning!', 'Please check')
    expect(id).toBeDefined()
    const toasts = toastManager.getToasts()
    expect(toasts[0].type).toBe('warning')
  })

  it('shows info toast', () => {
    const id = toast.info('Info', 'New data available')
    expect(id).toBeDefined()
    const toasts = toastManager.getToasts()
    expect(toasts[0].type).toBe('info')
  })

  it('closes toast by id', () => {
    const id = toast.success('Test')
    expect(toastManager.getToasts()).toHaveLength(1)
    toast.close(id)
    expect(toastManager.getToasts()).toHaveLength(0)
  })

  it('closes all toasts', () => {
    toast.success('Test 1')
    toast.error('Test 2')
    expect(toastManager.getToasts()).toHaveLength(2)
    toast.closeAll()
    expect(toastManager.getToasts()).toHaveLength(0)
  })

  it('auto-dismisses after duration', async () => {
    vi.useFakeTimers()
    toast.success('Test', 'Message', 1000)
    expect(toastManager.getToasts()).toHaveLength(1)
    
    vi.advanceTimersByTime(1000)
    await vi.runAllTimersAsync()
    
    expect(toastManager.getToasts()).toHaveLength(0)
    vi.useRealTimers()
  })
})

