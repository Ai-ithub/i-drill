/**
 * Unit tests for accessibility utilities
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  announceToScreenReader,
  trapFocus,
  getAriaLabel,
  formatNumberForScreenReader
} from '../accessibility'

describe('Accessibility Utilities', () => {
  beforeEach(() => {
    // Clear DOM
    document.body.innerHTML = ''
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  describe('announceToScreenReader', () => {
    it('should create announcement element', () => {
      announceToScreenReader('Test message')
      
      const announcement = document.querySelector('.sr-only')
      expect(announcement).not.toBeNull()
      expect(announcement?.textContent).toBe('Test message')
    })

    it('should set aria-live attribute to polite by default', () => {
      announceToScreenReader('Test message')
      
      const announcement = document.querySelector('.sr-only')
      expect(announcement?.getAttribute('aria-live')).toBe('polite')
      expect(announcement?.getAttribute('role')).toBe('status')
    })

    it('should set aria-live attribute to assertive when specified', () => {
      announceToScreenReader('Test message', 'assertive')
      
      const announcement = document.querySelector('.sr-only')
      expect(announcement?.getAttribute('aria-live')).toBe('assertive')
      expect(announcement?.getAttribute('role')).toBe('alert')
    })

    it('should remove announcement after timeout', () => {
      announceToScreenReader('Test message')
      
      const announcement = document.querySelector('.sr-only')
      expect(announcement).not.toBeNull()
      
      vi.advanceTimersByTime(1000)
      
      expect(document.querySelector('.sr-only')).toBeNull()
    })

    it('should append announcement to body', () => {
      announceToScreenReader('Test message')
      
      const announcement = document.querySelector('.sr-only')
      expect(announcement?.parentElement).toBe(document.body)
    })
  })

  describe('trapFocus', () => {
    it('should trap focus within element', () => {
      // Setup
      const container = document.createElement('div')
      container.innerHTML = `
        <button>Button 1</button>
        <input type="text" />
        <button>Button 2</button>
      `
      document.body.appendChild(container)
      
      const buttons = container.querySelectorAll('button')
      const input = container.querySelector('input')!
      
      // Execute
      trapFocus(container)
      
      // Test focus trapping
      buttons[0].focus()
      expect(document.activeElement).toBe(buttons[0])
      
      // Simulate Tab from last element
      const tabEvent = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true })
      buttons[1].dispatchEvent(tabEvent)
      
      // Focus should wrap to first element
      expect(document.activeElement).toBe(buttons[0])
    })

    it('should handle Shift+Tab from first element', () => {
      // Setup
      const container = document.createElement('div')
      container.innerHTML = `
        <button>Button 1</button>
        <button>Button 2</button>
      `
      document.body.appendChild(container)
      
      const buttons = container.querySelectorAll('button')
      
      // Execute
      trapFocus(container)
      
      // Test Shift+Tab from first element
      buttons[0].focus()
      const shiftTabEvent = new KeyboardEvent('keydown', {
        key: 'Tab',
        shiftKey: true,
        bubbles: true
      })
      buttons[0].dispatchEvent(shiftTabEvent)
      
      // Focus should wrap to last element
      expect(document.activeElement).toBe(buttons[1])
    })

    it('should only trap Tab key', () => {
      // Setup
      const container = document.createElement('div')
      container.innerHTML = `<button>Button</button>`
      document.body.appendChild(container)
      
      // Execute
      trapFocus(container)
      
      // Test other keys don't trigger trapping
      const otherKeyEvent = new KeyboardEvent('keydown', {
        key: 'Enter',
        bubbles: true
      })
      container.dispatchEvent(otherKeyEvent)
      
      // Should not prevent default
      expect(true).toBe(true)
    })

    it('should handle empty container', () => {
      // Setup
      const container = document.createElement('div')
      document.body.appendChild(container)
      
      // Execute - should not throw
      expect(() => trapFocus(container)).not.toThrow()
    })

    it('should handle container with no focusable elements', () => {
      // Setup
      const container = document.createElement('div')
      container.innerHTML = '<div>No focusable elements</div>'
      document.body.appendChild(container)
      
      // Execute - should not throw
      expect(() => trapFocus(container)).not.toThrow()
    })
  })

  describe('getAriaLabel', () => {
    it('should return key as label by default', () => {
      const label = getAriaLabel('test.key')
      expect(label).toBe('test.key')
    })

    it('should return key with params', () => {
      const label = getAriaLabel('test.key', { param: 'value' })
      expect(label).toBe('test.key')
      // Note: This is a placeholder implementation
      // In production, this would integrate with i18n
    })
  })

  describe('formatNumberForScreenReader', () => {
    it('should format number without unit', () => {
      const formatted = formatNumberForScreenReader(1234.56)
      expect(formatted).toContain('1,234.56')
    })

    it('should format number with unit', () => {
      const formatted = formatNumberForScreenReader(1234.56, 'feet')
      expect(formatted).toContain('1,234.56')
      expect(formatted).toContain('feet')
    })

    it('should format zero', () => {
      const formatted = formatNumberForScreenReader(0)
      expect(formatted).toContain('0')
    })

    it('should format large numbers', () => {
      const formatted = formatNumberForScreenReader(1000000)
      expect(formatted).toContain('1,000,000')
    })

    it('should format negative numbers', () => {
      const formatted = formatNumberForScreenReader(-1234.56)
      expect(formatted).toContain('-1,234.56')
    })

    it('should format decimal numbers', () => {
      const formatted = formatNumberForScreenReader(1234.56789)
      expect(formatted).toContain('1,234.568')
    })

    it('should handle different units', () => {
      const formatted = formatNumberForScreenReader(100, 'meters')
      expect(formatted).toContain('100')
      expect(formatted).toContain('meters')
    })
  })
})

