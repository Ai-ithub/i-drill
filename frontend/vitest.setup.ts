import '@testing-library/jest-dom'

class ResizeObserverPolyfill {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (typeof window !== 'undefined' && !('ResizeObserver' in window)) {
  // @ts-ignore
  window.ResizeObserver = ResizeObserverPolyfill as unknown as typeof ResizeObserver
}
if (typeof globalThis !== 'undefined' && !('ResizeObserver' in globalThis)) {
  // @ts-ignore
  globalThis.ResizeObserver = ResizeObserverPolyfill as unknown as typeof ResizeObserver
}
