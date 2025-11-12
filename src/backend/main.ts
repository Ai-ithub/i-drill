/**
 * Main entry point - Refactored structure
 * This file demonstrates improved code organization
 */

// Centralized configuration
export const config = {
  api: {
    baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001',
    timeout: 30000,
  },
  features: {
    enableWebSocket: true,
    enableCaching: true,
    enableAnalytics: false,
  },
};

// Error handling utility
export function handleError(error: unknown, context?: string): void {
  const message = error instanceof Error ? error.message : 'Unknown error';
  console.error(`[${context || 'App'}] Error:`, message);
  
  // Could integrate with error tracking service
  // errorTracker.captureException(error, { context });
}

// Performance monitoring
export function measurePerformance(name: string, fn: () => void): void {
  if (import.meta.env.DEV) {
    const start = performance.now();
    fn();
    const end = performance.now();
    console.log(`[Performance] ${name}: ${(end - start).toFixed(2)}ms`);
  } else {
    fn();
  }
}

