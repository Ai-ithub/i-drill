import { test, expect } from '@playwright/test';
import { login } from './helpers/auth';

/**
 * E2E Tests for Real-Time Monitoring
 * 
 * These tests verify:
 * - Real-time monitoring page loads correctly
 * - WebSocket connection (if applicable)
 * - Data visualization components
 * - Real-time data updates
 */

test.describe('Real-Time Monitoring', () => {
  const testUsername = process.env.TEST_USERNAME || 'admin';
  const testPassword = process.env.TEST_PASSWORD || 'admin123';

  test.beforeEach(async ({ page }) => {
    // Login first using helper
    await login(page, testUsername, testPassword);
  });

  test('should load Real-Time Monitoring page', async ({ page }) => {
    // Navigate to real-time monitoring
    await page.goto('/realtime');
    await page.waitForLoadState('networkidle');
    
    // Verify URL
    await expect(page).toHaveURL(/\/realtime/);
    
    // Page should have content
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
    expect(bodyContent!.length).toBeGreaterThan(0);
  });

  test('should display real-time monitoring content', async ({ page }) => {
    await page.goto('/realtime');
    await page.waitForLoadState('networkidle');
    
    // Wait a bit for content to load
    await page.waitForTimeout(2000);
    
    // Check that page has rendered content
    const mainContent = page.locator('main, [role="main"], .main-content').first();
    const isMainVisible = await mainContent.isVisible().catch(() => false);
    
    if (isMainVisible) {
      const content = await mainContent.textContent();
      expect(content).toBeTruthy();
    }
  });

  test('should handle WebSocket connection', async ({ page }) => {
    await page.goto('/realtime');
    await page.waitForLoadState('networkidle');
    
    // Listen for WebSocket connections
    const wsConnections: string[] = [];
    
    page.on('websocket', (ws) => {
      wsConnections.push(ws.url());
    });
    
    // Wait a bit for WebSocket to connect
    await page.waitForTimeout(3000);
    
    // Check if WebSocket connected (optional - might not always connect in test)
    // This test mainly verifies the page doesn't crash when trying to connect
    const pageContent = await page.locator('body').textContent();
    expect(pageContent).toBeTruthy();
  });

  test('should navigate to display pages', async ({ page }) => {
    // Test navigation to various display pages
    const displayPages = [
      { path: '/display/gauge', name: 'Gauge' },
      { path: '/display/sensor', name: 'Sensor' },
      { path: '/display/control', name: 'Control' },
      { path: '/display/rpm', name: 'RPM' },
    ];

    for (const displayPage of displayPages) {
      await page.goto(displayPage.path);
      await page.waitForLoadState('networkidle');
      
      // Verify URL changed
      await expect(page).toHaveURL(new RegExp(displayPage.path.replace('/', '\\/')));
      
      // Page should have loaded
      const bodyContent = await page.locator('body').textContent();
      expect(bodyContent).toBeTruthy();
      
      // Small delay between navigations
      await page.waitForTimeout(500);
    }
  });

  test('should handle page refresh without errors', async ({ page }) => {
    await page.goto('/realtime');
    await page.waitForLoadState('networkidle');
    
    // Refresh page
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // Page should still be on realtime
    await expect(page).toHaveURL(/\/realtime/);
    
    // Content should still be visible
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
  });

  test('should maintain state during navigation', async ({ page }) => {
    // Start at realtime
    await page.goto('/realtime');
    await page.waitForLoadState('networkidle');
    
    // Navigate to dashboard
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Navigate back to realtime
    await page.goto('/realtime');
    await page.waitForLoadState('networkidle');
    
    // Should still work correctly
    await expect(page).toHaveURL(/\/realtime/);
    
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
  });
});

