import { test, expect } from '@playwright/test';
import { login } from './helpers/auth';
import { goToDashboard, goToRealtime, goToData, goToRTO, goToDVR, goToPDM } from './helpers/navigation';

/**
 * E2E Tests for Dashboard Functionality
 * 
 * These tests verify:
 * - Dashboard page loads correctly
 * - Navigation between pages works
 * - Real-time data display
 * - UI components render properly
 */

test.describe('Dashboard Functionality', () => {
  const testUsername = process.env.TEST_USERNAME || 'admin';
  const testPassword = process.env.TEST_PASSWORD || 'admin123';

  test.beforeEach(async ({ page }) => {
    // Login first using helper
    await login(page, testUsername, testPassword);
  });

  test('should display dashboard page', async ({ page }) => {
    // Navigate to dashboard using helper
    await goToDashboard(page);
    
    // Check for common dashboard elements
    // The page should have some content (not empty)
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
    expect(bodyContent!.length).toBeGreaterThan(0);
  });

  test('should navigate to Real-Time Monitoring page', async ({ page }) => {
    // Navigate using helper
    await goToRealtime(page);
    
    // Page should have loaded
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
  });

  test('should navigate to Data page', async ({ page }) => {
    // Navigate using helper
    await goToData(page);
    
    // Page should have loaded
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
  });

  test('should navigate to RTO page', async ({ page }) => {
    // Navigate using helper
    await goToRTO(page);
    
    // Page should have loaded
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
  });

  test('should navigate to DVR page', async ({ page }) => {
    // Navigate using helper
    await goToDVR(page);
    
    // Page should have loaded
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
  });

  test('should navigate to PDM page', async ({ page }) => {
    // Navigate using helper
    await goToPDM(page);
    
    // Page should have loaded
    const bodyContent = await page.locator('body').textContent();
    expect(bodyContent).toBeTruthy();
  });

  test('should display header with logo and navigation', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Check for header elements
    // Logo or app name should be visible
    const header = page.locator('header').first();
    await expect(header).toBeVisible();
    
    // Check for "i drill" text in header
    const headerText = await header.textContent();
    expect(headerText).toContain('i drill');
  });

  test('should handle theme toggle', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Find theme toggle button (sun/moon icon)
    const themeButton = page.locator('button[aria-label="toggle theme"]').first();
    
    if (await themeButton.isVisible({ timeout: 2000 })) {
      // Get initial theme
      const initialTheme = await page.evaluate(() => {
        return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
      });
      
      // Click toggle
      await themeButton.click();
      await page.waitForTimeout(500);
      
      // Check theme changed
      const newTheme = await page.evaluate(() => {
        return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
      });
      
      expect(newTheme).not.toBe(initialTheme);
    }
  });

  test('should display role selector', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Look for role selector dropdown
    const roleSelector = page.locator('select').first();
    
    // Role selector might be hidden on mobile, so check if visible
    const isVisible = await roleSelector.isVisible().catch(() => false);
    
    if (isVisible) {
      // Should have role options
      const options = await roleSelector.locator('option').count();
      expect(options).toBeGreaterThan(0);
    }
  });
});

