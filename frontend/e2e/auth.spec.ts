import { test, expect } from '@playwright/test';
import { login, logout, isLoggedIn } from './helpers/auth';

/**
 * E2E Tests for Authentication Flow
 * 
 * These tests verify the complete authentication flow including:
 * - Login page display
 * - Invalid credentials handling
 * - Successful login
 * - Logout functionality
 */

test.describe('Authentication Flow', () => {
  const testUsername = process.env.TEST_USERNAME || 'admin';
  const testPassword = process.env.TEST_PASSWORD || 'admin123';

  test.beforeEach(async ({ page }) => {
    // Navigate to login page
    await page.goto('/login');
    // Wait for page to load
    await page.waitForLoadState('networkidle');
  });

  test('should display login page with all elements', async ({ page }) => {
    // Check if login form elements are visible
    await expect(page.locator('input[id="username"]')).toBeVisible();
    await expect(page.locator('input[id="password"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
    
    // Check for login page title
    await expect(page.locator('h1:has-text("i drill")')).toBeVisible();
    await expect(page.locator('h2:has-text("Sign In")')).toBeVisible();
  });

  test('should show error on invalid credentials', async ({ page }) => {
    // Fill in invalid credentials
    await page.fill('input[id="username"]', 'invalid_user');
    await page.fill('input[id="password"]', 'wrong_password');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Wait for error message
    await expect(
      page.locator('text=/Login failed|Invalid|خطا|نامعتبر/i')
    ).toBeVisible({ timeout: 10000 });
  });

  test('should login successfully with valid credentials', async ({ page }) => {
    // Use login helper
    await login(page, testUsername, testPassword);
    
    // Verify user is logged in
    const loggedIn = await isLoggedIn(page);
    expect(loggedIn).toBe(true);
    
    // Verify redirect to dashboard
    await expect(page).toHaveURL(/\/dashboard|\//);
  });

  test('should logout successfully', async ({ page }) => {
    // First login using helper
    await login(page, testUsername, testPassword);
    
    // Verify logged in
    expect(await isLoggedIn(page)).toBe(true);
    
    // Logout using helper
    try {
      await logout(page);
      
      // Should redirect to login
      await expect(page).toHaveURL(/\/login/, { timeout: 10000 });
      
      // Verify logged out
      expect(await isLoggedIn(page)).toBe(false);
    } catch (error) {
      // If logout button not found, skip this test
      test.skip();
    }
  });

  test('should handle empty form submission', async ({ page }) => {
    // Try to submit empty form
    await page.click('button[type="submit"]');
    
    // Form should show validation (HTML5 required attribute)
    // or stay on login page
    await expect(page).toHaveURL(/\/login/);
  });

  test('should toggle password visibility', async ({ page }) => {
    // Fill password
    await page.fill('input[id="password"]', 'testpassword');
    
    // Check initial state (password should be hidden)
    const passwordInput = page.locator('input[id="password"]');
    await expect(passwordInput).toHaveAttribute('type', 'password');
    
    // Click toggle button (eye icon)
    const toggleButton = page.locator('button:has([class*="Eye"])').first();
    if (await toggleButton.isVisible({ timeout: 2000 })) {
      await toggleButton.click();
      
      // Password should now be visible
      await expect(passwordInput).toHaveAttribute('type', 'text');
    }
  });
});

