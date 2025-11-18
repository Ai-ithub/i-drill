import { Page } from '@playwright/test';

/**
 * Authentication helper functions for E2E tests
 */

const DEFAULT_USERNAME = process.env.TEST_USERNAME || 'admin';
const DEFAULT_PASSWORD = process.env.TEST_PASSWORD || 'admin123';

/**
 * Login helper function
 * @param page - Playwright page object
 * @param username - Username (optional, uses default from env)
 * @param password - Password (optional, uses default from env)
 */
export async function login(
  page: Page,
  username: string = DEFAULT_USERNAME,
  password: string = DEFAULT_PASSWORD
): Promise<void> {
  await page.goto('/login');
  await page.waitForLoadState('networkidle');
  
  await page.fill('input[id="username"]', username);
  await page.fill('input[id="password"]', password);
  await page.click('button[type="submit"]');
  
  // Wait for redirect after login
  await page.waitForURL(/\/dashboard|\//, { timeout: 15000 });
  await page.waitForLoadState('networkidle');
}

/**
 * Logout helper function
 * @param page - Playwright page object
 */
export async function logout(page: Page): Promise<void> {
  // Find and click logout button
  // Try multiple selectors for logout button
  const logoutSelectors = [
    'button:has-text("Logout")',
    'button:has-text("خروج")',
    '[data-testid="logout"]',
    'a:has-text("Logout")',
    'a:has-text("خروج")',
  ];
  
  let logoutClicked = false;
  for (const selector of logoutSelectors) {
    try {
      const logoutButton = page.locator(selector).first();
      if (await logoutButton.isVisible({ timeout: 2000 })) {
        await logoutButton.click();
        logoutClicked = true;
        break;
      }
    } catch {
      continue;
    }
  }
  
  // If logout button not found, try to find user menu first
  if (!logoutClicked) {
    try {
      // Look for user menu or profile dropdown
      const userMenu = page.locator('[data-testid="user-menu"], button:has-text("admin")').first();
      if (await userMenu.isVisible({ timeout: 2000 })) {
        await userMenu.click();
        await page.waitForTimeout(500);
        // Try logout again
        for (const selector of logoutSelectors) {
          try {
            const logoutButton = page.locator(selector).first();
            if (await logoutButton.isVisible({ timeout: 1000 })) {
              await logoutButton.click();
              logoutClicked = true;
              break;
            }
          } catch {
            continue;
          }
        }
      }
    } catch {
      throw new Error('Logout button not found');
    }
  }
  
  if (!logoutClicked) {
    throw new Error('Logout button not found');
  }
  
  // Wait for redirect to login
  await page.waitForURL(/\/login/, { timeout: 10000 });
}

/**
 * Check if user is logged in
 * @param page - Playwright page object
 * @returns true if logged in, false otherwise
 */
export async function isLoggedIn(page: Page): Promise<boolean> {
  // Check if login page is visible
  const loginInput = page.locator('input[id="username"]');
  const isLoginPage = await loginInput.isVisible({ timeout: 2000 }).catch(() => false);
  
  // If login page is visible, user is not logged in
  if (isLoginPage) {
    return false;
  }
  
  // Check if we're on a protected page (not login)
  const currentURL = page.url();
  if (currentURL.includes('/login')) {
    return false;
  }
  
  // If we're on dashboard or other pages, assume logged in
  return true;
}
