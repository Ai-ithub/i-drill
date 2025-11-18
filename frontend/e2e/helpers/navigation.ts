import { Page, expect } from '@playwright/test';

/**
 * Navigation helper functions for E2E tests
 */

/**
 * Navigate to dashboard page
 */
export async function goToDashboard(page: Page): Promise<void> {
  await page.goto('/dashboard');
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(/\/dashboard/);
}

/**
 * Navigate to real-time monitoring page
 */
export async function goToRealtime(page: Page): Promise<void> {
  await page.goto('/realtime');
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(/\/realtime/);
}

/**
 * Navigate to data page
 */
export async function goToData(page: Page): Promise<void> {
  await page.goto('/data');
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(/\/data/);
}

/**
 * Navigate to RTO page
 */
export async function goToRTO(page: Page): Promise<void> {
  await page.goto('/rto');
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(/\/rto/);
}

/**
 * Navigate to DVR page
 */
export async function goToDVR(page: Page): Promise<void> {
  await page.goto('/dvr-page');
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(/\/dvr-page/);
}

/**
 * Navigate to PDM page
 */
export async function goToPDM(page: Page): Promise<void> {
  await page.goto('/pdm');
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveURL(/\/pdm/);
}
