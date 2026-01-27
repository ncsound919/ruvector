/**
 * Test setup file
 */

import { beforeAll, afterAll, afterEach, vi } from 'vitest';

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.RUVBOT_LOG_LEVEL = 'error';

beforeAll(() => {
  // Global setup
});

afterAll(() => {
  // Global cleanup
});

afterEach(() => {
  // Reset mocks after each test
  vi.clearAllMocks();
});
