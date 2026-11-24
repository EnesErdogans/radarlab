import { defineConfig } from 'vite';

// For GitHub Pages the site lives under /<repo>/. Relative base works for any repo name.
export default defineConfig({
  base: './',
  worker: { format: 'es' },
  test: { include: ['tests/**/*.test.ts'], testTimeout: 60000 },
} as any);
