import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Component/unit tests run under jsdom with fetch/SSE mocked — no browser, no network, no agent,
// and therefore no database I/O (plans/05 §8). Cannot write, delete, or drop any data.
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: false,
  },
});
