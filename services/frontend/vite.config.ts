import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The agent (uvicorn) binds 127.0.0.1 by default, so target IPv4 explicitly — using "localhost"
// can resolve to ::1 first and yield a spurious ECONNREFUSED. Override with VITE_AGENT_TARGET.
const AGENT_TARGET = process.env.VITE_AGENT_TARGET ?? "http://127.0.0.1:8100";

// Dev server proxies /agent → the agent SSE server (which has no CORS), so the browser sees a
// single origin. If the agent is down, the proxy would otherwise dump a raw Node AggregateError
// (ECONNREFUSED) to the terminal and leave the browser request hanging; the error handler below
// turns that into a clean one-line hint + a graceful SSE `error` frame the UI can render.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/agent": {
        target: AGENT_TARGET,
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("error", (err, _req, res) => {
            const hint =
              `Cannot reach the agent at ${AGENT_TARGET}. Start it: ` +
              "`cd services/agent && .venv/bin/python -m agent` (and Ollama). See COMMANDS.md §11e/§12.";
            // one clean line, not a stack trace
            console.warn(`[vite] /agent proxy: ${(err as NodeJS.ErrnoException).code ?? err.message} — ${hint}`);
            const r = res as any;
            if (r && typeof r.writeHead === "function" && !r.headersSent) {
              r.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
            }
            if (r && typeof r.end === "function") {
              r.end(`event: error\ndata: ${JSON.stringify({ message: hint })}\n\n`);
            }
          });
        },
      },
    },
  },
});
