"""``python -m agent`` -> run the SSE server. ``python -m agent.mcp_server`` -> the MCP server."""

from __future__ import annotations

import uvicorn

from .config import AgentConfig
from .runtime import get_runtime


def main() -> None:
    cfg = AgentConfig.from_env()
    get_runtime()  # fail fast on artifact/class issues before binding the port (plans/04 §11)
    uvicorn.run("agent.server:app", host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
