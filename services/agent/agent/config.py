"""Agent configuration (LLM provider + serving), separate from the ML ``Config``.

Model artifacts and the parquet dataset are located by the ML layer via ``GEO_ARTIFACTS_DIR`` /
``GEO_DATA_DIR`` (see ``ml.config.Config.from_env`` and plans/04 §11). This object only adds the
agent-specific knobs: which LLM to drive the ReAct loop and where to serve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """The geopolitic repo root (…/services/agent/agent/config.py → parents[3])."""
    return Path(__file__).resolve().parents[3]


def default_data_dir() -> str:
    """The parquet export shipped in the repo (used when GEO_DATA_DIR is unset/invalid)."""
    return str(repo_root() / "services" / "ml" / "dataset_parquet")


def default_artifacts_dir() -> str:
    """The trained-model bundle shipped in the repo (used when GEO_ARTIFACTS_DIR is unset/invalid)."""
    return str(repo_root() / "artifacts")


def load_dotenv_files() -> None:
    """Load `.env` from the repo root then `services/agent/.env` (existing env always wins), so
    `python -m agent` works without manually exporting vars. No-op if python-dotenv is missing."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    root = repo_root()
    for p in (root / ".env", root / "services" / "agent" / ".env"):
        if p.exists():
            load_dotenv(p, override=False)


@dataclass
class AgentConfig:
    # ---- LLM (plans/04 §4) ----
    llm_provider: str = "ollama"          # "ollama" | "anthropic" | "openai"
    llm_model: str = "qwen2.5:3b-instruct"  # default local tool-calling model (~3 GB, plans/04 §10)
    llm_temperature: float = 0.0          # deterministic tool selection
    ollama_base_url: str = "http://localhost:11434"

    # ---- ReAct loop ----
    recursion_limit: int = 12             # caps the reason->act->observe loop (plans/04 §4)

    # ---- serving ----
    host: str = "127.0.0.1"
    port: int = 8100

    @classmethod
    def from_env(cls) -> "AgentConfig":
        c = cls()
        c.llm_provider = os.environ.get("LLM_PROVIDER", c.llm_provider)
        c.llm_model = os.environ.get("LLM_MODEL", c.llm_model)
        c.llm_temperature = float(os.environ.get("LLM_TEMPERATURE", c.llm_temperature))
        c.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", c.ollama_base_url)
        c.recursion_limit = int(os.environ.get("AGENT_RECURSION_LIMIT", c.recursion_limit))
        c.host = os.environ.get("AGENT_HOST", c.host)
        c.port = int(os.environ.get("AGENT_PORT", c.port))
        return c
