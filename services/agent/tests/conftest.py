"""Shared fixtures. The whole suite is **read-only**: it loads the parquet export + trained
weights and never opens a writable database connection, so it cannot alter or delete the
ingested data (the Neo4j containers are not even required).
"""

from __future__ import annotations

import os
import pathlib
import warnings

warnings.filterwarnings("ignore")  # the shipped scaler was pickled with an older sklearn

# Locate the model artifacts + parquet export relative to the repo root (…/services/agent/tests).
ROOT = pathlib.Path(__file__).resolve().parents[3]
os.environ.setdefault("GEO_DATA_DIR", str(ROOT / "services" / "ml" / "dataset_parquet"))
os.environ.setdefault("GEO_ARTIFACTS_DIR", str(ROOT / "artifacts"))
# Keep the explainer cheap in tests (the viz node explains every answer).
os.environ.setdefault("GNN_EXPLAINER_EPOCHS", "6")
os.environ.setdefault("IG_STEPS", "4")

import pytest  # noqa: E402

import agent.runtime as runtime_mod  # noqa: E402
from agent.runtime import Runtime  # noqa: E402


@pytest.fixture(scope="session")
def rt():
    """The trained model + place resolver, built once (boot self-check runs)."""
    r = Runtime.build()
    runtime_mod._RUNTIME = r  # share with mcp_server.get_runtime()
    return r
