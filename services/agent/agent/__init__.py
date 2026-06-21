"""ReAct agent + MCP tools over the trained geopolitic GNN (plans/04-react-agent.md).

The agent answers natural-language geopolitical questions by composing the trained model's
atomic operation ``predict(source, target, T)`` through a small set of *task-shaped* tools
(one tool call answers a whole question). The tools run the exact ``ml.infer.Predictor``
in-process and resolve places from the same exported dataset the model was trained on, so the
agent can never drift from the model and never writes to any database.
"""

import warnings as _warnings

# Quiet two cosmetic, harmless warnings on a normal `python -m agent` / `-m agent.mcp_server` run:
#  - the shipped scaler was pickled with an older sklearn (unpickles fine, read-only inference);
#  - create_react_agent is deprecated in LangGraph v1 but still functional (no behavior change).
try:  # pragma: no cover - depends on sklearn version
    from sklearn.exceptions import InconsistentVersionWarning

    _warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:  # pragma: no cover
    pass
_warnings.filterwarnings("ignore", message=r".*create_react_agent has been moved.*")

from .config import AgentConfig

__all__ = ["AgentConfig"]
