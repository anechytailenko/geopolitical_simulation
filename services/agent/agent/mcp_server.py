"""FastMCP server (plans/04 §3): exposes the task-shaped tools over the Model Context Protocol.

Imports ``ml.infer.Predictor`` in-process (the exact trained model) via the ``Runtime``
singleton and resolves places from the read-only parquet export — it never writes to any
database. The LangGraph agent loads these tools through ``langchain-mcp-adapters`` (see
``graph.py``); any MCP client (e.g. Claude Desktop) can also attach to this server.

Run as a stdio MCP server:   python -m agent.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import tools_core as T
from .runtime import get_runtime

mcp = FastMCP("geopolitic-agent")


# ---- grounding ----------------------------------------------------------------------
@mcp.tool()
def get_latest_time_step() -> dict:
    """Return the default forecast step {time_step, input_period, forecast_period}. The model
    forecasts the month AFTER `time_step`; use `forecast_period` when naming the answer's month."""
    return T.get_latest_time_step(get_runtime())


@mcp.tool()
def resolve_place(text: str, time_step: int | None = None) -> dict:
    """Resolve a natural-language place to ids: a Country (ISO-3) or an IGO group with its member
    ISO-3 codes valid at `time_step`. Always call this first to turn names into ids."""
    return T.resolve_place(get_runtime(), text, time_step)


# ---- answer tools -------------------------------------------------------------------
@mcp.tool()
def predict_pair(source: str, target: str, time_step: int | None = None) -> dict:
    """Predict the 5-class relationship distribution for directed source->target next month.
    `source`/`target` may be country names, ISO-2 or ISO-3 codes."""
    return T.predict_pair(get_runtime(), source, target, time_step)


@mcp.tool()
def best_pair_in_group(group: str, relationship_class: str,
                       time_step: int | None = None, top_k: int = 5) -> dict:
    """Within an IGO group (e.g. EU, NATO, ASEAN), find the directed member pair most likely to
    show `relationship_class` next month. `relationship_class` must be one of the 5 class names."""
    return T.best_pair_in_group(get_runtime(), group, relationship_class, time_step, top_k)


@mcp.tool()
def most_likely_counterpart(country: str, relationship_class: str, direction: str = "incoming",
                            time_step: int | None = None, top_k: int = 5) -> dict:
    """Find the country most likely in `relationship_class` with a fixed country next month.
    direction='incoming' ranks others->country; 'outgoing' ranks country->others."""
    return T.most_likely_counterpart(get_runtime(), country, relationship_class, direction,
                                     time_step, top_k)


@mcp.tool()
def compare_pair(source: str, target: str, time_step: int | None = None) -> dict:
    """Compare conflict vs cooperation vs status-quo for directed source->target next month."""
    return T.compare_pair(get_runtime(), source, target, time_step)


@mcp.tool()
def explain_pair(source: str, target: str, time_step: int | None = None) -> dict:
    """Explain a prediction for source->target: the GNNExplainer subgraph (which countries/edges
    mattered) + named Integrated-Gradients feature attributions for the two countries."""
    return T.explain_pair(get_runtime(), source, target, time_step)


@mcp.tool()
def predict_counterfactual(intervene_source: str, intervene_target: str, intervene_class: str,
                           query_source: str, query_target: str,
                           time_step: int | None = None, symmetric: bool = True) -> dict:
    """What-if: assume (intervene_source,intervene_target) takes `intervene_class` this month,
    then re-predict (query_source -> query_target) next month. Single step only (no multi-month
    chains). Returns baseline vs counterfactual distributions and the per-class delta."""
    return T.predict_counterfactual(get_runtime(), intervene_source, intervene_target,
                                    intervene_class, query_source, query_target,
                                    time_step, symmetric)


def main() -> None:
    # Build the model once up front so a misconfigured artifact path fails fast (plans/04 §11).
    get_runtime()
    mcp.run()


if __name__ == "__main__":
    main()
