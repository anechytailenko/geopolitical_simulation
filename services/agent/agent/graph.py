"""LangGraph ReAct agent + the deterministic visualization step (plans/04 §4).

* ``build_tools`` wraps the task-shaped tools (``tools_core``) as LangChain ``StructuredTool``s.
  Each returns ``(summary, artifact)``: the LLM sees a short summary (keeps a small local model's
  context tight), while the full dict is kept as the message ``artifact`` for the viz step — so
  the left panel never has to re-parse free text.
* ``build_agent`` is ``langgraph.prebuilt.create_react_agent`` (reason->act->observe loop with a
  recursion cap) driven by the chosen LLM.
* ``assemble_viz`` is the **deterministic** node: it finds the answer's focus pair from the tool
  artifacts and always runs ``explain_pair`` on it, so the subgraph + IG popup can never disagree
  with the answer — even if the model never asked "why".
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from . import tools_core as T
from .config import AgentConfig
from .llm import SYSTEM_PROMPT
from .runtime import Runtime

CONFIDENCE_NOTE = "model confidence (temperature-scaled, ~uncalibrated on this checkpoint)"


# ---- tool wrappers ------------------------------------------------------------------
def _summarize(name: str, res: dict) -> str:
    fp = res.get("focus_pair")
    fc = res.get("forecast_period", "")
    if name == "get_latest_time_step":
        return f"latest input month {res['input_period']}, forecast {res['forecast_period']}"
    if name == "resolve_place":
        if res.get("kind") == "group":
            return f"group {res['name']} ({res['code']}): {len(res.get('members') or [])} members"
        return f"{res['name']} -> {res['iso3']}"
    if name == "compare_pair":
        return (f"{fp['src']}->{fp['tgt']}: verdict {res['verdict']} "
                f"(conflict {res['conflict_p']:.2f} / coop {res['cooperation_p']:.2f} / "
                f"status-quo {res['status_quo_p']:.2f}) in {fc}")
    if name in ("best_pair_in_group", "most_likely_counterpart"):
        top = res["ranked"][0]
        label = f"{top.get('src','')}->{top.get('tgt','')}".strip("->") or top.get("counterpart", "")
        return f"top {res['relationship_class']}: {label} p={list(top.values())[-1]:.2f} in {fc}"
    if name == "predict_counterfactual":
        b, c = res["baseline"], res["counterfactual"]
        return (f"{res['intervened_edge']['src']}-{res['intervened_edge']['tgt']} -> "
                f"{res['intervened_edge']['class']}: {fp['src']}->{fp['tgt']} "
                f"baseline {b['predicted_class']} ({b['confidence']:.2f}) vs counterfactual "
                f"{c['predicted_class']} ({c['confidence']:.2f}) in {fc}")
    if name == "explain_pair":
        return f"explained {fp['src']}->{fp['tgt']}: {res['target_class']}"
    if fp:  # predict_pair
        return f"{fp['src']}->{fp['tgt']}: {fp['predicted_class']} (p={fp['confidence']:.2f}) in {fc}"
    return name


def build_tools(rt: Runtime) -> list[StructuredTool]:
    """The §3 tools as LangChain StructuredTools (signatures/descriptions drive tool-calling).
    Each closes over the shared Runtime and returns (summary, full-artifact)."""

    def get_latest_time_step() -> dict:
        """Return the default forecast step {time_step, input_period, forecast_period}; name the
        answer's month using forecast_period (the predicted T+1 month)."""
        r = T.get_latest_time_step(rt); return _summarize("get_latest_time_step", r), r

    def resolve_place(text: str, time_step: Optional[int] = None) -> dict:
        """Resolve a place name to ids: a Country (ISO-3) or an IGO group with member ISO-3
        codes valid at time_step. Use for ambiguous names or to get a group's members."""
        r = T.resolve_place(rt, text, time_step); return _summarize("resolve_place", r), r

    def predict_pair(source: str, target: str, time_step: Optional[int] = None) -> dict:
        """Predict the 5-class relationship distribution for directed source->target next month
        (source/target may be country names or ISO-2/ISO-3 codes)."""
        r = T.predict_pair(rt, source, target, time_step); return _summarize("predict_pair", r), r

    def best_pair_in_group(group: str, relationship_class: str,
                           time_step: Optional[int] = None, top_k: int = 5) -> dict:
        """Within an IGO group (EU, NATO, ASEAN, ...), find the directed member pair most likely
        to show relationship_class next month."""
        r = T.best_pair_in_group(rt, group, relationship_class, time_step, top_k)
        return _summarize("best_pair_in_group", r), r

    def most_likely_counterpart(country: str, relationship_class: str, direction: str = "incoming",
                                time_step: Optional[int] = None, top_k: int = 5) -> dict:
        """Find the country most likely in relationship_class with a fixed country next month.
        direction='incoming' ranks others->country; 'outgoing' ranks country->others."""
        r = T.most_likely_counterpart(rt, country, relationship_class, direction, time_step, top_k)
        return _summarize("most_likely_counterpart", r), r

    def compare_pair(source: str, target: str, time_step: Optional[int] = None) -> dict:
        """Compare conflict vs cooperation vs status-quo for directed source->target next month."""
        r = T.compare_pair(rt, source, target, time_step); return _summarize("compare_pair", r), r

    def explain_pair(source: str, target: str, time_step: Optional[int] = None) -> dict:
        """Explain source->target: the GNNExplainer subgraph + named Integrated-Gradients feature
        attributions for the two countries. Use when the user asks WHY."""
        r = T.explain_pair(rt, source, target, time_step); return _summarize("explain_pair", r), r

    def predict_counterfactual(intervene_source: str, intervene_target: str, intervene_class: str,
                               query_source: str, query_target: str,
                               time_step: Optional[int] = None, symmetric: bool = True) -> dict:
        """What-if (single step): assume (intervene_source,intervene_target) takes intervene_class
        this month, then re-predict (query_source->query_target) next month. Returns baseline vs
        counterfactual and the per-class delta."""
        r = T.predict_counterfactual(rt, intervene_source, intervene_target, intervene_class,
                                     query_source, query_target, time_step, symmetric)
        return _summarize("predict_counterfactual", r), r

    fns = [get_latest_time_step, resolve_place, predict_pair, best_pair_in_group,
           most_likely_counterpart, compare_pair, explain_pair, predict_counterfactual]
    return [StructuredTool.from_function(f, response_format="content_and_artifact") for f in fns]


# ---- agent --------------------------------------------------------------------------
def build_agent(llm, rt: Runtime):
    return create_react_agent(llm, build_tools(rt), prompt=SYSTEM_PROMPT)


# ---- deterministic viz --------------------------------------------------------------
def _artifacts(messages) -> list[dict]:
    """Full tool-result dicts, in order. Prefers the structured ``artifact`` (in-process tools)
    and falls back to JSON in ``content`` (so it also works when tools arrive over MCP)."""
    import json
    out = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        art = getattr(m, "artifact", None)
        if isinstance(art, dict):
            out.append(art)
        elif isinstance(m.content, str):
            try:
                obj = json.loads(m.content)
                if isinstance(obj, dict):
                    out.append(obj)
            except (ValueError, TypeError):
                pass
    return out


def extract_focus(messages) -> Optional[dict]:
    """The last tool artifact carrying a focus_pair (the answer's focus), or None."""
    for art in reversed(_artifacts(messages)):
        if art.get("focus_pair"):
            return art
    return None


def assemble_viz(rt: Runtime, messages, answer_text: str) -> Optional[dict]:
    art = extract_focus(messages)
    if art is None:
        return None
    fp = dict(art["focus_pair"])
    ts = art.get("time_step", rt.max_ts)
    ex = rt.explainer.explain(fp["src"], fp["tgt"], ts)
    viz = {
        "answer": answer_text,
        "time_step": ts,
        "input_period": art.get("input_period"),
        "forecast_period": art.get("forecast_period"),
        "confidence_note": CONFIDENCE_NOTE,
        "focus_pairs": [fp],
        "intervention": art.get("intervened_edge"),
        "subgraph": ex["subgraph"],
        "feature_attributions": ex["feature_attributions"],
    }
    # Type-6 what-if: carry baseline vs counterfactual so the UI can show the shift (04 §6).
    if "baseline" in art and "counterfactual" in art:
        viz["counterfactual"] = {"baseline": art["baseline"],
                                 "counterfactual": art["counterfactual"],
                                 "delta": art.get("delta")}
    return viz


def run_chat(rt: Runtime, llm, message: str, cfg: AgentConfig | None = None) -> dict:
    """Synchronous convenience: run the ReAct agent + viz, return {answer, viz, messages}."""
    cfg = cfg or AgentConfig.from_env()
    agent = build_agent(llm, rt)
    state = agent.invoke({"messages": [("user", message)]},
                         config={"recursion_limit": cfg.recursion_limit})
    messages = state["messages"]
    answer = next((m.content for m in reversed(messages)
                   if isinstance(m, AIMessage) and m.content and not m.tool_calls), "")
    return {"answer": answer, "viz": assemble_viz(rt, messages, answer), "messages": messages}
