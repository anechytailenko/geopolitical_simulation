"""Task-shaped tools (plans/04 §3): one call answers a whole question.

Each function takes the shared ``Runtime`` (model + place resolver) and typed args, validates
them server-side (ISO-3 exists, class in CLASS_NAMES, time_step in range), and returns a plain
JSON-serializable dict with a ``focus_pair`` the deterministic viz step explains. The heavy
compute (batched prediction, ranking, explanation) runs here so a small local LLM only has to
pick a tool and fill a few args. These functions are the single source of truth; the MCP server
(``mcp_server.py``) and the LangGraph tools (``graph.py``) both wrap them.
"""

from __future__ import annotations

from ml import timestep
from ml.config import CLASS_NAMES

from .counterfactual import predict_counterfactual as _counterfactual
from .runtime import Runtime


class ToolError(ValueError):
    """A recoverable, user-facing validation error (bad id/class/time_step)."""


# ---- validation ---------------------------------------------------------------------
def _check_class(name: str) -> str:
    if name not in CLASS_NAMES:
        raise ToolError(f"unknown relationship_class {name!r}; valid: {CLASS_NAMES}")
    return name


def _check_ts(rt: Runtime, time_step: int | None) -> int:
    if time_step is None:
        return rt.max_ts
    ts = int(time_step)
    if ts < rt.min_ts or ts > rt.max_ts:
        raise ToolError(f"time_step {ts} out of range [{rt.min_ts}, {rt.max_ts}]")
    return ts


def _country(rt: Runtime, text: str) -> str:
    from .places import PlaceError
    try:
        return rt.resolver.resolve_country(text)
    except PlaceError as e:
        raise ToolError(str(e))


def _focus(pred) -> dict:
    return {"src": pred.source_id, "tgt": pred.target_id,
            "predicted_class": pred.predicted_class, "confidence": pred.confidence,
            "probabilities": pred.probabilities}


def _periods(rt: Runtime, ts: int) -> dict:
    return {"time_step": ts, "input_period": timestep.iso_period(ts),
            "forecast_period": timestep.iso_period(ts + 1)}


# ---- grounding ----------------------------------------------------------------------
def get_latest_time_step(rt: Runtime) -> dict:
    """The default forecast step: predict at T=max_ts, forecasting the unobserved T+1 month."""
    return _periods(rt, rt.max_ts)


def resolve_place(rt: Runtime, text: str, time_step: int | None = None) -> dict:
    """Map any natural-language place to ids: a Country (ISO-3) or an IGO group (+ members)."""
    from .places import PlaceError
    ts = _check_ts(rt, time_step)
    try:
        return rt.resolver.resolve(text, ts).to_dict()
    except PlaceError as e:
        raise ToolError(str(e))


# ---- answer tools -------------------------------------------------------------------
def predict_pair(rt: Runtime, source: str, target: str, time_step: int | None = None) -> dict:
    """Type 1 — the 5-class distribution for directed source->target at T+1."""
    ts = _check_ts(rt, time_step)
    s, t = _country(rt, source), _country(rt, target)
    pred = rt.predictor.predict(s, t, ts)
    return {**_periods(rt, ts), "probabilities": pred.probabilities,
            "predicted_class": pred.predicted_class, "confidence": pred.confidence,
            "focus_pair": _focus(pred)}


def best_pair_in_group(rt: Runtime, group: str, relationship_class: str,
                       time_step: int | None = None, top_k: int = 5) -> dict:
    """Type 2 — the directed member pair of a group most likely to show ``relationship_class``."""
    from .places import PlaceError
    ts = _check_ts(rt, time_step)
    cls = _check_class(relationship_class)
    try:
        place = rt.resolver.resolve_group(group, ts)
    except PlaceError as e:
        raise ToolError(str(e))
    members = place.members or []
    if len(members) < 2:
        raise ToolError(f"group {group!r} has <2 members in the model universe at T={ts}")

    pairs = [(u, v) for u in members for v in members if u != v]
    preds = rt.predictor.predict_batch(pairs, ts)
    preds.sort(key=lambda p: p.probabilities[cls], reverse=True)
    top = preds[:max(1, top_k)]
    ranked = [{"src": p.source_id, "tgt": p.target_id, "prob": p.probabilities[cls]} for p in top]
    return {**_periods(rt, ts), "group": place.code, "relationship_class": cls,
            "ranked": ranked, "focus_pair": _focus(top[0])}


def most_likely_counterpart(rt: Runtime, country: str, relationship_class: str,
                            direction: str = "incoming", time_step: int | None = None,
                            top_k: int = 5) -> dict:
    """Type 3 — the country most likely in ``relationship_class`` with the fixed country.
    direction='incoming' ranks X->country; 'outgoing' ranks country->X."""
    ts = _check_ts(rt, time_step)
    cls = _check_class(relationship_class)
    if direction not in ("incoming", "outgoing"):
        raise ToolError("direction must be 'incoming' or 'outgoing'")
    c = _country(rt, country)
    others = sorted(x for x in rt.country_ids if x != c)
    pairs = [(x, c) for x in others] if direction == "incoming" else [(c, x) for x in others]
    preds = rt.predictor.predict_batch(pairs, ts)
    preds.sort(key=lambda p: p.probabilities[cls], reverse=True)
    top = preds[:max(1, top_k)]
    ranked = [{"counterpart": (p.source_id if direction == "incoming" else p.target_id),
               "prob": p.probabilities[cls]} for p in top]
    return {**_periods(rt, ts), "country": c, "direction": direction,
            "relationship_class": cls, "ranked": ranked, "focus_pair": _focus(top[0])}


def compare_pair(rt: Runtime, source: str, target: str, time_step: int | None = None) -> dict:
    """Compare — conflict vs cooperation vs status-quo for source->target (5->3 collapse)."""
    ts = _check_ts(rt, time_step)
    s, t = _country(rt, source), _country(rt, target)
    pred = rt.predictor.predict(s, t, ts)
    p = pred.probabilities
    conflict = p["MATERIAL_CONFLICT"] + p["VERBAL_CONFLICT"]
    cooperation = p["MATERIAL_COOPERATION"] + p["VERBAL_COOPERATION"]
    status_quo = p["STATUS_QUO"]
    verdict = max({"conflict": conflict, "cooperation": cooperation, "status_quo": status_quo}.items(),
                  key=lambda kv: kv[1])[0]
    return {**_periods(rt, ts), "conflict_p": conflict, "cooperation_p": cooperation,
            "status_quo_p": status_quo, "verdict": verdict, "focus_pair": _focus(pred)}


def explain_pair(rt: Runtime, source: str, target: str, time_step: int | None = None) -> dict:
    """Why — the GNNExplainer subgraph + named Integrated-Gradients attributions (left panel)."""
    ts = _check_ts(rt, time_step)
    s, t = _country(rt, source), _country(rt, target)
    out = rt.explainer.explain(s, t, ts)
    out["focus_pair"] = {"src": s, "tgt": t, "predicted_class": out["target_class"]}
    out.update(_periods(rt, ts))
    return out


def predict_counterfactual(rt: Runtime, intervene_source: str, intervene_target: str,
                           intervene_class: str, query_source: str, query_target: str,
                           time_step: int | None = None, symmetric: bool = True) -> dict:
    """Type 6 — what-if: edit the month-T (intervene_source,intervene_target) edge to
    ``intervene_class`` and re-predict (query_source -> query_target) at T+1 (single step)."""
    ts = _check_ts(rt, time_step)
    cls = _check_class(intervene_class)
    isrc, itgt = _country(rt, intervene_source), _country(rt, intervene_target)
    qsrc, qtgt = _country(rt, query_source), _country(rt, query_target)
    out = _counterfactual(rt.predictor, isrc, itgt, cls, qsrc, qtgt, ts, symmetric=symmetric)
    out.update({"input_period": timestep.iso_period(ts),
                "forecast_period": timestep.iso_period(ts + 1)})
    return out
