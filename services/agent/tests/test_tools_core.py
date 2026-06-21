"""Task-shaped tools + server-side validation (plans/04 §12)."""

import pytest

from agent import tools_core as T
from agent.places import Place
from agent.tools_core import ToolError


def test_latest_time_step(rt):
    assert T.get_latest_time_step(rt) == {
        "time_step": 197, "input_period": "2026-06", "forecast_period": "2026-07"}


def test_predict_pair_distribution(rt):
    r = T.predict_pair(rt, "United States", "China")
    p = r["probabilities"]
    assert abs(sum(p.values()) - 1.0) < 1e-5
    assert r["predicted_class"] == max(p, key=p.get)
    assert abs(r["confidence"] - max(p.values())) < 1e-9
    assert r["focus_pair"]["src"] == "USA" and r["focus_pair"]["tgt"] == "CHN"
    assert r["forecast_period"] == "2026-07"


def test_predict_pair_deterministic(rt):
    a = T.predict_pair(rt, "USA", "CHN")["probabilities"]
    b = T.predict_pair(rt, "USA", "CHN")["probabilities"]
    assert a == b


def test_quiet_no_edge_dyad(rt):
    """Two tiny states with no SNAPSHOT_EDGE at T still get a valid distribution (zero edge vec)."""
    r = T.predict_pair(rt, "TUV", "NRU")
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-5


def test_validation_bad_country(rt):
    with pytest.raises(ToolError):
        T.predict_pair(rt, "USA", "Narnia")


def test_validation_bad_class_lists_valid(rt):
    with pytest.raises(ToolError) as e:
        T.best_pair_in_group(rt, "EU", "WAR")
    assert "MATERIAL_CONFLICT" in str(e.value)


def test_validation_time_step_range(rt):
    with pytest.raises(ToolError):
        T.predict_pair(rt, "USA", "CHN", time_step=5)      # below min
    with pytest.raises(ToolError):
        T.predict_pair(rt, "USA", "CHN", time_step=999)    # above max


def test_best_pair_in_group_ranking(rt):
    r = T.best_pair_in_group(rt, "EU", "MATERIAL_COOPERATION", top_k=4)
    probs = [x["prob"] for x in r["ranked"]]
    assert probs == sorted(probs, reverse=True)
    assert len(r["ranked"]) <= 4
    assert all(x["src"] != x["tgt"] for x in r["ranked"])
    assert r["focus_pair"]["src"] == r["ranked"][0]["src"]
    assert r["focus_pair"]["tgt"] == r["ranked"][0]["tgt"]


def test_best_pair_in_group_small_group_errors(rt, monkeypatch):
    monkeypatch.setattr(rt.resolver, "resolve_group",
                        lambda g, ts: Place(kind="group", name="X", code="X", qid="Q", members=["USA"]))
    with pytest.raises(ToolError):
        T.best_pair_in_group(rt, "X", "MATERIAL_CONFLICT")


def test_most_likely_counterpart(rt):
    inc = T.most_likely_counterpart(rt, "Ukraine", "MATERIAL_CONFLICT", "incoming", top_k=5)
    assert all(x["counterpart"] != "UKR" for x in inc["ranked"])
    assert all(x["counterpart"] in rt.country_ids for x in inc["ranked"])
    probs = [x["prob"] for x in inc["ranked"]]
    assert probs == sorted(probs, reverse=True)
    out = T.most_likely_counterpart(rt, "UKR", "MATERIAL_CONFLICT", "outgoing", top_k=5)
    assert out["direction"] == "outgoing"


def test_compare_pair_buckets(rt):
    r = T.compare_pair(rt, "India", "Pakistan")
    assert abs(r["conflict_p"] + r["cooperation_p"] + r["status_quo_p"] - 1.0) < 1e-5
    assert r["verdict"] in ("conflict", "cooperation", "status_quo")
