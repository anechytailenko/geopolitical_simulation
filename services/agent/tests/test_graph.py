"""ReAct graph + deterministic viz (plans/04 §12), driven by a scripted (no-LLM) model."""

from agent import graph as G
from agent.llm import ScriptedChatModel


def test_type1_direct(rt):
    llm = ScriptedChatModel(turns=[
        {"tool": "predict_pair", "args": {"source": "United States", "target": "China"}},
        "Next month (2026-07) USA->China is most likely MATERIAL_CONFLICT.",
    ])
    out = G.run_chat(rt, llm, "What is likely between the USA and China next month?")
    assert out["answer"].startswith("Next month")
    viz = out["viz"]
    assert viz["focus_pairs"][0]["src"] == "USA" and viz["focus_pairs"][0]["tgt"] == "CHN"
    assert set(viz["feature_attributions"].keys()) == {"USA", "CHN"}
    assert viz["forecast_period"] == "2026-07"
    assert viz["intervention"] is None


def test_type2_group(rt):
    llm = ScriptedChatModel(turns=[
        {"tool": "best_pair_in_group",
         "args": {"group": "EU", "relationship_class": "MATERIAL_COOPERATION"}},
        "Within the EU the top material-cooperation pair is the winner above.",
    ])
    out = G.run_chat(rt, llm, "Which EU pair is most likely material cooperation?")
    fp = out["viz"]["focus_pairs"][0]
    assert fp["src"] in rt.country_ids and fp["tgt"] in rt.country_ids and fp["src"] != fp["tgt"]


def test_type6_counterfactual(rt):
    llm = ScriptedChatModel(turns=[
        {"tool": "predict_counterfactual", "args": {
            "intervene_source": "United States", "intervene_target": "China",
            "intervene_class": "MATERIAL_COOPERATION",
            "query_source": "China", "query_target": "India"}},
        "If the USA and China cooperate, China->India stays roughly the same.",
    ])
    out = G.run_chat(rt, llm, "If USA & China sign a deal, what about China and India?")
    viz = out["viz"]
    assert viz["intervention"]["src"] == "USA" and viz["intervention"]["class"] == "MATERIAL_COOPERATION"
    assert viz["focus_pairs"][0]["src"] == "CHN" and viz["focus_pairs"][0]["tgt"] == "IND"


def test_viz_always_explains_focus(rt):
    """Even a non-'why' question yields a populated subgraph + IG (the deterministic viz node)."""
    llm = ScriptedChatModel(turns=[
        {"tool": "compare_pair", "args": {"source": "India", "target": "Pakistan"}},
        "Conflict is more likely than cooperation between India and Pakistan.",
    ])
    out = G.run_chat(rt, llm, "Is conflict or cooperation more likely between India and Pakistan?")
    viz = out["viz"]
    assert "subgraph" in viz and "feature_attributions" in viz
    assert set(viz["feature_attributions"].keys()) == {"IND", "PAK"}
    assert all(0.0 <= e["importance"] <= 1.0 for e in viz["subgraph"]["edges"])
