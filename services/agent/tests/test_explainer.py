"""Explanation enrichment (plans/04 §12): renderable subgraph + named IG, IG only for u/v."""

from agent.explainer import Explainer
from ml.config import CLASS_NAMES


def test_explain_payload(rt):
    # a slightly larger IG budget makes the completeness check tight without being slow.
    ex = Explainer(rt.predictor, gnn_epochs=8, ig_steps=24)
    out = ex.explain("USA", "CHN", 197)

    assert out["target_class"] in CLASS_NAMES

    sg = out["subgraph"]
    assert all(0.0 <= e["importance"] <= 1.0 for e in sg["edges"])
    assert all(e["src"] in rt.country_ids and e["tgt"] in rt.country_ids for e in sg["edges"])
    assert all(e["dominant_class"] in CLASS_NAMES for e in sg["edges"] if e["dominant_class"])

    # IG feature attributions exist ONLY for the focus pair's u and v (plans/04 §3 A1)
    assert set(out["feature_attributions"].keys()) == {"USA", "CHN"}
    feats = [f["feature"] for f in out["feature_attributions"]["USA"]]
    known = ("gdp", "military", "conflict", "region=", "member_of=", "population", "hdi", "vdem")
    assert any(any(k in f for k in known) for f in feats)

    # only u and v are clickable for the IG popup
    clickable = {n["id"] for n in sg["nodes"] if n["ig_clickable"]}
    assert clickable == {"USA", "CHN"}

    # Integrated Gradients completeness axiom holds (plans/03 §3, plans/04 §12)
    assert out["integrated_gradients_completeness_gap"] < 0.05


def test_feature_name_vocabularies(rt):
    ex = Explainer(rt.predictor)
    cnames = ex.country_feature_names()
    assert len(cnames) == rt.predictor.ds.pp.country_feat_dim
    enames = ex.edge_feature_names()
    assert len(enames) == rt.predictor.ds.pp.edge_dim
    assert enames[-len(CLASS_NAMES):] == [f"class_dist={c}" for c in CLASS_NAMES]
