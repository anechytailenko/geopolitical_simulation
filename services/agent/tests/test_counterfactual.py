"""Single-step counterfactual mechanism (plans/04 §12).

The trained checkpoint is overconfident and dominated by identity + the query pair's own
features, so a neighbor-edge intervention moves the output only marginally (a real model
property, not a bug). The tests therefore verify the **mechanism** — the baseline reproduces a
normal prediction, the edit is applied at the input level, the cached tensors are never mutated,
and the counterfactual stays a valid one-step distribution — rather than asserting a large
output swing the model does not produce.
"""

import torch

from agent import tools_core as T
from agent.counterfactual import _edit_window, _intervention_vector, predict_counterfactual
from ml.dataset import REL_SNAP


def test_baseline_reproduces_predict_pair(rt):
    cf = predict_counterfactual(rt.predictor, "USA", "CHN", "MATERIAL_COOPERATION",
                                "CHN", "IND", 197)
    pp = T.predict_pair(rt, "CHN", "IND")["probabilities"]
    for k in pp:
        assert abs(cf["baseline"]["probabilities"][k] - pp[k]) < 1e-6


def test_counterfactual_is_valid_distribution(rt):
    cf = predict_counterfactual(rt.predictor, "USA", "CHN", "MATERIAL_COOPERATION",
                                "CHN", "IND", 197)
    assert abs(sum(cf["counterfactual"]["probabilities"].values()) - 1.0) < 1e-5
    assert abs(sum(cf["delta"].values())) < 1e-5            # both distributions sum to 1
    assert all(abs(v) <= 1.0 for v in cf["delta"].values())
    assert cf["intervened_edge"] == {"src": "USA", "tgt": "CHN",
                                     "class": "MATERIAL_COOPERATION", "symmetric": True}
    assert cf["focus_pair"]["src"] == "CHN" and cf["focus_pair"]["tgt"] == "IND"


def test_no_global_state_mutation(rt):
    before = T.predict_pair(rt, "CHN", "IND")["probabilities"]
    predict_counterfactual(rt.predictor, "USA", "CHN", "MATERIAL_COOPERATION", "CHN", "IND", 197)
    after = T.predict_pair(rt, "CHN", "IND")["probabilities"]
    assert before == after


def test_edit_applies_at_input_and_keeps_cache(rt):
    p = rt.predictor
    cache_before = p.ds.snap_attr[197].clone()
    window = list(p.ds.build_window(197))
    iu, iv = p._indices("USA", "CHN")
    vec = _intervention_vector(p, "MATERIAL_COOPERATION")
    edited = _edit_window(p, window, len(window) - 1, [(iu, iv), (iv, iu)], vec)

    base_ea = window[-1][REL_SNAP].edge_attr
    new_ea = edited[-1][REL_SNAP].edge_attr
    # the edit changed the month-T snapshot edges (replaced rows and/or appended)
    if new_ea.shape == base_ea.shape:
        assert not torch.equal(new_ea, base_ea)
    else:
        assert new_ea.shape[0] > base_ea.shape[0]
    # the dataset's cached tensor was not mutated (edit is local to the call)
    assert torch.equal(p.ds.snap_attr[197], cache_before)


def test_intervention_vector_shape(rt):
    vec = _intervention_vector(rt.predictor, "MATERIAL_CONFLICT")
    assert vec.shape == (1, rt.predictor.ds.pp.edge_dim)
