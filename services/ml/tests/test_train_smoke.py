"""End-to-end smoke test (gated on the full stack). Trains 2 epochs on the tiny synthetic
dataset with W&B disabled, then checks the checkpoint policy, artifacts, calibration, a
served prediction, and explainer completeness. No Neo4j, no data deletion."""

import json
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("torchmetrics")
pytest.importorskip("sklearn")

import torch

from ml.train import train
from ml.infer import Predictor
from ml.explain import integrated_gradients


def test_train_creates_artifacts_and_serves(tiny_cfg):
    metrics = train(tiny_cfg)

    a = tiny_cfg.artifacts_dir
    for f in ("best.pt", "last.pt", "preprocess.pkl", "calibrator.pkl", "metrics.json", "node_index.json"):
        assert os.path.exists(os.path.join(a, f)), f"missing artifact {f}"

    assert "test_macro_f1" in metrics
    saved = json.load(open(os.path.join(a, "metrics.json")))
    assert 0.0 <= saved["test_macro_f1"] <= 1.0
    # ECE values are valid probabilities-of-miscalibration in [0,1]. (Temperature scaling
    # minimizes val NLL; on a tiny synthetic set it need not strictly lower *test* ECE, so we
    # don't assert ordering here — that's a research expectation, checked on real data.)
    assert 0.0 <= saved["test_ece_uncalibrated"] <= 1.0
    assert 0.0 <= saved["test_ece_calibrated"] <= 1.0

    # best.pt carries optimizer + epoch so a cut-off run can resume (checkpoint policy)
    ckpt = torch.load(os.path.join(a, "best.pt"), map_location="cpu", weights_only=False)
    for k in ("model_state", "optimizer_state", "epoch", "best_macro_f1", "preprocess"):
        assert k in ckpt

    # served prediction sums to 1 and restores from the same artifacts bundle
    pred = Predictor(tiny_cfg).predict("RUS", "UKR", 13)
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4
    assert pred.predicted_class in tiny_cfg.class_names


def test_integrated_gradients_completeness(tiny_cfg):
    from ml.dataset import GeopoliticDataset
    from ml.model import SpatioTemporalEdgeClassifier
    import numpy as np

    ds = GeopoliticDataset.from_parquet(tiny_cfg)
    model = SpatioTemporalEdgeClassifier.from_dataset(ds, tiny_cfg)
    model.eval()
    u, v = ds.country_index["RUS"], ds.country_index["UKR"]
    attr = torch.zeros((1, ds.pp.edge_dim))
    res = integrated_gradients(model, ds.build_window(13), u, v, attr, steps=128)
    # completeness axiom: sum(attributions over ALL perturbed inputs) ≈ F(x) - F(baseline);
    # the gap is the Riemann-sum error, which shrinks with steps.
    assert res.completeness_gap < 5e-2
