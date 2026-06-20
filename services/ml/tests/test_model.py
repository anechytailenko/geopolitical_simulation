"""Model + losses/metrics tests (gated on torch / torch_geometric / torchmetrics)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch

from ml.dataset import GeopoliticDataset
from ml.model import SpatioTemporalEdgeClassifier


def test_forward_shapes_and_softmax(tiny_cfg):
    import numpy as np
    ds = GeopoliticDataset.from_parquet(tiny_cfg)
    model = SpatioTemporalEdgeClassifier.from_dataset(ds, tiny_cfg)
    model.eval()
    win = ds.build_window(13)
    s = ds.make_samples("train", np.random.default_rng(0))[0]
    logits = model(win, s.pair_index, s.pair_attr)
    P = s.pair_index.shape[0]
    assert logits.shape == (P, len(tiny_cfg.class_names))
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(P), atol=1e-5)


def test_gradients_flow(tiny_cfg):
    import numpy as np
    ds = GeopoliticDataset.from_parquet(tiny_cfg)
    model = SpatioTemporalEdgeClassifier.from_dataset(ds, tiny_cfg)
    s = ds.make_samples("train", np.random.default_rng(0))[0]
    logits = model(ds.build_window(13), s.pair_index, s.pair_attr)
    loss = torch.nn.functional.cross_entropy(logits, s.labels)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
