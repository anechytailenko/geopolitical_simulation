"""Loss + metric unit tests (gated on torch / torchmetrics)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchmetrics")

import torch

from ml.losses import FocalLoss, inverse_frequency_weights, make_loss
from ml.metrics import Evaluator


def test_inverse_frequency_weights():
    counts = torch.tensor([10, 20, 5, 0, 100])  # includes a zero-count class
    w = inverse_frequency_weights(counts)
    assert w.shape == (5,)
    assert torch.isfinite(w).all()
    assert abs(float(w.min()) - 1.0) < 1e-6        # majority normalized to 1.0
    assert float(w[2]) > float(w[4])               # rarer class -> larger weight


def test_losses_are_finite_scalars():
    logits = torch.randn(16, 5)
    target = torch.randint(0, 5, (16,))
    w = torch.ones(5)
    for loss in (make_loss("weighted_ce", w), make_loss("focal", w, focal_gamma=2.0), FocalLoss()):
        val = loss(logits, target)
        assert val.dim() == 0 and torch.isfinite(val)


def test_evaluator_ranges():
    probs = torch.softmax(torch.randn(64, 5), dim=-1)
    target = torch.randint(0, 5, (64,))
    ev = Evaluator("cpu")
    ev.update(probs, target)
    res = ev.compute()
    assert 0.0 <= res.macro_f1 <= 1.0
    assert 0.0 <= res.ece <= 1.0
    assert len(res.confusion) == 5 and len(res.confusion[0]) == 5
    assert sum(sum(r) for r in res.confusion) == 64
