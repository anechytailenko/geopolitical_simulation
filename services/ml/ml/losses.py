"""Loss functions for the 5-class edge classifier (plans/03 §2.4).

Both losses are masked to Country->Country target pairs by construction: the caller only
passes logits/labels for those pairs. Class weights default to inverse-frequency on the
training label distribution; STATUS_QUO ends up ~1.0 and MATERIAL_CONFLICT the largest.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_frequency_weights(label_counts: torch.Tensor, normalize_to_min: bool = True) -> torch.Tensor:
    """Class weights ∝ 1 / frequency. `label_counts` is a length-C long/float tensor.

    Zero-count classes are clamped to 1 to avoid div-by-zero. With normalize_to_min the
    smallest weight is scaled to 1.0 (so the majority class ≈ 1.0, rarer classes > 1).
    """
    counts = label_counts.clamp(min=1).float()
    w = counts.sum() / counts
    if normalize_to_min:
        w = w / w.min()
    return w


class FocalLoss(nn.Module):
    """Multiclass focal loss: FL = -alpha_c * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight.gather(0, target)
        return loss.mean()


def make_loss(loss_name: str, class_weights: torch.Tensor | None, focal_gamma: float = 2.0) -> nn.Module:
    """Factory: 'weighted_ce' -> CrossEntropyLoss(weight), 'focal' -> FocalLoss."""
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    raise ValueError(f"unknown loss {loss_name!r} (expected 'weighted_ce' or 'focal')")
