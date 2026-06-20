"""Evaluation metrics (plans/03 §Evaluation): macro-F1 (primary), per-class F1,
5x5 confusion matrix, and Expected Calibration Error. Implemented with torchmetrics so the
numbers match the standard definitions; falls back to nothing exotic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassCalibrationError,
)

from .config import NUM_CLASSES


@dataclass
class EvalResult:
    macro_f1: float
    per_class_f1: list[float]
    ece: float
    confusion: list[list[int]]
    n: int


class Evaluator:
    """Accumulate probabilities + targets across batches, then `.compute()`.

    Usage:
        ev = Evaluator(device)
        ev.update(probs, target)   # probs: [P, C] softmax, target: [P] long
        result = ev.compute()
    """

    def __init__(self, device: torch.device | str = "cpu", n_bins: int = 15):
        self.device = torch.device(device)
        self._macro = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(self.device)
        self._per = MulticlassF1Score(num_classes=NUM_CLASSES, average=None).to(self.device)
        self._cm = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(self.device)
        self._ece = MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=n_bins, norm="l1").to(self.device)
        self._n = 0

    def reset(self) -> None:
        for m in (self._macro, self._per, self._cm, self._ece):
            m.reset()
        self._n = 0

    @torch.no_grad()
    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        probs = probs.to(self.device)
        target = target.to(self.device)
        preds = probs.argmax(dim=-1)
        self._macro.update(preds, target)
        self._per.update(preds, target)
        self._cm.update(preds, target)
        self._ece.update(probs, target)
        self._n += int(target.numel())

    @torch.no_grad()
    def compute(self) -> EvalResult:
        return EvalResult(
            macro_f1=float(self._macro.compute().item()),
            per_class_f1=[float(x) for x in self._per.compute().tolist()],
            ece=float(self._ece.compute().item()),
            confusion=[[int(x) for x in row] for row in self._cm.compute().tolist()],
            n=self._n,
        )
