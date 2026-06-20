"""Probability calibration (plans/03 §2.5). Temperature scaling: a single scalar T learned
on the validation set that divides the logits before softmax, minimizing NLL. Cheap, keeps
the argmax unchanged, and reliably lowers ECE. Serialized inside calibrator.pkl.
"""

from __future__ import annotations

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(temperature)))))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(logits), dim=-1)

    def fit(self, logits: torch.Tensor, target: torch.Tensor, max_iter: int = 100) -> "TemperatureScaler":
        """Learn T by minimizing NLL on held-out (val) logits with LBFGS."""
        logits = logits.detach()
        target = target.detach()
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), target)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def save(self, path: str) -> None:
        joblib.dump({"temperature": self.temperature}, path)

    @staticmethod
    def load(path: str) -> "TemperatureScaler":
        return TemperatureScaler(joblib.load(path)["temperature"])
