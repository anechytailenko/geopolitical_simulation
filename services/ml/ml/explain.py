"""Explainability (plans/03 §3): why the model predicted a class for a Country pair (u,v).

Two complementary methods, implemented to work with this model's custom temporal forward
signature (a window = list[HeteroData]) rather than the standard (x, edge_index) signature:

- `integrated_gradients` — feature-level. Integrates the gradient of the predicted-class
  probability from a zero baseline to the real input, for (a) the (u,v) edge feature vector
  and (b) u's and v's node features in the last window month. Satisfies the completeness
  axiom (sum of attributions ≈ F(x) − F(baseline)); `completeness_gap` reports the residual.
- `gnn_explainer_edge_mask` — structure-level. Learns a soft mask over the last month's
  SNAPSHOT edges (GNNExplainer-style) that preserves the predicted class with a sparsity
  penalty; returns per-edge importances in [0,1].

Both operate on a single pair at a time (the inference path), as the plan requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dataset import C, REL_SNAP


def _clone_window(window: list) -> list:
    return [d.clone() for d in window]


def _prob_for_pair(model, window, u: int, v: int, cls: int) -> torch.Tensor:
    pair = torch.tensor([[u, v]], dtype=torch.long, device=window[0][C].x.device)
    attr = window[-1][REL_SNAP].edge_attr  # placeholder; caller supplies pair_attr separately
    logits = model(window, pair, attr.new_zeros((1, attr.shape[1])))
    return F.softmax(logits, dim=-1)[0, cls]


@dataclass
class IGResult:
    edge_attribution: list[float]      # per dim of the (u,v) edge feature vector
    u_node_attribution: list[float]    # per dim of u's last-month node features
    v_node_attribution: list[float]
    completeness_gap: float            # |sum(attr) - (F(x) - F(baseline))|
    target_class: int


@torch.enable_grad()
def integrated_gradients(model, window: list, u: int, v: int, pair_attr: torch.Tensor,
                         target_class: int | None = None, steps: int = 50) -> IGResult:
    model.eval()
    device = window[0][C].x.device
    pair = torch.tensor([[u, v]], dtype=torch.long, device=device)

    base_attr = torch.zeros_like(pair_attr)
    last_x = window[-1][C].x
    base_x = torch.zeros_like(last_x)

    if target_class is None:
        with torch.no_grad():
            target_class = int(model(window, pair, pair_attr).argmax(dim=-1).item())

    grad_attr = torch.zeros_like(pair_attr)
    grad_x = torch.zeros_like(last_x)
    for k in range(1, steps + 1):
        alpha = k / steps
        a = (base_attr + alpha * (pair_attr - base_attr)).detach().requires_grad_(True)
        w = _clone_window(window)
        x = (base_x + alpha * (last_x - base_x)).detach().requires_grad_(True)
        w[-1][C].x = x
        p = F.softmax(model(w, pair, a), dim=-1)[0, target_class]
        ga, gx = torch.autograd.grad(p, [a, x], retain_graph=False)
        grad_attr += ga.detach()
        grad_x += gx.detach()

    ig_attr = ((pair_attr - base_attr) * grad_attr / steps).squeeze(0)
    ig_x = (last_x - base_x) * grad_x / steps

    with torch.no_grad():
        f_x = float(F.softmax(model(window, pair, pair_attr), dim=-1)[0, target_class])
        w0 = _clone_window(window)
        w0[-1][C].x = base_x
        f_base = float(F.softmax(model(w0, pair, base_attr), dim=-1)[0, target_class])
    # Completeness is over ALL perturbed inputs: the (u,v) edge vector + EVERY last-month
    # node feature interpolated from the zero baseline (not just u/v, which are reported only
    # for interpretability). Summing just u/v would leave a large spurious residual.
    total = float(ig_attr.sum() + ig_x.sum())
    return IGResult(
        edge_attribution=ig_attr.tolist(),
        u_node_attribution=ig_x[u].tolist(),
        v_node_attribution=ig_x[v].tolist(),
        completeness_gap=abs(total - (f_x - f_base)),
        target_class=target_class,
    )


@torch.enable_grad()
def gnn_explainer_edge_mask(model, window: list, u: int, v: int, pair_attr: torch.Tensor,
                            target_class: int | None = None, epochs: int = 200,
                            lr: float = 0.01, coeff_size: float = 1e-3) -> list[float]:
    """Learn a sigmoid mask over the last-month SNAPSHOT edges that preserves the predicted
    class (GNNExplainer-style). Returns per-edge importance in [0,1]."""
    model.eval()
    device = window[0][C].x.device
    pair = torch.tensor([[u, v]], dtype=torch.long, device=device)
    base_attr = window[-1][REL_SNAP].edge_attr
    n_edges = base_attr.shape[0]
    if n_edges == 0:
        return []

    if target_class is None:
        with torch.no_grad():
            target_class = int(model(window, pair, pair_attr).argmax(dim=-1).item())

    mask = torch.zeros(n_edges, device=device, requires_grad=True)
    opt = torch.optim.Adam([mask], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        w = _clone_window(window)
        m = torch.sigmoid(mask).unsqueeze(1)
        w[-1][REL_SNAP].edge_attr = base_attr * m
        logp = F.log_softmax(model(w, pair, pair_attr), dim=-1)[0, target_class]
        loss = -logp + coeff_size * torch.sigmoid(mask).mean()
        loss.backward()
        opt.step()
    return torch.sigmoid(mask).detach().cpu().tolist()
