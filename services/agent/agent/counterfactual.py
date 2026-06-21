"""Single-step counterfactual (plans/04 §1 Type 6, §3 ``predict_counterfactual``).

"If A and B sign a deal (-> MATERIAL_COOPERATION), what happens between B and C next month?"
We edit **one month-T input edge** and re-run the forward pass, reading the unchanged T+1 query.
This stays inside the model's trained one-step horizon: inputs <= T -> T+1, no autoregressive
rollout, no synthesis of future snapshots. The intervened edge's 10-dim feature vector is built
through the **same** ``edge_scaler`` the model trained with (never refit), and the edit is local
to the call (the dataset's cached tensors are never mutated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ml.config import CLASS_NAMES
from ml.dataset import REL_SNAP

# Per-class "active event" prior for the intervened edge (raw, pre-scaling). Sign follows the
# CAMEO/Goldstein convention: cooperation positive, conflict negative; material > verbal.
_PRIOR = {
    "MATERIAL_CONFLICT":    dict(event_count=120.0, weighted_intensity=-8.0, sentiment_mean=-3.0),
    "VERBAL_CONFLICT":      dict(event_count=40.0,  weighted_intensity=-3.0, sentiment_mean=-1.5),
    "MATERIAL_COOPERATION": dict(event_count=120.0, weighted_intensity=8.0,  sentiment_mean=3.0),
    "VERBAL_COOPERATION":   dict(event_count=40.0,  weighted_intensity=3.0,  sentiment_mean=1.5),
    "STATUS_QUO":           dict(event_count=5.0,   weighted_intensity=0.0,  sentiment_mean=0.0),
}


def _intervention_vector(predictor, intervene_class: str) -> np.ndarray:
    """[1, edge_dim] standardized feature vector for an edge whose dominant class is
    ``intervene_class`` (continuous fields scaled by the trained edge_scaler)."""
    pr = _PRIOR[intervene_class]
    onehot = [1.0 if c == intervene_class else 0.0 for c in CLASS_NAMES]
    row = {
        "event_count": pr["event_count"],
        "weighted_intensity": pr["weighted_intensity"],
        "sentiment_mean": pr["sentiment_mean"],
        "sentiment_std": 1.0,
        "days_since_last_event": 1.0,
        "class_distribution": onehot,
    }
    return predictor.ds.pp.edge_features(pd.DataFrame([row]))  # [1, 10] float32


def _edit_window(predictor, window: list, T_local_idx: int, edges: list[tuple[int, int]],
                 vec: np.ndarray) -> list:
    """Return a copy of ``window`` whose month at ``T_local_idx`` has ``edges`` inserted/updated
    in the SNAPSHOT relation with feature row ``vec`` — without mutating any cached tensor."""
    d = window[T_local_idx]
    device = predictor.device
    ei = d[REL_SNAP].edge_index
    ea = d[REL_SNAP].edge_attr
    new_ei = ei.clone()
    new_ea = ea.clone()
    row = torch.from_numpy(np.asarray(vec, dtype="float32"))[0].to(device)

    col_of = {(u, v): i for i, (u, v) in enumerate(ei.t().tolist())}
    add_pairs, add_rows = [], []
    for (pu, pv) in edges:
        if (pu, pv) in col_of:
            new_ea[col_of[(pu, pv)]] = row
        else:
            add_pairs.append((pu, pv)); add_rows.append(row)
    if add_pairs:
        add_ei = torch.tensor(add_pairs, dtype=torch.long, device=device).t().contiguous()
        new_ei = torch.cat([new_ei, add_ei], dim=1)
        new_ea = torch.cat([new_ea, torch.stack(add_rows, dim=0)], dim=0)

    d2 = d.clone()
    d2[REL_SNAP].edge_index = new_ei
    d2[REL_SNAP].edge_attr = new_ea
    return window[:T_local_idx] + [d2] + window[T_local_idx + 1:]


@torch.no_grad()
def predict_counterfactual(predictor, intervene_source: str, intervene_target: str,
                           intervene_class: str, query_source: str, query_target: str,
                           time_step: int, symmetric: bool = True) -> dict:
    iu, iv = predictor._indices(intervene_source, intervene_target)
    qu, qv = predictor._indices(query_source, query_target)
    names = predictor.cfg.class_names

    window = [d.to(predictor.device) for d in predictor.ds.build_window(time_step)]
    query_pair = torch.tensor([[qu, qv]], dtype=torch.long, device=predictor.device)
    query_attr = predictor._pair_attr(time_step, qu, qv)

    base_probs = predictor.calibrator.probs(predictor.model(window, query_pair, query_attr).cpu())[0]

    edges = [(iu, iv)] + ([(iv, iu)] if symmetric and iu != iv else [])
    vec = _intervention_vector(predictor, intervene_class)
    edited = _edit_window(predictor, window, len(window) - 1, edges, vec)
    cf_probs = predictor.calibrator.probs(predictor.model(edited, query_pair, query_attr).cpu())[0]

    def dist(p):
        i = int(p.argmax())
        return {"probabilities": {n: float(x) for n, x in zip(names, p.tolist())},
                "predicted_class": names[i], "confidence": float(p[i])}

    base, cf = dist(base_probs), dist(cf_probs)
    delta = {n: cf["probabilities"][n] - base["probabilities"][n] for n in names}
    return {
        "time_step": time_step,
        "intervened_edge": {"src": intervene_source, "tgt": intervene_target,
                            "class": intervene_class, "symmetric": symmetric},
        "focus_pair": {"src": query_source, "tgt": query_target,
                       "predicted_class": cf["predicted_class"], "confidence": cf["confidence"],
                       "probabilities": cf["probabilities"]},
        "baseline": base,
        "counterfactual": cf,
        "delta": delta,
    }
