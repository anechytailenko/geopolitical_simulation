"""Inference (plans/03 §Model Input/Output, §3). Loads the trained model + the persisted
preprocess bundle + temperature calibrator, rebuilds the live temporal window from the
aggregated Parquet export, and predicts the 5-class distribution for a directed Country pair
at a target month. Reuses the exact same model.py / dataset.py / features.py as training, so
serving can never drift from how it was trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .calibrate import TemperatureScaler
from .config import Config
from .dataset import GeopoliticDataset
from .explain import gnn_explainer_edge_mask, integrated_gradients
from .features import Preprocess
from .model import SpatioTemporalEdgeClassifier


@dataclass
class Prediction:
    source_id: str
    target_id: str
    time_step: int
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float


class Predictor:
    def __init__(self, cfg: Config, checkpoint: str = "best.pt"):
        import os
        self.cfg = cfg
        device = torch.device("cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
        self.device = device

        pp = Preprocess.load(os.path.join(cfg.artifacts_dir, "preprocess.pkl"))
        self.ds = GeopoliticDataset.from_parquet(cfg, preprocess=pp)

        ckpt_path = os.path.join(cfg.artifacts_dir, checkpoint)
        # weights_only=False: checkpoints embed the Preprocess bundle (trusted, self-written).
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = SpatioTemporalEdgeClassifier.from_dataset(self.ds, cfg).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        temp = ckpt.get("calibrator_temperature")
        if temp is None:
            try:
                self.calibrator = TemperatureScaler.load(os.path.join(cfg.artifacts_dir, "calibrator.pkl"))
            except Exception:
                self.calibrator = TemperatureScaler(1.0)
        else:
            self.calibrator = TemperatureScaler(temp)

    def _indices(self, source_id: str, target_id: str) -> tuple[int, int]:
        u = self.ds.country_index.get(source_id)
        v = self.ds.country_index.get(target_id)
        if u is None or v is None:
            raise KeyError(f"unknown country id(s): {source_id!r}/{target_id!r}")
        return u, v

    @torch.no_grad()
    def predict(self, source_id: str, target_id: str, time_step: int) -> Prediction:
        u, v = self._indices(source_id, target_id)
        window = [d.to(self.device) for d in self.ds.build_window(time_step)]
        pair = torch.tensor([[u, v]], dtype=torch.long, device=self.device)
        attr = self._pair_attr(time_step, u, v)
        logits = self.model(window, pair, attr)
        probs = self.calibrator.probs(logits.cpu())[0]
        names = self.cfg.class_names
        idx = int(probs.argmax())
        return Prediction(
            source_id=source_id, target_id=target_id, time_step=time_step,
            probabilities={n: float(p) for n, p in zip(names, probs.tolist())},
            predicted_class=names[idx], confidence=float(probs[idx]),
        )

    @torch.no_grad()
    def predict_batch(self, pairs: list[tuple[str, str]], time_step: int) -> list[Prediction]:
        if not pairs:
            return []
        window = [d.to(self.device) for d in self.ds.build_window(time_step)]
        idx_pairs, attrs = [], []
        for s, t in pairs:
            u, v = self._indices(s, t)
            idx_pairs.append((u, v))
            attrs.append(self._pair_attr(time_step, u, v))
        pair = torch.tensor(idx_pairs, dtype=torch.long, device=self.device)
        attr = torch.cat(attrs, dim=0)
        probs = self.calibrator.probs(self.model(window, pair, attr).cpu())
        names = self.cfg.class_names
        out = []
        for (s, t), p in zip(pairs, probs):
            i = int(p.argmax())
            out.append(Prediction(s, t, time_step,
                                  {n: float(x) for n, x in zip(names, p.tolist())}, names[i], float(p[i])))
        return out

    def explain(self, source_id: str, target_id: str, time_step: int) -> dict:
        u, v = self._indices(source_id, target_id)
        window = [d.to(self.device) for d in self.ds.build_window(time_step)]
        attr = self._pair_attr(time_step, u, v)
        ig = integrated_gradients(self.model, window, u, v, attr)
        edge_mask = gnn_explainer_edge_mask(self.model, window, u, v, attr, target_class=ig.target_class)
        return {
            "target_class": self.cfg.class_names[ig.target_class],
            "integrated_gradients": {
                "edge_feature_attribution": ig.edge_attribution,
                "source_node_attribution": ig.u_node_attribution,
                "target_node_attribution": ig.v_node_attribution,
                "completeness_gap": ig.completeness_gap,
            },
            "gnn_explainer_edge_importance": edge_mask,
        }

    def _pair_attr(self, time_step: int, u: int, v: int) -> torch.Tensor:
        lookup = self.ds._pair_lookup(time_step)
        hit = lookup.get((u, v))
        edim = self.ds.pp.edge_dim
        if hit is None:
            return torch.zeros((1, edim), dtype=torch.float32, device=self.device)
        return torch.tensor(hit, dtype=torch.float32, device=self.device).unsqueeze(0)
