"""Explanation enrichment (plans/04 §3, the two REQUIRED additive enrichments).

``Predictor.explain`` returns a *bare* GNNExplainer edge-importance vector plus Integrated-
Gradients attributions for only the focus pair's ``u`` and ``v``. Neither is directly
renderable. This module turns them into the left-panel payload:

1. **Subgraph edges** — map ``gnn_explainer_edge_importance[i] <-> ds.snap_pairs[T][i]`` to
   ISO-3 and read each edge's ``dominant_class@T`` from its ``class_distribution`` block.
2. **Named feature attributions** — label the IG vectors with the ordered feature names from
   the ``Preprocess`` bundle, so the popup shows ``military_expenditure_log +0.21`` not an index.

IG scope (plans/04 §3 A1): feature attributions exist **only** for the focus pair's source &
target Countries; every other subgraph node carries structural (GNNExplainer) importance only.
"""

from __future__ import annotations

import os

import numpy as np

from ml.config import CLASS_NAMES
from ml.explain import gnn_explainer_edge_mask, integrated_gradients
from ml.features import COUNTRY_BIN, COUNTRY_CONT, EDGE_CONT

from .groups import QID_TO_GROUP


class Explainer:
    def __init__(self, predictor, gnn_epochs: int | None = None, ig_steps: int | None = None):
        self.p = predictor
        self.ds = predictor.ds
        self.pp = predictor.ds.pp
        # Cost knobs (plans/04 §10/§12): the deterministic viz node explains every answer, so the
        # 200-epoch / 50-step defaults of ml.explain are too slow here. Tunable via env; tests
        # pass tiny values. Lower epochs/steps trade a little mask sharpness for latency.
        self.gnn_epochs = gnn_epochs if gnn_epochs is not None else int(os.environ.get("GNN_EXPLAINER_EPOCHS", 64))
        self.ig_steps = ig_steps if ig_steps is not None else int(os.environ.get("IG_STEPS", 24))

    # ---- feature-name vocabularies (match the Preprocess ordering) -------------------
    def country_feature_names(self) -> list[str]:
        names = list(COUNTRY_CONT) + list(COUNTRY_BIN)
        names += [f"region={r}" for r in self.pp.regions]
        names += [f"member_of={QID_TO_GROUP.get(q, q)}" for q in self.pp.actor_ids]
        return names

    def edge_feature_names(self) -> list[str]:
        return list(EDGE_CONT) + [f"class_dist={c}" for c in CLASS_NAMES]

    # ---- the enriched explanation ---------------------------------------------------
    def explain(self, source_id: str, target_id: str, time_step: int,
                top_k_edges: int = 12, top_k_features: int = 8) -> dict:
        # Reuse the exact ml.explain algorithms (same model, same inference path) but with the
        # agent's faster epoch/step budget — no change to the shared ml/ code.
        u, v = self.p._indices(source_id, target_id)
        window = [d.to(self.p.device) for d in self.ds.build_window(time_step)]
        attr = self.p._pair_attr(time_step, u, v)
        ig = integrated_gradients(self.p.model, window, u, v, attr, steps=self.ig_steps)
        edge_imp = gnn_explainer_edge_mask(self.p.model, window, u, v, attr,
                                           target_class=ig.target_class, epochs=self.gnn_epochs)

        subgraph = self._subgraph(source_id, target_id, time_step, edge_imp, top_k_edges)
        feats = {
            source_id: self._top_features(self.country_feature_names(),
                                          ig.u_node_attribution, top_k_features),
            target_id: self._top_features(self.country_feature_names(),
                                          ig.v_node_attribution, top_k_features),
        }
        return {
            "target_class": CLASS_NAMES[ig.target_class],
            "subgraph": subgraph,
            "feature_attributions": feats,
            "edge_feature_attribution": self._top_features(
                self.edge_feature_names(), ig.edge_attribution, top_k_features),
            "integrated_gradients_completeness_gap": ig.completeness_gap,
        }

    # ---- helpers --------------------------------------------------------------------
    def _subgraph(self, source_id: str, target_id: str, time_step: int,
                  importance: list[float], top_k: int) -> dict:
        cids = self.pp.country_ids
        pairs = self.ds.snap_pairs.get(time_step, [])
        attr = self.ds.snap_attr.get(time_step)
        n = min(len(importance), len(pairs))

        edges = []
        for i in range(n):
            u_idx, v_idx = pairs[i]
            dom = CLASS_NAMES[int(np.asarray(attr[i][-len(CLASS_NAMES):]).argmax())] if attr is not None else None
            edges.append({
                "src": cids[u_idx], "tgt": cids[v_idx],
                "dominant_class": dom, "importance": float(importance[i]),
            })
        # keep the focus edge (if present) + the top-k most important others
        focus = {(source_id, target_id), (target_id, source_id)}
        edges.sort(key=lambda e: e["importance"], reverse=True)
        kept, seen = [], set()
        for e in edges:
            key = (e["src"], e["tgt"])
            if key in focus or len(kept) < top_k:
                if key not in seen:
                    kept.append(e); seen.add(key)

        # node importance = max incident edge importance; focus pair pinned to 1.0
        node_imp: dict[str, float] = {}
        for e in kept:
            node_imp[e["src"]] = max(node_imp.get(e["src"], 0.0), e["importance"])
            node_imp[e["tgt"]] = max(node_imp.get(e["tgt"], 0.0), e["importance"])
        node_imp[source_id] = 1.0
        node_imp[target_id] = 1.0
        nodes = [{
            "id": cid,
            "name": self._name(cid),
            "type": "Country",
            "importance": imp,
            "ig_clickable": cid in (source_id, target_id),
        } for cid, imp in sorted(node_imp.items(), key=lambda kv: kv[1], reverse=True)]
        return {"nodes": nodes, "edges": kept}

    @staticmethod
    def _name(iso3: str) -> str:
        from .places import PlaceResolver
        return PlaceResolver.country_name(iso3)

    @staticmethod
    def _top_features(names: list[str], values: list[float], k: int) -> list[dict]:
        m = min(len(names), len(values))
        rows = [{"feature": names[i], "attribution": float(values[i])} for i in range(m)]
        rows.sort(key=lambda r: abs(r["attribution"]), reverse=True)
        return rows[:k]
