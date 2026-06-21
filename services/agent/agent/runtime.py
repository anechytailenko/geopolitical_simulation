"""Runtime: load the trained model + place resolver once, with a fail-fast boot self-check.

This is the single place that constructs ``ml.infer.Predictor`` (which loads ``best.pt`` +
``preprocess.pkl`` + ``calibrator.pkl`` from ``GEO_ARTIFACTS_DIR`` and the parquet dataset from
``GEO_DATA_DIR``). The boot self-check (plans/04 §11) guarantees the agent never serves a
mis-located or class-misaligned model.

The agent is **read-only**: it loads the parquet export and the trained weights, and never
opens a writable database connection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from ml.config import CLASS_NAMES, Config
from ml.infer import Predictor

from .config import default_artifacts_dir, default_data_dir, load_dotenv_files
from .explainer import Explainer
from .places import PlaceResolver


def _resolve_paths(cfg: Config) -> Config:
    """Make the agent runnable from any CWD: if GEO_DATA_DIR / GEO_ARTIFACTS_DIR are unset or
    don't actually contain the files, fall back to the bundle shipped in the repo (plans/04 §11).
    Avoids the classic `FileNotFoundError: dataset_parquet/node_snapshots.parquet` when launched
    from the repo root without exporting the env vars."""
    if not os.path.exists(os.path.join(cfg.data_dir, "node_snapshots.parquet")):
        cand = default_data_dir()
        if os.path.exists(os.path.join(cand, "node_snapshots.parquet")):
            cfg.data_dir = cand
    if not os.path.exists(os.path.join(cfg.artifacts_dir, "best.pt")):
        cand = default_artifacts_dir()
        if os.path.exists(os.path.join(cand, "best.pt")):
            cfg.artifacts_dir = cand
    return cfg


@dataclass
class Runtime:
    cfg: Config
    predictor: Predictor
    resolver: PlaceResolver
    explainer: Explainer

    @property
    def max_ts(self) -> int:
        return self.cfg.max_ts

    @property
    def min_ts(self) -> int:
        return self.cfg.min_target_ts

    @property
    def class_names(self) -> list[str]:
        return list(self.cfg.class_names)

    @property
    def country_ids(self) -> set[str]:
        return set(self.predictor.ds.country_index.keys())

    # ---- construction ---------------------------------------------------------------
    @classmethod
    def build(cls, cfg: Config | None = None, self_check: bool = True) -> "Runtime":
        if cfg is None:
            load_dotenv_files()            # pick up .env (repo root / services/agent), existing env wins
            cfg = Config.from_env()
        _resolve_paths(cfg)                # auto-discover the repo bundle if paths are unset/invalid
        predictor = Predictor(cfg)

        struct_path = os.path.join(cfg.data_dir, "structural_edges.parquet")
        member_df = pd.read_parquet(struct_path)
        resolver = PlaceResolver(set(predictor.ds.country_index.keys()), member_df)
        explainer = Explainer(predictor)

        rt = cls(cfg=cfg, predictor=predictor, resolver=resolver, explainer=explainer)
        if self_check:
            rt.boot_self_check()
        return rt

    def boot_self_check(self) -> None:
        """plans/04 §11: class order matches Go/Python canon and a probe prediction is valid."""
        pp_classes = list(self.predictor.ds.pp.class_names)
        if pp_classes != list(CLASS_NAMES):
            raise RuntimeError(
                f"class-order mismatch: preprocess {pp_classes} != CLASS_NAMES {list(CLASS_NAMES)}")

        ids = self.country_ids
        a, b = ("USA", "CHN") if {"USA", "CHN"} <= ids else tuple(sorted(ids)[:2])
        pred = self.predictor.predict(a, b, self.max_ts)
        s = sum(pred.probabilities.values())
        if abs(s - 1.0) > 1e-5:
            raise RuntimeError(f"probe prediction probabilities sum to {s}, expected 1.0")
        if set(pred.probabilities.keys()) != set(CLASS_NAMES):
            raise RuntimeError("probe prediction class names do not match CLASS_NAMES")


_RUNTIME: Runtime | None = None


def get_runtime() -> Runtime:
    """Process-wide singleton (built on first use)."""
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = Runtime.build()
    return _RUNTIME
