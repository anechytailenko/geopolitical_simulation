"""Central configuration for the ML stage. Dependency-free (stdlib only) so it is
importable on any machine and in unit tests without torch installed.

CLASS_NAMES is the single source of truth on the Python side and MUST stay byte-for-byte
identical to `internal/label.Classes` in the Go code (the index of each class is persisted
in `class_distribution` on every SNAPSHOT_EDGE). test_config.py asserts this against the Go
source so the two can never silently drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

# ---- canonical class order (mirror of internal/label/cameo.go `Classes`) -------------
CLASS_NAMES: list[str] = [
    "MATERIAL_CONFLICT",
    "VERBAL_CONFLICT",
    "MATERIAL_COOPERATION",
    "VERBAL_COOPERATION",
    "STATUS_QUO",
]
NUM_CLASSES = len(CLASS_NAMES)
STATUS_QUO_INDEX = CLASS_NAMES.index("STATUS_QUO")


def class_index(name: str) -> int:
    """Canonical index of a class name, or -1 if unknown (mirrors label.Index)."""
    try:
        return CLASS_NAMES.index(name)
    except ValueError:
        return -1


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default


@dataclass
class Config:
    """All knobs for export → dataset → train → calibrate → serve.

    Time convention (mirror of internal/timestep): time_step = (year-2010)*12 + (month-1),
    t=0 -> 2010-01. The aggregated DB spans ts 0..max_ts (197 = 2026-06).
    """

    # ---- paths ----
    data_dir: str = "dataset_parquet"     # holds node_snapshots/ snapshot_edges/ structural_edges.parquet
    artifacts_dir: str = "artifacts"      # best.pt / last.pt / preprocess.pkl / calibrator.pkl / metrics.json

    # ---- time / windowing ----
    window: int = 12                      # W: rolling input window in months
    max_ts: int = 197                     # MaxTimeStep present in the aggregated DB (2026-06)
    min_target_ts: int = 11               # smallest T with a full 12-month window
    # chronological split by TARGET month T (label is at T+1); see plans/03 §1.6
    train_max_ts: int = 172               # train: 11..172
    val_max_ts: int = 184                 # val:   173..184
    # test: (val_max_ts+1) .. (max_ts-1) = 185..196  (T+1 must be <= max_ts)

    # ---- sampling / imbalance ----
    k_neg: int = 5                        # STATUS_QUO negatives per positive Country->Country edge
    loss: str = "weighted_ce"            # "weighted_ce" | "focal"
    focal_gamma: float = 2.0

    # ---- model ----
    hidden: int = 128                     # d
    heads: int = 4                        # H (attention heads); hidden must be divisible by heads
    id_dim: int = 64                      # identity embedding width
    hops: int = 2                         # spatial message-passing hops
    gru_layers: int = 1
    dropout: float = 0.3

    # ---- optimization ----
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 5                     # early-stopping patience on val macro-F1
    seed: int = 42
    device: str = "auto"                  # "auto" | "cpu" | "cuda"

    # ---- experiment tracking (Weights & Biases) ----
    wandb_entity: str = "anna-nechytailenko-kyiv-school-of-economics"
    wandb_project: str = "geosimulation"
    wandb_mode: str = "online"            # "online" | "offline" | "disabled"
    run_name: str = ""

    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))

    # -------------------------------------------------------------------------
    @property
    def test_min_ts(self) -> int:
        return self.val_max_ts + 1

    @property
    def test_max_ts(self) -> int:
        # T+1 must exist (<= max_ts), so the last usable target month is max_ts-1.
        return self.max_ts - 1

    def split_of(self, target_ts: int) -> str:
        """Return 'train' | 'val' | 'test' for a target month T (or 'none')."""
        if target_ts < self.min_target_ts or target_ts > self.test_max_ts:
            return "none"
        if target_ts <= self.train_max_ts:
            return "train"
        if target_ts <= self.val_max_ts:
            return "val"
        return "test"

    def target_months(self, split: str) -> list[int]:
        return [t for t in range(self.min_target_ts, self.test_max_ts + 1) if self.split_of(t) == split]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config, overriding selected fields from environment variables.

        On Kaggle the data lives under /kaggle/input/<dataset>/; set GEO_DATA_DIR there.
        """
        c = cls()
        c.data_dir = os.environ.get("GEO_DATA_DIR", c.data_dir)
        c.artifacts_dir = os.environ.get("GEO_ARTIFACTS_DIR", c.artifacts_dir)
        c.epochs = _env_int("GEO_EPOCHS", c.epochs)
        c.patience = _env_int("GEO_PATIENCE", c.patience)
        c.seed = _env_int("GEO_SEED", c.seed)
        c.k_neg = _env_int("GEO_K_NEG", c.k_neg)
        c.lr = _env_float("GEO_LR", c.lr)
        c.loss = os.environ.get("GEO_LOSS", c.loss)
        c.device = os.environ.get("GEO_DEVICE", c.device)
        c.wandb_entity = os.environ.get("WANDB_ENTITY", c.wandb_entity)
        c.wandb_project = os.environ.get("WANDB_PROJECT", c.wandb_project)
        c.wandb_mode = os.environ.get("WANDB_MODE", c.wandb_mode)
        c.run_name = os.environ.get("WANDB_RUN_NAME", c.run_name)
        return c
