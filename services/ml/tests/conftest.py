"""Test fixtures. SAFETY: every fixture here builds a *tiny synthetic* Parquet dataset in a
pytest tmp dir. No test connects to Neo4j, and nothing here reads, writes, or deletes the
real databases or any project data — so running the suite can never harm the research data
or the running infrastructure. (Synthetic fixtures are test scaffolding, distinct from the
'no synthetic data in the pipeline' rule, which is about training.)
"""

from __future__ import annotations

import os
import sys

import pytest

# make the `ml` / `export` packages importable (services/ml is the parent of tests/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config import CLASS_NAMES, Config  # noqa: E402

COUNTRIES = ["USA", "CHN", "RUS", "UKR", "DEU", "FRA"]
REGIONS = {"USA": "North America", "CHN": "Asia", "RUS": "Europe",
           "UKR": "Europe", "DEU": "Europe", "FRA": "Europe"}
ACTORS = ["Q1065", "Q7184"]
MAX_TS = 20


def _build_parquet(out_dir: str) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    node_rows = []
    for ts in range(MAX_TS + 1):
        for c in COUNTRIES:
            node_rows.append({
                "node_id": c, "node_type": "Country", "ts": ts, "region": REGIONS[c],
                "gdp_log": 25 + rng.normal(), "gdp_per_capita": 20000 + 1000 * rng.normal(),
                "trade_openness_index": 50 + 5 * rng.normal(), "population_log": 18 + rng.normal(),
                "population_growth": rng.normal(), "land_area_log": 14 + rng.normal(),
                "political_stability": rng.normal(), "vdem_polyarchy_score": 0.5 + 0.1 * rng.normal(),
                "years_since_leadership_change": float(ts % 7), "military_expenditure_log": 22 + rng.normal(),
                "hdi": 0.8 + 0.02 * rng.normal(), "active_conflict_count": float(rng.integers(0, 50)),
                "conflict_intensity": float(rng.random() * 5),
                "sanctions_status": int(c == "RUS" and ts >= 12), "unsc_seat_flag": int(c in ("USA", "CHN", "RUS")),
                "nuclear_flag": int(c in ("USA", "CHN", "RUS", "FRA")), "coastline_flag": 1,
            })
        for a in ACTORS:
            node_rows.append({
                "node_id": a, "node_type": "Actor", "ts": ts,
                "member_count_log": 3 + 0.1 * rng.normal(),
                "recognized_legitimacy_score": float(20 + ts), "financial_resources_tier": 4,
            })
    pd.DataFrame(node_rows).to_parquet(os.path.join(out_dir, "node_snapshots.parquet"), index=False)

    pairs = [("USA", "CHN"), ("RUS", "UKR"), ("DEU", "FRA"), ("CHN", "RUS")]
    edge_rows = []
    for ts in range(MAX_TS + 1):
        for j, (s, t) in enumerate(pairs):
            if s == "RUS" and t == "UKR":
                cls = "MATERIAL_CONFLICT" if ts >= 12 else "VERBAL_CONFLICT"
            else:
                cls = CLASS_NAMES[(ts + j) % len(CLASS_NAMES)]
            dist = [0.05] * len(CLASS_NAMES)
            dist[CLASS_NAMES.index(cls)] = 1.0 - 0.05 * (len(CLASS_NAMES) - 1)
            edge_rows.append({
                "src": s, "tgt": t, "ts": ts, "event_count": int(rng.integers(1, 500)),
                "weighted_intensity": float(rng.normal()), "sentiment_mean": float(rng.normal() * 0.3),
                "sentiment_std": float(abs(rng.normal()) * 0.2), "days_since_last_event": int(30 * (ts % 3)),
                "class_distribution": dist, "dominant_class": cls, "data_source": "GDELT",
            })
    pd.DataFrame(edge_rows).to_parquet(os.path.join(out_dir, "snapshot_edges.parquet"), index=False)

    struct_rows = [
        {"rel": "BORDERS", "a": "RUS", "b": "UKR", "a_label": "Country", "b_label": "Country", "start": 0, "end": None},
        {"rel": "BORDERS", "a": "UKR", "b": "RUS", "a_label": "Country", "b_label": "Country", "start": 0, "end": None},
        {"rel": "BORDERS", "a": "DEU", "b": "FRA", "a_label": "Country", "b_label": "Country", "start": 0, "end": None},
        {"rel": "MEMBER_OF", "a": "USA", "b": "Q7184", "a_label": "Country", "b_label": "Actor", "start": 0, "end": None},
        {"rel": "MEMBER_OF", "a": "DEU", "b": "Q7184", "a_label": "Country", "b_label": "Actor", "start": 0, "end": None},
        {"rel": "MEMBER_OF", "a": "FRA", "b": "Q7184", "a_label": "Country", "b_label": "Actor", "start": 0, "end": None},
        {"rel": "MEMBER_OF", "a": "USA", "b": "Q1065", "a_label": "Country", "b_label": "Actor", "start": 0, "end": None},
        {"rel": "MEMBER_OF", "a": "CHN", "b": "Q1065", "a_label": "Country", "b_label": "Actor", "start": 0, "end": None},
        {"rel": "MEMBER_OF", "a": "RUS", "b": "Q7184", "a_label": "Country", "b_label": "Actor", "start": 0, "end": 5},
    ]
    pd.DataFrame(struct_rows).to_parquet(os.path.join(out_dir, "structural_edges.parquet"), index=False)


@pytest.fixture
def tiny_dataset_dir(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    out = tmp_path / "dataset_parquet"
    out.mkdir()
    _build_parquet(str(out))
    return str(out)


@pytest.fixture
def tiny_cfg(tiny_dataset_dir, tmp_path):
    return Config(
        data_dir=tiny_dataset_dir, artifacts_dir=str(tmp_path / "artifacts"),
        max_ts=MAX_TS, min_target_ts=11, train_max_ts=14, val_max_ts=16,
        window=12, k_neg=2, epochs=2, hidden=16, heads=2, id_dim=8, hops=2,
        gru_layers=1, dropout=0.0, seed=0, device="cpu", wandb_mode="disabled",
    )
