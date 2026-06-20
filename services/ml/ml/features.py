"""Feature engineering & the persisted preprocessing bundle (plans/03 §1.4, §1.7).

`Preprocess` is fit on the TRAIN months only (the #1 leakage guardrail), then applied to all
splits and reused unchanged at inference. It is serialized as part of `preprocess.pkl` so a
served model can never drift from how it was trained. It owns: per-node-type continuous
scalers + impute means, the region one-hot vocabulary, the actor-id column map (also the
alliance multi-hot columns), the fixed node ordering, and the edge-feature scaler.

The alliance multi-hot itself is built in dataset.py (it needs the temporal MEMBER_OF
validity, which is graph data); this module only owns the column ordering (`actor_ids`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ---- feature groups (names verified against the Go loaders) --------------------------
COUNTRY_CONT = [
    "gdp_log", "gdp_per_capita", "trade_openness_index", "population_log",
    "population_growth", "land_area_log", "political_stability", "vdem_polyarchy_score",
    "years_since_leadership_change", "military_expenditure_log", "hdi",
    "active_conflict_count", "conflict_intensity",
]
COUNTRY_LOG1P = ["active_conflict_count"]  # *_log + conflict_intensity are already log-scaled
COUNTRY_BIN = ["sanctions_status", "unsc_seat_flag", "nuclear_flag", "coastline_flag"]

ACTOR_CONT = ["member_count_log", "recognized_legitimacy_score", "financial_resources_tier"]
ACTOR_LOG1P = ["recognized_legitimacy_score"]

EDGE_CONT = ["event_count", "weighted_intensity", "sentiment_mean", "sentiment_std", "days_since_last_event"]
EDGE_LOG1P = ["event_count", "days_since_last_event"]
EDGE_DIST_LEN = 5  # class_distribution[5]


def _apply_log1p(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = np.log1p(out[c].astype("float64").clip(lower=0))
    return out


def _stack_distribution(series: pd.Series) -> np.ndarray:
    """Turn a parquet list<float>[5] column into a [N, 5] float array (robust to None)."""
    rows = []
    for v in series.tolist():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            rows.append([0.0] * EDGE_DIST_LEN)
        else:
            arr = list(v)
            arr = (arr + [0.0] * EDGE_DIST_LEN)[:EDGE_DIST_LEN]
            rows.append([float(x) for x in arr])
    return np.asarray(rows, dtype="float32")


@dataclass
class Preprocess:
    country_ids: list[str] = field(default_factory=list)   # fixed Country node ordering
    actor_ids: list[str] = field(default_factory=list)     # fixed Actor ordering = alliance columns
    regions: list[str] = field(default_factory=list)       # region one-hot vocabulary
    class_names: list[str] = field(default_factory=list)

    country_means: dict[str, float] = field(default_factory=dict)
    actor_means: dict[str, float] = field(default_factory=dict)
    edge_means: dict[str, float] = field(default_factory=dict)
    country_scaler: StandardScaler | None = None
    actor_scaler: StandardScaler | None = None
    edge_scaler: StandardScaler | None = None

    # ---- dims -----------------------------------------------------------------------
    @property
    def alliance_dim(self) -> int:
        return len(self.actor_ids)

    @property
    def country_block_dim(self) -> int:
        return len(COUNTRY_CONT) + len(COUNTRY_BIN) + len(self.regions)

    @property
    def country_feat_dim(self) -> int:
        return self.country_block_dim + self.alliance_dim  # + alliance multi-hot (added in dataset)

    @property
    def actor_feat_dim(self) -> int:
        return len(ACTOR_CONT)

    @property
    def edge_dim(self) -> int:
        return len(EDGE_CONT) + EDGE_DIST_LEN

    # ---- fit (TRAIN months only) ----------------------------------------------------
    def fit(self, node_df: pd.DataFrame, edge_df: pd.DataFrame, train_max_ts: int) -> "Preprocess":
        countries = node_df[node_df["node_type"] == "Country"]
        actors = node_df[node_df["node_type"] == "Actor"]

        self.country_ids = sorted(countries["node_id"].unique().tolist())
        self.actor_ids = sorted(actors["node_id"].unique().tolist())
        self.regions = sorted(x for x in countries["region"].dropna().unique().tolist() if str(x) != "")

        c_tr = _apply_log1p(countries[countries["ts"] <= train_max_ts], COUNTRY_LOG1P)
        self.country_means = {c: float(c_tr[c].mean()) for c in COUNTRY_CONT if c in c_tr}
        self.country_scaler = StandardScaler().fit(self._cont_matrix(c_tr, COUNTRY_CONT, self.country_means))

        a_tr = _apply_log1p(actors[actors["ts"] <= train_max_ts], ACTOR_LOG1P)
        self.actor_means = {c: float(a_tr[c].mean()) for c in ACTOR_CONT if c in a_tr}
        self.actor_scaler = StandardScaler().fit(self._cont_matrix(a_tr, ACTOR_CONT, self.actor_means))

        e_tr = _apply_log1p(edge_df[edge_df["ts"] <= train_max_ts], EDGE_LOG1P)
        self.edge_means = {c: float(e_tr[c].mean()) for c in EDGE_CONT if c in e_tr}
        self.edge_scaler = StandardScaler().fit(self._cont_matrix(e_tr, EDGE_CONT, self.edge_means))
        return self

    # ---- transforms -----------------------------------------------------------------
    @staticmethod
    def _cont_matrix(df: pd.DataFrame, cols: list[str], means: dict[str, float]) -> np.ndarray:
        out = np.empty((len(df), len(cols)), dtype="float64")
        for j, c in enumerate(cols):
            col = df[c].astype("float64") if c in df.columns else pd.Series(np.nan, index=df.index)
            out[:, j] = col.fillna(means.get(c, 0.0)).to_numpy()
        return out

    def _onehot_region(self, regions: pd.Series) -> np.ndarray:
        idx = {r: i for i, r in enumerate(self.regions)}
        out = np.zeros((len(regions), len(self.regions)), dtype="float32")
        for i, r in enumerate(regions.tolist()):
            j = idx.get(r)
            if j is not None:
                out[i, j] = 1.0
        return out

    def _reindex(self, df_month: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        return df_month.set_index("node_id").reindex(ids)

    def country_block(self, df_month: pd.DataFrame) -> np.ndarray:
        """[Nc, country_block_dim] = scaled continuous | binary | region one-hot. Ordered by
        country_ids; rows absent in df_month are imputed (mean for continuous, 0 for flags)."""
        df = self._reindex(df_month, self.country_ids)
        cont = _apply_log1p(df, COUNTRY_LOG1P)
        cont = self.country_scaler.transform(self._cont_matrix(cont, COUNTRY_CONT, self.country_means))
        binm = np.zeros((len(df), len(COUNTRY_BIN)), dtype="float32")
        for j, c in enumerate(COUNTRY_BIN):
            if c in df.columns:
                binm[:, j] = df[c].fillna(0).astype("float32").to_numpy()
        region = self._onehot_region(df["region"] if "region" in df.columns else pd.Series([None] * len(df)))
        return np.hstack([cont.astype("float32"), binm, region]).astype("float32")

    def actor_block(self, df_month: pd.DataFrame) -> np.ndarray:
        """[Na, actor_feat_dim] = scaled actor continuous features, ordered by actor_ids."""
        df = self._reindex(df_month, self.actor_ids)
        cont = _apply_log1p(df, ACTOR_LOG1P)
        return self.actor_scaler.transform(
            self._cont_matrix(cont, ACTOR_CONT, self.actor_means)
        ).astype("float32")

    def edge_features(self, df_rows: pd.DataFrame) -> np.ndarray:
        """[E, edge_dim] for edge rows already in the desired order (caller controls order).
        = scaled continuous | class_distribution[5]."""
        if len(df_rows) == 0:
            return np.zeros((0, self.edge_dim), dtype="float32")
        cont = _apply_log1p(df_rows, EDGE_LOG1P)
        cont = self.edge_scaler.transform(self._cont_matrix(cont, EDGE_CONT, self.edge_means)).astype("float32")
        dist = (_stack_distribution(df_rows["class_distribution"])
                if "class_distribution" in df_rows.columns
                else np.zeros((len(df_rows), EDGE_DIST_LEN), dtype="float32"))
        return np.hstack([cont, dist]).astype("float32")

    # ---- persistence ----------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Preprocess":
        return joblib.load(path)
