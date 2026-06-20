"""Dataset builder: Parquet export -> per-month HeteroData windows + labeled, negative-
sampled Country->Country target pairs (plans/03 §1.3–§1.6).

Design (the "full-graph-per-month" regime, plans/03 §4.1): for every month ts we build one
small heterogeneous graph (Country + Actor nodes, SNAPSHOT/BORDERS/MEMBER_OF edges). A
training/eval *sample* for target month T is the window of W=12 consecutive monthly graphs
[T-11 .. T] plus the set of (u,v) Country pairs whose label is the dominant_class at T+1,
augmented with K negative STATUS_QUO pairs.

Leakage controls: the Preprocess scaler is fit on TRAIN months only; a sample at T only ever
reads ts <= T for inputs (its one allowed peek at the future is the T+1 label); val/test
negatives are frozen with a fixed seed while train negatives are resampled each epoch.

NOTE on storage: the exporter writes three single Parquet files (node_snapshots.parquet,
snapshot_edges.parquet, structural_edges.parquet). On the ~232-node graph these load fully
into RAM; the per-month windowing happens in-memory, so the plan's ts-partitioning is an
optional optimization we don't need here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .config import Config, STATUS_QUO_INDEX, class_index
from .features import Preprocess

C, A = "Country", "Actor"
REL_SNAP = (C, "snapshot", C)
REL_BORDER = (C, "borders", C)
REL_MEMBER = (C, "member", A)
REL_RMEMBER = (A, "rev_member", C)


@dataclass
class Sample:
    """One training/eval example for target month T."""
    target_ts: int
    pair_index: torch.Tensor   # [P, 2] long, Country local indices (u, v)
    pair_attr: torch.Tensor    # [P, edge_dim] float, (u,v) edge features at month T (or zeros)
    labels: torch.Tensor       # [P] long, class index at T+1


def _valid_at(start: np.ndarray, end: np.ndarray, ts: int) -> np.ndarray:
    """Point-in-time validity predicate: start <= ts AND (end is null OR end > ts)."""
    s = np.nan_to_num(start, nan=0.0)
    e = end.astype("float64")
    return (s <= ts) & (np.isnan(e) | (e > ts))


class GeopoliticDataset:
    def __init__(self, cfg: Config, node_df: pd.DataFrame, edge_df: pd.DataFrame,
                 struct_df: pd.DataFrame, preprocess: Preprocess | None = None):
        self.cfg = cfg
        self.node_df = node_df
        self.edge_df = edge_df
        self.struct_df = struct_df

        if preprocess is None:
            preprocess = Preprocess(class_names=list(cfg.class_names))
            preprocess.fit(node_df, edge_df, cfg.train_max_ts)
        self.pp = preprocess

        self.country_index = {cid: i for i, cid in enumerate(self.pp.country_ids)}
        self.actor_index = {aid: i for i, aid in enumerate(self.pp.actor_ids)}
        self.num_country = len(self.pp.country_ids)
        self.num_actor = len(self.pp.actor_ids)

        self._build_per_ts_tensors()
        self._build_label_index()
        self._pair_cache: dict[int, dict[tuple[int, int], np.ndarray]] = {}

    # ---- construction ---------------------------------------------------------------
    def _build_per_ts_tensors(self) -> None:
        cfg = self.cfg
        nd = self.node_df
        # group node rows by ts for fast per-month blocks
        nd_country = {ts: g for ts, g in nd[nd["node_type"] == C].groupby("ts")}
        nd_actor = {ts: g for ts, g in nd[nd["node_type"] == A].groupby("ts")}

        # alliance multi-hot per ts from MEMBER_OF validity
        mem = self.struct_df[self.struct_df["rel"] == "MEMBER_OF"]
        mem_a = mem["a"].to_numpy(); mem_b = mem["b"].to_numpy()
        mem_start = mem["start"].to_numpy(dtype="float64")
        mem_end = mem["end"].to_numpy(dtype="float64")

        bord = self.struct_df[self.struct_df["rel"] == "BORDERS"]
        b_a = bord["a"].to_numpy(); b_b = bord["b"].to_numpy()
        b_start = bord["start"].to_numpy(dtype="float64")
        b_end = bord["end"].to_numpy(dtype="float64")

        ed_by_ts = {ts: g for ts, g in self.edge_df.groupby("ts")}

        self.country_x: dict[int, torch.Tensor] = {}
        self.actor_x: dict[int, torch.Tensor] = {}
        self.snap_index: dict[int, torch.Tensor] = {}
        self.snap_attr: dict[int, torch.Tensor] = {}
        self.snap_pairs: dict[int, list[tuple[int, int]]] = {}
        self.border_index: dict[int, torch.Tensor] = {}
        self.member_index: dict[int, torch.Tensor] = {}

        empty2 = torch.zeros((2, 0), dtype=torch.long)
        for ts in range(cfg.max_ts + 1):
            # ---- node features
            cdf = nd_country.get(ts, nd_country.get(max((t for t in nd_country if t <= ts), default=ts)))
            cblock = self.pp.country_block(cdf if cdf is not None else pd.DataFrame(columns=nd.columns))
            alliance = self._alliance_multihot(ts, mem_a, mem_b, mem_start, mem_end)
            self.country_x[ts] = torch.from_numpy(np.hstack([cblock, alliance]).astype("float32"))

            adf = nd_actor.get(ts, nd_actor.get(max((t for t in nd_actor if t <= ts), default=ts)))
            ablock = self.pp.actor_block(adf if adf is not None else pd.DataFrame(columns=nd.columns))
            self.actor_x[ts] = torch.from_numpy(ablock.astype("float32"))

            # ---- snapshot (event) edges at ts
            edf = ed_by_ts.get(ts)
            if edf is not None and len(edf):
                pairs, attr = self._edges_to_tensors(edf)
                self.snap_index[ts] = pairs
                self.snap_attr[ts] = attr
                self.snap_pairs[ts] = list(map(tuple, pairs.t().tolist()))
            else:
                self.snap_index[ts] = empty2.clone()
                self.snap_attr[ts] = torch.zeros((0, self.pp.edge_dim), dtype=torch.float32)
                self.snap_pairs[ts] = []

            # ---- structural edges valid at ts
            self.border_index[ts] = self._struct_to_index(b_a, b_b, _valid_at(b_start, b_end, ts),
                                                           self.country_index, self.country_index)
            self.member_index[ts] = self._struct_to_index(mem_a, mem_b, _valid_at(mem_start, mem_end, ts),
                                                           self.country_index, self.actor_index)

    def _alliance_multihot(self, ts, a, b, start, end) -> np.ndarray:
        out = np.zeros((self.num_country, self.num_actor), dtype="float32")
        valid = _valid_at(start, end, ts)
        for i in np.nonzero(valid)[0]:
            ci = self.country_index.get(a[i]); ai = self.actor_index.get(b[i])
            if ci is not None and ai is not None:
                out[ci, ai] = 1.0
        return out

    def _edges_to_tensors(self, edf: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        keep = []
        for pos, (s, t) in enumerate(zip(edf["src"].tolist(), edf["tgt"].tolist())):
            si = self.country_index.get(s); ti = self.country_index.get(t)
            if si is not None and ti is not None:
                rows.append((si, ti)); keep.append(pos)
        if not rows:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, self.pp.edge_dim), dtype=torch.float32)
        attr = self.pp.edge_features(edf.iloc[keep])
        index = torch.tensor(rows, dtype=torch.long).t().contiguous()
        return index, torch.from_numpy(attr)

    @staticmethod
    def _struct_to_index(a, b, valid, src_index, dst_index) -> torch.Tensor:
        rows = []
        for i in np.nonzero(valid)[0]:
            si = src_index.get(a[i]); di = dst_index.get(b[i])
            if si is not None and di is not None:
                rows.append((si, di))
        if not rows:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long).t().contiguous()

    def _build_label_index(self) -> None:
        """positives[T] = list of (u_idx, v_idx, label) from snapshot edges at month T (the
        label month for target T-1). We index by the *edge* month here; a target T reads T+1."""
        self.edges_at: dict[int, list[tuple[int, int, int]]] = {}
        for ts, g in self.edge_df.groupby("ts"):
            lst = []
            for s, t, dc in zip(g["src"].tolist(), g["tgt"].tolist(), g["dominant_class"].tolist()):
                si = self.country_index.get(s); ti = self.country_index.get(t)
                lab = class_index(dc)
                if si is not None and ti is not None and lab >= 0:
                    lst.append((si, ti, lab))
            self.edges_at[int(ts)] = lst

    # ---- pair feature lookup --------------------------------------------------------
    def _pair_lookup(self, ts: int) -> dict[tuple[int, int], np.ndarray]:
        if ts not in self._pair_cache:
            attr = self.snap_attr[ts].numpy()
            self._pair_cache[ts] = {p: attr[i] for i, p in enumerate(self.snap_pairs[ts])}
        return self._pair_cache[ts]

    # ---- public API -----------------------------------------------------------------
    def build_window(self, target_ts: int) -> list[HeteroData]:
        """The W=12 monthly graphs ending at target_ts (inputs only; never reads T+1)."""
        cfg = self.cfg
        out = []
        for ts in range(target_ts - cfg.window + 1, target_ts + 1):
            ts = max(0, ts)
            d = HeteroData()
            d[C].x = self.country_x[ts]
            d[A].x = self.actor_x[ts]
            d[REL_SNAP].edge_index = self.snap_index[ts]
            d[REL_SNAP].edge_attr = self.snap_attr[ts]
            d[REL_BORDER].edge_index = self.border_index[ts]
            d[REL_MEMBER].edge_index = self.member_index[ts]
            d[REL_RMEMBER].edge_index = self.member_index[ts].flip(0)
            out.append(d)
        return out

    def make_samples(self, split: str, rng: np.random.Generator) -> list[Sample]:
        """Positives (all edges at T+1) + K negative STATUS_QUO pairs per non-SQ positive.
        Pass a fresh rng each epoch for train; a fixed-seed rng for val/test (frozen)."""
        cfg = self.cfg
        samples: list[Sample] = []
        edim = self.pp.edge_dim
        for T in cfg.target_months(split):
            label_ts = T + 1
            pos = self.edges_at.get(label_ts, [])
            existing = {(u, v) for (u, v, _) in pos}
            n_nonsq = sum(1 for (_, _, lab) in pos if lab != STATUS_QUO_INDEX)
            n_neg = cfg.k_neg * max(1, n_nonsq)
            negs = self._sample_negatives(existing, n_neg, rng)

            pairs = [(u, v) for (u, v, _) in pos] + negs
            labels = [lab for (_, _, lab) in pos] + [STATUS_QUO_INDEX] * len(negs)
            if not pairs:
                continue

            lookup = self._pair_lookup(T)
            attr = np.zeros((len(pairs), edim), dtype="float32")
            for i, (u, v) in enumerate(pairs):
                hit = lookup.get((u, v))
                if hit is not None:
                    attr[i] = hit
            samples.append(Sample(
                target_ts=T,
                pair_index=torch.tensor(pairs, dtype=torch.long),
                pair_attr=torch.from_numpy(attr),
                labels=torch.tensor(labels, dtype=torch.long),
            ))
        return samples

    def _sample_negatives(self, existing: set, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
        if self.num_country < 2 or n <= 0:
            return []
        out: list[tuple[int, int]] = []
        seen = set(existing)
        attempts = 0
        budget = n * 20 + 50
        while len(out) < n and attempts < budget:
            attempts += 1
            u = int(rng.integers(self.num_country)); v = int(rng.integers(self.num_country))
            if u == v or (u, v) in seen:
                continue
            seen.add((u, v)); out.append((u, v))
        return out

    def class_counts(self, split: str = "train", seed: int = 0) -> torch.Tensor:
        counts = torch.zeros(len(self.cfg.class_names), dtype=torch.long)
        for s in self.make_samples(split, np.random.default_rng(seed)):
            counts += torch.bincount(s.labels, minlength=len(self.cfg.class_names))
        return counts

    # ---- loading --------------------------------------------------------------------
    @classmethod
    def from_parquet(cls, cfg: Config, preprocess: Preprocess | None = None) -> "GeopoliticDataset":
        d = cfg.data_dir
        node_df = pd.read_parquet(os.path.join(d, "node_snapshots.parquet"))
        edge_df = pd.read_parquet(os.path.join(d, "snapshot_edges.parquet"))
        struct_df = pd.read_parquet(os.path.join(d, "structural_edges.parquet"))
        for df in (node_df, edge_df):
            df["ts"] = df["ts"].astype(int)
        return cls(cfg, node_df, edge_df, struct_df, preprocess)
