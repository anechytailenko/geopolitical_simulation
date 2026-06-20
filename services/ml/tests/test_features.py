"""Preprocess tests (gated on numpy/pandas/sklearn). Synthetic fixtures only — no DB."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import numpy as np
import pandas as pd

from ml.features import Preprocess, COUNTRY_CONT, COUNTRY_BIN


def _frames(tiny_dataset_dir):
    import os
    nd = pd.read_parquet(os.path.join(tiny_dataset_dir, "node_snapshots.parquet"))
    ed = pd.read_parquet(os.path.join(tiny_dataset_dir, "snapshot_edges.parquet"))
    return nd, ed


def test_fit_and_block_dims(tiny_dataset_dir):
    nd, ed = _frames(tiny_dataset_dir)
    pp = Preprocess().fit(nd, ed, train_max_ts=14)
    assert len(pp.country_ids) == 6
    assert len(pp.actor_ids) == 2
    assert pp.country_block_dim == len(COUNTRY_CONT) + len(COUNTRY_BIN) + len(pp.regions)
    assert pp.country_feat_dim == pp.country_block_dim + pp.alliance_dim
    assert pp.edge_dim == 10  # 5 continuous + class_distribution[5]

    month = nd[(nd.node_type == "Country") & (nd.ts == 13)]
    block = pp.country_block(month)
    assert block.shape == (6, pp.country_block_dim)
    assert not np.isnan(block).any()

    amonth = nd[(nd.node_type == "Actor") & (nd.ts == 13)]
    ablock = pp.actor_block(amonth)
    assert ablock.shape == (2, pp.actor_feat_dim)

    e13 = ed[ed.ts == 13]
    eattr = pp.edge_features(e13)
    assert eattr.shape == (len(e13), pp.edge_dim)
    assert not np.isnan(eattr).any()


def test_scaler_fit_on_train_only(tiny_dataset_dir):
    """Means used for impute/scale must come from train months only (leakage guardrail)."""
    nd, ed = _frames(tiny_dataset_dir)
    pp = Preprocess().fit(nd, ed, train_max_ts=14)
    train_mean = nd[(nd.node_type == "Country") & (nd.ts <= 14)]["gdp_log"].mean()
    assert abs(pp.country_means["gdp_log"] - float(train_mean)) < 1e-6


def test_round_trip_serialization(tiny_dataset_dir, tmp_path):
    nd, ed = _frames(tiny_dataset_dir)
    pp = Preprocess().fit(nd, ed, train_max_ts=14)
    path = str(tmp_path / "preprocess.pkl")
    pp.save(path)
    pp2 = Preprocess.load(path)
    assert pp2.country_ids == pp.country_ids
    assert pp2.edge_dim == pp.edge_dim
