"""Dataset tests (gated on torch / torch_geometric). Synthetic fixtures only — no DB."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ml.dataset import GeopoliticDataset, C, A, REL_SNAP, REL_BORDER, REL_MEMBER, REL_RMEMBER
import numpy as np


def test_window_shape_and_keys(tiny_cfg):
    ds = GeopoliticDataset.from_parquet(tiny_cfg)
    win = ds.build_window(13)
    assert len(win) == tiny_cfg.window
    d = win[-1]
    assert d[C].x.shape == (ds.num_country, ds.pp.country_feat_dim)
    assert d[A].x.shape == (ds.num_actor, ds.pp.actor_feat_dim)
    for rel in (REL_SNAP, REL_BORDER, REL_MEMBER, REL_RMEMBER):
        assert d[rel].edge_index.shape[0] == 2
    assert d[REL_SNAP].edge_attr.shape[1] == ds.pp.edge_dim


def test_member_validity_interval(tiny_cfg):
    """RUS->Q7184 MEMBER_OF ends at ts=5, so it is present at ts=4 but gone at ts=6."""
    ds = GeopoliticDataset.from_parquet(tiny_cfg)
    rus, q = ds.country_index["RUS"], ds.actor_index["Q7184"]

    def has_member(ts):
        idx = ds.member_index[ts]
        return any(int(idx[0, k]) == rus and int(idx[1, k]) == q for k in range(idx.shape[1]))

    assert has_member(4)
    assert not has_member(6)


def test_labels_and_negative_determinism(tiny_cfg):
    ds = GeopoliticDataset.from_parquet(tiny_cfg)
    s1 = ds.make_samples("val", np.random.default_rng(123))
    s2 = ds.make_samples("val", np.random.default_rng(123))
    assert len(s1) > 0
    # frozen seed -> identical negatives/order
    assert s1[0].pair_index.tolist() == s2[0].pair_index.tolist()
    for s in s1:
        assert s.labels.min().item() >= 0 and s.labels.max().item() < len(tiny_cfg.class_names)
        assert s.pair_attr.shape == (s.pair_index.shape[0], ds.pp.edge_dim)


def test_target_pairs_use_T_plus_1_label(tiny_cfg):
    """RUS->UKR becomes MATERIAL_CONFLICT at ts>=12; the target month T=11 reads label@12."""
    ds = GeopoliticDataset.from_parquet(tiny_cfg)
    samples = {s.target_ts: s for s in ds.make_samples("train", np.random.default_rng(0))}
    s = samples[11]
    rus, ukr = ds.country_index["RUS"], ds.country_index["UKR"]
    mc = tiny_cfg.class_names.index("MATERIAL_CONFLICT")
    found = [(int(u), int(v), int(lab)) for (u, v), lab in zip(s.pair_index.tolist(), s.labels.tolist())
             if u == rus and v == ukr]
    assert found and found[0][2] == mc
