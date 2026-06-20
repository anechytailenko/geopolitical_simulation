"""Pure-stdlib tests for config + the all-important class-order invariant: Python's
CLASS_NAMES must match the canonical order in Go's internal/label/cameo.go exactly (a single
mismatched index silently corrupts every label). Runnable without the ML stack."""

import os

from ml.config import CLASS_NAMES, Config, STATUS_QUO_INDEX, class_index

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CAMEO_GO = os.path.join(REPO_ROOT, "internal", "label", "cameo.go")


def test_class_order_matches_go_source():
    assert os.path.exists(CAMEO_GO), f"missing {CAMEO_GO}"
    text = open(CAMEO_GO).read()
    # canonical order = order of first appearance of each class string literal in the Go file
    first_idx = {c: text.index(f'"{c}"') for c in CLASS_NAMES}
    assert all(v >= 0 for v in first_idx.values()), "a class name is absent from cameo.go"
    go_order = sorted(CLASS_NAMES, key=lambda c: first_idx[c])
    assert go_order == CLASS_NAMES, f"Python/Go class order drift: {go_order} != {CLASS_NAMES}"


def test_class_helpers():
    assert class_index("STATUS_QUO") == STATUS_QUO_INDEX
    assert class_index("MATERIAL_CONFLICT") == 0
    assert class_index("not_a_class") == -1
    assert len(CLASS_NAMES) == 5


def test_split_logic():
    cfg = Config(max_ts=197, min_target_ts=11, train_max_ts=172, val_max_ts=184)
    assert cfg.test_max_ts == 196
    assert cfg.split_of(10) == "none"          # below min window
    assert cfg.split_of(11) == "train"
    assert cfg.split_of(172) == "train"
    assert cfg.split_of(173) == "val"
    assert cfg.split_of(184) == "val"
    assert cfg.split_of(185) == "test"
    assert cfg.split_of(196) == "test"
    assert cfg.split_of(197) == "none"         # T+1 would exceed max_ts
    # contiguous, non-overlapping, in order
    tr, va, te = cfg.target_months("train"), cfg.target_months("val"), cfg.target_months("test")
    assert tr[-1] < va[0] < va[-1] < te[0]
    assert set(tr) | set(va) | set(te) == set(range(11, 197))
