"""Pure-stdlib tests for the time_step convention (mirror of internal/timestep). Runnable
without any ML dependency installed."""

from ml import timestep as ts


def test_known_anchors():
    assert ts.from_ym(2010, 1) == 0
    assert ts.from_ym(2026, 6) == 197
    assert ts.from_year(2011) == 12
    assert ts.year(0) == 2010 and ts.month(0) == 1
    assert ts.year(197) == 2026 and ts.month(197) == 6
    assert ts.iso_period(42) == "2013-07"


def test_roundtrip():
    for t in range(0, 198):
        assert ts.from_ym(ts.year(t), ts.month(t)) == t


def test_clamp_start():
    assert ts.clamp_start(-5) == 0
    assert ts.clamp_start(7) == 7
