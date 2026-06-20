"""Python mirror of internal/timestep/timestep.go. Dependency-free.

time_step = (year - 2010) * 12 + (month - 1), t=0 -> 2010-01. Calendar fields are pure
functions of time_step (no calendar node anywhere). Uses Python's flooring //,% which
already match Go's floorDiv/floorMod for negative inputs.
"""

from __future__ import annotations

EPOCH = 2010  # calendar year that maps to time_step 0 (January)


def from_ym(year: int, month: int) -> int:
    """(year, month[1-12]) -> time_step."""
    return (year - EPOCH) * 12 + (month - 1)


def from_year(year: int) -> int:
    """Calendar year -> time_step of its January."""
    return from_ym(year, 1)


def year(ts: int) -> int:
    return EPOCH + (ts // 12)


def month(ts: int) -> int:
    return (ts % 12) + 1


def iso_period(ts: int) -> str:
    """'YYYY-MM' for a time_step (e.g. '2013-07')."""
    return f"{year(ts):04d}-{month(ts):02d}"


def clamp_start(ts: int) -> int:
    """Structural edges that began before 2010 clamp to start_time_step = 0."""
    return 0 if ts < 0 else ts
