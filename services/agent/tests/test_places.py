"""Place resolution (plans/04 §12 Grounding)."""

import pytest

from agent.places import PlaceError


def test_country_aliases(rt):
    for name in ["Germany", "germany", "DEU", "DE"]:
        assert rt.resolver.resolve_country(name) == "DEU"
    assert rt.resolver.resolve_country("United States") == "USA"
    assert rt.resolver.resolve_country("the United States") == "USA"
    assert rt.resolver.resolve_country("China") == "CHN"
    assert rt.resolver.resolve_country("Russia") == "RUS"


def test_group_eu(rt):
    p = rt.resolver.resolve("the EU", rt.max_ts)
    assert p.kind == "group" and p.qid == "Q458"
    assert len(p.members) >= 2
    assert all(m in rt.country_ids for m in p.members)


def test_nato_membership_is_temporal(rt):
    """Finland/Sweden are NATO members at T=197 but not at an earlier month (real accessions)."""
    now = set(rt.resolver.group_members("Q7184", 197))
    old = set(rt.resolver.group_members("Q7184", 100))
    assert {"FIN", "SWE"} <= now
    assert not ({"FIN", "SWE"} & old)
    assert now > old


def test_unknown_place_raises(rt):
    with pytest.raises(PlaceError):
        rt.resolver.resolve_country("Atlantis")


def test_iso3_not_in_universe_raises(rt):
    with pytest.raises(PlaceError):
        rt.resolver.resolve_country("XYZ")
