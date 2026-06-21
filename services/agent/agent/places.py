"""Place resolution: natural-language -> ISO-3 country codes / IGO groups.

The agent reasons in ISO-3 codes; this module maps any free-text place a user might type to
the ids the model knows. It is deliberately **Neo4j-free**: a Country's identity (ISO-3) and a
group's membership both already live in the exported dataset the model trained on
(``structural_edges.parquet`` carries the temporal ``MEMBER_OF`` edges), and ``pycountry``
supplies ISO-3166 names/aliases. Resolving from the same read-only export the ``Predictor``
uses keeps the agent consistent with the model and unable to touch any live database.

Two resolutions:
  * Country  -> {kind:"country", iso3, name}
  * Group    -> {kind:"group", code, qid, name, members:[iso3 valid at T]}
Only ISO-3 codes that exist in the model's Country universe are returned (others raise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pycountry

from . import groups


class PlaceError(ValueError):
    """Unresolvable place / not in the model universe (recoverable, surfaced to the agent)."""


# Common geopolitical names that ISO-3166 lookup misses or spells differently.
COUNTRY_ALIASES: dict[str, str] = {
    "russia": "RUS", "russian federation": "RUS",
    "united states": "USA", "united states of america": "USA", "us": "USA",
    "usa": "USA", "america": "USA",
    "uk": "GBR", "britain": "GBR", "great britain": "GBR", "england": "GBR",
    "united kingdom": "GBR",
    "south korea": "KOR", "republic of korea": "KOR", "korea": "KOR",
    "north korea": "PRK", "dprk": "PRK",
    "czech republic": "CZE", "czechia": "CZE",
    "uae": "ARE", "emirates": "ARE",
    "ivory coast": "CIV", "cote d'ivoire": "CIV",
    "turkey": "TUR", "turkiye": "TUR",
    "syria": "SYR", "laos": "LAO", "vietnam": "VNM", "brunei": "BRN",
    "bolivia": "BOL", "venezuela": "VEN", "tanzania": "TZA", "moldova": "MDA",
    "iran": "IRN", "taiwan": "TWN",
}


@dataclass
class Place:
    kind: str                       # "country" | "group"
    name: str
    iso3: str | None = None         # country only
    code: str | None = None         # group only (canonical group code, e.g. "EU")
    qid: str | None = None          # group only (Wikidata QID)
    members: list[str] | None = None  # group only (ISO-3 valid at T, in the model universe)

    def to_dict(self) -> dict:
        if self.kind == "country":
            return {"kind": "country", "name": self.name, "iso3": self.iso3}
        return {"kind": "group", "name": self.name, "code": self.code,
                "qid": self.qid, "members": self.members}


class PlaceResolver:
    def __init__(self, country_ids: set[str], member_df: pd.DataFrame):
        """``country_ids``: the model's Country universe (ISO-3). ``member_df``: the MEMBER_OF
        rows of structural_edges.parquet (columns a=country, b=actor QID, start, end)."""
        self.country_ids = set(country_ids)
        m = member_df[member_df["rel"] == "MEMBER_OF"] if "rel" in member_df.columns else member_df
        self._mem_a = m["a"].to_numpy()
        self._mem_b = m["b"].to_numpy()
        self._mem_start = m["start"].to_numpy(dtype="float64")
        self._mem_end = m["end"].to_numpy(dtype="float64")

    # ---- countries ------------------------------------------------------------------
    def resolve_country(self, text: str) -> str:
        """Free text -> ISO-3 in the model universe, or raise PlaceError."""
        raw = (text or "").strip()
        if not raw:
            raise PlaceError("empty place")
        norm = groups.normalize(raw)

        # 1) curated geopolitical aliases
        iso = COUNTRY_ALIASES.get(norm)
        # 2) a literal ISO-3 token
        if iso is None and len(raw) == 3 and raw.upper().isalpha():
            iso = raw.upper()
        # 3) ISO-3166 lookup (handles ISO-2, ISO-3, official & common names)
        if iso is None:
            try:
                iso = pycountry.countries.lookup(raw).alpha_3
            except LookupError:
                iso = None
        # 4) fuzzy fallback
        if iso is None:
            try:
                iso = pycountry.countries.search_fuzzy(raw)[0].alpha_3
            except LookupError:
                iso = None

        if iso is None:
            raise PlaceError(f"could not resolve country: {text!r}")
        if iso not in self.country_ids:
            raise PlaceError(f"country {iso!r} is not in the model universe")
        return iso

    @staticmethod
    def country_name(iso3: str) -> str:
        try:
            c = pycountry.countries.get(alpha_3=iso3)
            return c.name if c else iso3
        except (KeyError, LookupError):
            return iso3

    # ---- groups ---------------------------------------------------------------------
    def group_members(self, qid: str, time_step: int) -> list[str]:
        """ISO-3 members of an IGO valid at ``time_step`` (start<=T and (end is null or end>T)),
        restricted to the model's Country universe, sorted."""
        s = np.nan_to_num(self._mem_start, nan=0.0)
        valid = (self._mem_b == qid) & (s <= time_step) & (
            np.isnan(self._mem_end) | (self._mem_end > time_step))
        out = sorted({a for a in self._mem_a[valid] if a in self.country_ids})
        return out

    def resolve_group(self, text: str, time_step: int) -> Place:
        code = groups.group_code_for(text)
        if code is None:
            raise PlaceError(f"unknown group: {text!r}")
        qid, name = groups.GROUP_QID[code]
        return Place(kind="group", name=name, code=code, qid=qid,
                     members=self.group_members(qid, time_step))

    # ---- unified --------------------------------------------------------------------
    def resolve(self, text: str, time_step: int) -> Place:
        """Resolve any place: a group if the text names one, else a Country."""
        if groups.group_code_for(text) is not None:
            return self.resolve_group(text, time_step)
        iso = self.resolve_country(text)
        return Place(kind="country", name=self.country_name(iso), iso3=iso)
