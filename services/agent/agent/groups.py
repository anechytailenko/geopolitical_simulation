"""Country-group (IGO) name/alias -> Wikidata QID map.

The 11 Actor (IGO) nodes in the aggregated dataset are keyed by Wikidata QID; a question that
names a group ("the EU", "NATO", ...) is resolved to its QID here, then to its member
Countries via the temporal ``MEMBER_OF`` edges valid at the query month (see ``places.py``).
QIDs verified against the curated ``ingestion/raw/wikidata.go`` major-orgs list and the
``MEMBER_OF`` actors present in ``structural_edges.parquet``.
"""

from __future__ import annotations

# canonical group code -> (QID, display name)
GROUP_QID: dict[str, tuple[str, str]] = {
    "UN": ("Q1065", "United Nations"),
    "EU": ("Q458", "European Union"),
    "NATO": ("Q7184", "NATO"),
    "ASEAN": ("Q7768", "ASEAN"),
    "AU": ("Q7159", "African Union"),
    "SCO": ("Q485207", "Shanghai Cooperation Organisation"),
    "ARAB_LEAGUE": ("Q7172", "League of Arab States"),
    "OAS": ("Q123759", "Organization of American States"),
    "CSTO": ("Q318693", "Collective Security Treaty Organization"),
    "WTO": ("Q7825", "World Trade Organization"),
    "INTERPOL": ("Q8475", "Interpol"),
}

# free-text alias -> canonical group code (compared lowercased, with a leading "the " stripped)
GROUP_ALIASES: dict[str, str] = {
    "un": "UN", "united nations": "UN",
    "eu": "EU", "european union": "EU", "europe": "EU",
    "nato": "NATO", "north atlantic treaty organization": "NATO",
    "north atlantic treaty organisation": "NATO",
    "asean": "ASEAN", "association of southeast asian nations": "ASEAN",
    "au": "AU", "african union": "AU",
    "sco": "SCO", "shanghai cooperation organisation": "SCO",
    "shanghai cooperation organization": "SCO",
    "arab league": "ARAB_LEAGUE", "league of arab states": "ARAB_LEAGUE",
    "oas": "OAS", "organization of american states": "OAS",
    "organisation of american states": "OAS",
    "csto": "CSTO", "collective security treaty organization": "CSTO",
    "wto": "WTO", "world trade organization": "WTO", "world trade organisation": "WTO",
    "interpol": "INTERPOL",
}

# QID -> canonical group code (reverse lookup; used to label MEMBER_OF features)
QID_TO_GROUP: dict[str, str] = {qid: code for code, (qid, _name) in GROUP_QID.items()}


def normalize(text: str) -> str:
    t = " ".join(text.strip().lower().split())
    if t.startswith("the "):
        t = t[4:]
    return t


def group_code_for(text: str) -> str | None:
    """Return the canonical group code for a free-text group name, or None."""
    t = normalize(text)
    if t.upper() in GROUP_QID:
        return t.upper()
    return GROUP_ALIASES.get(t)
