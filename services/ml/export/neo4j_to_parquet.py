"""Export geopolitic_aggregated (Neo4j) -> Parquet (plans/03 §1.1–§1.2).

Writes three files into the dataset dir (default services/ml/dataset_parquet/):
  node_snapshots.parquet   (one row per NodeSnapshot; all feature props, ts = time_step)
  snapshot_edges.parquet   (one row per SNAPSHOT_EDGE; features + dominant_class label)
  structural_edges.parquet (BORDERS + MEMBER_OF with validity intervals)

This is the Python equivalent of the Go exporter mentioned in the plan; CLASS_NAMES /
class ordering is mirrored in ml/config.py (Go's internal/label.Classes is the source of
truth). Reads ONLY (no writes/deletes) — safe to run against the live research DB.

Env: NEO4J_AGG_URI (default bolt://localhost:7688), NEO4J_USER (neo4j), NEO4J_PASSWORD.
Run:  python -m export.neo4j_to_parquet --out dataset_parquet
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from neo4j import GraphDatabase

DEFAULT_URI = os.environ.get("NEO4J_AGG_URI", "bolt://localhost:7688")
DEFAULT_USER = os.environ.get("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.environ.get("NEO4J_PASSWORD", "geopolitic")

NODE_Q = """
MATCH (ns:NodeSnapshot)
RETURN ns.node_id AS node_id, ns.node_type AS node_type, ns.time_step AS ts, properties(ns) AS p
"""

EDGE_Q = """
MATCH (s:Country)-[r:SNAPSHOT_EDGE {time_step:$ts}]->(t:Country)
RETURN s.id AS src, t.id AS tgt, properties(r) AS p
"""

STRUCT_Q = """
MATCH (a)-[r:BORDERS|MEMBER_OF]->(b)
RETURN type(r) AS rel, a.id AS a, b.id AS b,
       labels(a)[0] AS a_label, labels(b)[0] AS b_label,
       r.start_time_step AS start, r.end_time_step AS end
"""

MAXTS_Q = "MATCH (ns:NodeSnapshot) RETURN coalesce(max(ns.time_step), 0) AS m"


def _flatten(rows: list[dict], drop_dups: tuple[str, ...] = ()) -> pd.DataFrame:
    out = []
    for r in rows:
        props = dict(r.pop("p", {}) or {})
        merged = {**props, **r}            # explicit columns win over props of same name
        out.append(merged)
    df = pd.DataFrame(out)
    if "time_step" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"time_step": "ts"})
    if "time_step" in df.columns:
        df = df.drop(columns=["time_step"])
    return df


def export(out_dir: str, uri: str, user: str, password: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    counts = {}
    try:
        with driver.session() as sess:
            max_ts = int(sess.run(MAXTS_Q).single()["m"])

            print(f"[export] node snapshots ...")
            nodes = _flatten([dict(r) for r in sess.run(NODE_Q)])
            nodes.to_parquet(os.path.join(out_dir, "node_snapshots.parquet"), index=False)
            counts["node_snapshots"] = len(nodes)

            print(f"[export] snapshot edges (per month 0..{max_ts}) ...")
            frames = []
            for ts in range(max_ts + 1):
                batch = [dict(r) for r in sess.run(EDGE_Q, ts=ts)]
                if batch:
                    frames.append(_flatten(batch))
            edges = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            edges.to_parquet(os.path.join(out_dir, "snapshot_edges.parquet"), index=False)
            counts["snapshot_edges"] = len(edges)

            print(f"[export] structural edges ...")
            struct = pd.DataFrame([dict(r) for r in sess.run(STRUCT_Q)])
            struct.to_parquet(os.path.join(out_dir, "structural_edges.parquet"), index=False)
            counts["structural_edges"] = len(struct)
            counts["max_ts"] = max_ts
    finally:
        driver.close()
    print(f"[export] done -> {out_dir}: {counts}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Export geopolitic_aggregated to Parquet (read-only).")
    p.add_argument("--out", default="dataset_parquet")
    p.add_argument("--uri", default=DEFAULT_URI)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--password", default=DEFAULT_PASSWORD)
    args = p.parse_args()
    export(args.out, args.uri, args.user, args.password)


if __name__ == "__main__":
    main()
