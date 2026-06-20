# Commands

Every command for running the data-ingestion stack, with what it does and the
output to expect. Run from the repo root (`/Users/annanechytailenko/Desktop/geopolitic`).

Two Neo4j databases (one container each, since Community edition is single-database):
- **`geopolitic_raw`** — event-level store (`geopolitic-neo4j-raw`, Bolt `7687`)
- **`geopolitic_aggregated`** — snapshot store the ML service reads (`geopolitic-neo4j-aggregated`, Bolt `7688`)

**All data is real.** There is no synthetic generation. The country universe comes
from the live World Bank country list (~195 states); structural edges from Wikidata;
events from GDELT (BigQuery); features from World Bank, ACLED, V-Dem, SIPRI, UNDP.

Prerequisites: Docker + Docker Compose, Go 1.26+, `python3` (only to pretty-print JSON).

---

## 1. Start the databases

```bash
docker compose -f infra/docker/docker-compose.yml up -d
docker ps --format '{{.Names}}\t{{.Status}}' | grep geopolitic
```

Starts both Neo4j 5.26 containers. **Output:** `... Started` for each; wait until
both show `(healthy)` (~30 s). Browser UIs: raw http://localhost:7474, aggregated
http://localhost:7475 (login `neo4j` / `geopolitic`).

---

## 2. Build & unit tests (no database, no network)

```bash
go build ./...        # compiles every package; no output on success
go vet ./...          # static analysis; no output on success
go test ./tests/...   # pure-logic unit tests
```

**Output of the unit tests:**

```
ok  	geopolitic/tests	0.6s
```

Covers `time_step` math, the CAMEO→class label generator, GDELT event mapping
(`BuildEvent`), and the credentialed-source parsers (V-Dem, UNDP, ACLED aggregation,
SIPRI Excel) against in-memory real-format fixtures.

---

## 3. Integration tests (against the live databases)

> ⚠️ **Destructive.** These tests call `MATCH (n) DETACH DELETE n` on **both**
> Neo4j databases as setup. They are guarded: without the opt-in they SKIP. Run
> them against a throwaway/dev instance, never one holding data you want to keep.

```bash
# enable the destructive tests (point RAW_URI/AGG_URI at a throwaway DB if needed):
GEOPOLITIC_ALLOW_DB_WIPE=1 go test -tags=integration -count=1 ./tests/...
```

Runs the real loaders against the two dockerized DBs (external sources served from
in-process fixtures so assertions are deterministic). **Output:**

```
ok  	geopolitic/tests	~23s
```

Without `GEOPOLITIC_ALLOW_DB_WIPE=1`, `TestCoreIngestionAndAggregation` and
`TestWikidataLoader` report `--- SKIP` and the databases are left untouched.

What it verifies:
- **`TestCoreIngestionAndAggregation`** — country list → Country nodes; World Bank
  feature snapshots; **time-varying seeds (`unsc_seat_flag`, `sanctions_status`) are
  written as timestamped `FeatureSnapshot`s and are NOT properties on the identity
  node**; static `nuclear_flag` IS on the node; forward-fill into NodeSnapshots;
  conflict-priority dominance; Brexit point-in-time `MEMBER_OF`.
- **`TestWikidataLoader`** — region, temporal `BORDERS`, and IGO `MEMBER_OF` written
  from a fixture SPARQL endpoint.

---

## 4. Credentials for the real sources

The **core runs with no credentials** (World Bank country list + indicators +
Wikidata + JSON seeds). Each extra source activates only when its env vars are set
(`config.<Source>Enabled()`); otherwise it is reported `skipped` and the run still
succeeds. Set the env vars in the shell that launches the API (step 5). Use `!` in
this session to run an interactive login, e.g. `! gcloud auth application-default login`.
**Never commit secrets** — pass them as environment variables only.

### 4a. GDELT — events — all free

1. Create a Google Cloud project at https://console.cloud.google.com (free tier:
   1 TB of BigQuery queries/month).
2. Enable the BigQuery API for that project.
3. Authenticate with Application Default Credentials, either:
   - `gcloud auth application-default login` (interactive), **or**
   - a service-account JSON key → `export GOOGLE_APPLICATION_CREDENTIALS=/path/key.json`
4. `export GDELT_GCP_PROJECT=<your-project-id>`

GDELT tables: **`gdelt-bq.full.events`** (GDELT 1.0, 2010→2015-02) + **`gdelt-bq.gdeltv2.events`**
(2.0, 2015-02→present). (`gdelt-bq.gdeltv1.events` does not exist.) Actor codes are
**ISO-3** (verified against the live tables); `configs/cameo_country_to_iso3.json` is
identity over the ingested countries + legacy aliases, regenerated with
`go run tools/gen_cameo.go`.

**`GDELT_MODE` (default `aggregated`):**
- **`aggregated`** — aggregates the full 2010–present event set **in BigQuery** into
  ~2.4M monthly `SNAPSHOT_EDGE`s (~1.5–2 GB) written straight into
  `geopolitic_aggregated`. Raw events stay in BigQuery. The only feasible mode on a
  disk-constrained box. The two filtered segments scan ~89 GB (within the free tier).
- **`raw`** — streams every country↔country `[:EVENT]` edge into `geopolitic_raw`
  (16–48 GB; needs ample free disk). `GDELT_MAX_ROWS` caps rows per segment (raw mode
  only); `GDELT_START_DATE` / `GDELT_END_DATE` (YYYY-MM-DD) bound both modes.

**Verify (aggregated mode):** in `geopolitic_aggregated`,
`MATCH ()-[r:SNAPSHOT_EDGE]->() WHERE r.data_source='GDELT' RETURN count(r)` > 0.
**Verify (raw mode):** in `geopolitic_raw`, `MATCH ()-[e:EVENT]->() RETURN count(e)` > 0.

### 4b. ACLED — conflict-intensity feature — free

1. Register a free myACLED account at https://acleddata.com/register (OAuth; the
   legacy `key`+`email` API was retired 2025-09-15).
2. `export ACLED_EMAIL=<your-email>` and `export ACLED_PASSWORD=<your-password>`.

**Verify:** `MATCH (s:FeatureSnapshot) WHERE s.conflict_intensity IS NOT NULL RETURN count(s)` > 0.

### 4c. V-Dem — polyarchy + leadership change — free

1. Create a free account at https://v-dem.net and download
   `V-Dem-CY-Full+Others-v*.csv` (Country-Year: Full+Others).
2. `export VDEM_CSV_PATH=/path/V-Dem-CY-Full+Others-v15.csv`

**Verify:** `MATCH (s:FeatureSnapshot) WHERE s.vdem_polyarchy_score IS NOT NULL RETURN count(s)` > 0.

### 4d. SIPRI — military expenditure — free, no account

1. Download the Milex Excel from https://www.sipri.org/databases/milex
   (`SIPRI-Milex-data-1949-YYYY.xlsx`).
2. `export SIPRI_XLSX_PATH=/path/SIPRI-Milex-data-1949-2024.xlsx`
   (the loader auto-detects the “Constant US$” sheet; override with `SIPRI_SHEET`).

**Verify:** `MATCH (s:FeatureSnapshot) WHERE s.military_expenditure_log IS NOT NULL RETURN count(s)` > 0.

### 4e. UNDP — HDI — free, no account

1. Download the HDR “composite indices — complete time series” CSV from
   https://hdr.undp.org/data-center/documentation-and-downloads (has an `iso3`
   column + `hdi_YYYY` columns).
2. `export UNDP_HDI_CSV_PATH=/path/HDR-composite-indices.csv`

**Verify:** `MATCH (s:FeatureSnapshot) WHERE s.hdi IS NOT NULL RETURN count(s)` > 0.

---

## 5. Start the ingestion API

Credentials and dataset paths live in **`.env`** (gitignored; template in
`.env.example`). `internal/config.FromEnv()` loads it automatically — no manual
`export` needed. A real `.env` is already populated for this project:

```
GDELT_GCP_PROJECT=…           # GDELT events (also needs gcloud ADC, below)
ACLED_EMAIL=… ACLED_PASSWORD=…
VDEM_CSV_PATH=datasets_csv/V-Dem-CY-Full+Others-v16.csv
SIPRI_XLSX_PATH=datasets_csv/SIPRI-Milex-data-1949-2025_v1.2.xlsx
UNDP_HDI_CSV_PATH=datasets_csv/HDR25_Composite_indices_complete_time_series.csv
```

```bash
gcloud auth application-default login   # ONE-TIME, interactive — required for GDELT events
go run ./api                            # reads .env automatically
```

**Output:** `ingestion API listening on :8080 (raw=bolt://localhost:7687 agg=bolt://localhost:7688)`.
Liveness: `curl -s localhost:8080/healthz` → `{"status":"ok"}`.

> The dataset files in `datasets_csv/` are pre-trimmed to 2010+ (V-Dem rows,
> UNDP/SIPRI year-columns) by `go run tools/trim_datasets.go`.

### Selective re-run (add GDELT events without repeating World Bank)

GDELT is the only source of EVENT edges and needs GCP auth. After a full run, once
you have authenticated, add just the events without re-pulling the 38-min World
Bank data:

```bash
INGEST_ONLY=gdelt curl ...   # or: INGEST_ONLY=gdelt go run ./api  then POST /ingest
```

`INGEST_ONLY` accepts a comma list of: `countries, wikidata, worldbank, seeds,
gdelt, acled, vdem, sipri, undp`. Empty = run everything. Aggregation always runs.

---

## 6. Trigger ingestion

```bash
curl -s -X POST localhost:8080/api/v1/ingest -m 9000 | python3 -m json.tool
```

Runs the full pipeline: schema → country list → static flags → Wikidata → World
Bank → time-varying seeds → ACLED/V-Dem/SIPRI/UNDP → aggregate → GDELT
(`aggregated` mode: BigQuery → SNAPSHOT_EDGEs). With the full `.env`, a single POST
runs everything (~2 h on a laptop). On a disk- or rate-limit-constrained box it is
practical to run it in **two phases** with `INGEST_ONLY` (§5): first everything
except GDELT, then `INGEST_ONLY=gdelt`. Both produce the same database.

**Phase 1 — real non-GDELT run (countries, Wikidata, World Bank, seeds, ACLED,
V-Dem, SIPRI, UNDP + aggregation), ~56 min:**

```json
{
    "status": "ok",
    "countries": 217,
    "wikidata": { "Regions": 209, "Borders": 861, "Actors": 11, "Memberships": 763 },
    "worldbank_snapshots": 3255,
    "unsc_snapshots": 13, "sanctions_snapshots": 11, "financial_tier_snapshots": 7,
    "acled_features": 12000,
    "vdem_snapshots": 2863, "sipri_snapshots": 2325, "undp_snapshots": 2688,
    "aggregated": { "IdentityNodes": 232, "NodeSnapshots": 43152, "SnapshotEdges": 0, "MaxTimeStep": 185 },
    "sources": { "country_list":"ok","static_flags":"ok","wikidata":"ok","worldbank":"ok",
        "unsc":"ok","sanctions":"ok","financial_tier":"ok","acled":"ok",
        "vdem":"ok","sipri":"ok","undp":"ok" },
    "errors": null
}
```

**Phase 2 — `INGEST_ONLY=gdelt` (GDELT aggregated in BigQuery → SNAPSHOT_EDGEs),
~74 min** (≈23 min re-build of NodeSnapshots out to the present + BigQuery
aggregation + ~2.4M edge writes):

```json
{
    "status": "ok",
    "gdelt_snapshot_edges": 2396283,
    "aggregated": { "IdentityNodes": 232, "NodeSnapshots": 45936, "SnapshotEdges": 2396283, "MaxTimeStep": 197 },
    "sources": { "gdelt": "ok", "...": "skipped (INGEST_ONLY)" },
    "errors": null
}
```

- `countries: 217` — the live World Bank country universe (the old 20-country cap was
  a synthetic-seed artifact, since removed). Wikidata adds 4 more border-neighbour
  `Country` nodes (incl. **TWN**), so the raw DB holds **221** Country nodes.
- `acled_features: 12000`, `vdem 2863 / sipri 2325 / undp 2688` exactly match the
  trimmed dataset row counts — **no in-window data dropped**.
- `gdelt_snapshot_edges: 2396283` — full 2010-present country↔country event coverage,
  aggregated server-side. `MaxTimeStep 197` = 2026-06 (NodeSnapshots forward-filled to
  the present so every edge month has node features).
- Taiwan is added separately (`go run tools/add_gdelt_country.go TWN`, §8) →
  **+24,829 edges**, total **2,421,112**.
- 409 = an ingest is already running (the endpoint serializes runs).

---

## 7. Check last-run status

```bash
curl -s localhost:8080/api/v1/ingest/status | python3 -m json.tool
```

Reads the singleton `(:IngestState)` node — no re-ingest. Returns
`{"status":"never_run"}` (404) before the first run.

---

## 8. Verify the data in Neo4j

All outputs below are from the real core run (217 countries).

**Country universe + real Wikidata region + real World Bank GDP** (proves it is real,
not placeholder, data):

```bash
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (c:Country) RETURN count(c) AS countries"
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (c:Country) WHERE c.id IN ['USA','DEU','JPN','BRA','NGA'] RETURN c.id, c.name, c.iso2, c.region ORDER BY c.id"
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (s:FeatureSnapshot {node_id:'USA'}) WHERE s.gdp_log IS NOT NULL RETURN s.year, round(s.gdp_log,2) AS gdp_log ORDER BY s.year DESC LIMIT 3"
```

**Output:** `countries → 221` (217 from World Bank + 4 Wikidata border-neighbours
incl. TWN); regions are real continents (`BRA → South America`, `JPN → Asia`,
`USA → North America`); `USA gdp_log` ≈ `30.99` for 2024 (= ln of ~US$29 T).

**The refinement — time-varying features are timestamped, not static node props.**

```bash
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (c:Country {id:'USA'}) RETURN c.unsc_seat_flag AS on_node"
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (s:FeatureSnapshot {node_id:'USA', time_step:0}) RETURN s.unsc_seat_flag AS on_snapshot"
```

**Output:** `on_node → NULL`; `on_snapshot → 1`. Same for `sanctions_status`
(`node_id:'PRK'` → `NULL` on node, `1` on the 2010 snapshot). `nuclear_flag` (static)
*is* on the node (`= 1` for USA).

**Real Wikidata structural data** — France's borders (note overseas neighbours like
Brazil/Suriname via French Guiana confirm it is genuine Wikidata P47) and IGO members:

```bash
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (:Country {id:'FRA'})-[:BORDERS]->(n:Country) RETURN collect(n.id) AS france_borders"
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (a:Actor)<-[:MEMBER_OF]-(c:Country) RETURN a.name, count(c) AS members ORDER BY members DESC"
```

**Output:** France borders include `ESP, ITA, DEU, BEL, CHE, BRA, SUR …`; IGO member
counts are real (`United Nations 192`, `NATO 30`, `European Union 27`, …).

**Brexit on real Wikidata dates** — the UK's EU membership carries a real end date:

```bash
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (:Country{id:'GBR'})-[r:MEMBER_OF]->(a:Actor{id:'Q458'}) RETURN r.start_time_step AS start, r.end_time_step AS end"
```

**Output:** `start → 0`, `end → 120` (time_step 120 = 2020-01), so the point-in-time
predicate `start <= T AND (end IS NULL OR end > T)` correctly drops the UK after 2020.

Aggregated DB (forward-fill carries time-varying features into every month):

```bash
docker exec geopolitic-neo4j-aggregated cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (n:NodeSnapshot {node_id:'USA', time_step:66}) RETURN n.unsc_seat_flag AS unsc, round(n.gdp_log,2) AS gdp_log, n.nuclear_flag AS nuclear"
```

**ACLED conflict features (raw DB)** — real per-country-month conflict intensity:

```bash
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (s:FeatureSnapshot{node_id:'UKR'}) WHERE s.conflict_intensity IS NOT NULL RETURN s.year, s.active_conflict_count, round(s.conflict_intensity,2) ORDER BY s.conflict_intensity DESC LIMIT 3"
```

**Output:** UKR 2024 ~`4645` events/month, `conflict_intensity ≈ 9.1` (= ln Σfatalities).

**GDELT SNAPSHOT_EDGEs (aggregated DB)** — full 2010-present event coverage,
aggregated in BigQuery (`GDELT_MODE=aggregated`):

```bash
docker exec geopolitic-neo4j-aggregated cypher-shell -u neo4j -p geopolitic --format plain \

"
# conflict-priority dominant_class at the 2022 invasion (ts 146 = 2022-03):
docker exec geopolitic-neo4j-aggregated cypher-shell -u neo4j -p geopolitic --format plain \
"MATCH (:Country{id:'RUS'})-[r:SNAPSHOT_EDGE]->(:Country{id:'UKR'}) WHERE r.time_step=146 RETURN r.dominant_class, r.event_count, round(r.weighted_intensity,3)"
```

**Output:** `edges → 2421112`, `min_ts → 0` (2010-01), `max_ts → 197` (2026-06);
RUS→UKR @2022-03 → `MATERIAL_CONFLICT`, `131682` events, intensity `-0.11`. (China→
Taiwan spikes to `15713` events at ts 151 = 2022-08, the Pelosi-visit crisis.)

### Add a country missing from the cameo map (e.g. Taiwan)

World Bank omits Taiwan, but Wikidata seeds it as a `Country` node and GDELT codes it
(`TWN`). After adding it to `configs/cameo_country_to_iso3.json`, surgically add only
its GDELT edges (no full re-run needed — it shares no edges with the existing set):

```bash
set -a; source .env; set +a
go run tools/add_gdelt_country.go TWN     # → "added 24829 SNAPSHOT_EDGEs involving TWN"
```

`tools/gen_cameo.go` also seeds `TWN` so a regeneration keeps the map aligned.

**Output:** `unsc → 1, gdp_log → 30.53, nuclear → 1` (all carried into the 2015-07
NodeSnapshot). After a GDELT run, inspect edges:
`MATCH ()-[r:SNAPSHOT_EDGE]->() RETURN r.dominant_class, count(*) ORDER BY count(*) DESC`.

Schema:

```bash
docker exec -neo4j-raw cypher-shell -u neo4j -p geopolitic "SHOW INDEXES"
```

---

## 9. Reset / re-ingest

Ingestion is idempotent (every write is MERGE). To reset counts first:

```bash
docker exec geopolitic-neo4j-raw cypher-shell -u neo4j -p geopolitic "MATCH (n) DETACH DELETE n"
docker exec geopolitic-neo4j-aggregated cypher-shell -u neo4j -p geopolitic "MATCH (n) DETACH DELETE n"
```

Then re-run step 6.

---

## 10. Stop the stack

```bash
docker compose -f infra/docker/docker-compose.yml down       # keep data volumes
docker compose -f infra/docker/docker-compose.yml down -v    # delete all data
```
