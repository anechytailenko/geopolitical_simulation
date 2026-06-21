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

---

## 11. ReAct agent & MCP tools (`services/agent`, see plans/04-react-agent.md)

The agent answers natural-language geopolitical questions by calling **task-shaped MCP tools**
that run the trained GNN (`ml.infer.Predictor`) in-process. It is **read-only and database-free**:
it loads the model artifacts from `artifacts/` and the dataset from
`services/ml/dataset_parquet/` (the same export the model trained on) and resolves places with
`pycountry` + the parquet `MEMBER_OF` edges — it never connects to Neo4j, so it **cannot alter or
delete ingested data and never re-runs ingestion**. No GPU is needed.

Prerequisites: the trained bundle already exists at `artifacts/` (`best.pt`, `preprocess.pkl`,
`calibrator.pkl`, `metrics.json`) and the parquet export at `services/ml/dataset_parquet/`.

### 11a. One-time setup (isolated venv that reuses the system torch/PyG)

```bash
python3 -m venv --system-site-packages services/agent/.venv
PV=services/agent/.venv/bin/python
# (add --trusted-host pypi.org --trusted-host files.pythonhosted.org if your Python has the
#  macOS SSL-cert issue; omit otherwise)
$PV -m pip install -e services/ml --no-deps
$PV -m pip install -e services/agent          # langgraph, langchain-core, mcp,
                                              #   langchain-mcp-adapters, langchain-ollama,
                                              #   fastapi, sse-starlette, pycountry, pytest
```

**Output:** `Successfully installed … langgraph-1.x mcp-1.x pycountry-…`. The venv inherits
`torch` / `torch-geometric` / `scikit-learn` from the system, so only the agent stack downloads.

### 11b. Environment (optional — paths auto-discover)

**You do not need to export anything.** The agent **auto-discovers** the model bundle: if
`GEO_DATA_DIR` / `GEO_ARTIFACTS_DIR` are unset (or point somewhere without the files), it falls
back to the bundle shipped in the repo (`services/ml/dataset_parquet/` and `artifacts/`). So
`python -m agent` / `python -m agent.mcp_server` work from **any** directory — this is what avoids
the `FileNotFoundError: dataset_parquet/node_snapshots.parquet` you get when launching from the
repo root without the old exports.

To customize (LLM, explainer cost, or an alternate bundle), put the vars in a **`.env`** file the
agent loads at startup (repo-root `.env`, then `services/agent/.env`; an existing shell env always
wins). Copy the template:

```bash
cp services/agent/.env.example services/agent/.env     # then edit if desired
```

Available knobs (all optional; defaults shown):

```
LLM_PROVIDER=ollama   LLM_MODEL=qwen2.5:3b-instruct     # or LLM_PROVIDER=anthropic (+ ANTHROPIC_API_KEY)
GNN_EXPLAINER_EPOCHS=64   IG_STEPS=24                   # explainer cost (lower = faster, coarser masks)
# GEO_ARTIFACTS_DIR / GEO_DATA_DIR  — only to OVERRIDE the auto-discovered repo bundle (ABSOLUTE paths)
```

> Do **not** use `$PWD` inside a `.env` file — it is not expanded there. Thanks to
> auto-discovery you normally don't need the path vars at all.

### 11c. Run the tests (no LLM, no Neo4j, read-only)

```bash
cd services/agent
.venv/bin/python -m pytest -q
```

**Output:**

```
..................................                                       [100%]
34 passed in ~45–70s
```

The suite (34 tests) drives the **real trained model** and a **deterministic scripted LLM** (no
Ollama, no network) and an **in-memory MCP client/server session** (no subprocess). It covers:
place resolution incl. the temporal NATO membership (FIN/SWE present at T=197, absent at T=100);
server-side arg validation (bad ISO-3 / class / time_step); `predict_pair` summing to 1.0,
determinism, and a quiet no-edge dyad; group/counterpart ranking; the `compare_pair` 5→3 buckets;
the explainer (subgraph importances ∈ [0,1], IG only for the focus pair `u`/`v`, completeness
gap < 0.05); the single-step counterfactual (baseline reproduces `predict_pair`, edit applied at
input level, no cache mutation); the ReAct graph + deterministic viz for the 4 question types;
and the SSE server emitting `token`/`tool_call`/`tool_result`/`final`. The whole run loads the
model once (~10 s) and touches no database.

> Note on the shipped checkpoint: it is overconfident and dominated by identity + the query
> pair's own features, so a counterfactual neighbor-edit moves the output only marginally (max
> Δ ≈ 1e-4). The tests assert the **mechanism** (baseline correctness, input edit, no mutation,
> valid one-step distribution), not a large output swing the model does not produce.

### 11d. Run the MCP server (stdio — for an MCP client such as Claude Desktop)

```bash
services/agent/.venv/bin/python -m agent.mcp_server
```

**Output:** loads the model once (the §11 boot self-check asserts the class order matches Go/
Python canon and a probe `predict("USA","CHN",197)` sums to 1.0; it **fails fast** on a
mislocated/misaligned artifact), then serves these 8 tools over stdio:
`get_latest_time_step, resolve_place, predict_pair, best_pair_in_group, most_likely_counterpart,
compare_pair, explain_pair, predict_counterfactual`. It runs until the client disconnects.

### 11e. Run the chat agent (HTTP + Server-Sent Events)

The default LLM is local Ollama. Run it **either on the host** (uses Metal on macOS — the faster
path) **or in Docker** (reproducible, matches the Neo4j stack; CPU-only on macOS, GPU on
Linux+NVIDIA — see plans/04 §10.1). The agent reaches either at `OLLAMA_BASE_URL`
(default `http://localhost:11434`), so the two are interchangeable with no code change.

```bash
# Option A — host Ollama (recommended on Apple Silicon):
ollama serve &                           # start the local Ollama server (or launch the Ollama.app)
ollama pull qwen2.5:3b-instruct          # ~3 GB (plans/04 §10); needs ~4–5 GB free w/ the model

# Option B — dockerized Ollama (opt-in compose profile; NOT started by a plain `up`):
docker compose -f infra/docker/docker-compose.yml --profile llm up -d ollama
docker exec geopolitic-ollama ollama pull qwen2.5:3b-instruct   # one-time, into the volume

# then start the agent (same command for either option; no env exports needed — §11b):
services/agent/.venv/bin/python -m agent  # uvicorn on http://127.0.0.1:8100
```

> If `ollama pull` prints `could not connect to ollama server, run 'ollama serve'`, the local
> Ollama server isn't running — run `ollama serve` (or open the Ollama app) first, then pull. The
> agent itself still **boots** without Ollama (the LLM is built lazily); only a chat request fails,
> and it returns a clean error event rather than crashing.

**Output (docker option):** `geopolitic-ollama … Started`; the model volume `ollama_models`
caches the pulled weights so the pull is one-time. Stop it with
`docker compose -f infra/docker/docker-compose.yml --profile llm down`.

**Output:** `INFO: Uvicorn running on http://127.0.0.1:8100`. Liveness:
`curl -s localhost:8100/health` → `{"status":"ok"}`. Stream a question:

```bash
curl -N -s -X POST localhost:8100/agent/chat \
  -H 'content-type: application/json' \
  -d '{"message":"What is most likely between the USA and China next month?"}'
```

**Output:** an SSE stream — `event: tool_call` (the chosen tool + args) → `event: tool_result`
(a one-line summary) → `event: token` (the answer text) → `event: final` (the viz payload:
`focus_pairs`, the GNNExplainer `subgraph`, and Integrated-Gradients `feature_attributions` for
the two focus countries; plans/04 §6). The deterministic viz step always explains the answer's
focus pair, so the panel can never disagree with the answer.

No GPU and no Neo4j are required. To run without Ollama, set `LLM_PROVIDER=anthropic` (with
`langchain-anthropic` installed and `ANTHROPIC_API_KEY` set) — one env change, same tools.

---

## 12. Web frontend (`services/frontend`, see plans/05-frontend.md)

A Vite + React + TypeScript single-page app — the chat-driven demo. It talks to **one** origin:
the agent's SSE endpoint (`POST /agent/chat` on `:8100`), proxied by Vite in dev. It never
reaches Neo4j, the Go API, or BigQuery, and performs **no database access** — it only renders the
agent's payload. Requires Node 18+ (Node 22 here).

### 12a. Install

```bash
cd services/frontend
npm install
```

**Output:** `added N packages …` (react, zustand, d3-force + the Vite/Vitest/RTL dev stack).

### 12b. Run the frontend tests (no browser, no agent, no DB)

```bash
cd services/frontend
npm test                 # vitest run   (one-shot; `npm run test:watch` for watch mode)
npm run typecheck        # tsc --noEmit  (optional; strict TypeScript, no errors)
```

**Output:**

```
 Test Files  11 passed (11)
      Tests  26 passed (26)
   Duration  ~1.7s
```

The 26 specs run under **jsdom with `fetch`/SSE mocked** — no browser, no network, no agent, and
therefore **no database I/O**: the suite **cannot write, delete, or drop** any data (plans/05 §8).
They cover SSE frame parsing (incl. a frame split across chunks and CRLF endings), the store
reducer, the scrollable example chips (above the input), `ChatInput`, the probability bars, the
subgraph (input edges + the synthesized focus/prediction edge that renders even when it is not an
input edge), IG-popup gating (only the focus pair `u`/`v` are clickable), the class-color/theme
tokens (electric-purple accent `#a855f7`), a stubbed end-to-end stream → render → click-node flow,
the counterfactual baseline-vs-counterfactual view, error/malformed-frame resilience, and a
**graceful error bubble when the agent is unreachable** (the dev-proxy `ECONNREFUSED` case).

### 12c. Run the website locally (dev)

Start the agent first (§11e — needs Ollama, host or docker), then the Vite dev server:

```bash
# terminal 1 — the agent SSE server on :8100 (see §11e)
services/agent/.venv/bin/python -m agent

# terminal 2 — the frontend dev server on :5173 (proxies /agent → :8100)
cd services/frontend
npm run dev
```

**Output:** `VITE v5 … Local: http://localhost:5173/`. Open it: click an example chip (the
scrollable strip above the input) or type a question and press **Send** → the right panel streams
the tool steps (`▸ get_latest_time_step → 2026-07`, `▸ best_pair_in_group(…)`) then the answer +
probability chart; the left panel draws the GNNExplainer subgraph with the highlighted
prediction edge; clicking a highlighted (focus-pair) country node opens its Integrated-Gradients
popup. The theme is **electric purple**; the example chips scroll left/right above the input.

To point the UI at an agent on another host, set `VITE_AGENT_TARGET` before `npm run dev`
(e.g. `VITE_AGENT_TARGET=http://192.168.1.10:8100 npm run dev`).

### 12d. Production build (optional)

```bash
cd services/frontend
npm run build            # → dist/  (static bundle)
npm run preview          # serve dist/ locally to sanity-check
```

A deployed static `dist/` served from a different origin than the agent would need CORS on the
agent (one-line `app.add_middleware(CORSMiddleware, …)` in `services/agent/agent/server.py`,
plans/05 §10) — not needed for the Vite-proxied dev flow above.
