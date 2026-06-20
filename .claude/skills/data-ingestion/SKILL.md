---
name: data-ingestion
description: Populate the geopolitic_raw and geopolitic_aggregated Neo4j databases for the geopolitical-simulation GNN project. Use when ingesting/refreshing data, adding a data source, debugging the ingest pipeline or POST /api/v1/ingest, or working with FeatureSnapshot / NodeSnapshot / SNAPSHOT_EDGE / EVENT / temporal BORDERS|MEMBER_OF data. Study window starts at the 2010-01 epoch; all data is real (no synthetic generation).
---

# Data Ingestion

Go is the only language that writes to Neo4j. The pipeline loads the event-level
`geopolitic_raw` database from real external sources, then rebuilds the
snapshot-level `geopolitic_aggregated` database (the only DB the ML service
reads). Triggered by `POST /api/v1/ingest`. Design docs: `plans/02-data-ingestion.md`
(sources) and `plans/01-architecture.md` (model/graph). Full command + credential
reference: `COMMANDS.md`.

**No synthetic data.** The country universe (~195) comes from the live World Bank
country list; never hard-code a country subset.

## Code map

| Path | Responsibility |
|---|---|
| `internal/timestep/` | `time_step = (year-2010)*12 + (month-1)` (no TimeStep node) |
| `internal/label/` | CAMEO event code → 1 of 5 relationship classes |
| `internal/config/` | env-driven config + `*Enabled()` gates per source |
| `internal/neo4jdb/` | driver wrapper: Connect/Write/Count/ReadRows |
| `ingestion/raw/countries.go` | live World Bank country list → all Country nodes |
| `ingestion/raw/worldbank.go` | 7 WB indicators → FeatureSnapshots (concurrent + retry) |
| `ingestion/raw/wikidata.go` | SPARQL: region, temporal BORDERS/MEMBER_OF, IGO actors |
| `ingestion/raw/seed.go` | §9a static flags + §9b time-varying seeds (UNSC/sanctions/tier) |
| `ingestion/raw/gdelt.go` | GDELT via BigQuery → raw EVENT edges (`GDELT_MODE=raw`) |
| `ingestion/aggregated/gdelt.go` | GDELT aggregated in BigQuery → SNAPSHOT_EDGEs (`GDELT_MODE=aggregated`, default) |
| `ingestion/raw/acled.go` | ACLED OAuth → conflict-intensity features |
| `ingestion/raw/{vdem,sipri,undp}.go` | file-based feature loaders |
| `ingestion/raw/events.go` | `BuildEvent` (CAMEO→Event) + `WriteEvents` |
| `ingestion/raw/featuresnapshot.go` | `MergeFeatures` — the one path all features go through |
| `ingestion/aggregated/aggregate.go` | NodeSnapshots (forward-fill) + SNAPSHOT_EDGE |
| `ingestion/pipeline.go` | `Run(ctx,cfg)` orchestration (core always; creds-gated optional) |
| `api/main.go` | `POST /api/v1/ingest`, `GET /api/v1/ingest/status` |

## The one rule that bites: static vs time-varying features

A seed file being a static *input* does NOT make its feature static. Only
`nuclear_flag` and `coastline_flag` (§9a) live on the Country identity node.
Everything that changes over the window — `sanctions_status`, `unsc_seat_flag`,
`financial_resources_tier`, and every World Bank / V-Dem / SIPRI / UNDP / ACLED
value — is written per `time_step` via `raw.MergeFeatures(...)` as a
`FeatureSnapshot`. Putting a time-varying feature on the identity node reintroduces
temporal leakage and is the bug the integration test guards against.

## Source matrix

| Source | Free? | Auth / input | Writes |
|---|---|---|---|
| World Bank country list | yes, no key | — | all Country nodes |
| World Bank indicators | yes, no key | — | gdp/pop/trade/land/stability FeatureSnapshots |
| Wikidata SPARQL | yes, no key | — | region, BORDERS, MEMBER_OF, IGO actors |
| JSON seeds | yes | repo | nuclear/coastline (node) + unsc/sanctions/tier (FeatureSnapshots) |
| GDELT | yes | BigQuery ADC (`GDELT_GCP_PROJECT`) | `aggregated` (default): SNAPSHOT_EDGEs in agg DB; `raw`: EVENT edges in raw DB |
| ACLED | yes | OAuth (`ACLED_EMAIL`/`ACLED_PASSWORD`) | conflict features |
| V-Dem / SIPRI / UNDP | yes | downloaded file path env | polyarchy/milex/hdi features |

The credential-free core (country list + WB + Wikidata + seeds) always runs; each
other source activates only when `cfg.<Source>Enabled()` is true, else it is
reported as `skipped` in the run Result.

## Run + verify

```bash
docker compose -f infra/docker/docker-compose.yml up -d   # two Neo4j containers
go test ./tests/...                                        # unit (parsers, mapping)
go test -tags=integration -count=1 ./tests/...            # vs dockerized DBs
go run ./api && curl -X POST localhost:8080/api/v1/ingest # ingest + aggregate
```

Integration tests assert the static/time-varying split, forward-fill,
conflict-priority dominance, Brexit point-in-time `MEMBER_OF`, and the Wikidata
loader (fixture SPARQL). If one fails, the message names the offending value.

## Gotchas

- Two containers, not one DB (Neo4j Community is single-database). raw=7687, agg=7688.
- GDELT actor codes are **ISO-3** (`USA`/`GBR`/`DEU`/`RUS`/`TWN`), verified against the
  live `gdelt-bq.full.events` (1.0, 2010–2015) and `gdelt-bq.gdeltv2.events` (2.0) —
  *not* FIPS (GMY/UKG). `gdelt-bq.gdeltv1.events` does NOT exist. `configs/cameo_country_to_iso3.json`
  is identity over ingested countries + aliases (ROM→ROU); unmapped/regional codes (EUR)
  are dropped. Taiwan is WB-omitted but Wikidata-seeded — add a missing country with
  `go run tools/add_gdelt_country.go <ISO3>` (surgical) and seed it in `tools/gen_cameo.go`.
- Structural edges are temporal; always filter with
  `start_time_step <= T AND (end_time_step IS NULL OR end_time_step > T)`.
- Re-running is safe (all writes MERGE). Wipe with `MATCH (n) DETACH DELETE n` on
  both DBs to reset counts.
- GDELT full 2010–present is ~2.4M dyad-month SNAPSHOT_EDGEs in `aggregated` mode
  (default; aggregates in BigQuery, fits a local disk). `raw` mode loads 16–48 GB of
  EVENT edges and needs ample disk; `GDELT_MAX_ROWS` caps it (raw mode only).
