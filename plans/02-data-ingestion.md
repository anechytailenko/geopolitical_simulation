# Data Ingestion Plan — geopolitic_raw Population

## Context

This plan details every external data source, which node features each source populates, exactly how to fetch and parse it inside the Go ingestion service, and how records are written to `geopolitic_raw`. It also estimates total database size and recommends a free cloud storage strategy for backups and ML artifacts.

All fetching and writing is performed by the Go API service, triggered by `POST /api/v1/ingest`. No Python is involved in data ingestion.

---

## Where each dataset physically lives (and how to inspect it)

The pipeline writes to **three physically distinct stores**. Knowing which is which
avoids the most common confusion (looking for event relations in the wrong database):

| Dataset | Lives in | Contains / how to reach it |
|---|---|---|
| **Raw GDELT events** (every CAMEO event) | **BigQuery (Google Cloud)** — `gdelt-bq.full.events` (2010→2015) + `gdelt-bq.gdeltv2.events` (2015→present) | Queried at ingest time and aggregated server-side; in the default `aggregated` mode the raw events are **never copied into Neo4j**. |
| **`geopolitic_raw`** (Neo4j) | container `geopolitic-neo4j-raw` — Bolt **`bolt://localhost:7687`**, browser **http://localhost:7474** | Identity nodes, `FeatureSnapshot`s (`HAS_FEATURES`), and temporal `BORDERS`/`MEMBER_OF`. **No `SNAPSHOT_EDGE` and no `EVENT`** in aggregated mode. |
| **`geopolitic_aggregated`** (Neo4j) — the ML input | container `geopolitic-neo4j-aggregated` — Bolt **`bolt://localhost:7688`**, browser **http://localhost:7475** | `NodeSnapshot`s and **`SNAPSHOT_EDGE` (~2.4M, the country↔country relations the model predicts)**, plus the **same 221 Country + 11 Actor identity nodes** mirrored from raw (each with NodeSnapshots) and mirrored `BORDERS`/`MEMBER_OF`. |

> Both containers serve a Neo4j Browser, but on **different host ports** (raw = 7474,
> aggregated = 7475) because Community edition is single-database.

### Symptom & fix: "The provided relationship type is not in the database: SNAPSHOT_EDGE"

**Cause.** The query was run against **http://localhost:7474** — the **raw** DB browser,
which legitimately has **no** `SNAPSHOT_EDGE` (raw holds only features + structural
edges). The 2.4M `SNAPSHOT_EDGE`s live in the **aggregated** DB (`bolt://localhost:7688`).
Note `GET /api/v1/ingest/status` reads the `IngestState` node from the **raw** DB, so it
reports `agg_snapshot_edges` as *recorded metadata* — it describes the aggregated DB
without the edges being stored in raw, which is why the count looks present but the
browser query "finds nothing".

**Fix — reliable (cypher-shell, no port ambiguity):**
```bash
docker exec geopolitic-neo4j-aggregated cypher-shell -u neo4j -p geopolitic \
"MATCH ()-[r:SNAPSHOT_EDGE]->() RETURN r.dominant_class, count(*) ORDER BY count(*) DESC"
```

**Fix — Neo4j Browser:** open **http://localhost:7475/browser/** and in the Connect
dialog set the **Connection URL to `bolt://localhost:7688`** (user `neo4j`, password
`geopolitic`). ⚠️ Neo4j Browser **defaults the URL to `bolt://localhost:7687`**, which is
the *raw* container — leaving the default keeps you on the raw DB even when the page was
loaded from `:7475`. Always point it at `7688` for the aggregated DB.

---

## Feature-to-Source Mapping

### Country Node — Time-Varying Features

| Feature | Source | Source field / indicator |
|---|---|---|
| `region` | Wikidata | P30 (continent) |
| `land_area_log` | World Bank API | `AG.LND.TOTL.K2` (land area, sq km) |
| `coastline_flag` | Static seed | `configs/landlocked_countries.json` (only ~44 landlocked states; stable) |
| `neighbor_count` | Derived | Count of `[:BORDERS]` edges valid at `time_step` |
| `gdp_log` | World Bank API | `NY.GDP.MKTP.CD` |
| `gdp_per_capita` | World Bank API | `NY.GDP.PCAP.CD` |
| `trade_openness_index` | World Bank API | `NE.TRD.GNFS.ZS` (trade as % GDP) |
| `sanctions_status` | `configs/sanctions_registry.json` (per-year) → **FeatureSnapshot** | Time-varying: a country enters/exits sanction regimes over the window, so written per `time_step`, never as a static node flag |
| `population_log` | World Bank API | `SP.POP.TOTL` |
| `population_growth` | World Bank API | `SP.POP.GROW` |
| `political_stability` | World Bank API | `PV.EST` (WGI: Political Stability and Absence of Violence) |
| `hdi` | UNDP HDR | Annual CSV download (hdr.undp.org/data-center) |
| `vdem_polyarchy_score` | V-Dem | `v2x_polyarchy` column |
| `years_since_leadership_change` | V-Dem | `v2exnamhos` (head-of-state name; year changes detected by Go) |
| `military_expenditure_log` | SIPRI | Milex Excel, **Constant US$** sheet → `log(value+1)` |
| `nuclear_flag` | Static seed | 9 countries: USA, RUS, GBR, FRA, CHN, IND, PAK, PRK, ISR |
| `alliance_memberships` (multi-hot) | Derived | From temporal `[:MEMBER_OF]` edges valid at the queried `time_step` |
| `active_conflict_count` / `conflict_intensity` | ACLED (derived) | Count + fatalities of ACLED events per country per month |
| `unsc_seat_flag` | `configs/unsc_schedule.json` → **FeatureSnapshot** | Time-varying: non-permanent seats rotate on 2-year terms, so written per `time_step` (P5 set every step; elected members only during their term) |

### Actor Node — Time-Varying Features

| Feature | Source | Source field |
|---|---|---|
| `member_count_log` | Wikidata | P527 (has part / member) count, or manual for major IOs |
| `financial_resources_tier` | `configs/actor_financial_tier.json` (per-year) → **FeatureSnapshot** | Time-varying: 1–5 ordinal updated annually from public budgets, written per `time_step` on the Actor's FeatureSnapshot |
| `recognized_legitimacy_score` | Wikidata | Count of countries with P17 (country of origin) or P463 linking to this actor |

The Actor set is the **11 IGOs** sourced from Wikidata (UN, EU, NATO, ASEAN, African
Union, SCO, OAS, CSTO, Arab League, WTO, Interpol) — this is the complete actor universe
in both DBs. **Armed-group actors are deliberately not created** (ACLED feeds the
per-country `conflict_intensity`/`active_conflict_count` node feature, not Actor nodes —
see §7). So `count(Actor) = 11` in both `geopolitic_raw` and `geopolitic_aggregated`.

### Static / Structural Data

| Edge or field | Source |
|---|---|
| `[:BORDERS {start_time_step, end_time_step}]` between countries | Wikidata P47 (shares border with) + P580/P582 qualifiers |
| `[:MEMBER_OF {start_time_step, end_time_step}]` country→actor | Wikidata P463 (member of) + P580/P582 qualifiers |
| `[:MEMBER_OF {start_time_step, end_time_step}]` actor→actor | Wikidata P463 (e.g. EU member of UN) + P580/P582 |
| Actor identity + type (IGOs / intl orgs) | Wikidata type hierarchy (QIDs to verify at implementation) |
| Actor identity + type (armed groups) | ACLED actor list → `configs/acled_actor_lookup.json` |

Structural edges are **temporal** (validity intervals), not static — see §Wikidata for how P580/P582 → `start_time_step`/`end_time_step`. Where a statement has no date qualifier, default `start_time_step = 0` (epoch) and `end_time_step = null`.

The **country↔country event relationship** (the model's prediction target) is **not** a
raw structural edge: in the default `aggregated` mode it is the monthly `SNAPSHOT_EDGE`
written to `geopolitic_aggregated` (see §6 GDELT modes and §"Where each dataset lives").
`BORDERS`/`MEMBER_OF` above are the only country/actor relations in the raw DB.

---

## Source-by-Source Ingestion Steps

### 1. World Bank API

**URL pattern:** `https://api.worldbank.org/v2/country/{ISO2}/indicator/{INDICATOR}?format=json&per_page=200&date=2010:2026`

**Access:** Public REST API, no key required. Rate limit: ~100 req/min.

**Features populated (7 indicators):** `land_area_log` (`AG.LND.TOTL.K2`), `gdp_log` (`NY.GDP.MKTP.CD`), `gdp_per_capita` (`NY.GDP.PCAP.CD`), `trade_openness_index` (`NE.TRD.GNFS.ZS`), `population_log` (`SP.POP.TOTL`), `population_growth` (`SP.POP.GROW`), `political_stability` (`PV.EST`)

**Loading steps (Go):**
1. Iterate over all 195 country ISO-2 codes × 7 indicators = 1,365 requests (batch into goroutines, respect rate limit). Use `date=2010:2026` to pull the full study window in one call per (country, indicator)
2. Parse JSON response: `[{year, value}]` per indicator per country
3. For each (country, year) observation: compute `time_step = (year - 2010) * 12` (January of that year)
4. Write/MERGE a `(:FeatureSnapshot {node_id: ISO3, node_type: "Country", time_step, year})` node in `geopolitic_raw` and set the indicator value
5. Forward-fill (carrying last value to subsequent months) is handled at the aggregation step, not here

**Frequency:** Annual. Run once per year.

**Notes:** Use ISO-2 for the API call, store as ISO-3 internally. World Bank country list endpoint (`/v2/country?format=json&per_page=300`) provides the ISO-2 → ISO-3 mapping. The `NE.TRD.GNFS.ZS` indicator already gives total trade as % of GDP (no UN Comtrade needed). `PV.EST` is the Worldwide Governance Indicators "Political Stability" estimate, served by the same API — it replaces the earlier mislabeled use of V-Dem civil liberties as a stability proxy.

---

### 2. V-Dem Dataset

**URL:** `https://v-dem.net/data/the-v-dem-dataset/` (direct download, free with registration; ~150MB CSV)

**Access:** One-time download + annual update. No API. Requires free account.

**Features populated:** `vdem_polyarchy_score`, `years_since_leadership_change` (political stability now comes from World Bank `PV.EST`, not V-Dem)

**Loading steps (Go):**
1. Download `V-Dem-CY-Full+Others-v14.csv` (or latest) to local volume on first ingest
2. Parse CSV with Go's `encoding/csv`; relevant columns: `country_text_id` (this **is** the ISO-3166-1 alpha-3 code — no Correlates-of-War mapping needed; `COWcode` is a separate column), `year`, `v2x_polyarchy`, `v2exnamhos`
3. Detect leadership changes: for each country, compare `v2exnamhos` (head-of-state name) string between consecutive years; if changed → `years_since_leadership_change = 0`, else increment
4. MERGE the `(:FeatureSnapshot {node_id, node_type: "Country", time_step})` for each (country, year) — joins the World Bank snapshot at the same (node\_id, time\_step)

**Frequency:** Annual. V-Dem releases a new version each spring.

---

### 3. SIPRI Military Expenditure Database

**URL:** `https://www.sipri.org/databases/milex` (manual Excel download, free, no account)

**Access:** Direct Excel download. No API. File: `SIPRI-Milex-data-1949-{year}.xlsx`

**Features populated:** `military_expenditure_log`

**Loading steps (Go):**
1. Use `github.com/xuri/excelize/v2` Go library to parse the Excel file
2. Sheet of interest: **"Constant (20xx) US$"** — absolute spending, comparable across years; matches the `_log` feature name
3. Rows = countries (SIPRI uses country names → maintain a `configs/sipri_name_to_iso3.json` mapping table)
4. Columns = years (1949–present; only 2010+ is used)
5. For each (country, year) with a non-null value: write `military_expenditure_log = log(value + 1)` to the country's `(:FeatureSnapshot {node_id, node_type: "Country", time_step})`

**Frequency:** Annual. SIPRI releases updated data each April.

---

### 4. UNDP Human Development Report

**URL:** `https://hdr.undp.org/data-center/documentation-and-downloads` (CSV download, free, no account)

**Access:** Direct CSV/Excel download. No API.

**Features populated:** `hdi`

**Loading steps (Go):**
1. Download CSV; columns: country name, HDI value per year
2. Map UNDP country names to ISO-3 via `configs/undp_name_to_iso3.json`
3. Write HDI value to the country's `(:FeatureSnapshot {node_id, node_type: "Country", time_step})` for the corresponding year
4. Leave null where HDI is unavailable; the forward-fill pass at aggregation time will carry the last known value

**Frequency:** Annual. UNDP releases each September.

---

### 5. Wikidata (SPARQL)

**Endpoint:** `https://query.wikidata.org/sparql`

**Access:** Public SPARQL endpoint, no key. Rate limit: ~60 req/min. Large queries must paginate with `LIMIT`/`OFFSET`.

**Features populated:** `region` (Country); Actor identity + type; temporal `[:BORDERS]` edges; temporal `[:MEMBER_OF]` edges. (`coastline_flag` is **not** from Wikidata — it comes from the static `configs/landlocked_countries.json` seed. The earlier P17-based landlocked query was incorrect: P17 is "country", and landlocked status would be `wdt:P31 wd:Q123480`, but a curated static list of ~44 states is simpler and more reliable.)

**Use P298 (ISO 3166-1 alpha-3), not P297 (alpha-2)** — `Country.id` is ISO-3.

**Temporal edges via qualifiers:** P47 (borders) and P463 (member of) statements may carry `pq:P580` (start time) / `pq:P582` (end time). Query through the statement node (`p:`/`ps:`/`pq:`) and convert each date to `time_step` (`(year-2010)*12 + (month-1)`, clamped to ≥ 0 — edges that began before 2010 clamp to `start_time_step = 0`). Missing start → `start_time_step = 0`; missing end → `end_time_step = null` (still valid).

**Loading steps (Go):**

*Countries — region:*
```sparql
SELECT ?country ?iso3 ?continentLabel WHERE {
  ?country wdt:P31 wd:Q6256 ; wdt:P298 ?iso3 .
  OPTIONAL { ?country wdt:P30 ?continent .
             ?continent rdfs:label ?continentLabel FILTER(LANG(?continentLabel)="en") }
}
```
→ Write `region` (continent → one-hot index) to Country node static properties.

*Temporal BORDERS edges:*
```sparql
SELECT ?c1iso ?c2iso ?start ?end WHERE {
  ?c1 wdt:P31 wd:Q6256 ; wdt:P298 ?c1iso ; p:P47 ?stmt .
  ?stmt ps:P47 ?c2 .
  ?c2 wdt:P31 wd:Q6256 ; wdt:P298 ?c2iso .
  OPTIONAL { ?stmt pq:P580 ?start }
  OPTIONAL { ?stmt pq:P582 ?end }
}
```
→ Write `(:Country)-[:BORDERS {start_time_step, end_time_step}]->(:Country)`.

*Actor nodes + temporal MEMBER_OF edges:*
```sparql
SELECT ?org ?orgLabel ?orgType ?memberIso ?start ?end WHERE {
  VALUES ?orgType { wd:Q484652 wd:Q43229 }   # international organization, organization — verify QIDs
  ?org wdt:P31 ?orgType ; rdfs:label ?orgLabel FILTER(LANG(?orgLabel)="en") .
  ?member p:P463 ?stmt .
  ?stmt ps:P463 ?org .
  ?member wdt:P298 ?memberIso .
  OPTIONAL { ?stmt pq:P580 ?start }
  OPTIONAL { ?stmt pq:P582 ?end }
}
```
→ Write `(:Actor)` nodes and `(:Country)-[:MEMBER_OF {start_time_step, end_time_step}]->(:Actor)` edges. Armed-group actors are sourced from ACLED, not here.

**Frequency:** Once for seed. Re-run annually to pick up new memberships/border changes.

---

### 6. GDELT 1.0 + 2.0

GDELT 2.0's 15-minute export files only start **2015-02-18**. The study window is 2010–present, so two GDELT products are needed:

- **GDELT 1.0** — coverage 1979→2015. The BigQuery table is **`gdelt-bq.full.events`** (one uniform table). Use for **2010-01 → 2015-02**. (Note: `gdelt-bq.gdeltv1.events` does **not** exist — verified against the live project; `full.events` is the GDELT 1.0 historical events table, ~879M rows.)
- **GDELT 2.0** — `gdelt-bq.gdeltv2.events` (~898M rows), 2015→present.

**BigQuery — the selected event source for the project.** `gdelt-bq.full.events` (2010-01→2015-02-17 segment) and `gdelt-bq.gdeltv2.events` (2015-02-18+) — GCP free tier gives 1 TB queries/month. The two filtered segments scan ~89 GB combined, well within the free tier. Because `full.events` reaches back to 2010, GDELT alone covers the full 2010–present window (so ICEWS, §8, is only an optional cross-check). Auth uses Application Default Credentials; set `GDELT_GCP_PROJECT` (and either `gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`).

**Two ingest modes (`GDELT_MODE`):**
- **`aggregated` (default)** — the heavy aggregation runs **server-side in BigQuery**: one row per (source, target, month) with all SNAPSHOT_EDGE features (event_count, weighted_intensity, sentiment mean/std, per-class counts → dominant_class + class_distribution). Only those ~2.4M compact `SNAPSHOT_EDGE`s are written, **directly into `geopolitic_aggregated`** (~1.5–2 GB). Raw EVENT edges stay in BigQuery as the regenerable source of truth. This is the scalability mitigation (§Storage Estimates) and the only feasible mode on a disk-constrained local box. The per-event CAMEO→class CASE in SQL mirrors `internal/label.Classify` exactly; ISO-3 alias remaps (`ROM`→`ROU`, …) are combined in Go from per-group sums so collapsed dyads stay exact.
- **`raw`** — streams every country↔country EVENT edge into `geopolitic_raw` (16–48 GB; only with ample disk). Used when the strict "all raw events in Neo4j" design is wanted.

**Access:** Bulk CSV ZIP download, or BigQuery REST API. Both schemas share the same column layout for the fields below.

**Features populated:** `[:EVENT]` edges (CAMEO-coded events between country pairs)

**Column mapping:**

| GDELT column | Maps to |
|---|---|
| `GlobalEventID` | `event_id` |
| `Day` (YYYYMMDD) | `timestamp`; `time_step = (year-2010)*12 + (month-1)` |
| `Actor1CountryCode` / `Actor2CountryCode` | source/target — **ISO-3 country codes** (verified against live GDELT exports), validated/mapped via `configs/cameo_country_to_iso3.json` |
| `EventCode` (CAMEO) | `event_type` → label generator → `relationship_class` |
| `GoldsteinScale` | `goldstein_scale`; `intensity_score` = GoldsteinScale / 10 |
| `NumArticles` | `source_count` |
| `AvgTone` | `sentiment_score` = AvgTone / 10, clamped to [−1, 1] |

> **Verified empirically:** GDELT's `Actor1CountryCode`/`Actor2CountryCode` use **ISO-3 country codes** (Germany = `DEU`, UK = `GBR`, France = `FRA`), confirmed by sampling live GDELT 1.0 and 2.0 export files — *not* the FIPS-style CAMEO codes (GMY/UKG/FRN) sometimes assumed. The lookup `configs/cameo_country_to_iso3.json` is therefore (a) identity over the ISO-3 countries we ingest and (b) a filter: regional/sub-national codes (`EUR`, `AFR`, `WSB`, `GZS`, …) and a few legacy variants (`ROM`→`ROU`) are handled, and any code not in the map is dropped. Regenerate with `go run tools/gen_cameo.go`.

**Filtering (before Neo4j write):**
- Both actor country codes must convert to ISO-3 in the known 195-country set
- Deduplicate on `GlobalEventID` (GDELT re-emits events across files)

**Loading steps (Go), `aggregated` mode (default):**
1. Run the aggregation query against `gdelt-bq.full.events` (start→2015-02-17) and `gdelt-bq.gdeltv2.events` (2015-02-18→end). The query filters `Actor1/2CountryCode IN @codes` (the cameo-map country set, dropping `EUR`/`AFR`/… server-side), classifies each event with a CAMEO+Goldstein CASE, and `GROUP BY (src, tgt, month)` emitting per-group **sums + per-class counts**.
2. Stream the ~2.4M aggregated rows, remap ISO-3 aliases via `configs/cameo_country_to_iso3.json`, and combine collapsed dyads in Go (exact, because sums combine additively).
3. Finalize each group into a `SNAPSHOT_EDGE` (weighted_intensity, sentiment mean/std, dominant_class, class_distribution, days_since_last_event) and UNWIND-batch them into `geopolitic_aggregated`. Runs **after** `Build` so the mirrored Country nodes exist.

**Loading steps (Go), `raw` mode:** stream filtered rows from the same two tables, convert codes, run the label generator, and batch ~5,000 `[:EVENT]` writes (`data_source = "GDELT"`) into `geopolitic_raw`.

**Volume:** All ~195 countries, 2010–present, country↔country pairs → `raw` mode is **~40–120M EVENT edges (~16–48 GB)**; `aggregated` mode is **~2.4M SNAPSHOT_EDGEs (~1.5–2 GB)** (gdeltv2 ≈ 1.78M dyad-months + full.events ≈ 0.62M). See §Storage Estimates.

**Frequency:** Historical backfill once. Incremental on each ingest trigger.

---

### 7. ACLED

**API (OAuth, since 2025):** obtain a 24-hour access token by POSTing `username` (email), `password`, `grant_type=password`, `client_id=acled`, `scope=authenticated` to `https://acleddata.com/oauth/token`; then read from `https://acleddata.com/api/acled/read?_format=json&country={COUNTRY}&year={YEAR}&limit=5000` with header `Authorization: Bearer {token}`.

**Access:** Free myACLED account (email + password); set `ACLED_EMAIL` / `ACLED_PASSWORD`. Legacy `key`+`email` API keys were deprecated after 2025-09-15.

**Role:** ACLED events are overwhelmingly **intrastate** (battles, explosions, protests, riots between armed groups / governments / civilians *inside* one country) and rarely form a clean source-country→target-country dyad. So ACLED is **not** a source of Country→Country `[:EVENT]` edges; instead it populates the **node conflict-intensity feature** for each country.

**Features populated:** `active_conflict_count`, `conflict_intensity` on `(:FeatureSnapshot)` per country per month.

**ACLED field mapping (aggregation):**

| ACLED field | Used for |
|---|---|
| `country` | the affected Country node (the country the event occurs in) |
| `event_date` | `time_step` bucket |
| `event_type` | filter to violent types (battles, explosions/remote violence, violence against civilians) |
| `fatalities` | summed per country-month → `conflict_intensity = log(sum_fatalities + 1)` |
| `actor1`, `actor2` | `configs/acled_actor_lookup.json` — used only for the *optional* interstate-edge case below |

**Loading steps (Go):**
1. Iterate over country list × year range, paginate with `offset` (5,000 rows/page)
2. Aggregate per (country, month): `active_conflict_count` = number of violent events; `conflict_intensity` = `log(Σ fatalities + 1)`
3. Write these onto the country's `(:FeatureSnapshot {node_id, node_type: "Country", time_step})`
4. **Optional:** when both `actor1` and `actor2` resolve to *states* (a genuine interstate incident), also write a MATERIAL\_CONFLICT `[:EVENT]` edge with `data_source = "ACLED"`

**Frequency:** Weekly for recent data.

---

### 8. ICEWS (Harvard Dataverse)

**URL:** `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075`

**Access:** Free Harvard Dataverse account required. Annual tab-separated files.

**Role: OPTIONAL cross-check** (off by default). The selected event source is GDELT via BigQuery, whose `gdelt-bq.full.events` table already covers 2010–2015, so GDELT spans the entire window on its own. ICEWS remains a clean, well-attributed CAMEO source and can be enabled as a cross-check / dedup partner for 2010–2014, but it is **no longer required**. When enabled, the GDELT-vs-ICEWS dedup rule below applies.

**Features populated:** `[:EVENT]` edges with CAMEO coding.

**ICEWS field mapping:** Same CAMEO coding as GDELT. `story_id`/`Event ID` → `event_id`, `event_date` → `timestamp` + `time_step`, `Source Country` / `Target Country` → source/target ISO-3 via `configs/icews_country_to_iso3.json` (ICEWS stores full country **names**, not codes).

**Loading steps (Go):**
1. Download annual TSV files to local volume
2. Parse tab-separated rows; map country names → ISO-3; apply the same CAMEO → relationship\_class label generator as GDELT
3. Write `[:EVENT]` edges with `data_source = "ICEWS"`
4. Duplicate resolution: if the same (actors, date, CAMEO code) exists from GDELT, keep GDELT (higher source-count reliability), discard the ICEWS record

**Frequency:** Annual download of the new year's file.

---

### 9. Seed Data (Go-managed JSON configs)

Versioned JSON files in the repo, loaded at startup. **Critical distinction:** a
seed file being a static *input* does **not** make the feature it produces
static. Each file below is tagged by where its values are written.

**9a. Truly static → Country/Actor identity node (one value, never changes in-window):**

| File | Writes | Why static |
|---|---|---|
| `configs/nuclear_states.json` | `nuclear_flag` on Country | No country in the 2010+ set crosses the threshold within the window; re-stored per snapshot only for uniform schema |
| `configs/landlocked_countries.json` | `coastline_flag` on Country | Geography is fixed (the one in-window exception, South Sudan 2011, is handled as a new node, not a flag flip) |

**9b. Time-varying → written per `time_step` as `FeatureSnapshot` rows (value changes over the window):**

| File | Writes (per time_step) | Why time-varying |
|---|---|---|
| `configs/unsc_schedule.json` | `unsc_seat_flag` | Non-permanent seats rotate on 2-year terms; P5 set every step, elected members only during their term. Source: un.org/en/sc/members, 2008–present (covers the rolling window before the epoch) |
| `configs/sanctions_registry.json` | `sanctions_status` | Countries enter/exit UN/EU/US-OFAC sanction regimes; one entry per (country, year) |
| `configs/actor_financial_tier.json` | `financial_resources_tier` (Actor) | 1–5 ordinal updated annually from public budgets |

**9c. Lookup maps (static, used during parsing — never become node properties):**

| File | Contents |
|---|---|
| `configs/cameo_country_to_iso3.json` | **GDELT** country code → ISO-3 (ISO-3 identity over ingested countries + legacy variants; generated by `tools/gen_cameo.go`) |
| `configs/icews_country_to_iso3.json` | **ICEWS** full country name → ISO-3 |
| `configs/sipri_name_to_iso3.json` | SIPRI country name → ISO-3 mapping |
| `configs/acled_actor_lookup.json` | ACLED actor name → Country/Actor node ID mapping |
| `configs/undp_name_to_iso3.json` | UNDP country name → ISO-3 mapping |

**Loading steps (Go):**
- **9a** → MERGE the flag directly onto the Country/Actor identity node.
- **9b** → for each time-keyed entry, MERGE a `(:FeatureSnapshot {node_id, node_type, time_step})` and set the feature; for terms/years spanning multiple months, write the value at every `time_step` in range (the aggregation forward-fill then carries it). **Never** set these as a single identity-node property — doing so would erase their evolution and reintroduce the temporal leakage the snapshot design exists to prevent.
- **9c** → held in memory; consulted by the GDELT/ICEWS/SIPRI/ACLED/UNDP parsers.

---

## Ingest Ordering and Dependencies

The Go ingestion pipeline must run in this order:

```
Step 1 — Seed data (configs/*.json)
  → Creates Country nodes (all ~195, from the live World Bank country list), Actor nodes
  → Sets ONLY the truly-static identity flags: nuclear_flag, coastline_flag (§9a)
  → (unsc_seat_flag, sanctions_status, financial_resources_tier are NOT set here —
     they are time-varying and written as FeatureSnapshots in step 10, see §9b)

Step 2 — Wikidata (after Country nodes exist)
  → Writes region to Country nodes
  → Creates temporal [:BORDERS {start_time_step, end_time_step}] edges
  → Creates IGO/org Actor nodes + temporal [:MEMBER_OF {start, end}] edges

Steps 3–6 — Feature snapshot sources → (:FeatureSnapshot{node_type:"Country"})  (parallel, after step 1)
  3. World Bank API   → gdp_log, gdp_per_capita, trade_openness_index,
                        population_log, population_growth, land_area_log (AG.LND.TOTL.K2),
                        political_stability (PV.EST)
  4. V-Dem            → vdem_polyarchy_score, years_since_leadership_change
  5. SIPRI            → military_expenditure_log (Constant US$ sheet)
  6. UNDP HDR         → hdi

Step 7 — ACLED  →  node conflict feature on (:FeatureSnapshot)
  → active_conflict_count, conflict_intensity per country per month
  → (optional) interstate MATERIAL_CONFLICT [:EVENT] edges when both actors are states

Step 8 — GDELT via BigQuery (full.events 2010–2015 + gdeltv2.events 2015–present)
  → default `aggregated` mode: aggregate server-side → write SNAPSHOT_EDGEs into
    geopolitic_AGGREGATED (runs after Build); `raw` mode: write [:EVENT] into geopolitic_raw
  → bulk; full-window backfill takes longest — run as a background job
  → server-side filter to country↔country pairs; CAMEO→ISO-3 conversion required

Step 9 — ICEWS (optional, off by default)  →  [:EVENT] edges
  → cross-check / dedup partner for 2010–2014 only; country name→ISO-3 conversion

Step 10 — Derived + time-varying-seed feature computation (runs last)
  → neighbor_count: count [:BORDERS] edges valid at each time_step → FeatureSnapshot
  → sanctions_status: from sanctions_registry.json, one FeatureSnapshot per (country, year) (§9b)
  → unsc_seat_flag: from unsc_schedule.json, FeatureSnapshot per (country, time_step) — P5 every
    step, elected members only within their 2-year term (§9b)
  → financial_resources_tier: from actor_financial_tier.json, FeatureSnapshot per (actor, year) (§9b)
```

Steps 3–6 have no internal ordering constraints and can be goroutine-parallelized; steps 7–9 likewise. Step 10 must run after steps 2 (borders) and 7–9. Actor feature snapshots (`member_count_log`, `financial_resources_tier`, `recognized_legitimacy_score`) are written during steps 2 + static seed onto `(:FeatureSnapshot{node_type:"Actor"})`.

---

## Storage Estimates

### geopolitic_raw

Scope: **195 countries**, **~197 months** (2010-01 → present), **all-country** GDELT coverage.

| Content | Count | Avg size | Total |
|---|---|---|---|
| Country identity nodes | 195 | ~200 bytes | ~39 KB |
| Actor identity nodes | ~500 | ~300 bytes | ~150 KB |
| FeatureSnapshot nodes (Country + Actor) | (195+500) × 197 ≈ 137K | ~500 bytes | ~68 MB |
| [:BORDERS] temporal edges | ~600 (incl. change episodes) | ~120 bytes | ~70 KB |
| [:MEMBER_OF] temporal edges | ~3,500 (incl. change episodes) | ~120 bytes | ~420 KB |
| [:EVENT] edges — GDELT 1.0+2.0 (2010–present, all 195) | **~40–120M** | ~400 bytes | **~16–48 GB** |
| [:EVENT] edges — ICEWS (2010–2014, optional cross-check) | ~3–6M | ~400 bytes | ~1.5–2.5 GB |
| Neo4j page cache + index overhead | — | — | ~1.5–4 GB |

**Total geopolitic_raw: ~20–55 GB** — completely dominated by GDELT `[:EVENT]` edges. (ACLED no longer adds edges; it feeds node features.)

> **Scalability caveat (important at this scale).** Loading 50–150M raw GDELT events into Neo4j is heavy for a local research box — write throughput, page cache, and backup size all suffer. Two mitigations, decided at implementation:
> - **(a) Big disk:** run Neo4j on a fast local/external SSD with ≥100 GB free and a large page cache; acceptable if you want strict "all raw events in Neo4j".
> - **(b) Columnar raw store (recommended):** keep the event-level source of truth in **Parquet on disk queried with DuckDB**, or pre-aggregate in **BigQuery**, and load only the monthly `SNAPSHOT_EDGE` aggregates (+ an optional filtered raw subset, e.g. only non-STATUS_QUO events) into `geopolitic_raw`. This keeps Neo4j in the low-single-digit GB range while preserving a full, reproducible event archive. It is a deliberate deviation from the strict two-Neo4j-DB design and should be flagged in `01-architecture.md` if adopted.

### geopolitic_aggregated

| Content | Count | Avg size | Total |
|---|---|---|---|
| NodeSnapshot nodes | (195+500) × 197 ≈ 137K | ~600 bytes | ~82 MB |
| [:SNAPSHOT_EDGE] — active directed pairs (~12% of 37,830 possible = ~4,500/month) | 4,500 × 197 ≈ 887K | ~800 bytes | ~0.7 GB |
| [:BORDERS], [:MEMBER_OF] mirrored | ~4,100 | ~120 bytes | ~490 KB |
| Index overhead | — | — | ~150–300 MB |

**Total geopolitic_aggregated: ~1.5–2 GB** — stays manageable, and this is the only DB the ML training job and frontend actually read. (No `TimeStep` nodes — `time_step` is an indexed integer property.)

### ML Artifacts (local)

| Content | Size |
|---|---|
| Parquet training export (compressed, aggregated DB) | ~0.5–2 GB |
| Model weights per version (PyTorch .pt) | ~50–200 MB |
| Scaler + calibrator pickles | ~1–5 MB |
| metrics.json per run | negligible |

**Per model version: ~0.6–2 GB**

### Grand Total (local disk)

| Component | Estimated size |
|---|---|
| geopolitic_raw (Neo4j, strict design) | 20–55 GB |
| geopolitic_aggregated | 1.5–2 GB |
| ML artifacts (3 versions retained) | 2–6 GB |
| Raw source files (GDELT/ICEWS archives, SIPRI xlsx, V-Dem CSV) | 6–12 GB |
| **Total (strict "all raw in Neo4j")** | **~30–75 GB** |
| **Total (columnar raw store, mitigation b)** | **~10–25 GB** |

With the recommended columnar-raw mitigation the Neo4j footprint drops to ~1.5–2 GB (aggregated only) plus the Parquet/DuckDB event archive (~8–18 GB), which is far easier to host and back up.

---

## Free Cloud Storage Recommendation

At full scale the `geopolitic_raw` dump (~7–18 GB compressed) is too large for the free tiers — and unnecessary to back up, because it is **fully regenerable** from the archived GDELT/ICEWS source files (which are themselves permanently hosted by GDELT and Harvard Dataverse). So back up only what is expensive or impossible to recreate: (1) the **aggregated DB dump**, (2) versioned **Parquet training exports**, (3) **model artifacts**.

### Recommended: Three-service combination (all free)

| Service | Free tier | Primary use |
|---|---|---|
| **Cloudflare R2** | 10 GB + **zero egress fees** | `geopolitic_aggregated` dump (~1–2 GB compressed) + model artifacts |
| **Hugging Face Datasets** | 50 GB/repo (public) | Parquet training exports; version-controlled, ML-tooling-friendly |
| **Google Drive** | 15 GB | Source files / configs: SIPRI xlsx, V-Dem CSV, name-mapping JSONs |

**Backup footprint: ~5–10 GB**, comfortably within these free tiers. `geopolitic_raw` is intentionally **not** backed up (regenerate via `POST /api/v1/ingest`).

**Why Cloudflare R2 for dumps:** Zero egress charges (S3 and GCS charge per download). S3-compatible API means Go can use `aws-sdk-go-v2` with an endpoint override — no new SDK needed. 10 GB comfortably holds 2–3 compressed Neo4j dumps.

**Why Hugging Face for Parquet:** Native dataset versioning (Git-LFS backed), accessible to Python training scripts via `datasets` library without a custom download step, public repos have 50 GB limit per repo.

### Neo4j dump + upload (triggered via Go as shell subprocess)

```bash
# dump the aggregated DB only (raw is regenerable, not backed up)
neo4j-admin database dump geopolitic_aggregated --to-path=/tmp/dumps/
gzip /tmp/dumps/geopolitic_aggregated.dump

# upload to Cloudflare R2 (using aws-cli with R2 endpoint)
aws s3 cp /tmp/dumps/geopolitic_aggregated.dump.gz \
  s3://geopolitic-backups/$(date +%Y%m%d)/geopolitic_aggregated.dump.gz \
  --endpoint-url https://{ACCOUNT_ID}.r2.cloudflarestorage.com
```

### Alternative: Oracle Cloud Always Free (single-service option)

Oracle Cloud's Always Free tier provides **20 GB object storage** + 2 compute instances + managed DB, indefinitely. Sufficient to host the entire application if you want a single cloud provider. Trade-off: requires Oracle account setup and IAM configuration; more overhead than R2 for a local-first research project.

**Recommendation:** Start with Cloudflare R2 + Hugging Face (5-minute setup each). Migrate to Oracle Cloud only if the project needs hosted deployment.
