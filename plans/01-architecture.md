# Geopolitical Simulation — Project Architecture Plan

## Context

This is a greenfield research application that ingests structured international-relations event data, stores it in two Neo4j databases (raw events + aggregated snapshots), trains a Graph Neural Network to predict future geopolitical relationship types between countries, and renders predictions with explanations in an interactive React frontend. The project is local and research-focused — no authentication or multi-user concerns.

The core ML problem is **link-label prediction on a dynamic graph**: given the historical graph state up to time T, for a directed Country→Country pair, predict the probability distribution over five relationship classes at time T+1. Actors (non-state entities) are present in the graph and influence the model through their connections to countries, but are never prediction targets.

---

## ML Task Formulation

### What is being predicted

**Prediction target:** Every directed pair (Country source → Country target). The model outputs which of five classes best describes their relationship at the next discrete monthly time step.

| Class | Description |
|---|---|
| `MATERIAL_CONFLICT` | Military actions, cyberattacks, physical clashes |
| `VERBAL_CONFLICT` | Threats, sanctions, severing diplomatic relations |
| `MATERIAL_COOPERATION` | Financial aid, arms supplies, joint exercises |
| `VERBAL_COOPERATION` | Treaty signings, official visits, public support |
| `STATUS_QUO` | No significant change / noise |

**Output:** A 5-dimensional calibrated probability vector per (Country, Country) pair.

**Actors in training vs. inference:** Actor nodes and Actor↔Country / Actor↔Actor edges are included in the training graph because they provide contextual signal (e.g., a shared IO membership or a shared armed-group presence can be predictive of country-level behavior). However, the loss is computed only on Country→Country edges, and the inference endpoint only accepts Country IDs as inputs.

---

### Entities (Graph Nodes)

**Country node** — sovereign state (ISO 3166-1 alpha-3 ID). Identity is static; features are time-varying (see §Time-Varying Node Features below).

Static identity fields: `id` (alpha-3), `name`, `region`

Time-varying feature categories (stored per month, see database schema):
- Geographic: land area (log), coastline flag, neighbor count *(quasi-static but re-stored monthly for uniform schema)*
- Economic: GDP (log), GDP per capita, trade openness index, sanctions status (binary)
- Demographic: population (log), growth rate, HDI
- Political: V-Dem electoral democracy score (polyarchy), political stability (World Bank WGI), years since leadership change
- Military: SIPRI military expenditure (log), nuclear flag, alliance memberships (multi-hot: NATO/SCO/CSTO/AU/ASEAN… — derived from the temporal `MEMBER_OF` edges valid at `time_step`)
- Temporal state: active conflict count / conflict intensity (from ACLED), UNSC seat flag (rolling 2-year window)

**Actor node** — non-state entity (international orgs, armed groups, MNCs, political blocs). Present in graph for contextual signal only; never a prediction target.

Static identity fields: `id` (UUID), `name`, `actor_type`

Time-varying features:
- Member count (log), financial resources tier (1–5), recognized legitimacy score

Both node types carry a **64-dim trainable identity embedding** updated during training to capture latent characteristics not expressed by hand-crafted features.

---

### Time-Varying Node Features

A key modeling challenge is that node features are not static: GDP changes year to year, governments change, alliances shift. Treating node features as fixed would introduce temporal leakage (using 2024 GDP to predict a 2012 event).

**Solution:** Node features are versioned by `time_step` in both databases.

In the **raw database**, each entity (Country *or* Actor) has a separate `:FeatureSnapshot` node per year (annual data sources) or per month where data exists, linked by `[:HAS_FEATURES]->(:FeatureSnapshot {node_id, node_type, time_step})`. For months within the same year where no new data exists, the most recent snapshot is carried forward (forward-fill).

In the **aggregated database**, each `:NodeSnapshot {node_id, time_step, ...features}` stores the forward-filled feature vector for that entity at that specific month. The ML data extraction layer reads node features from the NodeSnapshot at the corresponding time_step, not from the static entity node.

**Model input consequence:** The node feature tensor is `[N_nodes × W × F_node]` — each of the W=12 rolling months has a distinct feature vector per node. This captures feature drift (e.g., a country's GDP declining over a year while in conflict).

---

### Relationships (Graph Edges)

**Raw event edge** — one edge per observed real-world event. **Stored in the raw
database only when `GDELT_MODE=raw`.** In the default `GDELT_MODE=aggregated`, GDELT
events are aggregated server-side in BigQuery and **never written to the raw DB** — so
in that mode there are no `:EVENT` edges in Neo4j (see §Data Sources and
`02-data-ingestion.md` §6). The schema, when present:
```
(source)-[:EVENT {
  event_id, relationship_class, timestamp, time_step (int),
  intensity_score, sentiment_score, source_count,
  goldstein_scale, event_type, data_source
}]->(target)
```

**Monthly snapshot edge** — one aggregated edge per (source, target) pair per calendar month (stored in the aggregated database). **This `SNAPSHOT_EDGE` *is* the country↔country relationship the model predicts** — its `dominant_class`/`class_distribution` is the training label and its other fields are the edge features. It is built either by aggregating raw `:EVENT` edges (`GDELT_MODE=raw`) or directly by the BigQuery aggregation (`GDELT_MODE=aggregated`, default); either way the result is identical and lives in the aggregated DB.
```
(source)-[:SNAPSHOT_EDGE {
  time_step,
  event_count, weighted_intensity,
  sentiment_mean, sentiment_std,
  dominant_class,
  class_distribution (Float[5]),
  class_transition_vector (Float[25]),  -- flattened 5×5 transition matrix (aggregated mode: omitted to save disk)
  days_since_last_event
}]->(target)
```
Actors carry **no** event edges by design: ACLED conflict signal becomes a *node*
feature (`active_conflict_count`/`conflict_intensity`) and GDELT events are
country↔country only — so an Actor's only relationships are the structural
`MEMBER_OF` edges that give the model contextual signal.

**Temporal structural edges** (stored in both databases). Borders and memberships are *not* static — within the study window alliances expand (NATO added Montenegro 2017, North Macedonia 2020, Finland 2023, Sweden 2024), states leave (Brexit 2020), and borders change (South Sudan 2011, Crimea 2014). Treating them as fixed would inject the same temporal leakage the time-varying node features avoid. Each structural edge therefore carries a validity interval:
```
(:Country)-[:BORDERS    {start_time_step, end_time_step}]->(:Country)
(:Country|:Actor)-[:MEMBER_OF {start_time_step, end_time_step}]->(:Actor)
```
`end_time_step = null` means "still valid". Point-in-time query — edges valid at time `T`:
`start_time_step <= T AND (end_time_step IS NULL OR end_time_step > T)`. Intervals are populated from Wikidata P580 (start time) / P582 (end time) qualifiers (see `02-data-ingestion.md`).

**Time-step convention (no TimeStep node).** `time_step` is a single integer = months since the 2010-01 epoch: `time_step = (year - 2010) * 12 + (month - 1)` (t=0 → 2010-01). `year`, `month`, and `iso_period` (e.g. `"2013-07"`) are pure functions of `time_step`, computed by a small shared helper in Go and Python — there is no separate calendar node. `time_step` is already an indexed integer property on `NodeSnapshot` and `SNAPSHOT_EDGE`, so range scans (`WHERE r.time_step >= T-11 AND r.time_step <= T`) are served directly by those indexes; a dimension node would only duplicate the value without adding a traversal.

**Where the country↔country relations live.** The relationships the model trains on
and predicts are the `SNAPSHOT_EDGE`s in **`geopolitic_aggregated`** (~2.4M edges over
~36K directed dyads for the full GDELT window). In the default `aggregated` mode the
**raw** DB intentionally holds *no* event edges — only `FeatureSnapshot`s and the
temporal `BORDERS`/`MEMBER_OF` structural edges — because the raw events are kept in
BigQuery (see `02-data-ingestion.md` §6/§Storage). So "raw shows only borders" is
expected, not a gap. To see the event relations, query the aggregated DB:
```cypher
MATCH (a:Country)-[r:SNAPSHOT_EDGE]->(b:Country) RETURN a, r, b LIMIT 100
// (a plain `MATCH (n) RETURN n LIMIT 300` is dominated by NodeSnapshots, which hides them)
```

---

### Time-Series Design: Monthly Snapshots + Rolling Window

**Discrete time steps = calendar months**, epoch 2010-01 (`time_step = 0`), yielding ~197 steps as of 2026-06.

**Why monthly:** Monthly aggregation provides regular structure for batching and temporal alignment. Event-level sequences have irregular spacing and variable length. Monthly granularity aligns with most political-science datasets.

**Rolling input window W = 12 months:** The model sees the last 12 months of SNAPSHOT_EDGEs and NodeSnapshots for every node and edge in the subgraph. This captures seasonal and annual diplomatic cycles while keeping the model sensitive to recent regime shifts.

**Temporal graph evolution:** The graph topology is genuinely time-varying — event edges appear and disappear monthly, and structural edges (`BORDERS`, `MEMBER_OF`) carry validity intervals (see §Relationships). The structural scaffold for a window is the union of structural edges valid at any month in that window (`start_time_step <= window_end AND (end_time_step IS NULL OR end_time_step > window_start)`); edge and node features encode per-time-step values.

---

### Model Input and Output

**For a given target Country pair (u, v) at inference time T+1:**

- **Spatial context:** 2-hop ego subgraph around both u and v from the aggregated graph at time T (includes Actor nodes within 2 hops)
- **Temporal context:** Last W=12 NodeSnapshot feature vectors per node + last W=12 SNAPSHOT_EDGE feature vectors per edge in the subgraph
- **Input tensors:**
  - Node features: `[N_nodes × W × F_node]`
  - Edge features: `[N_edges × W × F_edge]`
  - Adjacency structure (union over the W-window)
  - Target pair indicator mask (which two nodes are u and v)

**Output:** 5-dim softmax probability vector, calibrated via Platt scaling.

**Why live subgraph inference, not a precomputed-embedding lookup.** Inference runs a full forward pass over the freshly extracted temporal subgraph rather than looking up two stored node embeddings, and this is a requirement, not a preference:

- **Explainability needs the live computational graph.** GNNExplainer learns soft masks over the *actual* subgraph nodes/edges, and Integrated Gradients integrates over the *actual* input feature tensors. A frozen per-node embedding exposes neither structure nor features to attribute over — both methods would be impossible.
- **Simulation needs re-computation.** `POST /simulate` overrides edges and asks "what changes?". Against frozen embeddings an edge override is a no-op; only a forward pass over the modified subgraph produces a different prediction.
- **Temporal correctness.** The prediction depends on the last W=12 months *ending at T*. A single static embedding cannot represent the window that ends at an arbitrary query time T.

**Inductive over time, transductive over nodes.** The model generalizes to time steps it never saw (including T+1) because it consumes per-time-step features and structure at query time. It is transductive over the fixed ~195-country node set: each node has a learned 64-dim identity embedding (a model parameter) looked up by node id during the forward pass. New, never-trained nodes would have no identity embedding — acceptable here because the country set is fixed and known.

**Latency target:** < 500ms p99 for a single pair. Batch mode (simulation) uses `POST /predict/batch` to amortize Neo4j subgraph query overhead.

---

### Data Sources

| Source | Content | Role / coverage |
|---|---|---|
| GDELT 1.0 + 2.0 (via BigQuery) | CAMEO-coded events, Goldstein scale | **Selected** primary event source — `full.events` (2010→2015) + `gdeltv2.events` (2015→present); default `aggregated` mode aggregates server-side into `SNAPSHOT_EDGE`s (~2.4M, ~1.5–2 GB) rather than loading raw `[:EVENT]` edges (16–48 GB) |
| ICEWS (Harvard Dataverse) | CAMEO-coded events, strong source attribution | **Optional** cross-check (GDELT `full.events` already covers 2010–2014); off by default |
| ACLED | Armed conflict events (mostly intrastate) | **Node conflict-intensity feature** (per country-month), not Country→Country edges |
| Wikidata | Borders, memberships (with validity dates), region, actors | Annual refresh |
| World Bank API | Economic + demographic + political-stability indicators | Annual |
| SIPRI | Military expenditure | Annual |
| V-Dem | Electoral democracy + leadership change | Annual |
| UNDP HDR | Human Development Index | Annual |

See `02-data-ingestion.md` for the full source-by-source breakdown.

---

### Label Generation

`[:EVENT]` edges come from **GDELT + ICEWS** (both CAMEO-coded). Raw CAMEO codes map to the five classes:

- **MATERIAL\_CONFLICT:** CAMEO 18x/19x/20x + Goldstein < −5
- **VERBAL\_CONFLICT:** CAMEO 11x–15x + sanctions announcements + diplomatic severance
- **MATERIAL\_COOPERATION:** CAMEO 06x–08x + joint exercise / arms-transfer announcements
- **VERBAL\_COOPERATION:** CAMEO 01x–05x + treaty signings + Goldstein > +5
- **STATUS\_QUO:** No events, or all events with |Goldstein| < 1 and no dominant class

ACLED is *not* a source of Country→Country edges (most ACLED events are intrastate); it feeds the `active_conflict_count` / `conflict_intensity` node feature. Genuinely interstate ACLED events (both actors resolve to states) may optionally be added as MATERIAL\_CONFLICT edges.

**Window-level labeling rule:** If any MATERIAL\_CONFLICT event exists in the window, the window label is MATERIAL\_CONFLICT regardless of other events. Otherwise: modal class wins. This conflict-priority rule reflects that one armed incident is more diagnostic than many diplomatic calls.

---

### Class Imbalance

STATUS\_QUO will represent ~70–85% of labeled Country→Country windows. Mitigation is layered:

1. **Negative sampling:** Per positive edge, sample K=5 STATUS\_QUO Country→Country edges from the same snapshot (~1:5 ratio instead of 1:70)
2. **Class-weighted cross-entropy loss:** Weights ∝ inverse class frequency; STATUS\_QUO weight = 1.0, MATERIAL\_CONFLICT starts at ~5×
3. **Focal loss option:** Alternative to weighted CE; down-weights easy STATUS\_QUO examples
4. **Oversampling rare dyads:** Historically conflictual country pairs appear more frequently in training batches
5. **Threshold calibration:** Platt scaling / isotonic regression on a held-out validation set

---

### Evaluation Metrics

| Metric | Role |
|---|---|
| **Macro-F1** | Primary metric; penalizes equally for poor rare-class performance |
| Per-class F1 / Precision / Recall | MATERIAL\_CONFLICT recall is the headline secondary metric |
| 5×5 Confusion matrix | Primary debugging tool; directional confusions are meaningful |
| AUC-ROC (one-vs-rest) | Per-class calibration assessment |
| Expected Calibration Error (ECE) | Validates probability outputs are reliable (frontend shows confidence) |

**Evaluation split is always chronological.** Test set = most recent N months. Random splits leak future events into training.

---

### Explainability

Two complementary methods are planned, both surfaced in the frontend alongside predictions:

**GNNExplainer** — post-hoc, structure-aware. Learns a soft mask over the input subgraph (node importance weights + edge importance weights) that best explains the model's predicted class for a given (u, v) pair. Output: a ranked list of which neighboring countries/actors and which historical edges contributed most to the prediction. Runs at inference time on the 2-hop ego subgraph.

**Integrated Gradients** — feature-level attribution. Computes the gradient of the output class probability with respect to each input feature, integrated along a path from a baseline (zero features) to the actual input. Output: a signed importance score per feature per node in the subgraph, showing which node features (e.g., "military expenditure of country A" or "conflict count at T-3") drove the prediction. Runs at inference time; more expensive than GNNExplainer but provides feature-level rather than structure-level insight.

Both methods return their outputs through `POST /predict` (alongside the probability vector) and are displayed in the frontend's Explanation Panel as: (1) a highlighted subgraph with edge/node opacity scaled to importance, and (2) a ranked feature importance bar chart.

---

## Overall System Architecture

### Microservices

```
geopolitic/
├── plans/
│   ├── 01-architecture.md    # this file
│   └── 02-data-ingestion.md  # data source details + storage estimates
├── services/
│   ├── api/          # Go: REST API, ingestion pipeline, graph browse, prediction proxy
│   ├── ml/           # Python: training job (offline) + FastAPI inference + explainability
│   └── frontend/     # TypeScript + React: graph viz, timeline, prediction, simulation UI
├── infra/
│   ├── neo4j/        # two database configs, schema migrations, seed scripts
│   └── docker/       # docker-compose for local dev
└── shared/
    └── schemas/      # OpenAPI spec, shared JSON Schema types
```

**Go handles all database population.** There is no separate Python ingestion service. The Go API server owns: fetching from external sources, normalizing events, generating labels, building monthly snapshot aggregates, and writing to both Neo4j databases. Python is exclusively responsible for ML (training and inference).

---

### Two Neo4j Databases

**`geopolitic_raw`** — event-level store. Source of truth for all raw ingested data.

Contents:
- `(:Country)`, `(:Actor)` — identity nodes (static fields only)
- `(:FeatureSnapshot {node_id, node_type, time_step, ...features})` — one per entity (Country *or* Actor) per available time step, linked `(:Country|:Actor)-[:HAS_FEATURES]->(:FeatureSnapshot)`
- `(:EVENT)` edges — one per real-world event, all raw attributes — **`GDELT_MODE=raw` only**; in the default `aggregated` mode there are no `:EVENT` edges here (events are aggregated in BigQuery, not loaded into raw)
- `[:BORDERS {start_time_step, end_time_step}]`, `[:MEMBER_OF {start_time_step, end_time_step}]` — temporal structural edges

So in the default `aggregated` mode the raw DB contains only identity nodes,
`FeatureSnapshot`s, and the two temporal structural edge types — the country↔country
event relations live in the aggregated DB below.

**`geopolitic_aggregated`** — snapshot store. Derived from raw; what the ML service reads from.

Contents:
- `(:Country)`, `(:Actor)` — same identity nodes (mirrored from raw)
- `(:NodeSnapshot {node_id, node_type, time_step, ...features})` — one per entity per month, forward-filled
- `[:SNAPSHOT_EDGE]` — one per (source, target) pair per month, aggregated from raw EVENT edges
- `[:BORDERS {start_time_step, end_time_step}]`, `[:MEMBER_OF {start_time_step, end_time_step}]` — mirrored temporal structural edges

`time_step` is a plain indexed integer on `NodeSnapshot` and `SNAPSHOT_EDGE` (no `TimeStep` node); calendar fields derive from it via the shared helper (§Relationships).

Go builds the aggregated database by reading from raw, computing SNAPSHOT_EDGE aggregates for each completed month, and forward-filling NodeSnapshots. Triggered on demand via `POST /api/v1/ingest`.

---

### Data Flow

```
GDELT (BigQuery) / ICEWS / ACLED(→feature) / Wikidata / World Bank / SIPRI / V-Dem / UNDP
        │
        ▼  triggered by POST /api/v1/ingest
[Go API Server]
  → World Bank / Wikidata / seeds / ACLED / V-Dem / SIPRI / UNDP
       → write FeatureSnapshots + BORDERS/MEMBER_OF to geopolitic_raw   (NO event edges)
  → Build: forward-fill NodeSnapshots into geopolitic_aggregated
  → GDELT (default GDELT_MODE=aggregated):
       aggregate events server-side in BigQuery (CAMEO→relationship_class)
       → write SNAPSHOT_EDGEs straight into geopolitic_aggregated
       (raw events stay in BigQuery — never loaded into Neo4j)
     [alternate GDELT_MODE=raw: write [:EVENT] edges into geopolitic_raw,
      then Build aggregates them into SNAPSHOT_EDGEs]
        │
        ├──► geopolitic_raw       (FeatureSnapshots, BORDERS/MEMBER_OF; EVENT only in raw mode)
        └──► geopolitic_aggregated (NodeSnapshots, SNAPSHOT_EDGEs ← the ML reads this)
                    │
                    │ read
                    ▼
           [ML Training Job — Python]
             export to Parquet
             build temporal graph batches
             train GNN + calibrator + explainability hooks
                    │
        ┌───────────▼────────────┐
        │  local artifact store  │
        │  model_v{ts}.pt        │
        │  scaler.pkl            │
        │  calibrator.pkl        │
        └───────────┬────────────┘
                    │ loads at startup
[ML Inference Server — Python FastAPI]
  /predict        (probabilities + GNNExplainer output + IG attributions)
  /predict/batch  (simulation)
        ▲
        │ internal HTTP (Docker network only)
        │
[Go API Server]
  GET /api/v1/entities, /relationships, /snapshot  → reads geopolitic_aggregated
  POST /api/v1/predict   → proxies to Python inference server
  POST /api/v1/simulate  → fan-out batch to Python, assemble prediction graph
  POST /api/v1/ingest    → triggers ingestion + aggregation pipeline
        ▲
        │ HTTP (localhost)
        │
[React Frontend]
  graph canvas — countries + actors, edges colored by class
  timeline scrubber — T_0 → T_current
  prediction panel — probability chart + explanation panel
  simulation mode — override edges → predicted cascade overlay
```

---

### Neo4j Schema

**`geopolitic_raw` constraints and indexes:**
```cypher
CREATE CONSTRAINT FOR (c:Country) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT FOR (a:Actor) REQUIRE a.id IS UNIQUE;
CREATE INDEX FOR (s:FeatureSnapshot) ON (s.node_id, s.time_step);
CREATE INDEX FOR ()-[r:EVENT]-() ON (r.time_step);                          -- GDELT_MODE=raw only
CREATE INDEX FOR ()-[r:EVENT]-() ON (r.time_step, r.relationship_class);    -- GDELT_MODE=raw only
CREATE INDEX FOR ()-[r:MEMBER_OF]-() ON (r.start_time_step);
CREATE INDEX FOR ()-[r:BORDERS]-() ON (r.start_time_step);
```

**`geopolitic_aggregated` constraints and indexes:**
```cypher
CREATE CONSTRAINT FOR (c:Country) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT FOR (a:Actor) REQUIRE a.id IS UNIQUE;
CREATE INDEX FOR (n:NodeSnapshot) ON (n.node_id, n.time_step);
CREATE INDEX FOR ()-[r:SNAPSHOT_EDGE]-() ON (r.time_step);
CREATE INDEX FOR ()-[r:SNAPSHOT_EDGE]-() ON (r.time_step, r.dominant_class);
```

---

### Go API — Key Endpoints

```
-- Ingestion (manual trigger, no schedule)
POST   /api/v1/ingest                            trigger full ingestion + aggregation run
GET    /api/v1/ingest/status                     last run status, records written, errors

-- Graph browse (reads geopolitic_aggregated)
GET    /api/v1/entities                          list Country and Actor nodes
GET    /api/v1/entities/:id                      node detail + most recent features
GET    /api/v1/relationships?source=&target=&from=&to=   SNAPSHOT_EDGE history for a dyad
GET    /api/v1/snapshot?time_step=               full graph snapshot at time T

-- Prediction and simulation (proxy to Python ML service)
POST   /api/v1/predict
       body: { source_id: Country, target_id: Country, time_step }
       → { probabilities: {CLASS: float}, predicted_class, confidence,
           explanation: { edge_importances, node_importances, feature_attributions } }

POST   /api/v1/simulate
       body: { overrides: [{source_id, target_id, class}], time_step }
       → prediction graph for t+1 (all affected Country→Country edges with probabilities)

-- ML lifecycle
POST   /api/v1/ml/train                          trigger training job in Python ML service
GET    /api/v1/ml/status                         training job status + last metrics.json
POST   /api/v1/ml/reload                         hot-reload model in inference server
```

No authentication. All endpoints are local-only.

---

### Python ML Service — Internal API

```
POST /predict
     { source_id, target_id, time_step }
     → { probabilities: float[5], class_names: str[5],
         gnn_explainer: { node_mask: {id: float}, edge_mask: {(src,tgt): float} },
         integrated_gradients: { feature_attributions: {node_id: {feature: float}} },
         latency_ms }

POST /predict/batch
     { pairs: [{source_id, target_id}], time_step }
     → { predictions: [{source_id, target_id, probabilities}] }
     (no explanations in batch mode — used for simulation fan-out only)

GET  /health    { status, model_version, model_loaded_at }
POST /train     trigger training job (runs as subprocess, returns job_id)
GET  /train/:job_id   job status
POST /reload    hot-reload latest model artifact
```

---

### Training Job Pipeline

1. Read SNAPSHOT\_EDGE + NodeSnapshot records from `geopolitic_aggregated` → export to Parquet (decouples training from live Neo4j availability)
2. Build PyG `TemporalData` objects — node feature tensor `[N × W × F_node]`, edge feature tensor `[E × W × F_edge]`, edge index
3. **Chronological train/val/test split** — test set = most recent N months, never shuffle across time
4. Negative sampling (K=5 STATUS\_QUO Country→Country pairs per positive edge)
5. Training loop: class-weighted cross-entropy (or focal loss), Macro-F1 early stopping (patience=5)
6. Serialize model weights + scaler + calibrator + explainability configs to versioned artifact
7. **Embedding write-back (analytics/visualization only):** run a forward pass over all nodes and write the final-layer node embeddings to both Neo4j databases as an `embedding` property — used for similarity search and 2-D projection in the frontend. These are **not** used by the prediction path; inference always recomputes from the live subgraph (see §Model Input and Output)
8. Write `metrics.json` (test set Macro-F1, per-class F1, ECE, confusion matrix)

---

### Frontend Architecture

**Graph Canvas** — D3.js or Sigma.js. Country nodes are primary interactive elements; Actor nodes shown as smaller secondary nodes (dimmed) for structural context. Edges colored by `dominant_class`. Hover tooltip: class, intensity, event count. **Only Country→Country pairs can be selected for prediction.**

**Timeline Scrubber** — horizontal bar, T\_0 → T\_current. Scrubbing calls `GET /snapshot?time_step=T` and re-renders the graph.

**Prediction Panel** — Country pair picker → `POST /predict` → 5-bar horizontal probability chart + confidence indicator.

**Explanation Panel** — displayed alongside Prediction Panel:
- Subgraph highlight view: nodes and edges in the 2-hop context colored/weighted by GNNExplainer importance scores
- Feature attribution bar chart: top-N features ranked by integrated gradients magnitude (positive = pushed toward predicted class, negative = pushed away)

**Simulation Mode:**
1. User toggles simulation mode
2. Selects one or more Country→Country edges and overrides their class
3. `POST /simulate` → Go fans out batch prediction for the 1-hop neighborhood of overridden pairs
4. Frontend overlays predicted edges (saturated) on historical (muted) with legend

**State management:** React Query for server state. Zustand for local UI state (selected nodes, time step, simulation overrides, active filters).

---

### Recommended Build Order

| Step | Deliverable |
|---|---|
| 0 | Create `plans/` directory, place both plan documents |
| 1 | Both Neo4j database schemas: constraints, indexes; seed all ~195 countries from the live World Bank country list (no synthetic data — events come from real GDELT) |
| 2 | Go ingestion pipeline: GDELT fetcher + label generator + snapshot builder → writes to both DBs |
| 3 | Go graph browse endpoints: `/entities`, `/snapshot`, `/relationships` |
| 4 | Frontend graph canvas: render static snapshot from Go API, timeline scrubber |
| 5 | ML data layer: Parquet export from aggregated DB, PyG dataset construction, validate tensor shapes |
| 6 | ML training pipeline: trainer + sampler + metrics, run on seed data |
| 7 | ML inference server: FastAPI, subgraph loader, basic prediction endpoint |
| 8 | Go predict + simulate endpoints: proxy to Python ML service |
| 9 | Frontend prediction panel + simulation mode |
| 10 | Explainability: GNNExplainer + Integrated Gradients in ML service, Explanation Panel in frontend |
| 11 | Calibration + final evaluation: Platt scaling, ECE, metrics.json, test set report |

---

## Verification Plan

- **Neo4j schemas:** Apply constraints and indexes to both databases; verify with `CALL db.indexes()` and `CALL db.constraints()` on each
- **Ingestion:** Call `POST /api/v1/ingest`, then confirm the country↔country relations in the **aggregated** DB (`bolt://localhost:7688`, browser http://localhost:7475 — *not* the raw DB at 7474/7687, which has no `SNAPSHOT_EDGE` in the default `aggregated` mode): `MATCH (a:Country)-[r:SNAPSHOT_EDGE]->(b:Country) RETURN count(r), r.dominant_class` and `MATCH (n:NodeSnapshot) RETURN count(n)`; check label distribution. (In `GDELT_MODE=raw` only, also confirm `[:EVENT]` edges in the raw DB.)
- **Node feature versioning:** Query two different time\_steps for the same country; confirm feature vectors differ (e.g., GDP changes year over year)
- **Temporal structural edges:** Resolve NATO members at a `time_step` before vs. after an enlargement (e.g., 2016 vs. 2024, spanning the Finland/Sweden accessions) using the validity-interval predicate; confirm the member set differs
- **ML data extraction:** Assert PyG tensor shapes are `[N × W × F]`; confirm no NaN values in feature matrices
- **Training pipeline:** Train 3 epochs on seed data; confirm loss decreases and Macro-F1 > 0 on validation
- **Inference server:** `POST /predict` for a known Country pair; probabilities sum to 1.0; GNNExplainer mask values in [0, 1]; IG attributions satisfy completeness axiom
- **Go proxy:** Call `POST /api/v1/predict` through Go; response includes probabilities and explanation fields
- **Frontend end-to-end:** Render snapshot, scrub timeline, select two Country nodes, view prediction + explanation, toggle simulation and view overlay
