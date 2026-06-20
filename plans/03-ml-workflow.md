# ML Workflow — Data Formatting, Model, Explainability & Training

## Context

This is the third plan document. `01-architecture.md` defines the ML task and system
shape; `02-data-ingestion.md` describes how the two Neo4j databases are populated. Both are
done: `geopolitic_aggregated` (browser http://localhost:7475, Bolt `bolt://localhost:7688`)
holds **232 identity nodes** (221 Country + 11 Actor IGOs), **~45,936 NodeSnapshots**
(one per node per month, forward-filled, `MaxTimeStep = 197` = 2026-06), **~2.42M directed
monthly `SNAPSHOT_EDGE`s** (Country→Country), and the temporal `BORDERS` / `MEMBER_OF`
structural edges. This is the *only* database the ML service reads.

This document covers the ML stage:

1. **§1** — how to format the aggregated DB into model-ready tensors.
2. **§2** — the model: a temporal GNN that predicts the **country↔country relationship
   class one month ahead**.
3. **§3** — the explainability algorithm (why the model made a given decision).
4. **§4** — training-time estimate, the GPU (Kaggle), and Weights & Biases tracking.
5. **§5** — the end-to-end flow diagram.
6. **§6** — verification / acceptance checks.
7. **§7** — APIs, services & credentials required to execute this plan.

**The task, restated.** Link-label prediction on a dynamic, temporal, heterogeneous graph:
given the graph history up to month `T`, for a directed pair (Country `u` → Country `v`),
predict a calibrated probability distribution over five classes at `T+1`:
`MATERIAL_CONFLICT, VERBAL_CONFLICT, MATERIAL_COOPERATION, VERBAL_COOPERATION, STATUS_QUO`
(canonical order from `internal/label/cameo.go` — **do not reorder**, indices are persisted
in `class_distribution`). Loss is computed only on Country→Country edges; the 11 Actor nodes
are context only.

**Implementation principle — production research code, no stubs.** Every file specified
here (the Go exporter, the `ml/` modules, and the Kaggle notebook) is written **complete and
research-ready**: no stubs, no `TODO`s, no mocked or placeholder logic, and **no synthetic
or sub-sampled data**. The pipeline runs on the **full** exported dataset (all ~2.42M
`SNAPSHOT_EDGE`s, all 186 usable target months) end-to-end — export → train → calibrate →
explain → serve.

---

## §1 Data formatting: aggregated DB → model tensors

### 1.1 Three-stage pipeline (Neo4j → Parquet → PyG)

Training never touches live Neo4j. A snapshot is exported to **Parquet** once, then a
Python loader builds PyTorch Geometric (PyG) samples from it.

```
[Go exporter: cmd/export-parquet]   reads geopolitic_aggregated (bolt:7688)
        │  range scans served by existing indexes (snapshot_edge_ts, node_snapshot_ts)
        ▼
  Parquet on disk (partitioned by time_step)
        │
        ▼
[Python: services/ml/dataset.py]   per-target-month samples, scaler fit on TRAIN only
        ▼
  PyG DataLoader → train.py
```

**Why Parquet:** columnar, typed, compressed ~5–10× (the full ~2.42M-row edge set is a few
hundred MB), with column/partition pushdown so the loader reads only the months it needs.
This matters on the dev Mac (~2 GB free) and makes the whole `dataset_parquet/` folder small
enough to upload to Kaggle by hand as one Dataset (§4).

**Why the exporter is Go, not Python.** Go already has the Neo4j driver, the canonical class
ordering in `internal/label.Classes`, and the time-step helper in `internal/timestep`.
Driving the export from Go reuses both and removes the single most dangerous bug class — a
Go/Python disagreement about which index is `MATERIAL_CONFLICT`. Python mirrors the order
once, as a constant, and treats Go as the source of truth:

```python
# services/ml/ml/config.py — MUST match internal/label/cameo.go exactly
CLASS_NAMES = ["MATERIAL_CONFLICT", "VERBAL_CONFLICT",
               "MATERIAL_COOPERATION", "VERBAL_COOPERATION", "STATUS_QUO"]
```

The time-step convention is also mirrored from `internal/timestep/timestep.go`:
`time_step = (year-2010)*12 + (month-1)`, `t=0 → 2010-01`.

### 1.2 Parquet schema

Three files. Field names are taken from the Go writers (`ingestion/aggregated/aggregate.go`,
`ingestion/raw/featuresnapshot.go`) — **no invented properties**.

```
node_snapshots/ts=NN/part.parquet            # partition by time_step
  node_id: str          node_type: str ("Country"|"Actor")    ts: int32
  # Country continuous (already log-transformed in the DB where the name says _log — do NOT
  #   re-log; conflict_intensity is already log(Σfatalities+1); active_conflict_count is a
  #   raw count → log1p at load):
  gdp_log, gdp_per_capita, trade_openness_index, population_log, population_growth,
  land_area_log, political_stability, vdem_polyarchy_score, hdi, military_expenditure_log,
  years_since_leadership_change, active_conflict_count, conflict_intensity : float32
  # Country binary flags (pass-through, no scaling):
  sanctions_status, unsc_seat_flag, nuclear_flag, coastline_flag : int8
  region : str                        # static categorical → one-hot (Country)
  # Actor-only features (null/0 on Country rows; the Country columns are null on Actor rows):
  member_count_log, recognized_legitimacy_score : float32
  financial_resources_tier : int8     # 1..5 ordinal (Actor)
  # NOTE: neighbor_count is NOT stored (no loader writes it) — derive it at load time by
  #   counting BORDERS valid at ts from structural_edges.parquet if wanted.

snapshot_edges/ts=NN/part.parquet            # partition by time_step
  src: str  tgt: str  ts: int32
  event_count: int32
  weighted_intensity, sentiment_mean, sentiment_std : float32
  days_since_last_event: int32
  class_distribution: list<float32>[5]        # indices = CLASS_NAMES order
  dominant_class: str                         # the label source (label@T comes from ts=T+1)
  data_source: str                            # "GDELT" — provenance/filter, not a model feature
  # NOTE: class_transition_vector is intentionally NOT exported — it is never computed in
  #   aggregated mode, so the GDELT SNAPSHOT_EDGEs carry class_distribution only.

structural_edges.parquet                      # small, unpartitioned
  rel: str ("BORDERS"|"MEMBER_OF")  a: str  b: str
  a_label: str  b_label: str  start: int32  end: int32 (nullable = still valid)
```

Partitioning the two big tables by `ts` lets a sample for target month `T` read only the 13
relevant partitions (`[T-11, T]` for inputs, `T+1` for the label) instead of scanning
millions of rows.

### 1.3 Building a sample (per target month T)

The contract from `01-architecture.md` is: a rolling window `W = 12` months, node tensor
`[N × W × F_node]`, edge information per month, label from `T+1`.

- **Node set.** Fixed transductive set `N = 232`. Build a stable `node_id → row index` map
  once (sorted, persisted to `node_index.json`) so identity-embedding lookups and
  `edge_index` are consistent across every sample and at inference. This map is an artifact.
- **Node tensor `x` `[N × W × F_node]`.** For each node, gather its NodeSnapshots for
  `ts ∈ [T-11, T]`. `Build` already forward-fills every month `[0, maxTS]`, so there are no
  gaps — assert it. `F_node(Country)` = 17 numeric/binary (13 continuous + 4 binary) +
  region one-hot (~7) + 11-dim alliance multi-hot (see 1.4) ≈ 35; `F_node(Actor)` = 3
  (`member_count_log`, `recognized_legitimacy_score`, `financial_resources_tier`). The two
  node types use **separate input projections** (§2), so their differing widths are fine.
  The 64-dim identity embedding is **not** in `x`; it is an `nn.Embedding` looked up by node
  index inside the model (§2).
- **Edges — per-step graphs (recommended).** Rather than materialize a dense
  `[E × W × F_edge]` tensor over 2.42M edges, keep one graph per month: `edge_index_t` =
  the dyads active that month + the structural edges valid that month, with the per-month
  edge-feature matrix `edge_attr_t`. The spatial encoder runs once per month; the temporal
  module consumes the resulting sequence (§2). This is the standard spatio-temporal
  formulation and is far lighter in memory.
- **Structural scaffold.** Structural edges valid anywhere in the window:
  `start <= T AND (end IS NULL OR end > T-11)`. This is what makes "NATO at 2016" differ
  from "NATO at 2024" (Finland/Sweden accessions) — no temporal leakage.
- **Label.** For each *positive* directed Country→Country pair `(u,v)` that has a
  `SNAPSHOT_EDGE` at `T+1`, the label = index of its `dominant_class` at `T+1` (the
  conflict-priority modal rule is already baked into the DB by `aggregateGroup`).
- **Usable target months.** A sample at `T` needs `T+1 ≤ 197`, and a full window needs
  `T ≥ 11`. So `T ∈ [11, 196]` (186 target months).

### 1.4 Feature engineering

- **Standardization — fit on TRAIN months only.** Fit one `StandardScaler` (or
  `RobustScaler` for heavy-tailed counts) on the NodeSnapshot rows whose `ts` falls in the
  **training** range, and the same for continuous edge features. Apply that fitted scaler to
  val/test. **This is the single biggest leakage risk** — never fit on the whole table.
- **Log transforms.** Fields named `*_log` are already `log(x+1)` from the Go loaders, and
  `conflict_intensity` is already `log(Σfatalities+1)` — do **not** log these again, just
  standardize. Raw counts (`event_count`, `active_conflict_count`, `days_since_last_event`)
  get `log1p` then standardize.
- **Binary flags** (`sanctions_status`, `unsc_seat_flag`, `nuclear_flag`, `coastline_flag`)
  pass through as 0/1.
- **`region`** → static one-hot; fit the category list on train and persist it.
- **Alliance multi-hot (11-dim).** This is *not* a stored node feature; derive it per
  `(node, ts)` from the temporal `MEMBER_OF` edges valid at that month
  (`start <= ts AND (end IS NULL OR end > ts)`) against a fixed Actor-id → column map.
  Append to each month's node vector. This is what lets membership change over time.
- **Imputation.** Forward-fill is already done in the DB. For a node with no observation
  before some early `ts` (pre-coverage), impute to the train-set mean and optionally add a
  per-feature `missing_mask` channel. The Actor-only features (`member_count_log`,
  `recognized_legitimacy_score`, `financial_resources_tier`) are 0 on Country rows and the
  Country features are 0 on Actor rows — each node type has its own input projection, so the
  unused block is never mixed in.

### 1.5 Negative sampling (K = 5)

STATUS_QUO is ~70–85% of dyad-months, so positives are sampled against negatives at
build time, target ratio ~1:5:

- A "STATUS_QUO pair at `T+1`" is a directed `(u,v)` that either has a `SNAPSHOT_EDGE` at
  `T+1` with `dominant_class = STATUS_QUO`, or has **no** edge at `T+1` (a quiet dyad).
  Sample from both, biased toward "no-edge" dyads since those dominate reality and the model
  must handle them at inference.
- Use `torch_geometric.utils.negative_sampling` over the 221 Country nodes for the no-edge
  negatives, unioned with explicit STATUS_QUO edges.
- **Train negatives are resampled per epoch; val/test negatives are frozen with a fixed
  seed** (random negatives in eval make metrics non-comparable across runs).
- Materialize the sampled index as `samples_{split}.parquet`
  (`u, v, T, label, is_negative`); the Dataset reads it and assembles window tensors lazily.

### 1.6 Chronological split (no leakage)

Split by **target month `T`**, never shuffle across time. With 186 usable months:

| Split | Target months `T` | Calendar (approx.) |
|---|---|---|
| **train** | `T ≤ 172` | … → 2024-05 |
| **val** | `173 ≤ T ≤ 184` (12) | 2024-06 → 2025-05 |
| **test** | `185 ≤ T ≤ 196` (12) | 2025-06 → 2026-06 |

Rules: the scaler is fit only on train months; a test sample at `T` reads only
`ts ≤ T` for inputs (its single allowed peek into the future is the `T+1` label); the
embedding write-back (analytics) runs after splits are defined and is never read back into
training; optionally insert a 1-month purge gap between train-end and val-start so a train
label month never equals a val window month.

### 1.7 `services/ml/` layout and the one preprocessing bundle

```
services/ml/
├── pyproject.toml          # torch, torch-geometric, pyarrow/polars, scikit-learn,
│                           #   torchmetrics, wandb, joblib, captum, fastapi, uvicorn
├── export/neo4j_to_parquet.go   # (Go exporter; or a Python equivalent — pick one)
├── dataset_parquet/        # the EXPORT OUTPUT — the exact folder uploaded to Kaggle
│   ├── node_snapshots/ts=NN/part.parquet
│   ├── snapshot_edges/ts=NN/part.parquet
│   ├── structural_edges.parquet
│   └── samples_{train,val,test}.parquet
├── notebooks/
│   └── train_geopolitic_gnn.ipynb  # SELF-CONTAINED Kaggle notebook (all pipeline code
│                                    #   inline) — the second artifact uploaded to Kaggle
├── ml/                     # the same code, as importable modules for the inference server
│   ├── config.py           # W=12, K=5, split bounds, F dims, CLASS_NAMES (mirror Go)
│   ├── timestep.py         # mirror of internal/timestep (FromYM/Year/Month)
│   ├── dataset.py          # window build, neg-sampling, scaler fit/load → Data samples
│   ├── features.py         # scaler, region one-hot, alliance multi-hot from MEMBER_OF
│   ├── model.py            # the spatio-temporal GNN (§2)
│   ├── losses.py           # class-weighted CE + focal
│   ├── metrics.py          # macro-F1, per-class F1, confusion matrix, ECE
│   ├── calibrate.py        # Platt / temperature scaling → calibrator.pkl
│   ├── train.py            # loop, checkpointing, early stopping, W&B, artifact save
│   ├── infer.py            # subgraph forward pass for serving
│   └── explain.py          # GNNExplainer + Integrated Gradients (§3)
├── artifacts/              # best.pt, last.pt, preprocess.pkl, calibrator.pkl,
│                           #   node_index.json, metrics.json (also mirrored to W&B)
└── app.py                  # FastAPI inference server (01-architecture §Python ML Service)
```

The training entrypoint is the **self-contained notebook** `train_geopolitic_gnn.ipynb`: it
embeds the full pipeline (the `ml/` code inline) so running it on Kaggle needs nothing but
the attached `dataset_parquet/` Dataset — the user copies exactly two things to Kaggle (§4.3).
The `ml/` package holds the identical code as importable modules so the FastAPI inference
server (`app.py`) reuses the *same* `model.py` / `infer.py` (one model definition, no drift).

Bundle everything inference needs to stay consistent with training into **one**
`preprocess.pkl = {node_scaler, edge_scaler, region_onehot, node_index, class_names}`.
Inference loads this exact bundle and never refits — this prevents the classic
"served model with a mismatched scaler" bug.

---

## §2 Model

### 2.1 Decision: a Heterogeneous, edge-aware **Spatio-Temporal GNN**

The task is *temporal, heterogeneous, edge-feature-rich, and directed*, so the architecture
is a deliberate composition of four parts — a heterogeneous edge-aware spatial encoder, an
external temporal recurrence, a 5-class edge decoder, and a trainable identity embedding:

| Stage | Choice | Why this and not the alternatives |
|---|---|---|
| **Spatial encoder** | `HeteroConv` wrapping **TransformerConv** (for `SNAPSHOT_EDGE`) and **GATv2Conv** (structural edges) per relation | The only combination that simultaneously respects directedness, multi-dimensional edge features (both layers expose `edge_dim`/accept `edge_attr`), and node/edge heterogeneity (Country vs Actor; 3 relation types). Attention also yields per-edge importance "for free" (helps imbalance + explainability). |
| **Cheap structural conv** | **`SAGEConv`** for `BORDERS` / `MEMBER_OF` | Those edges carry only a validity interval, no continuous features — a lightweight aggregator suffices. (Neighbor-sampling is unnecessary on a 232-node graph; a plain SAGE encoder is also the natural simplicity baseline.) |
| **Temporal module** | shared encoder applied to each of `W=12` monthly snapshots → **`torch.nn.GRU`** over the per-node embedding sequence → **temporal-attention pooling** | Decouples time from convolution, so the rich spatial encoder is unconstrained. The attention-pool surfaces "which months mattered" for interpretability. |
| **Edge decoder (5-class)** | concat `[h_u ‖ h_v ‖ h_u⊙h_v ‖ e_uv^last]` → 2-layer MLP → 5 logits → softmax | Direction-aware, ingests the current-month edge features (strong autocorrelation of relationship state). A bilinear/DistMult or subgraph-pooling head is built for binary link *existence* and is awkward for a labeled-edge multiclass task. |
| **Identity** | 64-dim `nn.Embedding`, concatenated at **input** (before message passing) | Memorizes per-country latent traits not in the hand-crafted features; node set is fixed and known, so a transductive embedding is acceptable. |
| **Explainability** | **GNNExplainer** + **Integrated Gradients** | See §3. |

**Why not simpler / off-the-shelf options.** Static node-embedding methods (DeepWalk /
Node2Vec) and a plain MLP throw away time, edge features, or structure entirely and cannot
do live-subgraph inference or simulation. A vanilla `GCNConv` is undirected, takes only a
scalar `edge_weight`, and oversmooths fastest on a ~232-node graph. The pre-packaged
spatio-temporal cells in **PyTorch-Geometric-Temporal** (A3TGCN / GConvGRU / DCRNN / TGCN)
are the most tempting shortcut, but their `forward` accepts only a **scalar `edge_weight`**
and assumes **one homogeneous adjacency** with no node/edge types — adopting them would force
collapsing the edge-feature vector to one scalar, flattening Country/Actor into one type, and
merging the three relations into one, discarding exactly the signals this project is built
on. Hence the temporal mechanism lives **outside** the convolution, in a plain `nn.GRU`.

**Alternative architectures worth trying** (this design is the recommended starting point,
not the only valid one — benchmark against these if time allows):
- **GraphSAGE + GRU** — the same skeleton with a non-attentional encoder; the natural ablation
  baseline and a useful sanity floor.
- **Temporal Graph Networks (TGN)** — memory-based, event-time (continuous-time) model;
  attractive if we ever move off fixed monthly snapshots to event timestamps.
- **EvolveGCN** — evolves the GNN weights over time with an RNN instead of evolving node
  states; light and a good contrast to the GRU-over-embeddings approach here.
- **DySAT** — joint structural + temporal self-attention; a stronger (heavier) attention-only
  alternative to the GRU readout.
- **Graph-Transformer variant** — replace the per-month encoder with a full graph transformer
  if oversmoothing or long-range dependence becomes the bottleneck.

### 2.2 Layer-by-layer spec

Notation: `N` nodes in the (batched) subgraph, `W=12`,
`F_node(Country) ≈ 35` (17 numeric/binary + ~7 region one-hot + 11 alliance multi-hot),
`F_node(Actor) = 3` (`member_count_log`, `recognized_legitimacy_score`,
`financial_resources_tier`) — each node type has its own input projection;
`F_edge = 10` (`event_count` log1p, `weighted_intensity`, `sentiment_mean`, `sentiment_std`,
`days_since_last_event` log1p, `class_distribution[5]`); `d=128`, heads `H=4`, `id_dim=64`,
`P` = target pairs in batch.

| # | Stage | Layer / op | In | Out | Notes |
|---|---|---|---|---|---|
| 0 | Input | node features / per-step edges / node ids | `[N,W,F_node]` / `edge_index_t,edge_attr_t` / `[N]` | — | from §1 |
| 1 | Embed | `nn.Embedding(232, 64)` | `[N]` | `[N,64]` | trainable identity; broadcast over `W` |
| 2 | Input proj (per node type) | `Linear(F_node+64 → d)` + ReLU | `[N,W,F_node+64]` | `[N,W,d]` | separate Linear for Country vs Actor |
| 3 | Spatial hop 1 (per month `t`) | `HeteroConv({SNAPSHOT_EDGE: TransformerConv(d, d//H, heads=H, edge_dim=F_edge, beta=True); BORDERS/MEMBER_OF(+reverse): GATv2Conv/SAGEConv}, aggr='sum')` | `x_dict_t, edge_attr_dict_t` | `[N,d]` | reverse `MEMBER_OF` added so Actors influence Countries |
| 4 | Norm/act | `LayerNorm` → ELU → `Dropout(0.3)` | `[N,d]` | `[N,d]` | |
| 5 | Spatial hop 2 | second `HeteroConv` (same shapes) | `[N,d]` | `[N,d]` | **stop at 2 hops** = the 2-hop ego subgraph; avoids oversmoothing |
| 6 | Norm/act + residual | `LayerNorm` → ELU → `Dropout` + residual from #4 | `[N,d]` | `[N,d]` | residual mitigates oversmoothing |
| 7 | Stack over time | collect the `W` per-month embeddings | `W × [N,d]` | `[N,W,d]` | encoder weights shared across `t` |
| 8 | Temporal | `nn.GRU(d, d, batch_first=True)` (each node = a sequence) | `[N,W,d]` | out `[N,W,d]`, `h_T [N,d]` | |
| 9 | Temporal readout | attention-pool over the `W` outputs (or last hidden) | `[N,W,d]` | `[N,d]` | learned softmax over months → interpretable |
| 10 | Decoder gather | index target nodes `u,v` | `[N,d]` | `h_u,h_v [P,d]` | |
| 11 | Decoder concat | `[h_u ; h_v ; h_u⊙h_v ; e_uv^last]` | — | `[P, 3d+F_edge]` | direction-aware |
| 12 | Decoder MLP | `Linear(3d+F_edge → d)` → ReLU → `Dropout` → `Linear(d → 5)` | `[P,3d+F_edge]` | `[P,5]` | logits |
| 13 | Output | softmax (+ Platt/temperature at inference) | `[P,5]` | `[P,5]` | calibrated probabilities |

**Tensor-shape walkthrough:**
`[N×W×F_node] / per-step edges → (12× shared 2-hop HeteroConv) → [N×W×d] → GRU + temporal
attention → [N×d] → gather (u,v) → [P×(3d+F_edge)] → MLP → [P×5]`.

### 2.3 PyG class map (off-the-shelf vs custom glue)

| Piece | Class | Status |
|---|---|---|
| Identity embedding | `torch.nn.Embedding` | off-the-shelf |
| Per-type input proj | `torch.nn.Linear` (one per type) / `HeteroDictLinear` | off-the-shelf |
| Edge-feature conv | `torch_geometric.nn.TransformerConv(edge_dim=…, beta=True)` (or `GATv2Conv(edge_dim=…)`) | off-the-shelf |
| Structural conv | `torch_geometric.nn.GATv2Conv` / `SAGEConv` | off-the-shelf |
| Hetero wrapper | `torch_geometric.nn.HeteroConv(convs_dict, aggr='sum')` | off-the-shelf |
| Per-snapshot container | `torch_geometric.data.HeteroData` | off-the-shelf |
| Temporal recurrence | `torch.nn.GRU` | off-the-shelf |
| **Per-month encode-and-stack loop** | — | **custom glue (~30 lines in `forward`)** — the only non-trivial code |
| **Node alignment across snapshots** | — | **custom glue** in the Dataset/batching (fixed node ordering, union adjacency) |
| Edge decoder | `nn.Sequential` MLP | trivial custom |
| GNNExplainer | `torch_geometric.explain.Explainer(algorithm=GNNExplainer())`, `task_level='edge'`, `HeteroExplanation` | off-the-shelf |
| Integrated Gradients | `torch_geometric.explain.CaptumExplainer('IntegratedGradients')` | off-the-shelf |

### 2.4 Loss, optimization, imbalance

- **Loss:** class-weighted cross-entropy (weights ∝ inverse class frequency; STATUS_QUO=1.0,
  MATERIAL_CONFLICT ≈ 5×) by default; **focal loss** (`γ≈2`) as a config toggle. Masked to
  **Country→Country** pairs only (Actors never contribute to the loss).
- **Imbalance levers:** negative sampling `K=5` (§1.5) + weighted/focal loss + threshold
  calibration (§2.5). Early-stop on **Macro-F1** (not accuracy) to guard against collapse to
  all-STATUS_QUO.
- **Optimizer / regularization:** AdamW, `lr=1e-3`, `weight_decay=1e-4`, dropout 0.3,
  `LayerNorm` (not BatchNorm — small/variable per-subgraph batches), ELU. Early stopping
  patience 5.
- **Oversmoothing (the #1 risk on a ~232-node graph):** cap at **2 hops**, residual
  connections, LayerNorm. Do not go to 3+ hops.

### 2.5 Calibration

After training, fit **Platt scaling** (logistic regression on logits) or **temperature
scaling** on the validation set → `calibrator.pkl`. Log ECE before and after to show it
helps. The frontend's confidence display depends on calibrated probabilities.

---

## §3 Explainability algorithm (why the model decided)

Both methods run at inference time on the freshly extracted 2-hop temporal subgraph — which
is *why* `01-architecture.md` requires a live forward pass rather than a frozen-embedding
lookup (a static embedding exposes neither structure nor features to attribute over).

**GNNExplainer — structure-level.** For a given prediction on pair `(u,v)`, it learns two
soft masks by optimization: a node mask `M_node ∈ [0,1]^N` and an edge mask
`M_edge ∈ [0,1]^E` over the input subgraph. It multiplies the inputs by the (sigmoid) masks,
re-runs the forward pass, and minimizes the cross-entropy between the masked-graph prediction
and the original predicted class, plus sparsity/entropy regularizers that push masks toward
0/1. The result is a ranked list of *which neighboring countries/actors and which historical
edges* most support the prediction. In PyG:
`Explainer(model, algorithm=GNNExplainer(epochs=200), explanation_type='model',
node_mask_type='attributes', edge_mask_type='object',
model_config=ModelConfig(mode='multiclass_classification', task_level='edge',
return_type='probs'))`, producing a `HeteroExplanation`.

**Integrated Gradients — feature-level.** Attributes the predicted class probability to each
input feature by integrating the gradient along a straight path from a baseline (zero/mean
features) to the actual input:
`IG_i = (x_i − x'_i) · ∫₀¹ ∂F(x' + α(x−x'))/∂x_i dα`, approximated by a Riemann sum over
~50 steps. It satisfies the **completeness axiom** (attributions sum to `F(x) − F(x')`),
which is a built-in correctness check (§6). Output: a signed importance score per feature
per node (e.g. "military expenditure of `u`", "conflict count at `T−3`"), positive = pushed
toward the predicted class. In PyG: `CaptumExplainer('IntegratedGradients')`.

**Surfacing.** Both ride along the `POST /predict` response and render in the frontend's
Explanation Panel as (1) a subgraph highlight (node/edge opacity ∝ GNNExplainer mask) and
(2) a top-N feature bar chart (Integrated Gradients magnitude). The temporal-attention
weights from §2 layer 9 give a third, cheap view: *which of the 12 months* mattered.

---

## §4 Training time, GPU & Weights & Biases

### 4.1 Training time

This graph is tiny by GNN standards (232 nodes), so cost is dominated by **data assembly**,
not GPU math.

- **Recommended regime — full-graph-per-month.** Compute node embeddings `H_t` once per
  month for all 232 nodes, then score *all* of that month's pairs from the shared
  embeddings (no per-pair subgraph). Cost ≈ `186 months × 12 steps` graph-forwards per epoch
  (each sub-millisecond) + one big batched decoder pass → **~10 s–1 min/epoch**. With
  early stopping at ~20–30 epochs → **~5–30 minutes total** on a single T4/P100.
- **Inference-faithful regime — per-ego-subgraph-per-pair** (extracts exactly the
  inference computational graph): CPU-bound subgraph extraction → **~5–15 min/epoch →
  ~1.5–4 h total**. Only use if you must train the exact inference path.
- Either way you will **not** approach Kaggle's weekly quota; T4 vs P100 is negligible at
  this scale. Caching per-month node-feature tensors in RAM (Kaggle has ~30 GB) keeps the
  data-assembly cost down while still training on the **full** dataset.

### 4.2 GPU — Kaggle

Kaggle Notebooks free tier (2025–2026): **NVIDIA T4 ×2** or **P100 16 GB**; **~30 GPU-h/week**
(resets weekly; watch the live session meter); sessions up to **12 h**; `/kaggle/working`
**20 GB** persistent + read-only `/kaggle/input`; **~30 GB RAM**; **internet off by default**
(toggle on so the notebook can `pip install` and reach W&B). A phone-verified account is
required to enable GPU + internet. The full run is minutes, so one weekly quota covers many
experiments and sweeps.

### 4.3 Run on Kaggle (manual upload — two artifacts)

The whole flow is manual through the Kaggle website; **no `kaggle` CLI, no kernel metadata,
no remote execution**. Everything needed lives in the project directory and is copied up by
hand:

1. **Build the dataset locally.** Run the Go exporter against the aggregated DB to produce
   `services/ml/dataset_parquet/` (node_snapshots / snapshot_edges / structural_edges /
   samples_*). This is the full dataset — no subsampling.
2. **Upload the dataset.** kaggle.com → **Datasets → New Dataset** → drag in the whole
   `dataset_parquet/` folder → Create. (On a data refresh, open the dataset → **New Version**
   and re-upload.)
3. **Upload the notebook.** kaggle.com → **Code → New Notebook → File → Import Notebook** →
   pick `services/ml/notebooks/train_geopolitic_gnn.ipynb`. Because the notebook is
   self-contained, nothing else needs to come with it.
4. **Configure the session** (right-hand panel): **Add Input** → attach the Dataset from
   step 2; **Accelerator = GPU (T4 ×2 or P100)**; **Internet = On**; **Add-ons → Secrets →**
   add `WANDB_API_KEY`.
5. **Run All.** The notebook reads the Parquet from `/kaggle/input/<dataset>/`, builds the
   PyG dataset, trains, calibrates, runs the explainers, and logs everything to W&B.
   Checkpoints/artifacts land in `/kaggle/working` **and** stream to W&B (§4.4), so results
   survive even if the local machine never sees the `/kaggle/working` files.

That is the entire loop: **copy `dataset_parquet/` + the notebook to Kaggle, attach, Run All.**

### 4.4 Weights & Biases

```python
import wandb
from kaggle_secrets import UserSecretsClient          # on Kaggle
wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))

run = wandb.init(project="geopolitic-gnn", job_type="train", config={
    "W": 12, "K_neg": 5, "encoder": "HeteroConv+TransformerConv", "temporal": "GRU+attn",
    "hidden": 128, "heads": 4, "id_emb_dim": 64, "lr": 1e-3, "weight_decay": 1e-4,
    "loss": "weighted_ce", "epochs": 50, "patience": 5, "seed": 42,
})
# per epoch:
wandb.log({"epoch": e, "train/loss": loss, "val/macro_f1": mf1, "val/ece": ece,
           **{f"val/f1_{c}": f1[i] for i, c in enumerate(CLASS_NAMES)},
           "val/confusion": wandb.plot.confusion_matrix(
               y_true=y.cpu().numpy(), preds=p.cpu().numpy(), class_names=CLASS_NAMES)})
```

(The W&B API key is read from the Kaggle Secret `WANDB_API_KEY` — never committed; needs
Internet = On.)

**Checkpoint-saving conditions (explicit).** The trainer writes exactly two checkpoint files
and nothing on a fixed interval:

- **`best.pt` — saved only when validation Macro-F1 *strictly improves*** over the running
  best (`val/macro_f1 > best_macro_f1 + 1e-4`). This is the same quantity the early-stopping
  monitor watches (patience = 5 epochs with no improvement → stop). Macro-F1, not accuracy,
  so a model that collapses to all-`STATUS_QUO` never gets saved.
- **`last.pt` — overwritten every epoch**, purely so a run can resume if a 12 h Kaggle
  session is cut off mid-training.
- Each checkpoint stores the full state needed to resume *and* to serve consistently:
  `{model_state, optimizer_state, epoch, best_macro_f1, preprocess, calibrator, config,
  git_sha}`.
- At run end, `best.pt` (plus `preprocess.pkl`, `calibrator.pkl`, `metrics.json`) is logged
  as **one** W&B Artifact version with aliases `best` / `latest`, so the inference server
  always pulls a self-consistent bundle (no scaler/model mismatch).

**Backup & restore from the W&B website.** Because artifacts live in W&B's cloud, a
checkpoint survives a dead Kaggle session, an expired `/kaggle/working`, or a wiped local
disk. To recover, download it from the **run page → Artifacts** tab in the browser, or
programmatically:

```python
art = wandb.use_artifact("geopolitic-gnn:best")   # or :latest / :f1-0.62
ckpt_dir = art.download()                          # → best.pt, preprocess.pkl, calibrator.pkl
# single-file convenience for a checkpoint logged to the run:
wandb.restore("best.pt", run_path="<entity>/geopolitic-gnn/<run_id>")
```

This is the project's off-machine backup of every trained model — no checkpoint depends on
Kaggle or the local Mac staying alive.

**Sweeps (optional).** `wandb sweep sweep.yaml` (bayes over `lr`, `hidden`, `encoder`,
`loss`, `focal_gamma`, `K_neg`; hyperband early-terminate); call
`wandb.agent(SWEEP_ID, train, count=20)` inside the notebook. Given §4.1, ~20 trials fit one
12 h session.

---

## §5 End-to-end flow diagram

```
 geopolitic_aggregated (Neo4j, bolt:7688)
   NodeSnapshot[node×month]   SNAPSHOT_EDGE[u→v×month]   BORDERS/MEMBER_OF (validity)
            │  Go export → services/ml/dataset_parquet/  (reuses label.Classes + timestep)
            ▼
 dataset_parquet/ (partitioned by ts): node_snapshots / snapshot_edges / structural_edges
            │   ── manual upload: dataset_parquet/ + train_geopolitic_gnn.ipynb → Kaggle ──┐
            ▼                                                                              │
 [ Kaggle Notebook (GPU, Internet on) — self-contained ]                                  │
            │  dataset.py — window W=12, label = dominant_class@T+1, K=5 neg-sampling,     │
            │               scaler fit on TRAIN only → preprocess.pkl                      │
            ▼                                                                              │
 PyG samples ── DataLoader ─────────────────────────────────────────────┐                │
            ▼                                                            │                │
 ┌────────────────────────── MODEL (per target pair u→v) ─────────────┐ │                │
 │ input: x[N×W×F_node] (+ 64-dim identity emb)   per-step edge graphs │ │                │
 │   └─ per month t=1..12:  HeteroConv{ SNAPSHOT_EDGE:TransformerConv  │ │                │
 │        (edge_dim), BORDERS/MEMBER_OF:GATv2/SAGE } ×2 hops → H_t[N×d]│ │                │
 │   └─ stack → [N×W×d] → GRU → temporal-attention pool → h[N×d]       │ │                │
 │   └─ decoder MLP([h_u‖h_v‖h_u⊙h_v‖e_uv^last]) → logits[P×5]         │ │                │
 └──────────────────────────────┬─────────────────────────────────────┘ │                │
            softmax + Platt/temperature calibration                      │                │
            ▼                                                            │                │
   5-class P(MATERIAL_CONFLICT … STATUS_QUO)  ──►  GNNExplainer + Integrated Gradients     │
            │                                          (node/edge masks, feature attrib)   │
            ▼                                                                              │
  best.pt / preprocess.pkl / calibrator.pkl / metrics.json ──► W&B cloud (Artifact) ◄──────┘
            │   download / restore from W&B  (run page → Artifacts, or artifact.download())
            ▼
  FastAPI inference  /predict (probs + explanations)   /predict/batch (simulation)
            ▲
            │ proxied by the Go API  (POST /api/v1/predict, /simulate)
            ▼
  React frontend — prediction panel + explanation panel + simulation overlay
```

---

## §6 Verification / acceptance

- **Data layer.** Assert exported tensor shapes are `[N × W × F_node]` and per-step
  `edge_index_t`; no NaNs after scaling+imputation; `node_index.json` is stable across
  builds; the scaler was fit on train months only (assert min/max of fit `ts` ≤ train_end).
- **No leakage.** A test sample at `T` reads only `ts ≤ T` for inputs; splits are
  contiguous in time; val/test negatives are seed-frozen (two runs give identical eval sets).
- **Model smoke test.** A short run (e.g. 3 epochs) on the **full** dataset → loss decreases
  and Macro-F1 > 0 on validation; output probabilities sum to 1.0 (±1e-5) per pair.
- **Calibration.** ECE after Platt/temperature scaling ≤ ECE before, on validation.
- **Explainability.** GNNExplainer mask values ∈ [0,1]; Integrated Gradients satisfies the
  completeness axiom (sum of attributions ≈ `F(x) − F(baseline)` within tolerance).
- **Class order.** `CLASS_NAMES` in Python matches `internal/label.Classes` exactly
  (a single mismatched index silently corrupts every label).
- **Checkpointing.** `best.pt` is written only on a strict val Macro-F1 improvement;
  `last.pt` updates every epoch; both carry `optimizer_state + epoch` so a cut-off run
  resumes. The W&B run shows loss, Macro-F1, per-class F1, and the confusion matrix.
- **Backup/restore.** `wandb.use_artifact("geopolitic-gnn:best").download()` (or the run
  page → Artifacts) returns `best.pt` + `preprocess.pkl` + `calibrator.pkl`, and the FastAPI
  server loads that bundle and serves a `/predict` whose probabilities sum to 1.0.
- **Kaggle run.** Uploading `dataset_parquet/` + the notebook, attaching the dataset, GPU on,
  Internet on, **Run All** completes end-to-end and the best checkpoint appears in W&B.

---

## §7 APIs, services & credentials required

Everything needed to execute this plan. The external **data** APIs (GDELT/World Bank/etc.)
are **not** in this list — ingestion is already done and the ML stage reads only the
aggregated Neo4j DB.

| What | Used for | Auth / credential | Cost |
|---|---|---|---|
| **Neo4j — `geopolitic_aggregated`** (`bolt://localhost:7688`) | the Go exporter reads NodeSnapshots / SNAPSHOT_EDGEs / structural edges → `dataset_parquet/` | local Bolt user/password (already running in Docker) | free / local |
| **Python ML stack** | dataset build, model, training, calibration, explainers, serving | none | free (OSS) |
| → `torch`, `torch-geometric` | the GNN + `HeteroConv`/`TransformerConv`/`GATv2Conv`/`SAGEConv`, `torch_geometric.explain` | — | — |
| → `pyarrow` (or `polars`) | read/write the Parquet dataset | — | — |
| → `scikit-learn`, `joblib` | scaler, region one-hot, Platt scaling, `preprocess.pkl` | — | — |
| → `torchmetrics` | Macro-F1, per-class F1, confusion matrix, ECE | — | — |
| → `captum` | Integrated Gradients (`CaptumExplainer`) | — | — |
| → `wandb` | metric tracking, checkpoints/artifacts, sweeps | `WANDB_API_KEY` | free tier |
| → `fastapi`, `uvicorn` | the inference server (`app.py`) | none | free |
| **Weights & Biases account** | experiment tracking + off-machine checkpoint backup/restore | `WANDB_API_KEY` (env var locally; **Kaggle Secret** named `WANDB_API_KEY` on Kaggle) | free for personal/research |
| **Kaggle account** | free GPU to run the notebook | website login; **phone-verified** to unlock GPU + Internet | free (~30 GPU-h/week) |
| **Go toolchain** | run the existing exporter against Neo4j | none | free |

No paid or keyed third-party service is required: the only secrets are the local Neo4j
credentials and the single `WANDB_API_KEY`.
