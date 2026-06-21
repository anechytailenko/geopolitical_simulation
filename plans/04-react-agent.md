# ReAct Agent & Interactive Demo

## Context

This is the fourth plan document and the project's end goal. Plans 01–03 produced the data and
the model: a heterogeneous spatio-temporal GNN that, for a directed pair (Country `u` → Country
`v`) given history up to month `T`, predicts a **calibrated 5-class probability vector at
`T+1`** — `MATERIAL_CONFLICT, VERBAL_CONFLICT, MATERIAL_COOPERATION, VERBAL_COOPERATION,
STATUS_QUO` — plus **GNNExplainer** (which subgraph mattered) and **Integrated-Gradients**
(which features mattered) explanations, served by `ml.infer.Predictor` /
`services/ml/app.py` (`03-ml-workflow.md` §2–§3).

We now build a **ReAct agent** (Reason → Act → Observe) with an interactive demo:

- **Right** — a chat: the user asks a question in natural language.
- **Left** — the **subgraph that explains the answer**, chosen by **GNNExplainer** (a small
  relevant subgraph, *not* the full ~232-node graph). Clicking a country node that the prompt
  directly concerns opens a popup of the **Integrated-Gradients feature attributions** behind
  the prediction.
- **Top** — 2–3 example-question chips that prefill the prompt.

### Ground truth this design is built on (verified)

- The **Go API (`api/main.go`) is ingestion-only** (`/api/v1/ingest`, `/ingest/status`,
  `/healthz`) — no prediction endpoints. The prediction endpoints sketched in `01` were never
  built in Go.
- **Prediction is served by Python** (`services/ml`). The agent therefore runs on the
  **in-process `ml.infer.Predictor`** + a **read-only** Neo4j client (`bolt://localhost:7688`),
  never writing to the databases.
- **LLM = local Ollama** (user decision: free, no API key), provider-agnostic via LangChain
  `init_chat_model`; `LLM_PROVIDER=ollama|anthropic|openai`. Default tool-calling model
  **`qwen2.5:3b-instruct`** (~3 GB, sized for an 8 GB box — see §10), 7B/14B as upgrades.

### Central design choice: task-shaped tools + a deterministic viz step

A small **local** LLM is weak at chaining many tool calls. So the agent does **not** rely on a
free ReAct loop assembling low-level primitives. Instead:

1. **Task-shaped MCP tools** — each question type is answered by **one** tool call; the heavy
   compute (batched prediction, ranking, explanation) runs server-side. The LLM only has to
   pick the right tool and fill a few typed args (ISO-3 codes, a class name, a month).
2. **A deterministic visualization node** in the LangGraph graph always runs `explain_pair` on
   the answer's **focus pair** and assembles the left-panel payload — so the subgraph + IG
   popup are guaranteed to match the answer, even if the model never explicitly asks to explain.

This keeps the demo robust on a free local model while staying a genuine ReAct agent (the loop
still does NL→ids→answer, multi-turn memory, and "why?" follow-ups).

---

## §1 What the agent does (question taxonomy)

Every answer reduces to the model's atomic operation `predict(source, target, T)` (§2). The
agent reasons in **ISO-3 codes + class names + a month** — never tensors.

| Type | Example (chip) | One tool call |
|---|---|---|
| **1 — Direct** | "What is most likely between the USA and China next month?" | `predict_pair(USA, CHN)` → **argmax** of the 5 probabilities |
| **2 — Group search** | "Within the EU, which pair is most likely to see *material cooperation* next month?" | `best_pair_in_group("EU", MATERIAL_COOPERATION)` → top directed member pair by `P(class)` |
| **3 — Target-fixed search** | "Who is most likely to enter *material conflict* with Ukraine next month?" | `most_likely_counterpart("UKR", MATERIAL_CONFLICT, direction="incoming")` |
| **Why / explain** | "Why did you predict that for A → B?" | `explain_pair(A, B)` → subgraph + IG |
| **Compare** | "Is conflict or cooperation more likely between India and Pakistan?" | `compare_pair(IND, PAK)` |
| **6 — What-if** | "If the USA & China sign a trade deal, what's likely between China and India next month?" | `predict_counterfactual(intervene=(USA,CHN,MATERIAL_COOPERATION), query=(CHN,IND))` → baseline vs counterfactual (§3) |

**Example chips (above the chat):** (1) USA ↔ China direct; (2) EU material-cooperation pair;
(3) who → material conflict with Ukraine; (4) **what-if** "USA & China sign a deal → China–India?".
The left panel always visualizes the answer's **focus pair** (the prompt's A,B for Type 1; the
winning pair for Type 2/3; the **query** pair for Type 6).

**Model quality caveat (from `artifacts/metrics.json`, surfaced in the UI).** The shipped
checkpoint is strong at the extremes (per-class F1: STATUS_QUO 0.85, MATERIAL_CONFLICT 0.67) but
weak in the middle (VERBAL_CONFLICT 0.09, **MATERIAL_COOPERATION 0.07**, VERBAL_COOPERATION 0.40;
`test_macro_f1 ≈ 0.416`). Type 2's flagship chip ranks by `P(MATERIAL_COOPERATION)`, a class the
model predicts weakly — so its output is a **relative ranking, not a confident forecast**. The
demo therefore always shows the **full 5-class distribution + explanation**, never an argmax
alone, and surfaces a low-confidence note; consider defaulting one chip to a well-modelled class
(e.g. MATERIAL_CONFLICT).

---

## §2 Agent ↔ model alignment

The agent composes the model's primitives exactly as 03 serves them — it never re-implements ML
logic:

```
Predictor.predict(source, target, T)            # ml/infer.py:66
   → { source_id, target_id, time_step, probabilities:{CLASS:p ×5}, predicted_class, confidence }
Predictor.predict_batch(pairs, T)               # ml/infer.py:82
   → [ {source_id, target_id, time_step, probabilities, predicted_class, confidence} ]
Predictor.explain(source, target, T)            # ml/infer.py:102 — NOTE the exact keys
   → { target_class,                                              # ← NOT "predicted_class"
       integrated_gradients:{edge_feature_attribution,
         source_node_attribution, target_node_attribution, completeness_gap},
       gnn_explainer_edge_importance:[float] }                    # bare vector, see §3 enrichments
```

**`focus_pair` is added by the MCP tool layer, not by `Predictor`.** The raw `Predictor` returns
exactly the keys above; each task-shaped tool (§3) attaches the `focus_pair` the viz node
consumes. Likewise `explain` today returns a **bare** importance vector + per-node IG vectors —
the renderable subgraph and human-readable feature names are the §3 enrichments.

- **Type 1** = one `predict` → `argmax(probabilities)`.
- **Type 2/3** = build the directed candidate pairs, call `predict_batch`, then `argmax` over
  the **chosen class column** (e.g. `MATERIAL_COOPERATION`). This is exactly the user's "run
  prediction over the pairs and find where the probability for class X is highest".
- **Type 6** = build the window, edit one month-`T` input edge, re-forward (§3 `predict_counterfactual`).
- **Why** = `explain` → left panel.

**Confidence is shown honestly (not over-claimed).** The shipped `calibrator.pkl` is a single
temperature `T ≈ 1.03` (≈ identity), and `metrics.json` shows `test_ece` actually *rose* after
calibration (0.0715 → 0.0733). So the demo labels its number **"model confidence
(temperature-scaled, ≈ uncalibrated on this checkpoint)"** and does not assert calibrated
probabilities. (03 §2.5's "ECE after ≤ before" did not hold on this artifact — flag for a future
recalibration; not a blocker for the demo.)

**Class names, not indices.** The agent uses class *names* (`ml/config.CLASS_NAMES`, mirror of
Go `internal/label.Classes`), so it is immune to index drift.

**"The next discrete monthly time step."** The model predicts `T+1` from inputs up to `T` and
is **inductive over time** (consumes per-time-step features at query time). So "next month" =
predict at the **latest full-window month** `T = max_ts = 197` (2026-06), forecasting
**`T+1 = 198` = 2026-07** — a genuinely unobserved month. Tools default `time_step` to this.
(Training used `T ≤ 196` since it needed an observed label at `T+1`; inference needs none, so
`T = 197` is valid — its window `[186, 197]` is fully present.)

To stop input/forecast month confusion, `get_latest_time_step` returns **both** months
explicitly: `{time_step: 197, input_period: "2026-06", forecast_period: "2026-07"}`. Answers
always name the **`forecast_period`** (the predicted `T+1` month), never the input month.

---

## §3 MCP tools

A **FastMCP** server exposes the tools below. It imports `ml.infer.Predictor` **in-process**
(running the exact trained model — `best.pt` + `preprocess.pkl` + `calibrator.pkl`) and holds a
**read-only** Neo4j driver to `bolt://localhost:7688` for place resolution and subgraph
hydration. The LangGraph agent loads these via `langchain-mcp-adapters` `MultiServerMCPClient`.

**Grounding**

| Tool | Args | Returns |
|---|---|---|
| `get_latest_time_step` | – | `{time_step:197, input_period:"2026-06", forecast_period:"2026-07"}` |
| `resolve_place` | `text` | `{kind:"country"\|"group", name, iso3?, members?:[iso3]}` — one tool maps any NL place to ids: a Country (name/alias/ISO-2→ISO-3 via Neo4j) or a group (`EU`→Q458, `NATO`→Q7184, `ASEAN`→Q7768, … → `MEMBER_OF` members valid at T) |

**Answer (task-shaped — one call answers a whole question)**

| Tool | Args | Returns |
|---|---|---|
| `predict_pair` | `source, target, time_step?` | `{probabilities{CLASS:p}, predicted_class, confidence, focus_pair}` |
| `best_pair_in_group` | `group, relationship_class, time_step?, top_k=5` | `{ranked:[{src,tgt,prob}], focus_pair}` (best by `P(class)`) |
| `most_likely_counterpart` | `country, relationship_class, direction="incoming"\|"outgoing", time_step?, top_k=5` | `{ranked:[{counterpart,prob}], focus_pair}` |
| `compare_pair` | `source, target, time_step?` | `{conflict_p, cooperation_p, status_quo_p, verdict, focus_pair}` |
| `predict_counterfactual` | `intervene_source, intervene_target, intervene_class, query_source, query_target, time_step?, symmetric=true` | `{baseline{probabilities,predicted_class}, counterfactual{probabilities,predicted_class}, delta{CLASS:Δp}, intervened_edge, focus_pair=query}` |

**`compare_pair` aggregation (explicit 5→3 collapse).** `conflict_p = P(MATERIAL_CONFLICT) +
P(VERBAL_CONFLICT)`; `cooperation_p = P(MATERIAL_COOPERATION) + P(VERBAL_COOPERATION)`;
`status_quo_p = P(STATUS_QUO)`. The three buckets sum to 1.0; `verdict = argmax` of the three.

**Explain / render**

| Tool | Args | Returns |
|---|---|---|
| `explain_pair` | `source, target, time_step?` | `{target_class, subgraph{nodes,edges}, feature_attributions{ <source_iso3>:[…], <target_iso3>:[…] }}` — IG keys are **only** the focus pair's `u` and `v` (see below) |

- **Prediction primitives:** `predict_pair`, `best_pair_in_group`, `most_likely_counterpart`,
  `compare_pair`, `predict_counterfactual` — all wrap `Predictor.predict[_batch]` and return a
  **`focus_pair`** the viz node consumes.
- **Reasoning primitive (drives the left panel):** `explain_pair` — wraps `Predictor.explain`.
- Each answer tool returns its focus pair so the deterministic viz node (§4) can explain it.

**Robustness.** Every tool **validates args server-side** — ISO-3 exists, `relationship_class`
∈ `CLASS_NAMES`, `time_step` ∈ `[11, max_ts]` — and returns a clear, recoverable error.
Combined with the small, task-shaped tool set, this is what makes the agent reliable on a weak
local LLM.

**Two ML-serving enrichments — REQUIRED for the left panel to render** (additive to 03's
`explain`, no behavior in 03 is invalidated; a one-line note lands in 03 §3 when implemented).
`Predictor.explain` today returns a *bare* `gnn_explainer_edge_importance: list[float]` plus IG
vectors for only `u`,`v` — neither is directly renderable, so the MCP layer must add:
1. **Subgraph edges** — map `gnn_explainer_edge_importance[i] ↔ ds.snap_pairs[T][i]` (`(u_idx,
   v_idx)`, `dataset.py:101-129`) → ISO-3, emitting `(src_iso3, tgt_iso3, dominant_class@T,
   importance)`. This turns the importance vector into a drawable subgraph.
2. **Human-readable feature names** — label the IG vectors with the ordered names from the
   `Preprocess` bundle: country order = `features.py` `COUNTRY_CONT + COUNTRY_BIN + regions +
   alliance`; edge order = `EDGE_CONT + class_distribution`. So the popup shows
   `military_expenditure_log +0.21`, not an anonymous index.

**IG scope (A1) — only `u` and `v` are feature-clickable.** `Predictor.explain` computes IG for
exactly the focus pair's `source_node_attribution` (`u`), `target_node_attribution` (`v`), and
`edge_feature_attribution` — there is **no** per-node IG for other subgraph nodes. Therefore the
left panel makes **only `u` and `v` clickable for the IG popup**; every other node in the
GNNExplainer subgraph carries **structural importance only** (opacity ∝ mask), not a feature
breakdown.

**Counterfactual mechanism (`predict_counterfactual`, Type 6 — server-side, no retrain).**
1. Build the normal window `ds.build_window(T)` (the 12 monthly `HeteroData` graphs, `dataset.py`).
2. In month `T` (`window[-1][REL_SNAP]`) insert/replace the `intervene_source→intervene_target`
   edge (and its reverse if `symmetric=true`, since a signed deal is bilateral) with a 10-dim
   edge-feature vector whose `class_distribution = one_hot(intervene_class)` and whose continuous
   fields are a plausible "active event" prior (modest `event_count`, `sentiment_mean` signed by
   the class, small `days_since_last_event`) — **standardized with the same `edge_scaler` from
   `preprocess.pkl`** (never refit).
3. Re-run `model.forward(window', pair_index=[query_source, query_target], pair_attr =
   e_query^T)` (`model.py:87-97`) and return both the un-edited **baseline** and the
   **counterfactual** distributions + per-class `delta`. The edit is **local to the call** — no
   global state is mutated — reusing the exact "edit input, re-forward" path GNNExplainer uses.

**Why this does NOT contradict the timestep flow.** The model already maps `inputs ≤ T → T+1`.
The intervention edits **only month-`T` inputs**; the query stays at **`T+1`**. No autoregressive
rollout, no synthesis of future `NodeSnapshot`s beyond the single edited edge — it stays inside
the trained one-step horizon. The 2-hop receptive field means the intervention can only move the
query when the query pair is within 2 hops of the intervened edge — a built-in correctness
property (verified in §12).

**Out of scope (the boundary that *would* break the flow).** Multi-month "then-then" cascades
(`T → T+1 → T+2 → …`) require feeding a prediction back as the next month's input and
synthesizing every other feature for the future month — off-distribution for a one-step model.
Not built; `predict_counterfactual` answers a single step only and rejects multi-step requests.

---

## §4 Agent graph (LangGraph) + why LangGraph/LangChain

```
user message
   │
   ▼
[ ReAct agent ]  create_react_agent(LLM = Ollama, tools = grounding + answer)
   │  reason→act→observe:  resolve_place(...) → ONE answer tool → text answer (+ focus_pair)
   ▼
[ VIZ node ]  (deterministic, NOT LLM-driven)
   │  explain_pair(focus_pair, T) → build subgraph + IG feature_attributions
   ▼
stream `final{ answer, viz }`
```

- The **ReAct loop** turns NL into ids and makes the single answer call (plus multi-turn
  follow-ups / "why?" / compare). Recursion is capped.
- The **viz node is deterministic**: it always explains the answer's focus pair, so the
  left-panel subgraph + IG popup can never disagree with the answer.

**Why LangGraph + LangChain:**
- **ReAct, prebuilt.** `langgraph.prebuilt.create_react_agent` gives the tool-calling
  reason→act→observe loop with state and a recursion limit — no hand-rolled loop.
- **Streaming is the demo.** LangGraph natively streams **tokens + tool-call/tool-result
  events**; the UI shows live reasoning and each tool step, then the deterministic node emits
  the **final viz payload**. Re-creating reliable streaming by hand is the bulk of what
  LangGraph removes.
- **State + a place for the viz node.** A typed `StateGraph` + checkpointer gives multi-turn
  memory and a clean post-agent node for deterministic viz assembly.
- **MCP, zero glue.** `langchain-mcp-adapters` loads the FastMCP tools as LangChain tools.
- **Swappable LLM.** `init_chat_model` / `ChatOllama` — **default Ollama `qwen2.5:3b-instruct`**
  (free, local, tool-calling-capable, ~3 GB — sized for an 8 GB box, see §10);
  `LLM_PROVIDER` switches to Anthropic/OpenAI. Local-LLM weakness at tool-calling is mitigated
  by (a) task-shaped tools, (b) a small tool set, (c) server-side arg validation, (d) the
  deterministic viz step. If a local model still mis-calls tools, flipping
  `LLM_PROVIDER=anthropic` is a one-env change. **Apple-Silicon caveat:** Ollama uses unified
  memory via Metal, so the machine must have the model's RAM *free*; the dev Mac in 03 (~2 GB
  free) can't host even a 3B comfortably → either free RAM or run with `LLM_PROVIDER=anthropic`.

---

## §5 Architecture & services

```
services/
  ml/                      (existing) Predictor, /predict, /explain, best.pt + preprocess + calibrator
  agent/                   (NEW, Python)
    mcp_server.py          FastMCP server — the §3 tools (imports ml.infer.Predictor in-proc + read-only Neo4j 7688); sets GEO_ARTIFACTS_DIR + boot self-check (§5)
    llm.py                 init_chat_model / ChatOllama provider switch (default ollama)
    graph.py               create_react_agent + system prompt + deterministic VIZ node
    schemas.py             AgentAnswer + Viz payload (pydantic)
    server.py              FastAPI: POST /agent/chat → SSE stream of events
    pyproject.toml
frontend/                  (NEW, React + Vite + TypeScript)
    left subgraph canvas (Sigma.js/D3) · right chat · top example chips · IG popup
```

**Data flow:**

```
React (chat · example chips)
   │  POST /agent/chat  (Server-Sent Events)
   ▼
[agent/server.py] → LangGraph ReAct (Ollama) ──tool calls──► [MCP server (§3)]
   ▲   stream: token / tool_call / tool_result / final{answer,viz}      │
   │                                                                      ├─ ml.infer.Predictor (predict · predict_batch · explain)
   │                                                                      │     └─ loads best.pt + preprocess.pkl + calibrator.pkl (03)
   └──────────────────────────────────────────────────────────────────  ┴─ Neo4j geopolitic_aggregated (bolt:7688, READ-ONLY): places, subgraph
   │  final{viz}
   ▼
React: left renders the GNNExplainer subgraph; click an affected node → IG feature popup
```

The agent never writes to Neo4j; predict/explain run in-process against the trained model, so
the demo is a faithful forward pass over the live temporal subgraph (the `01`/`03` §3
requirement). **In-process `Predictor`, not an HTTP hop to `app.py`** — one model definition (no
client/server drift), lower latency, and direct access to the editable `window`/`pair_attr`
needed for `predict_counterfactual` (§3), which an HTTP `/predict` boundary would not expose.

**Artifact path (alignment fix).** The shipped artifacts (`best.pt`, `preprocess.pkl`,
`calibrator.pkl`, `metrics.json`) live at the **repo-root `artifacts/`**, but `Predictor`
defaults to `cfg.artifacts_dir = "artifacts"`, which is **resolved relative to the process CWD**
(`infer.py:39,42,52`, `config.py:55-56,130`). Launched from `services/agent/`, that relative
path misses the real files. So `mcp_server.py` **must set `GEO_ARTIFACTS_DIR` to the absolute
repo-root `artifacts/`** (or run with CWD = repo root) before constructing the `Predictor`.

**Boot self-check (fail fast).** On startup the MCP server (a) loads the `Predictor`, (b)
asserts the `Preprocess` class order == `ml.config.CLASS_NAMES` == Go `internal/label.Classes`,
and (c) runs a fixed `predict("USA", "CHN", 197)` and asserts the probabilities sum to
`1.0 ± 1e-5`. Any failure aborts boot with a clear message — the agent never serves a
mis-located or class-misaligned model.

---

## §6 Streaming protocol & visualization payload

`POST /agent/chat` (body `{message, thread_id?}`) returns an **SSE** stream:

| Event | Payload | UI effect |
|---|---|---|
| `token` | `{text}` | append to the streaming reply (right) |
| `tool_call` | `{name, args}` | show "▸ best_pair_in_group(EU, MATERIAL_COOPERATION)" |
| `tool_result` | `{name, summary}` | mark the step done |
| `final` | the viz payload below | render the left panel + probability chart |
| `error` | `{message}` | inline error bubble |

**Viz payload (`final`):**

```jsonc
{
  "answer": "Most likely DEU → FRA: MATERIAL_COOPERATION (p=0.62) in 2026-07.",
  "time_step": 197, "input_period": "2026-06", "forecast_period": "2026-07",
  "confidence_note": "model confidence (temperature-scaled, ≈ uncalibrated on this checkpoint)",
  "focus_pairs": [
    { "src": "DEU", "tgt": "FRA", "predicted_class": "MATERIAL_COOPERATION", "confidence": 0.62,
      "probabilities": { "MATERIAL_CONFLICT": 0.03, "VERBAL_CONFLICT": 0.05,
                         "MATERIAL_COOPERATION": 0.62, "VERBAL_COOPERATION": 0.22, "STATUS_QUO": 0.08 } }
  ],
  "intervention": null,                            // Type 6 only: {src,tgt,class} of the edited edge
  "subgraph": {                                   // GNNExplainer (top-k important edges)
    "nodes": [ {"id":"DEU","name":"Germany","type":"Country","importance":1.0,"ig_clickable":true},
               {"id":"FRA","name":"France","type":"Country","importance":1.0,"ig_clickable":true},
               {"id":"USA","name":"United States","type":"Country","importance":0.41,"ig_clickable":false} ],
    "edges": [ {"src":"USA","tgt":"DEU","dominant_class":"MATERIAL_COOPERATION","importance":0.41} ]
  },
  "feature_attributions": {                        // Integrated Gradients — ONLY the focus pair's u & v
    "DEU": [ {"feature":"military_expenditure_log","attribution":0.21},
             {"feature":"conflict_intensity","attribution":-0.14} ],
    "FRA": [ {"feature":"vdem_polyarchy_score","attribution":0.09} ]
  }
}
```

- **Left panel** draws `subgraph`: node/edge **opacity ∝ `importance`** (GNNExplainer mask). For
  Type 6 the **intervened** edge (`intervention.src→intervention.tgt`) is drawn in a distinct
  style.
- **IG popup is restricted to the focus pair's `u` and `v`** (the only nodes with
  `ig_clickable:true`). Clicking one opens its `feature_attributions` (top-N by `|attribution|`,
  signed: pushes toward vs away from the predicted class) — already in the payload, no extra
  call. Every other subgraph node shows **structural importance only** (no feature breakdown
  exists for it, per §3 A1).
- A small per-focus-pair probability bar chart renders under the chat answer; the `confidence`
  number is shown with `confidence_note` so it is never presented as a calibrated probability.
  For Type 6 the chart shows **baseline vs counterfactual** side by side with the per-class delta.

---

## §7 Frontend

**Stack:** Vite + React + TypeScript; **Zustand** for UI state (current viz, selected node);
**SSE** via `fetch` + `ReadableStream`; **Sigma.js** (WebGL) for the subgraph (D3-force for a
smaller view).

```
┌──────────────────────────── Geopolitic Agent ────────────────────────────┐
│ [USA↔CHN next month?] [EU material-coop pair?] [Conflict w/ Ukraine?]      │  ← example chips
│ [What-if: USA & CHN sign a deal → CHN–IND?]                                │  ← Type 6 chip
├───────────────────────────────────┬───────────────────────────────────────┤
│  SUBGRAPH  (why — GNNExplainer)   │  CHAT                                   │
│      ●USA                         │  you: Within the EU, which pair is most │
│        \  importance-weighted     │       likely material cooperation?      │
│   ●DEU ═══════ ●FRA  (affected)   │  agent ▸ get_latest_time_step → 2026-07 │
│     │ click → IG popup:           │        ▸ resolve_place(EU) → 27 members │
│     │  military_expenditure +0.21 │        ▸ best_pair_in_group(MAT_COOP)   │
│     │  conflict_intensity   −0.14 │  agent: DEU → FRA, p = 0.62 (2026-07)   │
│                                   │  ___________________________  [ Send ]  │
└───────────────────────────────────┴───────────────────────────────────────┘
```

Chips prefill the input; Send opens the SSE stream; reasoning + tool steps render live on the
right; on `final` the left panel renders the subgraph + probability chart; clicking an
`affected` node shows the IG popup.

---

## §8 Cross-plan alignment

- **03 (model) — exact reuse.** The agent calls `Predictor.predict / predict_batch / explain`
  with the same signatures, the canonical `CLASS_NAMES` order, and the `time_step` convention.
  The only changes to 03's serving code are **additive**: the **two `explain` enrichments** (§3:
  subgraph edges + `feature_names`) and `predict_counterfactual` (an input-edge edit +
  re-forward using the *same* model and `edge_scaler`, no retrain). Nothing in 03 is
  invalidated; a one-line note is added to 03 §3 when they're implemented.
- **Checkpoint quality (from `artifacts/metrics.json`) — keep claims honest.**
  `best_val_macro_f1 ≈ 0.418`, `test_macro_f1 ≈ 0.416`; per-class F1 STATUS_QUO 0.85,
  MATERIAL_CONFLICT 0.67, VERBAL_COOPERATION 0.40, VERBAL_CONFLICT 0.09, MATERIAL_COOPERATION
  0.07; calibration ≈ identity (T ≈ 1.03, ECE not improved). The demo therefore always shows the
  full distribution + explanation and surfaces the low-confidence caveat (§1, §2, §6) — it never
  implies more certainty than this checkpoint supports.
- **01 (architecture) — this realizes the frontend.** 01's "graph canvas + prediction panel +
  explanation panel + simulation" is **realized** by this chat-driven demo (the explanation
  panel = the GNNExplainer subgraph + IG popup; chat replaces the manual pair-picker). 01's
  **Go** prediction endpoints were **not built**; prediction is served by **Python** and the
  agent talks to it directly. (01 can later be reconciled to note Go is ingestion-only; no
  behavior depends on its predict endpoints.)
- **02 (ingestion) — untouched.** The agent only reads `geopolitic_aggregated`.
- **No contradiction across 01–04:** one model, one class order, one time convention; Go =
  ingestion, Python = prediction, Agent = orchestration, Frontend = display.

---

## §9 APIs, services & credentials required

| What | Used for | Credential | Cost |
|---|---|---|---|
| **Ollama** (local) | the ReAct LLM (default) — **`qwen2.5:3b-instruct`** (~3 GB, 8 GB box; 7B/14B as upgrades — §10) | none (local) | **free** |
| `langgraph`, `langchain`, `langchain-core` | ReAct agent, streaming, state | none | free (OSS) |
| `langchain-mcp-adapters`, `mcp` (FastMCP) | serve + load the §3 tools | none | free |
| `langchain-ollama` (+ optional `langchain-anthropic`/`langchain-openai`) | LLM bindings; provider switch | only if switching to a paid provider | free default |
| `services/ml` (`ml.infer.Predictor`) | predict / explain (in-process) | the trained `best.pt` bundle (03) | free |
| `neo4j` driver → `bolt://localhost:7688` | place resolution + subgraph (read-only) | local Bolt user/pass | free / local |
| `fastapi`, `uvicorn`, `sse-starlette` | `/agent/chat` SSE server | none | free |
| React + Vite + TypeScript (+ Sigma.js / D3) | the demo UI | none | free |

**No paid API key is required with the Ollama default.** Switching to Anthropic/OpenAI adds one
key via `LLM_PROVIDER`.

**Dependency gap (new `services/agent/pyproject.toml`).** `services/ml/pyproject.toml` does
**not** yet carry the agent stack — add `langchain`, `langchain-core`, `langchain-ollama` (+
optional `langchain-anthropic`/`langchain-openai`), `langgraph`, `langchain-mcp-adapters`,
`mcp`/`fastmcp`, and `sse-starlette` in the agent package. The agent imports
`ml.infer.Predictor` in-process, so it depends on the existing `services/ml` package too.

---

## §10 Memory & compute budget

**No GPU is required.** The local LLM runs on CPU/Metal and the 232-node GNN forward pass is
sub-second on CPU. The dominant consumer by far is the LLM.

| Component | Resident RAM | Notes |
|---|---|---|
| **Ollama `qwen2.5:3b-instruct` Q4_K_M** (default) | **~2.3 GB weights + ~0.5–1 GB KV ≈ ~3 GB** | tool-calling-capable; keep `num_ctx ≤ 8k` (prompts = system + tool schemas + a few turns); each +1k ctx ≈ +60–120 MB KV |
| `Predictor` (best.pt + per-ts dataset tensors in RAM) | **~0.5–1 GB** | CPU-only; node features ~16 MB, edges loaded per-ts |
| FastMCP + FastAPI + LangGraph process | **~0.3–0.5 GB** | |
| Neo4j **client** driver (read-only) | **~50 MB** | the Neo4j **server** Docker container (~1–2 GB) is already running from 02/03 — counted separately |
| Frontend (Vite dev / Node) | **~0.3 GB** | ~0 if served as a static build |
| **Agent-side working set (3B default)** | **~4–5 GB** | fits an **8 GB** machine alongside the existing Neo4j container |

**Upgrade tiers.** `qwen2.5:7b` / `llama3.1:8b` Q4 ≈ 5 GB weights → ~6–8 GB resident (better
tool-calling, needs **16 GB**); `qwen2.5:14b` Q4 ≈ ~9 GB (≥16 GB headroom); 32B/70B = 20–40+ GB,
unnecessary at this scale. **Apple-Silicon caveat (repeat of §4):** Ollama uses unified memory,
so the model's RAM must be *free*; the dev Mac in 03 (~2 GB free) can't host even a 3B
comfortably → free RAM or set `LLM_PROVIDER=anthropic` (one env change) and the Mac then only
runs the Predictor (~1–2 GB).

### §10.1 Running the LLM: host Ollama vs Docker (justification)

Ollama can run either **on the host** or **as a container**, and the agent supports both with
**no code change** — it reads `OLLAMA_BASE_URL` (default `http://localhost:11434`, which a
container maps to), so switching is one env var.

| | Host Ollama | Dockerized Ollama (`infra/docker`, `--profile llm`) |
|---|---|---|
| Reproducibility | depends on a host install | **pinned image + cached model volume**; one-command bring-up/teardown |
| Parity with the stack | separate from Docker | **matches Neo4j**, which is already dockerized |
| GPU on **macOS** | **uses Metal** (fast) | **CPU-only** — Docker Desktop containers get no Metal/GPU |
| GPU on **Linux+NVIDIA** | uses the GPU | uses the GPU via `nvidia-container-toolkit` (see the compose `deploy` block) |
| Footprint | model in `~/.ollama` | model in the `ollama_models` volume |

**Decision (implemented).** Ship Ollama as an **opt-in Docker Compose service** (`profiles:
["llm"]` in `infra/docker/docker-compose.yml`) — the reproducible default for **CI, Linux+GPU,
and anyone who wants the whole stack containerized** — while keeping **host Ollama the faster
path on Apple Silicon** (Metal). This is purely additive: nothing else in 04 changes, and the
`docker compose up` that starts the Neo4j DBs does **not** pull the 3 GB model unless you opt in
with `--profile llm`. Run/pull commands are in `COMMANDS.md` §11e/§12.

---

## §11 Artifact alignment (the model now exists at `artifacts/`)

- **Where the files are.** `best.pt`, `preprocess.pkl`, `calibrator.pkl`, `metrics.json` live at
  the **repo-root `artifacts/`**, not `services/ml/artifacts/`.
- **Path fix.** `Predictor` defaults to `cfg.artifacts_dir = "artifacts"` resolved against the
  process CWD (`infer.py:39,42,52`). Set **`GEO_ARTIFACTS_DIR=<repo-root>/artifacts`** (absolute)
  in the MCP server, or launch with CWD = repo root.
- **Boot self-check.** Load `Predictor`; assert `Preprocess` class order == `ml.config.CLASS_NAMES`
  == Go `internal/label.Classes`; run `predict("USA","CHN",197)` and assert probabilities sum to
  `1.0 ± 1e-5`; fail boot otherwise.
- **Calibrator reality.** `calibrator.pkl` is a single temperature `T ≈ 1.03` (≈ identity);
  ECE not improved (§2, §8). Confidence is shown as temperature-scaled, not "calibrated".

---

## §12 Verification (test cases)

**Grounding / `resolve_place`.** `Germany | DEU | DE → DEU`; `the EU | European Union → {group,
Q458, 27 members valid at T}`; `NATO → Q7184` **includes FIN/SWE at T=197 but excludes them at
an earlier T** (temporal `MEMBER_OF`); an unknown place → recoverable error.

**Arg validation (server-side).** bad ISO-3 `"XXX"` rejected; bad class `"WAR"` rejected **with
the valid `CLASS_NAMES` list** in the error; `time_step` outside `[11, 197]` rejected; omitted
`time_step` → defaults to **197**.

**`predict_pair`.** probabilities sum to `1.0 ± 1e-5`; `predicted_class = argmax`; `confidence =
max`; **deterministic across two calls** (eval mode, no dropout); a **quiet/no-edge dyad** (two
small countries with no SNAPSHOT_EDGE at 197) still returns a valid distribution (zero edge
vector, `infer.py:119-125`) without crashing.

**`best_pair_in_group`.** ranked **descending by `P(chosen class)`**; `focus_pair = ranked[0]`;
directed pairs only (`src ≠ tgt`); `top_k` honored; a single-member group → clear error.

**`most_likely_counterpart`.** `incoming` vs `outgoing` give **different rankings** for an
asymmetric case; self-pairs excluded; counterparts ∈ the 221 Countries (never Actors).

**`compare_pair`.** `conflict_p / cooperation_p / status_quo_p` sum to 1.0; computed per §3's
5→3 collapse; `verdict = argmax` of the three buckets.

**`explain_pair`.** `gnn_explainer_edge_importance ∈ [0,1]`; IG **`completeness_gap < 5e-2`**
(reuse the existing `test_train_smoke.py` threshold); subgraph edges resolve `importance[i] ↔
snap_pairs[T][i] →` ISO-3; `feature_attributions` carry **human-readable** names and contain
**only** the focus pair's `u`, `v` keys (per §3 A1); entries are top-N by `|attribution|`.

**`predict_counterfactual` (new).** returns valid `baseline` + `counterfactual` distributions;
when A,B,C are within 2 hops the **counterfactual ≠ baseline** (the intervention moves the
prediction); when the query pair is **>2 hops** from the intervened edge the counterfactual
**≈ baseline** (receptive-field correctness); the call **reads only ts ≤ T** (no T+1 peek) and
**mutates no global state** (a following `predict_pair` returns the un-intervened baseline);
a multi-month "then-then" request is **rejected / explained as one-step-only**.

**Agent behavior (4 chips).** Type 1 → argmax for (A,B); Type 2 → EU pair maximizing
`P(MATERIAL_COOPERATION)` **with the low-confidence caveat surfaced** (§1); Type 3 → country
maximizing `P(MATERIAL_CONFLICT)` toward Ukraine; Type 6 → counterfactual whose **focus pair =
query (B,C)** and whose subgraph **highlights the intervened A-B edge**. Each yields a populated
`final` viz; the deterministic viz node explains the focus pair even when the LLM never says
"why".

**Deterministic viz.** every answered question's `final` event has a non-empty `subgraph`
(importances ∈ [0,1]) and `feature_attributions` (human-readable names) for the focus pair;
clicking a `ig_clickable` node renders the popup; non-clickable nodes show structural importance
only.

**Streaming end-to-end.** `POST /agent/chat` emits `token` → `tool_call`/`tool_result` →
`final`; the React left panel renders the subgraph + probability chart.

**Local-LLM robustness.** a malformed tool call (missing/extra arg) → server-side validation
returns a recoverable error → the agent retries → correct answer within the recursion cap; cap
reached → graceful "couldn't resolve" (no infinite loop); `LLM_PROVIDER=anthropic` runs the same
graph/tools with **no code change**.

**Memory / perf smoke.** cold start (model + dataset load) within a few seconds; a single
`predict` < ~1–2 s CPU; `explain` (GNNExplainer 200 epochs + IG ~50 steps) < ~10–20 s; with
`qwen2.5:3b` resident the total agent-side RSS stays within the §10 budget (~4–5 GB).

**Artifact alignment & safety.** the MCP server boots with `GEO_ARTIFACTS_DIR` → repo-root
`artifacts/` and the §11 self-check passes (class order matches, probs sum to 1); class names +
`time_step` match 03; the agent never bypasses the `Predictor`; Neo4j access is read-only (no
writes/deletes) — the demo cannot harm the research data.
