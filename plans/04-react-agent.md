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

- Greenfield: no agent / MCP / frontend code yet.
- The **Go API (`api/main.go`) is ingestion-only** (`/api/v1/ingest`, `/ingest/status`,
  `/healthz`) — no prediction endpoints. The prediction endpoints sketched in `01` were never
  built in Go.
- **Prediction is served by Python** (`services/ml`). The agent therefore runs on the
  **in-process `ml.infer.Predictor`** + a **read-only** Neo4j client (`bolt://localhost:7688`),
  never writing to the databases.
- **LLM = local Ollama** (user decision: free, no API key), provider-agnostic via LangChain
  `init_chat_model`; `LLM_PROVIDER=ollama|anthropic|openai`. Use a tool-calling-capable local
  model (`qwen2.5` / `llama3.1`).

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

**Example chips (above the chat):** (1) USA ↔ China direct; (2) EU material-cooperation pair;
(3) who → material conflict with Ukraine. The left panel always visualizes the answer's
**focus pair** (the prompt's A,B for Type 1; the winning pair for Type 2/3).

---

## §2 Agent ↔ model alignment

The agent composes the model's primitives exactly as 03 serves them — it never re-implements ML
logic:

```
Predictor.predict(source, target, T)
   → { probabilities:{CLASS:p ×5}, predicted_class, confidence }
Predictor.predict_batch(pairs, T)
   → [ {source, target, probabilities, predicted_class} ]
Predictor.explain(source, target, T)
   → { predicted_class, integrated_gradients:{edge_feature_attribution,
       source_node_attribution, target_node_attribution, completeness_gap},
       gnn_explainer_edge_importance }
```

- **Type 1** = one `predict` → `argmax(probabilities)`.
- **Type 2/3** = build the directed candidate pairs, call `predict_batch`, then `argmax` over
  the **chosen class column** (e.g. `MATERIAL_COOPERATION`). This is exactly the user's "run
  prediction over the pairs and find where the probability for class X is highest".
- **Why** = `explain` → left panel.

**Class names, not indices.** The agent uses class *names* (`ml/config.CLASS_NAMES`, mirror of
Go `internal/label.Classes`), so it is immune to index drift.

**"The next discrete monthly time step."** The model predicts `T+1` from inputs up to `T` and
is **inductive over time** (consumes per-time-step features at query time). So "next month" =
predict at the **latest full-window month** `T = max_ts = 197` (2026-06), forecasting
**`T+1 = 198` = 2026-07** — a genuinely unobserved month. Tools default `time_step` to this;
`get_latest_time_step` returns it (with `iso_period`) so the LLM can name the forecast month.
(Training used `T ≤ 196` since it needed an observed label at `T+1`; inference needs none, so
`T = 197` is valid.)

---

## §3 MCP tools

A **FastMCP** server exposes the tools below. It imports `ml.infer.Predictor` **in-process**
(running the exact trained model — `best.pt` + `preprocess.pkl` + `calibrator.pkl`) and holds a
**read-only** Neo4j driver to `bolt://localhost:7688` for place resolution and subgraph
hydration. The LangGraph agent loads these via `langchain-mcp-adapters` `MultiServerMCPClient`.

**Grounding**

| Tool | Args | Returns |
|---|---|---|
| `get_latest_time_step` | – | `{time_step, iso_period}` |
| `resolve_place` | `text` | `{kind:"country"\|"group", name, iso3?, members?:[iso3]}` — one tool maps any NL place to ids: a Country (name/alias/ISO-2→ISO-3 via Neo4j) or a group (`EU`→Q458, `NATO`→Q7184, `ASEAN`→Q7768, … → `MEMBER_OF` members valid at T) |

**Answer (task-shaped — one call answers a whole question)**

| Tool | Args | Returns |
|---|---|---|
| `predict_pair` | `source, target, time_step?` | `{probabilities{CLASS:p}, predicted_class, confidence, focus_pair}` |
| `best_pair_in_group` | `group, relationship_class, time_step?, top_k=5` | `{ranked:[{src,tgt,prob}], focus_pair}` (best by `P(class)`) |
| `most_likely_counterpart` | `country, relationship_class, direction="incoming"\|"outgoing", time_step?, top_k=5` | `{ranked:[{counterpart,prob}], focus_pair}` |
| `compare_pair` | `source, target, time_step?` | `{conflict_p, cooperation_p, status_quo_p, verdict, focus_pair}` |

**Explain / render**

| Tool | Args | Returns |
|---|---|---|
| `explain_pair` | `source, target, time_step?` | `{predicted_class, subgraph{nodes,edges}, feature_attributions{iso3:[{feature,attribution}]}}` |

- **Prediction primitives:** `predict_pair`, `best_pair_in_group`, `most_likely_counterpart`,
  `compare_pair` — all wrap `Predictor.predict_batch` and return a **`focus_pair`** the viz node
  consumes.
- **Reasoning primitive (drives the left panel):** `explain_pair` — wraps `Predictor.explain`.
- Each answer tool returns its focus pair so the deterministic viz node (§4) can explain it.

**Robustness.** Every tool **validates args server-side** — ISO-3 exists, `relationship_class`
∈ `CLASS_NAMES`, `time_step` ∈ `[11, max_ts]` — and returns a clear, recoverable error.
Combined with the small, task-shaped tool set, this is what makes the agent reliable on a weak
local LLM.

**Two additive ML-serving enrichments** (implemented alongside this plan, **consistent with 03,
not a contradiction** — they only enrich the `explain` output):
1. `Predictor.explain` also returns the **subgraph edges** as
   `(src_iso3, tgt_iso3, dominant_class@T, importance)` by mapping
   `gnn_explainer_edge_importance[i] ↔ dataset.snap_pairs[T][i]` — so `explain_pair` returns a
   renderable subgraph, not a bare importance vector.
2. `Predictor.explain` also returns the ordered **`feature_names`** (already in the `Preprocess`
   bundle's country feature ordering) so the IG attributions are **human-labeled** in the popup
   (e.g. `military_expenditure_log +0.21`). A one-line note will be added to `03` §3 when these
   land.

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
- **Swappable LLM.** `init_chat_model` / `ChatOllama` — **default Ollama** (free, local);
  `LLM_PROVIDER` switches to Anthropic/OpenAI. Local-LLM weakness at tool-calling is mitigated
  by (a) task-shaped tools, (b) a small tool set, (c) server-side arg validation, (d) the
  deterministic viz step. If a local model still mis-calls tools, flipping
  `LLM_PROVIDER=anthropic` is a one-env change.

---

## §5 Architecture & services

```
services/
  ml/                      (existing) Predictor, /predict, /explain, best.pt + preprocess + calibrator
  agent/                   (NEW, Python)
    mcp_server.py          FastMCP server — the §3 tools (imports ml.infer.Predictor in-proc + read-only Neo4j 7688)
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
requirement).

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
  "time_step": 197, "iso_period": "2026-07",
  "focus_pairs": [
    { "src": "DEU", "tgt": "FRA", "predicted_class": "MATERIAL_COOPERATION", "confidence": 0.62,
      "probabilities": { "MATERIAL_CONFLICT": 0.03, "VERBAL_CONFLICT": 0.05,
                         "MATERIAL_COOPERATION": 0.62, "VERBAL_COOPERATION": 0.22, "STATUS_QUO": 0.08 } }
  ],
  "subgraph": {                                   // GNNExplainer (top-k important edges)
    "nodes": [ {"id":"DEU","name":"Germany","type":"Country","importance":1.0,"affected":true},
               {"id":"FRA","name":"France","type":"Country","importance":1.0,"affected":true},
               {"id":"USA","name":"United States","type":"Country","importance":0.41,"affected":false} ],
    "edges": [ {"src":"USA","tgt":"DEU","dominant_class":"MATERIAL_COOPERATION","importance":0.41} ]
  },
  "feature_attributions": {                        // Integrated Gradients, for the popup
    "DEU": [ {"feature":"military_expenditure_log","attribution":0.21},
             {"feature":"conflict_intensity","attribution":-0.14} ],
    "FRA": [ {"feature":"vdem_polyarchy_score","attribution":0.09} ]
  }
}
```

- **Left panel** draws `subgraph`: node/edge **opacity ∝ `importance`** (GNNExplainer mask);
  `affected:true` nodes are highlighted and **clickable**.
- **Clicking an affected node** opens a popup of its `feature_attributions` (top-N by
  `|attribution|`, signed: pushes toward vs away from the predicted class) — the requested
  Integrated-Gradients view. Data is already in the payload, so the popup needs no extra call.
- A small per-focus-pair probability bar chart renders under the chat answer.

---

## §7 Frontend

**Stack:** Vite + React + TypeScript; **Zustand** for UI state (current viz, selected node);
**SSE** via `fetch` + `ReadableStream`; **Sigma.js** (WebGL) for the subgraph (D3-force for a
smaller view).

```
┌──────────────────────────── Geopolitic Agent ────────────────────────────┐
│ [USA ↔ CHN next month?] [EU material-coop pair?] [Conflict with Ukraine?] │  ← example chips
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
  The only change to 03's serving code is the **two additive `explain` enrichments** (§3:
  subgraph edges + `feature_names`) — purely additive; nothing in 03 is invalidated. A one-line
  note is added to 03 §3 when they're implemented.
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
| **Ollama** (local) | the ReAct LLM (default) — `qwen2.5` / `llama3.1` (tool-calling) | none (local) | **free** |
| `langgraph`, `langchain`, `langchain-core` | ReAct agent, streaming, state | none | free (OSS) |
| `langchain-mcp-adapters`, `mcp` (FastMCP) | serve + load the §3 tools | none | free |
| `langchain-ollama` (+ optional `langchain-anthropic`/`langchain-openai`) | LLM bindings; provider switch | only if switching to a paid provider | free default |
| `services/ml` (`ml.infer.Predictor`) | predict / explain (in-process) | the trained `best.pt` bundle (03) | free |
| `neo4j` driver → `bolt://localhost:7688` | place resolution + subgraph (read-only) | local Bolt user/pass | free / local |
| `fastapi`, `uvicorn`, `sse-starlette` | `/agent/chat` SSE server | none | free |
| React + Vite + TypeScript (+ Sigma.js / D3) | the demo UI | none | free |

**No paid API key is required with the Ollama default.** Switching to Anthropic/OpenAI adds one
key via `LLM_PROVIDER`.

---

## §10 Verification

- **Tools (unit, synthetic — no live DB):** `resolve_place("Germany")→DEU`,
  `resolve_place("the EU")→{group, members…}`; `predict_pair` probabilities sum to 1.0;
  `best_pair_in_group`/`most_likely_counterpart` return pairs sorted by the requested class
  probability and a `focus_pair`; arg validation rejects bad ISO-3 / class / time_step.
- **Agent behavior (the 3 chips):** Type 1 returns the argmax class for (A,B); Type 2 the EU
  pair maximizing `P(MATERIAL_COOPERATION)`; Type 3 the country maximizing
  `P(MATERIAL_CONFLICT)` toward Ukraine — each with a populated `final` viz.
- **Deterministic viz:** for every answered question the `final` event contains a non-empty
  `subgraph` (importances ∈ [0,1]) and `feature_attributions` (human-readable feature names)
  for the focus pair; clicking an affected node renders the popup.
- **Streaming end-to-end:** `POST /agent/chat` emits `token` → `tool_call`/`tool_result` →
  `final`; the React left panel renders the subgraph + probability chart.
- **Alignment & safety:** class names + `time_step` match 03; the agent never bypasses the
  Predictor; Neo4j access is read-only (no writes/deletes) — the demo cannot harm the research
  data.
