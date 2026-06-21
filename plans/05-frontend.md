# Frontend & Interactive Demo

## Context

This is the fifth and final plan document. Plans 01–04 produced the data, the model, and the
**agent**: a ReAct agent served over Server-Sent Events at `POST /agent/chat` (`services/agent`,
port 8100), whose `final` event carries the visualization payload defined in
[`04-react-agent.md` §6](04-react-agent.md) — the answer, the focus pair's calibrated
probabilities, the **GNNExplainer** subgraph (which countries/edges mattered), and the
**Integrated-Gradients** feature attributions (which features mattered). The frontend is the last
missing piece: `01-architecture.md` sketched a graph canvas + prediction panel + explanation
panel, and `04-react-agent.md` §7 specified the **chat-driven realization** of it.

This document specifies that frontend — a single-page app that talks **only** to the agent SSE
server (never to the Go API or the Python ML service directly) and renders the live reasoning,
answer, subgraph, and explanations.

### Reference & the two changes from it

The visual reference is the static mockup `Screenshot 2026-06-19 at 22.27.07.png` (itself built
from the 04 §6 example payload). It is adopted **with two deliberate changes** and one stack
confirmation:

1. **Example-question chips move** from the top-right header to a **horizontally scrollable strip
   directly above the chat input** — the user scrolls left/right to pick a question.
2. **Theme recolored green → "Electric purple."** The brand/chrome green becomes a vivid purple;
   the **five relationship-class colors stay a separate semantic legend** (only the chrome
   changes).
3. The reasoning subgraph is rendered with **SVG + d3-force** (04 §7's "D3-force for a smaller
   view"; the explainer returns only ~10 nodes / ~12 edges), and tests use **Vitest + React
   Testing Library**.

### Ground truth this design is built on (verified)

- `services/frontend/` is **greenfield** (no code yet).
- The agent exposes `POST /agent/chat` (SSE via `EventSourceResponse`) and `GET /health` on
  `:8100` (`AGENT_PORT`), with **no CORS middleware** — so local dev uses a **Vite proxy**
  (`/agent → http://localhost:8100`); a deployed static build would add a one-line CORS
  middleware to the agent (§8, no behavior change to 04).
- Node 22 + npm are available. The stack (04 §7) is **Vite + React + TypeScript + Zustand**, SSE
  via `fetch` + `ReadableStream`.
- The class order is the canonical `CLASS_NAMES` (`MATERIAL_CONFLICT, VERBAL_CONFLICT,
  MATERIAL_COOPERATION, VERBAL_COOPERATION, STATUS_QUO`) — the single source of truth from
  03/04; the frontend mirrors it once and never reorders.

---

## End-to-end architecture (under the hood)

The system has two clearly separated phases. **Build-time** (ingestion → aggregate → export →
train) is where every database and external source lives; it runs once, on a data refresh.
**Serve-time** (a chat request) touches **no database at all** — the agent answers from the
parquet export + trained artifacts loaded in-process. The frontend lives entirely at the
serve-time end and speaks to exactly one origin.

```
BUILD-TIME  (run once / on data refresh — plans 02 & 03)
  External sources: GDELT (BigQuery) · World Bank · Wikidata · ACLED · V-Dem · SIPRI · UNDP
        │  POST /api/v1/ingest   (manual trigger; Go owns all fetching / labeling / aggregation)
        ▼
  Go ingestion API  (:8080, services/api)  ──writes──►  Neo4j geopolitic_raw        (:7687)
        │  Build + BigQuery server-side aggregation  └►  Neo4j geopolitic_aggregated (:7688) ◄ ML input
        ▼
  services/ml/export/neo4j_to_parquet.py   (READ-ONLY export of :7688 — "no writes/deletes")
        ▼
  services/ml/dataset_parquet/{node_snapshots, snapshot_edges, structural_edges}.parquet
        +  Kaggle training (plans 03)  ──►  artifacts/{best.pt, preprocess.pkl, calibrator.pkl}

SERVE-TIME  (every chat request — NO database is touched)
  React frontend (:5173 dev)
        │  POST /agent/chat  (SSE)        [ Vite proxy:  /agent → http://localhost:8100 ]
        ▼
  Agent SSE server (:8100, services/agent/agent/server.py)
        │  LangGraph ReAct loop · LLM = local Ollama qwen2.5:3b  (LLM_PROVIDER switch)
        ▼
  task-shaped tools (services/agent tools_core)   ── also exposed as a FastMCP server (stdio)
        ▼
  ml.infer.Predictor  (IN-PROCESS forward pass)  +  PlaceResolver (pycountry + parquet MEMBER_OF)
        │  loaded ONCE at boot from:
        ▼
  services/ml/dataset_parquet/   +   artifacts/best.pt · preprocess.pkl · calibrator.pkl
        ▲  stream back:  token → tool_call → tool_result → final{ viz }
        └──────────────────────────────────────────────────►  React renders subgraph + chart
```

- **The frontend only ever talks to `:8100`** (one origin, via the Vite proxy in dev). It never
  reaches Neo4j, the Go API, or BigQuery — those are not in its dependency graph at all.
- **Serve-time touches no database.** Prediction, explanation, and place resolution all run
  **in-process** over the **parquet export + artifacts**, so a chat request is a pure forward pass
  — lower latency and structurally unable to alter the research data. Neo4j + the Go API +
  BigQuery are **build-time only** (ingestion → aggregate → export), strictly upstream of the
  parquet the agent reads.
- **Reconciliation with 04.** 04 §3/§5 describes a "read-only Neo4j client" for place resolution;
  the shipped agent instead resolves places from the **same exported data** (parquet `MEMBER_OF`)
  + `pycountry`, so it is **DB-free at serve time**. Equivalent data, simpler and safer — recorded
  here (and in §10) so 01–05 stay consistent.

---

## §1 Layout & screens

A three-region single page (the chips relocated to above the chat input; chrome in purple):

```
┌──────────────────────────────── Geopolitic Agent ─────────────────────────── local model ─┐
│  ◑ Geopolitic Agent   forecasting country↔country relations · next month        ↺ reset    │
├───────────────────────────────────────────────┬────────────────────────────────────────────┤
│  REASONING SUBGRAPH            GNNExplainer ·  │  CHAT                                       │
│                                       2026-07  │  ┌────────────────────────────────────────┐ │
│   ┌───────────────────────────┐                │  │ you: Within the EU, which pair is most │ │
│   │ Germany (DEU)             │   ●ITA         │  │      likely material cooperation?      │ │
│   │ why MATERIAL_COOPERATION  │     \          │  │ ▸ get_latest_time_step → 2026-07       │ │
│   │ military_expenditure +.21 │  ●DEU ═══●FRA  │  │ ▸ resolve_place("EU") → 26 members     │ │
│   │ conflict_intensity   −.14 │    │  (focus)  │  │ ▸ best_pair_in_group(EU, MAT_COOP)     │ │
│   │ trade_openness_ind   +.08 │  ●ESP          │  │ ┌────────────────────────────────────┐ │ │
│   └───────────────────────────┘                │  │ │ DEU → FRA  MATERIAL_COOPERATION    │ │ │
│                                                │  │ │ p = 0.62 · 2026-07                 │ │ │
│                                                │  │ │ ▓▓▓▓▓▓▓▓▓▓▓▓ material coop   0.62   │ │ │
│   ● material conflict  ● verbal conflict       │  │ │ ▓▓▓▓ verbal coop            0.22   │ │ │
│   ● material coop      ● verbal coop  ● SQ     │  │ └────────────────────────────────────┘ │ │
│                                                │  └────────────────────────────────────────┘ │
│                                                │  ‹ [USA↔CHN?] [EU coop pair?] [→UKR?] [w… › │ ← scrollable chips
│                                                │  ┌──────────────────────────────┐ ┌──────┐ │
│                                                │  │ Ask about a country pair…    │ │ Send │ │
│                                                │  └──────────────────────────────┘ └──────┘ │
└───────────────────────────────────────────────┴────────────────────────────────────────────┘
```

- **Header** — logo "Geopolitic Agent" + subtitle "forecasting country↔country relations · next
  month"; a small "local model / ↺ reset" affordance on the right. **No chips here** (moved).
- **Left panel (~58%) "Reasoning subgraph"** — the SVG graph; top-right label
  `GNNExplainer · {forecast_period}`; bottom legend of the 5 classes; an IG popup that opens over
  `ig_clickable` nodes.
- **Right panel (~42%) "Chat"** — a scrollable message list (user bubble → streamed tool steps →
  answer card + probability bars), then the **horizontally scrollable example-chip strip**, then
  the input + Send.

---

## §2 Data contract (consumed — mirrors 04 §6, no drift)

The frontend speaks one protocol: the agent's SSE stream. It never re-implements ML logic and
never reorders classes.

**Request.** `POST {VITE_AGENT_URL or proxy}/agent/chat`, body `{ message, thread_id? }`, response
is an **SSE** stream. In dev, `VITE_AGENT_URL` is empty and Vite proxies `/agent` →
`http://localhost:8100` (the agent has no CORS; §7).

**Events.**

| Event | Payload | UI effect |
|---|---|---|
| `token` | `{ text }` | append to the streaming answer (right) |
| `tool_call` | `{ name, args }` | add a "▸ name(args)" reasoning step |
| `tool_result` | `{ name, summary }` | mark that step done with its summary |
| `final` | the viz payload below | render the left panel + probability chart |
| `error` | `{ message }` | inline error bubble |

**`final` viz payload** (exact shape from 04 §6):

```jsonc
{
  "answer": "Most likely DEU → FRA: MATERIAL_COOPERATION (p=0.62) in 2026-07.",
  "time_step": 197, "input_period": "2026-06", "forecast_period": "2026-07",
  "confidence_note": "model confidence (temperature-scaled, ~uncalibrated on this checkpoint)",
  "focus_pairs": [
    { "src": "DEU", "tgt": "FRA", "predicted_class": "MATERIAL_COOPERATION", "confidence": 0.62,
      "probabilities": { "MATERIAL_CONFLICT": 0.03, "VERBAL_CONFLICT": 0.05,
                         "MATERIAL_COOPERATION": 0.62, "VERBAL_COOPERATION": 0.22, "STATUS_QUO": 0.08 } }
  ],
  "intervention": null,                      // Type 6 only: { src, tgt, class, symmetric }
  "subgraph": {
    "nodes": [ { "id": "DEU", "name": "Germany", "type": "Country", "importance": 1.0, "ig_clickable": true }, … ],
    "edges": [ { "src": "USA", "tgt": "DEU", "dominant_class": "MATERIAL_COOPERATION", "importance": 0.41 }, … ]
  },
  "feature_attributions": {                  // ONLY the focus pair's u & v (04 §3 A1)
    "DEU": [ { "feature": "military_expenditure_log", "attribution": 0.21 }, … ],
    "FRA": [ { "feature": "vdem_polyarchy_score", "attribution": 0.09 }, … ]
  }
}
```

`type` mirrors these as TypeScript interfaces in `src/types.ts`. Class names come from
`src/lib/classColors.ts` (mirror of `CLASS_NAMES`).

---

## §3 Component tree & state

```
App
├─ Header
├─ SubgraphPanel
│   ├─ SubgraphCanvas   (SVG + d3-force)
│   ├─ Legend           (5 classes)
│   └─ IgPopup          (feature_attributions for the selected ig_clickable node)
└─ ChatPanel
    ├─ MessageList
    │   ├─ UserBubble
    │   ├─ ToolStep            (▸ name(args) → summary)
    │   └─ AnswerCard → ProbabilityBars   (+ baseline/counterfactual when intervention set)
    ├─ ExampleChips            (horizontally scrollable, above the input)
    └─ ChatInput               (text + Send)
```

**State — Zustand store (`src/store.ts`).** A single store with a reducer that applies SSE events
in order:

| Field | Meaning |
|---|---|
| `messages[]` | chat turns: user text, the in-flight assistant turn's `steps[]` (from tool events), answer text |
| `streaming` | true while a stream is open (disables the input) |
| `currentViz` | the latest `final` payload (drives the left panel) |
| `selectedNode` | the ISO-3 whose IG popup is open (or null) |
| `threadId` | conversation id echoed back to the agent for multi-turn |
| `error` | last `error` event message (or null) |

**SSE client (`src/sse.ts`) + `useAgentChat()` hook.** `fetch` POSTs the message, reads
`response.body` as a `ReadableStream`, decodes chunks, splits on the SSE frame delimiter
(`\n\n`), **buffers partial frames across chunk boundaries**, parses each `event:`/`data:` pair
into a typed event, and dispatches it to the store. Skips heartbeat/blank lines; never throws on a
malformed frame (logs + skips).

**Wire format.** The agent emits each frame as `event:<name>\n data:<json>\n\n` where the JSON is
single-line (`json.dumps` escapes any newline in the answer text). `sse.ts` nonetheless follows
the standard SSE rule and **concatenates multiple `data:` lines within a frame** before parsing,
so a future multi-line `data:` payload cannot corrupt the stream.

---

## §4 Subgraph rendering (SVG + d3-force)

`SubgraphCanvas` runs a `d3-force` simulation (`forceLink` + `forceManyBody` + `forceCenter`)
over `currentViz.subgraph` and draws plain SVG. **Two distinct edge layers** are drawn so the
answer is always visible:

- **Nodes** — `<circle>` + `<text>` label (`id`; full `name` on hover). Radius and fill-opacity
  scale with `importance`. `ig_clickable` focus nodes are filled with the purple `--accent` and
  get a pointer cursor; other nodes are muted (structural context only).
- **(a) Explainer input edges** — from `subgraph.edges`: `<line>` whose stroke is
  `classColor(dominant_class)` and whose opacity scales with `importance`. These are the
  GNNExplainer top-k **input** edges at month T (what the model attended to), *not* the answer.
- **(b) The prediction (focus) edge** — **one per `focus_pairs[i]`**, drawn between the focus
  nodes in the `--focus` purple, thicker, labeled `predicted_class · p` (the screenshot's
  highlighted DEU═══FRA). **Synthesize it even when it is absent from `subgraph.edges`** — the
  predicted relationship need not have been an input edge. Focus nodes are always present in
  `subgraph.nodes` (pinned `importance = 1.0`), so the edge always has endpoints. This is the edge
  that represents the agent's answer.
- **Click → IG popup** — clicking an `ig_clickable` node sets `selectedNode`; `IgPopup` renders
  that node's `feature_attributions` as **signed bars** (sorted by `|attribution|`, top-N; positive
  = pushed toward the predicted class, negative = away). Clicking a non-`ig_clickable` node opens
  **nothing** — only `u`/`v` have feature attributions (04 §3 A1); other nodes are structural.
- Top-right label `GNNExplainer · {forecast_period}`.

---

## §5 Chat, streaming & the scrollable chips

- **MessageList** — user bubbles; the in-flight assistant turn renders a stack of `ToolStep` rows
  (`▸ best_pair_in_group(EU, MATERIAL_COOPERATION) → DEU → FRA`) built from `tool_call` then
  completed by `tool_result`; then an `AnswerCard` with the streamed answer text and the
  `confidence_note` shown subtly; then `ProbabilityBars` for the focus pair's five probabilities
  (sorted descending, each a class-colored bar labeled `class + value`). When `intervention` is
  present (a Type-6 what-if), the card shows **baseline vs counterfactual** side by side.
- **ExampleChips** — a strip **directly above** the input with `overflow-x: auto`, scroll-snap,
  and left/right chevron buttons. The four chips embody the 04 question taxonomy:
  1. `What's most likely between the USA and China next month?`
  2. `Within the EU, which pair is most likely material cooperation?`
  3. `Who is most likely to enter material conflict with Ukraine?`
  4. `If the USA & China sign a deal, what about China and India?` *(Type 6 what-if)*

  Clicking a chip fills the input (and may auto-send); the active chip uses `--accent`.
- **ChatInput** — text field + Send button; Enter submits; both disabled while `streaming`.

---

## §6 Theme (Electric purple) — design tokens

Defined as CSS variables in `src/theme.css`. **Chrome** is purple; the **class palette** is a
separate categorical scale.

```css
:root {
  --bg:#150d22; --surface:#211633; --border:#3a2a55;
  --accent:#a855f7; --focus:#c084fc;            /* buttons, active chip, focus node/edge, links */
  --text:#f3eaff; --text-muted:#b3a0d0;
}
```

`--accent` drives the Send button, the active chip, the focus node/edge highlight, links, the
scrollbar thumb, and the IG-popup header.

**Class palette** (`src/lib/classColors.ts`, in `CLASS_NAMES` order — distinct from the accent so
the data reads clearly on purple chrome, matching the screenshot legend semantics):

| Class | Color |
|---|---|
| `MATERIAL_CONFLICT` | `#ef4444` (red) |
| `VERBAL_CONFLICT` | `#f59e0b` (amber) |
| `MATERIAL_COOPERATION` | `#22c55e` (green) |
| `VERBAL_COOPERATION` | `#38bdf8` (sky) |
| `STATUS_QUO` | `#6b7280` (gray) |

---

## §7 Project structure, build & run

```
services/frontend/
├── index.html
├── vite.config.ts          # @vitejs/plugin-react + dev proxy: /agent → http://localhost:8100
├── package.json
├── tsconfig.json
├── vitest.config.ts        # jsdom env, setupFiles (jest-dom)
└── src/
    ├── main.tsx  App.tsx  theme.css  store.ts  sse.ts  types.ts
    ├── lib/classColors.ts
    ├── components/{Header,SubgraphPanel,SubgraphCanvas,Legend,IgPopup,
    │              ChatPanel,MessageList,ToolStep,AnswerCard,ProbabilityBars,
    │              ExampleChips,ChatInput}.tsx
    └── test/               # Vitest + RTL specs (§8) + fixtures (the 04 §6 example payload)
```

- **deps:** `react`, `react-dom`, `zustand`, `d3-force`.
- **dev deps:** `vite`, `@vitejs/plugin-react`, `typescript`, `vitest`, `@testing-library/react`,
  `@testing-library/user-event`, `@testing-library/jest-dom`, `jsdom`.
- **env:** `VITE_AGENT_URL` (default empty → the Vite proxy handles `/agent`).
- **Vite dev proxy** (in `vite.config.ts`):
  ```ts
  server: { proxy: { "/agent": { target: "http://localhost:8100", changeOrigin: true } } }
  ```
- **Scripts:** `npm install` · `npm run dev` (Vite on `:5173`, proxies to the agent on `:8100`) ·
  `npm test` (Vitest) · `npm run build` (static bundle).

---

## §8 Tests (Vitest + React Testing Library, mocked SSE)

All component/unit, jsdom, no browser and no live agent — a mocked `ReadableStream` replays SSE
frames (fixtures = the 04 §6 example payload).

1. **`sse.ts` parsing** — a chunked stream of SSE frames yields typed events in order;
   a frame **split across two chunks** is correctly reassembled; blank/heartbeat lines are ignored.
2. **store reducer** — applying events in order: `tool_call` adds a step, `tool_result` completes
   it, `token` appends to the answer, `final` sets `currentViz`, `error` sets `error`.
3. **ExampleChips** — renders the 4 chips inside a horizontally scrollable container
   (`overflow-x`), positioned **above** the input; clicking a chip fills the input value.
4. **ChatInput** — Enter submits; the Send button calls the sender; both are disabled while
   `streaming` is true.
5. **ProbabilityBars** — renders 5 class-colored bars sorted descending; bar widths ∝ probability;
   each labeled with its class name + value.
6. **SubgraphCanvas** — one SVG node per `subgraph.node` and one input-edge line per
   `subgraph.edge`; node/edge opacity reflects `importance`; input-edge stroke =
   `classColor(dominant_class)`. The **prediction (focus) edge** between the focus nodes is drawn
   in `--focus` and labeled `predicted_class · p`.
7. **IG-popup gating** — clicking an `ig_clickable` node opens the popup with that node's
   `feature_attributions` (signed, sorted by `|attribution|`); clicking a non-`ig_clickable` node
   opens **nothing** (04 §3 A1).
8. **classColors / theme** — `classColors` has exactly 5 entries in `CLASS_NAMES` order; the theme
   exposes the purple tokens and the chrome `--accent` is `#a855f7` (no green chrome).
9. **App end-to-end (component-level, stubbed `fetch`)** — feed the 04 §6 example SSE stream:
   submit a question → tool steps stream in → the answer card + probability chart render → the
   subgraph draws → clicking the focus node opens the IG popup.
10. **Counterfactual** — a `final` whose `intervention` is set renders the baseline-vs-counterfactual
    view and highlights the intervened edge in the subgraph.
11. **Resilience** — an `error` event renders an inline error bubble; a malformed SSE frame is
    skipped without crashing the app.
12. **Prediction edge always shown** — given a `final` whose `subgraph.edges` does **not** contain
    the focus pair (its predicted relationship was not an input edge), the canvas still draws the
    focus/prediction edge between the focus nodes in `--focus` (Edit B / §4 layer b).

**Tests are read-only — they cannot delete data or drop a database.** The whole suite runs under
**jsdom with `fetch`/SSE mocked**: it opens **no socket, no Neo4j, no agent process** — so it
performs no database I/O of any kind and **cannot write, delete, or drop** the research data. (The
agent's own tests are likewise read-only — `COMMANDS.md` §11c — and the **only** DB-destructive
tests in the repository are the Go integration tests, gated behind `GEOPOLITIC_ALLOW_DB_WIPE=1`
and skipped by default; they are unrelated to the frontend.)

---

## §9 Verification / acceptance

- **Unit/component:** `npm test` → all §8 specs pass (jsdom, no browser, no live agent, **no
  database I/O** — `fetch`/SSE are mocked, so the suite cannot delete data or drop a DB).
- **Manual end-to-end:** start the agent (`python -m agent` + Ollama, `COMMANDS.md` §11e) →
  `npm run dev` → open `:5173` → click a chip → watch the tool steps stream → see the answer +
  probability chart + the reasoning subgraph → click the **DEU** node → the IG popup appears.
  Confirm the **purple** theme and that the **chips scroll horizontally above the input**.
- **Alignment & safety:** the SSE events and `final` fields match 04 §6; the class order matches
  `CLASS_NAMES`; the frontend calls **only** `/agent/chat` (never the Go API or Python ML), and
  performs **no** database access — it cannot affect the research data.

---

## §10 Cross-plan alignment

**Alignment verification (no logic bug found).** This document was checked field-by-field against
the shipped code and plans 01–04:
- **SSE events** (`token / tool_call / tool_result / final / error`) match the emitter in
  `services/agent/agent/server.py`.
- **`final` viz fields** match 04 §6 **and** the actual `graph.assemble_viz` output —
  `answer, time_step, input_period, forecast_period, confidence_note, focus_pairs[], intervention,
  subgraph{nodes,edges}, feature_attributions`; node fields `{id,name,type,importance,ig_clickable}`
  and edge fields `{src,tgt,dominant_class,importance}` match the explainer's output.
- **Class order** mirrors `CLASS_NAMES`; **time convention** (`forecast_period = iso_period(T+1)`)
  matches 03/04; **IG scope** (only the focus pair `u`/`v` are clickable) matches 04 §3 A1.
- **One gap fixed** (§4 / Edit B): the answer's predicted edge is *not* guaranteed to be in
  `subgraph.edges` (those are GNNExplainer **input** edges) — so the canvas draws the prediction
  edge as a separate, always-present focus layer.
- **One reconciliation** (under-the-hood §): 04's text mentions a read-only Neo4j client, but the
  shipped agent is **parquet + pycountry** (DB-free at serve time); 05 reflects the implementation.
- **Conclusion: no contradiction across 01–05** — one model, one class order, one time convention.

- **04 (agent) — exact reuse.** The frontend consumes the SSE/viz contract verbatim; the four
  chips embody the question taxonomy (incl. the Type-6 what-if). The only dependency on 04 is
  serving: local dev uses the **Vite proxy**; a deployed static build adds a small **CORS
  middleware** to `services/agent/agent/server.py` (a one-line `app.add_middleware(CORSMiddleware,
  …)`, no change to existing behavior).
- **01 (architecture) — this realizes the UI.** 01's "graph canvas + prediction panel +
  explanation panel" is realized by this chat-driven demo (explanation panel = the GNNExplainer
  subgraph + IG popup; chat replaces the manual pair-picker). The **timeline scrubber** and
  **simulation overlay** are **superseded** by chat + the what-if tool (04 §8) and are not built.
- **02/03 (ingestion, model) — untouched.** The UI only renders the agent's payload, which is
  already derived from the trained model over the aggregated data — no direct reads of either DB.
- **No contradiction across 01–05:** one model, one class order, one time convention; Go =
  ingestion, Python ML = prediction, Agent = orchestration, **Frontend = display**.
