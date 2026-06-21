# geopolitic-agent

A **ReAct agent + MCP tools** over the trained geopolitical GNN (see
[`plans/04-react-agent.md`](../../plans/04-react-agent.md)). It answers natural-language
questions about next-month country↔country relationships by calling **task-shaped tools** that
run `ml.infer.Predictor` in-process — one tool call answers a whole question, so a small local
LLM only has to pick a tool and fill a few ISO-3 / class / month args.

**Read-only & database-free.** The agent loads the trained weights from `artifacts/` and the
dataset from `services/ml/dataset_parquet/` (the same export the model trained on), and resolves
places with `pycountry` + the parquet `MEMBER_OF` edges. It never connects to Neo4j and never
writes anything — it cannot alter or delete ingested data.

## Layout

```
agent/
  config.py        AgentConfig — LLM provider/model + serving knobs
  groups.py        IGO name/alias -> Wikidata QID map (EU=Q458, NATO=Q7184, …)
  places.py        PlaceResolver — name/ISO-2/ISO-3 -> ISO-3, group -> members valid at T
  runtime.py       Runtime — loads Predictor once + boot self-check (plans/04 §11)
  explainer.py     enrich Predictor.explain -> renderable subgraph + named IG (plans/04 §3)
  counterfactual.py single-step what-if: edit one month-T edge, re-forward (plans/04 §3 Type 6)
  tools_core.py    the 8 task-shaped tools (validation + Predictor calls) — source of truth
  mcp_server.py    FastMCP server exposing tools_core over MCP (python -m agent.mcp_server)
  llm.py           Ollama/Anthropic/OpenAI switch + ScriptedChatModel (tests/offline)
  graph.py         LangGraph create_react_agent + deterministic viz node
  server.py        FastAPI POST /agent/chat -> SSE (token/tool_call/tool_result/final)
tests/             34 tests: real model + scripted LLM + in-memory MCP session (no Ollama/Neo4j)
```

## Question types (plans/04 §1)

| Type | Tool |
|---|---|
| Direct "A vs B next month" | `predict_pair` |
| Group "best `<class>` pair in `<group>`" | `best_pair_in_group` |
| Target-fixed "who is most likely `<class>` with `<country>`" | `most_likely_counterpart` |
| Compare "conflict or cooperation between A and B" | `compare_pair` |
| Why | `explain_pair` |
| What-if "if A & B `<class>`, what about B and C" (single step) | `predict_counterfactual` |

## Run

See [`COMMANDS.md` §11](../../COMMANDS.md) for setup, tests, the MCP server, and the SSE chat
server with expected output. Quick start:

```bash
export GEO_ARTIFACTS_DIR="$PWD/../../artifacts" GEO_DATA_DIR="$PWD/../ml/dataset_parquet"
.venv/bin/python -m pytest -q          # 34 passed (no LLM, no Neo4j)
.venv/bin/python -m agent.mcp_server   # MCP tools over stdio
.venv/bin/python -m agent              # chat SSE server on :8100 (needs Ollama for the default LLM)
```
