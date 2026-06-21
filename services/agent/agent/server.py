"""FastAPI SSE server (plans/04 §6). POST /agent/chat streams the ReAct run as Server-Sent
Events: token -> tool_call / tool_result -> final{viz}. The deterministic viz step always runs
explain_pair on the answer's focus pair, so the left panel matches the answer.

Run:  python -m agent   (or: uvicorn agent.server:app --port 8100)
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from langchain_core.messages import AIMessage, ToolMessage
from sse_starlette.sse import EventSourceResponse

from . import graph as G
from .config import AgentConfig
from .llm import build_llm
from .runtime import get_runtime
from .schemas import ChatRequest

app = FastAPI(title="geopolitic-agent", version="1.0")
_LLM = None


def get_llm():
    """LLM accessor (built once). Tests monkeypatch this to inject a scripted model."""
    global _LLM
    if _LLM is None:
        _LLM = build_llm()
    return _LLM


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


async def _event_stream(message: str):
    cfg = AgentConfig.from_env()
    rt = get_runtime()
    agent = G.build_agent(get_llm(), rt)
    collected: list = []
    try:
        async for chunk in agent.astream(
            {"messages": [("user", message)]},
            config={"recursion_limit": cfg.recursion_limit}, stream_mode="updates",
        ):
            for update in chunk.values():
                for m in (update or {}).get("messages", []):
                    collected.append(m)
                    if isinstance(m, AIMessage):
                        for tc in (m.tool_calls or []):
                            yield _sse("tool_call", {"name": tc["name"], "args": tc["args"]})
                        if m.content and not m.tool_calls:
                            yield _sse("token", {"text": m.content})
                    elif isinstance(m, ToolMessage):
                        yield _sse("tool_result", {"name": m.name, "summary": str(m.content)})

        answer = next((m.content for m in reversed(collected)
                       if isinstance(m, AIMessage) and m.content and not m.tool_calls), "")
        viz = G.assemble_viz(rt, collected, answer)
        yield _sse("final", viz or {"answer": answer})
    except Exception as e:  # surface as a recoverable error event
        yield _sse("error", {"message": str(e)})


@app.post("/agent/chat")
async def chat(req: ChatRequest):
    return EventSourceResponse(_event_stream(req.message))
