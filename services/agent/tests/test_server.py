"""SSE server (plans/04 §6): /agent/chat streams token/tool_call/tool_result/final, with the
deterministic viz in the final event. Uses a scripted model (no live LLM)."""

from fastapi.testclient import TestClient

import agent.runtime as runtime_mod
from agent import server as S
from agent.llm import ScriptedChatModel


def test_health(rt):
    runtime_mod._RUNTIME = rt
    client = TestClient(S.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_streams_events_and_final_viz(rt, monkeypatch):
    runtime_mod._RUNTIME = rt
    monkeypatch.setattr(S, "get_llm", lambda: ScriptedChatModel(turns=[
        {"tool": "predict_pair", "args": {"source": "USA", "target": "CHN"}},
        "USA->China is most likely MATERIAL_CONFLICT next month.",
    ]))
    client = TestClient(S.app)
    with client.stream("POST", "/agent/chat", json={"message": "USA and China next month?"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    assert "event: tool_call" in body
    assert "event: tool_result" in body
    assert "event: token" in body
    assert "event: final" in body
    # the final payload carries the deterministic viz
    assert "subgraph" in body and "feature_attributions" in body
