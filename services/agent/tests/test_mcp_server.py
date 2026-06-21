"""MCP server (plans/04 §3): tools are exposed and callable over the protocol, validation
errors surface as tool errors, and they load as LangChain tools via langchain-mcp-adapters.

Tests use an in-memory client<->server session (no subprocess) and drive the async calls with
asyncio.run, so no extra pytest plugin is needed.
"""

import asyncio
import json

import pytest

from agent import mcp_server as M
from mcp.shared.memory import create_connected_server_and_client_session as connect

TOOL_NAMES = {
    "get_latest_time_step", "resolve_place", "predict_pair", "best_pair_in_group",
    "most_likely_counterpart", "compare_pair", "explain_pair", "predict_counterfactual",
}


@pytest.fixture
def mcp_server(rt):
    # rt fixture already set agent.runtime._RUNTIME = rt, which mcp_server.get_runtime() reuses.
    return M.mcp


def _run(coro):
    return asyncio.run(coro)


async def _session_call(name, args):
    async with connect(M.mcp._mcp_server) as s:
        await s.initialize()
        return await s.call_tool(name, args)


def test_list_tools(mcp_server):
    async def go():
        async with connect(M.mcp._mcp_server) as s:
            await s.initialize()
            return (await s.list_tools()).tools
    tools = _run(go())
    assert {t.name for t in tools} == TOOL_NAMES


def test_call_predict_pair(mcp_server):
    r = _run(_session_call("predict_pair", {"source": "USA", "target": "CHN"}))
    assert not r.isError
    data = json.loads(r.content[0].text)
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-5
    assert data["focus_pair"]["src"] == "USA"


def test_validation_error_surfaces(mcp_server):
    r = _run(_session_call("predict_pair", {"source": "USA", "target": "CHN", "time_step": 9999}))
    assert r.isError
    assert "out of range" in r.content[0].text


def test_bad_class_error(mcp_server):
    r = _run(_session_call("best_pair_in_group", {"group": "EU", "relationship_class": "WAR"}))
    assert r.isError
    assert "MATERIAL_CONFLICT" in r.content[0].text


def test_loads_as_langchain_tools(mcp_server):
    from langchain_mcp_adapters.tools import load_mcp_tools

    async def go():
        async with connect(M.mcp._mcp_server) as s:
            await s.initialize()
            tools = await load_mcp_tools(s)
            pp = next(t for t in tools if t.name == "predict_pair")
            out = await pp.ainvoke({"source": "USA", "target": "CHN"})
            return {t.name for t in tools}, out
    names, out = _run(go())
    assert names == TOOL_NAMES
    assert out is not None
