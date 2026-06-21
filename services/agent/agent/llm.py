"""LLM provider switch + the ReAct system prompt (plans/04 §4).

Default is local Ollama (`qwen2.5:3b-instruct`, free, no key); `LLM_PROVIDER` flips to Anthropic
or OpenAI with one env change. ``ScriptedChatModel`` is a deterministic, dependency-free chat
model used by the tests (and any offline run) so the ReAct loop can be exercised without a live
LLM.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from .config import AgentConfig

SYSTEM_PROMPT = """You are a geopolitical forecasting assistant. The trained model predicts the \
relationship class between two countries for NEXT month, one of: MATERIAL_CONFLICT, \
VERBAL_CONFLICT, MATERIAL_COOPERATION, VERBAL_COOPERATION, STATUS_QUO.

Answer with exactly ONE answer tool per question:
- a direct "what is likely between A and B" -> predict_pair
- "within <group>, which pair is most likely <class>" -> best_pair_in_group
- "who is most likely <class> with <country>" -> most_likely_counterpart
- "conflict or cooperation between A and B" -> compare_pair
- a what-if "if A and B <class>, what about B and C" -> predict_counterfactual
Use explain_pair only when the user asks WHY.

Tools accept country names or ISO-2/ISO-3 codes directly; only call resolve_place when a name is \
ambiguous or you need a group's members. relationship_class must be one of the five class names. \
Name the forecast month using the tool's forecast_period field. Be concise; do not invent \
numbers — read them from the tool result."""


class ScriptedChatModel(BaseChatModel):
    """A fake tool-calling chat model driven by a fixed script (for tests / offline).

    ``turns`` is a list; each item is either a dict ``{"tool": name, "args": {...}}`` (emit a
    tool call) or a string (emit a final answer). One item is consumed per model step.
    """

    turns: list[Any] = []
    _idx: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001 - matches BaseChatModel API
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        turn = self.turns[self._idx] if self._idx < len(self.turns) else "Done."
        self._idx += 1
        if isinstance(turn, dict) and "tool" in turn:
            msg = AIMessage(content="", tool_calls=[{
                "name": turn["tool"], "args": turn.get("args", {}),
                "id": f"call_{self._idx}", "type": "tool_call",
            }])
        else:
            msg = AIMessage(content=str(turn))
        return ChatResult(generations=[ChatGeneration(message=msg)])


def build_llm(cfg: AgentConfig | None = None) -> BaseChatModel:
    cfg = cfg or AgentConfig.from_env()
    provider = cfg.llm_provider.lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg.llm_model, base_url=cfg.ollama_base_url,
                          temperature=cfg.llm_temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # optional extra
        return ChatAnthropic(model=cfg.llm_model, temperature=cfg.llm_temperature)
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # optional extra
        return ChatOpenAI(model=cfg.llm_model, temperature=cfg.llm_temperature)
    raise ValueError(f"unknown LLM_PROVIDER {cfg.llm_provider!r} (ollama|anthropic|openai)")
