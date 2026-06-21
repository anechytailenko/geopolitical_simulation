"""Pydantic payloads for the SSE server (plans/04 §6). The tools already return plain dicts;
these models document and validate the HTTP boundary."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class StreamEvent(BaseModel):
    event: str                 # token | tool_call | tool_result | final | error
    data: dict[str, Any]
