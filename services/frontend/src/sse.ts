// SSE client for POST /agent/chat (plans/05 §2-§3). Parses the agent's event stream into typed
// AgentEvents. Robust to \r\n vs \n line endings, frames split across chunk boundaries, multi-line
// `data:` payloads, comment/heartbeat lines, and malformed frames (skipped, never thrown).

import type { AgentEvent, Viz } from "./types";

/** Parse one SSE frame ("event: x\ndata: {...}") into a typed event, or null if not parseable. */
export function parseFrame(frame: string): AgentEvent | null {
  let event = "";
  const dataLines: string[] = [];
  for (const raw of frame.split("\n")) {
    const line = raw.replace(/\r$/, "");
    if (!line || line.startsWith(":")) continue; // blank or comment/heartbeat
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, "")); // strip one optional leading space
    }
  }
  if (!event) return null;
  const data = dataLines.join("\n");
  let payload: any = {};
  try {
    payload = data ? JSON.parse(data) : {};
  } catch {
    return null; // malformed JSON -> skip the frame
  }
  switch (event) {
    case "token":
      return { type: "token", text: String(payload.text ?? "") };
    case "tool_call":
      return { type: "tool_call", name: String(payload.name ?? ""), args: payload.args ?? {} };
    case "tool_result":
      return { type: "tool_result", name: String(payload.name ?? ""), summary: String(payload.summary ?? "") };
    case "final":
      return { type: "final", viz: payload as Viz };
    case "error":
      return { type: "error", message: String(payload.message ?? "") };
    default:
      return null;
  }
}

export interface StreamOptions {
  baseUrl?: string;
  threadId?: string | null;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
}

/** POST the message and invoke `onEvent` for each parsed SSE event until the stream ends. */
export async function streamChat(
  message: string,
  onEvent: (e: AgentEvent) => void,
  opts: StreamOptions = {},
): Promise<void> {
  const f = opts.fetchImpl ?? fetch;
  let res: Response;
  try {
    res = await f((opts.baseUrl ?? "") + "/agent/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message, thread_id: opts.threadId ?? undefined }),
      signal: opts.signal,
    });
  } catch {
    // connection refused / network error — the agent isn't reachable
    throw new Error(
      "Couldn't reach the agent. Is it running? Start it with `cd services/agent && " +
        ".venv/bin/python -m agent` (and have Ollama up). See COMMANDS.md §11e/§12.",
    );
  }
  if (!res.ok) {
    throw new Error(`Agent returned HTTP ${res.status}. Check the agent logs (is Ollama up?).`);
  }
  if (!res.body) {
    throw new Error("Agent response had no stream body.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true }).replace(/\r/g, ""); // normalize CRLF -> LF
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseFrame(frame);
      if (ev) onEvent(ev);
    }
  }
  // flush any trailing frame without the final blank line
  const tail = parseFrame(buf);
  if (tail) onEvent(tail);
}
