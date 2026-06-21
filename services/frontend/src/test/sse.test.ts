import { describe, expect, it } from "vitest";

import { parseFrame, streamChat } from "../sse";
import type { AgentEvent } from "../types";
import { mockSseFetch, sseFrame } from "./fixtures";

describe("sse parsing (plans/05 §8.1)", () => {
  it("parses event/data frames into typed events", () => {
    expect(parseFrame(sseFrame("token", { text: "hi" }))).toEqual({ type: "token", text: "hi" });
    expect(parseFrame(sseFrame("tool_call", { name: "predict_pair", args: { source: "USA" } }))).toEqual({
      type: "tool_call",
      name: "predict_pair",
      args: { source: "USA" },
    });
    expect(parseFrame(sseFrame("error", { message: "x" }))).toEqual({ type: "error", message: "x" });
  });

  it("reassembles a frame split across two chunks", async () => {
    const frame = sseFrame("token", { text: "hello world" });
    const mid = Math.floor(frame.length / 2);
    const events: AgentEvent[] = [];
    await streamChat("q", (e) => events.push(e), {
      fetchImpl: mockSseFetch([frame.slice(0, mid), frame.slice(mid)]),
    });
    expect(events).toEqual([{ type: "token", text: "hello world" }]);
  });

  it("ignores comments/heartbeats and is robust to CRLF line endings", async () => {
    const events: AgentEvent[] = [];
    await streamChat("q", (e) => events.push(e), {
      fetchImpl: mockSseFetch([": ping\n\n", 'event: token\r\ndata: {"text":"x"}\r\n\r\n']),
    });
    expect(events).toEqual([{ type: "token", text: "x" }]);
  });

  it("returns null for a malformed (bad JSON) frame and never throws", () => {
    expect(parseFrame("event: token\ndata: {not json")).toBeNull();
    expect(() => parseFrame("garbage with no fields")).not.toThrow();
  });

  it("throws an actionable error when the agent connection is refused", async () => {
    const reject = (async () => {
      throw new TypeError("fetch failed");
    }) as unknown as typeof fetch;
    await expect(streamChat("q", () => {}, { fetchImpl: reject })).rejects.toThrow(/reach the agent/i);
  });

  it("throws on a non-OK agent response", async () => {
    const notOk = (async () =>
      ({ ok: false, status: 502, body: null }) as unknown as Response) as typeof fetch;
    await expect(streamChat("q", () => {}, { fetchImpl: notOk })).rejects.toThrow(/HTTP 502/);
  });
});
