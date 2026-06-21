// Zustand store (plans/05 §3): applies SSE events in order and orchestrates a chat turn.
import { create } from "zustand";

import { streamChat } from "./sse";
import type { AgentEvent, Viz } from "./types";

export interface ToolStep {
  name: string;
  args?: Record<string, unknown>;
  summary?: string;
  done: boolean;
}

export type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; steps: ToolStep[]; answer: string };

export interface StoreState {
  messages: Message[];
  streaming: boolean;
  currentViz: Viz | null;
  selectedNode: string | null;
  error: string | null;
  threadId: string | null;
  // actions
  applyEvent: (e: AgentEvent) => void;
  send: (text: string) => Promise<void>;
  selectNode: (id: string | null) => void;
  reset: () => void;
}

const AGENT_BASE: string =
  ((import.meta as any).env?.VITE_AGENT_URL as string | undefined) ?? "";

function updateLastAssistant(
  messages: Message[],
  fn: (m: Extract<Message, { role: "assistant" }>) => void,
): Message[] {
  const out = messages.slice();
  for (let i = out.length - 1; i >= 0; i--) {
    const m = out[i];
    if (m.role === "assistant") {
      const copy = { role: "assistant" as const, steps: m.steps.slice(), answer: m.answer };
      fn(copy);
      out[i] = copy;
      return out;
    }
  }
  // no assistant message yet -> create one
  const fresh = { role: "assistant" as const, steps: [], answer: "" };
  fn(fresh);
  out.push(fresh);
  return out;
}

export const useStore = create<StoreState>((set, get) => ({
  messages: [],
  streaming: false,
  currentViz: null,
  selectedNode: null,
  error: null,
  threadId: null,

  applyEvent: (e: AgentEvent) =>
    set((s) => {
      switch (e.type) {
        case "tool_call":
          return {
            messages: updateLastAssistant(s.messages, (m) =>
              m.steps.push({ name: e.name, args: e.args, done: false }),
            ),
          };
        case "tool_result":
          return {
            messages: updateLastAssistant(s.messages, (m) => {
              const step = [...m.steps].reverse().find((st) => st.name === e.name && !st.done)
                ?? [...m.steps].reverse().find((st) => !st.done);
              if (step) {
                step.summary = e.summary;
                step.done = true;
              } else {
                m.steps.push({ name: e.name, summary: e.summary, done: true });
              }
            }),
          };
        case "token":
          return {
            messages: updateLastAssistant(s.messages, (m) => {
              m.answer += e.text;
            }),
          };
        case "final":
          return {
            currentViz: e.viz,
            messages: updateLastAssistant(s.messages, (m) => {
              if (!m.answer) m.answer = e.viz.answer ?? "";
            }),
          };
        case "error":
          return { error: e.message };
        default:
          return {};
      }
    }),

  send: async (text: string) => {
    if (get().streaming || !text.trim()) return;
    set((s) => ({
      streaming: true,
      error: null,
      currentViz: null,
      selectedNode: null,
      messages: [
        ...s.messages,
        { role: "user", text },
        { role: "assistant", steps: [], answer: "" },
      ],
    }));
    try {
      await streamChat(text, (e) => get().applyEvent(e), {
        baseUrl: AGENT_BASE,
        threadId: get().threadId,
      });
    } catch (err: any) {
      get().applyEvent({ type: "error", message: err?.message ?? String(err) });
    } finally {
      set({ streaming: false });
    }
  },

  selectNode: (id) => set({ selectedNode: id }),
  reset: () =>
    set({
      messages: [],
      streaming: false,
      currentViz: null,
      selectedNode: null,
      error: null,
    }),
}));
