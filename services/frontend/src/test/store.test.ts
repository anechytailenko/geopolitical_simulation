import { beforeEach, describe, expect, it } from "vitest";

import { useStore } from "../store";
import { VIZ_USA_CHN } from "./fixtures";

function resetStore() {
  useStore.setState({
    messages: [],
    streaming: false,
    currentViz: null,
    selectedNode: null,
    error: null,
    threadId: null,
  });
}

beforeEach(resetStore);

describe("store reducer (plans/05 §8.2)", () => {
  it("applies events in order", () => {
    useStore.setState({
      messages: [
        { role: "user", text: "q" },
        { role: "assistant", steps: [], answer: "" },
      ],
    });
    const { applyEvent } = useStore.getState();
    applyEvent({ type: "tool_call", name: "predict_pair", args: { source: "USA" } });
    applyEvent({ type: "tool_result", name: "predict_pair", summary: "done" });
    applyEvent({ type: "token", text: "Hello" });
    applyEvent({ type: "token", text: " world" });
    applyEvent({ type: "final", viz: VIZ_USA_CHN });

    const s = useStore.getState();
    const a = s.messages[1] as Extract<(typeof s.messages)[number], { role: "assistant" }>;
    expect(a.steps).toHaveLength(1);
    expect(a.steps[0]).toMatchObject({ name: "predict_pair", summary: "done", done: true });
    expect(a.answer).toBe("Hello world");
    expect(s.currentViz).toEqual(VIZ_USA_CHN);
  });

  it("error event sets the error field", () => {
    useStore.getState().applyEvent({ type: "error", message: "boom" });
    expect(useStore.getState().error).toBe("boom");
  });
});
