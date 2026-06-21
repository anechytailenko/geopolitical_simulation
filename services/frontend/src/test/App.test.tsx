import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import { useStore } from "../store";
import { mockSseFetch, scriptedPredictPair } from "./fixtures";

beforeEach(() =>
  useStore.setState({
    messages: [],
    streaming: false,
    currentViz: null,
    selectedNode: null,
    error: null,
    threadId: null,
  }),
);
afterEach(() => vi.unstubAllGlobals());

describe("App end-to-end with mocked SSE (plans/05 §8.9)", () => {
  it("streams tool steps then renders answer + chart + subgraph; node click opens IG popup", async () => {
    vi.stubGlobal("fetch", mockSseFetch(scriptedPredictPair()));
    render(<App />);

    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "USA and China next month?" },
    });
    fireEvent.click(screen.getByTestId("send-btn"));

    // a tool step streamed in
    expect(await screen.findByText(/predict_pair/)).toBeInTheDocument();
    // the answer card + probability chart rendered
    expect(await screen.findByTestId("answer-card")).toBeInTheDocument();
    expect(screen.getByTestId("probability-bars")).toBeInTheDocument();
    // the subgraph drew, including the focus edge
    expect(screen.getByTestId("subgraph")).toBeInTheDocument();
    expect(screen.getByTestId("focus-edge-USA-CHN")).toBeInTheDocument();

    // clicking the focus node opens the IG popup
    fireEvent.click(screen.getByTestId("node-USA"));
    expect(await screen.findByTestId("ig-popup")).toBeInTheDocument();
  });

  it("shows a friendly error bubble when the agent is unreachable (ECONNREFUSED)", async () => {
    vi.stubGlobal("fetch", (async () => {
      throw new TypeError("fetch failed"); // mimics the dev-proxy ECONNREFUSED
    }) as unknown as typeof fetch);
    render(<App />);

    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "conflict between the USA and China?" },
    });
    fireEvent.click(screen.getByTestId("send-btn"));

    const bubble = await screen.findByTestId("error-bubble");
    expect(bubble.textContent).toMatch(/reach the agent/i);
  });
});
