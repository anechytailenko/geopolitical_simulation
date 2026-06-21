import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import App from "../App";
import { parseFrame } from "../sse";
import { useStore } from "../store";

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

describe("resilience (plans/05 §8.11)", () => {
  it("renders an inline error bubble for an error event", () => {
    useStore.setState({ error: "something went wrong" });
    render(<App />);
    expect(screen.getByTestId("error-bubble")).toHaveTextContent("something went wrong");
  });

  it("skips a malformed SSE frame without throwing", () => {
    expect(parseFrame("event: final\ndata: {not json")).toBeNull();
    expect(() => parseFrame("garbage with no fields")).not.toThrow();
  });
});
