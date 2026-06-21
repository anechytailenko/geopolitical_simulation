import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { ChatPanel } from "../components/ChatPanel";
import { EXAMPLE_QUESTIONS } from "../components/ExampleChips";
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

describe("ExampleChips (plans/05 §8.3)", () => {
  it("renders the 4 chips in a horizontally scrollable strip above the input", () => {
    render(<ChatPanel />);
    for (let i = 0; i < 4; i++) expect(screen.getByTestId(`chip-${i}`)).toBeInTheDocument();

    const scroll = screen.getByTestId("chips-scroll");
    expect(scroll.style.overflowX).toBe("auto");

    // chips appear before the input in document order
    const chips = screen.getByTestId("example-chips");
    const input = screen.getByTestId("chat-input");
    expect(chips.compareDocumentPosition(input) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("clicking a chip fills the input value", () => {
    render(<ChatPanel />);
    fireEvent.click(screen.getByTestId("chip-0"));
    expect((screen.getByTestId("chat-input") as HTMLInputElement).value).toBe(EXAMPLE_QUESTIONS[0]);
  });
});
