import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ChatInput } from "../components/ChatInput";

describe("ChatInput (plans/05 §8.4)", () => {
  it("Enter submits when enabled", () => {
    const onSend = vi.fn();
    render(<ChatInput value="hi" onChange={() => {}} onSend={onSend} disabled={false} />);
    fireEvent.keyDown(screen.getByTestId("chat-input"), { key: "Enter" });
    expect(onSend).toHaveBeenCalledOnce();
  });

  it("the Send button calls the sender", () => {
    const onSend = vi.fn();
    render(<ChatInput value="hi" onChange={() => {}} onSend={onSend} disabled={false} />);
    fireEvent.click(screen.getByTestId("send-btn"));
    expect(onSend).toHaveBeenCalledOnce();
  });

  it("both input and Send are disabled while streaming", () => {
    render(<ChatInput value="hi" onChange={() => {}} onSend={() => {}} disabled={true} />);
    expect(screen.getByTestId("chat-input")).toBeDisabled();
    expect(screen.getByTestId("send-btn")).toBeDisabled();
  });
});
