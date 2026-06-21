import { useState } from "react";

import { useStore } from "../store";
import { ChatInput } from "./ChatInput";
import { ExampleChips } from "./ExampleChips";
import { MessageList } from "./MessageList";

// Right panel (plans/05 §1, §5): the chat transcript, the scrollable example chips ABOVE the
// input, and the input + Send.
export function ChatPanel() {
  const messages = useStore((s) => s.messages);
  const currentViz = useStore((s) => s.currentViz);
  const error = useStore((s) => s.error);
  const streaming = useStore((s) => s.streaming);
  const send = useStore((s) => s.send);

  const [text, setText] = useState("");
  const [activeChip, setActiveChip] = useState<number | null>(null);

  const doSend = () => {
    const t = text.trim();
    if (!t) return;
    setText("");
    setActiveChip(null);
    void send(t);
  };

  return (
    <div className="panel-right" data-testid="chat-panel">
      <div className="panel-title">
        <span>Chat</span>
        <span className="meta">local model</span>
      </div>
      <MessageList messages={messages} currentViz={currentViz} error={error} />
      <ExampleChips
        activeIndex={activeChip}
        onPick={(q, i) => {
          setText(q);
          setActiveChip(i);
        }}
      />
      <ChatInput value={text} onChange={setText} onSend={doSend} disabled={streaming} />
    </div>
  );
}
