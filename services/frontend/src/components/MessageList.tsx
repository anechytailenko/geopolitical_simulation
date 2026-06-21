import type { Message } from "../store";
import type { Viz } from "../types";
import { AnswerCard } from "./AnswerCard";
import { ToolStep } from "./ToolStep";

// The chat transcript (plans/05 §5): user bubbles, the assistant's streamed tool steps, and the
// answer card. The viz (left panel) is attached to the most recent assistant turn.
export function MessageList({
  messages,
  currentViz,
  error,
}: {
  messages: Message[];
  currentViz: Viz | null;
  error: string | null;
}) {
  let lastAssistant = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      lastAssistant = i;
      break;
    }
  }
  return (
    <div className="messages" data-testid="messages">
      {messages.map((m, i) =>
        m.role === "user" ? (
          <div className="bubble-user" data-testid="user-bubble" key={i}>
            {m.text}
          </div>
        ) : (
          <div className="assistant" data-testid="assistant-msg" key={i}>
            {m.steps.map((s, j) => (
              <ToolStep step={s} key={j} />
            ))}
            {(m.answer || (i === lastAssistant && currentViz)) && (
              <AnswerCard answer={m.answer} viz={i === lastAssistant ? currentViz : null} />
            )}
          </div>
        ),
      )}
      {error && (
        <div className="error-bubble" data-testid="error-bubble">
          {error}
        </div>
      )}
    </div>
  );
}
