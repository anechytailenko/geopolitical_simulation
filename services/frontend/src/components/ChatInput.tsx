// Chat input + Send (plans/05 §5). Enter submits; both disabled while streaming.
export function ChatInput({
  value,
  onChange,
  onSend,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  disabled: boolean;
}) {
  return (
    <div className="input-row">
      <input
        data-testid="chat-input"
        placeholder="Ask about a country pair or a group…"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !disabled) {
            e.preventDefault();
            onSend();
          }
        }}
      />
      <button data-testid="send-btn" disabled={disabled || !value.trim()} onClick={onSend}>
        Send
      </button>
    </div>
  );
}
