import { useStore } from "../store";

export function Header() {
  const reset = useStore((s) => s.reset);
  return (
    <div className="header" data-testid="header">
      <div className="brand">
        <span className="logo">Geopolitic Agent</span>
        <span className="subtitle">forecasting country↔country relations · next month</span>
      </div>
      <div className="meta">
        <button onClick={reset} data-testid="reset-btn">
          ↺ reset
        </button>
      </div>
    </div>
  );
}
