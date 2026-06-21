import type { FeatureAttribution } from "../types";

// Integrated-Gradients popup for a focus-pair node (plans/05 §4). Signed bars, top-N by |attr|.
export function IgPopup({
  nodeId,
  attrs,
  onClose,
}: {
  nodeId: string;
  attrs: FeatureAttribution[];
  onClose: () => void;
}) {
  const top = [...attrs]
    .sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution))
    .slice(0, 6);
  const max = Math.max(1e-6, ...top.map((a) => Math.abs(a.attribution)));
  return (
    <div className="ig-popup" data-testid="ig-popup">
      <div className="ig-head">
        <span>why {nodeId}</span>
        <button aria-label="close" onClick={onClose}>
          ×
        </button>
      </div>
      {top.map((a) => {
        const pos = a.attribution >= 0;
        return (
          <div className="ig-row" data-testid={`ig-row-${a.feature}`} key={a.feature}>
            <span>{a.feature}</span>
            <span style={{ color: pos ? "var(--pos)" : "var(--neg)" }}>
              <span
                className="ig-bar"
                style={{
                  display: "inline-block",
                  width: `${Math.round((Math.abs(a.attribution) / max) * 56) + 4}px`,
                  background: pos ? "var(--pos)" : "var(--neg)",
                }}
              />{" "}
              {pos ? "+" : ""}
              {a.attribution.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
