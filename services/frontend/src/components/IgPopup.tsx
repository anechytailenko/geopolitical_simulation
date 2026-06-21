import { formatAttribution } from "../lib/format";
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
            <span className="ig-feature">{a.feature}</span>
            <span
              className="ig-value"
              data-testid={`ig-value-${a.feature}`}
              style={{ color: pos ? "var(--pos)" : "var(--neg)", whiteSpace: "nowrap" }}
            >
              <span
                className="ig-bar"
                style={{
                  display: "inline-block",
                  width: `${Math.round((Math.abs(a.attribution) / max) * 44) + 3}px`,
                  background: pos ? "var(--pos)" : "var(--neg)",
                  verticalAlign: "middle",
                }}
              />{" "}
              {formatAttribution(a.attribution)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
