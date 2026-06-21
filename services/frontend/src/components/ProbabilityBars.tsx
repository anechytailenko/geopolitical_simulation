import { classColor, classLabel } from "../lib/classColors";

// A horizontal probability chart: one class-colored bar per class, sorted descending (plans/05 §5).
export function ProbabilityBars({
  probabilities,
  title,
}: {
  probabilities: Record<string, number>;
  title?: string;
}) {
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div data-testid="probability-bars">
      {title && <div className="prob-title">{title}</div>}
      {rows.map(([cls, p]) => (
        <div className="prob-row" data-testid={`prob-${cls}`} key={cls}>
          <span>{classLabel(cls)}</span>
          <div className="prob-track">
            <div
              className="prob-fill"
              data-testid={`probfill-${cls}`}
              style={{ width: `${Math.round(p * 100)}%`, background: classColor(cls) }}
            />
          </div>
          <span>{p.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}
