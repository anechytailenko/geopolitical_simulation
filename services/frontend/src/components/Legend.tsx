import { CLASS_NAMES, classColor, classLabel } from "../lib/classColors";

// The 5-class semantic legend (separate from the purple chrome).
export function Legend() {
  return (
    <div className="legend" data-testid="legend">
      {CLASS_NAMES.map((c) => (
        <span className="item" key={c} data-testid={`legend-${c}`}>
          <span className="dot" style={{ background: classColor(c) }} />
          {classLabel(c)}
        </span>
      ))}
    </div>
  );
}
