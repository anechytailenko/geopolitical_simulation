// Adaptive number formatting for Integrated-Gradients attributions (plans/05 §4).
// `toFixed(2)` collapses small attributions (e.g. 0.0045) to "0.00"; this keeps a signed,
// readable value across magnitudes so a non-zero attribution is never shown as zero.
export function formatAttribution(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a < 1e-9) return "0";
  const sign = v > 0 ? "+" : "-";
  let body: string;
  if (a >= 0.1) body = a.toFixed(2);
  else if (a >= 0.01) body = a.toFixed(3);
  else if (a >= 0.001) body = a.toFixed(4);
  else body = a.toExponential(1); // e.g. 6.6e-5
  return sign + body;
}
