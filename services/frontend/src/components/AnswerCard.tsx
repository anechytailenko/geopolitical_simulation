import type { Viz } from "../types";
import { ProbabilityBars } from "./ProbabilityBars";

// The answer card: answer text + the focus pair's probability chart, or a baseline-vs-counterfactual
// pair for a Type-6 what-if (plans/05 §5). The confidence note is shown subtly, never as "calibrated".
export function AnswerCard({ answer, viz }: { answer: string; viz: Viz | null }) {
  const fp = viz?.focus_pairs?.[0];
  const cf = viz?.counterfactual;
  return (
    <div className="answer-card" data-testid="answer-card">
      {viz?.intervention && (
        <div className="cf-banner" data-testid="cf-banner">
          assuming {viz.intervention.src}–{viz.intervention.tgt} → {viz.intervention.class}
        </div>
      )}
      <div className="answer-text">{answer}</div>
      {cf ? (
        <div className="cf-cols">
          <div>
            <ProbabilityBars title="baseline" probabilities={cf.baseline.probabilities} />
          </div>
          <div>
            <ProbabilityBars title="counterfactual" probabilities={cf.counterfactual.probabilities} />
          </div>
        </div>
      ) : fp ? (
        <ProbabilityBars
          title={`${fp.src} → ${fp.tgt}${viz?.forecast_period ? ` · ${viz.forecast_period}` : ""}`}
          probabilities={fp.probabilities}
        />
      ) : null}
      {viz?.confidence_note && (
        <div className="conf-note" data-testid="conf-note">
          {viz.confidence_note}
        </div>
      )}
    </div>
  );
}
