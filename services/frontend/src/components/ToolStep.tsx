import type { ToolStep as Step } from "../store";

// One reason→act→observe step: "▸ name(args) → summary" (plans/05 §5).
export function ToolStep({ step }: { step: Step }) {
  const args = step.args
    ? Object.values(step.args)
        .filter((v) => v !== undefined && v !== null && v !== "")
        .join(", ")
    : "";
  return (
    <div className="tool-step" data-testid="tool-step">
      <span className={step.done ? "" : "pending"}>
        ▸ {step.name}({args}){step.summary ? ` → ${step.summary}` : ""}
      </span>
    </div>
  );
}
