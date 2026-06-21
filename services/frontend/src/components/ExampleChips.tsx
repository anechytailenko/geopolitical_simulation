import { useRef } from "react";

// The 4 example questions (plans/05 §5) — the 3 from 04 §1 + the Type-6 what-if.
export const EXAMPLE_QUESTIONS = [
  "What's most likely between the USA and China next month?",
  "Within the EU, which pair is most likely material cooperation?",
  "Who is most likely to enter material conflict with Ukraine?",
  "If the USA & China sign a deal, what about China and India?",
];

const SHORT = ["USA↔CHN?", "EU coop pair?", "→UKR conflict?", "what-if: USA·CHN→CHN·IND"];

// A horizontally scrollable strip placed directly above the chat input. Clicking a chip prefills
// the input (the parent ChatPanel owns the input value).
export function ExampleChips({
  onPick,
  activeIndex,
}: {
  onPick: (q: string, i: number) => void;
  activeIndex: number | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const scroll = (dx: number) => ref.current?.scrollBy({ left: dx, behavior: "smooth" });
  return (
    <div className="chips-row" style={{ display: "flex", alignItems: "center" }} data-testid="example-chips">
      <button className="chip-scroll-btn" aria-label="scroll left" onClick={() => scroll(-220)}>
        ‹
      </button>
      <div className="chips" ref={ref} style={{ overflowX: "auto" }} data-testid="chips-scroll">
        {EXAMPLE_QUESTIONS.map((q, i) => (
          <button
            key={i}
            className={"chip" + (activeIndex === i ? " active" : "")}
            data-testid={`chip-${i}`}
            title={q}
            onClick={() => onPick(q, i)}
          >
            {SHORT[i]}
          </button>
        ))}
      </div>
      <button className="chip-scroll-btn" aria-label="scroll right" onClick={() => scroll(220)}>
        ›
      </button>
    </div>
  );
}
