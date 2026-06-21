import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ProbabilityBars } from "../components/ProbabilityBars";
import { VIZ_USA_CHN } from "./fixtures";

describe("ProbabilityBars (plans/05 §8.5)", () => {
  it("renders 5 class bars sorted descending, with widths ∝ probability and labels", () => {
    const probs = VIZ_USA_CHN.focus_pairs[0].probabilities;
    render(<ProbabilityBars probabilities={probs} />);

    const rows = screen.getAllByTestId(/^prob-/);
    expect(rows).toHaveLength(5);
    // sorted descending -> first row is the argmax class (MATERIAL_CONFLICT, 0.98)
    expect(rows[0]).toHaveAttribute("data-testid", "prob-MATERIAL_CONFLICT");

    // width ∝ probability
    expect(screen.getByTestId("probfill-MATERIAL_CONFLICT").style.width).toBe("98%");
    // labeled with its value
    expect(within(rows[0] as HTMLElement).getByText("0.98")).toBeInTheDocument();
  });
});
