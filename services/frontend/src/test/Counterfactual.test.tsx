import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import App from "../App";
import { useStore } from "../store";
import { VIZ_COUNTERFACTUAL } from "./fixtures";

beforeEach(() =>
  useStore.setState({
    messages: [
      { role: "user", text: "if USA & China sign a deal, China-India?" },
      { role: "assistant", steps: [], answer: VIZ_COUNTERFACTUAL.answer },
    ],
    streaming: false,
    currentViz: VIZ_COUNTERFACTUAL,
    selectedNode: null,
    error: null,
    threadId: null,
  }),
);

describe("counterfactual rendering (plans/05 §8.10)", () => {
  it("shows the intervention banner, baseline vs counterfactual, and the intervened edge", () => {
    render(<App />);

    const banner = screen.getByTestId("cf-banner");
    expect(banner.textContent).toContain("USA");
    expect(banner.textContent).toContain("CHN");
    expect(banner.textContent).toContain("MATERIAL_COOPERATION");

    // baseline + counterfactual = two probability charts
    expect(screen.getAllByTestId("probability-bars").length).toBeGreaterThanOrEqual(2);

    // the intervened edge (USA-CHN) and the focus/prediction edge (CHN-IND) are both drawn
    expect(screen.getByTestId("intervention-edge-USA-CHN")).toBeInTheDocument();
    expect(screen.getByTestId("focus-edge-CHN-IND")).toBeInTheDocument();
  });
});
