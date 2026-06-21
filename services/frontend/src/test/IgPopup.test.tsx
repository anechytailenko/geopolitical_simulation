import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { IgPopup } from "../components/IgPopup";
import { SubgraphPanel } from "../components/SubgraphPanel";
import { useStore } from "../store";
import { VIZ_USA_CHN } from "./fixtures";

beforeEach(() =>
  useStore.setState({
    messages: [],
    streaming: false,
    currentViz: VIZ_USA_CHN,
    selectedNode: null,
    error: null,
    threadId: null,
  }),
);

describe("IG popup gating (plans/05 §8.7)", () => {
  it("opens for an ig_clickable focus node and shows its named attributions", () => {
    render(<SubgraphPanel />);
    fireEvent.click(screen.getByTestId("node-USA"));
    expect(screen.getByTestId("ig-popup")).toBeInTheDocument();
    expect(screen.getByTestId("ig-row-military_expenditure_log")).toBeInTheDocument();
  });

  it("does nothing for a non-ig_clickable node", () => {
    render(<SubgraphPanel />);
    fireEvent.click(screen.getByTestId("node-DEU"));
    expect(screen.queryByTestId("ig-popup")).toBeNull();
  });

  it("renders the numeric attribution for small values (the reported popup bug)", () => {
    render(
      <IgPopup
        nodeId="CHN"
        attrs={[
          { feature: "gdp_log", attribution: 0.0283 },
          { feature: "trade_openness_index", attribution: 0.0045 },
        ]}
        onClose={() => {}}
      />,
    );
    // both numbers are visible (would both be "+0.00" under the old toFixed(2))
    expect(screen.getByTestId("ig-value-gdp_log").textContent).toContain("+0.028");
    expect(screen.getByTestId("ig-value-trade_openness_index").textContent).toContain("+0.0045");
  });
});
