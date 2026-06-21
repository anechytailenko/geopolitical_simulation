import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SubgraphCanvas } from "../components/SubgraphCanvas";
import { VIZ_USA_CHN } from "./fixtures";

describe("SubgraphCanvas (plans/05 §8.6, §8.12)", () => {
  it("renders nodes, input edges, and the focus prediction edge", () => {
    render(<SubgraphCanvas viz={VIZ_USA_CHN} onNodeClick={() => {}} />);
    expect(screen.getByTestId("node-USA")).toBeInTheDocument();
    expect(screen.getByTestId("node-CHN")).toBeInTheDocument();
    expect(screen.getByTestId("node-DEU")).toBeInTheDocument();
    expect(screen.getByTestId("edge-DEU-USA")).toBeInTheDocument(); // input edge
    expect(screen.getByTestId("focus-edge-USA-CHN")).toHaveAttribute("stroke", "var(--focus)");
  });

  it("node opacity reflects importance", () => {
    render(<SubgraphCanvas viz={VIZ_USA_CHN} onNodeClick={() => {}} />);
    const hi = Number(screen.getByTestId("node-circle-USA").getAttribute("fill-opacity"));
    const lo = Number(screen.getByTestId("node-circle-DEU").getAttribute("fill-opacity"));
    expect(hi).toBeGreaterThan(lo);
  });

  it("draws the prediction edge even when it is NOT an input edge", () => {
    // VIZ_USA_CHN.subgraph.edges contains only DEU→USA, not the focus pair USA→CHN
    render(<SubgraphCanvas viz={VIZ_USA_CHN} onNodeClick={() => {}} />);
    expect(screen.queryByTestId("edge-USA-CHN")).toBeNull(); // not an input edge
    expect(screen.getByTestId("focus-edge-USA-CHN")).toBeInTheDocument(); // but the focus edge is drawn
  });
});
