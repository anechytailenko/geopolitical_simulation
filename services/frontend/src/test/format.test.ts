import { describe, expect, it } from "vitest";

import { formatAttribution } from "../lib/format";

describe("formatAttribution (IG numbers must not collapse to 0.00)", () => {
  it("keeps small non-zero attributions visible (the reported bug)", () => {
    // these are real logit-IG magnitudes; toFixed(2) would render all of them as "0.00"
    expect(formatAttribution(0.0283)).toBe("+0.028");
    expect(formatAttribution(0.0045)).toBe("+0.0045");
    expect(formatAttribution(-0.0066)).toBe("-0.0066");
    expect(formatAttribution(0.000066)).toBe("+6.6e-5");
  });

  it("formats larger values compactly with a sign", () => {
    expect(formatAttribution(0.283)).toBe("+0.28");
    expect(formatAttribution(-0.12)).toBe("-0.12");
  });

  it("handles zero and non-finite", () => {
    expect(formatAttribution(0)).toBe("0");
    expect(formatAttribution(Number.NaN)).toBe("—");
  });
});
