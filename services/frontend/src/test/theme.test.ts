import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { CLASS_COLORS, CLASS_NAMES } from "../lib/classColors";

describe("classColors + theme (plans/05 §8.8)", () => {
  it("CLASS_NAMES is the canonical 5 classes in order", () => {
    expect([...CLASS_NAMES]).toEqual([
      "MATERIAL_CONFLICT",
      "VERBAL_CONFLICT",
      "MATERIAL_COOPERATION",
      "VERBAL_COOPERATION",
      "STATUS_QUO",
    ]);
    expect(Object.keys(CLASS_COLORS)).toHaveLength(5);
  });

  it("theme.css uses the electric-purple accent (no green chrome)", () => {
    // Vitest root is services/frontend, so the CSS is at src/theme.css.
    const css = readFileSync(resolve(process.cwd(), "src/theme.css"), "utf8");
    expect(css).toMatch(/--accent:\s*#a855f7/);
    expect(css).toMatch(/--focus:\s*#c084fc/);
  });
});
