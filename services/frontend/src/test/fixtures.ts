// Test fixtures: a self-consistent `final` viz payload (shape = plans/04 §6) and helpers to mock
// the agent's SSE stream. The focus pair (USA→CHN) is deliberately NOT among the input edges, so
// the prediction-edge test (plans/05 §8.12) is exercised.
import type { Viz } from "../types";

export const VIZ_USA_CHN: Viz = {
  answer: "Next month (2026-07) USA → China is most likely MATERIAL_CONFLICT.",
  time_step: 197,
  input_period: "2026-06",
  forecast_period: "2026-07",
  confidence_note: "model confidence (temperature-scaled, ~uncalibrated on this checkpoint)",
  focus_pairs: [
    {
      src: "USA",
      tgt: "CHN",
      predicted_class: "MATERIAL_CONFLICT",
      confidence: 0.98,
      probabilities: {
        MATERIAL_CONFLICT: 0.98,
        VERBAL_CONFLICT: 0.01,
        MATERIAL_COOPERATION: 0.003,
        VERBAL_COOPERATION: 0.004,
        STATUS_QUO: 0.003,
      },
    },
  ],
  intervention: null,
  subgraph: {
    nodes: [
      { id: "USA", name: "United States", type: "Country", importance: 1.0, ig_clickable: true },
      { id: "CHN", name: "China", type: "Country", importance: 1.0, ig_clickable: true },
      { id: "DEU", name: "Germany", type: "Country", importance: 0.41, ig_clickable: false },
    ],
    // an input edge that does NOT involve the focus pair directly
    edges: [{ src: "DEU", tgt: "USA", dominant_class: "MATERIAL_COOPERATION", importance: 0.41 }],
  },
  feature_attributions: {
    USA: [
      { feature: "military_expenditure_log", attribution: 0.21 },
      { feature: "conflict_intensity", attribution: -0.14 },
    ],
    CHN: [{ feature: "vdem_polyarchy_score", attribution: 0.09 }],
  },
};

// A Type-6 what-if payload: query CHN→IND under the assumption USA–CHN cooperate.
export const VIZ_COUNTERFACTUAL: Viz = {
  answer: "If the USA and China cooperate, China → India stays roughly MATERIAL_CONFLICT.",
  time_step: 197,
  input_period: "2026-06",
  forecast_period: "2026-07",
  confidence_note: "model confidence (temperature-scaled, ~uncalibrated on this checkpoint)",
  focus_pairs: [
    {
      src: "CHN",
      tgt: "IND",
      predicted_class: "MATERIAL_CONFLICT",
      confidence: 0.95,
      probabilities: {
        MATERIAL_CONFLICT: 0.95,
        VERBAL_CONFLICT: 0.02,
        MATERIAL_COOPERATION: 0.01,
        VERBAL_COOPERATION: 0.01,
        STATUS_QUO: 0.01,
      },
    },
  ],
  intervention: { src: "USA", tgt: "CHN", class: "MATERIAL_COOPERATION", symmetric: true },
  counterfactual: {
    baseline: {
      predicted_class: "MATERIAL_CONFLICT",
      probabilities: {
        MATERIAL_CONFLICT: 0.96,
        VERBAL_CONFLICT: 0.02,
        MATERIAL_COOPERATION: 0.005,
        VERBAL_COOPERATION: 0.01,
        STATUS_QUO: 0.005,
      },
    },
    counterfactual: {
      predicted_class: "MATERIAL_CONFLICT",
      probabilities: {
        MATERIAL_CONFLICT: 0.95,
        VERBAL_CONFLICT: 0.02,
        MATERIAL_COOPERATION: 0.01,
        VERBAL_COOPERATION: 0.01,
        STATUS_QUO: 0.01,
      },
    },
    delta: { MATERIAL_COOPERATION: 0.005 },
  },
  subgraph: {
    nodes: [
      { id: "CHN", name: "China", type: "Country", importance: 1.0, ig_clickable: true },
      { id: "IND", name: "India", type: "Country", importance: 1.0, ig_clickable: true },
      { id: "USA", name: "United States", type: "Country", importance: 0.5, ig_clickable: false },
    ],
    edges: [{ src: "USA", tgt: "CHN", dominant_class: "MATERIAL_CONFLICT", importance: 0.5 }],
  },
  feature_attributions: {
    CHN: [{ feature: "military_expenditure_log", attribution: 0.18 }],
    IND: [{ feature: "conflict_intensity", attribution: -0.07 }],
  },
};

/** Build one SSE frame. */
export function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** A ReadableStream that emits the given string chunks as UTF-8 (one per pull). */
export function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) controller.enqueue(enc.encode(chunks[i++]));
      else controller.close();
    },
  });
}

/** A `fetch` stub that returns an SSE response streaming the given chunks. */
export function mockSseFetch(chunks: string[]): typeof fetch {
  return (async () =>
    ({ ok: true, status: 200, body: streamOf(chunks) }) as unknown as Response) as typeof fetch;
}

/** The scripted stream for a successful predict_pair turn. */
export function scriptedPredictPair(): string[] {
  return [
    sseFrame("tool_call", { name: "predict_pair", args: { source: "USA", target: "CHN" } }),
    sseFrame("tool_result", { name: "predict_pair", summary: "USA->CHN: MATERIAL_CONFLICT" }),
    sseFrame("token", { text: VIZ_USA_CHN.answer }),
    sseFrame("final", VIZ_USA_CHN),
  ];
}
