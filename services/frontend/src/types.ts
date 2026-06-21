// TypeScript mirror of the agent's SSE contract (plans/04 §6, plans/05 §2). The frontend never
// re-implements ML logic and never reorders classes — it only renders these shapes.

export interface FocusPair {
  src: string;
  tgt: string;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface SubgraphNode {
  id: string;
  name: string;
  type: string;
  importance: number;
  ig_clickable: boolean;
}

export interface SubgraphEdge {
  src: string;
  tgt: string;
  dominant_class: string | null;
  importance: number;
}

export interface FeatureAttribution {
  feature: string;
  attribution: number;
}

export interface Intervention {
  src: string;
  tgt: string;
  class: string;
  symmetric?: boolean;
}

export interface CounterfactualDetail {
  baseline: { probabilities: Record<string, number>; predicted_class: string; confidence?: number };
  counterfactual: { probabilities: Record<string, number>; predicted_class: string; confidence?: number };
  delta?: Record<string, number>;
}

export interface Viz {
  answer: string;
  time_step: number;
  input_period?: string;
  forecast_period?: string;
  confidence_note?: string;
  focus_pairs: FocusPair[];
  intervention: Intervention | null;
  counterfactual?: CounterfactualDetail; // present only for a Type-6 what-if
  subgraph: { nodes: SubgraphNode[]; edges: SubgraphEdge[] };
  feature_attributions: Record<string, FeatureAttribution[]>;
}

// SSE events streamed by POST /agent/chat.
export type AgentEvent =
  | { type: "token"; text: string }
  | { type: "tool_call"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; summary: string }
  | { type: "final"; viz: Viz }
  | { type: "error"; message: string };
