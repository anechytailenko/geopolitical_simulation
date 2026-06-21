import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation } from "d3-force";
import { useMemo } from "react";

import { classColor } from "../lib/classColors";
import type { Viz } from "../types";

const W = 520;
const H = 420;

// The reasoning subgraph (plans/05 §4). d3-force is used purely as a layout calculator: the
// simulation is created STOPPED and ticked synchronously, so there are no timers/animation frames
// (deterministic and jsdom-friendly). Two edge layers are drawn: (a) GNNExplainer input edges and
// (b) the synthesized prediction/focus edge between the focus nodes (always drawn — the answer).
export function SubgraphCanvas({ viz, onNodeClick }: { viz: Viz; onNodeClick: (id: string) => void }) {
  const pos = useMemo(() => {
    const nodes: any[] = viz.subgraph.nodes.map((n) => ({ ...n }));
    const idx = new Map(nodes.map((n, i) => [n.id, i]));
    const links = viz.subgraph.edges
      .filter((e) => idx.has(e.src) && idx.has(e.tgt))
      .map((e) => ({ source: idx.get(e.src)!, target: idx.get(e.tgt)! }));
    const sim = forceSimulation(nodes)
      .force("link", forceLink(links).distance(95).strength(0.6))
      .force("charge", forceManyBody().strength(-280))
      .force("center", forceCenter(W / 2, H / 2))
      .force("collide", forceCollide(30))
      .stop();
    for (let i = 0; i < 300; i++) sim.tick();
    return new Map<string, { x: number; y: number }>(nodes.map((n) => [n.id, { x: n.x, y: n.y }]));
  }, [viz]);

  const get = (id: string) => pos.get(id);

  return (
    <svg className="subgraph-svg" viewBox={`0 0 ${W} ${H}`} data-testid="subgraph">
      {/* (a) GNNExplainer input edges — colored by dominant_class, opacity ∝ importance */}
      {viz.subgraph.edges.map((e, i) => {
        const a = get(e.src);
        const b = get(e.tgt);
        if (!a || !b) return null;
        return (
          <line
            key={`in-${i}`}
            data-testid={`edge-${e.src}-${e.tgt}`}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke={classColor(e.dominant_class)}
            strokeOpacity={0.2 + 0.7 * e.importance}
            strokeWidth={1.5}
          />
        );
      })}

      {/* intervention edge (Type-6 what-if) — dashed accent between the intervened pair */}
      {viz.intervention &&
        (() => {
          const a = get(viz.intervention.src);
          const b = get(viz.intervention.tgt);
          if (!a || !b) return null;
          return (
            <line
              data-testid={`intervention-edge-${viz.intervention.src}-${viz.intervention.tgt}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke="var(--accent)"
              strokeWidth={3}
              strokeDasharray="5 4"
            />
          );
        })()}

      {/* (b) prediction (focus) edges — synthesized, always drawn in --focus */}
      {viz.focus_pairs.map((fp, i) => {
        const a = get(fp.src);
        const b = get(fp.tgt);
        if (!a || !b) return null;
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2;
        return (
          <g key={`focus-${i}`}>
            <line
              data-testid={`focus-edge-${fp.src}-${fp.tgt}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke="var(--focus)"
              strokeWidth={4}
              strokeOpacity={0.95}
            />
            <text className="edge-label" x={mx} y={my - 6} textAnchor="middle">
              {fp.predicted_class} · p={fp.confidence.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* nodes */}
      {viz.subgraph.nodes.map((n) => {
        const p = get(n.id);
        if (!p) return null;
        const r = 8 + 10 * n.importance;
        return (
          <g
            key={n.id}
            transform={`translate(${p.x},${p.y})`}
            data-testid={`node-${n.id}`}
            data-ig-clickable={n.ig_clickable ? "true" : "false"}
            style={{ cursor: n.ig_clickable ? "pointer" : "default" }}
            onClick={() => n.ig_clickable && onNodeClick(n.id)}
          >
            <circle
              data-testid={`node-circle-${n.id}`}
              r={r}
              fill={n.ig_clickable ? "var(--accent)" : "var(--surface-2)"}
              fillOpacity={0.4 + 0.6 * n.importance}
              stroke="var(--border)"
            />
            <text className="node-label" textAnchor="middle" dy={r + 12}>
              {n.id}
            </text>
            <title>{n.name}</title>
          </g>
        );
      })}
    </svg>
  );
}
