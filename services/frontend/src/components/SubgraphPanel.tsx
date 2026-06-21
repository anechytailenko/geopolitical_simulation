import { useStore } from "../store";
import { IgPopup } from "./IgPopup";
import { Legend } from "./Legend";
import { SubgraphCanvas } from "./SubgraphCanvas";

// Left panel (plans/05 §1, §4): the reasoning subgraph + legend + IG popup. The popup opens only
// for ig_clickable focus nodes that have feature attributions (04 §3 A1).
export function SubgraphPanel() {
  const viz = useStore((s) => s.currentViz);
  const selected = useStore((s) => s.selectedNode);
  const selectNode = useStore((s) => s.selectNode);

  return (
    <div className="panel-left" data-testid="subgraph-panel">
      <div className="panel-title">
        <span>Reasoning subgraph</span>
        {viz?.forecast_period && (
          <span className="explainer-tag" data-testid="explainer-tag">
            GNNExplainer · {viz.forecast_period}
          </span>
        )}
      </div>
      <div className="subgraph-wrap">
        {viz ? (
          <SubgraphCanvas viz={viz} onNodeClick={selectNode} />
        ) : (
          <div className="empty-hint">Ask a question to see the reasoning subgraph.</div>
        )}
        {viz && selected && viz.feature_attributions[selected] && (
          <IgPopup
            nodeId={selected}
            attrs={viz.feature_attributions[selected]}
            onClose={() => selectNode(null)}
          />
        )}
      </div>
      <Legend />
    </div>
  );
}
