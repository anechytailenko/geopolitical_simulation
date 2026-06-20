package raw

import (
	"context"
	"fmt"

	"geopolitic/internal/neo4jdb"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// MergeFeatures upserts a FeatureSnapshot for an entity at one time_step and sets
// the given feature properties. Every time-varying loader (World Bank, V-Dem,
// SIPRI, UNDP, ACLED, sanctions, UNSC, financial tier) funnels through here, so
// feature values are always versioned by time_step — never stored as a static
// identity-node property. nodeType must be "Country" or "Actor".
func MergeFeatures(ctx context.Context, driver neo4j.DriverWithContext, nodeType, id string, ts, year int, features map[string]any) error {
	if nodeType != "Country" && nodeType != "Actor" {
		return fmt.Errorf("MergeFeatures: invalid nodeType %q", nodeType)
	}
	cypher := fmt.Sprintf(`
MERGE (n:%s {id: $id})
MERGE (n)-[:HAS_FEATURES]->(s:FeatureSnapshot {node_id: $id, node_type: $type, time_step: $ts})
SET s.year = $year, s += $features`, nodeType)
	return neo4jdb.Write(ctx, driver, cypher, map[string]any{
		"id":       id,
		"type":     nodeType,
		"ts":       ts,
		"year":     year,
		"features": features,
	})
}
