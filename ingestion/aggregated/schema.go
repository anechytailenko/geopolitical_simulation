// Package aggregated derives the geopolitic_aggregated database from
// geopolitic_raw: forward-filled NodeSnapshots and per-month SNAPSHOT_EDGE
// aggregates. This is the only database the ML training job and frontend read.
package aggregated

import (
	"context"

	"geopolitic/internal/neo4jdb"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

var aggSchema = []string{
	"CREATE CONSTRAINT agg_country_id IF NOT EXISTS FOR (c:Country) REQUIRE c.id IS UNIQUE",
	"CREATE CONSTRAINT agg_actor_id IF NOT EXISTS FOR (a:Actor) REQUIRE a.id IS UNIQUE",
	"CREATE INDEX node_snapshot_ts IF NOT EXISTS FOR (n:NodeSnapshot) ON (n.node_id, n.time_step)",
	"CREATE INDEX snapshot_edge_ts IF NOT EXISTS FOR ()-[r:SNAPSHOT_EDGE]-() ON (r.time_step)",
	"CREATE INDEX snapshot_edge_ts_class IF NOT EXISTS FOR ()-[r:SNAPSHOT_EDGE]-() ON (r.time_step, r.dominant_class)",
}

// ApplySchema creates all constraints and indexes on geopolitic_aggregated.
func ApplySchema(ctx context.Context, driver neo4j.DriverWithContext) error {
	for _, stmt := range aggSchema {
		if err := neo4jdb.Write(ctx, driver, stmt, nil); err != nil {
			return err
		}
	}
	return nil
}
