package raw

import (
	"context"

	"geopolitic/internal/neo4jdb"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// rawSchema is the constraint + index set for geopolitic_raw, mirroring the
// "Neo4j Schema" section of plans/01-architecture.md. Every statement is
// idempotent (IF NOT EXISTS) so ApplySchema is safe to re-run on every ingest.
var rawSchema = []string{
	"CREATE CONSTRAINT country_id IF NOT EXISTS FOR (c:Country) REQUIRE c.id IS UNIQUE",
	"CREATE CONSTRAINT actor_id IF NOT EXISTS FOR (a:Actor) REQUIRE a.id IS UNIQUE",
	"CREATE INDEX feature_node_ts IF NOT EXISTS FOR (s:FeatureSnapshot) ON (s.node_id, s.time_step)",
	"CREATE INDEX event_ts IF NOT EXISTS FOR ()-[r:EVENT]-() ON (r.time_step)",
	"CREATE INDEX event_ts_class IF NOT EXISTS FOR ()-[r:EVENT]-() ON (r.time_step, r.relationship_class)",
	"CREATE INDEX member_start IF NOT EXISTS FOR ()-[r:MEMBER_OF]-() ON (r.start_time_step)",
	"CREATE INDEX borders_start IF NOT EXISTS FOR ()-[r:BORDERS]-() ON (r.start_time_step)",
}

// ApplySchema creates all constraints and indexes on geopolitic_raw.
func ApplySchema(ctx context.Context, driver neo4j.DriverWithContext) error {
	for _, stmt := range rawSchema {
		if err := neo4jdb.Write(ctx, driver, stmt, nil); err != nil {
			return err
		}
	}
	return nil
}
