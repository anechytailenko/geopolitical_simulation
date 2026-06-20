// Package neo4jdb wraps the neo4j-go-driver with a few helpers used by both the
// raw and aggregated loaders.
package neo4jdb

import (
	"context"
	"fmt"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Connect opens a driver and verifies connectivity so callers fail fast if the
// container is not up yet.
func Connect(ctx context.Context, uri, user, pass string) (neo4j.DriverWithContext, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(user, pass, ""))
	if err != nil {
		return nil, fmt.Errorf("open driver %s: %w", uri, err)
	}
	if err := driver.VerifyConnectivity(ctx); err != nil {
		driver.Close(ctx)
		return nil, fmt.Errorf("verify connectivity %s: %w", uri, err)
	}
	return driver, nil
}

// Write runs a single write query in an auto-committed managed transaction.
func Write(ctx context.Context, driver neo4j.DriverWithContext, cypher string, params map[string]any) error {
	session := driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)
	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, cypher, params)
	})
	return err
}

// WriteBatch runs the same query once per parameter map inside one transaction —
// used to batch ~1,000 EVENT/snapshot writes per round-trip.
func WriteBatch(ctx context.Context, driver neo4j.DriverWithContext, cypher string, batch []map[string]any) error {
	if len(batch) == 0 {
		return nil
	}
	session := driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)
	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, params := range batch {
			if _, err := tx.Run(ctx, cypher, params); err != nil {
				return nil, err
			}
		}
		return nil, nil
	})
	return err
}

// Count returns a single integer result (e.g. a COUNT(*) query). Handy in tests.
func Count(ctx context.Context, driver neo4j.DriverWithContext, cypher string, params map[string]any) (int64, error) {
	session := driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer session.Close(ctx)
	res, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		r, err := tx.Run(ctx, cypher, params)
		if err != nil {
			return nil, err
		}
		rec, err := r.Single(ctx)
		if err != nil {
			return nil, err
		}
		return rec.Values[0], nil
	})
	if err != nil {
		return 0, err
	}
	n, ok := res.(int64)
	if !ok {
		return 0, fmt.Errorf("expected int64 result, got %T", res)
	}
	return n, nil
}

// ReadRows runs a read query and returns every record as a map.
func ReadRows(ctx context.Context, driver neo4j.DriverWithContext, cypher string, params map[string]any) ([]map[string]any, error) {
	session := driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer session.Close(ctx)
	res, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		r, err := tx.Run(ctx, cypher, params)
		if err != nil {
			return nil, err
		}
		var rows []map[string]any
		for r.Next(ctx) {
			rows = append(rows, r.Record().AsMap())
		}
		return rows, r.Err()
	})
	if err != nil {
		return nil, err
	}
	return res.([]map[string]any), nil
}
