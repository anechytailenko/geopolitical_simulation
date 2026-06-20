package aggregated

import (
	"context"
	"fmt"
	"math"
	"sort"

	"geopolitic/internal/label"
	"geopolitic/internal/neo4jdb"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Stats summarizes what Build wrote into geopolitic_aggregated.
type Stats struct {
	IdentityNodes   int
	NodeSnapshots   int
	SnapshotEdges   int
	StructuralEdges int
	MaxTimeStep     int
}

// bookkeeping FeatureSnapshot properties that are not model features.
var featureMeta = map[string]bool{
	"node_id": true, "node_type": true, "time_step": true, "year": true,
}

// Build reads geopolitic_raw and (re)builds geopolitic_aggregated:
// identity-node mirror, mirrored structural edges, forward-filled NodeSnapshots,
// and per-(source,target,month) SNAPSHOT_EDGE aggregates.
//
// eventMaxTS lets the caller extend NodeSnapshot forward-fill past the latest raw
// observation — used when events live outside Neo4j (GDELT aggregated in
// BigQuery), so node features still span every month that has SNAPSHOT_EDGEs.
// Pass 0 when there is no external event source.
func Build(ctx context.Context, rawDriver, aggDriver neo4j.DriverWithContext, eventMaxTS int) (Stats, error) {
	var stats Stats

	if err := ApplySchema(ctx, aggDriver); err != nil {
		return stats, fmt.Errorf("agg schema: %w", err)
	}

	identities, err := mirrorIdentities(ctx, rawDriver, aggDriver)
	if err != nil {
		return stats, err
	}
	stats.IdentityNodes = len(identities)

	structEdges, err := mirrorStructuralEdges(ctx, rawDriver, aggDriver)
	if err != nil {
		return stats, err
	}
	stats.StructuralEdges = structEdges

	maxTS, err := maxTimeStep(ctx, rawDriver)
	if err != nil {
		return stats, err
	}
	if eventMaxTS > maxTS {
		maxTS = eventMaxTS
	}
	stats.MaxTimeStep = maxTS

	nodeSnaps, err := buildNodeSnapshots(ctx, rawDriver, aggDriver, identities, maxTS)
	if err != nil {
		return stats, err
	}
	stats.NodeSnapshots = nodeSnaps

	edges, err := buildSnapshotEdges(ctx, rawDriver, aggDriver)
	if err != nil {
		return stats, err
	}
	stats.SnapshotEdges = edges

	return stats, nil
}

type identity struct {
	id    string
	label string // "Country" | "Actor"
	props map[string]any
}

// staticProps are the identity-node properties carried onto every NodeSnapshot.
func (i identity) staticProps() map[string]any {
	keep := map[string]bool{
		"region": true, "nuclear_flag": true, "coastline_flag": true,
		"unsc_seat_flag": true, "financial_resources_tier": true,
	}
	out := map[string]any{}
	for k, v := range i.props {
		if keep[k] {
			out[k] = v
		}
	}
	return out
}

func mirrorIdentities(ctx context.Context, raw, agg neo4j.DriverWithContext) ([]identity, error) {
	var identities []identity
	for _, l := range []string{"Country", "Actor"} {
		rows, err := neo4jdb.ReadRows(ctx, raw,
			fmt.Sprintf("MATCH (n:%s) RETURN n.id AS id, properties(n) AS props", l), nil)
		if err != nil {
			return nil, fmt.Errorf("read %s identities: %w", l, err)
		}
		for _, r := range rows {
			id, _ := r["id"].(string)
			props, _ := r["props"].(map[string]any)
			identities = append(identities, identity{id: id, label: l, props: props})
			cypher := fmt.Sprintf("MERGE (n:%s {id: $id}) SET n += $props", l)
			if err := neo4jdb.Write(ctx, agg, cypher, map[string]any{"id": id, "props": props}); err != nil {
				return nil, fmt.Errorf("mirror %s %s: %w", l, id, err)
			}
		}
	}
	return identities, nil
}

func mirrorStructuralEdges(ctx context.Context, raw, agg neo4j.DriverWithContext) (int, error) {
	count := 0
	// BORDERS (Country->Country).
	borders, err := neo4jdb.ReadRows(ctx, raw, `
MATCH (a:Country)-[r:BORDERS]->(b:Country)
RETURN a.id AS a, b.id AS b, r.start_time_step AS start, r.end_time_step AS end`, nil)
	if err != nil {
		return 0, fmt.Errorf("read borders: %w", err)
	}
	for _, r := range borders {
		if err := neo4jdb.Write(ctx, agg, `
MATCH (a:Country {id: $a}), (b:Country {id: $b})
MERGE (a)-[r:BORDERS]->(b)
SET r.start_time_step = $start, r.end_time_step = $end`, r); err != nil {
			return 0, fmt.Errorf("mirror border: %w", err)
		}
		count++
	}
	// MEMBER_OF (Country->Actor).
	members, err := neo4jdb.ReadRows(ctx, raw, `
MATCH (m:Country)-[r:MEMBER_OF]->(a:Actor)
RETURN m.id AS m, a.id AS a, r.start_time_step AS start, r.end_time_step AS end`, nil)
	if err != nil {
		return 0, fmt.Errorf("read memberships: %w", err)
	}
	for _, r := range members {
		if err := neo4jdb.Write(ctx, agg, `
MATCH (m:Country {id: $m}), (a:Actor {id: $a})
MERGE (m)-[r:MEMBER_OF]->(a)
SET r.start_time_step = $start, r.end_time_step = $end`, r); err != nil {
			return 0, fmt.Errorf("mirror membership: %w", err)
		}
		count++
	}
	return count, nil
}

func maxTimeStep(ctx context.Context, raw neo4j.DriverWithContext) (int, error) {
	fmax, err := neo4jdb.Count(ctx, raw,
		"MATCH (s:FeatureSnapshot) RETURN coalesce(max(s.time_step), 0)", nil)
	if err != nil {
		return 0, fmt.Errorf("max feature ts: %w", err)
	}
	emax, err := neo4jdb.Count(ctx, raw,
		"MATCH ()-[e:EVENT]->() RETURN coalesce(max(e.time_step), 0)", nil)
	if err != nil {
		return 0, fmt.Errorf("max event ts: %w", err)
	}
	if emax > fmax {
		return int(emax), nil
	}
	return int(fmax), nil
}

// buildNodeSnapshots forward-fills each identity node's FeatureSnapshots across
// every month [0, maxTS] and writes one NodeSnapshot per (node, month).
func buildNodeSnapshots(ctx context.Context, raw, agg neo4j.DriverWithContext, identities []identity, maxTS int) (int, error) {
	rows, err := neo4jdb.ReadRows(ctx, raw, `
MATCH (s:FeatureSnapshot)
RETURN s.node_id AS node_id, s.time_step AS time_step, properties(s) AS props`, nil)
	if err != nil {
		return 0, fmt.Errorf("read feature snapshots: %w", err)
	}

	// node_id -> time_step -> feature map (meta stripped)
	byNode := map[string]map[int]map[string]any{}
	for _, r := range rows {
		nodeID, _ := r["node_id"].(string)
		ts := int(getInt64(r["time_step"]))
		props, _ := r["props"].(map[string]any)
		feats := map[string]any{}
		for k, v := range props {
			if !featureMeta[k] {
				feats[k] = v
			}
		}
		if byNode[nodeID] == nil {
			byNode[nodeID] = map[int]map[string]any{}
		}
		byNode[nodeID][ts] = feats
	}

	written := 0
	const cypher = `
MATCH (n {id: $id})
MERGE (n)-[:HAS_SNAPSHOT]->(ns:NodeSnapshot {node_id: $id, time_step: $ts})
SET ns.node_type = $type, ns += $features`
	for _, ident := range identities {
		series := byNode[ident.id]
		static := ident.staticProps()
		current := map[string]any{}
		for ts := 0; ts <= maxTS; ts++ {
			if feats, ok := series[ts]; ok {
				// carry forward: overlay newly-observed features onto current
				for k, v := range feats {
					current[k] = v
				}
			}
			merged := map[string]any{}
			for k, v := range current {
				merged[k] = v
			}
			for k, v := range static {
				merged[k] = v
			}
			params := map[string]any{
				"id":       ident.id,
				"ts":       ts,
				"type":     ident.label,
				"features": merged,
			}
			if err := neo4jdb.Write(ctx, agg, cypher, params); err != nil {
				return written, fmt.Errorf("write nodesnapshot %s@%d: %w", ident.id, ts, err)
			}
			written++
		}
	}
	return written, nil
}

// buildSnapshotEdges groups raw EVENTs by (source, target, time_step) and writes
// one aggregated SNAPSHOT_EDGE per group.
func buildSnapshotEdges(ctx context.Context, raw, agg neo4j.DriverWithContext) (int, error) {
	rows, err := neo4jdb.ReadRows(ctx, raw, `
MATCH (src:Country)-[e:EVENT]->(tgt:Country)
RETURN src.id AS source, tgt.id AS target, e.time_step AS time_step,
       e.relationship_class AS relationship_class, e.intensity_score AS intensity_score,
       e.sentiment_score AS sentiment_score, e.source_count AS source_count`, nil)
	if err != nil {
		return 0, fmt.Errorf("read events: %w", err)
	}

	type key struct {
		src, tgt string
		ts       int
	}
	groups := map[key][]map[string]any{}
	dyadMonths := map[string][]int{} // "src|tgt" -> sorted unique ts list
	for _, r := range rows {
		k := key{
			src: r["source"].(string),
			tgt: r["target"].(string),
			ts:  int(getInt64(r["time_step"])),
		}
		groups[k] = append(groups[k], r)
	}
	for k := range groups {
		dyad := k.src + "|" + k.tgt
		dyadMonths[dyad] = append(dyadMonths[dyad], k.ts)
	}
	for dyad := range dyadMonths {
		sort.Ints(dyadMonths[dyad])
	}

	written := 0
	const cypher = `
MATCH (src:Country {id: $source}), (tgt:Country {id: $target})
MERGE (src)-[r:SNAPSHOT_EDGE {time_step: $time_step}]->(tgt)
SET r += $props`
	for k, evts := range groups {
		props := aggregateGroup(evts)
		props["days_since_last_event"] = daysSinceLast(dyadMonths[k.src+"|"+k.tgt], k.ts)
		params := map[string]any{
			"source":    k.src,
			"target":    k.tgt,
			"time_step": k.ts,
			"props":     props,
		}
		if err := neo4jdb.Write(ctx, agg, cypher, params); err != nil {
			return written, fmt.Errorf("write snapshot edge %s->%s@%d: %w", k.src, k.tgt, k.ts, err)
		}
		written++
	}
	return written, nil
}

// aggregateGroup computes the SNAPSHOT_EDGE feature bundle for one dyad-month.
func aggregateGroup(evts []map[string]any) map[string]any {
	n := len(evts)
	counts := make([]int, len(label.Classes))
	var weightedNum, weightSum, sentSum float64
	sentiments := make([]float64, 0, n)
	hasMaterialConflict := false

	for _, e := range evts {
		cls, _ := e["relationship_class"].(string)
		if idx := label.Index(cls); idx >= 0 {
			counts[idx]++
		}
		if cls == label.MaterialConflict {
			hasMaterialConflict = true
		}
		intensity := getFloat(e["intensity_score"])
		sources := float64(getInt64(e["source_count"]))
		weightedNum += intensity * sources
		weightSum += sources
		s := getFloat(e["sentiment_score"])
		sentSum += s
		sentiments = append(sentiments, s)
	}

	dist := make([]any, len(label.Classes))
	for i, c := range counts {
		dist[i] = float64(c) / float64(n)
	}

	weightedIntensity := 0.0
	if weightSum > 0 {
		weightedIntensity = weightedNum / weightSum
	}
	sentMean := sentSum / float64(n)
	sentStd := 0.0
	for _, s := range sentiments {
		sentStd += (s - sentMean) * (s - sentMean)
	}
	sentStd = math.Sqrt(sentStd / float64(n))

	// conflict-priority dominance, else modal class (canonical-order tie-break).
	dominant := label.StatusQuo
	if hasMaterialConflict {
		dominant = label.MaterialConflict
	} else {
		best := -1
		for i, c := range counts {
			if c > best {
				best = c
				dominant = label.Classes[i]
			}
		}
	}

	transition := make([]any, 25)
	for i := range transition {
		transition[i] = 0.0
	}

	return map[string]any{
		"event_count":             n,
		"weighted_intensity":      weightedIntensity,
		"sentiment_mean":          sentMean,
		"sentiment_std":           sentStd,
		"dominant_class":          dominant,
		"class_distribution":      dist,
		"class_transition_vector": transition,
	}
}

// daysSinceLast approximates the gap (in days, ~30/month) to the previous month
// in which this dyad had events; 0 for the dyad's first active month.
func daysSinceLast(months []int, ts int) int {
	prev := -1
	for _, m := range months {
		if m < ts && m > prev {
			prev = m
		}
	}
	if prev < 0 {
		return 0
	}
	return (ts - prev) * 30
}

func getInt64(v any) int64 {
	switch n := v.(type) {
	case int64:
		return n
	case int:
		return int64(n)
	case float64:
		return int64(n)
	default:
		return 0
	}
}

func getFloat(v any) float64 {
	switch f := v.(type) {
	case float64:
		return f
	case int64:
		return float64(f)
	case int:
		return float64(f)
	default:
		return 0
	}
}
