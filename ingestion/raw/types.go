// Package raw populates the geopolitic_raw database: Country/Actor identity
// nodes, FeatureSnapshot nodes, temporal BORDERS/MEMBER_OF edges, and EVENT
// edges. See plans/02-data-ingestion.md.
//
// The country universe and all structural edges come from live sources
// (World Bank country list + Wikidata SPARQL); there is no synthetic data.
package raw

// Country is one sovereign state (from the World Bank country list).
type Country struct {
	ISO3   string
	ISO2   string
	Name   string
	Region string
}

// Event is a normalized EVENT edge ready to be written to geopolitic_raw.
type Event struct {
	EventID           string
	Source            string // ISO-3
	Target            string // ISO-3
	TimeStep          int
	Timestamp         string // YYYY-MM-DD
	RelationshipClass string
	IntensityScore    float64
	SentimentScore    float64
	SourceCount       int
	GoldsteinScale    float64
	EventType         string // raw CAMEO code
	DataSource        string // "GDELT" | "ICEWS"
}

// toRow flattens an Event into a Cypher-friendly parameter map.
func (e Event) toRow() map[string]any {
	return map[string]any{
		"event_id":           e.EventID,
		"source":             e.Source,
		"target":             e.Target,
		"time_step":          e.TimeStep,
		"timestamp":          e.Timestamp,
		"relationship_class": e.RelationshipClass,
		"intensity_score":    e.IntensityScore,
		"sentiment_score":    e.SentimentScore,
		"source_count":       e.SourceCount,
		"goldstein_scale":    e.GoldsteinScale,
		"event_type":         e.EventType,
		"data_source":        e.DataSource,
	}
}
