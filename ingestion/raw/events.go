package raw

import (
	"context"
	"fmt"
	"strconv"

	"geopolitic/internal/label"
	"geopolitic/internal/neo4jdb"
	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// BuildEvent normalizes one raw CAMEO-coded event (GDELT or ICEWS shape) into an
// Event, converting actor country codes to ISO-3 via cameoMap. Returns
// (_, false) when the row must be dropped (unknown country or unparseable date).
//
// day is "YYYYMMDD"; goldstein is the Goldstein scale; avgTone is GDELT AvgTone
// (or 0 for ICEWS); numArticles is the source-count.
func BuildEvent(eventID, day, actor1Code, actor2Code, cameoCode string, goldstein, avgTone float64, numArticles int, dataSource string, cameoMap map[string]string) (Event, bool) {
	src, ok := cameoMap[actor1Code]
	if !ok || src == "" {
		return Event{}, false
	}
	tgt, ok := cameoMap[actor2Code]
	if !ok || tgt == "" {
		return Event{}, false
	}
	if src == tgt {
		return Event{}, false // self-loops are not country->country dyads
	}
	if len(day) != 8 {
		return Event{}, false
	}
	year, err1 := strconv.Atoi(day[:4])
	month, err2 := strconv.Atoi(day[4:6])
	if err1 != nil || err2 != nil || month < 1 || month > 12 || year < timestep.Epoch {
		return Event{}, false
	}
	ts := timestep.FromYM(year, month)

	return Event{
		EventID:           eventID,
		Source:            src,
		Target:            tgt,
		TimeStep:          ts,
		Timestamp:         fmt.Sprintf("%s-%s-%s", day[:4], day[4:6], day[6:8]),
		RelationshipClass: label.Classify(cameoCode, goldstein),
		IntensityScore:    goldstein / 10,
		SentimentScore:    clamp(avgTone/10, -1, 1),
		SourceCount:       numArticles,
		GoldsteinScale:    goldstein,
		EventType:         cameoCode,
		DataSource:        dataSource,
	}, true
}

// WriteEvents writes a batch of normalized EVENT edges in a single UNWIND round
// trip, deduplicating on event_id via MERGE.
func WriteEvents(ctx context.Context, driver neo4j.DriverWithContext, events []Event) (int, error) {
	if len(events) == 0 {
		return 0, nil
	}
	rows := make([]any, 0, len(events))
	for _, e := range events {
		rows = append(rows, e.toRow())
	}
	const cypher = `
UNWIND $rows AS row
MERGE (src:Country {id: row.source})
MERGE (tgt:Country {id: row.target})
MERGE (src)-[e:EVENT {event_id: row.event_id}]->(tgt)
SET e.relationship_class = row.relationship_class,
    e.timestamp = row.timestamp,
    e.time_step = row.time_step,
    e.intensity_score = row.intensity_score,
    e.sentiment_score = row.sentiment_score,
    e.source_count = row.source_count,
    e.goldstein_scale = row.goldstein_scale,
    e.event_type = row.event_type,
    e.data_source = row.data_source`
	if err := neo4jdb.Write(ctx, driver, cypher, map[string]any{"rows": rows}); err != nil {
		return 0, err
	}
	return len(events), nil
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
