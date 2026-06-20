package aggregated

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"

	"geopolitic/internal/label"
	"geopolitic/internal/neo4jdb"

	"cloud.google.com/go/bigquery"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"google.golang.org/api/iterator"
)

// defaultEndDate is today as an int YYYYMMDD — the GDELT window upper bound when
// GDELT_END_DATE is unset.
func defaultEndDate() int {
	now := time.Now().UTC()
	return now.Year()*10000 + int(now.Month())*100 + now.Day()
}

// GDELT is far too large to load as raw EVENT edges on a local box (tens of
// millions of edges, 16-48 GB). Instead we aggregate the full 2010-present event
// set *in BigQuery* into one row per (source, target, month) and load only those
// compact SNAPSHOT_EDGEs into geopolitic_aggregated. The raw events stay in
// BigQuery as the regenerable source of truth. This mirrors the per-dyad-month
// aggregation that aggregate.go does for raw Neo4j EVENTs, so the SNAPSHOT_EDGE
// schema is identical (minus the never-computed class_transition_vector, dropped
// here to save ~600 MB across ~2.4M edges). See plans/02-data-ingestion.md.

// gdeltSnapV1Table / gdeltSnapV2Table mirror the table split in raw/gdelt.go.
const (
	gdeltSnapV1Table    = "gdelt-bq.full.events"
	gdeltSnapV2Table    = "gdelt-bq.gdeltv2.events"
	gdeltSnapV2MinDate  = 20150218
	gdeltSnapWriteBatch = 2000
)

// gdeltAggRow is one aggregated (src, tgt, month) group returned by BigQuery.
// Sums (not finished averages) are returned so groups that collapse together
// after ISO-3 alias remapping (ROM->ROU, MTN->MNE, IMY->IMN) can be combined
// exactly in Go.
type gdeltAggRow struct {
	Src        string  `bigquery:"src"`
	Tgt        string  `bigquery:"tgt"`
	TS         int64   `bigquery:"ts"`
	EventCount int64   `bigquery:"event_count"`
	SumIntSrc  float64 `bigquery:"sum_int_src"`
	SumSrc     float64 `bigquery:"sum_src"`
	SumSent    float64 `bigquery:"sum_sent"`
	SumSentSq  float64 `bigquery:"sum_sent_sq"`
	C0         int64   `bigquery:"c0"`
	C1         int64   `bigquery:"c1"`
	C2         int64   `bigquery:"c2"`
	C3         int64   `bigquery:"c3"`
	C4         int64   `bigquery:"c4"`
}

type seKey struct {
	src, tgt string
	ts       int
}

type seAcc struct {
	n                                     int
	sumIntSrc, sumSrc, sumSent, sumSentSq float64
	c                                     [5]int
}

// GDELTMaxTimeStep returns the time_step of the GDELT window end, so Build can
// forward-fill NodeSnapshots across every month that has SNAPSHOT_EDGEs (events
// run to ~present, later than the latest annual feature observation).
func GDELTMaxTimeStep(end string) int {
	return tsFromDateInt(dateOnly(end, defaultEndDate()))
}

// LoadGDELTSnapshotEdges queries both GDELT tables, aggregates per (src,tgt,month)
// server-side, remaps/combines ISO-3 codes via cameoMap, and writes SNAPSHOT_EDGEs
// into geopolitic_aggregated. start/end are "YYYY-MM-DD" (end "" = today).
// maxRows caps rows PER SEGMENT (0 = unbounded).
func LoadGDELTSnapshotEdges(ctx context.Context, aggDriver neo4j.DriverWithContext, project, start, end string, maxRows int, cameoMap map[string]string) (int, error) {
	return loadGDELTSnapshotEdges(ctx, aggDriver, project, start, end, maxRows, cameoMap, "")
}

// LoadGDELTSnapshotEdgesForCode is LoadGDELTSnapshotEdges restricted to dyads where
// focus (an ISO-3 actor code) is the source or target. Because a newly-added
// country shares no SNAPSHOT_EDGEs with the existing set, this surgically adds its
// edges via MERGE without re-scanning/re-writing the rest. Used by
// tools/add_gdelt_country.go when the cameo map gains a country.
func LoadGDELTSnapshotEdgesForCode(ctx context.Context, aggDriver neo4j.DriverWithContext, project, start, end string, maxRows int, cameoMap map[string]string, focus string) (int, error) {
	return loadGDELTSnapshotEdges(ctx, aggDriver, project, start, end, maxRows, cameoMap, focus)
}

func loadGDELTSnapshotEdges(ctx context.Context, aggDriver neo4j.DriverWithContext, project, start, end string, maxRows int, cameoMap map[string]string, focus string) (int, error) {
	startI := dateOnly(start, 20100101)
	endI := dateOnly(end, defaultEndDate())

	client, err := bigquery.NewClient(ctx, project)
	if err != nil {
		return 0, fmt.Errorf("bigquery client: %w", err)
	}
	defer client.Close()

	// Restrict server-side to the known country codes so regional aggregates
	// (EUR/AFR/WSB...) never leave BigQuery.
	codes := make([]string, 0, len(cameoMap))
	for k := range cameoMap {
		codes = append(codes, k)
	}

	acc := map[seKey]*seAcc{}

	// pre-2015 segment from full.events
	if startI < gdeltSnapV2MinDate {
		hi := endI
		if hi >= gdeltSnapV2MinDate {
			hi = gdeltSnapV2MinDate - 1
		}
		if err := foldGDELT(ctx, client, gdeltSnapV1Table, startI, hi, maxRows, codes, focus, cameoMap, acc); err != nil {
			return 0, err
		}
	}
	// 2015+ segment from gdeltv2.events
	if endI >= gdeltSnapV2MinDate {
		lo := startI
		if lo < gdeltSnapV2MinDate {
			lo = gdeltSnapV2MinDate
		}
		if err := foldGDELT(ctx, client, gdeltSnapV2Table, lo, endI, maxRows, codes, focus, cameoMap, acc); err != nil {
			return 0, err
		}
	}

	return writeGDELTSnapshotEdges(ctx, aggDriver, acc)
}

// foldGDELT runs the aggregation query for one table and folds its groups into
// acc, remapping ISO-3 aliases so collapsed dyads combine exactly. When focus is
// non-empty, only dyads where focus is the source or target are aggregated.
func foldGDELT(ctx context.Context, client *bigquery.Client, table string, start, end, maxRows int, codes []string, focus string, cameoMap map[string]string, acc map[seKey]*seAcc) error {
	focusClause := ""
	if focus != "" {
		focusClause = "    AND (Actor1CountryCode = @focus OR Actor2CountryCode = @focus)\n"
	}
	sql := fmt.Sprintf(`
SELECT src, tgt, ts,
       COUNT(*)                       AS event_count,
       SUM(intensity * sources)       AS sum_int_src,
       SUM(sources)                   AS sum_src,
       SUM(sentiment)                 AS sum_sent,
       SUM(sentiment * sentiment)     AS sum_sent_sq,
       COUNTIF(cls = 0) AS c0, COUNTIF(cls = 1) AS c1, COUNTIF(cls = 2) AS c2,
       COUNTIF(cls = 3) AS c3, COUNTIF(cls = 4) AS c4
FROM (
  SELECT
    Actor1CountryCode AS src,
    Actor2CountryCode AS tgt,
    (CAST(SUBSTR(CAST(SQLDATE AS STRING),1,4) AS INT64) - 2010) * 12
      + (CAST(SUBSTR(CAST(SQLDATE AS STRING),5,2) AS INT64) - 1) AS ts,
    CAST(IFNULL(NumArticles, 0) AS FLOAT64)                 AS sources,
    IFNULL(GoldsteinScale, 0) / 10.0                        AS intensity,
    GREATEST(-1.0, LEAST(1.0, IFNULL(AvgTone, 0) / 10.0))   AS sentiment,
    CASE
      WHEN SAFE_CAST(SUBSTR(EventCode,1,2) AS INT64) BETWEEN 18 AND 20 THEN 0
      WHEN GoldsteinScale < -5                                         THEN 0
      WHEN SAFE_CAST(SUBSTR(EventCode,1,2) AS INT64) BETWEEN 11 AND 17 THEN 1
      WHEN SAFE_CAST(SUBSTR(EventCode,1,2) AS INT64) BETWEEN 6  AND 8  THEN 2
      WHEN SAFE_CAST(SUBSTR(EventCode,1,2) AS INT64) BETWEEN 1  AND 5  THEN 3
      WHEN GoldsteinScale > 5                                          THEN 3
      ELSE 4
    END AS cls
  FROM `+"`%s`"+`
  WHERE SQLDATE BETWEEN @start AND @end
    AND Actor1CountryCode IN UNNEST(@codes)
    AND Actor2CountryCode IN UNNEST(@codes)
    AND Actor1CountryCode != Actor2CountryCode
%s)
GROUP BY src, tgt, ts`, table, focusClause)
	if maxRows > 0 {
		sql += fmt.Sprintf("\nLIMIT %d", maxRows)
	}

	q := client.Query(sql)
	q.Parameters = []bigquery.QueryParameter{
		{Name: "start", Value: start},
		{Name: "end", Value: end},
		{Name: "codes", Value: codes},
	}
	if focus != "" {
		q.Parameters = append(q.Parameters, bigquery.QueryParameter{Name: "focus", Value: focus})
	}
	it, err := q.Read(ctx)
	if err != nil {
		return fmt.Errorf("query %s: %w", table, err)
	}
	for {
		var r gdeltAggRow
		err := it.Next(&r)
		if err == iterator.Done {
			break
		}
		if err != nil {
			return fmt.Errorf("iterate %s: %w", table, err)
		}
		src, ok := cameoMap[r.Src]
		if !ok || src == "" {
			continue
		}
		tgt, ok := cameoMap[r.Tgt]
		if !ok || tgt == "" || src == tgt {
			continue
		}
		k := seKey{src: src, tgt: tgt, ts: int(r.TS)}
		a := acc[k]
		if a == nil {
			a = &seAcc{}
			acc[k] = a
		}
		a.n += int(r.EventCount)
		a.sumIntSrc += r.SumIntSrc
		a.sumSrc += r.SumSrc
		a.sumSent += r.SumSent
		a.sumSentSq += r.SumSentSq
		a.c[0] += int(r.C0)
		a.c[1] += int(r.C1)
		a.c[2] += int(r.C2)
		a.c[3] += int(r.C3)
		a.c[4] += int(r.C4)
	}
	return nil
}

// writeGDELTSnapshotEdges finalizes each accumulator into a SNAPSHOT_EDGE and
// writes them in UNWIND batches. days_since_last_event is derived per dyad.
func writeGDELTSnapshotEdges(ctx context.Context, aggDriver neo4j.DriverWithContext, acc map[seKey]*seAcc) (int, error) {
	// per-dyad sorted active months for days_since_last_event.
	dyadMonths := map[string][]int{}
	for k := range acc {
		d := k.src + "|" + k.tgt
		dyadMonths[d] = append(dyadMonths[d], k.ts)
	}
	for d := range dyadMonths {
		sort.Ints(dyadMonths[d])
	}

	const cypher = `
UNWIND $rows AS row
MATCH (src:Country {id: row.source})
MATCH (tgt:Country {id: row.target})
MERGE (src)-[r:SNAPSHOT_EDGE {time_step: row.time_step}]->(tgt)
SET r += row.props`

	written := 0
	batch := make([]any, 0, gdeltSnapWriteBatch)
	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		if err := neo4jdb.Write(ctx, aggDriver, cypher, map[string]any{"rows": batch}); err != nil {
			return err
		}
		written += len(batch)
		batch = batch[:0]
		return nil
	}

	for k, a := range acc {
		n := float64(a.n)
		weightedIntensity := 0.0
		if a.sumSrc > 0 {
			weightedIntensity = a.sumIntSrc / a.sumSrc
		}
		mean := a.sumSent / n
		variance := a.sumSentSq/n - mean*mean
		if variance < 0 {
			variance = 0
		}
		dist := make([]any, 5)
		for i := 0; i < 5; i++ {
			dist[i] = float64(a.c[i]) / n
		}
		dominant := label.StatusQuo
		if a.c[0] > 0 {
			dominant = label.MaterialConflict
		} else {
			best := -1
			for i, c := range a.c {
				if c > best {
					best = c
					dominant = label.Classes[i]
				}
			}
		}
		batch = append(batch, map[string]any{
			"source":    k.src,
			"target":    k.tgt,
			"time_step": k.ts,
			"props": map[string]any{
				"event_count":           a.n,
				"weighted_intensity":    weightedIntensity,
				"sentiment_mean":        mean,
				"sentiment_std":         math.Sqrt(variance),
				"dominant_class":        dominant,
				"class_distribution":    dist,
				"days_since_last_event": daysSinceLast(dyadMonths[k.src+"|"+k.tgt], k.ts),
				"data_source":           "GDELT",
			},
		})
		if len(batch) >= gdeltSnapWriteBatch {
			if err := flush(); err != nil {
				return written, err
			}
		}
	}
	if err := flush(); err != nil {
		return written, err
	}
	return written, nil
}

// dateOnly turns "YYYY-MM-DD" into int YYYYMMDD, or def when blank/invalid.
func dateOnly(s string, def int) int {
	d := []rune{}
	for _, r := range s {
		if r >= '0' && r <= '9' {
			d = append(d, r)
		}
	}
	if len(d) < 8 {
		return def
	}
	n := 0
	for _, r := range d[:8] {
		n = n*10 + int(r-'0')
	}
	return n
}

// tsFromDateInt converts an int YYYYMMDD to a time_step.
func tsFromDateInt(d int) int {
	year := d / 10000
	month := (d / 100) % 100
	if month < 1 {
		month = 1
	}
	return (year-2010)*12 + (month - 1)
}
