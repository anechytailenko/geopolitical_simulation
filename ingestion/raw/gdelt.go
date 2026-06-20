package raw

import (
	"context"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"cloud.google.com/go/bigquery"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"google.golang.org/api/iterator"
)

// LoadCameoMap reads configs/cameo_country_to_iso3.json (GDELT CAMEO/FIPS → ISO-3).
func LoadCameoMap(configDir string) (map[string]string, error) {
	var m map[string]string
	if err := readJSON(filepath.Join(configDir, "cameo_country_to_iso3.json"), &m); err != nil {
		return nil, err
	}
	return m, nil
}

// gdeltV2MinDate is the first SQLDATE present in the gdeltv2.events table
// (2015-02-18). Earlier dates come from the GDELT 1.0 full.events table.
const gdeltV2MinDate = 20150218

// GDELT BigQuery table IDs. The 2010-2015 segment comes from gdelt-bq.full.events
// (the GDELT 1.0 historical events table); gdelt-bq.gdeltv1.events does NOT exist.
// 2015+ comes from gdelt-bq.gdeltv2.events. Both share the same column layout.
const (
	gdeltV1Table = "gdelt-bq.full.events"
	gdeltV2Table = "gdelt-bq.gdeltv2.events"
)

// gdeltWriteBatch is how many EVENT edges are flushed to Neo4j per round trip.
const gdeltWriteBatch = 5000

// gdeltRow is one BigQuery result row. Nullable types guard against NULL cells.
type gdeltRow struct {
	GlobalEventID int64                `bigquery:"GLOBALEVENTID"`
	SQLDATE       int64                `bigquery:"SQLDATE"`
	Actor1        bigquery.NullString  `bigquery:"Actor1CountryCode"`
	Actor2        bigquery.NullString  `bigquery:"Actor2CountryCode"`
	EventCode     bigquery.NullString  `bigquery:"EventCode"`
	Goldstein     bigquery.NullFloat64 `bigquery:"GoldsteinScale"`
	NumArticles   bigquery.NullInt64   `bigquery:"NumArticles"`
	AvgTone       bigquery.NullFloat64 `bigquery:"AvgTone"`
}

// gdeltRowToEvent maps a BigQuery row to a normalized Event (pure; unit-tested).
func gdeltRowToEvent(r gdeltRow, cameoMap map[string]string) (Event, bool) {
	day := fmt.Sprintf("%08d", r.SQLDATE)
	return BuildEvent(
		strconv.FormatInt(r.GlobalEventID, 10),
		day,
		r.Actor1.StringVal, r.Actor2.StringVal, r.EventCode.StringVal,
		r.Goldstein.Float64, r.AvgTone.Float64, int(r.NumArticles.Int64),
		"GDELT", cameoMap,
	)
}

// LoadGDELT streams country↔country events from GDELT in BigQuery (gdeltv1 for
// the pre-2015 segment, gdeltv2 from 2015 on), filtered server-side, converts
// CAMEO codes to ISO-3, and writes EVENT edges. startDate/endDate are
// "YYYY-MM-DD" (endDate "" = no upper bound). maxRows caps rows PER SEGMENT
// (0 = unbounded) — handy for a smoke run. Auth uses Application Default
// Credentials for the given GCP project.
func LoadGDELT(ctx context.Context, driver neo4j.DriverWithContext, project, startDate, endDate string, maxRows int, cameoMap map[string]string) (int, error) {
	start := dateInt(startDate, 20100101)
	end := dateInt(endDate, 99999999)

	client, err := bigquery.NewClient(ctx, project)
	if err != nil {
		return 0, fmt.Errorf("bigquery client: %w", err)
	}
	defer client.Close()

	total := 0
	// pre-2015 segment from gdeltv1.events
	if start < gdeltV2MinDate {
		hi := end
		if hi >= gdeltV2MinDate {
			hi = gdeltV2MinDate - 1
		}
		n, err := streamGDELT(ctx, client, driver, gdeltV1Table, start, hi, maxRows, cameoMap)
		total += n
		if err != nil {
			return total, err
		}
	}
	// 2015+ segment from gdeltv2.events
	if end >= gdeltV2MinDate {
		lo := start
		if lo < gdeltV2MinDate {
			lo = gdeltV2MinDate
		}
		n, err := streamGDELT(ctx, client, driver, gdeltV2Table, lo, end, maxRows, cameoMap)
		total += n
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

func streamGDELT(ctx context.Context, client *bigquery.Client, driver neo4j.DriverWithContext, table string, start, end, maxRows int, cameoMap map[string]string) (int, error) {
	sql := fmt.Sprintf(`
SELECT GLOBALEVENTID, SQLDATE, Actor1CountryCode, Actor2CountryCode,
       EventCode, GoldsteinScale, NumArticles, AvgTone
FROM `+"`%s`"+`
WHERE SQLDATE BETWEEN @start AND @end
  AND Actor1CountryCode IS NOT NULL AND Actor1CountryCode != ''
  AND Actor2CountryCode IS NOT NULL AND Actor2CountryCode != ''
  AND Actor1CountryCode != Actor2CountryCode`, table)
	if maxRows > 0 {
		sql += fmt.Sprintf("\nLIMIT %d", maxRows)
	}
	q := client.Query(sql)
	q.Parameters = []bigquery.QueryParameter{
		{Name: "start", Value: start},
		{Name: "end", Value: end},
	}
	it, err := q.Read(ctx)
	if err != nil {
		return 0, fmt.Errorf("query %s: %w", table, err)
	}

	written := 0
	batch := make([]Event, 0, gdeltWriteBatch)
	flush := func() error {
		n, err := WriteEvents(ctx, driver, batch)
		written += n
		batch = batch[:0]
		return err
	}
	for {
		var row gdeltRow
		err := it.Next(&row)
		if err == iterator.Done {
			break
		}
		if err != nil {
			return written, fmt.Errorf("iterate %s: %w", table, err)
		}
		if ev, ok := gdeltRowToEvent(row, cameoMap); ok {
			batch = append(batch, ev)
			if len(batch) >= gdeltWriteBatch {
				if err := flush(); err != nil {
					return written, err
				}
			}
		}
	}
	if err := flush(); err != nil {
		return written, err
	}
	return written, nil
}

// dateInt turns "YYYY-MM-DD" into an int YYYYMMDD, or def when blank/invalid.
func dateInt(s string, def int) int {
	s = strings.ReplaceAll(strings.TrimSpace(s), "-", "")
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return n
}
