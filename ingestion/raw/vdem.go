package raw

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"

	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// VDemRecord is one (country, year) row after leadership-change derivation.
type VDemRecord struct {
	ISO3                       string
	Year                       int
	Polyarchy                  float64
	HasPolyarchy               bool
	YearsSinceLeadershipChange int
}

// ParseVDem reads a V-Dem country-year CSV and returns per-(country,year) records
// with years_since_leadership_change derived from consecutive v2exnamhos values.
// country_text_id IS the ISO-3 code (no Correlates-of-War mapping needed).
func ParseVDem(r io.Reader) ([]VDemRecord, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1
	header, err := cr.Read()
	if err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	col := map[string]int{}
	for i, h := range header {
		col[h] = i
	}
	need := []string{"country_text_id", "year", "v2x_polyarchy", "v2exnamhos"}
	for _, c := range need {
		if _, ok := col[c]; !ok {
			return nil, fmt.Errorf("missing V-Dem column %q", c)
		}
	}

	type row struct {
		year      int
		polyarchy float64
		hasPoly   bool
		hos       string
	}
	byCountry := map[string][]row{}
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		iso3 := rec[col["country_text_id"]]
		year, err := strconv.Atoi(rec[col["year"]])
		if iso3 == "" || err != nil || year < timestep.Epoch {
			continue
		}
		r := row{year: year, hos: rec[col["v2exnamhos"]]}
		if v, err := strconv.ParseFloat(rec[col["v2x_polyarchy"]], 64); err == nil {
			r.polyarchy = v
			r.hasPoly = true
		}
		byCountry[iso3] = append(byCountry[iso3], r)
	}

	var out []VDemRecord
	for iso3, rows := range byCountry {
		sort.Slice(rows, func(i, j int) bool { return rows[i].year < rows[j].year })
		lastHOS := ""
		since := 0
		for i, r := range rows {
			switch {
			case i == 0 || r.hos != lastHOS:
				since = 0
			default:
				since++
			}
			lastHOS = r.hos
			out = append(out, VDemRecord{
				ISO3:                       iso3,
				Year:                       r.year,
				Polyarchy:                  r.polyarchy,
				HasPolyarchy:               r.hasPoly,
				YearsSinceLeadershipChange: since,
			})
		}
	}
	return out, nil
}

// LoadVDem parses the V-Dem CSV at path and writes vdem_polyarchy_score +
// years_since_leadership_change FeatureSnapshots. Returns the number of writes.
func LoadVDem(ctx context.Context, driver neo4j.DriverWithContext, path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	records, err := ParseVDem(f)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, rec := range records {
		feats := map[string]any{"years_since_leadership_change": rec.YearsSinceLeadershipChange}
		if rec.HasPolyarchy {
			feats["vdem_polyarchy_score"] = rec.Polyarchy
		}
		if err := MergeFeatures(ctx, driver, "Country", rec.ISO3, timestep.FromYear(rec.Year), rec.Year, feats); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}
