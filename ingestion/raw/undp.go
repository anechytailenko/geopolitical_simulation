package raw

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// UNDPRecord is one (country, year) HDI observation.
type UNDPRecord struct {
	ISO3 string
	Year int
	HDI  float64
}

// ParseUNDP reads the UNDP HDR "composite indices complete time series" CSV,
// which is wide: an `iso3` column plus `hdi_YYYY` columns. Emits one record per
// (country, year) with a present HDI value.
func ParseUNDP(r io.Reader) ([]UNDPRecord, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1
	header, err := cr.Read()
	if err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	isoCol := -1
	hdiYearCol := map[int]int{} // column index -> year
	for i, h := range header {
		h = strings.TrimSpace(strings.ToLower(h))
		if h == "iso3" {
			isoCol = i
			continue
		}
		if strings.HasPrefix(h, "hdi_") {
			if year, err := strconv.Atoi(strings.TrimPrefix(h, "hdi_")); err == nil {
				hdiYearCol[i] = year
			}
		}
	}
	if isoCol < 0 {
		return nil, fmt.Errorf("missing iso3 column in UNDP CSV")
	}
	if len(hdiYearCol) == 0 {
		return nil, fmt.Errorf("no hdi_YYYY columns found in UNDP CSV")
	}

	var out []UNDPRecord
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		iso3 := strings.TrimSpace(rec[isoCol])
		if len(iso3) != 3 {
			continue
		}
		for ci, year := range hdiYearCol {
			if ci >= len(rec) || year < timestep.Epoch {
				continue
			}
			v, err := strconv.ParseFloat(strings.TrimSpace(rec[ci]), 64)
			if err != nil {
				continue
			}
			out = append(out, UNDPRecord{ISO3: strings.ToUpper(iso3), Year: year, HDI: v})
		}
	}
	return out, nil
}

// LoadUNDP parses the UNDP HDI CSV at path and writes hdi FeatureSnapshots.
func LoadUNDP(ctx context.Context, driver neo4j.DriverWithContext, path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	records, err := ParseUNDP(f)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, rec := range records {
		if err := MergeFeatures(ctx, driver, "Country", rec.ISO3, timestep.FromYear(rec.Year), rec.Year, map[string]any{"hdi": rec.HDI}); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}
