package raw

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"

	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/xuri/excelize/v2"
)

// SIPRIRecord is one (country, year) military-expenditure observation.
type SIPRIRecord struct {
	ISO3     string
	Year     int
	MilexLog float64
}

// ParseSIPRIFile reads the SIPRI Milex "Constant US$" sheet (rows = countries,
// columns = years) and returns military_expenditure_log records. The sheet has
// title/notes rows before the data; the header is detected as the first row
// containing several 4-digit years. Country names are mapped to ISO-3 via nm;
// unmatched names and non-numeric cells (".."/"xxx"/"-") are skipped.
func ParseSIPRIFile(f *excelize.File, sheet string, nm NameMap) ([]SIPRIRecord, error) {
	if sheet == "" {
		for _, s := range f.GetSheetList() {
			if strings.Contains(strings.ToLower(s), "constant") {
				sheet = s
				break
			}
		}
	}
	if sheet == "" {
		return nil, fmt.Errorf("no Constant-US$ sheet found; set SIPRI_SHEET")
	}
	rows, err := f.GetRows(sheet)
	if err != nil {
		return nil, err
	}

	headerIdx, yearCols := findYearHeader(rows)
	if headerIdx < 0 {
		return nil, fmt.Errorf("could not locate a year header row in sheet %q", sheet)
	}

	var out []SIPRIRecord
	for i := headerIdx + 1; i < len(rows); i++ {
		row := rows[i]
		if len(row) == 0 {
			continue
		}
		name := strings.TrimSpace(row[0])
		if name == "" {
			continue
		}
		iso3, ok := nm.Lookup(name)
		if !ok {
			continue
		}
		for col, year := range yearCols {
			if col >= len(row) || year < timestep.Epoch {
				continue
			}
			val, ok := parseSIPRIValue(row[col])
			if !ok {
				continue
			}
			out = append(out, SIPRIRecord{ISO3: iso3, Year: year, MilexLog: math.Log(val + 1)})
		}
	}
	return out, nil
}

// LoadSIPRI opens the SIPRI xlsx at path and writes military_expenditure_log
// FeatureSnapshots. sheet may be "" to auto-detect the Constant-US$ sheet.
func LoadSIPRI(ctx context.Context, driver neo4j.DriverWithContext, path, sheet string, nm NameMap) (int, error) {
	f, err := excelize.OpenFile(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	records, err := ParseSIPRIFile(f, sheet, nm)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, rec := range records {
		if err := MergeFeatures(ctx, driver, "Country", rec.ISO3, timestep.FromYear(rec.Year), rec.Year, map[string]any{"military_expenditure_log": rec.MilexLog}); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

// findYearHeader returns the index of the first row containing >=5 four-digit
// year cells, plus a map of column index -> year.
func findYearHeader(rows [][]string) (int, map[int]int) {
	for i, row := range rows {
		cols := map[int]int{}
		for j, cell := range row {
			if y, err := strconv.Atoi(strings.TrimSpace(cell)); err == nil && y >= 1949 && y <= 2100 {
				cols[j] = y
			}
		}
		if len(cols) >= 5 {
			return i, cols
		}
	}
	return -1, nil
}

func parseSIPRIValue(s string) (float64, bool) {
	s = strings.TrimSpace(strings.ReplaceAll(s, ",", ""))
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}
