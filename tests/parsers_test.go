package tests

import (
	"math"
	"strings"
	"testing"

	"geopolitic/ingestion/raw"
	"geopolitic/internal/timestep"

	"github.com/xuri/excelize/v2"
)

func TestParseVDem(t *testing.T) {
	csv := `country_text_id,year,v2x_polyarchy,v2exnamhos
USA,2010,0.85,Obama
USA,2011,0.86,Obama
USA,2013,0.83,Trump
DEU,2010,0.90,Merkel`
	recs, err := raw.ParseVDem(strings.NewReader(csv))
	if err != nil {
		t.Fatalf("ParseVDem: %v", err)
	}
	since := map[int]int{}    // year -> years_since_leadership_change for USA
	poly := map[int]float64{} // year -> polyarchy for USA
	for _, r := range recs {
		if r.ISO3 == "USA" {
			since[r.Year] = r.YearsSinceLeadershipChange
			poly[r.Year] = r.Polyarchy
		}
	}
	if since[2010] != 0 || since[2011] != 1 || since[2013] != 0 {
		t.Errorf("years_since: 2010=%d 2011=%d 2013=%d, want 0/1/0", since[2010], since[2011], since[2013])
	}
	if poly[2011] != 0.86 {
		t.Errorf("USA polyarchy 2011 = %v, want 0.86", poly[2011])
	}
}

func TestParseUNDP(t *testing.T) {
	csv := `iso3,country,hdi_2010,hdi_2011
USA,United States,0.91,0.92
DEU,Germany,0.93,0.94`
	recs, err := raw.ParseUNDP(strings.NewReader(csv))
	if err != nil {
		t.Fatalf("ParseUNDP: %v", err)
	}
	if len(recs) != 4 {
		t.Fatalf("got %d records, want 4", len(recs))
	}
	for _, r := range recs {
		if r.ISO3 == "USA" && r.Year == 2010 && r.HDI != 0.91 {
			t.Errorf("USA 2010 HDI = %v, want 0.91", r.HDI)
		}
	}
}

func TestAggregateACLED(t *testing.T) {
	nm := raw.BuildNameMap([]raw.Country{{Name: "Nigeria", ISO3: "NGA"}})
	events := []raw.ACLEDEvent{
		{Country: "Nigeria", EventDate: "2014-03-10", EventType: "Battles", Fatalities: 5},
		{Country: "Nigeria", EventDate: "2014-03-20", EventType: "Battles", Fatalities: 3},
		{Country: "Nigeria", EventDate: "2014-03-15", EventType: "Protests", Fatalities: 0}, // non-violent, dropped
		{Country: "Atlantis", EventDate: "2014-03-15", EventType: "Battles", Fatalities: 9}, // unknown country, dropped
	}
	feats := raw.AggregateACLED(events, nm)
	if len(feats) != 1 {
		t.Fatalf("got %d features, want 1", len(feats))
	}
	f := feats[0]
	if f.ISO3 != "NGA" || f.TimeStep != timestep.FromYM(2014, 3) {
		t.Errorf("feature key = %s@%d, want NGA@%d", f.ISO3, f.TimeStep, timestep.FromYM(2014, 3))
	}
	if f.ActiveConflicts != 2 {
		t.Errorf("active_conflict_count = %d, want 2", f.ActiveConflicts)
	}
	if math.Abs(f.ConflictIntensity-math.Log(9)) > 1e-9 { // log(5+3+1)
		t.Errorf("conflict_intensity = %v, want log(9)=%v", f.ConflictIntensity, math.Log(9))
	}
}

func TestParseSIPRIFile(t *testing.T) {
	f := excelize.NewFile()
	sheet := "Constant (2022) US$"
	if _, err := f.NewSheet(sheet); err != nil {
		t.Fatalf("new sheet: %v", err)
	}
	f.SetCellValue(sheet, "A1", "SIPRI Military Expenditure Database — Constant US$")
	header := []any{"Country", 2010, 2011, 2012, 2013, 2014, 2015}
	for i, h := range header {
		cell, _ := excelize.CoordinatesToCellName(i+1, 2)
		f.SetCellValue(sheet, cell, h)
	}
	rows := [][]any{
		{"United States", 700, 710, 720, 730, 740, 750},
		{"Germany", 40, 41, 42, 43, 44, 45},
	}
	for r, row := range rows {
		for c, v := range row {
			cell, _ := excelize.CoordinatesToCellName(c+1, r+3)
			f.SetCellValue(sheet, cell, v)
		}
	}

	nm := raw.BuildNameMap([]raw.Country{
		{Name: "United States", ISO3: "USA"},
		{Name: "Germany", ISO3: "DEU"},
	})
	recs, err := raw.ParseSIPRIFile(f, "", nm) // "" => auto-detect Constant sheet
	if err != nil {
		t.Fatalf("ParseSIPRIFile: %v", err)
	}
	if len(recs) != 12 { // 2 countries x 6 years
		t.Fatalf("got %d records, want 12", len(recs))
	}
	for _, r := range recs {
		if r.ISO3 == "USA" && r.Year == 2010 && math.Abs(r.MilexLog-math.Log(701)) > 1e-9 {
			t.Errorf("USA 2010 milex_log = %v, want log(701)=%v", r.MilexLog, math.Log(701))
		}
	}
}
