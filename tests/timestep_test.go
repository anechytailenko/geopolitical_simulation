package tests

import (
	"testing"

	"geopolitic/internal/timestep"
)

func TestTimeStepRoundTrip(t *testing.T) {
	cases := []struct {
		year, month, ts int
	}{
		{2010, 1, 0},   // epoch
		{2010, 12, 11}, // first December
		{2011, 1, 12},  // second January
		{2013, 7, 42},  // iso_period example from the plan
		{2014, 3, 50},  // synthetic-event month
		{2026, 6, 197}, // ~current window end
	}
	for _, c := range cases {
		if got := timestep.FromYM(c.year, c.month); got != c.ts {
			t.Errorf("FromYM(%d,%d) = %d, want %d", c.year, c.month, got, c.ts)
		}
		if y := timestep.Year(c.ts); y != c.year {
			t.Errorf("Year(%d) = %d, want %d", c.ts, y, c.year)
		}
		if m := timestep.Month(c.ts); m != c.month {
			t.Errorf("Month(%d) = %d, want %d", c.ts, m, c.month)
		}
	}
}

func TestISOPeriod(t *testing.T) {
	if got := timestep.ISOPeriod(42); got != "2013-07" {
		t.Errorf("ISOPeriod(42) = %q, want 2013-07", got)
	}
	if got := timestep.ISOPeriod(0); got != "2010-01" {
		t.Errorf("ISOPeriod(0) = %q, want 2010-01", got)
	}
}

func TestClampStart(t *testing.T) {
	// A 2004 (pre-epoch) start must clamp to 0.
	if got := timestep.ClampStart(timestep.FromYear(2004)); got != 0 {
		t.Errorf("ClampStart(2004) = %d, want 0", got)
	}
	// An in-window start is unchanged.
	if got := timestep.ClampStart(timestep.FromYear(2014)); got != 48 {
		t.Errorf("ClampStart(2014) = %d, want 48", got)
	}
}
