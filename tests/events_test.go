package tests

import (
	"testing"

	"geopolitic/ingestion/raw"
	"geopolitic/internal/label"
	"geopolitic/internal/timestep"
)

// cameoMap mirrors a slice of configs/cameo_country_to_iso3.json.
var cameoMap = map[string]string{
	"USA": "USA", "RUS": "RUS", "GMY": "DEU", "UKG": "GBR", "UKR": "UKR",
}

func TestBuildEvent(t *testing.T) {
	ev, ok := raw.BuildEvent("12345", "20140315", "UKG", "GMY", "042", 2.0, 3.5, 7, "GDELT", cameoMap)
	if !ok {
		t.Fatal("expected event to build")
	}
	if ev.Source != "GBR" || ev.Target != "DEU" {
		t.Errorf("country conversion: got %s->%s, want GBR->DEU", ev.Source, ev.Target)
	}
	if ev.TimeStep != timestep.FromYM(2014, 3) {
		t.Errorf("time_step = %d, want %d", ev.TimeStep, timestep.FromYM(2014, 3))
	}
	if ev.RelationshipClass != label.VerbalCooperation {
		t.Errorf("class = %s, want VERBAL_COOPERATION", ev.RelationshipClass)
	}
	if ev.SourceCount != 7 || ev.SentimentScore != 0.35 {
		t.Errorf("source_count=%d sentiment=%v, want 7 / 0.35", ev.SourceCount, ev.SentimentScore)
	}
}

func TestBuildEventConflictClass(t *testing.T) {
	// CAMEO 190 with very negative Goldstein -> material conflict.
	ev, ok := raw.BuildEvent("1", "20140301", "RUS", "UKR", "190", -8, -5, 40, "GDELT", cameoMap)
	if !ok || ev.RelationshipClass != label.MaterialConflict {
		t.Fatalf("RUS->UKR 190 => ok=%v class=%s, want MATERIAL_CONFLICT", ok, ev.RelationshipClass)
	}
}

func TestBuildEventDrops(t *testing.T) {
	cases := []struct {
		name                   string
		id, day, a1, a2, cameo string
	}{
		{"unknown target", "1", "20140315", "USA", "XXX", "042"},
		{"unknown source", "1", "20140315", "ZZZ", "USA", "042"},
		{"self loop", "1", "20140315", "RUS", "RUS", "042"},
		{"bad date", "1", "2014", "USA", "RUS", "042"},
		{"pre-epoch", "1", "20081231", "USA", "RUS", "042"},
	}
	for _, c := range cases {
		if _, ok := raw.BuildEvent(c.id, c.day, c.a1, c.a2, c.cameo, 0, 0, 1, "GDELT", cameoMap); ok {
			t.Errorf("%s: expected drop, got event", c.name)
		}
	}
}
