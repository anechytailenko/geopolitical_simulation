package tests

import (
	"testing"

	"geopolitic/internal/label"
)

func TestClassify(t *testing.T) {
	cases := []struct {
		code      string
		goldstein float64
		want      string
	}{
		{"190", -8, label.MaterialConflict},   // root 19 + violent
		{"193", 0, label.MaterialConflict},    // root 19 alone
		{"200", 0, label.MaterialConflict},    // root 20
		{"010", -7, label.MaterialConflict},   // goldstein < -5 overrides cooperation
		{"112", 0, label.VerbalConflict},      // root 11
		{"160", 0, label.VerbalConflict},      // root 16
		{"070", 0, label.MaterialCooperation}, // root 07
		{"061", 0, label.MaterialCooperation}, // root 06
		{"043", 0, label.VerbalCooperation},   // root 04
		{"030", 0, label.VerbalCooperation},   // root 03
		{"091", 6, label.VerbalCooperation},   // neutral root but goldstein > +5
		{"091", 0, label.StatusQuo},           // root 09 neutral
		{"100", 0, label.StatusQuo},           // root 10 demand, neutral
		{"", 0, label.StatusQuo},              // unparseable
	}
	for _, c := range cases {
		if got := label.Classify(c.code, c.goldstein); got != c.want {
			t.Errorf("Classify(%q, %.0f) = %s, want %s", c.code, c.goldstein, got, c.want)
		}
	}
}

func TestClassIndex(t *testing.T) {
	if len(label.Classes) != 5 {
		t.Fatalf("expected 5 classes, got %d", len(label.Classes))
	}
	if label.Index(label.MaterialConflict) != 0 {
		t.Errorf("MATERIAL_CONFLICT must be index 0 (persisted order)")
	}
	if label.Index(label.StatusQuo) != 4 {
		t.Errorf("STATUS_QUO must be index 4 (persisted order)")
	}
	if label.Index("NOT_A_CLASS") != -1 {
		t.Errorf("unknown class must return -1")
	}
}
