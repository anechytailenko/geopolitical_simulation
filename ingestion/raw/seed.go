package raw

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"geopolitic/internal/neo4jdb"
	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// --- §9a: truly-static identity flags -----------------------------------------

// LoadStaticFlags sets the only two genuinely static seed flags directly on the
// Country identity node: nuclear_flag and coastline_flag (see plan §9a). Every
// country is defaulted (nuclear 0 / has-coast 1), then the seed lists flip the
// exceptions.
func LoadStaticFlags(ctx context.Context, driver neo4j.DriverWithContext, configDir string) error {
	var nuclear, landlocked []string
	if err := readJSON(filepath.Join(configDir, "nuclear_states.json"), &nuclear); err != nil {
		return err
	}
	if err := readJSON(filepath.Join(configDir, "landlocked_countries.json"), &landlocked); err != nil {
		return err
	}
	if err := neo4jdb.Write(ctx, driver,
		"MATCH (c:Country) SET c.nuclear_flag = 0, c.coastline_flag = 1", nil); err != nil {
		return err
	}
	if err := neo4jdb.Write(ctx, driver,
		"UNWIND $ids AS id MATCH (c:Country {id: id}) SET c.nuclear_flag = 1",
		map[string]any{"ids": toAny(nuclear)}); err != nil {
		return err
	}
	return neo4jdb.Write(ctx, driver,
		"UNWIND $ids AS id MATCH (c:Country {id: id}) SET c.coastline_flag = 0",
		map[string]any{"ids": toAny(landlocked)})
}

// --- §9b: time-varying seeds → timestamped FeatureSnapshots --------------------

type unscSchedule struct {
	Permanent []string            `json:"permanent"`
	Elected   map[string][]string `json:"elected"` // "YYYY" -> [iso3...]
}

// LoadUNSC writes unsc_seat_flag as FeatureSnapshots. Permanent members get the
// flag from the epoch onward (one snapshot at t=0, forward-filled forever);
// elected members get it switched on at the January of each listed term-year and
// off the following January. Returns the number of FeatureSnapshot writes.
func LoadUNSC(ctx context.Context, driver neo4j.DriverWithContext, configDir string) (int, error) {
	var sched unscSchedule
	if err := readJSON(filepath.Join(configDir, "unsc_schedule.json"), &sched); err != nil {
		return 0, err
	}
	n := 0
	for _, iso3 := range sched.Permanent {
		if err := MergeFeatures(ctx, driver, "Country", iso3, 0, timestep.Year(0), map[string]any{"unsc_seat_flag": 1}); err != nil {
			return n, err
		}
		n++
	}
	for yearStr, members := range sched.Elected {
		var year int
		if _, err := fmt.Sscanf(yearStr, "%d", &year); err != nil {
			continue
		}
		on := timestep.FromYear(year)
		off := timestep.FromYear(year + 1)
		for _, iso3 := range members {
			if err := MergeFeatures(ctx, driver, "Country", iso3, on, year, map[string]any{"unsc_seat_flag": 1}); err != nil {
				return n, err
			}
			// turn the seat off after the term so forward-fill carries 0
			if err := MergeFeatures(ctx, driver, "Country", iso3, off, year+1, map[string]any{"unsc_seat_flag": 0}); err != nil {
				return n, err
			}
			n += 2
		}
	}
	return n, nil
}

type sanctionInterval struct {
	StartYear int  `json:"start_year"`
	EndYear   *int `json:"end_year"`
}

// LoadSanctions writes sanctions_status as FeatureSnapshots: 1 at the January of
// each regime's start year, and 0 at the January it ends (so forward-fill carries
// the correct value between). Returns the number of FeatureSnapshot writes.
func LoadSanctions(ctx context.Context, driver neo4j.DriverWithContext, configDir string) (int, error) {
	var registry map[string][]sanctionInterval
	if err := readJSON(filepath.Join(configDir, "sanctions_registry.json"), &registry); err != nil {
		return 0, err
	}
	n := 0
	for iso3, intervals := range registry {
		for _, iv := range intervals {
			start := timestep.ClampStart(timestep.FromYear(iv.StartYear))
			if err := MergeFeatures(ctx, driver, "Country", iso3, start, timestep.Year(start), map[string]any{"sanctions_status": 1}); err != nil {
				return n, err
			}
			n++
			if iv.EndYear != nil {
				end := timestep.FromYear(*iv.EndYear)
				if err := MergeFeatures(ctx, driver, "Country", iso3, end, *iv.EndYear, map[string]any{"sanctions_status": 0}); err != nil {
					return n, err
				}
				n++
			}
		}
	}
	return n, nil
}

type tierEntry struct {
	Year int `json:"year"`
	Tier int `json:"tier"`
}

// LoadActorFinancialTier writes financial_resources_tier as FeatureSnapshots on
// Actor nodes, one per (actor, year). Keyed by Wikidata QID so it matches the
// actors created by the Wikidata loader. Returns the number of writes.
func LoadActorFinancialTier(ctx context.Context, driver neo4j.DriverWithContext, configDir string) (int, error) {
	var tiers map[string][]tierEntry
	if err := readJSON(filepath.Join(configDir, "actor_financial_tier.json"), &tiers); err != nil {
		return 0, err
	}
	n := 0
	for actorID, entries := range tiers {
		for _, e := range entries {
			ts := timestep.FromYear(e.Year)
			if err := MergeFeatures(ctx, driver, "Actor", actorID, ts, e.Year, map[string]any{"financial_resources_tier": e.Tier}); err != nil {
				return n, err
			}
			n++
		}
	}
	return n, nil
}

// --- helpers ------------------------------------------------------------------

func readJSON(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("parse %s: %w", path, err)
	}
	return nil
}

func toAny(xs []string) []any {
	out := make([]any, len(xs))
	for i, x := range xs {
		out[i] = x
	}
	return out
}
