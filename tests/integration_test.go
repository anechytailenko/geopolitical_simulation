//go:build integration

// Package tests — integration coverage that runs the real loaders against the
// two dockerized Neo4j databases. Run with: go test -tags=integration ./tests/...
// Requires the docker-compose stack up (raw=7687, aggregated=7688).
//
// External sources are served from in-process fixtures so the assertions are
// deterministic — there is NO synthetic data in the pipeline itself; the fixtures
// here are test inputs in the real source formats.
package tests

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"

	"geopolitic/ingestion/aggregated"
	"geopolitic/ingestion/raw"
	"geopolitic/internal/config"
	"geopolitic/internal/label"
	"geopolitic/internal/neo4jdb"
	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

const configDir = "../configs"

// fixtureCountries covers every ISO-3 referenced by the static/time-varying seed
// configs (UNSC permanent + elected, sanctions) plus the event/structural fixtures,
// so no loader has to create a stray Country node.
var fixtureCountries = []map[string]string{
	{"id": "USA", "iso2": "US"}, {"id": "RUS", "iso2": "RU"}, {"id": "GBR", "iso2": "GB"},
	{"id": "FRA", "iso2": "FR"}, {"id": "CHN", "iso2": "CN"}, {"id": "JPN", "iso2": "JP"},
	{"id": "KOR", "iso2": "KR"}, {"id": "BRA", "iso2": "BR"}, {"id": "PRK", "iso2": "KP"},
	{"id": "IRN", "iso2": "IR"}, {"id": "SYR", "iso2": "SY"}, {"id": "LBY", "iso2": "LY"},
	{"id": "VEN", "iso2": "VE"}, {"id": "BLR", "iso2": "BY"}, {"id": "MMR", "iso2": "MM"},
	{"id": "AFG", "iso2": "AF"}, {"id": "MLI", "iso2": "ML"}, {"id": "SSD", "iso2": "SS"},
	{"id": "UKR", "iso2": "UA"}, {"id": "DEU", "iso2": "DE"}, {"id": "POL", "iso2": "PL"},
	{"id": "IND", "iso2": "IN"}, {"id": "PAK", "iso2": "PK"}, {"id": "CAN", "iso2": "CA"},
}

// requireWipeOK guards the destructive integration tests: they call
// `MATCH (n) DETACH DELETE n` on both databases, so they must never run against a
// populated production DB by accident. Opt in with GEOPOLITIC_ALLOW_DB_WIPE=1 and
// point RAW_URI/AGG_URI at a throwaway instance.
func requireWipeOK(t *testing.T) {
	if os.Getenv("GEOPOLITIC_ALLOW_DB_WIPE") != "1" {
		t.Skip("destructive: wipes both Neo4j DBs. Set GEOPOLITIC_ALLOW_DB_WIPE=1 (use a throwaway DB) to run.")
	}
}

func TestCoreIngestionAndAggregation(t *testing.T) {
	requireWipeOK(t)
	ctx := context.Background()
	cfg := config.FromEnv()
	rawDriver, aggDriver := connect(ctx, t, cfg)
	defer rawDriver.Close(ctx)
	defer aggDriver.Close(ctx)
	wipe(ctx, t, rawDriver)
	wipe(ctx, t, aggDriver)

	if err := raw.ApplySchema(ctx, rawDriver); err != nil {
		t.Fatalf("raw schema: %v", err)
	}

	wb := worldBankFixture()
	defer wb.Close()

	// Step 1 — country universe from the (fixture) World Bank country list.
	countries, err := raw.LoadCountryList(ctx, rawDriver, wb.URL, http.DefaultClient)
	if err != nil {
		t.Fatalf("country list: %v", err)
	}
	if len(countries) != len(fixtureCountries) {
		t.Errorf("countries = %d, want %d", len(countries), len(fixtureCountries))
	}

	// Static flags + World Bank indicator snapshots.
	if err := raw.LoadStaticFlags(ctx, rawDriver, configDir); err != nil {
		t.Fatalf("static flags: %v", err)
	}
	if _, err := raw.LoadWorldBank(ctx, rawDriver, wb.URL, http.DefaultClient, countries); err != nil {
		t.Fatalf("worldbank: %v", err)
	}

	// Time-varying seeds → timestamped FeatureSnapshots (the refinement under test).
	if _, err := raw.LoadUNSC(ctx, rawDriver, configDir); err != nil {
		t.Fatalf("unsc: %v", err)
	}
	if _, err := raw.LoadSanctions(ctx, rawDriver, configDir); err != nil {
		t.Fatalf("sanctions: %v", err)
	}

	// Structural fixtures (would come from Wikidata in production) — exercise the
	// temporal MEMBER_OF/BORDERS path incl. Brexit.
	seedStructural(ctx, t, rawDriver)

	// Event fixtures via the real BuildEvent + WriteEvents path (GDELT-shaped).
	writeEventFixtures(ctx, t, rawDriver)

	// ---- Refinement assertions: time-varying seeds are NOT static node props ----
	if v := rawScalar(ctx, t, rawDriver,
		"MATCH (c:Country {id:'USA'}) RETURN c.unsc_seat_flag"); v != nil {
		t.Errorf("unsc_seat_flag must NOT be on the identity node, got %v", v)
	}
	if v := rawScalar(ctx, t, rawDriver,
		"MATCH (c:Country {id:'PRK'}) RETURN c.sanctions_status"); v != nil {
		t.Errorf("sanctions_status must NOT be on the identity node, got %v", v)
	}
	// nuclear_flag IS static and lives on the node.
	if v := readInt(ctx, t, rawDriver, "MATCH (c:Country {id:'USA'}) RETURN c.nuclear_flag", nil); v != 1 {
		t.Errorf("USA nuclear_flag (static) = %d, want 1", v)
	}
	// unsc_seat_flag is a timestamped FeatureSnapshot for the permanent member USA.
	if v := readInt(ctx, t, rawDriver,
		"MATCH (s:FeatureSnapshot {node_id:'USA', time_step:0}) RETURN s.unsc_seat_flag", nil); v != 1 {
		t.Errorf("USA unsc_seat_flag FeatureSnapshot@t0 = %d, want 1", v)
	}
	// sanctions_status is a timestamped FeatureSnapshot for PRK at 2010.
	if v := readInt(ctx, t, rawDriver,
		"MATCH (s:FeatureSnapshot {node_id:'PRK', time_step:0}) RETURN s.sanctions_status", nil); v != 1 {
		t.Errorf("PRK sanctions_status FeatureSnapshot@2010 = %d, want 1", v)
	}

	// ---- Build the aggregated DB ----
	stats, err := aggregated.Build(ctx, rawDriver, aggDriver, 0)
	if err != nil {
		t.Fatalf("aggregate: %v", err)
	}
	identityCount := readInt(ctx, t, aggDriver, "MATCH (n) WHERE n:Country OR n:Actor RETURN count(n)", nil)
	wantNodeSnaps := int(identityCount) * (stats.MaxTimeStep + 1)
	if stats.NodeSnapshots != wantNodeSnaps {
		t.Errorf("NodeSnapshots = %d, want %d (%d identities × %d months)",
			stats.NodeSnapshots, wantNodeSnaps, identityCount, stats.MaxTimeStep+1)
	}

	// Forward-fill carries time-varying seeds into NodeSnapshots.
	if v := readInt(ctx, t, aggDriver,
		"MATCH (n:NodeSnapshot {node_id:'USA'}) WHERE n.time_step=$ts RETURN n.unsc_seat_flag",
		map[string]any{"ts": timestep.FromYM(2015, 6)}); v != 1 {
		t.Errorf("USA unsc_seat_flag NodeSnapshot@2015-06 (forward-filled) = %d, want 1", v)
	}
	if v := readInt(ctx, t, aggDriver,
		"MATCH (n:NodeSnapshot {node_id:'PRK'}) WHERE n.time_step=$ts RETURN n.sanctions_status",
		map[string]any{"ts": timestep.FromYM(2013, 6)}); v != 1 {
		t.Errorf("PRK sanctions_status NodeSnapshot@2013-06 = %d, want 1", v)
	}
	// RUS sanctions begin 2014 -> 1 in 2015, absent/0 before.
	if v := readInt(ctx, t, aggDriver,
		"MATCH (n:NodeSnapshot {node_id:'RUS'}) WHERE n.time_step=$ts RETURN n.sanctions_status",
		map[string]any{"ts": timestep.FromYM(2015, 6)}); v != 1 {
		t.Errorf("RUS sanctions_status NodeSnapshot@2015-06 = %d, want 1", v)
	}
	// World Bank forward-fill (mid-year carries the year value).
	gdp := readFloat(ctx, t, aggDriver,
		"MATCH (n:NodeSnapshot {node_id:'USA'}) WHERE n.time_step=$ts RETURN n.gdp_log",
		map[string]any{"ts": timestep.FromYM(2014, 7)})
	if math.Abs(gdp-math.Log(1001)) > 1e-9 {
		t.Errorf("USA gdp_log forward-fill@2014-07 = %v, want %v", gdp, math.Log(1001))
	}

	// Conflict-priority dominance: IND->PAK has verbal-coop + material-conflict.
	dom := readString(ctx, t, aggDriver,
		"MATCH (:Country{id:'IND'})-[r:SNAPSHOT_EDGE {time_step:$ts}]->(:Country{id:'PAK'}) RETURN r.dominant_class",
		map[string]any{"ts": timestep.FromYM(2014, 3)})
	if dom != label.MaterialConflict {
		t.Errorf("IND->PAK dominant_class = %s, want MATERIAL_CONFLICT", dom)
	}

	// Brexit point-in-time membership (mirrored to aggregated).
	if !euMember(ctx, t, aggDriver, "GBR", timestep.FromYear(2018)) {
		t.Error("GBR should be an EU member in 2018")
	}
	if euMember(ctx, t, aggDriver, "GBR", timestep.FromYear(2021)) {
		t.Error("GBR should NOT be an EU member in 2021 (Brexit)")
	}
}

// TestWikidataLoader drives raw.LoadWikidata against a fixture SPARQL endpoint
// and verifies region, temporal BORDERS, and IGO MEMBER_OF are written.
func TestWikidataLoader(t *testing.T) {
	requireWipeOK(t)
	ctx := context.Background()
	cfg := config.FromEnv()
	rawDriver, _ := connect(ctx, t, cfg)
	defer rawDriver.Close(ctx)
	wipe(ctx, t, rawDriver)
	if err := raw.ApplySchema(ctx, rawDriver); err != nil {
		t.Fatalf("schema: %v", err)
	}
	// Countries must exist for MATCH-based edge attachment.
	for _, id := range []string{"USA", "CAN", "DEU", "FRA"} {
		if err := neo4jdb.Write(ctx, rawDriver, "MERGE (c:Country {id:$id})", map[string]any{"id": id}); err != nil {
			t.Fatal(err)
		}
	}

	sp := wikidataFixture()
	defer sp.Close()
	stats, err := raw.LoadWikidata(ctx, rawDriver, sp.URL, http.DefaultClient)
	if err != nil {
		t.Fatalf("LoadWikidata: %v", err)
	}
	if stats.Regions == 0 || stats.Borders == 0 || stats.Memberships == 0 {
		t.Errorf("wikidata stats = %+v, want all > 0", stats)
	}
	if v := readString(ctx, t, rawDriver, "MATCH (c:Country {id:'DEU'}) RETURN c.region", nil); v != "Europe" {
		t.Errorf("DEU region = %q, want Europe", v)
	}
	if n := readInt(ctx, t, rawDriver, "MATCH (:Country{id:'USA'})-[r:BORDERS]->(:Country{id:'CAN'}) RETURN count(r)", nil); n == 0 {
		t.Error("expected USA-CAN border edge")
	}
	if n := readInt(ctx, t, rawDriver, "MATCH (:Country{id:'DEU'})-[r:MEMBER_OF]->(:Actor{id:'Q458'}) RETURN count(r)", nil); n == 0 {
		t.Error("expected DEU MEMBER_OF EU (Q458)")
	}
}

// ---- fixtures ----------------------------------------------------------------

func worldBankFixture() *httptest.Server {
	values := map[string]float64{
		"NY.GDP.MKTP.CD": 1000, "NY.GDP.PCAP.CD": 250, "NE.TRD.GNFS.ZS": 60,
		"SP.POP.TOTL": 5000, "SP.POP.GROW": 1, "AG.LND.TOTL.K2": 9000, "PV.EST": 0.5,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/country", func(w http.ResponseWriter, r *http.Request) {
		list := make([]map[string]any, 0, len(fixtureCountries))
		for _, c := range fixtureCountries {
			list = append(list, map[string]any{
				"id": c["id"], "iso2Code": c["iso2"], "name": c["id"],
				"region": map[string]any{"id": "EUR", "value": "Europe & Central Asia"},
			})
		}
		_ = json.NewEncoder(w).Encode([]any{map[string]any{"pages": 1}, list})
	})
	mux.HandleFunc("/country/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		indicator := parts[len(parts)-1]
		val := values[indicator]
		obs := make([]map[string]any, 0, 6)
		for y := 2010; y <= 2015; y++ {
			obs = append(obs, map[string]any{"countryiso3code": "", "date": strconv.Itoa(y), "value": val})
		}
		_ = json.NewEncoder(w).Encode([]any{map[string]any{"pages": 1}, obs})
	})
	return httptest.NewServer(mux)
}

// wikidataFixture returns canned SPARQL JSON, branching on the query content.
func wikidataFixture() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("query")
		var bindings []map[string]any
		switch {
		case strings.Contains(q, "P47"): // borders
			bindings = []map[string]any{
				{"c1iso": lit("USA"), "c2iso": lit("CAN")},
				{"c1iso": lit("DEU"), "c2iso": lit("FRA"), "start": lit("2010-01-01T00:00:00Z")},
			}
		case strings.Contains(q, "P463"): // memberships
			bindings = []map[string]any{
				{"org": uri("Q458"), "orgLabel": lit("European Union"), "memberIso": lit("DEU")},
				{"org": uri("Q458"), "orgLabel": lit("European Union"), "memberIso": lit("FRA")},
			}
		default: // regions (P30)
			bindings = []map[string]any{
				{"iso3": lit("USA"), "continentLabel": lit("North America")},
				{"iso3": lit("DEU"), "continentLabel": lit("Europe")},
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": map[string]any{"bindings": bindings},
		})
	}))
}

func lit(v string) map[string]any { return map[string]any{"type": "literal", "value": v} }
func uri(qid string) map[string]any {
	return map[string]any{"type": "uri", "value": "http://www.wikidata.org/entity/" + qid}
}

func seedStructural(ctx context.Context, t *testing.T, d neo4j.DriverWithContext) {
	stmts := []struct {
		cypher string
		params map[string]any
	}{
		{"MERGE (a:Actor {id:'EU'}) SET a.name='European Union', a.actor_type='IGO'", nil},
		{`MATCH (m:Country {id:'GBR'}),(a:Actor{id:'EU'}) MERGE (m)-[r:MEMBER_OF]->(a)
		  SET r.start_time_step=0, r.end_time_step=$end`, map[string]any{"end": timestep.FromYear(2020)}},
		{`MATCH (m:Country {id:'DEU'}),(a:Actor{id:'EU'}) MERGE (m)-[r:MEMBER_OF]->(a)
		  SET r.start_time_step=0, r.end_time_step=null`, nil},
		{`MATCH (m:Country {id:'FRA'}),(a:Actor{id:'EU'}) MERGE (m)-[r:MEMBER_OF]->(a)
		  SET r.start_time_step=0, r.end_time_step=null`, nil},
		{`MATCH (a:Country{id:'RUS'}),(b:Country{id:'UKR'}) MERGE (a)-[r:BORDERS]->(b)
		  SET r.start_time_step=0, r.end_time_step=null`, nil},
	}
	for _, s := range stmts {
		if err := neo4jdb.Write(ctx, d, s.cypher, s.params); err != nil {
			t.Fatalf("seedStructural: %v", err)
		}
	}
}

func writeEventFixtures(ctx context.Context, t *testing.T, d neo4j.DriverWithContext) {
	// ISO-3 used directly as the actor "code" for these test fixtures.
	cm := map[string]string{"RUS": "RUS", "UKR": "UKR", "IND": "IND", "PAK": "PAK", "USA": "USA"}
	specs := []struct {
		id, day, a1, a2, cameo string
		gold                   float64
	}{
		{"E1", "20140301", "RUS", "UKR", "190", -8}, // material conflict
		{"E2", "20140310", "RUS", "UKR", "036", 4},  // verbal coop
		{"E3", "20140305", "IND", "PAK", "043", 2},  // verbal coop
		{"E4", "20140306", "IND", "PAK", "190", -7}, // material conflict -> priority
		{"E5", "20140307", "USA", "RUS", "112", -3}, // verbal conflict
	}
	var events []raw.Event
	for _, s := range specs {
		ev, ok := raw.BuildEvent(s.id, s.day, s.a1, s.a2, s.cameo, s.gold, s.gold, 10, "GDELT", cm)
		if !ok {
			t.Fatalf("BuildEvent failed for %s", s.id)
		}
		events = append(events, ev)
	}
	if _, err := raw.WriteEvents(ctx, d, events); err != nil {
		t.Fatalf("WriteEvents: %v", err)
	}
}

// ---- helpers -----------------------------------------------------------------

func connect(ctx context.Context, t *testing.T, cfg config.Config) (neo4j.DriverWithContext, neo4j.DriverWithContext) {
	rd, err := neo4jdb.Connect(ctx, cfg.RawURI, cfg.RawUser, cfg.RawPass)
	if err != nil {
		t.Fatalf("connect raw (is docker-compose up?): %v", err)
	}
	ad, err := neo4jdb.Connect(ctx, cfg.AggURI, cfg.AggUser, cfg.AggPass)
	if err != nil {
		t.Fatalf("connect aggregated (is docker-compose up?): %v", err)
	}
	return rd, ad
}

func euMember(ctx context.Context, t *testing.T, d neo4j.DriverWithContext, iso3 string, ts int) bool {
	return readInt(ctx, t, d, `
MATCH (m:Country {id:$id})-[r:MEMBER_OF]->(:Actor {id:'EU'})
WHERE r.start_time_step <= $ts AND (r.end_time_step IS NULL OR r.end_time_step > $ts)
RETURN count(r)`, map[string]any{"id": iso3, "ts": ts}) > 0
}

func wipe(ctx context.Context, t *testing.T, d neo4j.DriverWithContext) {
	if err := neo4jdb.Write(ctx, d, "MATCH (n) DETACH DELETE n", nil); err != nil {
		t.Fatalf("wipe: %v", err)
	}
}

func rawScalar(ctx context.Context, t *testing.T, d neo4j.DriverWithContext, cypher string) any {
	rows, err := neo4jdb.ReadRows(ctx, d, cypher, nil)
	if err != nil || len(rows) == 0 {
		t.Fatalf("rawScalar %q: err=%v rows=%d", cypher, err, len(rows))
	}
	for _, v := range rows[0] {
		return v
	}
	return nil
}

func readFloat(ctx context.Context, t *testing.T, d neo4j.DriverWithContext, cypher string, params map[string]any) float64 {
	rows, err := neo4jdb.ReadRows(ctx, d, cypher, params)
	if err != nil || len(rows) == 0 {
		t.Fatalf("readFloat %q: err=%v rows=%d", cypher, err, len(rows))
	}
	for _, v := range rows[0] {
		return toFloat(v)
	}
	return 0
}

func readInt(ctx context.Context, t *testing.T, d neo4j.DriverWithContext, cypher string, params map[string]any) int64 {
	rows, err := neo4jdb.ReadRows(ctx, d, cypher, params)
	if err != nil || len(rows) == 0 {
		t.Fatalf("readInt %q: err=%v rows=%d", cypher, err, len(rows))
	}
	for _, v := range rows[0] {
		return toInt(v)
	}
	return 0
}

func readString(ctx context.Context, t *testing.T, d neo4j.DriverWithContext, cypher string, params map[string]any) string {
	rows, err := neo4jdb.ReadRows(ctx, d, cypher, params)
	if err != nil || len(rows) == 0 {
		t.Fatalf("readString %q: err=%v rows=%d", cypher, err, len(rows))
	}
	for _, v := range rows[0] {
		s, _ := v.(string)
		return s
	}
	return ""
}

func toFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int64:
		return float64(n)
	default:
		return 0
	}
}

func toInt(v any) int64 {
	switch n := v.(type) {
	case int64:
		return n
	case float64:
		return int64(n)
	default:
		return 0
	}
}
