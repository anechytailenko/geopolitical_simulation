package raw

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strconv"

	"geopolitic/internal/neo4jdb"
	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// wikidataUserAgent is required by the WDQS endpoint (it rejects blank UAs).
const wikidataUserAgent = "geopolitic-ingestion/1.0 (research; contact via repo)"

// majorOrgs is the curated set of intergovernmental organizations whose
// memberships we pull. Keyed by Wikidata QID (matches actor_financial_tier.json).
// Querying a fixed VALUES set is far more reliable than an open subclass walk.
// QIDs verified against Wikidata (resolve with `?item rdfs:label` before editing —
// e.g. Q188749 is a chanterelle mushroom, not the SCO).
var majorOrgs = []string{
	"Q1065",   // United Nations
	"Q458",    // European Union
	"Q7184",   // NATO
	"Q7768",   // Association of Southeast Asian Nations (ASEAN)
	"Q7159",   // African Union
	"Q485207", // Shanghai Cooperation Organisation
	"Q7172",   // League of Arab States
	"Q123759", // Organization of American States
	"Q318693", // Collective Security Treaty Organization (CSTO)
	"Q7825",   // World Trade Organization
	"Q8475",   // Interpol
}

// WikidataStats reports what the Wikidata loader wrote.
type WikidataStats struct {
	Regions     int
	Borders     int
	Actors      int
	Memberships int
}

// LoadWikidata populates region (P30), temporal BORDERS (P47 + P580/P582), IGO
// Actor nodes and temporal MEMBER_OF edges (P463 + P580/P582), plus the Actor
// member_count_log / recognized_legitimacy_score features — all from the free
// SPARQL endpoint. Each step is independent; a failing step is returned as an
// error but the caller treats Wikidata as best-effort.
func LoadWikidata(ctx context.Context, driver neo4j.DriverWithContext, endpoint string, client *http.Client) (WikidataStats, error) {
	var stats WikidataStats

	if n, err := loadRegions(ctx, driver, endpoint, client); err != nil {
		return stats, fmt.Errorf("regions: %w", err)
	} else {
		stats.Regions = n
	}

	if n, err := loadBorders(ctx, driver, endpoint, client); err != nil {
		return stats, fmt.Errorf("borders: %w", err)
	} else {
		stats.Borders = n
	}

	actors, memberships, err := loadMemberships(ctx, driver, endpoint, client)
	if err != nil {
		return stats, fmt.Errorf("memberships: %w", err)
	}
	stats.Actors = actors
	stats.Memberships = memberships

	return stats, nil
}

func loadRegions(ctx context.Context, driver neo4j.DriverWithContext, endpoint string, client *http.Client) (int, error) {
	const q = `
SELECT ?iso3 ?continentLabel WHERE {
  ?country wdt:P31 wd:Q6256 ; wdt:P298 ?iso3 .
  OPTIONAL { ?country wdt:P30 ?continent .
             ?continent rdfs:label ?continentLabel FILTER(LANG(?continentLabel)="en") }
}`
	rows, err := sparql(ctx, client, endpoint, q)
	if err != nil {
		return 0, err
	}
	batch := make([]any, 0, len(rows))
	for _, r := range rows {
		region := r["continentLabel"]
		if r["iso3"] == "" || region == "" {
			continue
		}
		batch = append(batch, map[string]any{"iso3": r["iso3"], "region": region})
	}
	if err := neo4jdb.Write(ctx, driver, `
UNWIND $rows AS row
MATCH (c:Country {id: row.iso3})
SET c.region = row.region`, map[string]any{"rows": batch}); err != nil {
		return 0, err
	}
	return len(batch), nil
}

func loadBorders(ctx context.Context, driver neo4j.DriverWithContext, endpoint string, client *http.Client) (int, error) {
	const q = `
SELECT ?c1iso ?c2iso ?start ?end WHERE {
  ?c1 wdt:P31 wd:Q6256 ; wdt:P298 ?c1iso ; p:P47 ?stmt .
  ?stmt ps:P47 ?c2 .
  ?c2 wdt:P31 wd:Q6256 ; wdt:P298 ?c2iso .
  OPTIONAL { ?stmt pq:P580 ?start }
  OPTIONAL { ?stmt pq:P582 ?end }
}`
	rows, err := sparql(ctx, client, endpoint, q)
	if err != nil {
		return 0, err
	}
	batch := make([]any, 0, len(rows))
	for _, r := range rows {
		if r["c1iso"] == "" || r["c2iso"] == "" || r["c1iso"] == r["c2iso"] {
			continue
		}
		batch = append(batch, map[string]any{
			"a":     r["c1iso"],
			"b":     r["c2iso"],
			"start": startTS(r["start"]),
			"end":   endTS(r["end"]),
		})
	}
	// Symmetric: write both directions so neighbor_count is correct either way.
	if err := neo4jdb.Write(ctx, driver, `
UNWIND $rows AS row
MATCH (a:Country {id: row.a}), (b:Country {id: row.b})
MERGE (a)-[r:BORDERS]->(b) SET r.start_time_step = row.start, r.end_time_step = row.end
MERGE (b)-[r2:BORDERS]->(a) SET r2.start_time_step = row.start, r2.end_time_step = row.end`,
		map[string]any{"rows": batch}); err != nil {
		return 0, err
	}
	return len(batch), nil
}

func loadMemberships(ctx context.Context, driver neo4j.DriverWithContext, endpoint string, client *http.Client) (int, int, error) {
	values := ""
	for _, qid := range majorOrgs {
		values += "wd:" + qid + " "
	}
	q := fmt.Sprintf(`
SELECT ?org ?orgLabel ?memberIso ?start ?end WHERE {
  VALUES ?org { %s}
  ?member p:P463 ?stmt .
  ?stmt ps:P463 ?org .
  ?member wdt:P298 ?memberIso .
  OPTIONAL { ?stmt pq:P580 ?start }
  OPTIONAL { ?stmt pq:P582 ?end }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". ?org rdfs:label ?orgLabel }
}`, values)
	rows, err := sparql(ctx, client, endpoint, q)
	if err != nil {
		return 0, 0, err
	}

	type org struct {
		name    string
		members map[string]bool
	}
	orgs := map[string]*org{}
	memberBatch := make([]any, 0, len(rows))
	for _, r := range rows {
		qid := qidFromURI(r["org"])
		if qid == "" || r["memberIso"] == "" {
			continue
		}
		if orgs[qid] == nil {
			orgs[qid] = &org{name: r["orgLabel"], members: map[string]bool{}}
		}
		orgs[qid].members[r["memberIso"]] = true
		memberBatch = append(memberBatch, map[string]any{
			"member": r["memberIso"],
			"org":    qid,
			"start":  startTS(r["start"]),
			"end":    endTS(r["end"]),
		})
	}

	// Create Actor nodes.
	actorBatch := make([]any, 0, len(orgs))
	for qid, o := range orgs {
		name := o.name
		if name == "" {
			name = qid
		}
		actorBatch = append(actorBatch, map[string]any{"id": qid, "name": name})
	}
	if err := neo4jdb.Write(ctx, driver, `
UNWIND $rows AS row
MERGE (a:Actor {id: row.id}) SET a.name = row.name, a.actor_type = "IGO"`,
		map[string]any{"rows": actorBatch}); err != nil {
		return 0, 0, err
	}

	// MEMBER_OF edges (country -> actor).
	if err := neo4jdb.Write(ctx, driver, `
UNWIND $rows AS row
MATCH (m:Country {id: row.member}), (a:Actor {id: row.org})
MERGE (m)-[r:MEMBER_OF]->(a) SET r.start_time_step = row.start, r.end_time_step = row.end`,
		map[string]any{"rows": memberBatch}); err != nil {
		return 0, 0, err
	}

	// Actor features: member_count_log + recognized_legitimacy_score (proxy = member count).
	for qid, o := range orgs {
		count := float64(len(o.members))
		feats := map[string]any{
			"member_count_log":            math.Log(count + 1),
			"recognized_legitimacy_score": count,
		}
		if err := MergeFeatures(ctx, driver, "Actor", qid, 0, timestep.Year(0), feats); err != nil {
			return 0, 0, err
		}
	}

	return len(actorBatch), len(memberBatch), nil
}

// --- SPARQL plumbing ----------------------------------------------------------

type sparqlResult struct {
	Results struct {
		Bindings []map[string]struct {
			Value string `json:"value"`
		} `json:"bindings"`
	} `json:"results"`
}

// sparql runs a query and returns each binding flattened to var -> value.
func sparql(ctx context.Context, client *http.Client, endpoint, query string) ([]map[string]string, error) {
	u := endpoint + "?format=json&query=" + url.QueryEscape(query)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/sparql-results+json")
	req.Header.Set("User-Agent", wikidataUserAgent)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("sparql status %d: %s", resp.StatusCode, string(body))
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var parsed sparqlResult
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("decode sparql: %w", err)
	}
	out := make([]map[string]string, 0, len(parsed.Results.Bindings))
	for _, b := range parsed.Results.Bindings {
		row := make(map[string]string, len(b))
		for k, v := range b {
			row[k] = v.Value
		}
		out = append(out, row)
	}
	return out, nil
}

// qidFromURI turns "http://www.wikidata.org/entity/Q1065" into "Q1065".
func qidFromURI(uri string) string {
	if uri == "" {
		return ""
	}
	for i := len(uri) - 1; i >= 0; i-- {
		if uri[i] == '/' {
			return uri[i+1:]
		}
	}
	return uri
}

// startTS converts a Wikidata datetime literal to a clamped start time_step
// (defaults to epoch 0 when absent / pre-2010).
func startTS(s string) int {
	if ts, ok := wikidataDateTS(s); ok {
		return timestep.ClampStart(ts)
	}
	return 0
}

// endTS converts a Wikidata end datetime to a time_step, or nil when still valid.
func endTS(s string) any {
	if ts, ok := wikidataDateTS(s); ok {
		return ts
	}
	return nil
}

func wikidataDateTS(s string) (int, bool) {
	if len(s) < 7 {
		return 0, false
	}
	year, err := strconv.Atoi(s[:4])
	if err != nil {
		return 0, false
	}
	month, err := strconv.Atoi(s[5:7])
	if err != nil || month < 1 {
		month = 1
	}
	if month > 12 {
		month = 12
	}
	return timestep.FromYM(year, month), true
}
