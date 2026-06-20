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
	"strings"
	"time"

	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// acledViolentTypes are the ACLED event_types that count toward a country's
// conflict-intensity feature (see plans/02-data-ingestion.md §7).
var acledViolentTypes = map[string]bool{
	"battles":                    true,
	"explosions/remote violence": true,
	"violence against civilians": true,
}

// ACLEDEvent is the minimal slice of an ACLED record we use.
type ACLEDEvent struct {
	Country    string
	EventDate  string // YYYY-MM-DD
	EventType  string
	Fatalities int
}

// ACLEDFeature is an aggregated per-(country, month) conflict observation.
type ACLEDFeature struct {
	ISO3              string
	TimeStep          int
	Year              int
	ActiveConflicts   int
	ConflictIntensity float64 // log(sum_fatalities + 1)
}

type acledKey struct {
	iso3 string
	ts   int
}

type acledAcc struct {
	count int
	fat   int
}

// acledAgg folds violent ACLED events into per-(country, month) buckets. It is
// fed page-by-page so the full global dataset never has to be held in memory at
// once — the bucket map is bounded by countries × months (~41k max).
type acledAgg struct {
	buckets map[acledKey]*acledAcc
}

func newACLEDAgg() *acledAgg { return &acledAgg{buckets: map[acledKey]*acledAcc{}} }

func (a *acledAgg) add(events []ACLEDEvent, nm NameMap) {
	for _, e := range events {
		if !acledViolentTypes[strings.ToLower(strings.TrimSpace(e.EventType))] {
			continue
		}
		iso3, ok := nm.Lookup(e.Country)
		if !ok || len(e.EventDate) < 7 {
			continue
		}
		year, err1 := strconv.Atoi(e.EventDate[:4])
		month, err2 := strconv.Atoi(e.EventDate[5:7])
		if err1 != nil || err2 != nil || month < 1 || month > 12 || year < timestep.Epoch {
			continue
		}
		k := acledKey{iso3: iso3, ts: timestep.FromYM(year, month)}
		if a.buckets[k] == nil {
			a.buckets[k] = &acledAcc{}
		}
		a.buckets[k].count++
		a.buckets[k].fat += e.Fatalities
	}
}

func (a *acledAgg) features() []ACLEDFeature {
	out := make([]ACLEDFeature, 0, len(a.buckets))
	for k, acc := range a.buckets {
		out = append(out, ACLEDFeature{
			ISO3:              k.iso3,
			TimeStep:          k.ts,
			Year:              timestep.Year(k.ts),
			ActiveConflicts:   acc.count,
			ConflictIntensity: math.Log(float64(acc.fat) + 1),
		})
	}
	return out
}

// AggregateACLED buckets violent ACLED events by (country, month) into
// active_conflict_count and conflict_intensity. Pure — unit-tested with fixtures.
func AggregateACLED(events []ACLEDEvent, nm NameMap) []ACLEDFeature {
	a := newACLEDAgg()
	a.add(events, nm)
	return a.features()
}

// acledPageDelay paces page requests; ACLED returns 401 on back-to-back calls.
const acledPageDelay = 350 * time.Millisecond

// acledMaxRetries bounds the per-page retry/backoff (covers 401 throttling, 429,
// and 5xx).
const acledMaxRetries = 4

// LoadACLED authenticates via OAuth, pages through the read endpoint for each
// year in [startYear, endYear], aggregates, and writes conflict FeatureSnapshots.
func LoadACLED(ctx context.Context, driver neo4j.DriverWithContext, client *http.Client, tokenURL, readURL, email, password string, startYear, endYear int, nm NameMap) (int, error) {
	token, err := acledToken(ctx, client, tokenURL, email, password)
	if err != nil {
		return 0, fmt.Errorf("acled auth: %w", err)
	}

	agg := newACLEDAgg()
	for year := startYear; year <= endYear; year++ {
		for page := 1; ; page++ {
			batch, err := acledReadPageRetry(ctx, client, readURL, &token, tokenURL, email, password, year, page)
			if err != nil {
				return 0, fmt.Errorf("acled read %d p%d: %w", year, page, err)
			}
			if len(batch) == 0 {
				break
			}
			agg.add(batch, nm) // fold immediately; do not retain raw events
			time.Sleep(acledPageDelay)
		}
	}

	features := agg.features()
	n := 0
	for _, f := range features {
		feats := map[string]any{
			"active_conflict_count": f.ActiveConflicts,
			"conflict_intensity":    f.ConflictIntensity,
		}
		if err := MergeFeatures(ctx, driver, "Country", f.ISO3, f.TimeStep, f.Year, feats); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

func acledToken(ctx context.Context, client *http.Client, tokenURL, email, password string) (string, error) {
	form := url.Values{
		"username":   {email},
		"password":   {password},
		"grant_type": {"password"},
		"client_id":  {"acled"},
		"scope":      {"authenticated"},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("token status %d: %s", resp.StatusCode, string(body))
	}
	var tok struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tok); err != nil {
		return "", err
	}
	if tok.AccessToken == "" {
		return "", fmt.Errorf("empty access token")
	}
	return tok.AccessToken, nil
}

// acledReadPageRetry reads one page, retrying transient failures (401 throttling,
// 429, 5xx) with linear backoff and re-authenticating on a persistent 401 (the
// access token can be rate-throttled mid-run).
func acledReadPageRetry(ctx context.Context, client *http.Client, readURL string, token *string, tokenURL, email, password string, year, page int) ([]ACLEDEvent, error) {
	var lastErr error
	for attempt := 0; attempt <= acledMaxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt) * time.Second):
			}
		}
		events, status, err := acledReadPage(ctx, client, readURL, *token, year, page)
		if err == nil {
			return events, nil
		}
		lastErr = err
		if status == http.StatusUnauthorized && attempt >= 1 {
			if nt, e := acledToken(ctx, client, tokenURL, email, password); e == nil && nt != "" {
				*token = nt
			}
		}
		if status == http.StatusUnauthorized || status == http.StatusTooManyRequests || status >= 500 || status == 0 {
			continue // transient — retry
		}
		return nil, err // non-retryable
	}
	return nil, fmt.Errorf("exhausted retries: %w", lastErr)
}

func acledReadPage(ctx context.Context, client *http.Client, readURL, token string, year, page int) ([]ACLEDEvent, int, error) {
	u := fmt.Sprintf("%s?_format=json&year=%d&limit=5000&page=%d&fields=country|event_date|event_type|fatalities", readURL, year, page)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, resp.StatusCode, fmt.Errorf("read status %d: %s", resp.StatusCode, string(body))
	}
	ev, err := parseACLEDResponse(resp.Body)
	return ev, http.StatusOK, err
}

// parseACLEDResponse decodes the {"data":[...]} envelope; fatalities arrives as a
// string or number depending on the endpoint, so it is coerced.
func parseACLEDResponse(r io.Reader) ([]ACLEDEvent, error) {
	var payload struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.NewDecoder(r).Decode(&payload); err != nil {
		return nil, err
	}
	out := make([]ACLEDEvent, 0, len(payload.Data))
	for _, d := range payload.Data {
		out = append(out, ACLEDEvent{
			Country:    asString(d["country"]),
			EventDate:  asString(d["event_date"]),
			EventType:  asString(d["event_type"]),
			Fatalities: asInt(d["fatalities"]),
		})
	}
	return out, nil
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case string:
		i, _ := strconv.Atoi(strings.TrimSpace(n))
		return i
	default:
		return 0
	}
}
