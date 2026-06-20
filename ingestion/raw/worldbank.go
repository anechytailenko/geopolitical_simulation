package raw

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"

	"geopolitic/internal/timestep"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// worldBankConcurrency bounds simultaneous World Bank requests (the API allows
// ~100 req/min). See plans/02-data-ingestion.md step 1.
const worldBankConcurrency = 8

// wbIndicator maps a World Bank indicator code to a FeatureSnapshot property and
// the transform applied to the raw value.
type wbIndicator struct {
	code    string
	feature string
	log     bool // store log(value+1) instead of the raw value
}

// worldBankIndicators is the full World Bank indicator set from
// plans/02-data-ingestion.md §1 (7 indicators, all from the free public API).
var worldBankIndicators = []wbIndicator{
	{code: "NY.GDP.MKTP.CD", feature: "gdp_log", log: true},
	{code: "NY.GDP.PCAP.CD", feature: "gdp_per_capita", log: false},
	{code: "NE.TRD.GNFS.ZS", feature: "trade_openness_index", log: false},
	{code: "SP.POP.TOTL", feature: "population_log", log: true},
	{code: "SP.POP.GROW", feature: "population_growth", log: false},
	{code: "AG.LND.TOTL.K2", feature: "land_area_log", log: true},
	{code: "PV.EST", feature: "political_stability", log: false},
}

// wbResponse models the two-element World Bank JSON envelope: [meta, [obs...]].
type wbObservation struct {
	CountryISO3 string   `json:"countryiso3code"`
	Date        string   `json:"date"`
	Value       *float64 `json:"value"`
}

// LoadWorldBank fetches the core indicators for every supplied country and
// writes one FeatureSnapshot per (country, year) into geopolitic_raw. baseURL is
// overridable so tests can point it at a fixture server. Returns the number of
// FeatureSnapshot nodes written.
func LoadWorldBank(ctx context.Context, driver neo4j.DriverWithContext, baseURL string, client *http.Client, countries []Country) (int, error) {
	// accumulator: iso3 -> time_step -> {features}; guarded by mu (concurrent fetch).
	acc := map[string]map[int]map[string]any{}
	var mu sync.Mutex
	var errs []error

	type job struct {
		country Country
		ind     wbIndicator
	}
	jobs := make(chan job)
	var wg sync.WaitGroup
	for i := 0; i < worldBankConcurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				obs, err := fetchWorldBank(ctx, client, baseURL, j.country.ISO2, j.ind.code)
				mu.Lock()
				if err != nil {
					errs = append(errs, fmt.Errorf("%s/%s: %w", j.country.ISO2, j.ind.code, err))
					mu.Unlock()
					continue
				}
				for _, o := range obs {
					if o.Value == nil {
						continue
					}
					year, e := strconv.Atoi(o.Date)
					if e != nil || year < timestep.Epoch {
						continue
					}
					iso3 := o.CountryISO3
					if iso3 == "" {
						iso3 = j.country.ISO3
					}
					ts := timestep.FromYear(year)
					val := *o.Value
					if j.ind.log {
						val = math.Log(val + 1)
					}
					if acc[iso3] == nil {
						acc[iso3] = map[int]map[string]any{}
					}
					if acc[iso3][ts] == nil {
						acc[iso3][ts] = map[string]any{}
					}
					acc[iso3][ts][j.ind.feature] = val
				}
				mu.Unlock()
			}
		}()
	}
	for _, c := range countries {
		for _, ind := range worldBankIndicators {
			jobs <- job{country: c, ind: ind}
		}
	}
	close(jobs)
	wg.Wait()

	written := 0
	for iso3, byTS := range acc {
		for ts, features := range byTS {
			if err := MergeFeatures(ctx, driver, "Country", iso3, ts, timestep.Year(ts), features); err != nil {
				return written, fmt.Errorf("write featuresnapshot %s@%d: %w", iso3, ts, err)
			}
			written++
		}
	}
	// Partial fetch failures are non-fatal: we keep what we got and report the
	// rest so the pipeline can log a warning.
	if len(errs) > 0 {
		return written, fmt.Errorf("worldbank: %d fetch error(s): %w", len(errs), errors.Join(errs...))
	}
	return written, nil
}

// worldBankRetries is the number of extra attempts for a transient response.
const worldBankRetries = 3

func fetchWorldBank(ctx context.Context, client *http.Client, baseURL, iso2, indicator string) ([]wbObservation, error) {
	var lastErr error
	for attempt := 0; attempt <= worldBankRetries; attempt++ {
		if attempt > 0 {
			// linear backoff; the public API throttles bursts with transient 400/5xx.
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt) * 500 * time.Millisecond):
			}
		}
		obs, retryable, err := doFetchWorldBank(ctx, client, baseURL, iso2, indicator)
		if err == nil {
			return obs, nil
		}
		lastErr = err
		if !retryable {
			return nil, err
		}
	}
	return nil, lastErr
}

func doFetchWorldBank(ctx context.Context, client *http.Client, baseURL, iso2, indicator string) (obs []wbObservation, retryable bool, err error) {
	url := fmt.Sprintf("%s/country/%s/indicator/%s?format=json&per_page=200&date=2010:2026", baseURL, iso2, indicator)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, false, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, true, err // network errors are transient
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		// The World Bank load balancer returns sporadic 400/429/5xx under burst.
		retry := resp.StatusCode == 400 || resp.StatusCode == 429 || resp.StatusCode >= 500
		return nil, retry, fmt.Errorf("status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, true, err
	}

	// The envelope is [meta, [obs...]]; the second element is null when a series
	// has no data, so decode into two raw messages first.
	var envelope []json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, false, fmt.Errorf("decode envelope: %w", err)
	}
	if len(envelope) < 2 {
		return nil, false, nil
	}
	if err := json.Unmarshal(envelope[1], &obs); err != nil {
		return nil, false, nil // message payload (e.g. "no data"), not an error
	}
	return obs, false, nil
}
