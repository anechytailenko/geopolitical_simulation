// Package ingestion orchestrates a full ingest+aggregation run: it loads the
// raw database from every configured real source, then rebuilds the aggregated
// database. Triggered by POST /api/v1/ingest. See plans/02-data-ingestion.md.
//
// The credential-free core (World Bank country list + indicators, Wikidata
// structural edges, static + time-varying JSON seeds) always runs. Each
// credentialed source (GDELT BigQuery, ACLED, V-Dem, SIPRI, UNDP) runs only when
// its env vars / file paths are configured. There is no synthetic data.
package ingestion

import (
	"context"
	"fmt"
	"log"
	"time"

	"geopolitic/ingestion/aggregated"
	"geopolitic/ingestion/raw"
	"geopolitic/internal/config"
	"geopolitic/internal/neo4jdb"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Result is the structured outcome of one ingest run.
type Result struct {
	Status     string    `json:"status"`
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at"`
	DurationMS int64     `json:"duration_ms"`

	Countries          int               `json:"countries"`
	Wikidata           raw.WikidataStats `json:"wikidata"`
	WorldBankSnapshots int               `json:"worldbank_snapshots"`
	UNSC               int               `json:"unsc_snapshots"`
	Sanctions          int               `json:"sanctions_snapshots"`
	FinancialTier      int               `json:"financial_tier_snapshots"`
	GDELTEvents        int               `json:"gdelt_events"`
	GDELTSnapshotEdges int               `json:"gdelt_snapshot_edges"`
	ACLEDFeatures      int               `json:"acled_features"`
	VDem               int               `json:"vdem_snapshots"`
	SIPRI              int               `json:"sipri_snapshots"`
	UNDP               int               `json:"undp_snapshots"`
	Aggregated         aggregated.Stats  `json:"aggregated"`
	Sources            map[string]string `json:"sources"` // source -> "ok"/"skipped"/"error: ..."
	Errors             []string          `json:"errors"`
}

// Run performs the full pipeline. Infrastructure errors (connect, schema,
// country list, aggregation) abort; per-source failures are recorded and skipped.
func Run(ctx context.Context, cfg config.Config) (Result, error) {
	res := Result{Status: "ok", StartedAt: time.Now(), Sources: map[string]string{}}

	rawDriver, err := neo4jdb.Connect(ctx, cfg.RawURI, cfg.RawUser, cfg.RawPass)
	if err != nil {
		return res, fmt.Errorf("connect raw: %w", err)
	}
	defer rawDriver.Close(ctx)
	aggDriver, err := neo4jdb.Connect(ctx, cfg.AggURI, cfg.AggUser, cfg.AggPass)
	if err != nil {
		return res, fmt.Errorf("connect aggregated: %w", err)
	}
	defer aggDriver.Close(ctx)

	// Step 0 — raw schema.
	if err := raw.ApplySchema(ctx, rawDriver); err != nil {
		return res, fmt.Errorf("raw schema: %w", err)
	}

	// Step 1 — country universe (live World Bank country list). Fatal: nothing
	// else can attach without Country nodes. (Skipped only for a targeted
	// INGEST_ONLY re-run, where the nodes already exist from a prior full run.)
	var countries []raw.Country
	if cfg.Runs("countries") {
		countries, err = raw.LoadCountryList(ctx, rawDriver, cfg.WorldBankBaseURL, cfg.HTTPClient)
		if err != nil {
			return res, fmt.Errorf("country list: %w", err)
		}
		res.Countries = len(countries)
		res.Sources["country_list"] = "ok"

		// Static identity flags (§9a).
		if err := raw.LoadStaticFlags(ctx, rawDriver, cfg.ConfigDir); err != nil {
			res.note("static_flags", err)
		} else {
			res.Sources["static_flags"] = "ok"
		}
	} else {
		res.Sources["country_list"] = "skipped (INGEST_ONLY)"
		// ACLED/SIPRI map source country names -> ISO-3 via the country list. On a
		// targeted re-run the Country nodes already exist, but we still need the
		// list in memory to build the name map — fetch it without writing.
		if cfg.Runs("acled") || cfg.Runs("sipri") {
			if c, ferr := raw.FetchCountryList(ctx, cfg.WorldBankBaseURL, cfg.HTTPClient); ferr == nil {
				countries = c
			} else {
				res.note("country_list_for_namemap", ferr)
			}
		}
	}

	// Step 2 — Wikidata: region, temporal BORDERS/MEMBER_OF, IGO actors (free).
	if cfg.WikidataEnabled && cfg.Runs("wikidata") {
		if stats, err := raw.LoadWikidata(ctx, rawDriver, cfg.WikidataEndpoint, cfg.HTTPClient); err != nil {
			res.note("wikidata", err)
		} else {
			res.Wikidata = stats
			res.Sources["wikidata"] = "ok"
		}
	} else {
		res.Sources["wikidata"] = "skipped"
	}

	// Steps 3-6 — World Bank indicator FeatureSnapshots (free).
	if cfg.Runs("worldbank") {
		if n, err := raw.LoadWorldBank(ctx, rawDriver, cfg.WorldBankBaseURL, cfg.HTTPClient, countries); err != nil {
			res.WorldBankSnapshots = n
			res.note("worldbank", err)
		} else {
			res.WorldBankSnapshots = n
			res.Sources["worldbank"] = "ok"
		}
	} else {
		res.Sources["worldbank"] = "skipped (INGEST_ONLY)"
	}

	// Step 10 (time-varying seeds → FeatureSnapshots, §9b).
	if cfg.Runs("seeds") {
		if n, err := raw.LoadUNSC(ctx, rawDriver, cfg.ConfigDir); err != nil {
			res.note("unsc", err)
		} else {
			res.UNSC = n
			res.Sources["unsc"] = "ok"
		}
		if n, err := raw.LoadSanctions(ctx, rawDriver, cfg.ConfigDir); err != nil {
			res.note("sanctions", err)
		} else {
			res.Sanctions = n
			res.Sources["sanctions"] = "ok"
		}
		if n, err := raw.LoadActorFinancialTier(ctx, rawDriver, cfg.ConfigDir); err != nil {
			res.note("financial_tier", err)
		} else {
			res.FinancialTier = n
			res.Sources["financial_tier"] = "ok"
		}
	}

	nm := raw.BuildNameMap(countries)

	// Step 8 — GDELT events (BigQuery; needs GCP creds). Two modes:
	//  - "raw":        stream every EVENT edge into geopolitic_raw (16-48 GB).
	//  - "aggregated": aggregate in BigQuery, write only monthly SNAPSHOT_EDGEs
	//                  into geopolitic_aggregated AFTER Build (needs the mirrored
	//                  Country nodes). Default; fits a disk-constrained box.
	var gdeltAggCameo map[string]string
	gdeltEventMaxTS := 0
	if cfg.GDELTEnabled() && cfg.Runs("gdelt") {
		cameoMap, err := raw.LoadCameoMap(cfg.ConfigDir)
		if err != nil {
			res.note("gdelt", err)
		} else if cfg.GDELTMode == "raw" {
			if n, err := raw.LoadGDELT(ctx, rawDriver, cfg.GDELTProject, cfg.GDELTStart, cfg.GDELTEnd, cfg.GDELTMaxRows, cameoMap); err != nil {
				res.GDELTEvents = n
				res.note("gdelt", err)
			} else {
				res.GDELTEvents = n
				res.Sources["gdelt"] = "ok"
			}
		} else {
			gdeltAggCameo = cameoMap
			gdeltEventMaxTS = aggregated.GDELTMaxTimeStep(cfg.GDELTEnd)
		}
	} else {
		res.Sources["gdelt"] = "skipped (set GDELT_GCP_PROJECT)"
	}

	// Step 7 — ACLED conflict features (OAuth; needs myACLED creds).
	if cfg.ACLEDEnabled() && cfg.Runs("acled") {
		endYear := time.Now().Year()
		if n, err := raw.LoadACLED(ctx, rawDriver, cfg.HTTPClient, cfg.ACLEDTokenURL, cfg.ACLEDBaseURL, cfg.ACLEDEmail, cfg.ACLEDPassword, 2010, endYear, nm); err != nil {
			res.ACLEDFeatures = n
			res.note("acled", err)
		} else {
			res.ACLEDFeatures = n
			res.Sources["acled"] = "ok"
		}
	} else {
		res.Sources["acled"] = "skipped (set ACLED_EMAIL/ACLED_PASSWORD)"
	}

	// V-Dem / SIPRI / UNDP (file-based).
	if cfg.VDemEnabled() && cfg.Runs("vdem") {
		if n, err := raw.LoadVDem(ctx, rawDriver, cfg.VDemCSVPath); err != nil {
			res.note("vdem", err)
		} else {
			res.VDem = n
			res.Sources["vdem"] = "ok"
		}
	} else {
		res.Sources["vdem"] = "skipped (set VDEM_CSV_PATH)"
	}
	if cfg.SIPRIEnabled() && cfg.Runs("sipri") {
		if n, err := raw.LoadSIPRI(ctx, rawDriver, cfg.SIPRIXLSXPath, "", nm); err != nil {
			res.note("sipri", err)
		} else {
			res.SIPRI = n
			res.Sources["sipri"] = "ok"
		}
	} else {
		res.Sources["sipri"] = "skipped (set SIPRI_XLSX_PATH)"
	}
	if cfg.UNDPEnabled() && cfg.Runs("undp") {
		if n, err := raw.LoadUNDP(ctx, rawDriver, cfg.UNDPCSVPath); err != nil {
			res.note("undp", err)
		} else {
			res.UNDP = n
			res.Sources["undp"] = "ok"
		}
	} else {
		res.Sources["undp"] = "skipped (set UNDP_HDI_CSV_PATH)"
	}

	// Build aggregated DB from raw.
	aggStats, err := aggregated.Build(ctx, rawDriver, aggDriver, gdeltEventMaxTS)
	if err != nil {
		return res, fmt.Errorf("aggregate: %w", err)
	}
	res.Aggregated = aggStats

	// GDELT aggregated SNAPSHOT_EDGEs — runs after Build so the Country nodes the
	// edges attach to are already mirrored into geopolitic_aggregated.
	if gdeltAggCameo != nil {
		if n, err := aggregated.LoadGDELTSnapshotEdges(ctx, aggDriver, cfg.GDELTProject, cfg.GDELTStart, cfg.GDELTEnd, cfg.GDELTMaxRows, gdeltAggCameo); err != nil {
			res.GDELTSnapshotEdges = n
			res.Aggregated.SnapshotEdges += n
			res.note("gdelt", err)
		} else {
			res.GDELTSnapshotEdges = n
			res.Aggregated.SnapshotEdges += n
			res.Sources["gdelt"] = "ok"
		}
	}

	res.FinishedAt = time.Now()
	res.DurationMS = res.FinishedAt.Sub(res.StartedAt).Milliseconds()
	if len(res.Errors) > 0 {
		res.Status = "ok_with_warnings"
	}
	if err := recordState(ctx, rawDriver, res); err != nil {
		res.Errors = append(res.Errors, fmt.Sprintf("record state: %v", err))
	}
	return res, nil
}

// note records a non-fatal per-source failure.
func (r *Result) note(source string, err error) {
	r.Sources[source] = "error: " + err.Error()
	r.Errors = append(r.Errors, fmt.Sprintf("%s: %v", source, err))
	log.Printf("ingest source %s failed (non-fatal): %v", source, err)
}

func recordState(ctx context.Context, rawDriver neo4j.DriverWithContext, res Result) error {
	const cypher = `
MERGE (i:IngestState {id: "singleton"})
SET i.status = $status,
    i.last_run_at = $finished,
    i.duration_ms = $duration,
    i.countries = $countries,
    i.worldbank_snapshots = $wb,
    i.gdelt_events = $gdelt,
    i.acled_features = $acled,
    i.agg_node_snapshots = $node_snaps,
    i.agg_snapshot_edges = $snap_edges,
    i.errors = $errors`
	return neo4jdb.Write(ctx, rawDriver, cypher, map[string]any{
		"status":     res.Status,
		"finished":   res.FinishedAt.Format(time.RFC3339),
		"duration":   res.DurationMS,
		"countries":  res.Countries,
		"wb":         res.WorldBankSnapshots,
		"gdelt":      res.GDELTEvents,
		"acled":      res.ACLEDFeatures,
		"node_snaps": res.Aggregated.NodeSnapshots,
		"snap_edges": res.Aggregated.SnapshotEdges,
		"errors":     toAnySlice(res.Errors),
	})
}

func toAnySlice(xs []string) []any {
	out := make([]any, len(xs))
	for i, x := range xs {
		out[i] = x
	}
	return out
}
