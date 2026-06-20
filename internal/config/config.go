// Package config holds the runtime configuration for the ingestion service,
// populated from environment variables with local-dev defaults that match the
// docker-compose setup in infra/docker.
//
// The credential-free core (World Bank country list + indicators, Wikidata
// structural edges) always runs. Each credentialed source activates only when
// its env vars / file paths are set — see the *Enabled() helpers.
package config

import (
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config describes how to reach both Neo4j databases and every external source.
type Config struct {
	// geopolitic_raw — event-level store (its own Neo4j container).
	RawURI  string
	RawUser string
	RawPass string

	// geopolitic_aggregated — snapshot store the ML service reads from.
	AggURI  string
	AggUser string
	AggPass string

	// ConfigDir holds the static seed JSON files (configs/*.json).
	ConfigDir string

	// HTTPClient is shared by all HTTP-based loaders.
	HTTPClient *http.Client

	// --- credential-free core ---
	WorldBankBaseURL string // overridable for tests
	WikidataEndpoint string // SPARQL endpoint
	WikidataEnabled  bool   // default true; set WIKIDATA_ENABLED=0 to skip

	// --- GDELT via BigQuery (events) ---
	GDELTProject string // GDELT_GCP_PROJECT; empty disables GDELT
	GDELTStart   string // GDELT_START_DATE, YYYY-MM-DD (default 2010-01-01)
	GDELTEnd     string // GDELT_END_DATE, YYYY-MM-DD (default: today)
	GDELTMaxRows int    // GDELT_MAX_ROWS, 0 = unbounded
	// GDELTMode selects how GDELT events land: "aggregated" (default) aggregates
	// in BigQuery and writes only monthly SNAPSHOT_EDGEs to geopolitic_aggregated
	// (~2.4M edges, fits a local disk); "raw" streams every EVENT edge into
	// geopolitic_raw (16-48 GB — only with ample disk). GDELT_MODE.
	GDELTMode string

	// --- ACLED (conflict-intensity node feature; OAuth since 2025) ---
	ACLEDEmail    string // ACLED_EMAIL (myACLED account)
	ACLEDPassword string // ACLED_PASSWORD
	ACLEDTokenURL string // OAuth token endpoint (overridable for tests)
	ACLEDBaseURL  string // read endpoint (overridable for tests)

	// --- file-based sources (user downloads the file, we read the path) ---
	VDemCSVPath   string // VDEM_CSV_PATH
	SIPRIXLSXPath string // SIPRI_XLSX_PATH
	UNDPCSVPath   string // UNDP_HDI_CSV_PATH

	// OnlySources, if non-empty, restricts the pipeline to the listed source keys
	// (INGEST_ONLY="gdelt" lets you add events later without re-pulling World Bank).
	// Empty = run everything. Aggregation always runs.
	OnlySources map[string]bool
}

// FromEnv builds a Config from environment variables, first loading a local
// .env file (if present) so credentials/paths are picked up automatically.
func FromEnv() Config {
	loadDotEnv(env("DOTENV_PATH", ".env"))
	return Config{
		RawURI:  env("RAW_URI", "bolt://localhost:7687"),
		RawUser: env("RAW_USER", "neo4j"),
		RawPass: env("RAW_PASS", "geopolitic"),
		AggURI:  env("AGG_URI", "bolt://localhost:7688"),
		AggUser: env("AGG_USER", "neo4j"),
		AggPass: env("AGG_PASS", "geopolitic"),

		ConfigDir:  env("CONFIG_DIR", "configs"),
		HTTPClient: &http.Client{Timeout: 120 * time.Second},

		WorldBankBaseURL: env("WORLDBANK_BASE_URL", "https://api.worldbank.org/v2"),
		WikidataEndpoint: env("WIKIDATA_ENDPOINT", "https://query.wikidata.org/sparql"),
		WikidataEnabled:  env("WIKIDATA_ENABLED", "1") != "0",

		GDELTProject: env("GDELT_GCP_PROJECT", ""),
		GDELTStart:   env("GDELT_START_DATE", "2010-01-01"),
		GDELTEnd:     env("GDELT_END_DATE", ""),
		GDELTMaxRows: envInt("GDELT_MAX_ROWS", 0),
		GDELTMode:    env("GDELT_MODE", "aggregated"),

		ACLEDEmail:    env("ACLED_EMAIL", ""),
		ACLEDPassword: env("ACLED_PASSWORD", ""),
		ACLEDTokenURL: env("ACLED_TOKEN_URL", "https://acleddata.com/oauth/token"),
		ACLEDBaseURL:  env("ACLED_BASE_URL", "https://acleddata.com/api/acled/read"),

		VDemCSVPath:   env("VDEM_CSV_PATH", ""),
		SIPRIXLSXPath: env("SIPRI_XLSX_PATH", ""),
		UNDPCSVPath:   env("UNDP_HDI_CSV_PATH", ""),

		OnlySources: parseCSVSet(env("INGEST_ONLY", "")),
	}
}

// Runs reports whether a named source should run given INGEST_ONLY.
func (c Config) Runs(source string) bool {
	if len(c.OnlySources) == 0 {
		return true
	}
	return c.OnlySources[source]
}

func parseCSVSet(s string) map[string]bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	out := map[string]bool{}
	for _, p := range strings.Split(s, ",") {
		if p = strings.ToLower(strings.TrimSpace(p)); p != "" {
			out[p] = true
		}
	}
	return out
}

// GDELTEnabled reports whether the GDELT BigQuery loader should run.
func (c Config) GDELTEnabled() bool { return c.GDELTProject != "" }

// ACLEDEnabled reports whether the ACLED loader has OAuth credentials.
func (c Config) ACLEDEnabled() bool { return c.ACLEDEmail != "" && c.ACLEDPassword != "" }

// VDemEnabled / SIPRIEnabled / UNDPEnabled gate the file-based loaders.
func (c Config) VDemEnabled() bool  { return c.VDemCSVPath != "" }
func (c Config) SIPRIEnabled() bool { return c.SIPRIXLSXPath != "" }
func (c Config) UNDPEnabled() bool  { return c.UNDPCSVPath != "" }

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
