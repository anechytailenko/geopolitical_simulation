//go:build ignore

// gen_cameo regenerates configs/cameo_country_to_iso3.json, aligned to the exact
// country universe we ingest (the live World Bank country list).
//
// Empirically (verified against live GDELT 1.0 + 2.0 export files), GDELT's
// Actor1CountryCode/Actor2CountryCode use ISO-3 country codes — NOT the FIPS-style
// CAMEO codes (GMY/UKG/FRN) often assumed. So the map is identity over our ISO-3
// countries, plus any code variant from the authoritative CAMEO.country.txt whose
// label resolves to one of our countries. Regional/sub-national codes (EUR, AFR,
// WSB, ...) have no country match and are intentionally excluded.
//
// Run: go run tools/gen_cameo.go
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"geopolitic/ingestion/raw"
)

const (
	wbBase   = "https://api.worldbank.org/v2"
	cameoURL = "https://www.gdeltproject.org/data/lookups/CAMEO.country.txt"
	outPath  = "configs/cameo_country_to_iso3.json"
)

func main() {
	ctx := context.Background()
	client := &http.Client{Timeout: 60 * time.Second}

	countries, err := raw.FetchCountryList(ctx, wbBase, client)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fetch countries:", err)
		os.Exit(1)
	}
	nm := raw.BuildNameMap(countries)

	m := map[string]string{}
	iso := map[string]bool{}
	for _, c := range countries {
		m[c.ISO3] = c.ISO3 // GDELT uses ISO-3 codes -> identity for every inserted country
		iso[c.ISO3] = true
	}

	// World Bank omits a few states that Wikidata seeds as Country nodes and that
	// GDELT actively codes — most importantly Taiwan (TWN). Add them so a
	// regeneration stays aligned with the ingested country set (otherwise their
	// events would be silently dropped).
	for _, code := range []string{"TWN"} {
		if _, ok := m[code]; !ok {
			m[code] = code
			iso[code] = true
		}
	}

	// Cross-reference the authoritative CAMEO list to catch any non-ISO code whose
	// label resolves to one of our countries (and to confirm the ISO ones).
	added, skipped := 0, 0
	for code, label := range fetchCameo(client) {
		if _, ok := m[code]; ok {
			continue
		}
		if iso3, ok := nm.Lookup(label); ok {
			m[code] = iso3
			added++
		} else {
			skipped++ // regions / sub-national / unmatched
		}
	}

	data, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(outPath, append(data, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s: %d entries (%d identity from %d countries, %d label-matched, %d CAMEO codes skipped)\n",
		outPath, len(m), len(iso), len(countries), added, skipped)
}

func fetchCameo(client *http.Client) map[string]string {
	out := map[string]string{}
	resp, err := client.Get(cameoURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "warn: CAMEO list fetch failed, identity map only:", err)
		return out
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	first := true
	for sc.Scan() {
		line := sc.Text()
		if first { // header "CODE\tLABEL"
			first = false
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			continue
		}
		out[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
	}
	return out
}
