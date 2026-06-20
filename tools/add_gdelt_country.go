//go:build ignore

// add_gdelt_country surgically adds the GDELT SNAPSHOT_EDGEs for one country to
// geopolitic_aggregated, without re-running the full ~2.4M-edge pull. Use it when
// configs/cameo_country_to_iso3.json gains a country that was missing (e.g. TWN,
// which World Bank omits but Wikidata seeds as a Country node) so its events stop
// being dropped. The country must already exist as a Country node in the
// aggregated DB (mirrored by a prior Build) and must be a key in the cameo map.
//
//	go run tools/add_gdelt_country.go TWN
//
// Reads the same .env as the API (GDELT_GCP_PROJECT, AGG_URI, …). Idempotent
// (MERGE); a country that shares no edges with the existing set is purely additive.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"geopolitic/ingestion/aggregated"
	"geopolitic/ingestion/raw"
	"geopolitic/internal/config"
	"geopolitic/internal/neo4jdb"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: go run tools/add_gdelt_country.go <ISO3>")
	}
	focus := os.Args[1]

	cfg := config.FromEnv()
	if cfg.GDELTProject == "" {
		log.Fatal("GDELT_GCP_PROJECT not set (.env)")
	}
	cameoMap, err := raw.LoadCameoMap(cfg.ConfigDir)
	if err != nil {
		log.Fatalf("load cameo map: %v", err)
	}
	if _, ok := cameoMap[focus]; !ok {
		log.Fatalf("%s is not a key in configs/cameo_country_to_iso3.json — add it first", focus)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	aggDriver, err := neo4jdb.Connect(ctx, cfg.AggURI, cfg.AggUser, cfg.AggPass)
	if err != nil {
		log.Fatalf("connect aggregated: %v", err)
	}
	defer aggDriver.Close(ctx)

	n, err := aggregated.LoadGDELTSnapshotEdgesForCode(ctx, aggDriver, cfg.GDELTProject, cfg.GDELTStart, cfg.GDELTEnd, cfg.GDELTMaxRows, cameoMap, focus)
	if err != nil {
		log.Fatalf("load %s edges: %v", focus, err)
	}
	fmt.Printf("added %d SNAPSHOT_EDGEs involving %s\n", n, focus)
}
