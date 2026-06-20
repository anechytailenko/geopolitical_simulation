// Command api exposes the data-gathering trigger endpoint. It is the "another
// folder" referenced in the task: the HTTP entrypoint that kicks off ingestion
// into both Neo4j databases.
//
//	POST /api/v1/ingest          run the full ingest + aggregation pipeline
//	GET  /api/v1/ingest/status   report the last run (from the IngestState node)
//	GET  /healthz                liveness probe
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"geopolitic/ingestion"
	"geopolitic/internal/config"
	"geopolitic/internal/neo4jdb"
)

type server struct {
	cfg     config.Config
	running sync.Mutex // serialize ingest runs
}

func main() {
	srv := &server{cfg: config.FromEnv()}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/ingest", srv.handleIngest)
	mux.HandleFunc("GET /api/v1/ingest/status", srv.handleStatus)
	mux.HandleFunc("GET /healthz", srv.handleHealth)

	addr := ":" + env("PORT", "8080")
	log.Printf("ingestion API listening on %s (raw=%s agg=%s)", addr, srv.cfg.RawURI, srv.cfg.AggURI)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}

// handleIngest runs the pipeline synchronously and returns the Result. A 409 is
// returned if another ingest is already in progress.
func (s *server) handleIngest(w http.ResponseWriter, r *http.Request) {
	if !s.running.TryLock() {
		writeJSON(w, http.StatusConflict, map[string]string{"error": "ingest already in progress"})
		return
	}
	defer s.running.Unlock()

	// Full real ingests (especially GDELT + ACLED) run for a long time; cap
	// generously and allow override via INGEST_TIMEOUT_MINUTES (default 12h).
	timeout := time.Duration(envInt("INGEST_TIMEOUT_MINUTES", 720)) * time.Minute
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	res, err := ingestion.Run(ctx, s.cfg)
	if err != nil {
		log.Printf("ingest failed: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]any{
			"status": "error",
			"error":  err.Error(),
		})
		return
	}
	log.Printf("ingest %s: countries=%d gdelt_events=%d node_snapshots=%d snapshot_edges=%d errors=%d",
		res.Status, res.Countries, res.GDELTEvents, res.Aggregated.NodeSnapshots, res.Aggregated.SnapshotEdges, len(res.Errors))
	writeJSON(w, http.StatusOK, res)
}

// handleStatus reads the singleton IngestState node from geopolitic_raw.
func (s *server) handleStatus(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	driver, err := neo4jdb.Connect(ctx, s.cfg.RawURI, s.cfg.RawUser, s.cfg.RawPass)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	defer driver.Close(ctx)

	rows, err := neo4jdb.ReadRows(ctx, driver,
		`MATCH (i:IngestState {id: "singleton"}) RETURN properties(i) AS props`, nil)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	if len(rows) == 0 {
		writeJSON(w, http.StatusNotFound, map[string]string{"status": "never_run"})
		return
	}
	writeJSON(w, http.StatusOK, rows[0]["props"])
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(body)
}

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
