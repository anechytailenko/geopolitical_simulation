package raw

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"geopolitic/internal/neo4jdb"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// wbCountry is one entry of the World Bank /country endpoint.
type wbCountry struct {
	ID     string `json:"id"`       // ISO-3
	ISO2   string `json:"iso2Code"` // ISO-2
	Name   string `json:"name"`
	Region struct {
		ID    string `json:"id"`
		Value string `json:"value"`
	} `json:"region"`
}

// FetchCountryList fetches the full sovereign-country list from the World Bank
// country endpoint (free, no key) without touching the database. Aggregate rows
// (region "Aggregates") and entries missing an ISO-2/ISO-3 are dropped.
func FetchCountryList(ctx context.Context, baseURL string, client *http.Client) ([]Country, error) {
	url := fmt.Sprintf("%s/country?format=json&per_page=400", baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch country list: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("country list status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var envelope []json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, fmt.Errorf("decode country envelope: %w", err)
	}
	if len(envelope) < 2 {
		return nil, fmt.Errorf("unexpected country list shape")
	}
	var raw []wbCountry
	if err := json.Unmarshal(envelope[1], &raw); err != nil {
		return nil, fmt.Errorf("decode countries: %w", err)
	}

	var countries []Country
	for _, c := range raw {
		// Aggregates (e.g. "World", "Euro area") carry region.value = "Aggregates".
		if c.Region.Value == "Aggregates" || c.ID == "" || c.ISO2 == "" {
			continue
		}
		countries = append(countries, Country{ISO3: c.ID, ISO2: c.ISO2, Name: c.Name, Region: c.Region.Value})
	}
	return countries, nil
}

// LoadCountryList fetches the full country list and MERGEs a Country identity
// node for each. It is the single source of the country universe (~195 states);
// `region` is provisional here and overwritten by the Wikidata P30 continent.
func LoadCountryList(ctx context.Context, driver neo4j.DriverWithContext, baseURL string, client *http.Client) ([]Country, error) {
	countries, err := FetchCountryList(ctx, baseURL, client)
	if err != nil {
		return nil, err
	}
	const cypher = `
MERGE (c:Country {id: $id})
SET c.iso2 = $iso2, c.name = $name, c.region = $region`
	for _, c := range countries {
		params := map[string]any{"id": c.ISO3, "iso2": c.ISO2, "name": c.Name, "region": c.Region}
		if err := neo4jdb.Write(ctx, driver, cypher, params); err != nil {
			return nil, fmt.Errorf("merge country %s: %w", c.ISO3, err)
		}
	}
	return countries, nil
}
