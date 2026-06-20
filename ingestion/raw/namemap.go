package raw

import "strings"

// NameMap resolves source-specific country names (SIPRI, UNDP, ACLED) to ISO-3.
// It is built from the live World Bank country list plus a curated alias table
// for the names those sources spell differently from the World Bank.
type NameMap struct {
	byName map[string]string // normalized name -> ISO-3
}

// BuildNameMap indexes the loaded countries by normalized name and layers the
// common aliases on top.
func BuildNameMap(countries []Country) NameMap {
	m := NameMap{byName: map[string]string{}}
	for _, c := range countries {
		m.byName[normalizeName(c.Name)] = c.ISO3
	}
	for alias, iso3 := range nameAliases {
		m.byName[normalizeName(alias)] = iso3
	}
	return m
}

// Lookup returns the ISO-3 for a source country name, or ("", false).
func (m NameMap) Lookup(name string) (string, bool) {
	iso3, ok := m.byName[normalizeName(name)]
	return iso3, ok
}

func normalizeName(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// nameAliases maps common alternative spellings to ISO-3, independent of the
// (quirky) World Bank names. Extend as new source spellings surface.
var nameAliases = map[string]string{
	"United States of America":         "USA",
	"United States":                    "USA",
	"Russian Federation":               "RUS",
	"Russia":                           "RUS",
	"South Korea":                      "KOR",
	"Korea, Rep.":                      "KOR",
	"Republic of Korea":                "KOR",
	"North Korea":                      "PRK",
	"Korea, Dem. People's Rep.":        "PRK",
	"Iran":                             "IRN",
	"Iran, Islamic Rep.":               "IRN",
	"Iran (Islamic Republic of)":       "IRN",
	"Egypt":                            "EGY",
	"Egypt, Arab Rep.":                 "EGY",
	"Syria":                            "SYR",
	"Syrian Arab Republic":             "SYR",
	"Turkey":                           "TUR",
	"Turkiye":                          "TUR",
	"Venezuela":                        "VEN",
	"Venezuela, RB":                    "VEN",
	"Vietnam":                          "VNM",
	"Viet Nam":                         "VNM",
	"Laos":                             "LAO",
	"Lao PDR":                          "LAO",
	"Democratic Republic of Congo":     "COD",
	"Congo, Dem. Rep.":                 "COD",
	"DR Congo":                         "COD",
	"Republic of the Congo":            "COG",
	"Congo, Rep.":                      "COG",
	"Ivory Coast":                      "CIV",
	"Cote d'Ivoire":                    "CIV",
	"Tanzania":                         "TZA",
	"Bolivia":                          "BOL",
	"Bolivia (Plurinational State of)": "BOL",
	"Czechia":                          "CZE",
	"Czech Republic":                   "CZE",
	"Slovakia":                         "SVK",
	"Slovak Republic":                  "SVK",
	"United Kingdom":                   "GBR",
	"Gambia":                           "GMB",
	"Gambia, The":                      "GMB",
	"Kyrgyzstan":                       "KGZ",
	"Kyrgyz Republic":                  "KGZ",
	"Yemen":                            "YEM",
	"Yemen, Rep.":                      "YEM",
}
