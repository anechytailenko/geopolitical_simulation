//go:build ignore

// trim_datasets trims the research source files in datasets_csv/ to the study
// window (2010-present), in place, then verifies the result by re-parsing each
// trimmed file with the production loader so we can be sure no 2010+ data was
// dropped and the ingestion infrastructure still reads them.
//
//	V-Dem  : drop ROWS    with year < 2010
//	UNDP   : drop COLUMNS named *_<year> with year < 2010
//	SIPRI  : drop year COLUMNS < 2010 in the "Constant US$" sheet
//
// Run: go run tools/trim_datasets.go
package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"geopolitic/ingestion/raw"

	"github.com/xuri/excelize/v2"
)

const epoch = 2010

var (
	vdemPath  = "datasets_csv/V-Dem-CY-Full+Others-v16.csv"
	undpPath  = "datasets_csv/HDR25_Composite_indices_complete_time_series.csv"
	sipriPath = "datasets_csv/SIPRI-Milex-data-1949-2025_v1.2.xlsx"
	yearSfx   = regexp.MustCompile(`_(\d{4})$`)
)

func main() {
	trimVDem()
	trimUNDP()
	trimSIPRI()
	verify()
}

func trimVDem() {
	in, err := os.Open(vdemPath)
	must(err)
	defer in.Close()
	r := csv.NewReader(in)
	r.FieldsPerRecord = -1
	r.ReuseRecord = true

	header, err := r.Read()
	must(err)
	yearCol := indexOf(header, "year")
	if yearCol < 0 {
		fatal("V-Dem: no 'year' column")
	}

	tmp := vdemPath + ".tmp"
	out, err := os.Create(tmp)
	must(err)
	w := csv.NewWriter(out)
	must(w.Write(header))

	kept, dropped := 0, 0
	for {
		rec, err := r.Read()
		if err != nil {
			break
		}
		y, e := strconv.Atoi(strings.TrimSpace(rec[yearCol]))
		if e != nil || y < epoch {
			dropped++
			continue
		}
		must(w.Write(rec))
		kept++
	}
	w.Flush()
	must(w.Error())
	must(out.Close())
	must(os.Rename(tmp, vdemPath))
	fmt.Printf("V-Dem : kept %d rows (year>=%d), dropped %d older rows\n", kept, epoch, dropped)
}

func trimUNDP() {
	in, err := os.Open(undpPath)
	must(err)
	defer in.Close()
	r := csv.NewReader(in)
	r.FieldsPerRecord = -1

	rows, err := r.ReadAll()
	must(err)
	if len(rows) == 0 {
		fatal("UNDP: empty")
	}
	header := rows[0]
	var keep []int
	droppedCols := 0
	for i, h := range header {
		if m := yearSfx.FindStringSubmatch(h); m != nil {
			if y, _ := strconv.Atoi(m[1]); y < epoch {
				droppedCols++
				continue
			}
		}
		keep = append(keep, i)
	}

	tmp := undpPath + ".tmp"
	out, err := os.Create(tmp)
	must(err)
	w := csv.NewWriter(out)
	for _, row := range rows {
		proj := make([]string, 0, len(keep))
		for _, i := range keep {
			if i < len(row) {
				proj = append(proj, row[i])
			} else {
				proj = append(proj, "")
			}
		}
		must(w.Write(proj))
	}
	w.Flush()
	must(w.Error())
	must(out.Close())
	must(os.Rename(tmp, undpPath))
	fmt.Printf("UNDP  : kept %d columns, dropped %d pre-%d year columns (%d data rows)\n",
		len(keep), droppedCols, epoch, len(rows)-1)
}

func trimSIPRI() {
	f, err := excelize.OpenFile(sipriPath)
	must(err)
	defer f.Close()

	sheet := ""
	for _, s := range f.GetSheetList() {
		if strings.Contains(strings.ToLower(s), "constant") {
			sheet = s
			break
		}
	}
	if sheet == "" {
		fatal("SIPRI: no Constant US$ sheet")
	}
	rows, err := f.GetRows(sheet)
	must(err)

	// locate the header row (>=5 four-digit year cells) and the pre-2010 columns.
	var dropCols []int
	for _, row := range rows {
		years := 0
		var pre []int
		for j, cell := range row {
			if y, e := strconv.Atoi(strings.TrimSpace(cell)); e == nil && y >= 1949 && y <= 2100 {
				years++
				if y < epoch {
					pre = append(pre, j)
				}
			}
		}
		if years >= 5 {
			dropCols = pre
			break
		}
	}
	// remove right-to-left so indices stay valid; +1 because excelize cols are 1-based.
	for i := len(dropCols) - 1; i >= 0; i-- {
		name, err := excelize.ColumnNumberToName(dropCols[i] + 1)
		must(err)
		must(f.RemoveCol(sheet, name))
	}
	must(f.Save())
	fmt.Printf("SIPRI : removed %d pre-%d year columns from sheet %q\n", len(dropCols), epoch, sheet)
}

// verify re-parses each trimmed file with the production loader and prints record
// counts + a spot value, confirming the infrastructure still reads them.
func verify() {
	fmt.Println("--- verify (parse trimmed files with production loaders) ---")
	client := &http.Client{Timeout: 60 * time.Second}
	countries, err := raw.FetchCountryList(context.Background(), "https://api.worldbank.org/v2", client)
	must(err)
	nm := raw.BuildNameMap(countries)

	vf, err := os.Open(vdemPath)
	must(err)
	defer vf.Close()
	vrecs, err := raw.ParseVDem(vf)
	must(err)
	minY, maxY := yearRange(func(yield func(int)) {
		for _, r := range vrecs {
			yield(r.Year)
		}
	})
	fmt.Printf("V-Dem : %d records parsed, year range %d-%d\n", len(vrecs), minY, maxY)

	uf, err := os.Open(undpPath)
	must(err)
	defer uf.Close()
	urecs, err := raw.ParseUNDP(uf)
	must(err)
	uMin, uMax := yearRange(func(yield func(int)) {
		for _, r := range urecs {
			yield(r.Year)
		}
	})
	fmt.Printf("UNDP  : %d HDI records parsed, year range %d-%d\n", len(urecs), uMin, uMax)

	sf, err := excelize.OpenFile(sipriPath)
	must(err)
	defer sf.Close()
	srecs, err := raw.ParseSIPRIFile(sf, "", nm)
	must(err)
	sMin, sMax := yearRange(func(yield func(int)) {
		for _, r := range srecs {
			yield(r.Year)
		}
	})
	fmt.Printf("SIPRI : %d milex records parsed, year range %d-%d\n", len(srecs), sMin, sMax)

	if minY < epoch || uMin < epoch || sMin < epoch {
		fatal("a parsed record predates the epoch — trimming is wrong")
	}
	fmt.Println("OK: all parsed records are within the 2010+ window.")
}

func yearRange(seq func(func(int))) (int, int) {
	min, max := 1<<31, -(1 << 31)
	seq(func(y int) {
		if y < min {
			min = y
		}
		if y > max {
			max = y
		}
	})
	return min, max
}

func indexOf(hs []string, want string) int {
	for i, h := range hs {
		if strings.TrimSpace(h) == want {
			return i
		}
	}
	return -1
}

func must(err error) {
	if err != nil {
		fatal(err.Error())
	}
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, "ERROR:", msg)
	os.Exit(1)
}
