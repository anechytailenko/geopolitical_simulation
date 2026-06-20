package config

import (
	"bufio"
	"os"
	"strings"
)

// loadDotEnv loads KEY=VALUE pairs from a .env file into the process environment,
// without overriding variables that are already set (explicit exports win). A
// missing file is not an error. Lines starting with '#' and blank lines are
// ignored; surrounding single/double quotes on values are stripped.
//
// This is called by FromEnv so every entrypoint (API server, tools) automatically
// picks up the project's .env without an external dependency or a wrapper script.
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq < 0 {
			continue
		}
		key := strings.TrimSpace(line[:eq])
		val := strings.TrimSpace(line[eq+1:])
		if len(val) >= 2 && (val[0] == '"' || val[0] == '\'') && val[len(val)-1] == val[0] {
			val = val[1 : len(val)-1]
		}
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		_ = os.Setenv(key, val)
	}
}
