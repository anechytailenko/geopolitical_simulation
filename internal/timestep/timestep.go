// Package timestep implements the canonical time-step convention shared across
// the ingestion pipeline (and mirrored in the Python ML service).
//
// time_step is a single integer = months since the 2010-01 epoch:
//
//	time_step = (year - 2010) * 12 + (month - 1)   // t=0 -> 2010-01
//
// year, month and iso_period are pure functions of time_step, so there is no
// separate TimeStep node anywhere in the graph (see plans/01-architecture.md).
package timestep

import "fmt"

// Epoch is the calendar year that maps to time_step 0 (January).
const Epoch = 2010

// FromYM converts a (year, month) pair to a time_step. month is 1-12.
func FromYM(year, month int) int {
	return (year-Epoch)*12 + (month - 1)
}

// FromYear converts a calendar year to the time_step of its January.
func FromYear(year int) int {
	return FromYM(year, 1)
}

// Year returns the calendar year for a time_step.
func Year(ts int) int {
	return Epoch + floorDiv(ts, 12)
}

// Month returns the 1-12 calendar month for a time_step.
func Month(ts int) int {
	return floorMod(ts, 12) + 1
}

// ISOPeriod returns the "YYYY-MM" string for a time_step (e.g. "2013-07").
func ISOPeriod(ts int) string {
	return fmt.Sprintf("%04d-%02d", Year(ts), Month(ts))
}

// ClampStart clamps a start time_step to the epoch: structural edges that began
// before 2010 clamp to start_time_step = 0 (see Wikidata section of the plan).
func ClampStart(ts int) int {
	if ts < 0 {
		return 0
	}
	return ts
}

// floorDiv / floorMod give Python-style flooring division so that negative
// time_steps (pre-2010 dates before clamping) still map to the right calendar.
func floorDiv(a, b int) int {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

func floorMod(a, b int) int {
	m := a % b
	if m != 0 && ((m < 0) != (b < 0)) {
		m += b
	}
	return m
}
