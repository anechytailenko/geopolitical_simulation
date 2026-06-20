// Package label maps raw CAMEO event codes (used by both GDELT and ICEWS) to
// the five relationship classes the model predicts. See plans/01-architecture.md
// "Label Generation".
package label

import "strconv"

// The five relationship classes, in canonical order. The index of each class in
// this slice is the index used by class_distribution (Float[5]) and the flattened
// class_transition_vector (Float[25]) on SNAPSHOT_EDGE.
const (
	MaterialConflict    = "MATERIAL_CONFLICT"
	VerbalConflict      = "VERBAL_CONFLICT"
	MaterialCooperation = "MATERIAL_COOPERATION"
	VerbalCooperation   = "VERBAL_COOPERATION"
	StatusQuo           = "STATUS_QUO"
)

// Classes is the canonical ordering. Do not reorder — indices are persisted.
var Classes = []string{
	MaterialConflict,
	VerbalConflict,
	MaterialCooperation,
	VerbalCooperation,
	StatusQuo,
}

// Index returns the canonical position of a class, or -1 if unknown.
func Index(class string) int {
	for i, c := range Classes {
		if c == class {
			return i
		}
	}
	return -1
}

// Classify maps a CAMEO event code + Goldstein scale to a relationship class.
//
// CAMEO codes are 2-4 digit strings whose leading two digits are the "root" code
// (01..20). The rules below follow plans/01-architecture.md:
//
//	MATERIAL_CONFLICT    root 18/19/20, or Goldstein < -5
//	VERBAL_CONFLICT      root 11..17 (demands/disapproval/rejection/threats/coercion)
//	MATERIAL_COOPERATION root 06..08 (material cooperation, aid, yield)
//	VERBAL_COOPERATION   root 01..05, or Goldstein > +5
//	STATUS_QUO           everything else (incl. 09 investigate, 10 demand-neutral)
func Classify(cameoCode string, goldstein float64) string {
	root := rootCode(cameoCode)

	switch {
	case root >= 18 && root <= 20:
		return MaterialConflict
	case goldstein < -5:
		return MaterialConflict
	case root >= 11 && root <= 17:
		return VerbalConflict
	case root >= 6 && root <= 8:
		return MaterialCooperation
	case root >= 1 && root <= 5:
		return VerbalCooperation
	case goldstein > 5:
		return VerbalCooperation
	default:
		return StatusQuo
	}
}

// rootCode extracts the leading 1-2 digit CAMEO root (01..20) from an event code.
// Returns 0 for unparseable codes (-> STATUS_QUO via Classify).
func rootCode(code string) int {
	if len(code) < 2 {
		if n, err := strconv.Atoi(code); err == nil {
			return n
		}
		return 0
	}
	n, err := strconv.Atoi(code[:2])
	if err != nil {
		return 0
	}
	return n
}
