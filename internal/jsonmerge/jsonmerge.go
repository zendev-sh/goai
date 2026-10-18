// Package jsonmerge overlays caller-supplied JSON objects on SDK-built request
// bodies without mutating either input.
package jsonmerge

// Deep returns base with overlay applied: nested objects merge key by key at
// every level, and any other overlay value replaces the base value.
func Deep(base, overlay map[string]any) map[string]any {
	merged := make(map[string]any, len(base)+len(overlay))
	for k, v := range base {
		merged[k] = v
	}
	for k, v := range overlay {
		existing, baseIsMap := merged[k].(map[string]any)
		next, overlayIsMap := v.(map[string]any)
		if baseIsMap && overlayIsMap {
			merged[k] = Deep(existing, next)
			continue
		}
		merged[k] = v
	}
	return merged
}
