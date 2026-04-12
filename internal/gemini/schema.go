// Package gemini provides shared utilities for Google Gemini API providers.
package gemini

import (
	"fmt"
	"strings"
)

// maxRefInlineDepth caps recursive $ref resolution to prevent infinite loops
// on cyclic schemas. Real-world schemas rarely exceed 3-4 levels of nesting.
const maxRefInlineDepth = 10

// SanitizeSchema sanitizes a JSON Schema for Gemini compatibility.
// It converts enum integer/number types to string, filters invalid required fields,
// ensures array items have a type, inlines $ref/$defs (which Gemini rejects),
// and strips JSON Schema conditionals (if/then/else) which Gemini's OpenAPI
// validator does not understand.
func SanitizeSchema(schema map[string]any) map[string]any {
	// Step 1: resolve $ref → $defs and strip unsupported keywords.
	// Gemini's function_declarations validator cannot resolve
	// "#/$defs/X" references and rejects JSON Schema conditional keywords
	// (if/then/else) even inside allOf clauses. This preprocess pass
	// handles both issues before the existing compatibility sanitizer runs.
	defs, _ := schema["$defs"].(map[string]any)
	processed := inlineRefsAndStripDrafts(schema, defs, 0)
	if m, ok := processed.(map[string]any); ok {
		schema = m
	}
	// Step 2: existing compatibility pass (nullables, enums, array items, etc.).
	return sanitizeImpl(schema).(map[string]any)
}

// inlineRefsAndStripDrafts walks an arbitrary schema value and:
//   - Replaces every {"$ref": "#/$defs/X"} with a deep copy of $defs.X
//     (recursively, so nested refs are also resolved).
//   - Strips "$defs", "if", "then", "else" keys anywhere they appear.
//   - Bails out at maxRefInlineDepth to prevent cyclic-ref infinite loops;
//     at the depth cap, unresolved refs become {type: "object"}.
//
// The function does not mutate its input: new maps/slices are allocated
// for every container so callers can safely reuse the original schema.
func inlineRefsAndStripDrafts(obj any, defs map[string]any, depth int) any {
	if obj == nil {
		return nil
	}
	switch v := obj.(type) {
	case map[string]any:
		// $ref resolution: replace with the deep-cloned target.
		if ref, ok := v["$ref"].(string); ok {
			if depth >= maxRefInlineDepth {
				// Cap reached: fall back to a generic type rather than
				// infinite-recurse or panic.
				return map[string]any{"type": "object"}
			}
			if target := resolveDefRef(ref, defs); target != nil {
				return inlineRefsAndStripDrafts(target, defs, depth+1)
			}
			// Dangling ref (not in $defs): fall through to generic type.
			return map[string]any{"type": "object"}
		}
		// Recurse into every key except the blocklisted ones.
		result := make(map[string]any, len(v))
		for k, val := range v {
			if k == "$defs" || k == "if" || k == "then" || k == "else" {
				continue
			}
			result[k] = inlineRefsAndStripDrafts(val, defs, depth)
		}
		return result
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = inlineRefsAndStripDrafts(item, defs, depth)
		}
		return out
	default:
		return obj
	}
}

// resolveDefRef returns the target map for a "#/$defs/X" reference, or nil
// if the ref does not point into the provided $defs table. Only the local
// "#/$defs/" JSON Pointer form is supported: external URIs and deep
// pointers are returned as nil (and inlineRefsAndStripDrafts will fall
// back to a generic type).
func resolveDefRef(ref string, defs map[string]any) any {
	const prefix = "#/$defs/"
	if defs == nil || !strings.HasPrefix(ref, prefix) {
		return nil
	}
	key := ref[len(prefix):]
	return defs[key]
}

func sanitizeImpl(obj any) any {
	if obj == nil {
		return nil
	}

	arr, isArr := obj.([]any)
	if isArr {
		result := make([]any, len(arr))
		for i, v := range arr {
			result[i] = sanitizeImpl(v)
		}
		return result
	}

	m, isMap := obj.(map[string]any)
	if !isMap {
		return obj
	}

	result := make(map[string]any)

	// Normalize nullable type arrays: ["object","null"] → "object" + nullable: true.
	// SchemaFrom produces []string, JSON unmarshal produces []any.
	if rawType, ok := m["type"]; ok {
		switch tt := rawType.(type) {
		case []string:
			var nonNull string
			for _, s := range tt {
				if s != "null" {
					nonNull = s
				}
			}
			if nonNull != "" {
				result["type"] = nonNull
				result["nullable"] = true
			}
		case []any:
			var nonNull string
			for _, v := range tt {
				if s, ok := v.(string); ok && s != "null" {
					nonNull = s
				}
			}
			if nonNull != "" {
				result["type"] = nonNull
				result["nullable"] = true
			}
		}
	}

	for k, v := range m {
		// Skip "type" if we already normalized it from a nullable array above.
		if k == "type" {
			if _, already := result["type"]; already {
				continue
			}
		}
		if k == "enum" {
			if enumArr, ok := v.([]any); ok {
				strEnum := make([]any, len(enumArr))
				for i, ev := range enumArr {
					strEnum[i] = fmt.Sprint(ev)
				}
				result[k] = strEnum
				continue
			}
		}

		switch vv := v.(type) {
		case map[string]any:
			result[k] = sanitizeImpl(vv)
		case []any:
			result[k] = sanitizeImpl(vv)
		default:
			result[k] = v
		}
	}

	// If enum present with integer/number type, convert to string.
	if _, hasEnum := result["enum"]; hasEnum {
		if t, ok := result["type"]; ok {
			if t == "integer" || t == "number" {
				result["type"] = "string"
			}
		}
	}

	// Filter required to only include fields in properties.
	// SchemaFrom produces []string; JSON unmarshal produces []any.
	if result["type"] == "object" {
		if props, ok := result["properties"].(map[string]any); ok {
			switch req := result["required"].(type) {
			case []any:
				filtered := make([]any, 0)
				for _, r := range req {
					if s, ok := r.(string); ok {
						if _, exists := props[s]; exists {
							filtered = append(filtered, r)
						}
					}
				}
				result["required"] = filtered
			case []string:
				filtered := make([]any, 0)
				for _, s := range req {
					if _, exists := props[s]; exists {
						filtered = append(filtered, s)
					}
				}
				result["required"] = filtered
			}
		}
	}

	// Ensure array items has type.
	if result["type"] == "array" {
		if result["items"] == nil {
			result["items"] = map[string]any{}
		}
		if items, ok := result["items"].(map[string]any); ok {
			if _, hasType := items["type"]; !hasType {
				items["type"] = "string"
				result["items"] = items
			}
		}
	}

	// Remove properties/required from non-object types.
	if t, ok := result["type"]; ok && t != "object" {
		delete(result, "properties")
		delete(result, "required")
	}

	// Gemini API does not support additionalProperties or format in JSON Schema.
	delete(result, "additionalProperties")
	delete(result, "format")

	return result
}
