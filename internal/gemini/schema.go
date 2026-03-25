// Package gemini provides shared utilities for Google Gemini API providers.
package gemini

import "fmt"

// SanitizeSchema sanitizes a JSON Schema for Gemini compatibility.
// It converts enum integer/number types to string, filters invalid required fields,
// and ensures array items have a type.
func SanitizeSchema(schema map[string]any) map[string]any {
	// sanitizeImpl always returns map[string]any for map input; cast is safe.
	return sanitizeImpl(schema).(map[string]any)
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
				m["type"] = nonNull
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
				m["type"] = nonNull
				result["nullable"] = true
			}
		}
	}

	for k, v := range m {
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
