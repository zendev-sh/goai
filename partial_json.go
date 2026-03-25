package goai

import (
	"encoding/json"
	"strings"
)

// parsePartialJSON attempts to repair and parse incomplete JSON into type T.
// It handles truncated strings, arrays, objects, numbers, and keywords.
func parsePartialJSON[T any](incomplete string) (*T, error) {
	// Try parsing as-is first.
	var result T
	if err := json.Unmarshal([]byte(incomplete), &result); err == nil {
		return &result, nil
	}

	// Try repairing.
	repaired := repairJSON(incomplete)
	var result2 T
	if err := json.Unmarshal([]byte(repaired), &result2); err != nil {
		return nil, err
	}
	return &result2, nil
}

// repairJSON attempts to fix incomplete JSON by closing open structures.
func repairJSON(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "{}"
	}

	var buf strings.Builder
	var stack []byte // Stack of opening chars: '{' or '['
	inString := false
	escaped := false

	for i := range len(s) {
		c := s[i]

		if escaped {
			buf.WriteByte(c)
			escaped = false
			continue
		}

		if inString {
			if c == '\\' {
				buf.WriteByte(c)
				escaped = true
				continue
			}
			if c == '"' {
				buf.WriteByte(c)
				inString = false
				continue
			}
			buf.WriteByte(c)
			continue
		}

		switch c {
		case '"':
			buf.WriteByte(c)
			inString = true
		case '{':
			buf.WriteByte(c)
			stack = append(stack, '{')
		case '}':
			if len(stack) > 0 && stack[len(stack)-1] == '{' {
				buf.WriteByte(c)
				stack = stack[:len(stack)-1]
			}
		case '[':
			buf.WriteByte(c)
			stack = append(stack, '[')
		case ']':
			if len(stack) > 0 && stack[len(stack)-1] == '[' {
				buf.WriteByte(c)
				stack = stack[:len(stack)-1]
			}
		default:
			buf.WriteByte(c)
		}
	}

	result := buf.String()

	// Close open string.
	if inString || escaped {
		// If we ended mid-escape (trailing backslash), drop the backslash
		// so closing with `"` doesn't produce `\"` (escaped quote).
		if escaped && len(result) > 0 && result[len(result)-1] == '\\' {
			result = result[:len(result)-1]
		}

		// Strip incomplete \uXXXX escape sequences at the end of the string.
		// A valid unicode escape is exactly \uXXXX (6 chars). If we see \u
		// followed by fewer than 4 hex digits at the tail, remove the partial escape.
		if idx := strings.LastIndex(result, `\u`); idx >= 0 {
			tail := result[idx:]
			if len(tail) < 6 { // incomplete \uXXXX
				result = result[:idx]
			}
		}

		result += `"`
	}

	// Complete truncated keywords and numbers.
	result = completeTrailing(result)

	// Close open containers.
	for i := len(stack) - 1; i >= 0; i-- {
		result = trimTrailingIncomplete(result)
		if stack[i] == '{' {
			result += "}"
		} else {
			result += "]"
		}
	}

	return result
}

// completeTrailing completes truncated JSON keywords and numbers.
func completeTrailing(s string) string {
	trimmed := strings.TrimRight(s, " \t\n\r")
	if trimmed == "" {
		return s
	}

	// Complete truncated keywords.
	keywords := []string{"true", "false", "null"}
	for _, kw := range keywords {
		// Check exact match first (complete keyword should not fall through to number handling).
		if strings.HasSuffix(trimmed, kw) {
			pos := len(trimmed) - len(kw)
			if pos > 0 {
				prev := trimmed[pos-1]
				if prev != ':' && prev != ',' && prev != '[' && prev != '{' && prev != ' ' && prev != '\t' && prev != '\n' {
					continue
				}
			}
			return s
		}
		for prefixLen := 1; prefixLen < len(kw); prefixLen++ {
			if strings.HasSuffix(trimmed, kw[:prefixLen]) {
				// Verify it's a word boundary (preceded by : , [ { or whitespace).
				pos := len(trimmed) - prefixLen
				if pos > 0 {
					prev := trimmed[pos-1]
					if prev != ':' && prev != ',' && prev != '[' && prev != '{' && prev != ' ' && prev != '\t' && prev != '\n' {
						continue
					}
				}
				return trimmed[:len(trimmed)-prefixLen] + kw
			}
		}
	}

	// Complete truncated numbers.
	last := trimmed[len(trimmed)-1]
	switch last {
	case '.', '-', '+', 'e', 'E':
		return trimmed + "0"
	}

	return s
}

// trimTrailingIncomplete removes trailing incomplete entries before closing a container.
func trimTrailingIncomplete(s string) string {
	s = strings.TrimRight(s, " \t\n\r")
	if s == "" {
		return s
	}

	// Remove trailing comma.
	if s[len(s)-1] == ',' {
		return strings.TrimRight(s[:len(s)-1], " \t\n\r")
	}

	// Remove trailing colon + key (incomplete key-value pair).
	if s[len(s)-1] == ':' {
		s = s[:len(s)-1]
		s = strings.TrimRight(s, " \t\n\r")
		// Remove the key string.
		if len(s) > 0 && s[len(s)-1] == '"' {
			s = s[:len(s)-1]
			idx := strings.LastIndex(s, `"`)
			if idx >= 0 {
				s = s[:idx]
			}
			s = strings.TrimRight(s, " \t\n\r,")
		}
		return s
	}

	// Remove dangling key string inside an object (e.g. `{"a":1, "b"` → `{"a":1`).
	// A trailing `"..."` is a dangling key if preceded by `{` or `,`.
	if s[len(s)-1] == '"' {
		idx := strings.LastIndex(s[:len(s)-1], `"`)
		if idx >= 0 {
			before := strings.TrimRight(s[:idx], " \t\n\r")
			if len(before) > 0 {
				switch before[len(before)-1] {
				case '{':
					// Keep the opening brace, just remove the dangling key.
					return before
				case ',':
					// Strip the comma and the dangling key.
					return strings.TrimRight(before[:len(before)-1], " \t\n\r")
				}
			}
		}
	}

	return s
}
