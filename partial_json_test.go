package goai

import (
	"testing"
)

func TestRepairJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		// Valid JSON passes through unchanged.
		{
			name:     "complete object",
			input:    `{"name": "Alice", "age": 30}`,
			expected: `{"name": "Alice", "age": 30}`,
		},
		{
			name:     "complete array",
			input:    `[1, 2, 3]`,
			expected: `[1, 2, 3]`,
		},
		{
			name:     "complete nested",
			input:    `{"a": {"b": [1, 2]}}`,
			expected: `{"a": {"b": [1, 2]}}`,
		},

		// Empty string → "{}".
		{
			name:     "empty string",
			input:    "",
			expected: "{}",
		},
		{
			name:     "whitespace only",
			input:    "   \t\n",
			expected: "{}",
		},

		// Truncated string values.
		{
			name:     "truncated string value",
			input:    `{"name": "Pa`,
			expected: `{"name": "Pa"}`,
		},
		{
			name:     "truncated string mid-word",
			input:    `{"greeting": "hello wor`,
			expected: `{"greeting": "hello wor"}`,
		},

		// Truncated arrays.
		{
			name:     "truncated array trailing comma",
			input:    `[1, 2,`,
			expected: `[1, 2]`,
		},
		{
			name:     "truncated array no comma",
			input:    `[1, 2`,
			expected: `[1, 2]`,
		},
		{
			name:     "truncated array after opening",
			input:    `[`,
			expected: `[]`,
		},

		// Truncated objects.
		{
			name:     "truncated object trailing colon",
			input:    `{"a": 1, "b":`,
			expected: `{"a": 1}`,
		},
		{
			name:     "truncated object trailing comma",
			input:    `{"a": 1,`,
			expected: `{"a": 1}`,
		},
		{
			name:     "truncated object after opening",
			input:    `{`,
			expected: `{}`,
		},
		{
			name:     "truncated object key only",
			input:    `{"a": 1, "b"`,
			expected: `{"a": 1, "b"}`,
		},

		// Nested truncation.
		{
			name:     "nested object truncation",
			input:    `{"a": {"b": 1`,
			expected: `{"a": {"b": 1}}`,
		},
		{
			name:     "deeply nested truncation",
			input:    `{"a": {"b": {"c": 2`,
			expected: `{"a": {"b": {"c": 2}}}`,
		},

		// Array in object.
		{
			name:     "array in object truncated",
			input:    `{"a": [1, 2`,
			expected: `{"a": [1, 2]}`,
		},
		{
			name:     "array in object trailing comma",
			input:    `{"a": [1, 2,`,
			expected: `{"a": [1, 2]}`,
		},

		// Truncated keywords.
		{
			name:     "truncated true",
			input:    `{"valid": tru`,
			expected: `{"valid": true}`,
		},
		{
			name:     "truncated false",
			input:    `{"valid": fal`,
			expected: `{"valid": false}`,
		},
		{
			name:     "truncated null",
			input:    `{"val": nul`,
			expected: `{"val": null}`,
		},
		{
			name:     "truncated true single char",
			input:    `{"ok": t`,
			expected: `{"ok": true}`,
		},

		// Truncated numbers.
		{
			name:     "truncated decimal",
			input:    `{"val": 1.`,
			expected: `{"val": 1.0}`,
		},
		{
			name:     "truncated negative sign",
			input:    `{"val": -`,
			expected: `{"val": -0}`,
		},
		{
			name:     "truncated exponent",
			input:    `{"val": 1e`,
			expected: `{"val": 1e0}`,
		},
		{
			name:     "truncated exponent uppercase",
			input:    `{"val": 1E`,
			expected: `{"val": 1E0}`,
		},

		// Escaped quotes in strings.
		{
			name:     "escaped quote truncated",
			input:    `{"a": "he said \"hi`,
			expected: `{"a": "he said \"hi"}`,
		},
		{
			name:     "escaped quote complete",
			input:    `{"a": "he said \"hi\""}`,
			expected: `{"a": "he said \"hi\""}`,
		},
		{
			name:     "escaped backslash in string",
			input:    `{"a": "path\\`,
			expected: `{"a": "path\\"}`,
		},
		{
			name:     "trailing lone backslash in string",
			input:    "{\"a\": \"ab\\",
			expected: `{"a": "ab"}`,
		},

		// Object in array.
		{
			name:     "object in array truncated",
			input:    `[{"a": 1}, {"b":`,
			expected: `[{"a": 1}, {}]`,
		},

		// Mixed nesting.
		{
			name:     "array of arrays truncated",
			input:    `[[1, 2], [3`,
			expected: `[[1, 2], [3]]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := repairJSON(tt.input)
			if got != tt.expected {
				t.Errorf("repairJSON(%q)\n  got:  %q\n  want: %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestCompleteTrailing(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "complete true keyword",
			input:    `{"a": tr`,
			expected: `{"a": true`,
		},
		{
			name:     "complete false keyword",
			input:    `{"a": fa`,
			expected: `{"a": false`,
		},
		{
			name:     "complete null keyword",
			input:    `{"a": nu`,
			expected: `{"a": null`,
		},
		{
			name:     "trailing dot",
			input:    `{"a": 3.`,
			expected: `{"a": 3.0`,
		},
		{
			name:     "trailing minus",
			input:    `{"a": -`,
			expected: `{"a": -0`,
		},
		{
			name:     "trailing plus",
			input:    `{"a": +`,
			expected: `{"a": +`,
		},
		{
			name:     "trailing e",
			input:    `{"a": 5e`,
			expected: `{"a": 5e0`,
		},
		{
			name:     "no trailing issue",
			input:    `{"a": 5`,
			expected: `{"a": 5`,
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "keyword not at word boundary -- e treated as exponent",
			input:    `{"nature`,
			expected: `{"nature0`,
		},
		{
			name:     "keyword at start of string",
			input:    `tru`,
			expected: `true`,
		},
		{
			name:     "keyword prefix not at word boundary -- continue branch",
			input:    `{"count`,
			expected: `{"count`,
		},
		{
			name:     "trailing uppercase E",
			input:    `{"a": 9E`,
			expected: `{"a": 9E0`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := completeTrailing(tt.input)
			if got != tt.expected {
				t.Errorf("completeTrailing(%q)\n  got:  %q\n  want: %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestTrimTrailingIncomplete(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "trailing comma",
			input:    `{"a": 1,`,
			expected: `{"a": 1`,
		},
		{
			name:     "trailing colon with key",
			input:    `{"a": 1, "b":`,
			expected: `{"a": 1`,
		},
		{
			name:     "trailing colon with key and spaces",
			input:    `{"a": 1, "b" :`,
			expected: `{"a": 1`,
		},
		{
			name:     "no trailing issue",
			input:    `{"a": 1`,
			expected: `{"a": 1`,
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := trimTrailingIncomplete(tt.input)
			if got != tt.expected {
				t.Errorf("trimTrailingIncomplete(%q)\n  got:  %q\n  want: %q", tt.input, got, tt.expected)
			}
		})
	}
}

// --- parsePartialJSON tests with typed structs ---

type simpleObj struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

type nestedObj struct {
	A innerObj `json:"a"`
}

type innerObj struct {
	B int `json:"b"`
}

type objWithArray struct {
	Items []int `json:"items"`
}

type objWithBool struct {
	Valid bool `json:"valid"`
}

type objWithFloat struct {
	Val float64 `json:"val"`
}

type objWithNull struct {
	Val *string `json:"val"`
}

func TestParsePartialJSON_CompleteJSON(t *testing.T) {
	input := `{"name": "Alice", "age": 30}`
	got, err := parsePartialJSON[simpleObj](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "Alice" || got.Age != 30 {
		t.Errorf("got %+v, want {Name:Alice Age:30}", got)
	}
}

func TestParsePartialJSON_TruncatedString(t *testing.T) {
	input := `{"name": "Ali`
	got, err := parsePartialJSON[simpleObj](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "Ali" {
		t.Errorf("got Name=%q, want %q", got.Name, "Ali")
	}
}

func TestParsePartialJSON_NestedObject(t *testing.T) {
	input := `{"a": {"b": 42`
	got, err := parsePartialJSON[nestedObj](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.A.B != 42 {
		t.Errorf("got A.B=%d, want 42", got.A.B)
	}
}

func TestParsePartialJSON_ArrayInObject(t *testing.T) {
	input := `{"items": [1, 2, 3`
	got, err := parsePartialJSON[objWithArray](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Items) != 3 || got.Items[0] != 1 || got.Items[1] != 2 || got.Items[2] != 3 {
		t.Errorf("got Items=%v, want [1 2 3]", got.Items)
	}
}

func TestParsePartialJSON_TruncatedBool(t *testing.T) {
	input := `{"valid": tru`
	got, err := parsePartialJSON[objWithBool](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !got.Valid {
		t.Errorf("got Valid=%v, want true", got.Valid)
	}
}

func TestParsePartialJSON_TruncatedFloat(t *testing.T) {
	input := `{"val": 3.`
	got, err := parsePartialJSON[objWithFloat](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Val != 3.0 {
		t.Errorf("got Val=%f, want 3.0", got.Val)
	}
}

func TestParsePartialJSON_NullValue(t *testing.T) {
	input := `{"val": nul`
	got, err := parsePartialJSON[objWithNull](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Val != nil {
		t.Errorf("got Val=%v, want nil", got.Val)
	}
}

func TestParsePartialJSON_EmptyString(t *testing.T) {
	// Empty string repairs to "{}", which unmarshals to zero-value struct.
	got, err := parsePartialJSON[simpleObj]("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "" || got.Age != 0 {
		t.Errorf("got %+v, want zero-value simpleObj", got)
	}
}

func TestParsePartialJSON_Array(t *testing.T) {
	input := `[1, 2, 3`
	got, err := parsePartialJSON[[]int](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(*got) != 3 {
		t.Errorf("got len=%d, want 3", len(*got))
	}
}

func TestParsePartialJSON_Map(t *testing.T) {
	input := `{"key": "val`
	got, err := parsePartialJSON[map[string]string](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if (*got)["key"] != "val" {
		t.Errorf("got key=%q, want %q", (*got)["key"], "val")
	}
}

func TestParsePartialJSON_EscapedQuotes(t *testing.T) {
	input := `{"name": "he said \"hi`
	got, err := parsePartialJSON[simpleObj](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := `he said "hi`
	if got.Name != expected {
		t.Errorf("got Name=%q, want %q", got.Name, expected)
	}
}

func TestParsePartialJSON_AlreadyValid(t *testing.T) {
	// Ensure valid JSON does NOT go through repair (no mutation).
	input := `{"name":"Bob","age":25}`
	got, err := parsePartialJSON[simpleObj](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "Bob" || got.Age != 25 {
		t.Errorf("got %+v, want {Name:Bob Age:25}", got)
	}
}

func TestParsePartialJSON_TruncatedObjectKey(t *testing.T) {
	// Truncated after key + colon → should drop the incomplete pair.
	input := `{"name": "Alice", "age":`
	got, err := parsePartialJSON[simpleObj](input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "Alice" {
		t.Errorf("got Name=%q, want %q", got.Name, "Alice")
	}
	// age should be zero since the value was truncated.
	if got.Age != 0 {
		t.Errorf("got Age=%d, want 0", got.Age)
	}
}
