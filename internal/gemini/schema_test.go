package gemini

import (
	"reflect"
	"testing"
)

func TestSanitizeSchema_Nil(t *testing.T) {
	// sanitizeImpl(nil) returns nil; SanitizeSchema wraps it and
	// falls back to returning the original (nil) schema.
	// A nil map[string]any is returned, which is the original input.
	result := SanitizeSchema(nil)
	if len(result) != 0 {
		t.Errorf("SanitizeSchema(nil) len = %d, want 0", len(result))
	}
}

func TestSanitizeSchema_EmptyMap(t *testing.T) {
	result := SanitizeSchema(map[string]any{})
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result) != 0 {
		t.Errorf("expected empty map, got %v", result)
	}
}

func TestSanitizeSchema_RemovesAdditionalProperties(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
		"additionalProperties": false,
	}
	result := SanitizeSchema(schema)
	if _, ok := result["additionalProperties"]; ok {
		t.Error("additionalProperties should be removed")
	}
	if result["type"] != "object" {
		t.Errorf("type = %v", result["type"])
	}
}

func TestSanitizeSchema_EnumIntegerToString(t *testing.T) {
	schema := map[string]any{
		"type": "integer",
		"enum": []any{1, 2, 3},
	}
	result := SanitizeSchema(schema)
	if result["type"] != "string" {
		t.Errorf("type = %v, want string", result["type"])
	}
	enumArr := result["enum"].([]any)
	if enumArr[0] != "1" || enumArr[1] != "2" || enumArr[2] != "3" {
		t.Errorf("enum = %v, want [1 2 3] as strings", enumArr)
	}
}

func TestSanitizeSchema_EnumNumberToString(t *testing.T) {
	schema := map[string]any{
		"type": "number",
		"enum": []any{1.5, 2.5},
	}
	result := SanitizeSchema(schema)
	if result["type"] != "string" {
		t.Errorf("type = %v, want string", result["type"])
	}
}

func TestSanitizeSchema_EnumStringNoChange(t *testing.T) {
	schema := map[string]any{
		"type": "string",
		"enum": []any{"a", "b", "c"},
	}
	result := SanitizeSchema(schema)
	if result["type"] != "string" {
		t.Errorf("type = %v, want string", result["type"])
	}
	enumArr := result["enum"].([]any)
	if enumArr[0] != "a" {
		t.Errorf("enum[0] = %v", enumArr[0])
	}
}

func TestSanitizeSchema_FilterRequiredFields(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
		"required": []any{"name", "nonexistent"},
	}
	result := SanitizeSchema(schema)
	required := result["required"].([]any)
	if len(required) != 1 {
		t.Fatalf("required = %v, want [name]", required)
	}
	if required[0] != "name" {
		t.Errorf("required[0] = %v", required[0])
	}
}

func TestSanitizeSchema_ArrayItemsNoType(t *testing.T) {
	schema := map[string]any{
		"type":  "array",
		"items": map[string]any{},
	}
	result := SanitizeSchema(schema)
	items := result["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("items.type = %v, want string", items["type"])
	}
}

func TestSanitizeSchema_ArrayItemsNil(t *testing.T) {
	schema := map[string]any{
		"type": "array",
	}
	result := SanitizeSchema(schema)
	items := result["items"].(map[string]any)
	if items == nil {
		t.Fatal("items should be set")
	}
}

func TestSanitizeSchema_ArrayItemsWithType(t *testing.T) {
	schema := map[string]any{
		"type":  "array",
		"items": map[string]any{"type": "integer"},
	}
	result := SanitizeSchema(schema)
	items := result["items"].(map[string]any)
	if items["type"] != "integer" {
		t.Errorf("items.type = %v, want integer (unchanged)", items["type"])
	}
}

func TestSanitizeSchema_RemovePropertiesFromNonObject(t *testing.T) {
	schema := map[string]any{
		"type": "string",
		"properties": map[string]any{
			"x": map[string]any{"type": "string"},
		},
		"required": []any{"x"},
	}
	result := SanitizeSchema(schema)
	if _, ok := result["properties"]; ok {
		t.Error("properties should be removed from non-object type")
	}
	if _, ok := result["required"]; ok {
		t.Error("required should be removed from non-object type")
	}
}

func TestSanitizeSchema_NestedObjects(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"address": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"street": map[string]any{"type": "string"},
				},
				"additionalProperties": true,
			},
		},
	}
	result := SanitizeSchema(schema)
	props := result["properties"].(map[string]any)
	addr := props["address"].(map[string]any)
	if _, ok := addr["additionalProperties"]; ok {
		t.Error("nested additionalProperties should be removed")
	}
}

func TestSanitizeSchema_ArrayWithNestedSchema(t *testing.T) {
	schema := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id": map[string]any{"type": "integer"},
			},
			"additionalProperties": false,
		},
	}
	result := SanitizeSchema(schema)
	items := result["items"].(map[string]any)
	if _, ok := items["additionalProperties"]; ok {
		t.Error("nested additionalProperties should be removed")
	}
}

func TestSanitizeImpl_ScalarPassthrough(t *testing.T) {
	// Non-map, non-array, non-nil values pass through unchanged.
	result := sanitizeImpl("hello")
	if result != "hello" {
		t.Errorf("expected hello, got %v", result)
	}

	result = sanitizeImpl(42)
	if result != 42 {
		t.Errorf("expected 42, got %v", result)
	}

	result = sanitizeImpl(true)
	if result != true {
		t.Errorf("expected true, got %v", result)
	}
}

func TestSanitizeImpl_Array(t *testing.T) {
	input := []any{
		map[string]any{"type": "string", "additionalProperties": true},
		"plain",
	}
	result := sanitizeImpl(input).([]any)
	if len(result) != 2 {
		t.Fatalf("len = %d, want 2", len(result))
	}
	// First element (map) should have additionalProperties removed.
	m := result[0].(map[string]any)
	if _, ok := m["additionalProperties"]; ok {
		t.Error("additionalProperties should be removed from array element")
	}
	// Second element (string) passes through.
	if result[1] != "plain" {
		t.Errorf("result[1] = %v", result[1])
	}
}

func TestSanitizeImpl_Nil(t *testing.T) {
	result := sanitizeImpl(nil)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestSanitizeSchema_EnumNotArray(t *testing.T) {
	// enum is not an array -- should pass through as a regular value.
	schema := map[string]any{
		"type": "string",
		"enum": "not-an-array",
	}
	result := SanitizeSchema(schema)
	if result["enum"] != "not-an-array" {
		t.Errorf("enum = %v", result["enum"])
	}
}

func TestSanitizeSchema_RequiredNonStringEntries(t *testing.T) {
	// required array with non-string entries -- should be filtered out.
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
		"required": []any{"name", 42},
	}
	result := SanitizeSchema(schema)
	required := result["required"].([]any)
	if len(required) != 1 {
		t.Fatalf("required = %v, want [name]", required)
	}
}

func TestSanitizeSchema_ObjectWithoutProperties(t *testing.T) {
	// object type without properties -- required filtering skipped.
	schema := map[string]any{
		"type":     "object",
		"required": []any{"name"},
	}
	result := SanitizeSchema(schema)
	// Without properties map, the required filtering branch (line 71 props check) is skipped.
	// required stays as-is.
	if result["type"] != "object" {
		t.Errorf("type = %v", result["type"])
	}
}

func TestSanitizeSchema_EnumWithNoType(t *testing.T) {
	// enum present but no type field.
	schema := map[string]any{
		"enum": []any{1, 2},
	}
	result := SanitizeSchema(schema)
	enumArr := result["enum"].([]any)
	if enumArr[0] != "1" {
		t.Errorf("enum[0] = %v, want '1'", enumArr[0])
	}
}

func TestSanitizeSchema_PointerToStructNullableType(t *testing.T) {
	// SchemaFrom produces []string{"object","null"} for pointer-to-struct fields.
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"address": map[string]any{
				"type": []string{"object", "null"},
				"properties": map[string]any{
					"street": map[string]any{"type": "string"},
				},
				"required":             []string{"street"},
				"additionalProperties": false,
			},
		},
		"required":             []string{"address"},
		"additionalProperties": false,
	}
	result := SanitizeSchema(schema)

	props := result["properties"].(map[string]any)
	addr := props["address"].(map[string]any)

	// Type should be normalized to plain "object".
	if addr["type"] != "object" {
		t.Errorf("address.type = %v, want \"object\"", addr["type"])
	}
	// Should be marked nullable.
	if addr["nullable"] != true {
		t.Errorf("address.nullable = %v, want true", addr["nullable"])
	}
	// Properties should be preserved (not stripped).
	if _, ok := addr["properties"]; !ok {
		t.Error("address.properties should be preserved for object type")
	}
	// additionalProperties should be removed.
	if _, ok := addr["additionalProperties"]; ok {
		t.Error("additionalProperties should be removed")
	}
	// Nested required ([]string) should be filtered correctly.
	if req, ok := addr["required"]; ok {
		reqArr := req.([]any)
		if len(reqArr) != 1 || reqArr[0] != "street" {
			t.Errorf("address.required = %v, want [street]", reqArr)
		}
	}
}

func TestSanitizeSchema_TimeDateTimeFormat(t *testing.T) {
	// time.Time produces type: "string", format: "date-time".
	// Gemini doesn't support format, so it should be stripped.
	schema := map[string]any{
		"type":   "string",
		"format": "date-time",
	}
	result := SanitizeSchema(schema)
	if _, ok := result["format"]; ok {
		t.Error("format should be removed for Gemini compatibility")
	}
	if result["type"] != "string" {
		t.Errorf("type = %v, want string", result["type"])
	}
}

func TestSanitizeSchema_RequiredAsStringSlice(t *testing.T) {
	// SchemaFrom produces required as []string, not []any.
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
		"required": []string{"name", "age", "nonexistent"},
	}
	result := SanitizeSchema(schema)
	required := result["required"].([]any)
	if len(required) != 2 {
		t.Fatalf("required = %v, want [name age]", required)
	}
	// Check both fields are present.
	got := map[string]bool{}
	for _, r := range required {
		got[r.(string)] = true
	}
	if !got["name"] || !got["age"] {
		t.Errorf("required = %v, want name and age", required)
	}
}

func TestSanitizeSchema_NullableTypeAsAnySlice(t *testing.T) {
	// If type comes as []any (e.g., from JSON unmarshal), it should also be normalized.
	schema := map[string]any{
		"type": []any{"string", "null"},
	}
	result := SanitizeSchema(schema)
	if result["type"] != "string" {
		t.Errorf("type = %v, want \"string\"", result["type"])
	}
	if result["nullable"] != true {
		t.Errorf("nullable = %v, want true", result["nullable"])
	}
}

func TestSanitizeSchema_FullSchemaFromOutput(t *testing.T) {
	// Simulate what SchemaFrom produces for a struct with pointer and time fields.
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"created_at": map[string]any{
				"type":   "string",
				"format": "date-time",
			},
			"metadata": map[string]any{
				"type": []string{"object", "null"},
				"properties": map[string]any{
					"key": map[string]any{"type": "string"},
				},
				"required":             []string{"key"},
				"additionalProperties": false,
			},
		},
		"required":             []string{"name", "created_at", "metadata"},
		"additionalProperties": false,
	}
	result := SanitizeSchema(schema)

	// Top-level: additionalProperties removed, required filtered.
	if _, ok := result["additionalProperties"]; ok {
		t.Error("top-level additionalProperties should be removed")
	}
	req := result["required"].([]any)
	if !reflect.DeepEqual(req, []any{"name", "created_at", "metadata"}) {
		t.Errorf("required = %v", req)
	}

	props := result["properties"].(map[string]any)

	// created_at: format removed.
	createdAt := props["created_at"].(map[string]any)
	if _, ok := createdAt["format"]; ok {
		t.Error("created_at.format should be removed")
	}

	// metadata: type normalized, nullable set, properties preserved.
	meta := props["metadata"].(map[string]any)
	if meta["type"] != "object" {
		t.Errorf("metadata.type = %v, want \"object\"", meta["type"])
	}
	if meta["nullable"] != true {
		t.Error("metadata.nullable should be true")
	}
	if _, ok := meta["properties"]; !ok {
		t.Error("metadata.properties should be preserved")
	}
}

func TestSanitizeSchema_NoInputMutation(t *testing.T) {
	// Verify that SanitizeSchema does not mutate the input map.
	input := map[string]any{
		"type": []string{"object", "null"},
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}
	// Save original type value.
	origType := input["type"].([]string)
	origLen := len(origType)

	result := SanitizeSchema(input)

	// Input must not be mutated.
	inputType := input["type"]
	switch tt := inputType.(type) {
	case []string:
		if len(tt) != origLen {
			t.Errorf("input type was mutated: got %v", tt)
		}
		for _, s := range tt {
			if s != "object" && s != "null" {
				t.Errorf("input type element mutated to %q", s)
			}
		}
	default:
		t.Errorf("input type was mutated from []string to %T: %v", inputType, inputType)
	}

	// Result should have the nullable conversion applied.
	if result["type"] != "object" {
		t.Errorf("result type = %v, want 'object'", result["type"])
	}
	if result["nullable"] != true {
		t.Error("result should have nullable: true")
	}
}

func TestSanitizeSchema_NoInputMutation_AnySlice(t *testing.T) {
	// Test the []any path (from JSON unmarshal) also does not mutate input.
	input := map[string]any{
		"type": []any{"string", "null"},
	}

	result := SanitizeSchema(input)

	// Input must not be mutated.
	inputType, ok := input["type"].([]any)
	if !ok {
		t.Fatalf("input type was mutated from []any to %T", input["type"])
	}
	if len(inputType) != 2 {
		t.Errorf("input type length mutated: got %d", len(inputType))
	}

	// Result should have the nullable conversion.
	if result["type"] != "string" {
		t.Errorf("result type = %v, want 'string'", result["type"])
	}
	if result["nullable"] != true {
		t.Error("result should have nullable: true")
	}
}
