package goai

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// --- Test types ---

type Person struct {
	Name string  `json:"name" jsonschema:"description=Full name"`
	Age  int     `json:"age"`
	Bio  *string `json:"bio"`
}

type Address struct {
	Street string `json:"street"`
	City   string `json:"city"`
}

type Employee struct {
	Person            // embedded
	Address Address   `json:"address"`
	Role    string    `json:"role" jsonschema:"enum=engineer|manager|director"`
}

type WithSkipped struct {
	Visible string `json:"visible"`
	Hidden  string `json:"-"`
}

type WithUnexported struct {
	Public  string `json:"public"`
	private string //nolint:unused
}

type WithSlice struct {
	Tags []string `json:"tags"`
}

type WithNestedSlice struct {
	Addresses []Address `json:"addresses"`
}

type WithMap struct {
	Metadata map[string]string `json:"metadata"`
}

type WithMapComplex struct {
	Data map[string]Address `json:"data"`
}

type AllBasicTypes struct {
	S   string  `json:"s"`
	B   bool    `json:"b"`
	I   int     `json:"i"`
	I8  int8    `json:"i8"`
	I16 int16   `json:"i16"`
	I32 int32   `json:"i32"`
	I64 int64   `json:"i64"`
	U   uint    `json:"u"`
	U8  uint8   `json:"u8"`
	U16 uint16  `json:"u16"`
	U32 uint32  `json:"u32"`
	U64 uint64  `json:"u64"`
	F32 float32 `json:"f32"`
	F64 float64 `json:"f64"`
}

type Base struct {
	ID string `json:"id"`
}

type Inner struct {
	Value int `json:"value"`
}

type Outer struct {
	Base        // embedded
	Inner Inner `json:"inner"`
}

type MultiTag struct {
	Status string `json:"status" jsonschema:"description=Current status,enum=active|inactive|pending"`
}

type PointerNested struct {
	Addr *Address `json:"addr"`
}

type EmbeddedPtr struct {
	*Base
	Name string `json:"name"`
}

type NoJSONTag struct {
	Name    string
	Visible string `json:"visible"`
}

type SliceOfPointers struct {
	Items []*Address `json:"items"`
}

// --- Helpers ---

func mustUnmarshal(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("failed to unmarshal schema: %v", err)
	}
	return m
}

func prop(t *testing.T, schema map[string]any, name string) map[string]any {
	t.Helper()
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties")
	}
	p, ok := props[name].(map[string]any)
	if !ok {
		t.Fatalf("property %q not found", name)
	}
	return p
}

func requiredList(t *testing.T, schema map[string]any) []string {
	t.Helper()
	raw, ok := schema["required"].([]any)
	if !ok {
		return nil
	}
	out := make([]string, len(raw))
	for i, v := range raw {
		out[i] = v.(string)
	}
	return out
}

func assertType(t *testing.T, p map[string]any, expected string) {
	t.Helper()
	got, ok := p["type"].(string)
	if !ok {
		t.Fatalf("expected type %q, got %v", expected, p["type"])
	}
	if got != expected {
		t.Errorf("expected type %q, got %q", expected, got)
	}
}

func assertNullableType(t *testing.T, p map[string]any, expectedBase string) {
	t.Helper()
	raw, ok := p["type"].([]any)
	if !ok {
		t.Fatalf("expected nullable type array, got %v (%T)", p["type"], p["type"])
	}
	if len(raw) != 2 {
		t.Fatalf("expected 2-element type array, got %d", len(raw))
	}
	if raw[0] != expectedBase {
		t.Errorf("expected base type %q, got %v", expectedBase, raw[0])
	}
	if raw[1] != "null" {
		t.Errorf("expected second element \"null\", got %v", raw[1])
	}
}

func assertHasProperty(t *testing.T, schema map[string]any, name string) {
	t.Helper()
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties")
	}
	if _, ok := props[name]; !ok {
		t.Errorf("expected property %q to exist", name)
	}
}

func assertNoProperty(t *testing.T, schema map[string]any, name string) {
	t.Helper()
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		return // no properties at all is fine
	}
	if _, ok := props[name]; ok {
		t.Errorf("expected property %q to NOT exist", name)
	}
}


func containsStr(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}

// --- Tests ---

func TestSchemaFrom_BasicTypes(t *testing.T) {
	tests := []struct {
		name     string
		field    string
		wantType string
	}{
		{"string", "s", "string"},
		{"bool", "b", "boolean"},
		{"int", "i", "integer"},
		{"int8", "i8", "integer"},
		{"int16", "i16", "integer"},
		{"int32", "i32", "integer"},
		{"int64", "i64", "integer"},
		{"uint", "u", "integer"},
		{"uint8", "u8", "integer"},
		{"uint16", "u16", "integer"},
		{"uint32", "u32", "integer"},
		{"uint64", "u64", "integer"},
		{"float32", "f32", "number"},
		{"float64", "f64", "number"},
	}

	schema := mustUnmarshal(t, SchemaFrom[AllBasicTypes]())

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := prop(t, schema, tt.field)
			assertType(t, p, tt.wantType)
		})
	}
}

func TestSchemaFrom_PointerNullable(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Person]())

	// Bio is *string → nullable string.
	bio := prop(t, schema, "bio")
	assertNullableType(t, bio, "string")

	// Name is string → non-nullable.
	name := prop(t, schema, "name")
	assertType(t, name, "string")
}

func TestSchemaFrom_PointerToStruct(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[PointerNested]())
	addr := prop(t, schema, "addr")

	// *Address → type should be ["object", "null"].
	assertNullableType(t, addr, "object")

	// Should still have properties from Address.
	innerProps, ok := addr["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected nested object to have properties")
	}
	if _, ok := innerProps["street"]; !ok {
		t.Error("expected nested object to have street property")
	}
}

func TestSchemaFrom_StructProperties(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Person]())

	assertType(t, schema, "object")
	assertHasProperty(t, schema, "name")
	assertHasProperty(t, schema, "age")
	assertHasProperty(t, schema, "bio")

	// additionalProperties: false
	ap, ok := schema["additionalProperties"].(bool)
	if !ok || ap != false {
		t.Errorf("expected additionalProperties: false, got %v", schema["additionalProperties"])
	}
}

func TestSchemaFrom_AllFieldsRequired(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Person]())

	req := requiredList(t, schema)
	for _, name := range []string{"name", "age", "bio"} {
		if !containsStr(req, name) {
			t.Errorf("expected %q in required list", name)
		}
	}
}

func TestSchemaFrom_NestedStruct(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Employee]())
	addr := prop(t, schema, "address")

	assertType(t, addr, "object")
	assertHasProperty(t, addr, "street")
	assertHasProperty(t, addr, "city")

	// Nested struct also gets additionalProperties: false.
	ap, ok := addr["additionalProperties"].(bool)
	if !ok || ap != false {
		t.Errorf("expected nested additionalProperties: false, got %v", addr["additionalProperties"])
	}
}

func TestSchemaFrom_EmbeddedStructFlatten(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Employee]())

	// Person's fields should be flattened into Employee.
	assertHasProperty(t, schema, "name")
	assertHasProperty(t, schema, "age")
	assertHasProperty(t, schema, "bio")
	// address and role are Employee's own fields.
	assertHasProperty(t, schema, "address")
	assertHasProperty(t, schema, "role")

	req := requiredList(t, schema)
	for _, name := range []string{"name", "age", "bio", "address", "role"} {
		if !containsStr(req, name) {
			t.Errorf("expected %q in required list after embedding", name)
		}
	}
}

func TestSchemaFrom_EmbeddedPointerStruct(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[EmbeddedPtr]())

	// *Base embedded → fields flattened.
	assertHasProperty(t, schema, "id")
	assertHasProperty(t, schema, "name")
}

func TestSchemaFrom_Slice(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[WithSlice]())
	tags := prop(t, schema, "tags")

	assertType(t, tags, "array")
	items, ok := tags["items"].(map[string]any)
	if !ok {
		t.Fatal("expected items in array schema")
	}
	assertType(t, items, "string")
}

func TestSchemaFrom_SliceOfStructs(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[WithNestedSlice]())
	addrs := prop(t, schema, "addresses")

	assertType(t, addrs, "array")
	items, ok := addrs["items"].(map[string]any)
	if !ok {
		t.Fatal("expected items in array schema")
	}
	assertType(t, items, "object")
	assertHasProperty(t, items, "street")
}

func TestSchemaFrom_SliceOfPointers(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[SliceOfPointers]())
	items := prop(t, schema, "items")

	assertType(t, items, "array")
	inner, ok := items["items"].(map[string]any)
	if !ok {
		t.Fatal("expected items schema for []*Address")
	}
	// Each item is *Address → nullable object.
	assertNullableType(t, inner, "object")
}

func TestSchemaFrom_MapStringKey(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[WithMap]())
	meta := prop(t, schema, "metadata")

	assertType(t, meta, "object")
	ap, ok := meta["additionalProperties"].(map[string]any)
	if !ok {
		t.Fatal("expected additionalProperties for map")
	}
	assertType(t, ap, "string")
}

func TestSchemaFrom_MapComplexValue(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[WithMapComplex]())
	data := prop(t, schema, "data")

	assertType(t, data, "object")
	ap, ok := data["additionalProperties"].(map[string]any)
	if !ok {
		t.Fatal("expected additionalProperties for map[string]Address")
	}
	assertType(t, ap, "object")
	assertHasProperty(t, ap, "street")
}

func TestSchemaFrom_JSONTagSkip(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[WithSkipped]())

	assertHasProperty(t, schema, "visible")
	assertNoProperty(t, schema, "Hidden")
	assertNoProperty(t, schema, "hidden")
}

func TestSchemaFrom_JSONTagCustomName(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Person]())

	// Uses json:"name" not "Name".
	assertHasProperty(t, schema, "name")
	assertNoProperty(t, schema, "Name")
}

func TestSchemaFrom_UnexportedFieldsSkipped(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[WithUnexported]())

	assertHasProperty(t, schema, "public")
	assertNoProperty(t, schema, "private")

	req := requiredList(t, schema)
	if containsStr(req, "private") {
		t.Error("unexported field should not be in required list")
	}
	if len(req) != 1 {
		t.Errorf("expected 1 required field, got %d", len(req))
	}
}

func TestSchemaFrom_DescriptionTag(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Person]())
	name := prop(t, schema, "name")

	desc, ok := name["description"].(string)
	if !ok {
		t.Fatal("expected description on name field")
	}
	if desc != "Full name" {
		t.Errorf("expected description \"Full name\", got %q", desc)
	}
}

func TestSchemaFrom_EnumTag(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Employee]())
	role := prop(t, schema, "role")

	rawEnum, ok := role["enum"].([]any)
	if !ok {
		t.Fatal("expected enum on role field")
	}
	expected := []string{"engineer", "manager", "director"}
	if len(rawEnum) != len(expected) {
		t.Fatalf("expected %d enum values, got %d", len(expected), len(rawEnum))
	}
	for i, v := range expected {
		if rawEnum[i] != v {
			t.Errorf("enum[%d]: expected %q, got %v", i, v, rawEnum[i])
		}
	}
}

func TestSchemaFrom_MultipleSchemaTagValues(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[MultiTag]())
	status := prop(t, schema, "status")

	// Should have both description and enum.
	desc, ok := status["description"].(string)
	if !ok || desc != "Current status" {
		t.Errorf("expected description \"Current status\", got %v", status["description"])
	}

	rawEnum, ok := status["enum"].([]any)
	if !ok {
		t.Fatal("expected enum on status field")
	}
	expected := []string{"active", "inactive", "pending"}
	if len(rawEnum) != len(expected) {
		t.Fatalf("expected %d enum values, got %d", len(expected), len(rawEnum))
	}
	for i, v := range expected {
		if rawEnum[i] != v {
			t.Errorf("enum[%d]: expected %q, got %v", i, v, rawEnum[i])
		}
	}
}

func TestSchemaFrom_NoJSONTagUsesFieldName(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[NoJSONTag]())

	// Field without json tag uses Go field name.
	assertHasProperty(t, schema, "Name")
	// Field with json tag uses tag name.
	assertHasProperty(t, schema, "visible")
}

func TestSchemaFrom_DeepNesting(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[Outer]())

	// Base embedded → id flattened.
	assertHasProperty(t, schema, "id")
	// Inner is a nested struct field.
	assertHasProperty(t, schema, "inner")

	inner := prop(t, schema, "inner")
	assertType(t, inner, "object")
	assertHasProperty(t, inner, "value")

	valueProp := prop(t, inner, "value")
	assertType(t, valueProp, "integer")
}

func TestSchemaFrom_OutputIsValidJSON(t *testing.T) {
	raw := SchemaFrom[Employee]()

	// Must be valid JSON.
	if !json.Valid(raw) {
		t.Fatal("SchemaFrom output is not valid JSON")
	}

	// Round-trip: unmarshal and re-marshal should not error.
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	if _, err := json.Marshal(m); err != nil {
		t.Fatalf("re-marshal failed: %v", err)
	}
}

func TestSchemaFrom_TopLevelPointerType(t *testing.T) {
	// SchemaFrom[*Person]() should unwrap the pointer and return the struct schema.
	schema := mustUnmarshal(t, SchemaFrom[*Person]())

	assertType(t, schema, "object")
	assertHasProperty(t, schema, "name")
	assertHasProperty(t, schema, "age")
	assertHasProperty(t, schema, "bio")
}

func TestSchemaFrom_EmptyStruct(t *testing.T) {
	type Empty struct{}
	schema := mustUnmarshal(t, SchemaFrom[Empty]())

	assertType(t, schema, "object")
	// No required field when there are no fields.
	if _, ok := schema["required"]; ok {
		t.Error("expected no required key for empty struct")
	}
}

func TestTypeToSchema_NonStringKeyMap(t *testing.T) {
	// map[int]string → object with no additionalProperties schema.
	result := typeToSchema(reflect.TypeOf(map[int]string{}), make(map[reflect.Type]bool))
	if result["type"] != "object" {
		t.Errorf("expected type object, got %v", result["type"])
	}
	if _, ok := result["additionalProperties"]; ok {
		t.Error("expected no additionalProperties for non-string-key map")
	}
}

func TestTypeToSchema_UnsupportedType(t *testing.T) {
	// chan int → empty schema.
	result := typeToSchema(reflect.TypeOf(make(chan int)), make(map[reflect.Type]bool))
	if len(result) != 0 {
		t.Errorf("expected empty schema for unsupported type, got %v", result)
	}
}

// InvalidTag has a jsonschema tag part without '=' (should be skipped).
type InvalidTag struct {
	Name string `json:"name" jsonschema:"required,description=A name"`
}

func TestSchemaFrom_InvalidTagPartSkipped(t *testing.T) {
	schema := mustUnmarshal(t, SchemaFrom[InvalidTag]())
	name := prop(t, schema, "name")
	// "required" without '=' should be ignored.
	desc, ok := name["description"].(string)
	if !ok || desc != "A name" {
		t.Errorf("expected description 'A name', got %v", name["description"])
	}
}

func TestSchemaFrom_MarshalFailurePanics(t *testing.T) {
	// Inject a failing marshal function to cover the panic path.
	orig := schemaMarshalFunc
	schemaMarshalFunc = func(v any) ([]byte, error) {
		return nil, fmt.Errorf("injected marshal failure")
	}
	defer func() { schemaMarshalFunc = orig }()

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on marshal failure")
		}
		msg, ok := r.(string)
		if !ok || !strings.Contains(msg, "SchemaFrom marshal failed") {
			t.Errorf("unexpected panic: %v", r)
		}
	}()
	SchemaFrom[Person]()
}
