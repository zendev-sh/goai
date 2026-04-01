package goai

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"
)

// schemaMarshalFunc is swappable for testing the panic path.
var schemaMarshalFunc = json.Marshal

// schemaMarshalFuncIsDefault returns true when schemaMarshalFunc has not been
// replaced. Caching is only safe when the default marshal function is in use;
// swapping it (in tests) must bypass the cache to ensure the injected function
// is actually called.
func schemaMarshalFuncIsDefault() bool {
	// Compare function pointers via reflect to detect test overrides.
	return reflect.ValueOf(schemaMarshalFunc).Pointer() == reflect.ValueOf(json.Marshal).Pointer()
}

// schemaCache caches computed JSON schemas keyed by reflect.Type to avoid
// recomputing on every call to SchemaFrom[T]().
var schemaCache sync.Map

// SchemaFrom generates a JSON Schema from the Go type T. It panics if the
// generated schema cannot be marshaled to JSON, which indicates a bug in
// typeToSchema rather than a runtime error. In normal usage this cannot occur.
//
// The schema is compatible with OpenAI strict mode:
//   - All properties are required (pointer types become nullable)
//   - additionalProperties: false on all objects
//
// Supports struct tags:
//   - json:"name" for field naming, json:"-" to skip
//   - jsonschema:"description=...,enum=a|b|c" for descriptions and enums
//
// Results are cached: repeated calls with the same T return a cached value.
// The cache is bypassed when schemaMarshalFunc has been replaced (test-only).
func SchemaFrom[T any]() json.RawMessage {
	t := reflect.TypeFor[T]()
	useCache := schemaMarshalFuncIsDefault()
	if useCache {
		if cached, ok := schemaCache.Load(t); ok {
			return cached.(json.RawMessage)
		}
	}
	// Unwrap pointer types.
	unwrapped := t
	for unwrapped.Kind() == reflect.Ptr {
		unwrapped = unwrapped.Elem()
	}
	schema := typeToSchema(unwrapped, make(map[reflect.Type]bool))
	// typeToSchema only produces JSON-safe types (map[string]any with string/bool/int/slice values),
	// so json.Marshal cannot fail here. Panic on impossible error to surface bugs in typeToSchema.
	data, err := schemaMarshalFunc(schema)
	if err != nil {
		panic(fmt.Sprintf("goai: SchemaFrom marshal failed (bug in typeToSchema): %v", err))
	}
	if useCache {
		schemaCache.Store(t, json.RawMessage(data))
	}
	return data
}

var (
	timeType       = reflect.TypeOf(time.Time{})
	rawMessageType = reflect.TypeOf(json.RawMessage{})
)

func typeToSchema(t reflect.Type, seen map[reflect.Type]bool) map[string]any {
	// Unwrap pointer: nullable type.
	if t.Kind() == reflect.Ptr {
		inner := typeToSchema(t.Elem(), seen)
		// Make nullable: type becomes array ["<type>", "null"].
		if baseType, ok := inner["type"].(string); ok {
			inner["type"] = []string{baseType, "null"}
		}
		return inner
	}

	// Special case: time.Time → string with date-time format.
	if t == timeType {
		return map[string]any{"type": "string", "format": "date-time"}
	}

	// Special case: json.RawMessage → any type (empty schema).
	if t == rawMessageType {
		return map[string]any{}
	}

	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]any{"type": "integer"}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice:
		elem := t.Elem()
		// Break cycles:
		// - seen[elem]: elem is a struct currently being processed (set by the struct case).
		// - elem == t: self-referential named slice type (e.g. type Foo []Foo), where the
		//   element type IS the slice type itself. Without this guard, recursion is unbounded.
		//   Do NOT set seen[elem] here; that would block the struct case from fully processing
		//   non-recursive slice-of-struct types like []SomeStruct.
		// Limitation: mutually recursive named slice types (e.g. type A []B; type B []A) are
		// NOT detected by this guard and will cause a stack overflow. Use struct wrappers
		// instead of raw named-slice mutual recursion.
		if seen[elem] || elem == t {
			return map[string]any{"type": "array"}
		}
		return map[string]any{"type": "array", "items": typeToSchema(elem, seen)}
	case reflect.Map:
		if t.Key().Kind() == reflect.String {
			elem := t.Elem()
			// Same cycle-break logic as the Slice case above.
			if seen[elem] || elem == t {
				return map[string]any{"type": "object"}
			}
			return map[string]any{"type": "object", "additionalProperties": typeToSchema(elem, seen)}
		}
		return map[string]any{"type": "object"}
	case reflect.Struct:
		// Cycle detection: if we're already processing this struct, break the cycle.
		if seen[t] {
			return map[string]any{}
		}
		seen[t] = true
		result := structToSchema(t, seen)
		delete(seen, t)
		return result
	default:
		return map[string]any{}
	}
}

func structToSchema(t reflect.Type, seen map[reflect.Type]bool) map[string]any {
	properties := make(map[string]any)
	var required []string

	collectFields(t, properties, &required, seen)

	schema := map[string]any{
		"type":                 "object",
		"properties":           properties,
		"additionalProperties": false,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

// collectFields recursively processes struct fields, flattening embedded structs.
func collectFields(t reflect.Type, properties map[string]any, required *[]string, seen map[reflect.Type]bool) {
	for i := range t.NumField() {
		field := t.Field(i)

		// Flatten embedded (anonymous) structs.
		if field.Anonymous {
			ft := field.Type
			if ft.Kind() == reflect.Ptr {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				if seen[ft] {
					continue
				}
				collectFields(ft, properties, required, seen)
				continue
			}
		}

		if !field.IsExported() {
			continue
		}

		name := field.Name

		// Parse json tag.
		if tag := field.Tag.Get("json"); tag != "" {
			parts := strings.Split(tag, ",")
			if parts[0] == "-" && (len(parts) == 1 || parts[1] != "") {
				continue
			}
			if parts[0] != "" {
				name = parts[0]
			}
		}

		prop := typeToSchema(field.Type, seen)

		// Parse jsonschema tag.
		if tag := field.Tag.Get("jsonschema"); tag != "" {
			parseSchemaTag(tag, prop)
		}

		properties[name] = prop

		// All properties are required (OpenAI strict mode).
		// Pointer types are already nullable via type: ["<base>", "null"].
		*required = append(*required, name)
	}
}

func parseSchemaTag(tag string, prop map[string]any) {
	for part := range strings.SplitSeq(tag, ",") {
		k, v, ok := strings.Cut(part, "=")
		if !ok {
			continue
		}
		key, value := strings.TrimSpace(k), strings.TrimSpace(v)
		switch key {
		case "description":
			prop["description"] = value
		case "enum":
			prop["enum"] = strings.Split(value, "|")
		}
	}
}
