package goai

import (
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/zendev-sh/goai/provider"
)

func TestDefaultOptions(t *testing.T) {
	o := defaultOptions()
	if o.MaxSteps != 1 {
		t.Errorf("MaxSteps = %d, want 1", o.MaxSteps)
	}
	if o.MaxRetries != 2 {
		t.Errorf("MaxRetries = %d, want 2", o.MaxRetries)
	}
}

func TestApplyOptions(t *testing.T) {
	o := applyOptions(
		WithSystem("You are helpful."),
		WithPrompt("hello"),
		WithMaxOutputTokens(1000),
		WithTemperature(0.7),
		WithTopP(0.9),
		WithStopSequences("END", "STOP"),
		WithMaxSteps(5),
		WithMaxRetries(3),
		WithTimeout(30*time.Second),
		WithPromptCaching(true),
		WithToolChoice("auto"),
		WithHeaders(map[string]string{"X-Custom": "value"}),
		WithProviderOptions(map[string]any{"key": "val"}),
	)

	if o.System != "You are helpful." {
		t.Errorf("System = %q", o.System)
	}
	if o.Prompt != "hello" {
		t.Errorf("Prompt = %q", o.Prompt)
	}
	if o.MaxOutputTokens != 1000 {
		t.Errorf("MaxOutputTokens = %d", o.MaxOutputTokens)
	}
	if o.Temperature == nil || *o.Temperature != 0.7 {
		t.Errorf("Temperature = %v", o.Temperature)
	}
	if o.TopP == nil || *o.TopP != 0.9 {
		t.Errorf("TopP = %v", o.TopP)
	}
	if len(o.StopSequences) != 2 {
		t.Errorf("StopSequences = %v", o.StopSequences)
	}
	if o.MaxSteps != 5 {
		t.Errorf("MaxSteps = %d", o.MaxSteps)
	}
	if o.MaxRetries != 3 {
		t.Errorf("MaxRetries = %d", o.MaxRetries)
	}
	if o.Timeout != 30*time.Second {
		t.Errorf("Timeout = %v", o.Timeout)
	}
	if !o.PromptCaching {
		t.Error("PromptCaching should be true")
	}
	if o.ToolChoice != "auto" {
		t.Errorf("ToolChoice = %q", o.ToolChoice)
	}
	if o.Headers["X-Custom"] != "value" {
		t.Errorf("Headers = %v", o.Headers)
	}
	if o.ProviderOptions["key"] != "val" {
		t.Errorf("ProviderOptions = %v", o.ProviderOptions)
	}
}

func TestWithMessages(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
	}

	o := applyOptions(WithMessages(msgs...))
	if len(o.Messages) != 2 {
		t.Fatalf("Messages = %d, want 2", len(o.Messages))
	}
	if o.Messages[0].Role != provider.RoleUser {
		t.Errorf("Messages[0].Role = %v", o.Messages[0].Role)
	}
}

func TestWithTools(t *testing.T) {
	tool := Tool{
		Name:        "read",
		Description: "Read a file",
	}
	o := applyOptions(WithTools(tool))
	if len(o.Tools) != 1 {
		t.Fatalf("Tools = %d, want 1", len(o.Tools))
	}
	if o.Tools[0].Name != "read" {
		t.Errorf("Tools[0].Name = %q", o.Tools[0].Name)
	}
}

func TestWithTemperature_Zero(t *testing.T) {
	o := applyOptions(WithTemperature(0.0))
	if o.Temperature == nil {
		t.Fatal("Temperature should not be nil")
	}
	if *o.Temperature != 0.0 {
		t.Errorf("Temperature = %v, want 0.0", *o.Temperature)
	}
}

func TestWithProviderOptions_ValidValues(t *testing.T) {
	// Should not panic with JSON-serializable values.
	applyOptions(WithProviderOptions(map[string]any{
		"string":  "hello",
		"number":  42,
		"bool":    true,
		"nil":     nil,
		"nested":  map[string]any{"key": "val"},
		"slice":   []any{1, "two", 3.0},
		"pointer": (*int)(nil),
	}))
}

func TestWithProviderOptions_Channel(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for channel value")
		}
	}()
	applyOptions(WithProviderOptions(map[string]any{
		"ch": make(chan int),
	}))
}

func TestWithProviderOptions_Func(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for func value")
		}
	}()
	applyOptions(WithProviderOptions(map[string]any{
		"fn": func() {},
	}))
}

func TestWithProviderOptions_NestedChannel(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nested channel")
		}
	}()
	applyOptions(WithProviderOptions(map[string]any{
		"nested": map[string]any{
			"ch": make(chan string),
		},
	}))
}

func TestWithEmbeddingProviderOptions_Func(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for func value")
		}
	}()
	applyOptions(WithEmbeddingProviderOptions(map[string]any{
		"callback": func() {},
	}))
}

func TestWithProviderOptions_UnsafePointer(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for unsafe.Pointer value")
		}
	}()
	var x int
	applyOptions(WithProviderOptions(map[string]any{
		"ptr": unsafe.Pointer(&x),
	}))
}

func TestWithProviderOptions_StructWithChan(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for struct containing chan field")
		}
	}()
	type badStruct struct{ Ch chan int }
	applyOptions(WithProviderOptions(map[string]any{
		"x": badStruct{Ch: make(chan int)},
	}))
}

func TestWithProviderOptions_UnexportedEmbeddedStruct(t *testing.T) {
	// Unexported embedded struct types: encoding/json promotes their exported fields
	// into the outer struct (by design ;  see encoding/json's typeFields logic).
	// json.Marshal(outer{inner{make(chan int)}}) returns "unsupported type: chan int",
	// so the validator must also reject this. Both panic for the same reason.
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic: unexported embedded struct with chan must be detected")
		}
	}()
	type inner struct{ Ch chan int }
	type outer struct{ inner }
	applyOptions(WithProviderOptions(map[string]any{
		"x": outer{inner{make(chan int)}},
	}))
}

func TestWithProviderOptions_CyclicMap(t *testing.T) {
	// A cyclic map causes json.Marshal to fail; the validator must reject it early.
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic: cyclic map would cause json.Marshal to fail downstream")
		}
	}()
	m := map[string]any{}
	m["self"] = m
	applyOptions(WithProviderOptions(m))
}

func TestWithProviderOptions_CyclicSlice(t *testing.T) {
	// A cyclic slice causes infinite recursion caught by the depth limit.
	//
	// How the cycle persists through reflect: `s[0] = s` stores a copy of the slice
	// header (same backing array pointer) as an interface{} at index 0. Reflect traversal
	// then cycles as: Slice → Index(0) → Interface → Elem() → Slice → Index(0) → ...
	// Each complete cycle increments depth by 2: the Interface case recurses with depth+1
	// into Elem(), then the Slice case recurses with depth+1 into Index(0). Slices are not tracked
	// by address in seen (only Ptr and Map are), so the depth limit > 1000 is the only
	// guard. The test verifies that the depth guard fires before a stack overflow.
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic: cyclic slice must be caught by depth limit")
		}
	}()
	s := []any{nil}
	s[0] = s
	applyOptions(WithProviderOptions(map[string]any{"s": s}))
}

func TestWithProviderOptions_DiamondPattern(t *testing.T) {
	// Diamond pattern: same pointer reachable via two paths within a single top-level key.
	// json.Marshal handles this fine (serializes the value at each path); must NOT panic.
	// This exercises the seenDone early-return in checkJSONSerializable: whichever of
	// "p" or "q" is visited first (Go map iteration order is random) marks shared as
	// seenDone; the second visit hits the seenDone early-return and skips re-traversal.
	shared := map[string]any{"x": 1}
	applyOptions(WithProviderOptions(map[string]any{
		"a": map[string]any{"p": shared, "q": shared}, // same pointer via two paths under key "a"
	}))
}

func TestWithProviderOptions_CrossKeyDiamond(t *testing.T) {
	// Cross-key diamond: same clean map pointer appears under two different top-level keys.
	// Because validateProviderOptions creates a fresh seen map per top-level key, each
	// key's traversal is independent ;  no false-positive panic from stale seenDone entries.
	shared := map[string]any{"x": 1}
	applyOptions(WithProviderOptions(map[string]any{
		"a": shared,
		"b": shared,
	}))
}

func TestWithProviderOptions_MapWithNonStringKey(t *testing.T) {
	// Maps with non-string/non-integer key types are not JSON-serializable.
	// json.Marshal returns "unsupported type: map[chan int]string" for such maps.
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic: map with chan key type must be detected")
		}
	}()
	applyOptions(WithProviderOptions(map[string]any{
		"m": map[chan int]string{},
	}))
}

func TestWithProviderOptions_StructWithJSONDashChan(t *testing.T) {
	// A field tagged json:"-" is excluded by json.Marshal; must NOT panic.
	type structWithIgnored struct {
		Name string  `json:"name"`
		Skip chan int `json:"-"`
	}
	applyOptions(WithProviderOptions(map[string]any{
		"x": structWithIgnored{Name: "ok", Skip: make(chan int)},
	}))
}

func TestWithProviderOptions_StructWithJSONDashComma_Panics(t *testing.T) {
	// json:"-," means the field IS marshaled (with key "-"), unlike json:"-" which excludes it.
	// The validator must panic here because json.Marshal would also fail on a chan field.
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic: json:\"-,\" field with chan is serialized by json.Marshal and must be rejected")
		}
	}()
	type structWithDashName struct {
		Ch chan int `json:"-,"`
	}
	applyOptions(WithProviderOptions(map[string]any{
		"x": structWithDashName{Ch: make(chan int)},
	}))
}

func TestWithProviderOptions_PanicMessageContainsCaller(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("panic value is not a string: %v", r)
		}
		if !strings.Contains(msg, "WithProviderOptions") {
			t.Errorf("panic message %q should contain caller name WithProviderOptions", msg)
		}
	}()
	applyOptions(WithProviderOptions(map[string]any{
		"fn": func() {},
	}))
}

func TestWithEmbeddingProviderOptions_PanicMessageContainsCaller(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("panic value is not a string: %v", r)
		}
		if !strings.Contains(msg, "WithEmbeddingProviderOptions") {
			t.Errorf("panic message %q should contain caller name WithEmbeddingProviderOptions", msg)
		}
	}()
	applyOptions(WithEmbeddingProviderOptions(map[string]any{
		"fn": func() {},
	}))
}

func TestWithProviderOptions_ReusedSubmapAlwaysChecked(t *testing.T) {
	// "AlwaysChecked" means: a bad value in a reused submap is always detected,
	// regardless of which key is visited first ;  not that all keys are traversed.
	//
	// Go map iteration is non-deterministic. Both "a" and "b" point to the same bad
	// submap, so whichever key is iterated first finds the func and panics immediately.
	// The panic (not `seen` map deduplication) stops execution before the other key
	// is reached. This test verifies the observable guarantee: panic always fires.
	// See TestWithProviderOptions_CrossKeyDiamond for the clean-map cross-key case
	// (which verifies no false-positive panic).
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic: bad value in reused submap must always be detected")
		}
	}()
	bad := map[string]any{"fn": func() {}}
	applyOptions(WithProviderOptions(map[string]any{
		"a": bad,
		"b": bad,
	}))
}

func TestWithProviderOptions_NilMap(t *testing.T) {
	// A nil map value is JSON-serializable (encodes as null); must NOT panic.
	// Exercises the v.IsNil() early-return in the reflect.Map case.
	applyOptions(WithProviderOptions(map[string]any{
		"m": (map[string]any)(nil),
	}))
}

func TestWithProviderOptions_NestedNilValue(t *testing.T) {
	// A nil interface{} value nested inside a non-nil map is JSON-serializable (encodes as null).
	// v.MapIndex on an existing key with a nil any value returns Kind()==Interface with IsNil()==true.
	// The `case reflect.Interface: if !v.IsNil()` branch in checkJSONSerializable handles this:
	// IsNil()==true → skip recursion, return safely. Must NOT panic.
	applyOptions(WithProviderOptions(map[string]any{
		"outer": map[string]any{
			"nil_val": nil,
			"ok_val":  "hello",
		},
	}))
}

func TestWithProviderOptions_StructUnexportedField(t *testing.T) {
	// Unexported non-anonymous fields are skipped by json.Marshal; must NOT panic
	// even when the unexported field holds a non-serializable type.
	// Exercises the !f.IsExported() && !f.Anonymous skip branch in the Struct case.
	type withUnexported struct {
		Name string
		skip chan int // unexported, non-anonymous: json.Marshal ignores it
	}
	applyOptions(WithProviderOptions(map[string]any{
		"x": withUnexported{Name: "ok", skip: make(chan int)},
	}))
}

func TestWithProviderOptions_CyclicPointer(t *testing.T) {
	// A self-referential pointer creates a true cycle: n → n.Next → n → ...
	// The Ptr seenInProgress branch in checkJSONSerializable detects this.
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic: cyclic pointer must be detected")
		}
	}()
	type node struct{ Next *node }
	n := &node{}
	n.Next = n
	applyOptions(WithProviderOptions(map[string]any{"n": n}))
}

func TestWithProviderOptions_DiamondViaPointer(t *testing.T) {
	// Same non-nil pointer reachable via two struct fields ;  diamond, not cycle.
	// json.Marshal handles this fine; the Ptr seenDone early-return must NOT panic.
	type leaf struct{ Value int }
	shared := &leaf{Value: 1}
	type branch struct{ A, B *leaf }
	applyOptions(WithProviderOptions(map[string]any{
		"b": branch{A: shared, B: shared},
	}))
}

func TestWithProviderOptions_IntegerMapKey(t *testing.T) {
	// Maps with integer key types are JSON-serializable (keys become string digits).
	// The integer-key branches in the map key switch must NOT panic.
	applyOptions(WithProviderOptions(map[string]any{
		"m": map[int]string{1: "one", 2: "two"},
	}))
}

func TestWithProviderOptions_Array(t *testing.T) {
	// Fixed-size Go arrays are JSON-serializable; must NOT panic.
	// Array uses the same case branch as Slice in checkJSONSerializable.
	applyOptions(WithProviderOptions(map[string]any{
		"arr": [3]int{1, 2, 3},
	}))
}

func TestWithProviderOptions_DeepNestingAllowed(t *testing.T) {
	// Verify that legitimate (non-cyclic) nesting up to 100 levels deep does not panic.
	// The depth limit is > 1000, so 100-level nesting should always be accepted.
	nested := map[string]any{"leaf": "ok"}
	for range 100 {
		nested = map[string]any{"next": nested}
	}
	applyOptions(WithProviderOptions(nested))
}

func TestWithProviderOptions_NestingNearLimit(t *testing.T) {
	// Verify that nesting at the exact depth limit does not panic.
	// validateProviderOptions calls checkJSONSerializable with the VALUE of each top-level
	// key at depth=0. Each nesting level adds 2 (one Map→value step, one Interface→Elem step).
	// 500 added wrappers: validateProviderOptions strips one level (iterates the top-level opts map),
	// so the value passed to checkJSONSerializable at depth=0 is the 499-wrapper map. Each
	// nesting level adds 2 depth increments (Map + Interface), so string "ok" is entered at
	// depth=1000 (499×2 + 2 for the leaf {"leaf":"ok"} map). 1000 is NOT > 1000 → no panic.
	// Together with NestingExceedsLimit (501 wrappers, panics), this pair tightly constrains
	// the boundary: 500 passes, 501 fails.
	nested := map[string]any{"leaf": "ok"}
	for range 500 {
		nested = map[string]any{"next": nested}
	}
	applyOptions(WithProviderOptions(nested))
}

func TestWithProviderOptions_NestingExceedsLimit(t *testing.T) {
	// Verify that nesting one level beyond the depth limit panics.
	// 501 added wrappers: validateProviderOptions strips one level, so value at depth=0 is
	// the 500-wrapper map. 500×2 = 1000 depth increments reach the leaf {"leaf":"ok"} at
	// depth=1000 (no panic yet). The leaf map's Map case calls checkJSONSerializable at
	// depth=1001 for its Interface-wrapped string value; the depth guard fires immediately.
	// The depth guard at the top of checkJSONSerializable fires immediately (before the type
	// switch), so the string is never examined.
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nesting exceeding depth limit")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("panic value is not a string: %v", r)
		}
		if !strings.Contains(msg, "exceeds maximum nesting depth") {
			t.Errorf("panic message %q should contain 'exceeds maximum nesting depth'", msg)
		}
	}()
	nested := map[string]any{"leaf": "ok"}
	for range 501 {
		nested = map[string]any{"next": nested}
	}
	applyOptions(WithProviderOptions(nested))
}

func TestApplyOptions_Defaults(t *testing.T) {
	// No options applied -- should get defaults.
	o := applyOptions()
	if o.Temperature != nil {
		t.Errorf("Temperature should be nil by default, got %v", o.Temperature)
	}
	if o.TopP != nil {
		t.Errorf("TopP should be nil by default, got %v", o.TopP)
	}
	if o.System != "" {
		t.Errorf("System should be empty by default")
	}
	if len(o.Messages) != 0 {
		t.Errorf("Messages should be empty by default")
	}
}
