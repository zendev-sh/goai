package goai

import (
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// Option configures a generation call.
type Option func(*options)

// options holds all configuration for a generation call.
type options struct {
	// System is the system prompt.
	System string

	// Prompt is a shorthand for a single user message.
	// If set, it is auto-wrapped as a UserMessage and prepended to Messages.
	Prompt string

	// Messages is the conversation history.
	Messages []provider.Message

	// Tools available to the model.
	Tools []Tool

	// MaxOutputTokens limits the response length. 0 means provider default.
	MaxOutputTokens int

	// Temperature controls randomness. nil means provider default.
	Temperature *float64

	// TopP controls nucleus sampling. nil means provider default.
	TopP *float64

	// TopK limits sampling to the top K tokens. nil means provider default.
	TopK *int

	// FrequencyPenalty penalizes tokens based on frequency in the text so far. nil means provider default.
	FrequencyPenalty *float64

	// PresencePenalty penalizes tokens that have appeared in the text so far. nil means provider default.
	PresencePenalty *float64

	// Seed for deterministic generation. nil means provider default.
	Seed *int

	// StopSequences causes generation to stop when encountered.
	StopSequences []string

	// MaxSteps is the maximum number of auto tool loop iterations (default 1 = no loop).
	MaxSteps int

	// MaxRetries is the retry count for transient errors (default 2).
	MaxRetries int

	// Timeout for the entire generation call.
	Timeout time.Duration

	// Headers are additional HTTP headers for this request.
	Headers map[string]string

	// ProviderOptions are provider-specific request parameters.
	ProviderOptions map[string]any

	// PromptCaching enables provider-specific prompt caching.
	PromptCaching bool

	// ToolChoice controls tool selection: "auto", "none", "required", or a specific tool name.
	ToolChoice string

	// OnStepFinish is called after each generation step completes (including tool execution).
	OnStepFinish func(StepResult)

	// OnRequest is called before each model API call.
	OnRequest func(RequestInfo)

	// OnResponse is called after each model API call completes.
	OnResponse func(ResponseInfo)

	// OnToolCall is called after each tool execution.
	OnToolCall func(ToolCallInfo)

	// OnToolCallStart is called before each tool execution.
	OnToolCallStart func(ToolCallStartInfo)

	// ExplicitSchema overrides auto-generated JSON Schema for GenerateObject/StreamObject.
	ExplicitSchema json.RawMessage

	// SchemaName is the schema name sent to providers (default "response").
	SchemaName string

	// MaxParallelCalls controls batch parallelism for EmbedMany (default 4).
	MaxParallelCalls int

	// EmbeddingProviderOptions are provider-specific parameters for embedding requests.
	EmbeddingProviderOptions map[string]any
}

// defaultOptions returns options with sensible defaults.
func defaultOptions() options {
	return options{
		MaxSteps:   1,
		MaxRetries: 2,
	}
}

// applyOptions creates options from defaults and applies all given options.
func applyOptions(opts ...Option) options {
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// WithSystem sets the system prompt.
func WithSystem(s string) Option {
	return func(o *options) { o.System = s }
}

// WithPrompt sets a shorthand user message prompt.
func WithPrompt(s string) Option {
	return func(o *options) { o.Prompt = s }
}

// WithMessages sets the conversation history.
func WithMessages(msgs ...provider.Message) Option {
	return func(o *options) { o.Messages = msgs }
}

// WithTools sets the tools available to the model.
func WithTools(tools ...Tool) Option {
	return func(o *options) { o.Tools = tools }
}

// WithMaxOutputTokens limits the response length.
func WithMaxOutputTokens(n int) Option {
	return func(o *options) { o.MaxOutputTokens = n }
}

// WithTemperature controls randomness.
func WithTemperature(t float64) Option {
	return func(o *options) { o.Temperature = &t }
}

// WithTopP controls nucleus sampling.
func WithTopP(p float64) Option {
	return func(o *options) { o.TopP = &p }
}

// WithTopK limits sampling to the top K tokens.
func WithTopK(k int) Option {
	return func(o *options) { o.TopK = &k }
}

// WithFrequencyPenalty sets the frequency penalty.
func WithFrequencyPenalty(p float64) Option {
	return func(o *options) { o.FrequencyPenalty = &p }
}

// WithPresencePenalty sets the presence penalty.
func WithPresencePenalty(p float64) Option {
	return func(o *options) { o.PresencePenalty = &p }
}

// WithSeed sets the seed for deterministic generation.
func WithSeed(s int) Option {
	return func(o *options) { o.Seed = &s }
}

// WithStopSequences sets stop sequences.
func WithStopSequences(seqs ...string) Option {
	return func(o *options) { o.StopSequences = seqs }
}

// WithMaxSteps sets the maximum auto tool loop iterations.
func WithMaxSteps(n int) Option {
	return func(o *options) {
		if n < 1 {
			n = 1
		}
		o.MaxSteps = n
	}
}

// WithMaxRetries sets the retry count for transient errors.
// Values below 0 are clamped to 0 (no retries).
func WithMaxRetries(n int) Option {
	return func(o *options) {
		if n < 0 {
			n = 0
		}
		o.MaxRetries = n
	}
}

// WithTimeout sets the timeout for the entire generation call.
func WithTimeout(d time.Duration) Option {
	return func(o *options) { o.Timeout = d }
}

// WithHeaders sets additional HTTP headers.
func WithHeaders(h map[string]string) Option {
	return func(o *options) { o.Headers = h }
}

// WithProviderOptions sets provider-specific request parameters.
// Values must be JSON-serializable (no channels, functions, or unsafe pointers).
func WithProviderOptions(opts map[string]any) Option {
	validateProviderOptions("WithProviderOptions", opts)
	return func(o *options) {
		o.ProviderOptions = opts
	}
}

// WithPromptCaching enables provider-specific prompt caching.
// Currently supported by: Anthropic, Bedrock (Anthropic-family models only), and MiniMax
// (which delegates to Anthropic). Other providers log a warning to stderr when
// this option is set.
func WithPromptCaching(b bool) Option {
	return func(o *options) { o.PromptCaching = b }
}

// Tool choice constants for use with WithToolChoice.
const (
	// ToolChoiceAuto lets the model decide whether to call a tool (default behavior).
	ToolChoiceAuto = "auto"

	// ToolChoiceNone prevents the model from calling any tools.
	ToolChoiceNone = "none"

	// ToolChoiceRequired forces the model to call at least one tool.
	ToolChoiceRequired = "required"
)

// WithToolChoice controls tool selection.
func WithToolChoice(tc string) Option {
	return func(o *options) { o.ToolChoice = tc }
}

// WithExplicitSchema overrides auto-generated JSON Schema for GenerateObject/StreamObject.
func WithExplicitSchema(schema json.RawMessage) Option {
	return func(o *options) { o.ExplicitSchema = schema }
}

// WithSchemaName sets the schema name sent to providers (default "response").
func WithSchemaName(name string) Option {
	return func(o *options) { o.SchemaName = name }
}

// WithMaxParallelCalls sets batch parallelism for EmbedMany.
func WithMaxParallelCalls(n int) Option {
	return func(o *options) { o.MaxParallelCalls = n }
}

// WithEmbeddingProviderOptions sets provider-specific parameters for embedding requests.
// Values must be JSON-serializable (no channels, functions, or unsafe pointers).
func WithEmbeddingProviderOptions(opts map[string]any) Option {
	validateProviderOptions("WithEmbeddingProviderOptions", opts)
	return func(o *options) {
		o.EmbeddingProviderOptions = opts
	}
}

// validateProviderOptions panics if any value in the map is not JSON-serializable.
// This catches programming errors (channels, functions, unsafe pointers, cycles) early,
// before they reach MustMarshalJSON deep in provider code.
// caller is the WithXxxProviderOptions function name, included in the panic message.
func validateProviderOptions(caller string, opts map[string]any) {
	for k, v := range opts {
		if v == nil {
			continue
		}
		// Fresh seen map per top-level key: prevents stale entries from one key
		// suppressing cycle/bad-value detection for an unrelated key.
		// Values: seenInProgress (currently on stack) or seenDone (fully processed).
		seen := make(map[uintptr]int)
		checkJSONSerializable(caller, k, reflect.ValueOf(v), seen, 0)
	}
}

// seenInProgress and seenDone are the two states tracked in the seen map passed to
// checkJSONSerializable. In-progress means the address is currently on the call stack
// (true cycle); done means it was fully processed via a different path (diamond pattern).
const (
	seenInProgress = 1
	seenDone       = 2
)

// checkJSONSerializable panics if v contains a type that json.Marshal cannot handle.
// seen tracks map/pointer addresses using a two-state DFS to distinguish true cycles
// (same address on the current call stack → json.Marshal would fail) from diamond patterns
// (same address reachable via two different paths → json.Marshal handles fine).
// depth guards against slice cycles, which cannot be detected by address (slices are not
// tracked in seen). The limit of 1000 is a pragmatic guard against pathological input ; 
// cyclic slices that escape address-based detection, or unreasonably deep nesting.
func checkJSONSerializable(caller, key string, v reflect.Value, seen map[uintptr]int, depth int) {
	if depth > 1000 {
		// Depth > 1000 indicates either a cyclic slice/array (not address-tracked) or
		// unreasonably deep nesting. Either way the value is not safe for provider use.
		panic(fmt.Sprintf("goai: %s: option key %q exceeds maximum nesting depth (cyclic or too deeply nested)", caller, key))
	}
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.UnsafePointer:
		panic(fmt.Sprintf("goai: %s: option key %q has non-serializable value of type %s", caller, key, v.Type()))
	}

	// If the type implements json.Marshaler, call MarshalJSON in a recover
	// to catch panicking implementations early.
	marshalerType := reflect.TypeOf((*json.Marshaler)(nil)).Elem()
	if v.Type().Implements(marshalerType) || (v.Kind() != reflect.Ptr && reflect.PointerTo(v.Type()).Implements(marshalerType)) {
		func() {
			defer func() {
				if r := recover(); r != nil {
					panic(fmt.Sprintf("goai: %s: option key %q MarshalJSON panicked: %v", caller, key, r))
				}
			}()
			if _, err := json.Marshal(v.Interface()); err != nil {
				panic(fmt.Sprintf("goai: %s: option key %q MarshalJSON failed: %v", caller, key, err))
			}
		}()
		return // Marshaler handles its own serialization
	}

	switch v.Kind() {
	case reflect.Ptr:
		if !v.IsNil() {
			addr := v.Pointer()
			if seen[addr] == seenInProgress {
				panic(fmt.Sprintf("goai: %s: option key %q has a cyclic value (json.Marshal would fail)", caller, key))
			}
			if seen[addr] == seenDone {
				return // diamond pattern: already validated via another path, safe to skip
			}
			seen[addr] = seenInProgress
			checkJSONSerializable(caller, key, v.Elem(), seen, depth+1)
			seen[addr] = seenDone
		}
	case reflect.Interface:
		if !v.IsNil() {
			checkJSONSerializable(caller, key, v.Elem(), seen, depth+1)
		}
	case reflect.Map:
		if v.IsNil() {
			return
		}
		// Validate key type: json.Marshal requires string or integer key types.
		// Keys of other kinds (chan, func, struct, etc.) are not JSON-serializable.
		// TextMarshaler keys are theoretically valid but are not expected in provider options.
		kk := v.Type().Key().Kind()
		switch kk {
		case reflect.String,
			reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			// Valid JSON map key types.
		default:
			panic(fmt.Sprintf("goai: %s: option key %q has a map with non-JSON-serializable key type %s", caller, key, v.Type().Key()))
		}
		addr := v.Pointer()
		if seen[addr] == seenInProgress {
			panic(fmt.Sprintf("goai: %s: option key %q has a cyclic value (json.Marshal would fail)", caller, key))
		}
		if seen[addr] == seenDone {
			return // diamond pattern: already validated via another path, safe to skip
		}
		seen[addr] = seenInProgress
		for _, mk := range v.MapKeys() {
			checkJSONSerializable(caller, key, v.MapIndex(mk), seen, depth+1)
		}
		seen[addr] = seenDone
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			checkJSONSerializable(caller, key, v.Index(i), seen, depth+1)
		}
	case reflect.Struct:
		t := v.Type()
		for i := range t.NumField() {
			f := t.Field(i)
			// Skip unexported non-anonymous fields: json.Marshal ignores them.
			// Anonymous (embedded) fields are included even if the embedding field name is
			// unexported, because json.Marshal promotes their exported fields into the outer struct.
			if !f.IsExported() && !f.Anonymous {
				continue
			}
			// Fields with the exact json tag value "-" are skipped by json.Marshal; do not validate them.
			// Note: json:"-," (comma suffix) means the field name IS "-" and IS marshaled ;  not skipped here.
			if f.Tag.Get("json") == "-" {
				continue
			}
			checkJSONSerializable(caller, key, v.Field(i), seen, depth+1)
		}
	}
}
