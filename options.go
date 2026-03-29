package goai

import (
	"encoding/json"
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
func WithProviderOptions(opts map[string]any) Option {
	return func(o *options) { o.ProviderOptions = opts }
}

// WithPromptCaching enables provider-specific prompt caching.
func WithPromptCaching(b bool) Option {
	return func(o *options) { o.PromptCaching = b }
}

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
func WithEmbeddingProviderOptions(opts map[string]any) Option {
	return func(o *options) { o.EmbeddingProviderOptions = opts }
}
