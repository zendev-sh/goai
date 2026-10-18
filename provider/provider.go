// Package provider defines the interfaces and types that AI providers implement.
//
// GoAI ships provider implementations in sub-packages (openai, anthropic, google, etc.).
// Each provider implements one or more of the model interfaces defined here.
package provider

import "context"

// LanguageModel generates text and tool calls from messages.
type LanguageModel interface {
	// ModelID returns the provider-specific model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514").
	ModelID() string

	// DoGenerate performs a non-streaming generation request.
	DoGenerate(ctx context.Context, params GenerateParams) (*GenerateResult, error)

	// DoStream performs a streaming generation request.
	DoStream(ctx context.Context, params GenerateParams) (*StreamResult, error)
}

// CapableModel is an optional interface that LanguageModel implementations
// can satisfy to declare their capabilities. Use ModelCapabilitiesOf to query.
type CapableModel interface {
	Capabilities() ModelCapabilities
}

// ModelCapabilitiesOf returns the model's capabilities if it implements CapableModel,
// or a zero-value ModelCapabilities otherwise.
func ModelCapabilitiesOf(m LanguageModel) ModelCapabilities {
	if c, ok := m.(CapableModel); ok {
		return c.Capabilities()
	}
	return ModelCapabilities{}
}

// RealtimeModel represents a model that supports a long-lived, bidirectional
// realtime session. Unlike LanguageModel, a realtime model can receive client
// events while provider events are being emitted.
//
// Experimental: the realtime API may change incompatibly until it covers the
// Gemini Live, OpenAI Realtime, and OpenAI GPT-Live adapters.
type RealtimeModel interface {
	// ModelID returns the provider-specific model identifier.
	ModelID() string

	// OpenRealtime opens a provider connection using params as the initial
	// session configuration. Implementations must not mutate params or values
	// reachable through its slices, maps, or byte buffers.
	OpenRealtime(ctx context.Context, params RealtimeParams) (RealtimeConnection, error)
}

// RealtimeConnection is a live bidirectional connection to a realtime model.
//
// Implementations must allow Events to be consumed concurrently with Send.
// Send and Close must be safe for concurrent use. Close is idempotent. Events
// is closed exactly once when the connection terminates; Err then reports the
// terminal error, if any.
type RealtimeConnection interface {
	// Events returns the provider event stream. The channel is closed when the
	// connection terminates.
	Events() <-chan RealtimeEvent

	// Send delivers one normalized client event to the provider. Implementations
	// must not retain or mutate caller-owned byte slices or maps after Send
	// returns.
	Send(ctx context.Context, event RealtimeClientEvent) error

	// Close terminates the connection and releases its resources.
	Close() error

	// Err returns the terminal connection error after Events is closed. It
	// returns nil after a normal close and may return nil while still active.
	Err() error
}

// RealtimeCapableModel is implemented by realtime models that declare their
// supported modalities, transports, and session behavior.
type RealtimeCapableModel interface {
	RealtimeCapabilities() RealtimeCapabilities
}

// RealtimeCapabilitiesOf returns a defensive copy of the model's realtime
// capabilities, or zero-value capabilities when the model does not implement
// RealtimeCapableModel.
func RealtimeCapabilitiesOf(m RealtimeModel) RealtimeCapabilities {
	if c, ok := m.(RealtimeCapableModel); ok {
		return c.RealtimeCapabilities().Clone()
	}
	return RealtimeCapabilities{}
}

// EmbeddingModel generates vector embeddings from text.
type EmbeddingModel interface {
	// ModelID returns the provider-specific model identifier.
	ModelID() string

	// DoEmbed generates embeddings for the given values.
	DoEmbed(ctx context.Context, values []string, params EmbedParams) (*EmbedResult, error)

	// MaxValuesPerCall returns the maximum number of values that can be embedded in a single call.
	// Returns 0 if there is no limit.
	MaxValuesPerCall() int
}

// ImageModel generates images from text prompts.
type ImageModel interface {
	// ModelID returns the provider-specific model identifier.
	ModelID() string

	// DoGenerate generates images from the given parameters.
	DoGenerate(ctx context.Context, params ImageParams) (*ImageResult, error)
}

// VideoModel generates videos from text and media prompts.
type VideoModel interface {
	// ModelID returns the provider-specific model identifier.
	ModelID() string

	// DoGenerate generates videos from the given parameters.
	DoGenerate(ctx context.Context, params VideoParams) (*VideoResult, error)
}
