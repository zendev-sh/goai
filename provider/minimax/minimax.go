// Package minimax provides a MiniMax language model implementation for GoAI.
//
// MiniMax uses an Anthropic-compatible API (recommended by MiniMax). This
// provider delegates to GoAI's anthropic.Chat() with MiniMax's endpoint,
// following the same pattern as Azure's Anthropic delegation.
//
// Key differences from the Anthropic provider:
//   - Default base URL: https://api.minimax.io/anthropic
//   - Env vars: MINIMAX_API_KEY, MINIMAX_BASE_URL (not ANTHROPIC_*)
//   - No image/document input support
package minimax

import (
	"context"
	"net/http"
	"os"

	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/anthropic"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel = (*chatModel)(nil)
	_ provider.CapableModel  = (*chatModel)(nil)
)

const defaultBaseURL = "https://api.minimax.io/anthropic"

// Option configures the MiniMax provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets a static API key for authentication.
func WithAPIKey(key string) Option {
	return func(o *options) {
		o.tokenSource = provider.StaticToken(key)
	}
}

// WithTokenSource sets a dynamic token source for authentication.
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) {
		o.tokenSource = ts
	}
}

// WithBaseURL overrides the default API base URL.
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.baseURL = url
	}
}

// WithHeaders sets additional HTTP headers sent with every request.
func WithHeaders(h map[string]string) Option {
	return func(o *options) {
		o.headers = h
	}
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) {
		o.httpClient = c
	}
}

// Chat creates a MiniMax language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("MINIMAX_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("MINIMAX_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}

	// Delegate to the Anthropic provider with MiniMax's endpoint.
	anthropicOpts := []anthropic.Option{anthropic.WithBaseURL(o.baseURL)}
	if o.tokenSource != nil {
		anthropicOpts = append(anthropicOpts, anthropic.WithTokenSource(o.tokenSource))
	}
	if o.httpClient != nil {
		anthropicOpts = append(anthropicOpts, anthropic.WithHTTPClient(o.httpClient))
	}
	if len(o.headers) > 0 {
		anthropicOpts = append(anthropicOpts, anthropic.WithHeaders(o.headers))
	}

	return &chatModel{
		inner: anthropic.Chat(modelID, anthropicOpts...),
		id:    modelID,
	}
}

type chatModel struct {
	inner provider.LanguageModel
	id    string
}

func (m *chatModel) ModelID() string { return m.id }

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature:      true,
		ToolCall:         true,
		Reasoning:        true, // Native thinking blocks via Anthropic format
		InputModalities:  provider.ModalitySet{Text: true},
		OutputModalities: provider.ModalitySet{Text: true},
	}
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	return m.inner.DoGenerate(ctx, params)
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	return m.inner.DoStream(ctx, params)
}
