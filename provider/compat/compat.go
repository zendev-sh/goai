// Package compat provides a generic OpenAI-compatible language model for GoAI.
//
// Unlike other providers, compat has no default base URL -- callers must provide one
// via WithBaseURL. Authentication is optional: when no API key or token source is
// configured, the Authorization header is omitted entirely.
//
// This provider is useful for connecting to any OpenAI-compatible API such as
// LiteLLM, LocalAI, or custom deployments. For Ollama and vLLM specifically,
// see the dedicated convenience packages.
package compat

import (
	"net/http"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

// Option configures the generic OpenAI-compatible provider.
type Option func(*options)

type options struct {
	providerID  string
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithProviderID overrides the provider name used in error messages.
// Defaults to "compat".
func WithProviderID(id string) Option {
	return func(o *options) { o.providerID = id }
}

// WithAPIKey sets a static API key for authentication.
// When not set, the Authorization header is omitted.
func WithAPIKey(key string) Option {
	return func(o *options) { o.tokenSource = provider.StaticToken(key) }
}

// WithTokenSource sets a dynamic token source for authentication.
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) { o.tokenSource = ts }
}

// WithBaseURL sets the API base URL. This is required for the generic provider.
func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

// WithHeaders sets additional HTTP headers sent with every request.
func WithHeaders(h map[string]string) Option {
	return func(o *options) { o.headers = h }
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.httpClient = c }
}

func resolveOptions(opts []Option) options {
	o := options{providerID: "compat"}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// Chat creates a generic OpenAI-compatible language model.
// WithBaseURL is required; calls will fail without it.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := resolveOptions(opts)
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:           o.providerID,
		ModelID:              modelID,
		BaseURL:              o.baseURL,
		BaseURLRequired:      true,
		TokenSource:          o.tokenSource,
		TokenRequired:        false,
		Headers:              o.headers,
		HTTPClient:           o.httpClient,
		Capabilities:         chatCaps,
		IncludeStreamOptions: true,
		WarnPromptCaching:    true,
	})
}

// Embedding creates a generic OpenAI-compatible embedding model.
// WithBaseURL is required; calls will fail without it.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := resolveOptions(opts)
	return openaicompat.NewEmbeddingModel(openaicompat.EmbeddingModelConfig{
		ProviderID:      o.providerID,
		ModelID:         modelID,
		BaseURL:         o.baseURL,
		BaseURLRequired: true,
		TokenSource:     o.tokenSource,
		TokenRequired:   false,
		Headers:         o.headers,
		HTTPClient:      o.httpClient,
	})
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
