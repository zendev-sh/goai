// Package openrouter provides an OpenRouter language model implementation for GoAI.
//
// OpenRouter is a unified API for accessing multiple LLM providers through
// a single OpenAI-compatible endpoint.
package openrouter

import (
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

// Option configures the OpenRouter provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets a static API key for authentication.
func WithAPIKey(key string) Option {
	return func(o *options) { o.tokenSource = provider.StaticToken(key) }
}

// WithTokenSource sets a dynamic token source for authentication.
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) { o.tokenSource = ts }
}

// WithBaseURL overrides the default API base URL.
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

// Chat creates an OpenRouter language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("OPENROUTER_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("OPENROUTER_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:           "openrouter",
		ModelID:              modelID,
		BaseURL:              o.baseURL,
		TokenSource:          o.tokenSource,
		TokenRequired:        true,
		Headers:              mergeHeaders(o.headers),
		HTTPClient:           o.httpClient,
		Capabilities:         chatCaps,
		IncludeStreamOptions: true,
		WarnPromptCaching:    true,
		ExtraBody:            map[string]any{"usage": map[string]any{"include": true}},
	})
}

// mergeHeaders returns user-provided headers with OpenRouter-specific headers added.
// Requested per https://openrouter.ai/docs/api-reference/overview#headers
func mergeHeaders(user map[string]string) map[string]string {
	merged := map[string]string{
		"HTTP-Referer": "https://github.com/zendev-sh/goai",
		"X-Title":      "goai",
	}
	for k, v := range user {
		merged[k] = v
	}
	return merged
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
