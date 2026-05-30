// Package llamacpp provides a llama.cpp server language model implementation for GoAI.
//
// llama.cpp server exposes an OpenAI-compatible API at http://localhost:8080 by default.
// No authentication is required. This package is a convenience wrapper around
// the generic compat provider.
//
// Example usage:
//
//	model := llamacpp.Chat("llama3")
//	result, err := goai.GenerateText(ctx, model, goai.WithMessages(
//		goai.UserMessage("Hello"),
//	))
package llamacpp

import (
	"net/http"
	"net/url"

	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/compat"
)

const defaultBaseURL = "http://localhost:8080"

// Option configures the llama.cpp provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets a static API key for authentication.
// llama.cpp server does not require authentication by default, but this can
// be used if the server is behind a reverse proxy that enforces auth.
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

// WithBaseURL overrides the default llama.cpp server base URL.
// The URL must use http:// or https:// scheme.
//
// llama.cpp is a local-first provider (default http://localhost:8080).
// The user explicitly controls the target URL, so SSRF-style host
// validation is intentionally not applied (same as ollama, vllm).
func WithBaseURL(rawURL string) Option {
	return func(o *options) {
		if u, err := url.Parse(rawURL); err == nil && u.Scheme != "" {
			o.baseURL = rawURL
		}
	}
}

// WithHeaders sets additional HTTP headers sent with every request.
// The map is copied; subsequent modifications by the caller have no effect.
func WithHeaders(h map[string]string) Option {
	return func(o *options) {
		copied := make(map[string]string, len(h))
		for k, v := range h {
			copied[k] = v
		}
		o.headers = copied
	}
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) {
		o.httpClient = c
	}
}

// Chat creates a llama.cpp server language model for the given model ID.
// Defaults to http://localhost:8080, no authentication.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	return compat.Chat(modelID, toCompatOpts(o)...)
}

// Embedding creates a llama.cpp server embedding model for the given model ID.
// Defaults to http://localhost:8080, no authentication.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	return compat.Embedding(modelID, toCompatOpts(o)...)
}

func toCompatOpts(o options) []compat.Option {
	copts := []compat.Option{compat.WithProviderID("llamacpp"), compat.WithBaseURL(o.baseURL)}
	if o.tokenSource != nil {
		copts = append(copts, compat.WithTokenSource(o.tokenSource))
	}
	if o.headers != nil {
		copts = append(copts, compat.WithHeaders(o.headers))
	}
	if o.httpClient != nil {
		copts = append(copts, compat.WithHTTPClient(o.httpClient))
	}
	return copts
}
