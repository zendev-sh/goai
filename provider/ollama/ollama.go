// Package ollama provides an Ollama language model implementation for GoAI.
//
// Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1 by default.
// No authentication is required. This package is a convenience wrapper around
// the generic compat provider.
package ollama

import (
	"net/http"

	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/compat"
)

const defaultBaseURL = "http://localhost:11434/v1"

// Option configures the Ollama provider.
type Option func(*options)

type options struct {
	baseURL    string
	headers    map[string]string
	httpClient *http.Client
}

// WithBaseURL overrides the default Ollama API base URL.
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

// Chat creates an Ollama language model for the given model ID.
// Defaults to http://localhost:11434/v1, no authentication.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	return compat.Chat(modelID, toCompatOpts(o)...)
}

// Embedding creates an Ollama embedding model for the given model ID.
// Defaults to http://localhost:11434/v1, no authentication.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	return compat.Embedding(modelID, toCompatOpts(o)...)
}

func toCompatOpts(o options) []compat.Option {
	copts := []compat.Option{compat.WithProviderID("ollama"), compat.WithBaseURL(o.baseURL)}
	if o.headers != nil {
		copts = append(copts, compat.WithHeaders(o.headers))
	}
	if o.httpClient != nil {
		copts = append(copts, compat.WithHTTPClient(o.httpClient))
	}
	return copts
}
