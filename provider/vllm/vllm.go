// Package vllm provides a vLLM language model implementation for GoAI.
//
// vLLM exposes an OpenAI-compatible API at http://localhost:8000/v1 by default.
// Authentication is optional. This package is a convenience wrapper around
// the generic compat provider.
package vllm

import (
	"net/http"

	"github.com/zendev-sh/goai/provider"
	"github.com/zendev-sh/goai/provider/compat"
)

const defaultBaseURL = "http://localhost:8000/v1"

// Option configures the vLLM provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets a static API key for authentication.
// When not set, the Authorization header is omitted.
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

// WithBaseURL overrides the default vLLM API base URL.
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

// Chat creates a vLLM language model for the given model ID.
// Defaults to http://localhost:8000/v1, no authentication.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	return compat.Chat(modelID, toCompatOpts(o)...)
}

// Embedding creates a vLLM embedding model for the given model ID.
// Defaults to http://localhost:8000/v1, no authentication.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	return compat.Embedding(modelID, toCompatOpts(o)...)
}

func toCompatOpts(o options) []compat.Option {
	copts := []compat.Option{compat.WithProviderID("vllm"), compat.WithBaseURL(o.baseURL)}
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
