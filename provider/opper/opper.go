// Package opper provides an Opper language model implementation for GoAI.
//
// Opper is an EU-hosted, OpenAI-compatible AI gateway that serves models from
// many providers behind one API key. Requests go to the /chat/completions
// endpoint at https://api.opper.ai/v3/compat. See https://opper.ai and
// https://docs.opper.ai.
package opper

import (
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const defaultBaseURL = "https://api.opper.ai/v3/compat"

// Option configures the Opper provider.
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

// Chat creates an Opper language model for the given model ID.
//
// Model IDs are pool names such as "claude-sonnet-4-6" or "gpt-5.4-mini",
// which Opper routes across the providers serving that model. A
// provider/model ID such as "anthropic/claude-sonnet-4-6" pins one route.
// The API key is read from OPPER_API_KEY unless WithAPIKey or
// WithTokenSource is set, and OPPER_BASE_URL overrides the default
// endpoint unless WithBaseURL is set.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("OPPER_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("OPPER_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:        "opper",
		ModelID:           modelID,
		BaseURL:           o.baseURL,
		TokenSource:       o.tokenSource,
		TokenRequired:     true,
		Headers:           o.headers,
		HTTPClient:        o.httpClient,
		Capabilities:      chatCaps,
		WarnPromptCaching: true,
		RequestConfig: openaicompat.RequestConfig{
			IncludeStreamOptions: true,
		},
	})
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
