// Package fptcloud provides an FPT Smart Cloud AI Marketplace language and
// embedding model implementation for GoAI, using the OpenAI-compatible
// endpoints.
//
// The platform offers two regions: "global" (default, mkp-api.fptcloud.com)
// and "jp" (mkp-api.fptcloud.jp). Use WithRegion to switch.
package fptcloud

import (
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const (
	baseURLGlobal = "https://mkp-api.fptcloud.com/v1"
	baseURLJP     = "https://mkp-api.fptcloud.jp/v1"
)

// Option configures the FPT Smart Cloud provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	region      string
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

// WithRegion selects the API region. Valid values: "global" (default) or "jp".
// Unknown values fall back to "global".
func WithRegion(region string) Option {
	return func(o *options) { o.region = region }
}

// WithBaseURL overrides the region-derived base URL entirely.
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

func regionBaseURL(region string) string {
	switch region {
	case "jp":
		return baseURLJP
	default:
		return baseURLGlobal
	}
}

func resolveOptions(opts []Option) options {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("FPT_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.region == "" {
		o.region = os.Getenv("FPT_REGION")
	}
	if o.baseURL == "" {
		if envBase := os.Getenv("FPT_BASE_URL"); envBase != "" {
			o.baseURL = envBase
		} else {
			o.baseURL = regionBaseURL(o.region)
		}
	}
	return o
}

// Chat creates an FPT Smart Cloud language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := resolveOptions(opts)
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:           "fptcloud",
		ModelID:              modelID,
		BaseURL:              o.baseURL,
		TokenSource:          o.tokenSource,
		TokenRequired:        true,
		Headers:              o.headers,
		HTTPClient:           o.httpClient,
		Capabilities:         chatCaps,
		IncludeStreamOptions: true,
		WarnPromptCaching:    true,
	})
}

// Embedding creates an FPT Smart Cloud embedding model for the given model ID.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := resolveOptions(opts)
	return openaicompat.NewEmbeddingModel(openaicompat.EmbeddingModelConfig{
		ProviderID:    "fptcloud",
		ModelID:       modelID,
		BaseURL:       o.baseURL,
		TokenSource:   o.tokenSource,
		TokenRequired: true,
		Headers:       o.headers,
		HTTPClient:    o.httpClient,
	})
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
