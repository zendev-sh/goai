// Package cloudflare provides a Cloudflare Workers AI language and embedding
// model implementation for GoAI, using the OpenAI-compatible endpoints.
//
// See https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/
package cloudflare

import (
	"fmt"
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const defaultAPIBase = "https://api.cloudflare.com/client/v4"

// Option configures the Cloudflare provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	accountID   string
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets a static Cloudflare API token for authentication.
func WithAPIKey(key string) Option {
	return func(o *options) { o.tokenSource = provider.StaticToken(key) }
}

// WithTokenSource sets a dynamic token source for authentication.
func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) { o.tokenSource = ts }
}

// WithAccountID sets the Cloudflare account ID used to build the API URL.
func WithAccountID(id string) Option {
	return func(o *options) { o.accountID = id }
}

// WithBaseURL overrides the full API base URL (including account path and
// `/v1` suffix). Useful for AI Gateway or a proxy. When set, WithAccountID is
// ignored.
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
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("CLOUDFLARE_API_TOKEN"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.accountID == "" {
		o.accountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	}
	if o.baseURL == "" {
		if envBase := os.Getenv("CLOUDFLARE_BASE_URL"); envBase != "" {
			o.baseURL = envBase
		} else if o.accountID != "" {
			o.baseURL = fmt.Sprintf("%s/accounts/%s/ai/v1", defaultAPIBase, o.accountID)
		}
	}
	return o
}

// Chat creates a Cloudflare Workers AI language model for the given model ID.
// Either WithBaseURL or WithAccountID (plus an account ID from env) is required.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := resolveOptions(opts)
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:           "cloudflare",
		ModelID:              modelID,
		BaseURL:              o.baseURL,
		TokenSource:          o.tokenSource,
		TokenRequired:        true,
		BaseURLRequired:      true,
		Headers:              o.headers,
		HTTPClient:           o.httpClient,
		Capabilities:         chatCaps,
		IncludeStreamOptions: true,
		WarnPromptCaching:    true,
	})
}

// Embedding creates a Cloudflare Workers AI embedding model for the given model ID.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := resolveOptions(opts)
	return openaicompat.NewEmbeddingModel(openaicompat.EmbeddingModelConfig{
		ProviderID:          "cloudflare",
		ModelID:             modelID,
		BaseURL:             o.baseURL,
		TokenSource:         o.tokenSource,
		TokenRequired:       true,
		BaseURLRequired:     true,
		Headers:             o.headers,
		HTTPClient:          o.httpClient,
		MaxValuesPerCall:    100,
		ProviderOptionsKeys: []string{"dimensions"},
		EncodingFormat:      "-",
	})
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
