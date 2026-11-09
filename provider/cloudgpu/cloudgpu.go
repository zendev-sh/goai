// Package cloudgpu provides a CloudGPU language model implementation for GoAI.
//
// CloudGPU serves an OpenAI-compatible Chat Completions API at
// https://cloudgpu.app/v1 with models such as DeepSeek, Kimi, GLM and
// MiniMax. See https://cloudgpu.app/api/docs.
package cloudgpu

import (
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const defaultBaseURL = "https://cloudgpu.app/v1"

// Option configures the CloudGPU provider.
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

// Chat creates a CloudGPU language model for the given model ID.
//
// Model IDs are bare names such as "deepseek-v4.1-flash", "kimi-k3" or
// "glm-5.3-flash". The API key is read from CLOUDGPU_API_KEY unless
// WithAPIKey or WithTokenSource is set, and CLOUDGPU_BASE_URL overrides the
// default endpoint unless WithBaseURL is set.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("CLOUDGPU_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("CLOUDGPU_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:        "cloudgpu",
		ModelID:           modelID,
		BaseURL:           o.baseURL,
		TokenSource:       o.tokenSource,
		TokenRequired:     true,
		Headers:           o.headers,
		HTTPClient:        o.httpClient,
		Capabilities:      chatCaps,
		WarnPromptCaching: true,
		RequestConfig: openaicompat.RequestConfig{
			IncludeStreamOptions:    true,
			IncludeReasoningContent: true,
		},
	})
}

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
