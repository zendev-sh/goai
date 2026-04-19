// Package runpod provides a RunPod language model implementation for GoAI.
//
// RunPod exposes an OpenAI-compatible API via serverless vLLM workers.
// Each deployment has a unique endpoint ID visible in the RunPod console.
//
// Usage:
//
//	model := runpod.Chat("your-endpoint-id", "meta-llama/Llama-3.3-70B-Instruct",
//		runpod.WithAPIKey("your-runpod-api-key"),
//	)
//	result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
package runpod

import (
	"fmt"
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const baseURLTemplate = "https://api.runpod.ai/v2/%s/openai/v1"

// Option configures the RunPod provider.
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

// Chat creates a RunPod language model for the given endpoint and model IDs.
// endpointID is the RunPod serverless endpoint ID (visible in the RunPod console).
func Chat(endpointID, modelID string, opts ...Option) provider.LanguageModel {
	defaultBaseURL := fmt.Sprintf(baseURLTemplate, endpointID)
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("RUNPOD_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("RUNPOD_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:           "runpod",
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

var chatCaps = provider.ModelCapabilities{
	Temperature:      true,
	ToolCall:         true,
	InputModalities:  provider.ModalitySet{Text: true},
	OutputModalities: provider.ModalitySet{Text: true},
}
