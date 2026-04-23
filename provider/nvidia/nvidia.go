package nvidia

import (
	"net/http"
	"os"

	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

const defaultBaseURL = "https://integrate.api.nvidia.com/v1"

type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

func WithAPIKey(key string) Option {
	return func(o *options) { o.tokenSource = provider.StaticToken(key) }
}

func WithTokenSource(ts provider.TokenSource) Option {
	return func(o *options) { o.tokenSource = ts }
}

func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

func WithHeaders(h map[string]string) Option {
	return func(o *options) { o.headers = h }
}

func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.httpClient = c }
}

func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("NVIDIA_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("NVIDIA_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return openaicompat.NewChatModel(openaicompat.ChatModelConfig{
		ProviderID:           "nvidia",
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

func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	if o.tokenSource == nil {
		if key := os.Getenv("NVIDIA_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("NVIDIA_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return openaicompat.NewEmbeddingModel(openaicompat.EmbeddingModelConfig{
		ProviderID:    "nvidia",
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