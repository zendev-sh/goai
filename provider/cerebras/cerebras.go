// Package cerebras provides a Cerebras language model implementation for GoAI.
package cerebras

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel = (*chatModel)(nil)
	_ provider.CapableModel  = (*chatModel)(nil)
)

const defaultBaseURL = "https://api.cerebras.ai/v1"

// Option configures the Cerebras provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithAPIKey sets the Cerebras API key.
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

// WithBaseURL overrides the default API base URL.
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

// Chat creates a Cerebras language model for the given model ID.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve API key from env if not set.
	if o.tokenSource == nil {
		if key := os.Getenv("CEREBRAS_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	// Resolve base URL from env if not overridden.
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("CEREBRAS_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return &chatModel{id: modelID, opts: o}
}

type chatModel struct {
	id   string
	opts options
}

func (m *chatModel) ModelID() string { return m.id }

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{
		Temperature:      true,
		ToolCall:         true,
		InputModalities:  provider.ModalitySet{Text: true},
		OutputModalities: provider.ModalitySet{Text: true},
	}
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	if params.PromptCaching {
		fmt.Fprintf(os.Stderr, "goai: cerebras: WithPromptCaching is not supported and will be ignored\n")
	}
	body := openaicompat.BuildRequest(params, m.id, false, openaicompat.RequestConfig{})

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/chat/completions", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	return openaicompat.ParseResponse(respBody)
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	if params.PromptCaching {
		fmt.Fprintf(os.Stderr, "goai: cerebras: WithPromptCaching is not supported and will be ignored\n")
	}
	body := openaicompat.BuildRequest(params, m.id, true, openaicompat.RequestConfig{
		IncludeStreamOptions: true,
	})

	resp, err := m.doHTTP(ctx, m.opts.baseURL+"/chat/completions", body)
	if err != nil {
		return nil, err
	}

	return openaicompat.NewSSEStream(ctx, resp.Body), nil
}

func (m *chatModel) doHTTP(ctx context.Context, url string, body map[string]any) (*http.Response, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	return httpc.DoJSONRequest(ctx, httpc.RequestConfig{
		URL:        url,
		Token:      token,
		Body:       body,
		Headers:    m.opts.headers,
		HTTPClient: m.opts.httpClient,
		ProviderID: "cerebras",
	}, goai.ParseHTTPErrorWithHeaders)
}

func (m *chatModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("goai: no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}
