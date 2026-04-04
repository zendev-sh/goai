// Package compat provides a generic OpenAI-compatible language model for GoAI.
//
// Unlike other providers, compat has no default base URL -- callers must provide one
// via WithBaseURL. Authentication is optional: when no API key or token source is
// configured, the Authorization header is omitted entirely.
//
// This provider is useful for connecting to any OpenAI-compatible API such as
// LiteLLM, LocalAI, or custom deployments. For Ollama and vLLM specifically,
// see the dedicated convenience packages.
package compat

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"io"
	"net/http"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/internal/openaicompat"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.LanguageModel  = (*chatModel)(nil)
	_ provider.CapableModel   = (*chatModel)(nil)
	_ provider.EmbeddingModel = (*embeddingModel)(nil)
)

// Option configures the generic OpenAI-compatible provider.
type Option func(*options)

type options struct {
	providerID  string
	tokenSource provider.TokenSource
	baseURL     string
	headers     map[string]string
	httpClient  *http.Client
}

// WithProviderID overrides the provider name used in error messages.
// Defaults to "compat".
func WithProviderID(id string) Option {
	return func(o *options) {
		o.providerID = id
	}
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

// WithBaseURL sets the API base URL. This is required for the generic provider.
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

// Chat creates a generic OpenAI-compatible language model.
// WithBaseURL is required; calls will fail without it.
func Chat(modelID string, opts ...Option) provider.LanguageModel {
	o := options{providerID: "compat"}
	for _, opt := range opts {
		opt(&o)
	}
	return &chatModel{id: modelID, opts: o}
}

// Embedding creates a generic OpenAI-compatible embedding model.
// WithBaseURL is required; calls will fail without it.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{providerID: "compat"}
	for _, opt := range opts {
		opt(&o)
	}
	return &embeddingModel{id: modelID, opts: o}
}

// --- Chat Model ---

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
		fmt.Fprintf(os.Stderr, "goai: %s: WithPromptCaching is not supported and will be ignored\n", m.opts.providerID)
	}
	if m.opts.baseURL == "" {
		return nil, fmt.Errorf("compat: base URL is required (use WithBaseURL)")
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
		fmt.Fprintf(os.Stderr, "goai: %s: WithPromptCaching is not supported and will be ignored\n", m.opts.providerID)
	}
	if m.opts.baseURL == "" {
		return nil, fmt.Errorf("compat: base URL is required (use WithBaseURL)")
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
	// Only resolve token when a token source is configured (auth is optional for compat).
	var token string
	if m.opts.tokenSource != nil {
		var err error
		token, err = m.opts.tokenSource.Token(ctx)
		if err != nil {
			return nil, fmt.Errorf("resolving auth token: %w", err)
		}
	}

	return httpc.DoJSONRequest(ctx, httpc.RequestConfig{
		URL:        url,
		Token:      token,
		Body:       body,
		Headers:    m.opts.headers,
		HTTPClient: m.opts.httpClient,
		ProviderID: m.opts.providerID,
	}, goai.ParseHTTPErrorWithHeaders)
}

// --- Embedding Model ---

type embeddingModel struct {
	id   string
	opts options
}

func (m *embeddingModel) ModelID() string { return m.id }

// MaxValuesPerCall returns the maximum batch size. Returns 2048 as a safe default.
func (m *embeddingModel) MaxValuesPerCall() int { return 2048 }

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	if m.opts.baseURL == "" {
		return nil, fmt.Errorf("compat: base URL is required (use WithBaseURL)")
	}

	body := map[string]any{
		"model":           m.id,
		"input":           values,
		"encoding_format": "float",
	}

	if params.ProviderOptions != nil {
		if v, ok := params.ProviderOptions["dimensions"]; ok {
			body["dimensions"] = v
		}
		if v, ok := params.ProviderOptions["user"]; ok {
			body["user"] = v
		}
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", m.opts.baseURL+"/embeddings", jsonBody)
	req.Header.Set("Content-Type", "application/json")

	// Only set Authorization when a token source is configured.
	if m.opts.tokenSource != nil {
		token, err := m.opts.tokenSource.Token(ctx)
		if err != nil {
			return nil, fmt.Errorf("resolving auth token: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
	}

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}

	resp, err := m.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, goai.ParseHTTPErrorWithHeaders(m.opts.providerID, resp.StatusCode, respBody, resp.Header)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	var result struct {
		Model string `json:"model"`
		Data  []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	embeddings := make([][]float64, len(result.Data))
	for _, d := range result.Data {
		if d.Index >= 0 && d.Index < len(embeddings) {
			embeddings[d.Index] = d.Embedding
		}
	}

	return &provider.EmbedResult{
		Embeddings: embeddings,
		Usage:      provider.Usage{InputTokens: result.Usage.PromptTokens, TotalTokens: result.Usage.TotalTokens},
		Response:   provider.ResponseMetadata{Model: result.Model},
	}, nil
}

func (m *embeddingModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}
