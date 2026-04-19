// Package cloudflare provides a Cloudflare Workers AI language and embedding
// model implementation for GoAI, using the OpenAI-compatible endpoints.
//
// See https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/
package cloudflare

import (
	"context"
	"encoding/json"
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
	_ provider.LanguageModel  = (*chatModel)(nil)
	_ provider.CapableModel   = (*chatModel)(nil)
	_ provider.EmbeddingModel = (*embeddingModel)(nil)
)

const defaultAPIBase = "https://api.cloudflare.com/client/v4"

// Option configures the Cloudflare provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	accountID   string
	baseURL     string // full base URL including /accounts/{id}/ai/v1, overrides accountID-derived URL
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
	return &chatModel{id: modelID, opts: resolveOptions(opts)}
}

// Embedding creates a Cloudflare Workers AI embedding model for the given model ID.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	return &embeddingModel{id: modelID, opts: resolveOptions(opts)}
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
		fmt.Fprintf(os.Stderr, "goai: cloudflare: WithPromptCaching is not supported and will be ignored\n")
	}
	if err := m.checkBaseURL(); err != nil {
		return nil, err
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
		fmt.Fprintf(os.Stderr, "goai: cloudflare: WithPromptCaching is not supported and will be ignored\n")
	}
	if err := m.checkBaseURL(); err != nil {
		return nil, err
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

func (m *chatModel) checkBaseURL() error {
	if m.opts.baseURL == "" {
		return errors.New("cloudflare: account ID is required (use WithAccountID or CLOUDFLARE_ACCOUNT_ID)")
	}
	return nil
}

func (m *chatModel) doHTTP(ctx context.Context, url string, body map[string]any) (*http.Response, error) {
	token, err := resolveToken(ctx, m.opts.tokenSource)
	if err != nil {
		return nil, err
	}
	return httpc.DoJSONRequest(ctx, httpc.RequestConfig{
		URL:        url,
		Token:      token,
		Body:       body,
		Headers:    m.opts.headers,
		HTTPClient: m.opts.httpClient,
		ProviderID: "cloudflare",
	}, goai.ParseHTTPErrorWithHeaders)
}

// --- Embedding Model ---

type embeddingModel struct {
	id   string
	opts options
}

func (m *embeddingModel) ModelID() string { return m.id }

// MaxValuesPerCall returns the maximum batch size. Cloudflare accepts multiple
// inputs per request; 100 is a safe default aligned with their documented limits.
func (m *embeddingModel) MaxValuesPerCall() int { return 100 }

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	if m.opts.baseURL == "" {
		return nil, errors.New("cloudflare: account ID is required (use WithAccountID or CLOUDFLARE_ACCOUNT_ID)")
	}

	body := map[string]any{
		"model": m.id,
		"input": values,
	}
	if params.ProviderOptions != nil {
		if v, ok := params.ProviderOptions["dimensions"]; ok {
			body["dimensions"] = v
		}
	}

	token, err := resolveToken(ctx, m.opts.tokenSource)
	if err != nil {
		return nil, err
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", m.opts.baseURL+"/embeddings", jsonBody)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}

	client := m.opts.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, goai.ParseHTTPErrorWithHeaders("cloudflare", resp.StatusCode, respBody, resp.Header)
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

func resolveToken(ctx context.Context, ts provider.TokenSource) (string, error) {
	if ts == nil {
		return "", errors.New("goai: no API token or token source configured")
	}
	return ts.Token(ctx)
}
