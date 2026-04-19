// Package fptcloud provides an FPT Smart Cloud AI Marketplace language and
// embedding model implementation for GoAI, using the OpenAI-compatible
// endpoints.
//
// The platform offers two regions: "global" (default, mkp-api.fptcloud.com)
// and "jp" (mkp-api.fptcloud.jp). Use WithRegion to switch.
package fptcloud

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

const (
	baseURLGlobal = "https://mkp-api.fptcloud.com/v1"
	baseURLJP     = "https://mkp-api.fptcloud.jp/v1"
)

// Option configures the FPT Smart Cloud provider.
type Option func(*options)

type options struct {
	tokenSource provider.TokenSource
	region      string // "global" | "jp"
	baseURL     string // explicit override
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
	return &chatModel{id: modelID, opts: resolveOptions(opts)}
}

// Embedding creates an FPT Smart Cloud embedding model for the given model ID.
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
		fmt.Fprintf(os.Stderr, "goai: fptcloud: WithPromptCaching is not supported and will be ignored\n")
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
		fmt.Fprintf(os.Stderr, "goai: fptcloud: WithPromptCaching is not supported and will be ignored\n")
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
		ProviderID: "fptcloud",
	}, goai.ParseHTTPErrorWithHeaders)
}

// --- Embedding Model ---

type embeddingModel struct {
	id   string
	opts options
}

func (m *embeddingModel) ModelID() string { return m.id }

// MaxValuesPerCall returns the maximum batch size. 2048 is a safe default.
func (m *embeddingModel) MaxValuesPerCall() int { return 2048 }

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
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
		return nil, goai.ParseHTTPErrorWithHeaders("fptcloud", resp.StatusCode, respBody, resp.Header)
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
		return "", errors.New("goai: no API key or token source configured")
	}
	return ts.Token(ctx)
}
