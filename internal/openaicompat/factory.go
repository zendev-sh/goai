// Factory constructors for OpenAI-compatible chat and embedding models.
//
// Provider packages use NewChatModel / NewEmbeddingModel to obtain ready-made
// provider.LanguageModel / provider.EmbeddingModel implementations that handle
// the common HTTP dispatch, auth, error parsing, and response decoding. The
// provider package only needs to build the ChatModelConfig / EmbeddingModelConfig
// (including its public With* options) and delegate to the factory.

package openaicompat

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
	"github.com/zendev-sh/goai/provider"
)

// ChatModelConfig configures a chat language model backed by the OpenAI
// Chat Completions wire format.
type ChatModelConfig struct {
	// ProviderID identifies the provider in error messages and stderr warnings
	// (for example "deepinfra", "cloudflare").
	ProviderID string

	// ModelID is the model identifier passed to the upstream API.
	ModelID string

	// BaseURL is the root URL, without the /chat/completions suffix
	// (for example "https://api.deepinfra.com/v1/openai").
	BaseURL string

	// TokenSource resolves the bearer token on each request. When nil and
	// TokenRequired is false, the Authorization header is omitted.
	TokenSource provider.TokenSource

	// TokenRequired controls whether a missing TokenSource causes an error at
	// request time. Providers like compat and ollama set this to false.
	TokenRequired bool

	// BaseURLRequired causes DoGenerate / DoStream to error out when BaseURL is
	// empty. Used by providers that refuse to build a default (compat,
	// cloudflare, fptcloud before region resolution).
	BaseURLRequired bool

	// Headers are sent on every request. Callers are expected to have merged
	// any user-provided and provider-fixed headers before setting this field.
	Headers map[string]string

	// HTTPClient overrides the default HTTP client. Nil uses http.DefaultClient.
	HTTPClient *http.Client

	// Capabilities is returned from the model's Capabilities() method.
	Capabilities provider.ModelCapabilities

	// ExtraBody merges provider-specific fields into the request body
	// (for example OpenRouter's "usage": {"include": true}).
	ExtraBody map[string]any

	// IncludeStreamOptions adds stream_options.include_usage on streaming requests.
	// Most providers want this true.
	IncludeStreamOptions bool

	// WarnPromptCaching emits a one-line stderr warning when the caller sets
	// GenerateParams.PromptCaching and the provider does not support it.
	WarnPromptCaching bool
}

// NewChatModel returns a provider.LanguageModel backed by the shared
// OpenAI-compatible codec. The returned model also implements provider.CapableModel.
func NewChatModel(cfg ChatModelConfig) provider.LanguageModel {
	return &chatModel{cfg: cfg}
}

// EmbeddingModelConfig configures an embedding model backed by the OpenAI
// /embeddings wire format.
type EmbeddingModelConfig struct {
	// ProviderID identifies the provider in error messages.
	ProviderID string

	// ModelID is the model identifier passed to the upstream API.
	ModelID string

	// BaseURL is the root URL, without the /embeddings suffix.
	BaseURL string

	// TokenSource resolves the bearer token on each request.
	TokenSource provider.TokenSource

	// TokenRequired controls whether a missing TokenSource causes an error
	// at request time.
	TokenRequired bool

	// BaseURLRequired causes DoEmbed to error out when BaseURL is empty.
	BaseURLRequired bool

	// Headers are sent on every request.
	Headers map[string]string

	// HTTPClient overrides the default HTTP client.
	HTTPClient *http.Client

	// MaxValuesPerCall is returned from the model's MaxValuesPerCall() method.
	// Defaults to 2048 when zero.
	MaxValuesPerCall int

	// ProviderOptionsKeys is the list of provider option keys forwarded to the
	// request body verbatim (for example "dimensions", "user").
	// When nil, the default set {"dimensions", "user"} is used.
	ProviderOptionsKeys []string

	// EncodingFormat is sent as the encoding_format field. Defaults to "float"
	// when empty. Set to "-" to omit the field entirely (some providers reject
	// unknown fields).
	EncodingFormat string
}

// NewEmbeddingModel returns a provider.EmbeddingModel backed by the shared
// OpenAI-compatible codec.
func NewEmbeddingModel(cfg EmbeddingModelConfig) provider.EmbeddingModel {
	return &embeddingModel{cfg: cfg}
}

// --- Internal chat model ---

type chatModel struct {
	cfg ChatModelConfig
}

// Compile-time interface assertions for the internal types.
var (
	_ provider.LanguageModel  = (*chatModel)(nil)
	_ provider.CapableModel   = (*chatModel)(nil)
	_ provider.EmbeddingModel = (*embeddingModel)(nil)
)

func (m *chatModel) ModelID() string { return m.cfg.ModelID }

func (m *chatModel) Capabilities() provider.ModelCapabilities {
	return m.cfg.Capabilities
}

func (m *chatModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	m.warnPromptCaching(params)
	if err := m.checkBaseURL(); err != nil {
		return nil, err
	}
	body := BuildRequest(params, m.cfg.ModelID, false, RequestConfig{
		ExtraBody: m.cfg.ExtraBody,
	})

	resp, err := m.doHTTP(ctx, m.cfg.BaseURL+"/chat/completions", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}
	return ParseResponse(respBody)
}

func (m *chatModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	m.warnPromptCaching(params)
	if err := m.checkBaseURL(); err != nil {
		return nil, err
	}
	body := BuildRequest(params, m.cfg.ModelID, true, RequestConfig{
		IncludeStreamOptions: m.cfg.IncludeStreamOptions,
		ExtraBody:            m.cfg.ExtraBody,
	})

	resp, err := m.doHTTP(ctx, m.cfg.BaseURL+"/chat/completions", body)
	if err != nil {
		return nil, err
	}
	return NewSSEStream(ctx, resp.Body), nil
}

func (m *chatModel) warnPromptCaching(params provider.GenerateParams) {
	if params.PromptCaching && m.cfg.WarnPromptCaching {
		fmt.Fprintf(os.Stderr, "goai: %s: WithPromptCaching is not supported and will be ignored\n", m.cfg.ProviderID)
	}
}

func (m *chatModel) checkBaseURL() error {
	if m.cfg.BaseURLRequired && m.cfg.BaseURL == "" {
		return fmt.Errorf("%s: base URL is required", m.cfg.ProviderID)
	}
	return nil
}

func (m *chatModel) doHTTP(ctx context.Context, url string, body map[string]any) (*http.Response, error) {
	token, err := resolveToken(ctx, m.cfg.TokenSource, m.cfg.TokenRequired)
	if err != nil {
		return nil, err
	}
	return httpc.DoJSONRequest(ctx, httpc.RequestConfig{
		URL:        url,
		Token:      token,
		Body:       body,
		Headers:    m.cfg.Headers,
		HTTPClient: m.cfg.HTTPClient,
		ProviderID: m.cfg.ProviderID,
	}, goai.ParseHTTPErrorWithHeaders)
}

// --- Internal embedding model ---

type embeddingModel struct {
	cfg EmbeddingModelConfig
}

func (m *embeddingModel) ModelID() string { return m.cfg.ModelID }

func (m *embeddingModel) MaxValuesPerCall() int {
	if m.cfg.MaxValuesPerCall > 0 {
		return m.cfg.MaxValuesPerCall
	}
	return 2048
}

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	if m.cfg.BaseURLRequired && m.cfg.BaseURL == "" {
		return nil, fmt.Errorf("%s: base URL is required", m.cfg.ProviderID)
	}

	body := map[string]any{
		"model": m.cfg.ModelID,
		"input": values,
	}
	if enc := m.cfg.EncodingFormat; enc != "-" {
		if enc == "" {
			enc = "float"
		}
		body["encoding_format"] = enc
	}

	keys := m.cfg.ProviderOptionsKeys
	if keys == nil {
		keys = []string{"dimensions", "user"}
	}
	if params.ProviderOptions != nil {
		for _, k := range keys {
			if v, ok := params.ProviderOptions[k]; ok {
				body[k] = v
			}
		}
	}

	token, err := resolveToken(ctx, m.cfg.TokenSource, m.cfg.TokenRequired)
	if err != nil {
		return nil, err
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", m.cfg.BaseURL+"/embeddings", jsonBody)
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	for k, v := range m.cfg.Headers {
		req.Header.Set(k, v)
	}

	client := m.cfg.HTTPClient
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
		return nil, goai.ParseHTTPErrorWithHeaders(m.cfg.ProviderID, resp.StatusCode, respBody, resp.Header)
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

// --- Helpers ---

func resolveToken(ctx context.Context, ts provider.TokenSource, required bool) (string, error) {
	if ts == nil {
		if required {
			return "", errors.New("goai: no API key or token source configured")
		}
		return "", nil
	}
	tok, err := ts.Token(ctx)
	if err != nil {
		return "", fmt.Errorf("resolving auth token: %w", err)
	}
	return tok, nil
}
