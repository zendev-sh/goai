package openai

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

// Compile-time interface compliance check.
var _ provider.EmbeddingModel = (*embeddingModel)(nil)

const (
	// maxEmbeddingResponseBytes bounds the size of a successful embedding
	// response body to defend against a malicious/misbehaving server returning
	// an unbounded payload.
	maxEmbeddingResponseBytes = 64 << 20 // 64 MiB
	// maxEmbeddingErrorBytes bounds the size of an error response body read
	// solely to extract the error message.
	maxEmbeddingErrorBytes = 1 << 20 // 1 MiB
)

// Embedding creates an OpenAI embedding model for the given model ID.
// Supports text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002, etc.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve API key from env if not set.
	if o.tokenSource == nil {
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	// Resolve base URL from env if not overridden.
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("OPENAI_BASE_URL"); base != "" {
			o.baseURL = base
		}
	}
	return &embeddingModel{id: modelID, opts: o}
}

type embeddingModel struct {
	id   string
	opts options
}

func (m *embeddingModel) ModelID() string { return m.id }

// MaxValuesPerCall returns the maximum batch size for OpenAI embeddings.
func (m *embeddingModel) MaxValuesPerCall() int { return 2048 }

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	body := map[string]any{
		"model":           m.id,
		"input":           values,
		"encoding_format": "float",
	}

	// Item 16: add dimensions and user parameters from ProviderOptions.
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

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}

	// Auth is applied last so a caller-supplied header named "Authorization"
	// cannot override the real credential.
	req.Header.Set("Authorization", "Bearer "+token)

	resp, err := httpc.Do(m.httpClient(), req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxEmbeddingErrorBytes))
		return nil, goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, respBody, resp.Header)
	}

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxEmbeddingResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}
	if len(respBody) > maxEmbeddingResponseBytes {
		return nil, fmt.Errorf("openai: embedding response body exceeds %d bytes", maxEmbeddingResponseBytes)
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

	// Sort by index (usually already sorted).
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

func (m *embeddingModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("goai: no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}

func (m *embeddingModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}
