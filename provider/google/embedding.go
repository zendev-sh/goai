package google

import (
	"cmp"
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

// Embedding creates a Google Gemini embedding model for the given model ID.
// Supports text-embedding-004, embedding-001, etc.
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	// Resolve API key from env if not set.
	// Support both GOOGLE_GENERATIVE_AI_API_KEY (Vercel AI SDK convention)
	// and GEMINI_API_KEY (Google's own convention / models.dev).
	if o.tokenSource == nil {
		if key := cmp.Or(os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY"), os.Getenv("GEMINI_API_KEY")); key != "" {
			o.tokenSource = provider.StaticToken(key)
		}
	}
	// Resolve base URL from env if not overridden.
	if o.baseURL == defaultBaseURL {
		if base := os.Getenv("GOOGLE_GENERATIVE_AI_BASE_URL"); base != "" {
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

// MaxValuesPerCall returns the maximum batch size for Gemini embeddings.
func (m *embeddingModel) MaxValuesPerCall() int { return 100 }

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	// Extract Google-specific embedding options.
	var gopts map[string]any
	if g, ok := params.ProviderOptions["google"].(map[string]any); ok {
		gopts = g
	}

	// Build batch embed request.
	requests := make([]map[string]any, len(values))
	for i, v := range values {
		req := map[string]any{
			"model": "models/" + m.id,
			"content": map[string]any{
				"parts": []map[string]any{{"text": v}},
			},
		}
		// Add taskType if specified.
		if gopts != nil {
			if tt, ok := gopts["taskType"].(string); ok && tt != "" {
				req["taskType"] = tt
			}
		}
		// Add outputDimensionality if specified.
		if gopts != nil {
			if od, ok := gopts["outputDimensionality"]; ok {
				req["outputDimensionality"] = od
			}
		}
		requests[i] = req
	}

	body := map[string]any{"requests": requests}
	url := fmt.Sprintf("%s/v1beta/models/%s:batchEmbedContents", m.opts.baseURL, m.id)

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", url, jsonBody)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", token)

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
		return nil, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, respBody, resp.Header)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	var result struct {
		Embeddings []struct {
			Values []float64 `json:"values"`
		} `json:"embeddings"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	embeddings := make([][]float64, len(result.Embeddings))
	for i, e := range result.Embeddings {
		embeddings[i] = e.Values
	}

	return &provider.EmbedResult{
		Embeddings: embeddings,
		// Gemini doesn't return token usage for embeddings.
		Usage: provider.Usage{},
	}, nil
}

func (m *embeddingModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}

func (m *embeddingModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}
