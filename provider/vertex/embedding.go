package vertex

import (
	"context"
	"encoding/json"

	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance check.
var _ provider.EmbeddingModel = (*embeddingModel)(nil)

// Embedding creates a Vertex AI embedding model for the given model ID.
//
// Uses the native Vertex AI :predict endpoint (not OpenAI-compat).
// Supports text-embedding-004, textembedding-gecko, etc.
//
// Provider options (under "vertex" key in EmbedParams.ProviderOptions):
//   - outputDimensionality (int): Reduced dimension for the output embedding.
//   - taskType (string): SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING,
//     RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION,
//     CODE_RETRIEVAL_QUERY.
//   - title (string): Document title (only valid with RETRIEVAL_DOCUMENT task).
//   - autoTruncate (bool): Truncate input text if too long (default: true).
func Embedding(modelID string, opts ...Option) provider.EmbeddingModel {
	o := resolveOpts(opts)
	return &embeddingModel{id: modelID, opts: o}
}

type embeddingModel struct {
	id   string
	opts options
}

func (m *embeddingModel) ModelID() string { return m.id }

// MaxValuesPerCall returns the maximum batch size for Vertex AI embeddings.
func (m *embeddingModel) MaxValuesPerCall() int { return 250 }

func (m *embeddingModel) DoEmbed(ctx context.Context, values []string, params provider.EmbedParams) (*provider.EmbedResult, error) {
	// Extract vertex-specific options.
	var vopts map[string]any
	if v, ok := params.ProviderOptions["vertex"].(map[string]any); ok {
		vopts = v
	}

	// Build instances array.
	instances := make([]map[string]any, len(values))
	for i, v := range values {
		inst := map[string]any{"content": v}
		if vopts != nil {
			if tt, ok := vopts["taskType"].(string); ok && tt != "" {
				inst["task_type"] = tt
			}
			if title, ok := vopts["title"].(string); ok && title != "" {
				inst["title"] = title
			}
		}
		instances[i] = inst
	}

	// Build parameters.
	parameters := map[string]any{}
	if vopts != nil {
		if od, ok := vopts["outputDimensionality"]; ok {
			parameters["outputDimensionality"] = od
		}
		if at, ok := vopts["autoTruncate"]; ok {
			parameters["autoTruncate"] = at
		}
	}

	body := map[string]any{
		"instances": instances,
	}
	if len(parameters) > 0 {
		body["parameters"] = parameters
	}

	reqURL, err := nativeURL(m.opts, fmt.Sprintf("models/%s:predict", url.PathEscape(m.id)))
	if err != nil {
		return nil, err
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", reqURL, jsonBody)
	req.Header.Set("Content-Type", "application/json")

	// Native endpoints use ?key= for API keys (already in URL), Bearer for OAuth.
	if m.opts.tokenSource != nil && !isNativeAPIKeyAuth(m.opts.tokenSource) {
		if err := setAuth(ctx, req, m.opts.tokenSource); err != nil {
			return nil, err
		}
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
		return nil, goai.ParseHTTPErrorWithHeaders("vertex", resp.StatusCode, respBody, resp.Header)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	var result struct {
		Predictions []struct {
			Embeddings struct {
				Values     []float64 `json:"values"`
				Statistics struct {
					TokenCount int `json:"token_count"`
				} `json:"statistics"`
			} `json:"embeddings"`
		} `json:"predictions"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	embeddings := make([][]float64, len(result.Predictions))
	totalTokens := 0
	for i, p := range result.Predictions {
		embeddings[i] = p.Embeddings.Values
		totalTokens += p.Embeddings.Statistics.TokenCount
	}

	return &provider.EmbedResult{
		Embeddings: embeddings,
		Usage:      provider.Usage{InputTokens: totalTokens},
	}, nil
}

func (m *embeddingModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}
