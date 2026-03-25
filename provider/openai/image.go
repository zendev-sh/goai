package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance check.
var _ provider.ImageModel = (*imageModel)(nil)

// Image creates an OpenAI image model (DALL-E 3, gpt-image-1, etc.).
func Image(modelID string, opts ...Option) provider.ImageModel {
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
	return &imageModel{id: modelID, opts: o}
}

type imageModel struct {
	id   string
	opts options
}

func (m *imageModel) ModelID() string { return m.id }

func (m *imageModel) DoGenerate(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	body := map[string]any{
		"model":  m.id,
		"prompt": params.Prompt,
		"n":      params.N,
	}
	// gpt-image-1/1.5 default to b64_json; older models (dall-e-3) need it explicit.
	// Matches Vercel AI SDK's hasDefaultResponseFormat set.
	if !hasDefaultResponseFormat(m.id) {
		body["response_format"] = "b64_json"
	}
	if params.Size != "" {
		body["size"] = params.Size
	}

	// Note: params.AspectRatio is not mapped because OpenAI's image API uses
	// explicit "size" dimensions (e.g. "1024x1024") rather than aspect ratios.
	// Callers should use params.Size or pass "size" via ProviderOptions instead.

	// Item 13: provider options passthrough (quality, style, seed, etc.).
	for k, v := range params.ProviderOptions {
		body[k] = v
	}

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", m.opts.baseURL+"/images/generations", jsonBody)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

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
		return nil, goai.ParseHTTPErrorWithHeaders("openai", resp.StatusCode, respBody, resp.Header)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	var result struct {
		Data []struct {
			B64JSON       string `json:"b64_json"`
			RevisedPrompt string `json:"revised_prompt,omitempty"`
		} `json:"data"`
		OutputFormat string `json:"output_format,omitempty"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	images := make([]provider.ImageData, len(result.Data))
	imagesMeta := make([]map[string]any, len(result.Data))
	for i, d := range result.Data {
		decoded, err := base64.StdEncoding.DecodeString(d.B64JSON)
		if err != nil {
			return nil, fmt.Errorf("decoding image %d: %w", i, err)
		}

		// Item 15: detect media type from response output_format or data prefix.
		mediaType := detectMediaType(result.OutputFormat, d.B64JSON)

		images[i] = provider.ImageData{
			Data:      decoded,
			MediaType: mediaType,
		}

		// Item 14: extract revised_prompt in metadata.
		meta := map[string]any{}
		if d.RevisedPrompt != "" {
			meta["revisedPrompt"] = d.RevisedPrompt
		}
		imagesMeta[i] = meta
	}

	imgResult := &provider.ImageResult{Images: images}

	// Build provider metadata if any image has metadata.
	if slices.ContainsFunc(imagesMeta, func(m map[string]any) bool { return len(m) > 0 }) {
		imgResult.ProviderMetadata = map[string]map[string]any{
			"openai": {"images": imagesMeta},
		}
	}

	return imgResult, nil
}

// detectMediaType determines the image media type from the API response.
// Item 15: don't hardcode "image/png" -- detect from output_format or fallback.
func detectMediaType(outputFormat string, _ string) string {
	switch outputFormat {
	case "png":
		return "image/png"
	case "jpeg", "jpg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	case "gif":
		return "image/gif"
	default:
		// Default to PNG for backwards compat when format not specified.
		return "image/png"
	}
}

// hasDefaultResponseFormat returns true for models that default to b64_json
// and reject an explicit response_format parameter.
// Matches Vercel AI SDK's hasDefaultResponseFormat set.
func hasDefaultResponseFormat(modelID string) bool {
	switch modelID {
	case "gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5":
		return true
	default:
		return false
	}
}

func (m *imageModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}

func (m *imageModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}
