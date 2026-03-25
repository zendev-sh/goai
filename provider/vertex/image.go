package vertex

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"

	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance check.
var _ provider.ImageModel = (*imageModel)(nil)

// Image creates a Vertex AI image model for the given model ID.
//
// Uses the native Vertex AI :predict endpoint (not OpenAI-compat).
// Supports Imagen 3/4 models (imagen-3.0-generate-002, etc.).
//
// Provider options (under "vertex" key in ImageParams.ProviderOptions):
//   - negativePrompt (string): Text describing what to avoid.
//   - personGeneration (string): "dont_allow", "allow_adult", "allow_all".
//   - safetySetting (string): "block_low_and_above", "block_medium_and_above",
//     "block_only_high", "block_none".
//   - addWatermark (bool): Whether to add a watermark.
//   - sampleImageSize (string): "1K" or "2K".
func Image(modelID string, opts ...Option) provider.ImageModel {
	o := resolveOpts(opts)
	return &imageModel{id: modelID, opts: o}
}

type imageModel struct {
	id   string
	opts options
}

func (m *imageModel) ModelID() string { return m.id }

func (m *imageModel) DoGenerate(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
	// Build parameters from standard params.
	parameters := map[string]any{
		"sampleCount": params.N,
	}
	if params.AspectRatio != "" {
		parameters["aspectRatio"] = params.AspectRatio
	}

	// Extract vertex-specific options.
	if vopts, ok := params.ProviderOptions["vertex"].(map[string]any); ok {
		if v, ok := vopts["negativePrompt"].(string); ok && v != "" {
			parameters["negativePrompt"] = v
		}
		if v, ok := vopts["personGeneration"].(string); ok && v != "" {
			parameters["personGeneration"] = v
		}
		if v, ok := vopts["safetySetting"].(string); ok && v != "" {
			parameters["safetySetting"] = v
		}
		if v, ok := vopts["addWatermark"]; ok {
			parameters["addWatermark"] = v
		}
		if v, ok := vopts["sampleImageSize"].(string); ok && v != "" {
			parameters["sampleImageSize"] = v
		}
		if v, ok := vopts["seed"]; ok {
			parameters["seed"] = v
		}
	}

	body := map[string]any{
		"instances": []map[string]any{
			{"prompt": params.Prompt},
		},
		"parameters": parameters,
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
			BytesBase64Encoded string `json:"bytesBase64Encoded"`
			MimeType           string `json:"mimeType"`
			Prompt             string `json:"prompt,omitempty"`
		} `json:"predictions"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	images := make([]provider.ImageData, len(result.Predictions))
	imagesMeta := make([]map[string]any, len(result.Predictions))
	for i, p := range result.Predictions {
		decoded, err := base64.StdEncoding.DecodeString(p.BytesBase64Encoded)
		if err != nil {
			return nil, fmt.Errorf("decoding image %d: %w", i, err)
		}
		mediaType := cmp.Or(p.MimeType, "image/png")
		images[i] = provider.ImageData{
			Data:      decoded,
			MediaType: mediaType,
		}
		meta := map[string]any{}
		if p.Prompt != "" {
			meta["revisedPrompt"] = p.Prompt
		}
		imagesMeta[i] = meta
	}

	imgResult := &provider.ImageResult{Images: images}

	// Build provider metadata if any image has metadata.
	if slices.ContainsFunc(imagesMeta, func(m map[string]any) bool { return len(m) > 0 }) {
		imgResult.ProviderMetadata = map[string]map[string]any{
			"vertex": {"images": imagesMeta},
		}
	}

	return imgResult, nil
}

func (m *imageModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}
