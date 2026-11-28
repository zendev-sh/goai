package google

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// Compile-time interface compliance checks.
var (
	_ provider.ImageModel = (*imagenModel)(nil)
	_ provider.ImageModel = (*geminiImageModel)(nil)
)

// Image creates a Google Generative AI image model.
//
// For Imagen models (imagen-4.0-generate-001, etc.), uses the :predict endpoint
// with instances/parameters format -- same wire format as Vertex AI but with
// Gemini API auth (x-goog-api-key).
//
// For Gemini image models (gemini-2.5-flash-image, etc.), uses the generateContent
// endpoint with responseModalities: ["IMAGE"] -- Nano Banana pattern.
//
// Provider options (under "google" key in ImageParams.ProviderOptions):
//
// Imagen models:
//   - personGeneration (string): "dont_allow", "allow_adult", "allow_all"
//   - aspectRatio (string): overrides params.AspectRatio
//
// Gemini image models:
//   - imageConfig (map): {"aspectRatio": "1:1", "imageSize": "1K"}
func Image(modelID string, opts ...Option) provider.ImageModel {
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
	if isImagenModel(modelID) {
		// Imagen models on the Gemini API are deprecated and scheduled for
		// shutdown on 2026-08-17. Keep them functional until then, but warn so
		// callers migrate to the Gemini image path (gemini-*-flash-image via
		// generateContent) ahead of the cutoff.
		fmt.Fprintf(os.Stderr,
			"goai: google: %s is deprecated on the Gemini API and will be shut down on 2026-08-17; "+
				"migrate to a Gemini image model (e.g. google.Image(\"gemini-2.5-flash-image\")) before then\n",
			modelID)
		return &imagenModel{id: modelID, opts: o}
	}
	return &geminiImageModel{id: modelID, opts: o}
}

// isImagenModel returns true for Imagen model IDs (imagen-*).
func isImagenModel(modelID string) bool {
	return strings.HasPrefix(modelID, "imagen-")
}

// --- Imagen models (via :predict endpoint) ---

type imagenModel struct {
	id   string
	opts options
}

func (m *imagenModel) ModelID() string { return m.id }

func (m *imagenModel) DoGenerate(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
	if err := validateNonChatOptions(m.opts); err != nil {
		return nil, err
	}
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	parameters := map[string]any{
		"sampleCount": params.N,
	}
	if params.AspectRatio != "" {
		parameters["aspectRatio"] = params.AspectRatio
	}

	// Extract google-specific options.
	if gopts, ok := params.ProviderOptions["google"].(map[string]any); ok {
		if v, ok := gopts["personGeneration"].(string); ok && v != "" {
			parameters["personGeneration"] = v
		}
		if v, ok := gopts["aspectRatio"].(string); ok && v != "" {
			parameters["aspectRatio"] = v
		}
	}

	body := map[string]any{
		"instances":  []map[string]any{{"prompt": params.Prompt}},
		"parameters": parameters,
	}

	reqURL := fmt.Sprintf("%s/v1beta/models/%s:predict", m.opts.baseURL, url.PathEscape(m.id))

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", reqURL, jsonBody)
	req.Header.Set("Content-Type", "application/json")

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}
	// Set the credential last so it wins over any caller-supplied header
	// named like the auth header.
	req.Header.Set("x-goog-api-key", token)

	resp, err := httpc.Do(m.httpClient(), req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxGoogleErrorBodyBytes))
		return nil, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, respBody, resp.Header)
	}

	respBody, err := readSuccessBody(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	var result struct {
		Predictions []struct {
			BytesBase64Encoded string `json:"bytesBase64Encoded"`
			MimeType           string `json:"mimeType"`
		} `json:"predictions"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	images := make([]provider.ImageData, len(result.Predictions))
	for i, p := range result.Predictions {
		decoded, err := base64.StdEncoding.DecodeString(p.BytesBase64Encoded)
		if err != nil {
			return nil, fmt.Errorf("decoding image %d: %w", i, err)
		}
		images[i] = provider.ImageData{
			Data:      decoded,
			MediaType: cmp.Or(p.MimeType, "image/png"),
		}
	}

	return &provider.ImageResult{
		Images:   images,
		Response: provider.ResponseMetadata{Model: m.id},
	}, nil
}

func (m *imagenModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("goai: no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}

func (m *imagenModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}

// --- Gemini image models (via generateContent endpoint, Nano Banana) ---

type geminiImageModel struct {
	id   string
	opts options
}

func (m *geminiImageModel) ModelID() string { return m.id }

func (m *geminiImageModel) DoGenerate(ctx context.Context, params provider.ImageParams) (*provider.ImageResult, error) {
	if err := validateNonChatOptions(m.opts); err != nil {
		return nil, err
	}
	token, err := m.resolveToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolving auth token: %w", err)
	}

	genConfig := map[string]any{
		"responseModalities": []string{"TEXT", "IMAGE"},
	}

	// Map params.N to numberOfImages in the generation config.
	// Gemini's generateContent supports numberOfImages (top-level GenerationConfig
	// field) to request multiple images.
	if params.N > 1 {
		genConfig["numberOfImages"] = params.N
	}

	// Note: params.Size is not supported by the Gemini generateContent endpoint.
	// Use provider options {"google": {"imageSize": "1K"}} instead (top-level
	// GenerationConfig field, not nested under imageConfig).

	// Build imageConfig from params and provider options.
	imageConfig := map[string]any{}

	// Map params.AspectRatio into imageConfig if set.
	if params.AspectRatio != "" {
		imageConfig["aspectRatio"] = params.AspectRatio
	}

	// Extract google-specific options for image config (overrides params).
	// imageSize is not an official ImageConfig field -- it belongs at the
	// top level of generationConfig, so it is hoisted out of imageConfig.
	if gopts, ok := params.ProviderOptions["google"].(map[string]any); ok {
		if ic, ok := gopts["imageConfig"].(map[string]any); ok {
			for k, v := range ic {
				if k == "imageSize" {
					genConfig["imageSize"] = v
					continue
				}
				imageConfig[k] = v
			}
		}
		// Support a top-level imageSize provider option as well.
		if sz, ok := gopts["imageSize"].(string); ok && sz != "" {
			genConfig["imageSize"] = sz
		}
	}

	if len(imageConfig) > 0 {
		genConfig["imageConfig"] = imageConfig
	}

	body := map[string]any{
		"contents": []map[string]any{
			{
				"role": "user",
				"parts": []map[string]any{
					{"text": params.Prompt},
				},
			},
		},
		"generationConfig": genConfig,
	}

	reqURL := fmt.Sprintf("%s/v1beta/models/%s:generateContent", m.opts.baseURL, url.PathEscape(m.id))

	jsonBody := httpc.MustMarshalJSON(body)
	req := httpc.MustNewRequest(ctx, "POST", reqURL, jsonBody)
	req.Header.Set("Content-Type", "application/json")

	for k, v := range m.opts.headers {
		req.Header.Set(k, v)
	}
	// Set the credential last so it wins over any caller-supplied header
	// named like the auth header.
	req.Header.Set("x-goog-api-key", token)

	resp, err := httpc.Do(m.httpClient(), req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, maxGoogleErrorBodyBytes))
		return nil, goai.ParseHTTPErrorWithHeaders("google", resp.StatusCode, respBody, resp.Header)
	}

	respBody, err := readSuccessBody(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	var result struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text       string `json:"text,omitempty"`
					InlineData *struct {
						MimeType string `json:"mimeType"`
						Data     string `json:"data"`
					} `json:"inlineData,omitempty"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	var images []provider.ImageData
	for _, candidate := range result.Candidates {
		for _, part := range candidate.Content.Parts {
			if part.InlineData != nil && part.InlineData.Data != "" {
				decoded, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
				if err != nil {
					return nil, fmt.Errorf("decoding inline image: %w", err)
				}
				mediaType := cmp.Or(part.InlineData.MimeType, "image/png")
				images = append(images, provider.ImageData{
					Data:      decoded,
					MediaType: mediaType,
				})
			}
		}
	}

	if len(images) == 0 {
		return nil, fmt.Errorf("no image data in response")
	}

	return &provider.ImageResult{
		Images:   images,
		Response: provider.ResponseMetadata{Model: m.id},
	}, nil
}

func (m *geminiImageModel) resolveToken(ctx context.Context) (string, error) {
	if m.opts.tokenSource == nil {
		return "", errors.New("goai: no API key or token source configured")
	}
	return m.opts.tokenSource.Token(ctx)
}

func (m *geminiImageModel) httpClient() *http.Client {
	if m.opts.httpClient != nil {
		return m.opts.httpClient
	}
	return http.DefaultClient
}
