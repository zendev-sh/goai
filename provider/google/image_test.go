package google

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// --- Imagen model tests ---

func TestImagenModel_Generate(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("fake-png"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/models/imagen-4.0-generate-001:predict") {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("api key = %q", r.Header.Get("x-goog-api-key"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		instances := req["instances"].([]any)
		inst := instances[0].(map[string]any)
		if inst["prompt"] != "a cat" {
			t.Errorf("prompt = %v", inst["prompt"])
		}

		params := req["parameters"].(map[string]any)
		if params["sampleCount"] != float64(2) {
			t.Errorf("sampleCount = %v", params["sampleCount"])
		}
		if params["aspectRatio"] != "16:9" {
			t.Errorf("aspectRatio = %v", params["aspectRatio"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"bytesBase64Encoded": imgData},
				{"bytesBase64Encoded": imgData},
			},
		})
	}))
	defer srv.Close()

	model := Image("imagen-4.0-generate-001", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt:      "a cat",
		N:           2,
		AspectRatio: "16:9",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Images) != 2 {
		t.Fatalf("images = %d, want 2", len(result.Images))
	}
	if string(result.Images[0].Data) != "fake-png" {
		t.Errorf("data = %q", string(result.Images[0].Data))
	}
	if result.Images[0].MediaType != "image/png" {
		t.Errorf("mediaType = %s", result.Images[0].MediaType)
	}
}

func TestImagenModel_ProviderOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		params := req["parameters"].(map[string]any)
		if params["personGeneration"] != "allow_adult" {
			t.Errorf("personGeneration = %v", params["personGeneration"])
		}
		// Provider option aspectRatio should override params.AspectRatio.
		if params["aspectRatio"] != "9:16" {
			t.Errorf("aspectRatio = %v, want 9:16", params["aspectRatio"])
		}

		imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{{"bytesBase64Encoded": imgData}},
		})
	}))
	defer srv.Close()

	model := Image("imagen-4.0-generate-001", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt:      "test",
		N:           1,
		AspectRatio: "1:1", // overridden by provider option
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"personGeneration": "allow_adult",
				"aspectRatio":      "9:16",
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImagenModel_CustomHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Errorf("custom header missing")
		}
		imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{{"bytesBase64Encoded": imgData}},
		})
	}))
	defer srv.Close()

	model := Image("imagen-4.0-fast-generate-001",
		WithAPIKey("k"),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"X-Custom": "val"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestImagenModel_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer srv.Close()

	model := Image("imagen-4.0-generate-001", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "429") && !strings.Contains(err.Error(), "rate") {
		t.Errorf("error = %v", err)
	}
}

func TestImagenModel_NoToken(t *testing.T) {
	model := Image("imagen-4.0-generate-001") // no API key
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error for missing token")
	}
}

func TestImagenModel_InvalidBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{{"bytesBase64Encoded": "!!!invalid!!!"}},
		})
	}))
	defer srv.Close()

	model := Image("imagen-4.0-generate-001", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected decode error")
	}
	if !strings.Contains(err.Error(), "decoding") {
		t.Errorf("error = %v", err)
	}
}

func TestImagenModel_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{bad json`))
	}))
	defer srv.Close()

	model := Image("imagen-4.0-generate-001", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected parse error")
	}
}

func TestImagenModel_ReadBodyError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", "100")
		_, _ = w.Write([]byte(`{`)) // truncated
	}))
	defer srv.Close()

	model := Image("imagen-4.0-generate-001", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestImagenModel_CustomHTTPClient(t *testing.T) {
	called := false
	client := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			called = true
			imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
			body := fmt.Sprintf(`{"predictions":[{"bytesBase64Encoded":"%s"}]}`, imgData)
			return &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(strings.NewReader(body)),
				Header:     http.Header{"Content-Type": []string{"application/json"}},
			}, nil
		}),
	}

	model := Image("imagen-4.0-generate-001", WithAPIKey("k"), WithHTTPClient(client))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("custom http client not used")
	}
}

func TestImagenModel_NetworkError(t *testing.T) {
	client := &http.Client{
		Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			return nil, fmt.Errorf("connection refused")
		}),
	}

	model := Image("imagen-4.0-generate-001", WithAPIKey("k"), WithHTTPClient(client))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("error = %v", err)
	}
}

// --- Gemini image model tests (Nano Banana) ---

func TestGeminiImage_Generate(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("banana-png"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("method = %s", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/models/gemini-2.5-flash-image:generateContent") {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("api key = %q", r.Header.Get("x-goog-api-key"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Verify contents structure.
		contents := req["contents"].([]any)
		content := contents[0].(map[string]any)
		if content["role"] != "user" {
			t.Errorf("role = %v", content["role"])
		}
		parts := content["parts"].([]any)
		part := parts[0].(map[string]any)
		if part["text"] != "draw a banana" {
			t.Errorf("text = %v", part["text"])
		}

		// Verify generationConfig has responseModalities.
		gc := req["generationConfig"].(map[string]any)
		rm := gc["responseModalities"].([]any)
		if len(rm) != 2 || rm[0] != "TEXT" || rm[1] != "IMAGE" {
			t.Errorf("responseModalities = %v", rm)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"parts": []map[string]any{
							{
								"inlineData": map[string]any{
									"mimeType": "image/png",
									"data":     imgData,
								},
							},
							{"text": "here is a banana"},
						},
					},
				},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "draw a banana",
		N:      1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Images) != 1 {
		t.Fatalf("images = %d, want 1", len(result.Images))
	}
	if string(result.Images[0].Data) != "banana-png" {
		t.Errorf("data = %q", string(result.Images[0].Data))
	}
	if result.Images[0].MediaType != "image/png" {
		t.Errorf("mediaType = %s", result.Images[0].MediaType)
	}
}

func TestGeminiImage_MultipleImages(t *testing.T) {
	img1 := base64.StdEncoding.EncodeToString([]byte("img1"))
	img2 := base64.StdEncoding.EncodeToString([]byte("img2"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"parts": []map[string]any{
							{"inlineData": map[string]any{"mimeType": "image/jpeg", "data": img1}},
							{"inlineData": map[string]any{"mimeType": "image/webp", "data": img2}},
						},
					},
				},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Images) != 2 {
		t.Fatalf("images = %d, want 2", len(result.Images))
	}
	if result.Images[0].MediaType != "image/jpeg" {
		t.Errorf("img0 mediaType = %s", result.Images[0].MediaType)
	}
	if result.Images[1].MediaType != "image/webp" {
		t.Errorf("img1 mediaType = %s", result.Images[1].MediaType)
	}
}

func TestGeminiImage_DefaultMimeType(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("data"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"parts": []map[string]any{
							{"inlineData": map[string]any{"mimeType": "", "data": imgData}},
						},
					},
				},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Images[0].MediaType != "image/png" {
		t.Errorf("default mediaType = %s, want image/png", result.Images[0].MediaType)
	}
}

func TestGeminiImage_ImageConfig(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		gc := req["generationConfig"].(map[string]any)
		ic := gc["imageConfig"].(map[string]any)
		if ic["aspectRatio"] != "16:9" {
			t.Errorf("imageConfig.aspectRatio = %v", ic["aspectRatio"])
		}

		imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{
					{"inlineData": map[string]any{"mimeType": "image/png", "data": imgData}},
				}}},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "x",
		N:      1,
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"imageConfig": map[string]any{"aspectRatio": "16:9"},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGeminiImage_NoImageInResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{
					{"text": "I cannot generate images"},
				}}},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error for no image data")
	}
	if !strings.Contains(err.Error(), "no image data") {
		t.Errorf("error = %v", err)
	}
}

func TestGeminiImage_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":{"message":"forbidden"}}`))
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestGeminiImage_NoToken(t *testing.T) {
	model := Image("gemini-2.5-flash-image")
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error for missing token")
	}
}

func TestGeminiImage_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`not json`))
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected parse error")
	}
}

func TestGeminiImage_InvalidBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{
					{"inlineData": map[string]any{"mimeType": "image/png", "data": "!!!bad!!!"}},
				}}},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected decode error")
	}
}

func TestGeminiImage_NetworkError(t *testing.T) {
	client := &http.Client{
		Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			return nil, fmt.Errorf("network down")
		}),
	}

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithHTTPClient(client))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("error = %v", err)
	}
}

func TestGeminiImage_CustomHTTPClient(t *testing.T) {
	called := false
	imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
	client := &http.Client{
		Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			called = true
			body := fmt.Sprintf(`{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"%s"}}]}}]}`, imgData)
			return &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(strings.NewReader(body)),
				Header:     http.Header{"Content-Type": []string{"application/json"}},
			}, nil
		}),
	}

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithHTTPClient(client))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("custom http client not used")
	}
}

func TestGeminiImage_CustomHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Test") != "yes" {
			t.Errorf("custom header missing")
		}
		imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{
					{"inlineData": map[string]any{"mimeType": "image/png", "data": imgData}},
				}}},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image",
		WithAPIKey("k"),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"X-Test": "yes"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGeminiImage_ReadBodyError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", "100")
		_, _ = w.Write([]byte(`{`))
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- isImagenModel tests ---

func TestIsImagenModel(t *testing.T) {
	tests := []struct {
		id   string
		want bool
	}{
		{"imagen-4.0-generate-001", true},
		{"imagen-4.0-ultra-generate-001", true},
		{"imagen-3.0-generate-002", true},
		{"gemini-2.5-flash-image", false},
		{"gemini-3-pro-image-preview", false},
		{"gpt-4o", false},
	}
	for _, tt := range tests {
		if got := isImagenModel(tt.id); got != tt.want {
			t.Errorf("isImagenModel(%q) = %v, want %v", tt.id, got, tt.want)
		}
	}
}

// --- Image factory routing ---

func TestImage_RoutesToImagen(t *testing.T) {
	model := Image("imagen-4.0-generate-001", WithAPIKey("k"))
	if _, ok := model.(*imagenModel); !ok {
		t.Errorf("expected *imagenModel, got %T", model)
	}
	if model.ModelID() != "imagen-4.0-generate-001" {
		t.Errorf("modelID = %s", model.ModelID())
	}
}

func TestImage_RoutesToGemini(t *testing.T) {
	model := Image("gemini-2.5-flash-image", WithAPIKey("k"))
	if _, ok := model.(*geminiImageModel); !ok {
		t.Errorf("expected *geminiImageModel, got %T", model)
	}
	if model.ModelID() != "gemini-2.5-flash-image" {
		t.Errorf("modelID = %s", model.ModelID())
	}
}

// roundTripFunc is a helper for custom http.RoundTripper in tests.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestImagenModel_TokenSourceError(t *testing.T) {
	model := &imagenModel{
		id: "imagen-4.0-generate-001",
		opts: options{
			tokenSource: failingTokenSource{},
		},
	}
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected token error")
	}
}

func TestGeminiImage_TokenSourceError(t *testing.T) {
	model := &geminiImageModel{
		id: "gemini-2.5-flash-image",
		opts: options{
			tokenSource: failingTokenSource{},
		},
	}
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected token error")
	}
}

type failingTokenSource struct{}

func (failingTokenSource) Token(context.Context) (string, error) {
	return "", fmt.Errorf("token fetch failed")
}

func TestImagenModel_EmptyProviderOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		params := req["parameters"].(map[string]any)
		// Should not have personGeneration when not set.
		if _, ok := params["personGeneration"]; ok {
			t.Error("personGeneration should not be set")
		}

		imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{{"bytesBase64Encoded": imgData}},
		})
	}))
	defer srv.Close()

	model := Image("imagen-4.0-generate-001", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt:          "x",
		N:               1,
		ProviderOptions: map[string]any{"google": map[string]any{}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGeminiImage_EmptyProviderOptions(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		gc := req["generationConfig"].(map[string]any)
		if _, ok := gc["imageConfig"]; ok {
			t.Error("imageConfig should not be set when not provided")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{
					{"inlineData": map[string]any{"mimeType": "image/png", "data": imgData}},
				}}},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt:          "x",
		N:               1,
		ProviderOptions: map[string]any{"google": map[string]any{}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Test with nil InlineData (part has no image).
func TestGeminiImage_NilInlineData(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("ok"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{
					{"text": "describing image"},
					{"inlineData": map[string]any{"mimeType": "image/png", "data": imgData}},
				}}},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should only have 1 image (text parts skipped).
	if len(result.Images) != 1 {
		t.Errorf("images = %d, want 1", len(result.Images))
	}
}

// Test InlineData with empty data string is skipped.
func TestGeminiImage_EmptyInlineData(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("real"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{
					{"inlineData": map[string]any{"mimeType": "image/png", "data": ""}},
					{"inlineData": map[string]any{"mimeType": "image/png", "data": imgData}},
				}}},
			},
		})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Images) != 1 {
		t.Errorf("images = %d, want 1 (empty data skipped)", len(result.Images))
	}
}

// Test empty candidates.
func TestGeminiImage_EmptyCandidates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"candidates": []map[string]any{}})
	}))
	defer srv.Close()

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithBaseURL(srv.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error for empty candidates")
	}
}

// Ensure ReadBodyError works for Gemini too.
func TestGeminiImage_ReadBodyErrorTransport(t *testing.T) {
	client := &http.Client{
		Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(bytes.NewReader(nil)),
				Header:     http.Header{},
			}, nil
		}),
	}

	model := Image("gemini-2.5-flash-image", WithAPIKey("k"), WithHTTPClient(client))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "x", N: 1})
	if err == nil {
		t.Fatal("expected error for empty body")
	}
}
