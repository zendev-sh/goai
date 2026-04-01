package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestEmbedding_SingleValue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/embeddings" {
			t.Errorf("expected /embeddings, got %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("invalid request JSON: %v", err)
		}
		if req["model"] != "text-embedding-3-small" {
			t.Errorf("expected model text-embedding-3-small, got %v", req["model"])
		}
		if req["encoding_format"] != "float" {
			t.Errorf("expected encoding_format float, got %v", req["encoding_format"])
		}
		input, ok := req["input"].([]any)
		if !ok || len(input) != 1 || input[0] != "hello" {
			t.Errorf("expected input [\"hello\"], got %v", req["input"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{0.1, 0.2, 0.3}, "index": 0},
			},
			"usage": map[string]any{
				"prompt_tokens": 5,
				"total_tokens":  5,
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 3 {
		t.Fatalf("expected 3 dimensions, got %d", len(result.Embeddings[0]))
	}
	if result.Embeddings[0][0] != 0.1 || result.Embeddings[0][1] != 0.2 || result.Embeddings[0][2] != 0.3 {
		t.Errorf("unexpected embedding values: %v", result.Embeddings[0])
	}
	if result.Usage.InputTokens != 5 {
		t.Errorf("expected 5 input tokens, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_MultipleValues(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		input, _ := req["input"].([]any)
		if len(input) != 3 {
			t.Errorf("expected 3 inputs, got %d", len(input))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{0.1, 0.2}, "index": 0},
				{"embedding": []float64{0.3, 0.4}, "index": 1},
				{"embedding": []float64{0.5, 0.6}, "index": 2},
			},
			"usage": map[string]any{
				"prompt_tokens": 15,
				"total_tokens":  15,
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello", "world", "foo"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result.Embeddings))
	}
	if result.Embeddings[0][0] != 0.1 {
		t.Errorf("expected first embedding [0]=0.1, got %f", result.Embeddings[0][0])
	}
	if result.Embeddings[2][1] != 0.6 {
		t.Errorf("expected third embedding [1]=0.6, got %f", result.Embeddings[2][1])
	}
	if result.Usage.InputTokens != 15 {
		t.Errorf("expected 15 input tokens, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "invalid request",
				"type":    "invalid_request_error",
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestEmbedding_MaxValuesPerCall(t *testing.T) {
	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"))
	if got := model.MaxValuesPerCall(); got != 2048 {
		t.Errorf("expected MaxValuesPerCall=2048, got %d", got)
	}
}

func TestEmbedding_ModelID(t *testing.T) {
	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"))
	if got := model.ModelID(); got != "text-embedding-3-small" {
		t.Errorf("expected ModelID=text-embedding-3-small, got %s", got)
	}
}

func TestEmbedding_WithAPIKey_SetsAuthorizationHeader(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("my-secret-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gotAuth != "Bearer my-secret-key" {
		t.Errorf("expected Authorization 'Bearer my-secret-key', got %q", gotAuth)
	}
}

func TestEmbedding_WithBaseURL_OverridesURL(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gotPath != "/embeddings" {
		t.Errorf("expected path /embeddings, got %s", gotPath)
	}
}

func TestEmbedding_NoTokenSource(t *testing.T) {
	model := Embedding("text-embedding-3-small")
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error when no token source configured, got nil")
	}
	if !strings.Contains(err.Error(), "no API key or token source configured") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestEmbedding_WithHTTPClient(t *testing.T) {
	// Create model WITH WithHTTPClient -- should use the custom client.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	customClient := &http.Client{}
	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL), WithHTTPClient(customClient))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 1 {
		t.Errorf("expected 1 embedding, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_ReadBodyError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "9999")
		w.WriteHeader(http.StatusOK)
		// Write partial body then close -- causes read error due to Content-Length mismatch.
		_, _ = w.Write([]byte(`{"data":`))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error from reading truncated body, got nil")
	}
}

func TestEmbedding_UnmarshalError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`not valid json`))
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error from invalid JSON, got nil")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestEmbedding_WithHeaders(t *testing.T) {
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"X-Custom": "hello"}),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotHeader != "hello" {
		t.Errorf("expected X-Custom header 'hello', got %q", gotHeader)
	}
}

func TestEmbedding_SendRequestError(t *testing.T) {
	// Use a custom client that fails on Do.
	model := Embedding("text-embedding-3-small",
		WithAPIKey("test-key"),
		WithBaseURL("http://127.0.0.1:1"), // Connection refused.
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error from connection refused")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("expected 'sending request' error, got: %v", err)
	}
}

func TestEmbedding_EnvVarResolution(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	m := Embedding("text-embedding-3-small")
	em := m.(*embeddingModel)
	if em.opts.tokenSource == nil {
		t.Error("tokenSource should be set from OPENAI_API_KEY")
	}
}

func TestEmbedding_EnvVarBaseURL(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	t.Setenv("OPENAI_BASE_URL", "https://custom.openai.com/v1")
	m := Embedding("text-embedding-3-small")
	em := m.(*embeddingModel)
	if em.opts.baseURL != "https://custom.openai.com/v1" {
		t.Errorf("baseURL = %q", em.opts.baseURL)
	}
}

func TestEmbedding_EnvVarNotOverrideExplicit(t *testing.T) {
	t.Setenv("OPENAI_BASE_URL", "https://env.url")
	m := Embedding("text-embedding-3-small", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"))
	em := m.(*embeddingModel)
	if em.opts.baseURL != "https://explicit.url" {
		t.Errorf("baseURL = %q", em.opts.baseURL)
	}
}

func TestEmbedding_WithDimensions(t *testing.T) {
	var gotDimensions any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		gotDimensions = req["dimensions"]

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1, 0.2}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 3, "total_tokens": 3},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"dimensions": 256},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// JSON numbers unmarshal as float64; 256 == float64(256).
	if gotDimensions != float64(256) {
		t.Errorf("request body dimensions = %v (%T), want 256", gotDimensions, gotDimensions)
	}
}

func TestEmbedding_ResponseModelPopulated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "text-embedding-3-small",
			"data": []map[string]any{
				{"embedding": []float64{0.1, 0.2, 0.3}, "index": 0},
			},
			"usage": map[string]any{"prompt_tokens": 3, "total_tokens": 3},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Response.Model != "text-embedding-3-small" {
		t.Errorf("Response.Model = %q, want %q", result.Response.Model, "text-embedding-3-small")
	}
}
