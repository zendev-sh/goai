package google

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
		expectedPath := "/v1beta/models/text-embedding-004:batchEmbedContents"
		if r.URL.Path != expectedPath {
			t.Errorf("expected path %s, got %s", expectedPath, r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("invalid request JSON: %v", err)
		}

		requests, ok := req["requests"].([]any)
		if !ok || len(requests) != 1 {
			t.Fatalf("expected 1 request, got %v", req["requests"])
		}
		first := requests[0].(map[string]any)
		if first["model"] != "models/text-embedding-004" {
			t.Errorf("expected model models/text-embedding-004, got %v", first["model"])
		}
		content := first["content"].(map[string]any)
		parts := content["parts"].([]any)
		part := parts[0].(map[string]any)
		if part["text"] != "hello" {
			t.Errorf("expected text 'hello', got %v", part["text"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1, 0.2, 0.3}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
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
	// Google embedding API does not return token counts; InputTokens should be 0.
	if result.Usage.InputTokens != 0 {
		t.Errorf("expected 0 input tokens, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_MultipleValues(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		requests, _ := req["requests"].([]any)
		if len(requests) != 3 {
			t.Errorf("expected 3 requests, got %d", len(requests))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1, 0.2}},
				{"values": []float64{0.3, 0.4}},
				{"values": []float64{0.5, 0.6}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
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
	if result.Usage.InputTokens != 0 {
		t.Errorf("expected 0 input tokens, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "invalid request",
				"code":    400,
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestEmbedding_MaxValuesPerCall(t *testing.T) {
	model := Embedding("text-embedding-004", WithAPIKey("test-key"))
	if got := model.MaxValuesPerCall(); got != 100 {
		t.Errorf("expected MaxValuesPerCall=100, got %d", got)
	}
}

func TestEmbedding_ModelID(t *testing.T) {
	model := Embedding("text-embedding-004", WithAPIKey("test-key"))
	if got := model.ModelID(); got != "text-embedding-004" {
		t.Errorf("expected ModelID=text-embedding-004, got %s", got)
	}
}

func TestEmbedding_APIKeyHeader(t *testing.T) {
	var gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("my-google-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gotKey != "my-google-key" {
		t.Errorf("expected x-goog-api-key 'my-google-key', got %q", gotKey)
	}
}

func TestEmbedding_BaseURL(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.HasPrefix(gotPath, "/v1beta/models/text-embedding-004") {
		t.Errorf("expected path starting with /v1beta/models/text-embedding-004, got %s", gotPath)
	}
}

func TestEmbedding_NoTokenSource(t *testing.T) {
	model := Embedding("text-embedding-004")
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error when no token source configured, got nil")
	}
	if !strings.Contains(err.Error(), "no API key or token source configured") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestEmbedding_WithHTTPClient(t *testing.T) {
	// Cover the "httpClient IS set" branch.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{{"values": []float64{0.1}}},
		})
	}))
	defer srv.Close()

	customClient := &http.Client{}
	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL), WithHTTPClient(customClient))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 1 {
		t.Errorf("expected 1 embedding, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_DefaultHTTPClient(t *testing.T) {
	// Create model without WithHTTPClient -- should use http.DefaultClient.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
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
		_, _ = w.Write([]byte(`{"embeddings":`))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
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

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
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
			"embeddings": []map[string]any{{"values": []float64{0.1}}},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004",
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
	model := Embedding("text-embedding-004",
		WithAPIKey("test-key"),
		WithBaseURL("http://127.0.0.1:1"),
	)
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error from connection refused")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("expected 'sending request' error, got: %v", err)
	}
}

// --- BF.15: Embedding ProviderOptions tests ---

func TestEmbedding_OutputDimensionality(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		requests := req["requests"].([]any)
		first := requests[0].(map[string]any)

		if first["outputDimensionality"] != float64(256) {
			t.Errorf("outputDimensionality = %v, want 256", first["outputDimensionality"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1, 0.2}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"outputDimensionality": 256,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_TaskType(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		requests := req["requests"].([]any)
		first := requests[0].(map[string]any)

		if first["taskType"] != "RETRIEVAL_QUERY" {
			t.Errorf("taskType = %v, want RETRIEVAL_QUERY", first["taskType"])
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"taskType": "RETRIEVAL_QUERY",
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_BothOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		requests := req["requests"].([]any)
		for i, r := range requests {
			req := r.(map[string]any)
			if req["taskType"] != "CLASSIFICATION" {
				t.Errorf("requests[%d].taskType = %v, want CLASSIFICATION", i, req["taskType"])
			}
			if req["outputDimensionality"] != float64(128) {
				t.Errorf("requests[%d].outputDimensionality = %v, want 128", i, req["outputDimensionality"])
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{
				{"values": []float64{0.1}},
				{"values": []float64{0.2}},
			},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello", "world"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"taskType":             "CLASSIFICATION",
				"outputDimensionality": 128,
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_NoProviderOptions(t *testing.T) {
	// Verify that requests work fine without ProviderOptions.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		requests := req["requests"].([]any)
		first := requests[0].(map[string]any)

		// Should not have taskType or outputDimensionality.
		if _, ok := first["taskType"]; ok {
			t.Error("taskType should not be set without ProviderOptions")
		}
		if _, ok := first["outputDimensionality"]; ok {
			t.Error("outputDimensionality should not be set without ProviderOptions")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": []map[string]any{{"values": []float64{0.1}}},
		})
	}))
	defer srv.Close()

	model := Embedding("text-embedding-004", WithAPIKey("test-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbedding_EnvVarResolution(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "env-key")
	m := Embedding("text-embedding-004")
	em := m.(*embeddingModel)
	if em.opts.tokenSource == nil {
		t.Error("tokenSource should be set from GOOGLE_GENERATIVE_AI_API_KEY")
	}
}
