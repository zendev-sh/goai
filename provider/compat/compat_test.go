package compat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// failingTokenSource is a TokenSource that always returns an error.
type failingTokenSource struct{}

func (f failingTokenSource) Token(_ context.Context) (string, error) {
	return "", errors.New("token fetch failed")
}

// --- Chat Tests ---

func TestChat_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-123","model":"my-model","choices":[{"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
	}))
	defer server.Close()

	model := Chat("my-model", WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want 'Hello world'", result.Text)
	}
}

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("my-model", WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "Hello" {
		t.Errorf("texts = %v, want [Hello]", texts)
	}
}

func TestChat_NoBaseURL(t *testing.T) {
	model := Chat("model")
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error when no base URL")
	}
	if !strings.Contains(err.Error(), "base URL is required") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestChat_Stream_NoBaseURL(t *testing.T) {
	model := Chat("model")
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error when no base URL")
	}
	if !strings.Contains(err.Error(), "base URL is required") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestChat_NoAuthSkipsHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("expected no Authorization header, got %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_WithAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_WithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer dynamic" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithTokenSource(provider.StaticToken("dynamic")), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("model", WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChat_ConnectionError(t *testing.T) {
	model := Chat("m", WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestChat_ReadError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "100")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"id"`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChat_WithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Chat("model", WithHTTPClient(c))
	cm := model.(*chatModel)
	if cm.opts.httpClient != c {
		t.Error("custom client not set")
	}
}

func TestChat_WithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_PerRequestHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Per-Request") != "val" {
			t.Error("missing per-request header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{"X-Per-Request": "val"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestChat_ToolCallStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		tools, _ := req["tools"].([]any)
		if len(tools) != 1 {
			t.Errorf("tools count = %d, want 1", len(tools))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\"\"}}]},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\": \\\"Paris\\\"}\"}}]},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("my-model", WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather in Paris"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "Get weather", InputSchema: []byte(`{"type":"object","properties":{"city":{"type":"string"}}}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotToolCall bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			gotToolCall = true
			if chunk.ToolName != "get_weather" {
				t.Errorf("ToolName = %q", chunk.ToolName)
			}
		}
	}
	if !gotToolCall {
		t.Error("no tool call chunk received")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m", WithBaseURL("http://example.com"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("my-model", WithBaseURL("http://example.com"))
	if model.ModelID() != "my-model" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

// --- Embedding Tests ---

func TestEmbedding_SingleValue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("expected /embeddings, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{0.1, 0.2, 0.3}, "index": 0},
			},
			"usage": map[string]any{"prompt_tokens": 5, "total_tokens": 5},
		})
	}))
	defer srv.Close()

	model := Embedding("my-embed", WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Embeddings))
	}
	if result.Embeddings[0][0] != 0.1 {
		t.Errorf("unexpected embedding: %v", result.Embeddings[0])
	}
	if result.Usage.InputTokens != 5 {
		t.Errorf("expected 5 input tokens, got %d", result.Usage.InputTokens)
	}
}

func TestEmbedding_MultipleValues(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float64{0.1, 0.2}, "index": 0},
				{"embedding": []float64{0.3, 0.4}, "index": 1},
			},
			"usage": map[string]any{"prompt_tokens": 10, "total_tokens": 10},
		})
	}))
	defer srv.Close()

	model := Embedding("my-embed", WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello", "world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(result.Embeddings))
	}
}

func TestEmbedding_NoBaseURL(t *testing.T) {
	model := Embedding("model")
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error when no base URL")
	}
	if !strings.Contains(err.Error(), "base URL is required") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_NoAuthSkipsHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("expected no Authorization header, got %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	model := Embedding("m", WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_WithAPIKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer my-key" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	model := Embedding("m", WithAPIKey("my-key"), WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"message":"bad request"}}`)
	}))
	defer srv.Close()

	model := Embedding("m", WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbedding_ConnectionError(t *testing.T) {
	model := Embedding("m", WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestEmbedding_ReadBodyError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "9999")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data":`))
	}))
	defer srv.Close()

	model := Embedding("m", WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbedding_UnmarshalError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`not valid json`))
	}))
	defer srv.Close()

	model := Embedding("m", WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "parsing response") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestEmbedding_WithHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	model := Embedding("m", WithBaseURL(srv.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_WithHTTPClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	c := &http.Client{}
	model := Embedding("m", WithBaseURL(srv.URL), WithHTTPClient(c))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_MaxValuesPerCall(t *testing.T) {
	model := Embedding("m", WithBaseURL("http://example.com"))
	if got := model.MaxValuesPerCall(); got != 2048 {
		t.Errorf("MaxValuesPerCall = %d, want 2048", got)
	}
}

func TestEmbedding_ModelID(t *testing.T) {
	model := Embedding("my-embed", WithBaseURL("http://example.com"))
	if model.ModelID() != "my-embed" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestChat_TokenSourceError(t *testing.T) {
	// doHTTP should return error when token source fails.
	model := Chat("m", WithBaseURL("http://example.com"), WithTokenSource(failingTokenSource{}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_TokenSourceError(t *testing.T) {
	// DoEmbed should return error when token source fails.
	model := Embedding("m", WithBaseURL("http://example.com"), WithTokenSource(failingTokenSource{}))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestEmbedding_ProviderOptionsUser(t *testing.T) {
	// Verify the "user" provider option is passed in the request body.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["user"] != "test-user" {
			t.Errorf("expected user=test-user, got %v", req["user"])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	model := Embedding("m", WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"user": "test-user"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedding_ProviderOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["dimensions"] != float64(256) {
			t.Errorf("expected dimensions=256, got %v", req["dimensions"])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer srv.Close()

	model := Embedding("m", WithBaseURL(srv.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"dimensions": 256},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestEmbedding_ResponseModelPopulated verifies that EmbedResult.Response.Model
// is populated with the model ID passed to Embedding().
func TestEmbedding_ResponseModelPopulated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1, 0.2}, "index": 0}},
			"model": "my-embed-model",
			"usage": map[string]any{"prompt_tokens": 3, "total_tokens": 3},
		})
	}))
	defer srv.Close()

	model := Embedding("my-embed-model", WithBaseURL(srv.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Response.Model != "my-embed-model" {
		t.Errorf("Response.Model = %q, want %q", result.Response.Model, "my-embed-model")
	}
}

// TestChat_PromptCachingIgnored verifies that passing PromptCaching=true to the compat
// provider succeeds (warning is written to stderr, not returned as error).
func TestChat_PromptCachingIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-test","model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	}))
	defer server.Close()

	model := Chat("my-model", WithBaseURL(server.URL))

	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatalf("DoGenerate unexpected error: %v", err)
	}
	if result.Text != "ok" {
		t.Errorf("DoGenerate Text = %q, want ok", result.Text)
	}

	streamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer streamServer.Close()

	streamModel := Chat("my-model", WithBaseURL(streamServer.URL))

	streamResult, err := streamModel.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatalf("DoStream unexpected error: %v", err)
	}
	var texts []string
	for chunk := range streamResult.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
	}
	if len(texts) != 1 || texts[0] != "ok" {
		t.Errorf("DoStream texts = %v, want [ok]", texts)
	}
}
