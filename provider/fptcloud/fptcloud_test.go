package fptcloud

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestChat_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"Qwen2.5-Coder-32B-Instruct","choices":[{"message":{"role":"assistant","content":"xin chào"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`)
	}))
	defer server.Close()

	m := Chat("Qwen2.5-Coder-32B-Instruct", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "xin chào" {
		t.Errorf("Text = %q", result.Text)
	}
}

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := m.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
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
	if len(texts) != 1 || texts[0] != "Hi" {
		t.Errorf("texts = %v", texts)
	}
}

func TestRegionDefault(t *testing.T) {
	m := Chat("m", WithAPIKey("k"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != baseURLGlobal {
		t.Errorf("baseURL = %q, want %q", cm.opts.baseURL, baseURLGlobal)
	}
}

func TestRegionJP(t *testing.T) {
	m := Chat("m", WithAPIKey("k"), WithRegion("jp"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != baseURLJP {
		t.Errorf("baseURL = %q, want %q", cm.opts.baseURL, baseURLJP)
	}
}

func TestRegionUnknownFallback(t *testing.T) {
	m := Chat("m", WithAPIKey("k"), WithRegion("mars"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != baseURLGlobal {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"rate limited"}}`)
	}))
	defer server.Close()

	m := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNoTokenSource(t *testing.T) {
	t.Setenv("FPT_API_KEY", "")
	m := Chat("m", WithBaseURL("http://x"))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "v" {
			t.Error("missing header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	m := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "v"}))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer dyn" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	m := Chat("m", WithTokenSource(provider.StaticToken("dyn")), WithBaseURL(server.URL))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithHTTPClient(t *testing.T) {
	c := &http.Client{}
	m := Chat("m", WithAPIKey("k"), WithHTTPClient(c))
	if m.(*chatModel).opts.httpClient != c {
		t.Error("client not set")
	}
}

func TestCapabilities(t *testing.T) {
	m := Chat("m", WithAPIKey("k"))
	caps := provider.ModelCapabilitiesOf(m)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	m := Chat("Qwen2.5-Coder-32B-Instruct", WithAPIKey("k"))
	if m.ModelID() != "Qwen2.5-Coder-32B-Instruct" {
		t.Errorf("ModelID = %q", m.ModelID())
	}
}

func TestEnvVarResolution(t *testing.T) {
	t.Setenv("FPT_API_KEY", "env-key")
	t.Setenv("FPT_REGION", "jp")
	m := Chat("m")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource nil")
	}
	if cm.opts.baseURL != baseURLJP {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestEnvVarBaseURL(t *testing.T) {
	t.Setenv("FPT_API_KEY", "k")
	t.Setenv("FPT_BASE_URL", "https://custom.example.com/v1")
	m := Chat("m")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://custom.example.com/v1" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestEnvVarNotOverrideExplicit(t *testing.T) {
	t.Setenv("FPT_API_KEY", "env")
	t.Setenv("FPT_BASE_URL", "https://env.example.com")
	m := Chat("m", WithAPIKey("explicit"), WithBaseURL("https://explicit.example.com"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://explicit.example.com" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestConnectionError(t *testing.T) {
	m := Chat("m", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected: %s", err)
	}
}

func TestPromptCachingIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	m := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestToolCallStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		tools, _ := req["tools"].([]any)
		if len(tools) != 1 {
			t.Errorf("tools count = %d", len(tools))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := m.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools:    []provider.ToolDefinition{{Name: "f", Description: "x", InputSchema: []byte(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var got bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			got = true
		}
	}
	if !got {
		t.Error("no tool call")
	}
}

// --- Embeddings ---

func TestEmbed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"m","data":[{"embedding":[0.1,0.2],"index":0}],"usage":{"prompt_tokens":2,"total_tokens":2}}`)
	}))
	defer server.Close()

	m := Embedding("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 || result.Embeddings[0][0] != 0.1 {
		t.Errorf("embeddings = %v", result.Embeddings)
	}
	if result.Usage.InputTokens != 2 {
		t.Errorf("InputTokens = %d", result.Usage.InputTokens)
	}
}

func TestEmbed_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = fmt.Fprint(w, `{"error":"unauthorized"}`)
	}))
	defer server.Close()

	m := Embedding("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbed_NoToken(t *testing.T) {
	t.Setenv("FPT_API_KEY", "")
	m := Embedding("m", WithBaseURL("http://x"))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbed_MaxValuesPerCall(t *testing.T) {
	m := Embedding("m", WithAPIKey("k"))
	if m.MaxValuesPerCall() != 2048 {
		t.Errorf("MaxValuesPerCall = %d", m.MaxValuesPerCall())
	}
}

func TestEmbed_ModelID(t *testing.T) {
	m := Embedding("embed-v1", WithAPIKey("k"))
	if m.ModelID() != "embed-v1" {
		t.Errorf("ModelID = %q", m.ModelID())
	}
}

func TestEmbed_Dimensions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["dimensions"] != float64(512) {
			t.Errorf("dimensions = %v", req["dimensions"])
		}
		if req["user"] != "u1" {
			t.Errorf("user = %v", req["user"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"m","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)
	}))
	defer server.Close()

	m := Embedding("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"dimensions": 512, "user": "u1"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbed_WithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "v" {
			t.Error("missing header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"m","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)
	}))
	defer server.Close()

	m := Embedding("m", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "v"}))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbed_ConnectionError(t *testing.T) {
	m := Embedding("m", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected: %s", err)
	}
}

func TestEmbed_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"not-valid-json`)
	}))
	defer server.Close()

	m := Embedding("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil || !strings.Contains(err.Error(), "parsing response") {
		t.Fatalf("expected parsing error, got %v", err)
	}
}

func TestStream_PromptCachingIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := m.DoStream(t.Context(), provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
}

func TestStream_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, `{"error":"oops"}`)
	}))
	defer server.Close()

	m := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := m.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}
