package cloudflare

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
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"@cf/meta/llama-3.1-8b-instruct","choices":[{"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
	}))
	defer server.Close()

	model := Chat("@cf/meta/llama-3.1-8b-instruct", WithAPIKey("test-token"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello world" {
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

	model := Chat("@cf/meta/llama-3.1-8b-instruct", WithAPIKey("k"), WithBaseURL(server.URL))
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
	if len(texts) != 1 || texts[0] != "Hi" {
		t.Errorf("texts = %v", texts)
	}
}

func TestChat_MissingAccountID(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_BASE_URL", "")
	model := Chat("m", WithAPIKey("k"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil || !strings.Contains(err.Error(), "account ID") {
		t.Fatalf("expected account ID error, got: %v", err)
	}

	_, err = model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil || !strings.Contains(err.Error(), "account ID") {
		t.Fatalf("expected account ID error, got: %v", err)
	}
}

func TestChat_AccountIDBuildsURL(t *testing.T) {
	m := Chat("m", WithAPIKey("k"), WithAccountID("acc123"))
	cm := m.(*chatModel)
	want := "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1"
	if cm.opts.baseURL != want {
		t.Errorf("baseURL = %q, want %q", cm.opts.baseURL, want)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNoTokenSource(t *testing.T) {
	t.Setenv("CLOUDFLARE_API_TOKEN", "")
	model := Chat("m", WithBaseURL("http://x"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
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

	model := Chat("m", WithTokenSource(provider.StaticToken("dyn")), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithHTTPClient(t *testing.T) {
	c := &http.Client{}
	m := Chat("m", WithAPIKey("k"), WithHTTPClient(c), WithAccountID("a"))
	cm := m.(*chatModel)
	if cm.opts.httpClient != c {
		t.Error("client not set")
	}
}

func TestCapabilities(t *testing.T) {
	m := Chat("m", WithAPIKey("k"), WithAccountID("a"))
	caps := provider.ModelCapabilitiesOf(m)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	m := Chat("@cf/meta/llama-3.1-8b-instruct", WithAPIKey("k"), WithAccountID("a"))
	if m.ModelID() != "@cf/meta/llama-3.1-8b-instruct" {
		t.Errorf("ModelID = %q", m.ModelID())
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

func TestEnvVarResolution(t *testing.T) {
	t.Setenv("CLOUDFLARE_API_TOKEN", "env-tok")
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "env-acc")
	m := Chat("m")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource should be set")
	}
	if !strings.Contains(cm.opts.baseURL, "env-acc") {
		t.Errorf("baseURL should include account ID: %q", cm.opts.baseURL)
	}
}

func TestEnvVarBaseURL(t *testing.T) {
	t.Setenv("CLOUDFLARE_API_TOKEN", "k")
	t.Setenv("CLOUDFLARE_BASE_URL", "https://gateway.example.com/v1")
	m := Chat("m")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://gateway.example.com/v1" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestConnectionError(t *testing.T) {
	model := Chat("m", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("expected error")
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
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}]},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := m.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "x", InputSchema: []byte(`{"type":"object"}`)},
		},
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
		_, _ = fmt.Fprint(w, `{"model":"@cf/baai/bge-base-en-v1.5","data":[{"embedding":[0.1,0.2],"index":0},{"embedding":[0.3,0.4],"index":1}],"usage":{"prompt_tokens":4,"total_tokens":4}}`)
	}))
	defer server.Close()

	m := Embedding("@cf/baai/bge-base-en-v1.5", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := m.DoEmbed(t.Context(), []string{"a", "b"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 || result.Embeddings[0][0] != 0.1 {
		t.Errorf("embeddings = %v", result.Embeddings)
	}
	if result.Usage.InputTokens != 4 {
		t.Errorf("InputTokens = %d", result.Usage.InputTokens)
	}
}

func TestEmbed_MissingAccountID(t *testing.T) {
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_BASE_URL", "")
	m := Embedding("m", WithAPIKey("k"))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil || !strings.Contains(err.Error(), "account ID") {
		t.Fatalf("expected account ID error, got %v", err)
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
	t.Setenv("CLOUDFLARE_API_TOKEN", "")
	m := Embedding("m", WithBaseURL("http://x"))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{})
	if err == nil {
		t.Fatal("expected error")
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
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"m","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`)
	}))
	defer server.Close()

	m := Embedding("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := m.DoEmbed(t.Context(), []string{"a"}, provider.EmbedParams{
		ProviderOptions: map[string]any{"dimensions": 512},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbed_MaxValuesPerCall(t *testing.T) {
	m := Embedding("m", WithAPIKey("k"), WithAccountID("a"))
	if m.MaxValuesPerCall() != 100 {
		t.Errorf("MaxValuesPerCall = %d", m.MaxValuesPerCall())
	}
}
