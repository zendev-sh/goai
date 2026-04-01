package fireworks

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

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("accounts/fireworks/models/llama-v3p3-70b-instruct", WithAPIKey("test-key"), WithBaseURL(server.URL))
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

func TestChat_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-123","model":"accounts/fireworks/models/llama-v3p3-70b-instruct","choices":[{"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
	}))
	defer server.Close()

	model := Chat("accounts/fireworks/models/llama-v3p3-70b-instruct", WithAPIKey("test-key"), WithBaseURL(server.URL))
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

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("model", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestNoTokenSource(t *testing.T) {
	model := Chat("model")
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Chat("model", WithAPIKey("key"), WithHTTPClient(c))
	cm := model.(*chatModel)
	if cm.httpClient() != c {
		t.Error("custom client not set")
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithTokenSource(t *testing.T) {
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

func TestCapabilities(t *testing.T) {
	model := Chat("m", WithAPIKey("k"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("accounts/fireworks/models/llama-v3p3-70b-instruct", WithAPIKey("k"))
	if model.ModelID() != "accounts/fireworks/models/llama-v3p3-70b-instruct" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestConnectionError(t *testing.T) {
	model := Chat("m", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"))
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

func TestPerRequestHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Per-Request") != "val" {
			t.Error("missing per-request header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
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

func TestToolCallStreaming(t *testing.T) {
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

	model := Chat("accounts/fireworks/models/llama-v3p3-70b-instruct", WithAPIKey("test-key"), WithBaseURL(server.URL))
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

func TestReadError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "100")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"id"`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChat_EnvVarResolution(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "env-key")
	m := Chat("m")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource should be set from FIREWORKS_API_KEY")
	}
}

func TestChat_EnvVarBaseURL(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "env-key")
	t.Setenv("FIREWORKS_BASE_URL", "https://custom.fireworks.ai/v1")
	m := Chat("m")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://custom.fireworks.ai/v1" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestChat_EnvVarNotOverrideExplicit(t *testing.T) {
	t.Setenv("FIREWORKS_BASE_URL", "https://env.url")
	m := Chat("m", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://explicit.url" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestChat_PromptCachingIgnored(t *testing.T) {
	// DoGenerate with JSON server
	genServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-test","model":"accounts/fireworks/models/llama-v3p1-70b-instruct","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	}))
	defer genServer.Close()

	genModel := Chat("accounts/fireworks/models/llama-v3p1-70b-instruct", WithAPIKey("test-key"), WithBaseURL(genServer.URL))
	genResult, err := genModel.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatalf("DoGenerate unexpected error: %v", err)
	}
	if genResult.Text != "ok" {
		t.Errorf("DoGenerate Text = %q, want ok", genResult.Text)
	}

	// DoStream with SSE server
	streamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer streamServer.Close()

	streamModel := Chat("accounts/fireworks/models/llama-v3p1-70b-instruct", WithAPIKey("test-key"), WithBaseURL(streamServer.URL))
	streamResult, err := streamModel.DoStream(t.Context(), provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		PromptCaching: true,
	})
	if err != nil {
		t.Fatalf("DoStream unexpected error: %v", err)
	}
	for range streamResult.Stream {
	}
}
