package nvidia

import (
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
		_, _ = strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\ndata: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n").WriteTo(w)
	}))
	defer server.Close()

	model := Chat("meta/llama-3.3-70b-instruct", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
		_, _ = strings.NewReader(`{"id":"chatcmpl-123","model":"meta/llama-3.3-70b-instruct","choices":[{"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`).WriteTo(w)
	}))
	defer server.Close()

	model := Chat("meta/llama-3.3-70b-instruct", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
		_, _ = strings.NewReader(`{"error":{"message":"Rate limited"}}`).WriteTo(w)
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

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func okResponse() *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)),
	}
}

func TestWithHTTPClient(t *testing.T) {
	called := false
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		called = true
		return okResponse(), nil
	})
	m := Chat("model", WithAPIKey("key"), WithHTTPClient(&http.Client{Transport: tr}))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Error("custom HTTP client transport was not invoked")
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = strings.NewReader(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`).WriteTo(w)
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
		_, _ = strings.NewReader(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`).WriteTo(w)
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
	model := Chat("meta/llama-3.3-70b-instruct", WithAPIKey("k"))
	if model.ModelID() != "meta/llama-3.3-70b-instruct" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestChat_EnvVarResolution(t *testing.T) {
	var gotAuth string
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotAuth = req.Header.Get("Authorization")
		return okResponse(), nil
	})
	t.Setenv("NVIDIA_API_KEY", "env-key")
	t.Setenv("NVIDIA_BASE_URL", "")
	m := Chat("meta/llama-3.3-70b-instruct", WithHTTPClient(&http.Client{Transport: tr}))
	if m.ModelID() != "meta/llama-3.3-70b-instruct" {
		t.Errorf("ModelID() = %q", m.ModelID())
	}
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer env-key" {
		t.Errorf("auth = %q", gotAuth)
	}
}

func TestChat_EnvVarBaseURL(t *testing.T) {
	var gotURL string
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotURL = req.URL.String()
		return okResponse(), nil
	})
	t.Setenv("NVIDIA_API_KEY", "k")
	t.Setenv("NVIDIA_BASE_URL", "https://custom.example.com/v1")
	m := Chat("m", WithHTTPClient(&http.Client{Transport: tr}))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(gotURL, "https://custom.example.com/v1/") {
		t.Errorf("URL = %q", gotURL)
	}
}

func TestChat_EnvVarNotOverrideExplicit(t *testing.T) {
	var gotURL string
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotURL = req.URL.String()
		return okResponse(), nil
	})
	t.Setenv("NVIDIA_API_KEY", "env-key")
	t.Setenv("NVIDIA_BASE_URL", "https://env.url")
	m := Chat("m", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"), WithHTTPClient(&http.Client{Transport: tr}))
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(gotURL, "https://explicit.url/") {
		t.Errorf("URL = %q", gotURL)
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
		_, _ = strings.NewReader(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`).WriteTo(w)
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

func TestEmbedding_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = strings.NewReader(`{"data":[{"embedding":[0.1,0.2,0.3],"index":0}],"usage":{"prompt_tokens":10}}`).WriteTo(w)
	}))
	defer server.Close()

	model := Embedding("nvidia/embed-qa-4", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoEmbed(t.Context(), []string{"hello world"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(result.Embeddings))
	}
	if len(result.Embeddings[0]) != 3 {
		t.Errorf("embedding len = %d, want 3", len(result.Embeddings[0]))
	}
}

func TestEmbedding_EnvVarResolution(t *testing.T) {
	var gotAuth string
	tr := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		gotAuth = req.Header.Get("Authorization")
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1}}`)),
		}, nil
	})
	t.Setenv("NVIDIA_API_KEY", "env-key")
	m := Embedding("nvidia/embed-qa-4", WithHTTPClient(&http.Client{Transport: tr}))
	_, err := m.DoEmbed(t.Context(), []string{"hi"}, provider.EmbedParams{})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer env-key" {
		t.Errorf("auth = %q, want Bearer env-key", gotAuth)
	}
}