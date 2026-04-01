package vertex

import (
	"context"
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
		if !strings.Contains(r.URL.Path, "/chat/completions") {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("test-token")), WithBaseURL(server.URL))
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
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-123","model":"gemini-2.5-pro","choices":[{"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
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

func TestNoProject(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	model := Chat("model", WithTokenSource(provider.StaticToken("tok")))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "GOOGLE_CLOUD_PROJECT") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("model", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDefaultLocation(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithProject("proj"))
	cm := model.(*chatModel)
	if cm.opts.location != "us-central1" {
		t.Errorf("location = %q, want us-central1", cm.opts.location)
	}
}

func TestCustomLocation(t *testing.T) {
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithLocation("europe-west1"))
	cm := model.(*chatModel)
	if cm.opts.location != "europe-west1" {
		t.Errorf("location = %q", cm.opts.location)
	}
}

func TestWithHTTPClient(t *testing.T) {
	c := &http.Client{}
	model := Chat("model", WithTokenSource(provider.StaticToken("tok")), WithBaseURL("http://x"), WithHTTPClient(c))
	cm := model.(*chatModel)
	if cm.httpClient() != c {
		t.Error("custom client not set")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL("http://x"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.ToolCall {
		t.Error("unexpected capabilities")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("tok")), WithBaseURL("http://x"))
	if model.ModelID() != "gemini-2.5-pro" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestConnectionError(t *testing.T) {
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL("http://127.0.0.1:1"))
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

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// errReader is an io.ReadCloser that returns an error on Read.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, fmt.Errorf("forced read error") }
func (errReader) Close() error             { return nil }

type errBodyTransport struct{}

func (t *errBodyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       errReader{},
		Header:     make(http.Header),
	}, nil
}

func TestDoGenerate_ReadError(t *testing.T) {
	model := Chat("m",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL("http://fake"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestTokenSourceError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token fetch failed")
	})
	model := Chat("m", WithTokenSource(ts), WithBaseURL("http://fake"))
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

func TestDefaultURLWithProject(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	// Use baseURL to point at test server, but verify project/location env fallbacks work.
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "my-project")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	cm := model.(*chatModel)
	if cm.opts.project != "my-project" {
		t.Errorf("project = %q, want my-project", cm.opts.project)
	}
	if cm.opts.location != "europe-west4" {
		t.Errorf("location = %q, want europe-west4", cm.opts.location)
	}
}

func TestNoTokenSource(t *testing.T) {
	// No token source + custom baseURL -- auth is skipped, request goes through unauthenticated.
	// With a fake URL this should fail with a connection error.
	model := Chat("m", WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for unreachable endpoint")
	}
	// Should be a connection error, not an auth error.
	if strings.Contains(err.Error(), "no API key") {
		t.Errorf("should not be auth error: %v", err)
	}
}

type urlCapturingTransport struct {
	captured string
}

func (tr *urlCapturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	tr.captured = req.URL.String()
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(`{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)),
		Header:     make(http.Header),
	}, nil
}

func TestRequestHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Request-Header") != "from-params" {
			t.Errorf("X-Request-Header = %q", r.Header.Get("X-Request-Header"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{"X-Request-Header": "from-params"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDefaultURL(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Chat("gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"),
		WithLocation("us-central1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	expected := "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi/chat/completions"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestWithAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer my-api-key" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("m", WithAPIKey("my-api-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- autoTokenSource + resolveOpts auto-resolve tests ---

func TestAutoTokenSource_GoogleAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "gkey-123")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")

	ts := autoTokenSource(false)
	if ts == nil {
		t.Fatal("expected non-nil token source")
	}
	aks, ok := ts.(*apiKeyTokenSource)
	if !ok {
		t.Fatal("expected apiKeyTokenSource")
	}
	if aks.key != "gkey-123" {
		t.Errorf("key = %q, want gkey-123", aks.key)
	}
}

func TestAutoTokenSource_GeminiAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "gemini-456")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")

	ts := autoTokenSource(false)
	aks, ok := ts.(*apiKeyTokenSource)
	if !ok {
		t.Fatal("expected apiKeyTokenSource")
	}
	if aks.key != "gemini-456" {
		t.Errorf("key = %q, want gemini-456", aks.key)
	}
}

func TestAutoTokenSource_GoogleGenerativeAIAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "gen-789")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")

	ts := autoTokenSource(false)
	aks, ok := ts.(*apiKeyTokenSource)
	if !ok {
		t.Fatal("expected apiKeyTokenSource")
	}
	if aks.key != "gen-789" {
		t.Errorf("key = %q, want gen-789", aks.key)
	}
}

func TestAutoTokenSource_FallsBackToADC(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	// ADC will also fail without real creds, but autoTokenSource still returns a non-nil source.
	ts := autoTokenSource(false)
	if ts == nil {
		t.Fatal("expected non-nil token source from ADC fallback")
	}
	// Should NOT be an apiKeyTokenSource.
	if _, ok := ts.(*apiKeyTokenSource); ok {
		t.Error("expected ADC-based token source, got apiKeyTokenSource")
	}
}

func TestResolveOpts_AutoResolveAuth(t *testing.T) {
	// No explicit tokenSource, no baseURL → should auto-resolve.
	t.Setenv("GOOGLE_API_KEY", "auto-key")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")

	o := resolveOpts(nil)
	if o.tokenSource == nil {
		t.Fatal("expected auto-resolved token source")
	}
	aks, ok := o.tokenSource.(*apiKeyTokenSource)
	if !ok {
		t.Fatal("expected apiKeyTokenSource from auto-resolve")
	}
	if aks.key != "auto-key" {
		t.Errorf("key = %q, want auto-key", aks.key)
	}
}

// --- API key URL routing tests ---

func TestResolveURL_APIKeyUsesGeminiEndpoint(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Chat("gemini-2.5-pro",
		WithAPIKey("my-key"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	expected := "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestWireModelID_APIKeyReturnsBareModelName(t *testing.T) {
	model := Chat("gemini-2.5-pro", WithAPIKey("my-key"))
	cm := model.(*chatModel)
	if got := cm.wireModelID(); got != "gemini-2.5-pro" {
		t.Errorf("wireModelID() = %q, want gemini-2.5-pro", got)
	}
}

func TestWireModelID_VertexAddsPrefix(t *testing.T) {
	model := Chat("gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("proj"),
	)
	cm := model.(*chatModel)
	if got := cm.wireModelID(); got != "google/gemini-2.5-pro" {
		t.Errorf("wireModelID() = %q, want google/gemini-2.5-pro", got)
	}
}

func TestWireModelID_AlreadyPrefixed(t *testing.T) {
	model := Chat("google/gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("proj"),
	)
	cm := model.(*chatModel)
	if got := cm.wireModelID(); got != "google/gemini-2.5-pro" {
		t.Errorf("wireModelID() = %q, want google/gemini-2.5-pro (should not double-prefix)", got)
	}
}

func TestNativeBaseURL_APIKeyUsesGeminiAPI(t *testing.T) {
	o := options{
		tokenSource: &apiKeyTokenSource{key: "test-key"},
	}
	got := nativeBaseURL(o)
	expected := "https://generativelanguage.googleapis.com/v1beta"
	if got != expected {
		t.Errorf("nativeBaseURL() = %q, want %q", got, expected)
	}
}

func TestNativeURL_APIKeyAppendsKeyParam(t *testing.T) {
	o := options{
		tokenSource: &apiKeyTokenSource{key: "test-key"},
	}
	got, err := nativeURL(o, "models/text-embedding-004:predict")
	if err != nil {
		t.Fatal(err)
	}
	expected := "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:predict?key=test-key"
	if got != expected {
		t.Errorf("nativeURL() = %q, want %q", got, expected)
	}
}

func TestAutoTokenSource_HasProjectUsesADC(t *testing.T) {
	// When hasProject=true, autoTokenSource should prefer ADC (not API key env vars).
	t.Setenv("GOOGLE_API_KEY", "should-not-use")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
	ts := autoTokenSource(true)
	if ts == nil {
		t.Fatal("expected non-nil token source")
	}
	// Should NOT be an apiKeyTokenSource -- ADC is preferred when project is set.
	if _, ok := ts.(*apiKeyTokenSource); ok {
		t.Error("expected ADC token source, got apiKeyTokenSource")
	}
}

func TestStripGeminiProviderOptions_RemovesThinkingConfig(t *testing.T) {
	params := &provider.GenerateParams{
		ProviderOptions: map[string]any{
			"thinkingConfig": map[string]any{"thinkingBudget": 1024},
			"otherOption":    "keep",
		},
	}
	stripGeminiProviderOptions(params)
	if _, ok := params.ProviderOptions["thinkingConfig"]; ok {
		t.Error("expected thinkingConfig to be removed")
	}
	if _, ok := params.ProviderOptions["otherOption"]; !ok {
		t.Error("expected otherOption to be kept")
	}
}

func TestStripGeminiProviderOptions_NilOptions(t *testing.T) {
	params := &provider.GenerateParams{}
	stripGeminiProviderOptions(params) // should not panic
}

func TestSanitizeToolSchemas_CleansTool(t *testing.T) {
	// Schema with additionalProperties (which Gemini doesn't support).
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
		"additionalProperties": false,
	}
	raw, _ := json.Marshal(schema)
	params := &provider.GenerateParams{
		Tools: []provider.ToolDefinition{
			{Name: "test_tool", InputSchema: raw},
		},
	}
	sanitizeToolSchemas(params)
	// After sanitization, additionalProperties should be removed.
	var result map[string]any
	if err := json.Unmarshal(params.Tools[0].InputSchema, &result); err != nil {
		t.Fatal(err)
	}
	if _, ok := result["additionalProperties"]; ok {
		t.Error("expected additionalProperties to be removed by sanitization")
	}
}

func TestSanitizeToolSchemas_EmptySchema(t *testing.T) {
	params := &provider.GenerateParams{
		Tools: []provider.ToolDefinition{
			{Name: "no_schema", InputSchema: nil},
		},
	}
	sanitizeToolSchemas(params) // should skip gracefully
}

func TestSanitizeToolSchemas_InvalidJSON(t *testing.T) {
	params := &provider.GenerateParams{
		Tools: []provider.ToolDefinition{
			{Name: "bad_json", InputSchema: json.RawMessage(`{invalid}`)},
		},
	}
	sanitizeToolSchemas(params) // should skip gracefully, not panic
}

func TestResolveOpts_EnvVarBaseURL(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_BASE_URL", "https://custom.vertex.com")
	t.Setenv("GOOGLE_VERTEX_PROJECT", "myproject")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "us-east1")
	o := resolveOpts(nil)
	if o.baseURL != "https://custom.vertex.com" {
		t.Errorf("baseURL = %q", o.baseURL)
	}
	if o.project != "myproject" {
		t.Errorf("project = %q", o.project)
	}
	if o.location != "us-east1" {
		t.Errorf("location = %q", o.location)
	}
}

func TestResolveOpts_FallbackEnvVars(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "cloud-proj")
	t.Setenv("GCLOUD_PROJECT", "")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")
	t.Setenv("GOOGLE_VERTEX_BASE_URL", "")
	o := resolveOpts(nil)
	if o.project != "cloud-proj" {
		t.Errorf("project = %q, want cloud-proj", o.project)
	}
	if o.location != "asia-east1" {
		t.Errorf("location = %q, want asia-east1", o.location)
	}
}

func TestResolveOpts_GcloudProjectFallback(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "gcloud-proj")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("GOOGLE_VERTEX_BASE_URL", "")
	o := resolveOpts(nil)
	if o.project != "gcloud-proj" {
		t.Errorf("project = %q, want gcloud-proj", o.project)
	}
	// location should default to us-central1
	if o.location != "us-central1" {
		t.Errorf("location = %q, want us-central1", o.location)
	}
}

// --- Coverage: setAuth ---

func TestSetAuth_NilTokenSource(t *testing.T) {
	req, _ := http.NewRequest("POST", "http://example.com", nil)
	err := setAuth(t.Context(), req, nil)
	if err != nil {
		t.Fatalf("expected nil error for nil token source, got %v", err)
	}
	if req.Header.Get("Authorization") != "" {
		t.Error("Authorization header should not be set for nil token source")
	}
}

func TestSetAuth_TokenError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token error")
	})
	req, _ := http.NewRequest("POST", "http://example.com", nil)
	err := setAuth(t.Context(), req, ts)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("error = %v", err)
	}
}

func TestSetAuth_Success(t *testing.T) {
	ts := provider.StaticToken("my-token")
	req, _ := http.NewRequest("POST", "http://example.com", nil)
	err := setAuth(t.Context(), req, ts)
	if err != nil {
		t.Fatal(err)
	}
	if req.Header.Get("Authorization") != "Bearer my-token" {
		t.Errorf("Authorization = %q", req.Header.Get("Authorization"))
	}
}

func TestSanitizeToolSchemas_MarshalError(t *testing.T) {
	// Swap jsonMarshalFunc to simulate a marshal error.
	orig := jsonMarshalFunc
	jsonMarshalFunc = func(v any) ([]byte, error) {
		return nil, fmt.Errorf("forced marshal error")
	}
	defer func() { jsonMarshalFunc = orig }()

	schema := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
	params := &provider.GenerateParams{
		Tools: []provider.ToolDefinition{
			{Name: "tool1", InputSchema: schema},
		},
	}
	sanitizeToolSchemas(params)
	// Schema should remain unchanged (marshal error → continue).
	if string(params.Tools[0].InputSchema) != string(schema) {
		t.Error("expected schema unchanged after marshal error")
	}
}

func TestValidGCPIdentifier(t *testing.T) {
	valid := []string{
		"us-central1",
		"my-project-123",
		"example.com:my-project", // domain-scoped project
		"a",
	}
	for _, v := range valid {
		if !validGCPIdentifier(v) {
			t.Errorf("expected %q to be valid", v)
		}
	}
	invalid := []string{
		"",
		"../evil",
		"us-central1/../../x",
		"has spaces",
		"-starts-with-dash",
		"foo..bar", // path traversal
	}
	for _, v := range invalid {
		if validGCPIdentifier(v) {
			t.Errorf("expected %q to be invalid", v)
		}
	}
}

func TestNativeURL_InvalidLocation(t *testing.T) {
	o := options{
		tokenSource: provider.StaticToken("tok"),
		project:     "my-project",
		location:    "../evil",
	}
	_, err := nativeURL(o, "models/text-embedding-004:predict")
	if err == nil {
		t.Fatal("expected error for invalid location")
	}
	if !strings.Contains(err.Error(), "invalid location") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestNativeURL_InvalidProject(t *testing.T) {
	o := options{
		tokenSource: provider.StaticToken("tok"),
		project:     "has spaces",
		location:    "us-central1",
	}
	_, err := nativeURL(o, "models/text-embedding-004:predict")
	if err == nil {
		t.Fatal("expected error for invalid project")
	}
	if !strings.Contains(err.Error(), "invalid project") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestResolveURL_InvalidLocation(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "my-project")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "../evil")
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"), WithLocation("../evil"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for invalid location")
	}
	if !strings.Contains(err.Error(), "invalid location") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestResolveURL_InvalidProject(t *testing.T) {
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")),
		WithProject("has spaces"), WithLocation("us-central1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for invalid project")
	}
	if !strings.Contains(err.Error(), "invalid project") {
		t.Errorf("unexpected error: %s", err)
	}
}

// TestChat_PromptCachingIgnored verifies that passing PromptCaching=true to the Vertex
// provider succeeds (warning is written to stderr, not returned as error).
func TestChat_PromptCachingIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-test","model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("test-token")), WithBaseURL(server.URL))

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

	streamModel := Chat("gemini-2.5-pro", WithTokenSource(provider.StaticToken("test-token")), WithBaseURL(streamServer.URL))

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
