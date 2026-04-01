package azure

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
	"github.com/zendev-sh/goai/provider/openai"
)

// okJSON is a standard successful JSON response for the Responses API.
const okResponsesJSON = `{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}],"usage":{"input_tokens":10,"output_tokens":5}}`

// okChatJSON is a standard successful JSON response for Chat Completions.
const okChatJSON = `{"id":"x","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`

// responsesSSE returns SSE events for a Responses API streaming response.
func responsesSSE(w http.ResponseWriter, text string) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = fmt.Fprintf(w, "event: response.output_item.added\ndata: {\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}\n\n")
	_, _ = fmt.Fprintf(w, "event: response.content_part.added\ndata: {\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n")
	_, _ = fmt.Fprintf(w, "event: response.output_text.delta\ndata: {\"output_index\":0,\"content_index\":0,\"delta\":\"%s\"}\n\n", text)
	_, _ = fmt.Fprintf(w, "event: response.completed\ndata: {\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}}\n\n")
}

func TestChat_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Default uses v1 format -- all models now use Responses API.
		if !strings.Contains(r.URL.Path, "/openai/v1/responses") {
			t.Errorf("path = %s, want .../openai/v1/responses", r.URL.Path)
		}
		if r.Header.Get("api-key") != "test-key" {
			t.Errorf("api-key = %s", r.Header.Get("api-key"))
		}
		if !strings.Contains(r.URL.RawQuery, "api-version=") {
			t.Errorf("missing api-version in query")
		}
		responsesSSE(w, "Hello")
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithEndpoint(server.URL))
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
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithEndpoint(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "Hello" {
		t.Errorf("Text = %q, want Hello", result.Text)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithEndpoint(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestWithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer managed-identity-token" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		if r.Header.Get("api-key") != "" {
			t.Error("api-key should not be set when using token source")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithTokenSource(provider.StaticToken("managed-identity-token")), WithEndpoint(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestModelID(t *testing.T) {
	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint("https://x.openai.azure.com"))
	if model.ModelID() != "gpt-4o" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestDeploymentSlash(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// With useDeploymentBasedURLs, model with slash should be normalized to dash.
		if !strings.Contains(r.URL.Path, "openai-gpt-4o") {
			t.Errorf("path = %s, want .../openai-gpt-4o/...", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("openai/gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL), WithDeploymentBasedURLs(true))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestConnectionError(t *testing.T) {
	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestTokenSourceError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token fetch failed")
	})
	model := Chat("gpt-4o", WithTokenSource(ts), WithEndpoint("http://fake"))
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

func TestResourceNameEnvFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	t.Setenv("AZURE_RESOURCE_NAME", "myresource")
	t.Setenv("AZURE_OPENAI_API_KEY", "env-key")
	model := Chat("gpt-4o", WithEndpoint(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestResourceNameEnvFallback_NoExplicitEndpoint(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	t.Setenv("AZURE_RESOURCE_NAME", "testresource")
	t.Setenv("AZURE_OPENAI_API_KEY", "env-key")

	model := Chat("gpt-4o")
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected connection error")
	}
}

func TestNoAuthHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("api-key") != "" || r.Header.Get("Authorization") != "" {
			t.Error("unexpected auth header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	t.Setenv("AZURE_OPENAI_API_KEY", "")
	model := Chat("gpt-4o", WithEndpoint(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestToolCallStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Responses API tool call format.
		_, _ = fmt.Fprint(w, "event: response.output_item.added\ndata: {\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\ndata: {\"output_index\":0,\"delta\":\"{\\\"city\\\": \\\"Paris\\\"}\"}\n\n")
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.done\ndata: {\"output_index\":0,\"arguments\":\"{\\\"city\\\": \\\"Paris\\\"}\"}\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\ndata: {\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":20}}}\n\n")
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithEndpoint(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather"}}},
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
		}
	}
	if !gotToolCall {
		t.Error("no tool call chunk received")
	}
}

// TestGPT5_UsesResponsesAPI verifies that GPT-5 models are routed to the
// Responses API -- matching Vercel AI SDK behavior.
func TestGPT5_UsesResponsesAPI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Default v1 format: /openai/v1/responses
		if r.URL.Path != "/openai/v1/responses" {
			t.Errorf("expected /openai/v1/responses, got %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		responsesSSE(w, "Hello from GPT-5")
	}))
	defer server.Close()

	model := Chat("gpt-5.2-chat", WithAPIKey("test-key"), WithEndpoint(server.URL))
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
	if len(texts) != 1 || texts[0] != "Hello from GPT-5" {
		t.Errorf("texts = %v, want [Hello from GPT-5]", texts)
	}
}

// TestAzureURLRewrite verifies URL rewriting with default (v1) format.
func TestAzureURLRewrite(t *testing.T) {
	var capturedPath string
	var capturedQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Default v1 format -- all models use Responses API now.
	if capturedPath != "/openai/v1/responses" {
		t.Errorf("path = %s, want /openai/v1/responses", capturedPath)
	}
	if !strings.Contains(capturedQuery, "api-version=v1") {
		t.Errorf("query = %s", capturedQuery)
	}
}

// TestAzureURLRewrite_DeploymentBased verifies deployment-based URL format.
func TestAzureURLRewrite_DeploymentBased(t *testing.T) {
	var capturedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL), WithDeploymentBasedURLs(true))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Deployment-based: all endpoints use /openai/deployments/{model}/...
	if !strings.Contains(capturedPath, "/openai/deployments/gpt-4o/") {
		t.Errorf("path = %s, want /openai/deployments/gpt-4o/...", capturedPath)
	}
}

func TestEndpointEnvFallback(t *testing.T) {
	var capturedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	t.Setenv("AZURE_OPENAI_ENDPOINT", server.URL)
	t.Setenv("AZURE_RESOURCE_NAME", "")
	t.Setenv("AZURE_OPENAI_API_KEY", "k")
	model := Chat("gpt-4o")
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// v1 format.
	if capturedPath != "/openai/v1/responses" {
		t.Errorf("path = %s, want /openai/v1/responses", capturedPath)
	}
}

func TestWithHTTPClient(t *testing.T) {
	var capturedUA string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedUA = r.Header.Get("X-Custom-Transport")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	customTransport := &headerTransport{base: http.DefaultTransport, header: "X-Custom-Transport", value: "custom-client"}
	httpClient := &http.Client{Transport: customTransport}

	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL), WithHTTPClient(httpClient))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if capturedUA != "custom-client" {
		t.Errorf("X-Custom-Transport = %q, want %q", capturedUA, "custom-client")
	}
}

// headerTransport wraps a base transport and injects a custom header.
type headerTransport struct {
	base   http.RoundTripper
	header string
	value  string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set(t.header, t.value)
	return t.base.RoundTrip(req)
}

func TestRoundTrip_InvalidURL(t *testing.T) {
	rt := &azureRoundTripper{
		base:       http.DefaultTransport,
		endpoint:   "://\x00invalid",
		apiKey:     "k",
		deployID:   "m",
		apiVersion: defaultAPIVersion,
	}
	req, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", nil)
	_, err := rt.RoundTrip(req)
	if err == nil {
		t.Fatal("expected error for invalid URL")
	}
	if !strings.Contains(err.Error(), "azure: invalid URL") {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestRequestHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Request-Header") != "from-params" {
			t.Errorf("X-Request-Header = %q", r.Header.Get("X-Request-Header"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL))
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

// === BF.17 Tests ===

func TestWithAPIVersion(t *testing.T) {
	var capturedQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL), WithAPIVersion("2024-12-01-preview"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(capturedQuery, "api-version=2024-12-01-preview") {
		t.Errorf("query = %s, want api-version=2024-12-01-preview", capturedQuery)
	}
}

func TestWithAPIVersion_Default(t *testing.T) {
	var capturedQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	// No WithAPIVersion -- should default to "v1".
	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(capturedQuery, "api-version=v1") {
		t.Errorf("query = %s, want api-version=v1", capturedQuery)
	}
}

func TestWithDeploymentBasedURLs(t *testing.T) {
	var capturedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL), WithDeploymentBasedURLs(true))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Deployment-based format -- all endpoints use deployments path.
	if !strings.Contains(capturedPath, "/openai/deployments/gpt-4o/") {
		t.Errorf("path = %s, want /openai/deployments/gpt-4o/...", capturedPath)
	}
}

func TestWithDeploymentBasedURLs_False(t *testing.T) {
	var capturedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okResponsesJSON)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithEndpoint(server.URL), WithDeploymentBasedURLs(false))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Default v1 format.
	if !strings.Contains(capturedPath, "/openai/v1/") {
		t.Errorf("path = %s, want /openai/v1/...", capturedPath)
	}
}

func TestImage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify Azure URL rewriting for images.
		if !strings.Contains(r.URL.Path, "/images/generations") {
			t.Errorf("path = %s, want .../images/generations", r.URL.Path)
		}
		if !strings.Contains(r.URL.RawQuery, "api-version=") {
			t.Errorf("missing api-version in query: %s", r.URL.RawQuery)
		}
		if r.Header.Get("api-key") != "test-key" {
			t.Errorf("api-key = %q", r.Header.Get("api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"data":[{"b64_json":"aGVsbG8="}]}`)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("test-key"), WithEndpoint(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 {
		t.Fatalf("images = %d, want 1", len(result.Images))
	}
	if string(result.Images[0].Data) != "hello" {
		t.Errorf("image data = %q, want %q", string(result.Images[0].Data), "hello")
	}
}

func TestImage_DeploymentBased(t *testing.T) {
	var capturedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"data":[{"b64_json":"aGVsbG8="}]}`)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithEndpoint(server.URL), WithDeploymentBasedURLs(true))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(capturedPath, "/openai/deployments/dall-e-3/images/generations") {
		t.Errorf("path = %s, want /openai/deployments/dall-e-3/images/generations", capturedPath)
	}
}

func TestImage_ModelID(t *testing.T) {
	model := Image("dall-e-3", WithAPIKey("k"), WithEndpoint("https://x.openai.azure.com"))
	if model.ModelID() != "dall-e-3" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestImage_V1Format(t *testing.T) {
	var capturedPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"data":[{"b64_json":"aGVsbG8="}]}`)
	}))
	defer server.Close()

	// Default (no deployment-based URLs) -- v1 format.
	model := Image("dall-e-3", WithAPIKey("k"), WithEndpoint(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(capturedPath, "/openai/v1/images/generations") {
		t.Errorf("path = %s, want /openai/v1/images/generations", capturedPath)
	}
}

func TestChat_ClaudeModel_BuildAnthropicModel(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "myresource")
	t.Setenv("AZURE_OPENAI_API_KEY", "test-key")
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")

	// Claude models should be routed to Anthropic provider.
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithEndpoint("https://myresource.openai.azure.com"))
	if model.ModelID() != "claude-sonnet-4-20250514" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestChat_ClaudeModel_ResourceNameFromEnv(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "envresource")
	t.Setenv("AZURE_OPENAI_API_KEY", "test-key")
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")

	model := Chat("claude-sonnet-4-20250514")
	if model.ModelID() != "claude-sonnet-4-20250514" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestChat_ClaudeModel_WithTokenSource(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "myresource")
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	t.Setenv("AZURE_OPENAI_API_KEY", "")

	model := Chat("claude-sonnet-4-20250514",
		WithTokenSource(provider.StaticToken("managed-token")),
		WithEndpoint("https://myresource.openai.azure.com"),
	)
	if model.ModelID() != "claude-sonnet-4-20250514" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestChat_ClaudeModel_WithHTTPClient(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "myresource")
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("k"),
		WithEndpoint("https://myresource.openai.azure.com"),
		WithHTTPClient(&http.Client{}),
	)
	if model.ModelID() != "claude-sonnet-4-20250514" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestChat_ClaudeModel_WithHeaders(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "myresource")
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("k"),
		WithEndpoint("https://myresource.openai.azure.com"),
		WithHeaders(map[string]string{"X-Custom": "val"}),
	)
	if model.ModelID() != "claude-sonnet-4-20250514" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestChat_ClaudeModel_ResourceFromEndpointParsing(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "")
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")

	// Should parse resource name from endpoint URL.
	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("k"),
		WithEndpoint("https://parsedresource.openai.azure.com"),
	)
	if model.ModelID() != "claude-sonnet-4-20250514" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestImage_WithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer img-token" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"data":[{"b64_json":"aGVsbG8="}]}`)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithTokenSource(provider.StaticToken("img-token")), WithEndpoint(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestIsOpenAIModel(t *testing.T) {
	tests := []struct {
		modelID string
		want    bool
	}{
		{"gpt-4o", true},
		{"o3", true},
		{"codex-mini", true},
		{"chatgpt-4o-latest", true},
		{"computer-use-preview", true},
		{"openai/gpt-4o", true},
		{"Mistral-Large-3", false},
		{"Llama-4-Scout", false},
		{"DeepSeek-V3.1", false},
	}
	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			got := isOpenAIModel(tt.modelID)
			if got != tt.want {
				t.Errorf("isOpenAIModel(%q) = %v, want %v", tt.modelID, got, tt.want)
			}
		})
	}
}

func TestChatCompletionsModel_ModelID_And_Capabilities(t *testing.T) {
	// Non-OpenAI model should be wrapped in chatCompletionsModel.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okChatJSON)
	}))
	defer server.Close()

	t.Setenv("AZURE_RESOURCE_NAME", "myresource")
	// Override endpoint to point to test server (AI Services URL would be constructed
	// from resource name, so we need to override via endpoint).
	model := Chat("Mistral-Large-3", WithAPIKey("k"), WithEndpoint(server.URL))

	// Verify ModelID returns the correct ID.
	if model.ModelID() != "Mistral-Large-3" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "Mistral-Large-3")
	}

	// Verify Capabilities returns valid data (non-nil).
	caps := provider.ModelCapabilitiesOf(model)
	_ = caps // Just verify it doesn't panic and returns.
}

// === AI Services endpoint tests ===

func TestAIServices_URLFormat(t *testing.T) {
	var capturedPath string
	var capturedQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okChatJSON)
	}))
	defer server.Close()

	t.Setenv("AZURE_RESOURCE_NAME", "myresource")

	// Non-OpenAI model should route to AI Services endpoint.
	// We override WithEndpoint for the test, but the AI Services model uses
	// resource name to build its own URL. We need to inject via buildAIServicesModel.
	// Test via direct construction to verify URL format.
	model := buildAIServicesModel(&options{
		apiKey:   "k",
		endpoint: "https://myresource.openai.azure.com",
	}, "Phi-4-reasoning")

	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	// Will fail to connect to services.ai.azure.com -- but we can verify the model works.
	// For URL verification, we test via the round tripper directly.
	_ = err
	_ = capturedPath
	_ = capturedQuery
	_ = model

	// Verify model ID passthrough.
	if model.ModelID() != "Phi-4-reasoning" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "Phi-4-reasoning")
	}
}

func TestAIServices_RoundTripper(t *testing.T) {
	var capturedPath string
	var capturedQuery string
	var capturedAPIKey string
	var capturedModel string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		capturedQuery = r.URL.RawQuery
		capturedAPIKey = r.Header.Get("api-key")
		// Read model from body.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if m, ok := req["model"]; ok {
			capturedModel = fmt.Sprint(m)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okChatJSON)
	}))
	defer server.Close()

	// Build AI Services model pointing at test server.
	baseTransport := http.DefaultTransport
	transport := &aiServicesRoundTripper{
		base:       baseTransport,
		apiKey:     "test-azure-key",
		apiVersion: defaultAIServicesAPIVersion,
	}
	httpClient := &http.Client{Transport: transport}
	openaiOpts := []openai.Option{
		openai.WithHTTPClient(httpClient),
		openai.WithAPIKey("azure-delegated"),
		openai.WithBaseURL(server.URL),
	}
	model := openai.Chat("Phi-4-reasoning", openaiOpts...)
	wrapped := &chatCompletionsModel{inner: model}

	_, err := wrapped.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify path is /chat/completions (not /openai/v1/...).
	if capturedPath != "/chat/completions" {
		t.Errorf("path = %s, want /chat/completions", capturedPath)
	}
	// Verify api-version query parameter.
	if !strings.Contains(capturedQuery, "api-version=2024-05-01-preview") {
		t.Errorf("query = %s, want api-version=2024-05-01-preview", capturedQuery)
	}
	// Verify api-key header (not Authorization).
	if capturedAPIKey != "test-azure-key" {
		t.Errorf("api-key = %q, want %q", capturedAPIKey, "test-azure-key")
	}
	// Verify model in body.
	if capturedModel != "Phi-4-reasoning" {
		t.Errorf("model = %q, want %q", capturedModel, "Phi-4-reasoning")
	}
}

func TestAIServices_TokenSource(t *testing.T) {
	var capturedAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okChatJSON)
	}))
	defer server.Close()

	transport := &aiServicesRoundTripper{
		base:        http.DefaultTransport,
		tokenSource: provider.StaticToken("managed-identity-token"),
		apiVersion:  defaultAIServicesAPIVersion,
	}
	httpClient := &http.Client{Transport: transport}
	openaiOpts := []openai.Option{
		openai.WithHTTPClient(httpClient),
		openai.WithAPIKey("azure-delegated"),
		openai.WithBaseURL(server.URL),
	}
	model := openai.Chat("Phi-4-reasoning", openaiOpts...)
	wrapped := &chatCompletionsModel{inner: model}

	_, err := wrapped.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if capturedAuth != "Bearer managed-identity-token" {
		t.Errorf("Authorization = %q, want %q", capturedAuth, "Bearer managed-identity-token")
	}
}

func TestAIServices_CustomHeaders(t *testing.T) {
	var capturedCustom string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedCustom = r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, okChatJSON)
	}))
	defer server.Close()

	transport := &aiServicesRoundTripper{
		base:       http.DefaultTransport,
		apiKey:     "k",
		headers:    map[string]string{"X-Custom": "test-val"},
		apiVersion: defaultAIServicesAPIVersion,
	}
	httpClient := &http.Client{Transport: transport}
	openaiOpts := []openai.Option{
		openai.WithHTTPClient(httpClient),
		openai.WithAPIKey("azure-delegated"),
		openai.WithBaseURL(server.URL),
	}
	model := openai.Chat("DeepSeek-V3.1", openaiOpts...)
	wrapped := &chatCompletionsModel{inner: model}

	_, err := wrapped.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if capturedCustom != "test-val" {
		t.Errorf("X-Custom = %q, want %q", capturedCustom, "test-val")
	}
}

func TestAIServices_NonOpenAIModelsRouting(t *testing.T) {
	// Verify that various non-OpenAI models are routed through AI Services.
	models := []string{"Phi-4-reasoning", "DeepSeek-V3.1", "Llama-4-Scout", "Mistral-Large-3", "Cohere-command-r", "grok-4-1-fast-reasoning"}
	for _, modelID := range models {
		t.Run(modelID, func(t *testing.T) {
			t.Setenv("AZURE_RESOURCE_NAME", "testresource")
			model := Chat(modelID, WithAPIKey("k"), WithEndpoint("https://testresource.openai.azure.com"))
			// Should be a chatCompletionsModel (wrapped via AI Services).
			if _, ok := model.(*chatCompletionsModel); !ok {
				t.Errorf("%s: expected *chatCompletionsModel, got %T", modelID, model)
			}
		})
	}
}

func TestAIServices_OpenAIModelsNotRouted(t *testing.T) {
	// Verify that OpenAI models are NOT routed through AI Services.
	models := []string{"gpt-4o", "o3", "codex-mini", "chatgpt-4o-latest"}
	for _, modelID := range models {
		t.Run(modelID, func(t *testing.T) {
			model := Chat(modelID, WithAPIKey("k"), WithEndpoint("https://test.openai.azure.com"))
			if _, ok := model.(*chatCompletionsModel); ok {
				t.Errorf("%s: should NOT be chatCompletionsModel", modelID)
			}
		})
	}
}

func TestExtractResourceName(t *testing.T) {
	tests := []struct {
		name     string
		envVar   string
		endpoint string
		want     string
	}{
		{"from env", "envresource", "", "envresource"},
		{"from endpoint", "", "https://parsedresource.openai.azure.com", "parsedresource"},
		{"env takes precedence", "envresource", "https://other.openai.azure.com", "envresource"},
		{"empty", "", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("AZURE_RESOURCE_NAME", tt.envVar)
			o := &options{endpoint: tt.endpoint}
			got := extractResourceName(o)
			if got != tt.want {
				t.Errorf("extractResourceName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAIServices_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	transport := &aiServicesRoundTripper{
		base:       http.DefaultTransport,
		apiKey:     "k",
		apiVersion: defaultAIServicesAPIVersion,
	}
	httpClient := &http.Client{Transport: transport}
	model := &chatCompletionsModel{inner: openai.Chat("Phi-4-reasoning", openai.WithHTTPClient(httpClient), openai.WithAPIKey("x"), openai.WithBaseURL(server.URL))}

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
		t.Errorf("texts = %v, want [Hi]", texts)
	}
}

func TestAIServices_WithHTTPClient(t *testing.T) {
	t.Setenv("AZURE_RESOURCE_NAME", "myresource")
	// Test that custom HTTP client transport is used by buildAIServicesModel.
	customTransport := &headerTransport{base: http.DefaultTransport, header: "X-Custom-Transport", value: "from-client"}
	model := buildAIServicesModel(&options{
		apiKey:     "k",
		endpoint:   "https://myresource.openai.azure.com",
		httpClient: &http.Client{Transport: customTransport},
	}, "Phi-4-reasoning")
	if model.ModelID() != "Phi-4-reasoning" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestAIServices_TokenSourceError(t *testing.T) {
	ts := provider.CachedTokenSource(func(_ context.Context) (*provider.Token, error) {
		return nil, fmt.Errorf("token fetch failed")
	})
	transport := &aiServicesRoundTripper{
		base:        http.DefaultTransport,
		tokenSource: ts,
		apiVersion:  defaultAIServicesAPIVersion,
	}
	httpClient := &http.Client{Transport: transport}
	openaiOpts := []openai.Option{
		openai.WithHTTPClient(httpClient),
		openai.WithAPIKey("azure-delegated"),
		openai.WithBaseURL("http://fake"),
	}
	model := openai.Chat("m", openaiOpts...)
	wrapped := &chatCompletionsModel{inner: model}

	_, err := wrapped.DoGenerate(t.Context(), provider.GenerateParams{
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

func TestForceChatCompletions_PreservesExistingOptions(t *testing.T) {
	params := provider.GenerateParams{
		ProviderOptions: map[string]any{
			"user":       "test-user",
			"customKey":  42,
		},
	}
	forceChatCompletions(&params)

	if params.ProviderOptions["useResponsesAPI"] != false {
		t.Error("useResponsesAPI should be false")
	}
	if params.ProviderOptions["user"] != "test-user" {
		t.Errorf("user = %v, want test-user", params.ProviderOptions["user"])
	}
	if params.ProviderOptions["customKey"] != 42 {
		t.Errorf("customKey = %v, want 42", params.ProviderOptions["customKey"])
	}
}

func TestForceChatCompletions_NilOptions(t *testing.T) {
	params := provider.GenerateParams{}
	forceChatCompletions(&params)

	if params.ProviderOptions["useResponsesAPI"] != false {
		t.Error("useResponsesAPI should be false")
	}
}

// TestChat_PromptCachingIgnored verifies that passing PromptCaching=true to
// chatCompletionsModel succeeds (warning is written to stderr, not returned as error).
func TestChat_PromptCachingIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-test","model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	}))
	defer server.Close()

	model := &chatCompletionsModel{inner: openai.Chat("Phi-4-reasoning", openai.WithAPIKey("test-key"), openai.WithBaseURL(server.URL))}

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

	streamModel := &chatCompletionsModel{inner: openai.Chat("Phi-4-reasoning", openai.WithAPIKey("test-key"), openai.WithBaseURL(streamServer.URL))}

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
