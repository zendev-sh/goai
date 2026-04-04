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

func TestAnthropicChat_StreamURL(t *testing.T) {
	var gotPath, gotAuth, gotVersion string
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotVersion = r.Header.Get("anthropic-version")

		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)

		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: message_start\n")
		fmt.Fprint(w, `data: {"type":"message_start","message":{"id":"msg-123","model":"claude-sonnet-4-20250514","role":"assistant","usage":{"input_tokens":10,"cache_read_input_tokens":5,"cache_creation_input_tokens":3}}}`+"\n\n")
		fmt.Fprint(w, "event: content_block_start\n")
		fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		fmt.Fprint(w, "event: content_block_delta\n")
		fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`+"\n\n")
		fmt.Fprint(w, "event: content_block_stop\n")
		fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		fmt.Fprint(w, "event: message_delta\n")
		fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`+"\n\n")
		fmt.Fprint(w, "event: message_stop\n")
		fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := AnthropicChat("claude-sonnet-4-20250514",
		WithTokenSource(provider.StaticToken("test-token")),
		WithBaseURL(server.URL),
	)

	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	var usage provider.Usage
	var hasUsage bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
		if chunk.Usage != (provider.Usage{}) {
			usage = chunk.Usage
			hasUsage = true
		}
	}

	// Verify streaming URL pattern.
	if !strings.HasSuffix(gotPath, "/claude-sonnet-4-20250514:streamRawPredict") {
		t.Errorf("path = %s, want suffix /claude-sonnet-4-20250514:streamRawPredict", gotPath)
	}

	// Verify Bearer auth (not x-api-key).
	if gotAuth != "Bearer test-token" {
		t.Errorf("auth = %s, want Bearer test-token", gotAuth)
	}

	// Verify anthropic-version header.
	if gotVersion != "2023-06-01" {
		t.Errorf("anthropic-version = %s, want 2023-06-01", gotVersion)
	}

	// Verify body transformation: no "model", has "anthropic_version".
	if _, ok := gotBody["model"]; ok {
		t.Error("body should not contain 'model' field")
	}
	if v, ok := gotBody["anthropic_version"].(string); !ok || v != "vertex-2023-10-16" {
		t.Errorf("anthropic_version = %v, want vertex-2023-10-16", gotBody["anthropic_version"])
	}

	// Verify text content.
	if len(texts) != 1 || texts[0] != "Hello" {
		t.Errorf("texts = %v, want [Hello]", texts)
	}

	// Verify cache token extraction.
	if !hasUsage {
		t.Fatal("no usage in stream")
	}
	if usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", usage.InputTokens)
	}
	if usage.CacheReadTokens != 5 {
		t.Errorf("CacheReadTokens = %d, want 5", usage.CacheReadTokens)
	}
	if usage.CacheWriteTokens != 3 {
		t.Errorf("CacheWriteTokens = %d, want 3", usage.CacheWriteTokens)
	}
	if usage.OutputTokens != 7 {
		t.Errorf("OutputTokens = %d, want 7", usage.OutputTokens)
	}
}

func TestAnthropicChat_GenerateURL(t *testing.T) {
	var gotPath string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"id": "msg-456",
			"type": "message",
			"model": "claude-sonnet-4-20250514",
			"role": "assistant",
			"content": [{"type": "text", "text": "Hi there"}],
			"stop_reason": "end_turn",
			"usage": {
				"input_tokens": 20,
				"output_tokens": 5,
				"cache_read_input_tokens": 8,
				"cache_creation_input_tokens": 12
			}
		}`)
	}))
	defer server.Close()

	model := AnthropicChat("claude-sonnet-4-20250514",
		WithTokenSource(provider.StaticToken("tok")),
		WithBaseURL(server.URL),
	)

	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Non-streaming uses :rawPredict.
	if !strings.HasSuffix(gotPath, "/claude-sonnet-4-20250514:rawPredict") {
		t.Errorf("path = %s, want suffix /claude-sonnet-4-20250514:rawPredict", gotPath)
	}

	if result.Text != "Hi there" {
		t.Errorf("text = %q, want %q", result.Text, "Hi there")
	}

	if result.Usage.CacheReadTokens != 8 {
		t.Errorf("CacheReadTokens = %d, want 8", result.Usage.CacheReadTokens)
	}
	if result.Usage.CacheWriteTokens != 12 {
		t.Errorf("CacheWriteTokens = %d, want 12", result.Usage.CacheWriteTokens)
	}
}

func TestAnthropicBaseURL(t *testing.T) {
	tests := []struct {
		name     string
		project  string
		location string
		baseURL  string
		want     string
	}{
		{
			name:     "standard location",
			project:  "my-project",
			location: "us-east5",
			want:     "https://us-east5-aiplatform.googleapis.com/v1/projects/my-project/locations/us-east5/publishers/anthropic/models",
		},
		{
			name:     "global location",
			project:  "my-project",
			location: "global",
			want:     "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/anthropic/models",
		},
		{
			name:    "custom baseURL",
			baseURL: "http://localhost:8080/custom",
			want:    "http://localhost:8080/custom",
		},
		{
			name:    "custom baseURL trailing slash",
			baseURL: "http://localhost:8080/custom/",
			want:    "http://localhost:8080/custom",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := options{
				project:  tt.project,
				location: tt.location,
				baseURL:  tt.baseURL,
			}
			got := anthropicBaseURL(o)
			if got != tt.want {
				t.Errorf("anthropicBaseURL() = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestAnthropicBodyTransformer(t *testing.T) {
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"stream":     true,
		"max_tokens": 1024,
		"messages":   []any{},
	}

	result := anthropicBodyTransformer(body)

	if _, ok := result["model"]; ok {
		t.Error("model should be removed")
	}
	if v := result["anthropic_version"]; v != "vertex-2023-10-16" {
		t.Errorf("anthropic_version = %v, want vertex-2023-10-16", v)
	}
	// Other fields preserved.
	if v := result["max_tokens"]; v != 1024 {
		t.Errorf("max_tokens = %v, want 1024", v)
	}
}

func TestAnthropicURLBuilder(t *testing.T) {
	base := "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5/publishers/anthropic/models"
	builder := anthropicURLBuilder(base)

	stream := builder(base, "claude-sonnet-4-20250514", true)
	if !strings.HasSuffix(stream, ":streamRawPredict") {
		t.Errorf("streaming URL = %s, want :streamRawPredict suffix", stream)
	}

	nonStream := builder(base, "claude-sonnet-4-20250514", false)
	if !strings.HasSuffix(nonStream, ":rawPredict") {
		t.Errorf("non-streaming URL = %s, want :rawPredict suffix", nonStream)
	}
}

func TestAnthropicChat_NoEnvLeak(t *testing.T) {
	// Verify that AnthropicChat does not read ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL.
	t.Setenv("ANTHROPIC_API_KEY", "should-not-be-used")
	t.Setenv("ANTHROPIC_BASE_URL", "http://should-not-be-used")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify it did NOT use the Anthropic env base URL.
		if strings.Contains(r.Host, "should-not-be-used") {
			t.Error("used ANTHROPIC_BASE_URL env var")
		}
		// Verify it did NOT use the Anthropic env API key.
		if r.Header.Get("x-api-key") == "should-not-be-used" {
			t.Error("used ANTHROPIC_API_KEY env var as x-api-key")
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"msg-1","type":"message","model":"claude-sonnet-4-20250514","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := AnthropicChat("claude-sonnet-4-20250514",
		WithTokenSource(provider.StaticToken("vertex-token")),
		WithBaseURL(server.URL),
	)

	_, err := model.DoGenerate(context.Background(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}
