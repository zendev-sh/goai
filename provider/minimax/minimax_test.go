package minimax

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// mockAnthropicServer returns a test server that speaks the Anthropic Messages API format.
func mockAnthropicServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(handler)
}

func anthropicResponse(content, finishReason string) string {
	return fmt.Sprintf(`{
		"id": "msg_test",
		"type": "message",
		"role": "assistant",
		"model": "MiniMax-M2.7",
		"content": [{"type": "text", "text": %q}],
		"stop_reason": %q,
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`, content, finishReason)
}

func TestChat_Generate(t *testing.T) {
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/messages") {
			t.Errorf("path = %s, want */v1/messages", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("x-api-key = %q", r.Header.Get("x-api-key"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["model"] != "MiniMax-M2.7" {
			t.Errorf("model = %v", req["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, anthropicResponse("Hello world", "end_turn"))
	})
	defer server.Close()

	model := Chat("MiniMax-M2.7", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"MiniMax-M2.7\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n")
		fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n")
		fmt.Fprint(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
		fmt.Fprint(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n")
		fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	})
	defer server.Close()

	model := Chat("MiniMax-M2.7", WithAPIKey("test-key"), WithBaseURL(server.URL))
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

func TestChat_ToolCall(t *testing.T) {
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		tools, _ := req["tools"].([]any)
		if len(tools) != 1 {
			t.Errorf("tools count = %d, want 1", len(tools))
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "MiniMax-M2.7",
			"content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Tokyo"}}],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 10, "output_tokens": 20}
		}`)
	})
	defer server.Close()

	model := Chat("MiniMax-M2.7", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather in Tokyo"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "get_weather", Description: "Get weather", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "get_weather" {
		t.Errorf("ToolName = %q", result.ToolCalls[0].Name)
	}
	if !strings.Contains(string(result.ToolCalls[0].Input), "Tokyo") {
		t.Errorf("ToolInput = %s, want contains Tokyo", result.ToolCalls[0].Input)
	}
}

func TestChat_HTTPError(t *testing.T) {
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}`)
	})
	defer server.Close()

	model := Chat("model", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *goai.APIError, got %T", err)
	}
	if !apiErr.IsRetryable {
		t.Error("429 should be retryable")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("m", WithAPIKey("k"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature {
		t.Error("Temperature should be true")
	}
	if !caps.ToolCall {
		t.Error("ToolCall should be true")
	}
	if !caps.Reasoning {
		t.Error("Reasoning should be true (Anthropic thinking blocks)")
	}
	if !caps.InputModalities.Text {
		t.Error("InputModalities.Text should be true")
	}
	if caps.InputModalities.Image {
		t.Error("InputModalities.Image should be false")
	}
	if !caps.OutputModalities.Text {
		t.Error("OutputModalities.Text should be true")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("MiniMax-M2.7", WithAPIKey("k"))
	if model.ModelID() != "MiniMax-M2.7" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestNoTokenSource(t *testing.T) {
	t.Setenv("MINIMAX_API_KEY", "")
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
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, anthropicResponse("ok", "end_turn"))
	})
	defer server.Close()

	c := &http.Client{}
	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL), WithHTTPClient(c))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want 'ok'", result.Text)
	}
}

func TestWithHeaders(t *testing.T) {
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "val" {
			t.Error("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, anthropicResponse("ok", "end_turn"))
	})
	defer server.Close()

	model := Chat("m", WithAPIKey("k"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "val"}))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want 'ok'", result.Text)
	}
}

func TestWithTokenSource(t *testing.T) {
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "dynamic" {
			t.Errorf("x-api-key = %q", r.Header.Get("x-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, anthropicResponse("ok", "end_turn"))
	})
	defer server.Close()

	model := Chat("m", WithTokenSource(provider.StaticToken("dynamic")), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want 'ok'", result.Text)
	}
}

func TestChat_EnvVarResolution(t *testing.T) {
	t.Setenv("MINIMAX_API_KEY", "env-key")
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "env-key" {
			t.Errorf("x-api-key = %q, want 'env-key'", r.Header.Get("x-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, anthropicResponse("ok", "end_turn"))
	})
	defer server.Close()

	m := Chat("m", WithBaseURL(server.URL))
	result, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want 'ok'", result.Text)
	}
}

func TestChat_EnvVarBaseURL(t *testing.T) {
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, anthropicResponse("ok", "end_turn"))
	})
	defer server.Close()

	t.Setenv("MINIMAX_API_KEY", "env-key")
	t.Setenv("MINIMAX_BASE_URL", server.URL)
	m := Chat("m")
	result, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want 'ok'", result.Text)
	}
}

func TestChat_EnvVarNotOverrideExplicit(t *testing.T) {
	server := mockAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "explicit" {
			t.Errorf("x-api-key = %q, want 'explicit'", r.Header.Get("x-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, anthropicResponse("ok", "end_turn"))
	})
	defer server.Close()

	t.Setenv("MINIMAX_API_KEY", "env-key")
	model := Chat("m", WithAPIKey("explicit"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want 'ok'", result.Text)
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
}

// --- E2E Tests (require MINIMAX_API_KEY) ---

func skipWithoutKey(t *testing.T) string {
	t.Helper()
	key := os.Getenv("MINIMAX_API_KEY")
	if key == "" {
		t.Skip("MINIMAX_API_KEY not set")
	}
	return key
}

func skipAPIError(t *testing.T, err error) {
	t.Helper()
	var apiErr *goai.APIError
	if errors.As(err, &apiErr) {
		// Skip on account issues or server errors (MiniMax 520s are common).
		if apiErr.StatusCode == 1008 || apiErr.StatusCode == 429 || apiErr.StatusCode >= 500 {
			t.Skipf("API issue (code %d): %s", apiErr.StatusCode, apiErr.Message)
		}
	}
	if strings.Contains(err.Error(), "520") || strings.Contains(err.Error(), "502") || strings.Contains(err.Error(), "503") {
		t.Skipf("Server error: %s", err.Error())
	}
}

func TestE2E_Generate(t *testing.T) {
	key := skipWithoutKey(t)
	models := []string{"MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"}
	for _, m := range models {
		t.Run(m, func(t *testing.T) {
			model := Chat(m, WithAPIKey(key))
			result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
				Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "Reply exactly: OK"}}}},
				MaxOutputTokens: 300,
			})
			if err != nil {
				skipAPIError(t, err)
				t.Fatal(err)
			}
			if !strings.Contains(result.Text, "OK") {
				t.Errorf("Text = %q, want contains 'OK'", result.Text)
			}
		})
	}
}

func TestE2E_Stream(t *testing.T) {
	key := skipWithoutKey(t)
	models := []string{"MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"}
	for _, m := range models {
		t.Run(m, func(t *testing.T) {
			model := Chat(m, WithAPIKey(key))
			sr, err := model.DoStream(t.Context(), provider.GenerateParams{
				Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "Reply exactly: OK"}}}},
				MaxOutputTokens: 300,
			})
			if err != nil {
				skipAPIError(t, err)
				t.Fatal(err)
			}
			var texts []string
			for chunk := range sr.Stream {
				if chunk.Type == provider.ChunkText {
					texts = append(texts, chunk.Text)
				}
			}
			joined := strings.Join(texts, "")
			if !strings.Contains(joined, "OK") {
				t.Errorf("stream text = %q, want contains 'OK'", joined)
			}
		})
	}
}

func TestE2E_ToolCall(t *testing.T) {
	key := skipWithoutKey(t)
	models := []string{"MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"}
	for _, m := range models {
		t.Run(m, func(t *testing.T) {
			model := Chat(m, WithAPIKey(key))
			result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
				// System prompt required for M2.1 tool call reliability.
				System:          "You are a helpful assistant. Use tools when appropriate.",
				Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "What is the weather in Tokyo?"}}}},
				MaxOutputTokens: 500,
				Tools: []provider.ToolDefinition{{
					Name:        "get_weather",
					Description: "Get current weather for a city",
					InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
				}},
				ToolChoice: "auto",
			})
			if err != nil {
				skipAPIError(t, err)
				t.Fatal(err)
			}
			if len(result.ToolCalls) == 0 {
				t.Errorf("no tool calls, text = %q", result.Text)
				return
			}
			if result.ToolCalls[0].Name != "get_weather" {
				t.Errorf("tool = %q, want get_weather", result.ToolCalls[0].Name)
			}
		})
	}
}

func TestE2E_Thinking(t *testing.T) {
	key := skipWithoutKey(t)
	models := []string{"MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"}
	for _, m := range models {
		t.Run(m, func(t *testing.T) {
			model := Chat(m, WithAPIKey(key))

			// Generate with thinking
			result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
				Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "What is 7 * 8? Reply with just the number."}}}},
				MaxOutputTokens: 500,
				ProviderOptions: map[string]any{
					"thinking": map[string]any{
						"type":         "enabled",
						"budgetTokens": 300,
					},
				},
			})
			if err != nil {
				skipAPIError(t, err)
				t.Fatal(err)
			}
			if !strings.Contains(result.Text, "56") {
				t.Errorf("Text = %q, want contains '56'", result.Text)
			}
			// Check for reasoning content via ProviderMetadata or usage tokens.
			hasReasoning := result.Usage.ReasoningTokens > 0
			if meta, ok := result.ProviderMetadata["anthropic"]; ok {
				if reasoning, ok := meta["reasoning"]; ok {
					r := fmt.Sprintf("%v", reasoning)
					if len(r) > 2 { // more than "[]"
						hasReasoning = true
						t.Logf("reasoning: %s", r[:min(len(r), 100)])
					}
				}
			}
			if !hasReasoning {
				t.Error("expected reasoning content in ProviderMetadata or ReasoningTokens > 0")
			}
		})
	}
}

func TestE2E_ThinkingStream(t *testing.T) {
	key := skipWithoutKey(t)
	models := []string{"MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"}
	for _, m := range models {
		t.Run(m, func(t *testing.T) {
			testThinkingStream(t, m, key)
		})
	}
}

func testThinkingStream(t *testing.T, modelID, key string) {
	model := Chat(modelID, WithAPIKey(key))

	sr, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "What is 7 * 8? Reply with just the number."}}}},
		MaxOutputTokens: 500,
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"type":         "enabled",
				"budgetTokens": 300,
			},
		},
	})
	if err != nil {
		skipAPIError(t, err)
		t.Fatal(err)
	}

	var texts, reasoning []string
	for chunk := range sr.Stream {
		if chunk.Type == provider.ChunkText {
			texts = append(texts, chunk.Text)
		}
		if chunk.Type == provider.ChunkReasoning {
			reasoning = append(reasoning, chunk.Text)
		}
	}
	joined := strings.Join(texts, "")
	if !strings.Contains(joined, "56") {
		t.Errorf("stream text = %q, want contains '56'", joined)
	}
	if len(reasoning) == 0 {
		t.Error("expected reasoning chunks")
	}
	t.Logf("reasoning chunks=%d, text=%q", len(reasoning), joined)
}
