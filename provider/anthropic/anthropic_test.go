package anthropic

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

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// --- Streaming tests ---

func TestChat_Stream_TextResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("unexpected auth: %s", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != apiVersion {
			t.Errorf("unexpected version: %s", r.Header.Get("anthropic-version"))
		}
		if r.Header.Get("anthropic-beta") != betaFeatures {
			t.Errorf("unexpected beta: %s", r.Header.Get("anthropic-beta"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["model"] != "claude-sonnet-4-20250514" {
			t.Errorf("model = %v, want claude-sonnet-4-20250514", req["model"])
		}
		if req["stream"] != true {
			t.Errorf("stream = %v, want true", req["stream"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"id":"msg_stream_1","model":"claude-sonnet-4-20250514","usage":{"input_tokens":15,"cache_read_input_tokens":5,"cache_creation_input_tokens":2}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	// text + text + step_finish + finish = 4
	if len(chunks) != 4 {
		t.Fatalf("expected 4 chunks, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Type != provider.ChunkText || chunks[0].Text != "Hello" {
		t.Errorf("chunks[0] = %+v, want text Hello", chunks[0])
	}
	if chunks[1].Type != provider.ChunkText || chunks[1].Text != " world" {
		t.Errorf("chunks[1] = %+v, want text  world", chunks[1])
	}
	if chunks[2].Type != provider.ChunkStepFinish || chunks[2].FinishReason != provider.FinishStop {
		t.Errorf("chunks[2] = %+v, want step_finish stop", chunks[2])
	}
	if chunks[3].Type != provider.ChunkFinish {
		t.Errorf("chunks[3] = %+v, want finish", chunks[3])
	}
	// Check usage on finish chunk.
	if chunks[3].Usage.InputTokens != 15 {
		t.Errorf("InputTokens = %d, want 15", chunks[3].Usage.InputTokens)
	}
	if chunks[3].Usage.OutputTokens != 8 {
		t.Errorf("OutputTokens = %d, want 8", chunks[3].Usage.OutputTokens)
	}
	if chunks[3].Usage.CacheReadTokens != 5 {
		t.Errorf("CacheReadTokens = %d, want 5", chunks[3].Usage.CacheReadTokens)
	}
	if chunks[3].Usage.CacheWriteTokens != 2 {
		t.Errorf("CacheWriteTokens = %d, want 2", chunks[3].Usage.CacheWriteTokens)
	}
	if chunks[3].Response.ID != "msg_stream_1" {
		t.Errorf("Response.ID = %q, want %q", chunks[3].Response.ID, "msg_stream_1")
	}
	if chunks[3].Response.Model != "claude-sonnet-4-20250514" {
		t.Errorf("Response.Model = %q, want %q", chunks[3].Response.Model, "claude-sonnet-4-20250514")
	}
}

func TestChat_Stream_ToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"read_file"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":": \"test.go\"}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read test.go"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	// start + delta + delta + tool_call (accumulated) + step_finish + finish = 6
	if len(chunks) != 6 {
		t.Fatalf("expected 6 chunks, got %d", len(chunks))
	}
	if chunks[0].Type != provider.ChunkToolCallStreamStart {
		t.Errorf("chunks[0].Type = %s, want tool_call_streaming_start", chunks[0].Type)
	}
	if chunks[0].ToolCallID != "toolu_123" || chunks[0].ToolName != "read_file" {
		t.Errorf("chunks[0] = %+v, want toolu_123/read_file", chunks[0])
	}
	// Two partial deltas streamed for UI progress.
	if chunks[1].Type != provider.ChunkToolCallDelta || chunks[1].ToolInput != `{"path"` {
		t.Errorf("chunks[1] = %+v, want tool_call_delta with first partial", chunks[1])
	}
	if chunks[2].Type != provider.ChunkToolCallDelta || chunks[2].ToolInput != `: "test.go"}` {
		t.Errorf("chunks[2] = %+v, want tool_call_delta with second partial", chunks[2])
	}
	// Accumulated complete JSON emitted on content_block_stop.
	if chunks[3].Type != provider.ChunkToolCall || chunks[3].ToolInput != `{"path": "test.go"}` {
		t.Errorf("chunks[3] = %+v, want tool_call with complete json", chunks[3])
	}
	if chunks[4].FinishReason != provider.FinishToolCalls {
		t.Errorf("finish reason = %s, want tool_calls", chunks[4].FinishReason)
	}
}

func TestChat_Stream_Reasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify thinking config is in the request body.
		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]any
		if err := json.Unmarshal(body, &reqBody); err != nil {
			t.Errorf("failed to parse request body: %v", err)
		}
		thinking, ok := reqBody["thinking"].(map[string]any)
		if !ok {
			t.Errorf("thinking not in request body: %v", reqBody)
		} else {
			if thinking["type"] != "enabled" {
				t.Errorf("thinking.type = %v, want enabled", thinking["type"])
			}
			if thinking["budget_tokens"] != float64(16000) {
				t.Errorf("thinking.budget_tokens = %v, want 16000", thinking["budget_tokens"])
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":1}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think about this"}}},
		},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"type":         "enabled",
				"budgetTokens": 16000,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	// reasoning + text + step_finish + finish = 4
	if len(chunks) != 4 {
		t.Fatalf("expected 4 chunks, got %d", len(chunks))
	}
	if chunks[0].Type != provider.ChunkReasoning || chunks[0].Text != "Let me think..." {
		t.Errorf("chunks[0] = %+v, want reasoning", chunks[0])
	}
	if chunks[1].Type != provider.ChunkText || chunks[1].Text != "The answer is 42" {
		t.Errorf("chunks[1] = %+v, want text", chunks[1])
	}
}

func TestChat_Stream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	// Find error chunk.
	found := false
	for _, chunk := range chunks {
		if chunk.Type == provider.ChunkError {
			found = true
			if chunk.Error == nil {
				t.Error("expected error, got nil")
			}
		}
	}
	if !found {
		t.Error("expected error chunk, none found")
	}
}

func TestChat_Stream_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"type":"rate_limit_error","message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestChat_Stream_ContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		// Don't send message_stop -- let context cancel handle it.
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(t.Context())
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(ctx, provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	cancel()
	// Drain to ensure the goroutine exits.
	for range result.Stream {
	}
}

// --- Non-streaming tests ---

func TestChat_Generate_TextResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["stream"] != false {
			t.Errorf("stream = %v, want false", req["stream"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "msg_123",
			"model": "claude-sonnet-4-20250514",
			"type": "message",
			"content": [{"type": "text", "text": "Hello world"}],
			"stop_reason": "end_turn",
			"usage": {
				"input_tokens": 15,
				"output_tokens": 8,
				"cache_read_input_tokens": 5,
				"cache_creation_input_tokens": 2
			}
		}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want %q", result.Text, "Hello world")
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %s, want stop", result.FinishReason)
	}
	if result.Usage.InputTokens != 15 {
		t.Errorf("InputTokens = %d, want 15", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 8 {
		t.Errorf("OutputTokens = %d, want 8", result.Usage.OutputTokens)
	}
	if result.Usage.CacheReadTokens != 5 {
		t.Errorf("CacheReadTokens = %d, want 5", result.Usage.CacheReadTokens)
	}
	if result.Usage.CacheWriteTokens != 2 {
		t.Errorf("CacheWriteTokens = %d, want 2", result.Usage.CacheWriteTokens)
	}
	if result.Response.ID != "msg_123" {
		t.Errorf("Response.ID = %q, want %q", result.Response.ID, "msg_123")
	}
}

func TestChat_Generate_ToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "msg_456",
			"model": "claude-sonnet-4-20250514",
			"type": "message",
			"content": [
				{"type": "text", "text": "Let me read that file."},
				{"type": "tool_use", "id": "toolu_789", "name": "read_file", "input": {"path": "test.go"}}
			],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 20, "output_tokens": 15}
		}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read test.go"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "Let me read that file." {
		t.Errorf("Text = %q, want %q", result.Text, "Let me read that file.")
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %s, want tool_calls", result.FinishReason)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
	if result.ToolCalls[0].ID != "toolu_789" {
		t.Errorf("ToolCall.ID = %q, want %q", result.ToolCalls[0].ID, "toolu_789")
	}
	if result.ToolCalls[0].Name != "read_file" {
		t.Errorf("ToolCall.Name = %q, want %q", result.ToolCalls[0].Name, "read_file")
	}
}

func TestChat_Generate_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"type":"invalid_request_error","message":"Invalid request"}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

// --- Request building tests ---

func TestBuildRequest_System(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		System: "You are helpful.",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	}, true)

	system, ok := body["system"].([]map[string]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system = %v, want array of 1", body["system"])
	}
	if system[0]["text"] != "You are helpful." {
		t.Errorf("system text = %v, want 'You are helpful.'", system[0]["text"])
	}
	// No cache_control by default -- only when PromptCaching is enabled.
	if _, ok := system[0]["cache_control"]; ok {
		t.Errorf("cache_control should not be set without PromptCaching, got %v", system[0]["cache_control"])
	}
}

func TestBuildRequest_SystemCacheControl(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		System: "You are helpful.",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		PromptCaching: true,
	}, true)

	system, ok := body["system"].([]map[string]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system = %v, want array of 1", body["system"])
	}
	cc, _ := system[0]["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control = %v, want ephemeral", cc)
	}
}

func TestBuildRequest_Tools(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "read_file", Description: "Read a file", InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)},
		},
	}, false)

	tools, ok := body["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want array of 1", body["tools"])
	}
	if tools[0]["name"] != "read_file" {
		t.Errorf("tool name = %v, want read_file", tools[0]["name"])
	}
}

func TestBuildRequest_Temperature(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	temp := 0.7
	body := m.buildRequest(provider.GenerateParams{
		Messages:    []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Temperature: &temp,
	}, false)

	if body["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", body["temperature"])
	}
}

func TestBuildRequest_MaxTokens(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}

	// Default.
	body := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	}, false)
	if body["max_tokens"] != defaultMaxTokens {
		t.Errorf("max_tokens = %v, want %d", body["max_tokens"], defaultMaxTokens)
	}

	// Custom.
	body = m.buildRequest(provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		MaxOutputTokens: 4096,
	}, false)
	if body["max_tokens"] != 4096 {
		t.Errorf("max_tokens = %v, want 4096", body["max_tokens"])
	}
}

func TestBuildRequest_ToolChoice(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}

	tests := []struct {
		choice string
		want   any
	}{
		{"auto", map[string]any{"type": "auto"}},
		{"required", map[string]any{"type": "any"}},
		{"read_file", map[string]any{"type": "tool", "name": "read_file"}},
	}

	for _, tt := range tests {
		body := m.buildRequest(provider.GenerateParams{
			Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
			ToolChoice: tt.choice,
		}, false)

		got, _ := json.Marshal(body["tool_choice"])
		want, _ := json.Marshal(tt.want)
		if string(got) != string(want) {
			t.Errorf("tool_choice(%s) = %s, want %s", tt.choice, got, want)
		}
	}

	// "none" should not set tool_choice.
	body := m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "none",
	}, false)
	if body["tool_choice"] != nil {
		t.Errorf("tool_choice(none) = %v, want nil", body["tool_choice"])
	}
}

// --- Message conversion tests ---

func TestConvertMessages_Text(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "hi there"}}},
	})

	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want 2", len(msgs))
	}
	if msgs[0]["role"] != "user" {
		t.Errorf("msgs[0].role = %v, want user", msgs[0]["role"])
	}
	content := msgs[0]["content"].([]map[string]any)
	if content[0]["text"] != "hello" {
		t.Errorf("content[0].text = %v, want hello", content[0]["text"])
	}
}

func TestConvertMessages_SkipsSystem(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleSystem, Content: []provider.Part{{Type: provider.PartText, Text: "system prompt"}}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
	})

	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1 (system should be skipped)", len(msgs))
	}
}

func TestConvertMessages_SkipsEmptyContent(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: ""}}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
	})

	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1 (empty content should be skipped)", len(msgs))
	}
}

func TestConvertMessages_ToolCallAndResult(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolCallID: "toolu_1", ToolName: "read", ToolInput: json.RawMessage(`{"path":"a.go"}`)},
		}},
		{Role: provider.RoleTool, Content: []provider.Part{
			{Type: provider.PartToolResult, ToolCallID: "toolu_1", ToolOutput: "file contents"},
		}},
	})

	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want 2", len(msgs))
	}

	// Assistant message with tool_use.
	aContent := msgs[0]["content"].([]map[string]any)
	if aContent[0]["type"] != "tool_use" {
		t.Errorf("assistant content type = %v, want tool_use", aContent[0]["type"])
	}
	if aContent[0]["id"] != "toolu_1" {
		t.Errorf("tool_use id = %v, want toolu_1", aContent[0]["id"])
	}

	// Tool message should have role "user" (Anthropic API requires it).
	if msgs[1]["role"] != "user" {
		t.Errorf("tool message role = %v, want user", msgs[1]["role"])
	}
	tContent := msgs[1]["content"].([]map[string]any)
	if tContent[0]["type"] != "tool_result" {
		t.Errorf("tool content type = %v, want tool_result", tContent[0]["type"])
	}
	if tContent[0]["tool_use_id"] != "toolu_1" {
		t.Errorf("tool_use_id = %v, want toolu_1", tContent[0]["tool_use_id"])
	}
}

func TestConvertMessages_Reasoning(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartReasoning, Text: "thinking...", ProviderOptions: map[string]any{"signature": "sig123"}},
			{Type: provider.PartText, Text: "answer"},
		}},
	})

	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 2 {
		t.Fatalf("got %d parts, want 2", len(content))
	}
	if content[0]["type"] != "thinking" {
		t.Errorf("part[0].type = %v, want thinking", content[0]["type"])
	}
	if content[0]["thinking"] != "thinking..." {
		t.Errorf("part[0].thinking = %v, want thinking...", content[0]["thinking"])
	}
	if content[0]["signature"] != "sig123" {
		t.Errorf("part[0].signature = %v, want sig123", content[0]["signature"])
	}
}

func TestConvertMessages_ReasoningWithoutSignature(t *testing.T) {
	// Reasoning from other providers (e.g. Gemini) may lack signature.
	// These should be skipped to avoid API validation errors.
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartReasoning, Text: "gemini thinking"},
			{Type: provider.PartText, Text: "answer"},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("got %d parts, want 1 (reasoning without signature skipped)", len(content))
	}
	if content[0]["type"] != "text" {
		t.Errorf("part[0].type = %v, want text", content[0]["type"])
	}
}

func TestConvertMessages_CacheControl(t *testing.T) {
	// Part-level cache control.
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartText, Text: "hello", CacheControl: "ephemeral"},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	cc, _ := content[0]["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control = %v, want ephemeral", cc)
	}

	// Message-level cache control on last part.
	msgs = convertMessages([]provider.Message{
		{
			Role: provider.RoleUser,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "first"},
				{Type: provider.PartText, Text: "second"},
			},
			ProviderOptions: map[string]any{
				"anthropic": map[string]any{
					"cacheControl": map[string]any{"type": "ephemeral"},
				},
			},
		},
	})
	content = msgs[0]["content"].([]map[string]any)
	// First part should NOT have cache_control.
	if content[0]["cache_control"] != nil {
		t.Errorf("first part should not have cache_control, got %v", content[0]["cache_control"])
	}
	// Last part should have cache_control.
	cc, _ = content[1]["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Errorf("last part cache_control = %v, want ephemeral", cc)
	}
}

func TestConvertMessages_Image(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartImage, URL: "data:image/png;base64,iVBORw0KGgo="},
		}},
	})

	content := msgs[0]["content"].([]map[string]any)
	if content[0]["type"] != "image" {
		t.Errorf("type = %v, want image", content[0]["type"])
	}
	source, _ := content[0]["source"].(map[string]any)
	if source["type"] != "base64" {
		t.Errorf("source.type = %v, want base64", source["type"])
	}
	if source["media_type"] != "image/png" {
		t.Errorf("source.media_type = %v, want image/png", source["media_type"])
	}
	if source["data"] != "iVBORw0KGgo=" {
		t.Errorf("source.data = %v, want iVBORw0KGgo=", source["data"])
	}
}

func TestConvertMessages_File(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartFile, URL: "data:application/pdf;base64,JVBERi0="},
		}},
	})

	content := msgs[0]["content"].([]map[string]any)
	if content[0]["type"] != "document" {
		t.Errorf("type = %v, want document", content[0]["type"])
	}
	source, _ := content[0]["source"].(map[string]any)
	if source["media_type"] != "application/pdf" {
		t.Errorf("source.media_type = %v, want application/pdf", source["media_type"])
	}
}

// --- parseDataURL tests ---

func TestParseDataURL(t *testing.T) {
	tests := []struct {
		url       string
		mediaType string
		data      string
		ok        bool
	}{
		{"data:image/png;base64,abc123", "image/png", "abc123", true},
		{"data:application/pdf;base64,xyz", "application/pdf", "xyz", true},
		{"https://example.com/img.png", "", "", false},
		{"data:image/png;abc123", "", "", false},
	}

	for _, tt := range tests {
		mt, d, ok := httpc.ParseDataURL(tt.url)
		if ok != tt.ok || mt != tt.mediaType || d != tt.data {
			t.Errorf("ParseDataURL(%q) = (%q, %q, %v), want (%q, %q, %v)",
				tt.url, mt, d, ok, tt.mediaType, tt.data, tt.ok)
		}
	}
}

// --- mapFinishReason tests ---

func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		input string
		want  provider.FinishReason
	}{
		{"end_turn", provider.FinishStop},
		{"stop_sequence", provider.FinishStop},
		{"tool_use", provider.FinishToolCalls},
		{"max_tokens", provider.FinishLength},
		{"unknown", provider.FinishOther},
	}

	for _, tt := range tests {
		if got := mapFinishReason(tt.input); got != tt.want {
			t.Errorf("mapFinishReason(%q) = %s, want %s", tt.input, got, tt.want)
		}
	}
}

// --- Option tests ---

func TestWithHTTPClient(t *testing.T) {
	customClient := &http.Client{}
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("key"), WithHTTPClient(customClient))
	cm := model.(*chatModel)
	if cm.httpClient() != customClient {
		t.Error("WithHTTPClient should set custom client")
	}
}

func TestWithHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "value" {
			t.Errorf("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","model":"claude-sonnet-4-20250514","type":"message","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithHeaders(map[string]string{"X-Custom": "value"}),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

func TestWithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "dynamic-token" {
			t.Errorf("x-api-key = %q, want dynamic-token", r.Header.Get("x-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","model":"claude-sonnet-4-20250514","type":"message","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	ts := provider.StaticToken("dynamic-token")
	model := Chat("claude-sonnet-4-20250514", WithTokenSource(ts), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

func TestNoTokenSource(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514")
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for no token source")
	}
	if !strings.Contains(err.Error(), "no API key or token source") {
		t.Errorf("unexpected error: %s", err)
	}
}

// --- Capabilities test ---

func TestCapabilities(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("key"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature {
		t.Error("expected Temperature=true")
	}
	if !caps.Reasoning {
		t.Error("expected Reasoning=true")
	}
	if !caps.ToolCall {
		t.Error("expected ToolCall=true")
	}
	if !caps.Attachment {
		t.Error("expected Attachment=true")
	}
	if !caps.InputModalities.PDF {
		t.Error("expected InputModalities.PDF=true")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("key"))
	if model.ModelID() != "claude-sonnet-4-20250514" {
		t.Errorf("ModelID() = %q, want claude-sonnet-4-20250514", model.ModelID())
	}
}

// --- parseResponse tests ---

func TestParseResponse_ErrorInBody(t *testing.T) {
	body := `{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}`
	_, err := parseResponse([]byte(body))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestParseResponse_InvalidJSON(t *testing.T) {
	_, err := parseResponse([]byte("not json"))
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- StopSequences and TopP test ---

func TestBuildRequest_StopSequencesAndTopP(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	topP := 0.9
	body := m.buildRequest(provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		TopP:          &topP,
		StopSequences: []string{"END"},
	}, false)

	if body["top_p"] != 0.9 {
		t.Errorf("top_p = %v, want 0.9", body["top_p"])
	}
	stops, ok := body["stop_sequences"].([]string)
	if !ok || len(stops) != 1 || stops[0] != "END" {
		t.Errorf("stop_sequences = %v, want [END]", body["stop_sequences"])
	}
}

// --- Per-request headers test ---

func TestBuildRequest_PerRequestHeaders(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Headers:  map[string]string{"X-Custom": "value"},
	}, false)

	headers, ok := body["_headers"].(map[string]string)
	if !ok || headers["X-Custom"] != "value" {
		t.Errorf("_headers = %v, want X-Custom=value", body["_headers"])
	}
}

func TestBuildRequest_ThinkingEnabled(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{
				"type":         "enabled",
				"budgetTokens": 16000,
			},
		},
	}, true)

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not in request body")
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want enabled", thinking["type"])
	}
	if thinking["budget_tokens"] != 16000 {
		t.Errorf("thinking.budget_tokens = %v, want 16000", thinking["budget_tokens"])
	}
}

func TestBuildRequest_ThinkingAdaptive(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-6-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{"type": "adaptive"},
		},
	}, true)

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not in request body")
	}
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking.type = %v, want adaptive", thinking["type"])
	}
	if _, hasBudget := thinking["budget_tokens"]; hasBudget {
		t.Error("adaptive thinking should not have budget_tokens")
	}
}

func TestBuildRequest_ProviderOptionsPassthrough(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"custom_field": "custom_value",
		},
	}, true)

	if body["custom_field"] != "custom_value" {
		t.Errorf("custom_field = %v, want custom_value", body["custom_field"])
	}
}

// --- Additional coverage tests ---

func TestConvertMessages_InvalidImageURL(t *testing.T) {
	// Image with invalid URL should be skipped.
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartImage, URL: "https://example.com/img.png"},
			{Type: provider.PartText, Text: "hello"},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content part (image skipped), got %d", len(content))
	}
}

func TestConvertMessages_EmptyImageURL(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartImage, URL: ""},
			{Type: provider.PartText, Text: "hello"},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content part (empty image skipped), got %d", len(content))
	}
}

func TestConvertMessages_EmptyFileURL(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartFile, URL: ""},
			{Type: provider.PartText, Text: "hello"},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content part (empty file skipped), got %d", len(content))
	}
}

func TestConvertMessages_InvalidFileURL(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartFile, URL: "https://example.com/file.pdf"},
			{Type: provider.PartText, Text: "hello"},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content part (invalid file URL skipped), got %d", len(content))
	}
}

func TestConvertMessages_NilToolInput(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolCallID: "toolu_1", ToolName: "test", ToolInput: nil},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	// With nil ToolInput, input should default to empty object.
	input := content[0]["input"]
	if input == nil {
		t.Error("expected non-nil input for nil ToolInput")
	}
}

func TestConvertMessages_InvalidJSONToolInput(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolCallID: "toolu_1", ToolName: "test", ToolInput: json.RawMessage(`{invalid}`)},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	// With invalid JSON ToolInput, input should fall back to empty object.
	input, ok := content[0]["input"].(map[string]any)
	if !ok {
		t.Fatal("expected map[string]any input on invalid JSON")
	}
	if len(input) != 0 {
		t.Errorf("expected empty map, got %v", input)
	}
}

func TestConvertMessages_EmptyReasoningSkipped(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartReasoning, Text: ""},
			{Type: provider.PartText, Text: "answer"},
		}},
	})
	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content part (empty reasoning skipped), got %d", len(content))
	}
}

func TestParseResponse_OverflowError(t *testing.T) {
	body := `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 200000 tokens > 100000 maximum"}}`
	_, err := parseResponse([]byte(body))
	if err == nil {
		t.Fatal("expected error")
	}
	// Should be detected as overflow.
	var overflow *goai.ContextOverflowError
	if !errors.As(err, &overflow) {
		t.Errorf("expected ContextOverflowError, got %T: %v", err, err)
	}
}

func TestParseResponse_TypeErrorNoErrorField(t *testing.T) {
	body := `{"type":"error"}`
	_, err := parseResponse([]byte(body))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChat_Stream_ContextOverflowError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 200000 tokens > 100000 maximum"}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	found := false
	for _, c := range chunks {
		if c.Type == provider.ChunkError {
			found = true
			var overflow *goai.ContextOverflowError
			if !errors.As(c.Error, &overflow) {
				t.Errorf("expected ContextOverflowError, got %T: %v", c.Error, c.Error)
			}
		}
	}
	if !found {
		t.Error("expected overflow error chunk")
	}
}

func TestChat_Stream_ScannerError(t *testing.T) {
	// Create a server that closes connection mid-stream.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		// Close connection abruptly (no message_stop).
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Drain -- should not hang.
	for range result.Stream {
	}
}

func TestNoTokenSource_Stream(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514")
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for no token source")
	}
}

func TestChat_Stream_ClassifyStreamError_ContextOverflow(t *testing.T) {
	// This tests the ClassifyStreamError path with a known error code.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"error","error":{"code":"context_length_exceeded","message":"Context too long"}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var foundOverflow bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var overflow *goai.ContextOverflowError
			if errors.As(chunk.Error, &overflow) {
				foundOverflow = true
			}
		}
	}
	if !foundOverflow {
		t.Error("expected ContextOverflowError from ClassifyStreamError")
	}
}

func TestChat_Stream_ClassifyStreamError_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"error","error":{"code":"insufficient_quota","message":"Quota exceeded"}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var foundAPIError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var apiErr *goai.APIError
			if errors.As(chunk.Error, &apiErr) {
				foundAPIError = true
			}
		}
	}
	if !foundAPIError {
		t.Error("expected APIError from ClassifyStreamError")
	}
}

func TestChat_Stream_ErrorNoMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Error event with empty error object -- triggers "unknown stream error".
		_, _ = fmt.Fprint(w, `data: {"type":"error","error":{}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var found bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError && chunk.Error != nil {
			found = true
			if !strings.Contains(chunk.Error.Error(), "unknown stream error") {
				t.Errorf("expected 'unknown stream error', got %q", chunk.Error.Error())
			}
		}
	}
	if !found {
		t.Error("expected error chunk")
	}
}

func TestChat_Stream_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: not valid json\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
	// Should still get the text despite invalid JSON line.
	if len(texts) != 1 || texts[0] != "ok" {
		t.Errorf("texts = %v, want [ok]", texts)
	}
}

func TestDoHTTP_PerRequestHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom-Per-Request") != "value" {
			t.Errorf("missing per-request header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","model":"claude-sonnet-4-20250514","type":"message","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{"X-Custom-Per-Request": "value"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestParseSSE_ScannerError(t *testing.T) {
	// Test parseSSE with a reader that produces a scanner error.
	out := make(chan provider.StreamChunk, 64)
	// Create a reader with a line exceeding buffer.
	longLine := "data: " + strings.Repeat("x", 2*1024*1024) + "\n"
	go parseSSE(t.Context(), strings.NewReader(longLine), out, false)

	var foundError bool
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			foundError = true
		}
	}
	if !foundError {
		t.Error("expected error chunk from scanner overflow")
	}
}

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
	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("test-key"),
		WithBaseURL("http://localhost"),
		WithHTTPClient(&http.Client{Transport: &errBodyTransport{}}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error from read failure")
	}
	if !strings.Contains(err.Error(), "reading response") {
		t.Errorf("expected 'reading response' error, got: %v", err)
	}
}

func TestDoGenerate_ParseResponseError(t *testing.T) {
	// Server returns 200 OK with invalid JSON -- io.ReadAll succeeds but parseResponse fails.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `not valid json at all`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error from invalid response JSON")
	}
}

func TestDoHTTP_ConnectionError(t *testing.T) {
	// Use a URL that will fail to connect.
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected connection error")
	}
	if !strings.Contains(err.Error(), "sending request") {
		t.Errorf("unexpected error: %s", err)
	}
}

// --- Response Format (RF mode) tests ---

func TestInjectResponseFormatTool(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`)
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "existing_tool", Description: "existing", InputSchema: json.RawMessage(`{}`)},
		},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test_schema",
			Schema: schema,
		},
	}

	result := injectResponseFormatTool(params)

	// Should prepend the synthetic tool.
	if len(result.Tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(result.Tools))
	}
	if result.Tools[0].Name != responseFormatToolName {
		t.Errorf("first tool name = %q, want %q", result.Tools[0].Name, responseFormatToolName)
	}
	if string(result.Tools[0].InputSchema) != string(schema) {
		t.Errorf("schema = %s, want %s", result.Tools[0].InputSchema, schema)
	}
	if result.Tools[1].Name != "existing_tool" {
		t.Errorf("second tool name = %q, want existing_tool", result.Tools[1].Name)
	}
	// Should force tool_choice to the synthetic tool name.
	if result.ToolChoice != responseFormatToolName {
		t.Errorf("ToolChoice = %q, want %q", result.ToolChoice, responseFormatToolName)
	}
	// Original params should be unmodified.
	if len(params.Tools) != 1 {
		t.Errorf("original params.Tools modified: len=%d", len(params.Tools))
	}
}

func TestExtractResponseFormatResult(t *testing.T) {
	result := &provider.GenerateResult{
		ToolCalls: []provider.ToolCall{
			{ID: "toolu_1", Name: responseFormatToolName, Input: json.RawMessage(`{"name":"Alice"}`)},
		},
		FinishReason: provider.FinishToolCalls,
	}

	extractResponseFormatResult(result)

	if result.Text != `{"name":"Alice"}` {
		t.Errorf("Text = %q, want %q", result.Text, `{"name":"Alice"}`)
	}
	if len(result.ToolCalls) != 0 {
		t.Errorf("ToolCalls should be empty, got %d", len(result.ToolCalls))
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
}

func TestExtractResponseFormatResult_WithOtherToolCalls(t *testing.T) {
	result := &provider.GenerateResult{
		ToolCalls: []provider.ToolCall{
			{ID: "toolu_1", Name: "real_tool", Input: json.RawMessage(`{"x":1}`)},
			{ID: "toolu_2", Name: responseFormatToolName, Input: json.RawMessage(`{"name":"Bob"}`)},
		},
		FinishReason: provider.FinishToolCalls,
	}

	extractResponseFormatResult(result)

	if result.Text != `{"name":"Bob"}` {
		t.Errorf("Text = %q, want %q", result.Text, `{"name":"Bob"}`)
	}
	// Should keep the real tool call.
	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 remaining tool call, got %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "real_tool" {
		t.Errorf("remaining tool = %q, want real_tool", result.ToolCalls[0].Name)
	}
	// FinishReason stays tool_calls since there are still real tool calls.
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishToolCalls)
	}
}

func TestExtractResponseFormatResult_NoMatch(t *testing.T) {
	result := &provider.GenerateResult{
		ToolCalls: []provider.ToolCall{
			{ID: "toolu_1", Name: "other_tool", Input: json.RawMessage(`{}`)},
		},
		FinishReason: provider.FinishToolCalls,
	}

	extractResponseFormatResult(result)

	// Nothing should change.
	if result.Text != "" {
		t.Errorf("Text should be empty, got %q", result.Text)
	}
	if len(result.ToolCalls) != 1 {
		t.Errorf("ToolCalls should still have 1, got %d", len(result.ToolCalls))
	}
}

func TestDoGenerate_WithResponseFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Verify the synthetic tool was injected.
		tools, ok := req["tools"].([]any)
		if !ok || len(tools) == 0 {
			t.Errorf("expected tools in request")
		}
		firstTool, _ := tools[0].(map[string]any)
		if firstTool["name"] != responseFormatToolName {
			t.Errorf("first tool = %v, want %s", firstTool["name"], responseFormatToolName)
		}
		// Verify tool_choice is forced.
		tc, _ := req["tool_choice"].(map[string]any)
		if tc["type"] != "tool" || tc["name"] != responseFormatToolName {
			t.Errorf("tool_choice = %v, want type=tool name=%s", tc, responseFormatToolName)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "msg_rf1",
			"model": "claude-sonnet-4-20250514",
			"type": "message",
			"content": [
				{
					"type": "tool_use",
					"id": "toolu_rf1",
					"name": "json_response",
					"input": {"name": "Alice", "age": 30}
				}
			],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 20, "output_tokens": 15}
		}`)
	}))
	defer server.Close()

	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}`)
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "Return name and age"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "person",
			Schema: schema,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Text should be set from the tool input.
	if result.Text == "" {
		t.Fatal("expected non-empty Text from RF extraction")
	}
	// Should contain the JSON fields.
	if !strings.Contains(result.Text, "Alice") {
		t.Errorf("Text should contain Alice, got %q", result.Text)
	}
	// No tool calls should remain.
	if len(result.ToolCalls) != 0 {
		t.Errorf("expected 0 ToolCalls, got %d", len(result.ToolCalls))
	}
	// FinishReason should be mapped to stop.
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
}

func TestDoStream_WithResponseFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Verify the synthetic tool was injected.
		tools, ok := req["tools"].([]any)
		if !ok || len(tools) == 0 {
			t.Errorf("expected tools in request")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		// message_start
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":20}}}`+"\n\n")
		// content_block_start: tool_use with the synthetic json_response tool -- should be suppressed
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_rf1","name":"json_response"}}`+"\n\n")
		// content_block_delta: input_json_delta -- should emit ChunkText, not ChunkToolCall
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"name\""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":\"Alice\",\"age\":30}"}}`+"\n\n")
		// content_block_stop
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		// message_delta with stop_reason=tool_use -- should map to stop, not tool_calls
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}`+"\n\n")
		// message_stop
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}`)
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "Return name and age"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "person",
			Schema: schema,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	// Verify: NO ChunkToolCallStreamStart (the synthetic tool block is suppressed).
	for _, c := range chunks {
		if c.Type == provider.ChunkToolCallStreamStart {
			t.Errorf("unexpected ChunkToolCallStreamStart -- RF mode should suppress synthetic tool start")
		}
		if c.Type == provider.ChunkToolCall {
			t.Errorf("unexpected ChunkToolCall -- RF mode should emit ChunkText instead")
		}
	}

	// Verify: input_json_delta emitted as ChunkText.
	var textParts []string
	for _, c := range chunks {
		if c.Type == provider.ChunkText {
			textParts = append(textParts, c.Text)
		}
	}
	if len(textParts) == 0 {
		t.Fatal("expected ChunkText chunks from RF mode input_json_delta")
	}
	combined := strings.Join(textParts, "")
	if !strings.Contains(combined, "Alice") {
		t.Errorf("text = %q, expected to contain Alice", combined)
	}

	// Verify: stop_reason=tool_use mapped to FinishStop (not FinishToolCalls).
	var stepFinishReason provider.FinishReason
	for _, c := range chunks {
		if c.Type == provider.ChunkStepFinish {
			stepFinishReason = c.FinishReason
		}
	}
	if stepFinishReason != provider.FinishStop {
		t.Errorf("step finish reason = %q, want %q", stepFinishReason, provider.FinishStop)
	}

	// Verify: ChunkFinish present with usage.
	var finishFound bool
	for _, c := range chunks {
		if c.Type == provider.ChunkFinish {
			finishFound = true
			if c.Usage.InputTokens != 20 {
				t.Errorf("InputTokens = %d, want 20", c.Usage.InputTokens)
			}
			if c.Usage.OutputTokens != 10 {
				t.Errorf("OutputTokens = %d, want 10", c.Usage.OutputTokens)
			}
		}
	}
	if !finishFound {
		t.Error("expected ChunkFinish")
	}
}

func TestDoStream_WithResponseFormat_MixedWithRealTool(t *testing.T) {
	// Tests RF mode when there's also a real tool_use block alongside the synthetic one.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		// Real tool block -- should emit normally.
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_real","name":"calculator"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"expr\":\"1+1\"}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		// Synthetic RF tool block -- should be suppressed.
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_rf","name":"json_response"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"result\":42}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":1}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	schema := json.RawMessage(`{"type":"object","properties":{"result":{"type":"integer"}}}`)
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "calc"}}},
		},
		ResponseFormat: &provider.ResponseFormat{Name: "result", Schema: schema},
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	// Real tool should produce ChunkToolCallStreamStart and ChunkToolCall.
	var toolStartCount, toolCallCount, textCount int
	for _, c := range chunks {
		switch c.Type {
		case provider.ChunkToolCallStreamStart:
			toolStartCount++
			if c.ToolName != "calculator" {
				t.Errorf("tool start name = %q, want calculator", c.ToolName)
			}
		case provider.ChunkToolCall:
			toolCallCount++
			if c.ToolName != "calculator" {
				t.Errorf("tool call name = %q, want calculator", c.ToolName)
			}
		case provider.ChunkText:
			textCount++
		}
	}

	if toolStartCount != 1 {
		t.Errorf("expected 1 tool start (real), got %d", toolStartCount)
	}
	if toolCallCount != 1 {
		t.Errorf("expected 1 tool call (real), got %d", toolCallCount)
	}
	if textCount != 1 {
		t.Errorf("expected 1 text chunk (RF synthetic), got %d", textCount)
	}
}

// --- Coverage gap tests ---

// DoStream with native output format (covers line 150-151: useOutputFormat && ResponseFormat != nil).
func TestDoStream_NativeOutputFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Verify output_format is passed through (via ProviderOptions injection).
		if of, ok := req["output_format"]; !ok {
			t.Errorf("expected output_format in request, got none")
		} else {
			ofm := of.(map[string]any)
			if ofm["type"] != "json_schema" {
				t.Errorf("output_format.type = %v, want json_schema", ofm["type"])
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"name\":\"test\"}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-5-20241022", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ResponseFormat: &provider.ResponseFormat{
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
		ProviderOptions: map[string]any{
			"structuredOutputMode": "outputFormat",
		},
	})
	if err != nil {
		t.Fatalf("DoStream error: %v", err)
	}
	for range result.Stream {
	}
}

// injectNativeOutputFormat with nil ProviderOptions (covers line 571-573).
func TestInjectNativeOutputFormat_NilProviderOptions(t *testing.T) {
	params := provider.GenerateParams{
		ResponseFormat: &provider.ResponseFormat{
			Schema: json.RawMessage(`{"type":"object"}`),
		},
	}
	result := injectNativeOutputFormat(params)
	if result.ProviderOptions == nil {
		t.Fatal("expected ProviderOptions to be initialized")
	}
	of, ok := result.ProviderOptions["output_format"]
	if !ok {
		t.Fatal("output_format not set")
	}
	ofm := of.(map[string]any)
	if ofm["type"] != "json_schema" {
		t.Errorf("type = %v, want json_schema", ofm["type"])
	}
	if result.ResponseFormat != nil {
		t.Error("ResponseFormat should be cleared after injection")
	}
}

// injectNativeOutputFormat with invalid schema bytes (covers unmarshal error fallback).
func TestInjectNativeOutputFormat_InvalidSchema(t *testing.T) {
	params := provider.GenerateParams{
		ResponseFormat: &provider.ResponseFormat{
			Schema: json.RawMessage(`not valid json`),
		},
		ProviderOptions: map[string]any{"existing": "value"},
	}
	result := injectNativeOutputFormat(params)
	// Should return original params unchanged (fall back to tool trick).
	if result.ResponseFormat == nil {
		t.Error("ResponseFormat should NOT be cleared when schema is invalid")
	}
	if _, ok := result.ProviderOptions["output_format"]; ok {
		t.Error("output_format should NOT be set when schema is invalid")
	}
	// Original ProviderOptions should be preserved.
	if result.ProviderOptions["existing"] != "value" {
		t.Error("original ProviderOptions should be preserved")
	}
}

// buildRequest contextManagement with all sub-field variants (covers lines 337-348).
func TestBuildRequest_ContextManagementAllSubFields(t *testing.T) {
	model := &chatModel{id: "claude-sonnet-4-20250514"}
	body := model.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"contextManagement": map[string]any{
				"edits": []any{
					map[string]any{
						"type":         "clear_tool_uses_20250919",
						"clearAtLeast": float64(50000),
						"clearToolInputs": map[string]any{
							"type": "tool_input_clear_20250919",
						},
						"excludeTools":         []any{"bash"},
						"customUnknownField":    "passthrough",
					},
				},
			},
		},
	}, false)

	cm, ok := body["context_management"].(map[string]any)
	if !ok {
		t.Fatal("context_management not present in request")
	}
	edits, ok := cm["edits"].([]map[string]any)
	if !ok {
		t.Fatal("edits not present")
	}
	if len(edits) != 1 {
		t.Fatalf("expected 1 edit, got %d", len(edits))
	}
	edit := edits[0]
	if edit["clear_at_least"] != float64(50000) {
		t.Errorf("clear_at_least = %v, want 50000", edit["clear_at_least"])
	}
	if edit["clear_tool_inputs"] == nil {
		t.Error("clear_tool_inputs not set")
	}
	excludeTools := edit["exclude_tools"]
	if excludeTools == nil {
		t.Error("exclude_tools not set")
	}
	// The default case: unknown field passed through.
	if edit["customUnknownField"] != "passthrough" {
		t.Errorf("customUnknownField = %v, want passthrough", edit["customUnknownField"])
	}
}

// parseResponse with thinking and redacted_thinking blocks (covers lines 1054-1073).
func TestParseResponse_ThinkingAndRedactedThinking(t *testing.T) {
	body := `{
		"id": "msg_001",
		"model": "claude-sonnet-4-20250514",
		"type": "message",
		"content": [
			{"type": "thinking", "thinking": "Let me reason about this", "signature": "sig123"},
			{"type": "redacted_thinking", "data": "encrypted_data_here"},
			{"type": "text", "text": "The answer is 42"}
		],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50,
			"cache_read_input_tokens": 0,
			"cache_creation_input_tokens": 0,
			"output_tokens_details": {"thinking_tokens": 30}
		}
	}`

	result, err := parseResponse([]byte(body))
	if err != nil {
		t.Fatalf("parseResponse error: %v", err)
	}
	if result.Text != "The answer is 42" {
		t.Errorf("text = %q, want 'The answer is 42'", result.Text)
	}

	pm := result.Response.ProviderMetadata
	if pm == nil {
		t.Fatal("expected provider metadata")
	}
	reasoning, ok := pm["reasoning"].([]map[string]any)
	if !ok {
		t.Fatal("reasoning not found in provider metadata")
	}
	if len(reasoning) != 2 {
		t.Fatalf("expected 2 reasoning entries, got %d", len(reasoning))
	}
	// First entry: thinking.
	if reasoning[0]["type"] != "thinking" {
		t.Errorf("reasoning[0].type = %v, want thinking", reasoning[0]["type"])
	}
	if reasoning[0]["text"] != "Let me reason about this" {
		t.Errorf("reasoning[0].text = %v", reasoning[0]["text"])
	}
	if reasoning[0]["signature"] != "sig123" {
		t.Errorf("reasoning[0].signature = %v, want sig123", reasoning[0]["signature"])
	}
	// Second entry: redacted_thinking.
	if reasoning[1]["type"] != "redacted_thinking" {
		t.Errorf("reasoning[1].type = %v, want redacted_thinking", reasoning[1]["type"])
	}
	if reasoning[1]["data"] != "encrypted_data_here" {
		t.Errorf("reasoning[1].data = %v", reasoning[1]["data"])
	}
}

// parseResponse with only redacted_thinking (no prior thinking block -- covers providerMeta == nil branch).
func TestParseResponse_RedactedThinkingOnly(t *testing.T) {
	body := `{
		"id": "msg_003",
		"model": "claude-sonnet-4-20250514",
		"type": "message",
		"content": [
			{"type": "redacted_thinking", "data": "encrypted_only"},
			{"type": "text", "text": "answer"}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	result, err := parseResponse([]byte(body))
	if err != nil {
		t.Fatalf("parseResponse error: %v", err)
	}
	if result.Text != "answer" {
		t.Errorf("text = %q, want 'answer'", result.Text)
	}
	pm := result.Response.ProviderMetadata
	if pm == nil {
		t.Fatal("expected provider metadata")
	}
	reasoning, ok := pm["reasoning"].([]map[string]any)
	if !ok || len(reasoning) != 1 {
		t.Fatalf("expected 1 reasoning entry, got %v", reasoning)
	}
	if reasoning[0]["type"] != "redacted_thinking" {
		t.Errorf("type = %v, want redacted_thinking", reasoning[0]["type"])
	}
	if reasoning[0]["data"] != "encrypted_only" {
		t.Errorf("data = %v, want encrypted_only", reasoning[0]["data"])
	}
}

// parseResponse with page_location and char_location citations (covers lines 1034-1048).
func TestParseResponse_Citations_PageAndCharLocation(t *testing.T) {
	docTitle := "My Document"
	body := fmt.Sprintf(`{
		"id": "msg_002",
		"model": "claude-sonnet-4-20250514",
		"type": "message",
		"content": [
			{
				"type": "text",
				"text": "According to the document...",
				"citations": [
					{
						"type": "page_location",
						"cited_text": "important fact",
						"document_index": 0,
						"document_title": %q,
						"start_page_number": 5,
						"end_page_number": 6
					},
					{
						"type": "char_location",
						"cited_text": "another fact",
						"document_index": 1,
						"document_title": %q,
						"start_char_index": 100,
						"end_char_index": 200
					},
					{
						"type": "char_location",
						"cited_text": "no title citation",
						"document_index": 2,
						"start_char_index": 300,
						"end_char_index": 400
					}
				]
			}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 50, "output_tokens": 25}
	}`, docTitle, docTitle)

	result, err := parseResponse([]byte(body))
	if err != nil {
		t.Fatalf("parseResponse error: %v", err)
	}

	pm := result.Response.ProviderMetadata
	if pm == nil {
		t.Fatal("expected provider metadata")
	}
	citations, ok := pm["citations"].([]map[string]any)
	if !ok {
		t.Fatal("citations not found in provider metadata")
	}
	if len(citations) != 3 {
		t.Fatalf("expected 3 citations, got %d", len(citations))
	}

	// page_location citation.
	c0 := citations[0]
	if c0["type"] != "page_location" {
		t.Errorf("citation[0].type = %v, want page_location", c0["type"])
	}
	if c0["documentIndex"] != 0 {
		t.Errorf("citation[0].documentIndex = %v, want 0", c0["documentIndex"])
	}
	if c0["documentTitle"] != docTitle {
		t.Errorf("citation[0].documentTitle = %v, want %q", c0["documentTitle"], docTitle)
	}
	if c0["startPageNumber"] != 5 {
		t.Errorf("citation[0].startPageNumber = %v, want 5", c0["startPageNumber"])
	}
	if c0["endPageNumber"] != 6 {
		t.Errorf("citation[0].endPageNumber = %v, want 6", c0["endPageNumber"])
	}

	// char_location citation with title.
	c1 := citations[1]
	if c1["type"] != "char_location" {
		t.Errorf("citation[1].type = %v, want char_location", c1["type"])
	}
	if c1["documentTitle"] != docTitle {
		t.Errorf("citation[1].documentTitle = %v, want %q", c1["documentTitle"], docTitle)
	}
	if c1["startCharIndex"] != 100 {
		t.Errorf("citation[1].startCharIndex = %v, want 100", c1["startCharIndex"])
	}
	if c1["endCharIndex"] != 200 {
		t.Errorf("citation[1].endCharIndex = %v, want 200", c1["endCharIndex"])
	}

	// char_location citation without title (documentTitle should not be present).
	c2 := citations[2]
	if _, hasTitle := c2["documentTitle"]; hasTitle {
		t.Errorf("citation[2] should not have documentTitle when nil, got %v", c2["documentTitle"])
	}
}

func TestDoGenerate_ProviderDefinedToolBeta(t *testing.T) {
	var gotBeta string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBeta = r.Header.Get("anthropic-beta")

		// Verify tool body includes provider-defined type.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		tools, ok := req["tools"].([]any)
		if !ok || len(tools) == 0 {
			t.Fatal("expected tools in request")
		}
		tool0 := tools[0].(map[string]any)
		if tool0["type"] != "computer_20250124" {
			t.Errorf("tool type = %v, want computer_20250124", tool0["type"])
		}
		if tool0["display_width_px"] != float64(1920) {
			t.Errorf("display_width_px = %v", tool0["display_width_px"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-sonnet-4-20250514","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)

	computerTool := Tools.Computer(ComputerToolOptions{
		DisplayWidthPx: 1920, DisplayHeightPx: 1080,
	})

	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{computerTool},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Beta should include both base features AND computer-use beta.
	if !strings.Contains(gotBeta, "computer-use-2025-01-24") {
		t.Errorf("beta header should contain computer-use-2025-01-24, got %q", gotBeta)
	}
	if !strings.Contains(gotBeta, "claude-code-20250219") {
		t.Errorf("beta header should still contain base betas, got %q", gotBeta)
	}
}

func TestChat_EnvVarResolution(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	m := Chat("claude-sonnet-4-20250514")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource should be set from ANTHROPIC_API_KEY")
	}
}

func TestChat_EnvVarBaseURL(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	t.Setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")
	m := Chat("claude-sonnet-4-20250514")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://custom.anthropic.com" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestChat_EnvVarNotOverrideExplicit(t *testing.T) {
	t.Setenv("ANTHROPIC_BASE_URL", "https://env.url")
	m := Chat("claude-sonnet-4-20250514", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://explicit.url" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestConvertMessages_ThinkingWithSignature(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{
					Type: provider.PartReasoning,
					Text: "I am thinking...",
					ProviderOptions: map[string]any{
						"signature": "sig-abc123",
					},
				},
			},
		},
	}
	result := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}
	content, ok := result[0]["content"].([]map[string]any)
	if !ok {
		t.Fatal("content is not []map[string]any")
	}
	if len(content) != 1 {
		t.Fatalf("len(content) = %d, want 1", len(content))
	}
	if content[0]["type"] != "thinking" {
		t.Errorf("type = %q, want thinking", content[0]["type"])
	}
	if content[0]["signature"] != "sig-abc123" {
		t.Errorf("signature = %q, want sig-abc123", content[0]["signature"])
	}
}

func TestConvertMessages_RedactedThinking(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{
					Type: provider.PartReasoning,
					Text: "", // empty text = redacted thinking
					ProviderOptions: map[string]any{
						"redactedData": "encrypted-data-here",
					},
				},
			},
		},
	}
	result := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}
	content, ok := result[0]["content"].([]map[string]any)
	if !ok {
		t.Fatal("content is not []map[string]any")
	}
	if len(content) != 1 {
		t.Fatalf("len(content) = %d, want 1", len(content))
	}
	if content[0]["type"] != "redacted_thinking" {
		t.Errorf("type = %q, want redacted_thinking", content[0]["type"])
	}
	if content[0]["data"] != "encrypted-data-here" {
		t.Errorf("data = %q", content[0]["data"])
	}
}

func TestParseSSE_ContextCancel_AllBranches(t *testing.T) {
	// Exercise every TrySend early-return path in parseSSE with a cancelled
	// context and unbuffered channel. Each sub-test crafts SSE data so that
	// the target TrySend is the first one in the execution path.

	tests := []struct {
		name  string
		input string
	}{
		{
			// tool_use start (line 696)
			name: "tool_use_start",
			input: "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"fn\"}}\n",
		},
		{
			// server_tool_use start (line 710)
			name: "server_tool_use_start",
			input: "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"t2\",\"name\":\"bash_code_execution\"}}\n",
		},
		{
			// text_delta (line 727)
			name: "text_delta",
			input: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n",
		},
		{
			// thinking_delta (line 734)
			name: "thinking_delta",
			input: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n",
		},
		{
			// signature_delta (line 741)
			name: "signature_delta",
			input: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig123\"}}\n",
		},
		{
			// citations_delta (line 753)
			name: "citations_delta",
			input: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"citations_delta\",\"citation\":{\"url\":\"http://x\"}}}\n",
		},
		{
			// input_json_delta as text (RF mode handled separately -- normal tool delta, line 786)
			// This needs tool_use start first, which TrySends. Use pipe for that.
			// For pre-cancel: use isRFBlock=true path -- but parseSSE is called with isRFMode.
			// For RF mode text: just set isRFMode=true and the text TrySend at 768 fires.
			// But we can't control isRFMode here. Skip -- use a separate sub-test below.
			name: "message_stop",
			input: "data: {\"type\":\"message_stop\"}\n",
		},
		{
			// message_delta stop_reason (line 828)
			name: "message_delta",
			input: "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n",
		},
		{
			// error - overflow via ClassifyStreamError (line 916)
			name: "error_overflow",
			input: "data: {\"type\":\"error\",\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"too long\"}}\n",
		},
		{
			// error - overflow via IsOverflow (line 937)
			name: "error_overflow_msg",
			input: "data: {\"type\":\"error\",\"error\":{\"message\":\"prompt is too long\"}}\n",
		},
		{
			// error - generic (line 945)
			name: "error_generic",
			input: "data: {\"type\":\"error\",\"error\":{\"message\":\"some error\"}}\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			out := make(chan provider.StreamChunk) // unbuffered
			done := make(chan struct{})
			go func() {
				parseSSE(ctx, strings.NewReader(tc.input), out, false)
				close(done)
			}()
			<-done
			for range out {
			}
		})
	}

	// Nested TrySend paths that need prior sends to succeed.

	t.Run("tool_call_delta_cancel", func(t *testing.T) {
		// Goal: cover line 786 (ChunkToolCallDelta TrySend).
		// Needs tool_use start TrySend (696) to succeed first.
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"fn\"}}\n" +
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\"\"}}\n"

		done := make(chan struct{})
		go func() {
			parseSSE(ctx, strings.NewReader(input), out, false)
			close(done)
		}()

		<-out // receive tool start chunk
		cancel()
		<-done
		for range out {
		}
	})

	t.Run("content_block_stop_tool_cancel", func(t *testing.T) {
		// Goal: cover line 803 (ChunkToolCall at content_block_stop).
		// Needs tool_use start (696) + delta to succeed first.
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"fn\"}}\n" +
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n" +
			"data: {\"type\":\"content_block_stop\"}\n"

		done := make(chan struct{})
		go func() {
			parseSSE(ctx, strings.NewReader(input), out, false)
			close(done)
		}()

		<-out // tool start
		<-out // tool delta
		cancel()
		<-done
		for range out {
		}
	})

	t.Run("rf_mode_text_cancel", func(t *testing.T) {
		// Goal: cover line 768 (RF mode text TrySend).
		// In RF mode, tool_use with name=responseFormatTool doesn't emit start,
		// and input_json_delta emits ChunkText instead.
		ctx, cancel := context.WithCancel(t.Context())
		cancel() // cancel before parsing
		out := make(chan provider.StreamChunk) // unbuffered

		// content_block_start with tool_use named "json_response" sets isRFBlock=true,
		// skipping the tool start TrySend. Then input_json_delta emits text.
		input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"json_response\"}}\n" +
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"hello\"}}\n"

		done := make(chan struct{})
		go func() {
			parseSSE(ctx, strings.NewReader(input), out, true)
			close(done)
		}()
		<-done
		for range out {
		}
	})

	t.Run("scanner_error_cancel", func(t *testing.T) {
		// Goal: cover line 906 (scanner.Err() TrySend).
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		r := &slowErrorReader{
			data: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n",
			err:  fmt.Errorf("read fail"),
		}

		done := make(chan struct{})
		go func() {
			parseSSE(ctx, r, out, false)
			close(done)
		}()

		<-out // text chunk
		cancel()
		<-done
		for range out {
		}
	})

	t.Run("error_parsed_overflow_cancel", func(t *testing.T) {
		// Goal: cover line 919 (handleStreamError context_overflow TrySend).
		// Need to receive something before the error event.
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n" +
			"data: {\"type\":\"error\",\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"too long\"}}\n"

		done := make(chan struct{})
		go func() {
			parseSSE(ctx, strings.NewReader(input), out, false)
			close(done)
		}()

		<-out // text chunk
		cancel()
		<-done
		for range out {
		}
	})

	t.Run("error_parsed_other_cancel", func(t *testing.T) {
		// Goal: cover line 926 (handleStreamError non-overflow TrySend).
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n" +
			"data: {\"type\":\"error\",\"error\":{\"code\":\"insufficient_quota\",\"message\":\"quota exceeded\"}}\n"

		done := make(chan struct{})
		go func() {
			parseSSE(ctx, strings.NewReader(input), out, false)
			close(done)
		}()

		<-out // text chunk
		cancel()
		<-done
		for range out {
		}
	})
}

// --- Coverage gap tests ---

// buildRequest: TopK branch (covers the ~1% gap).
func TestBuildRequest_TopK(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	topK := 40
	body := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		TopK:     &topK,
	}, false)

	if body["top_k"] != 40 {
		t.Errorf("top_k = %v, want 40", body["top_k"])
	}
}

// injectNativeOutputFormat: nil ResponseFormat returns original params unchanged.
func TestInjectNativeOutputFormat_NilResponseFormat(t *testing.T) {
	params := provider.GenerateParams{
		ProviderOptions: map[string]any{"existing": "value"},
	}
	result := injectNativeOutputFormat(params)
	// Should return the original params (not the copy with cloned ProviderOptions).
	if result.ProviderOptions["existing"] != "value" {
		t.Error("should preserve original ProviderOptions")
	}
	if _, ok := result.ProviderOptions["output_format"]; ok {
		t.Error("should not set output_format when ResponseFormat is nil")
	}
}

// mapFinishReason: cover pause_turn, model_context_window_exceeded, and refusal.
func TestMapFinishReason_AllBranches(t *testing.T) {
	tests := []struct {
		input string
		want  provider.FinishReason
	}{
		{"pause_turn", provider.FinishStop},
		{"model_context_window_exceeded", provider.FinishLength},
		{"refusal", provider.FinishContentFilter},
	}
	for _, tt := range tests {
		if got := mapFinishReason(tt.input); got != tt.want {
			t.Errorf("mapFinishReason(%q) = %s, want %s", tt.input, got, tt.want)
		}
	}
}

// slowErrorReader returns data then an error on subsequent reads.
type slowErrorReader struct {
	data string
	pos  int
	err  error
}

func (r *slowErrorReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, r.err
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}
