package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/httpc"
	"github.com/zendev-sh/goai/provider"
)

// failReader is an io.ReadCloser that always fails on Read.
type failReader struct{}

func (f *failReader) Read(_ []byte) (int, error) { return 0, fmt.Errorf("read error") }
func (f *failReader) Close() error               { return nil }

// failTokenSource is a provider.TokenSource that always returns an error.
type failTokenSource struct{}

func (f failTokenSource) Token(_ context.Context) (string, error) {
	return "", fmt.Errorf("token error")
}

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

func TestChat_Stream_RedactedThinking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`+"\n\n")
		// Redacted thinking arrives as a complete block in content_block_start (no deltas).
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"encrypted-blob-xyz"}}`+"\n\n")
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
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	// The redacted_thinking block must surface as a ChunkReasoning carrying the
	// encrypted data so it can be replayed on a later turn.
	var redacted *provider.StreamChunk
	for i := range chunks {
		if chunks[i].Type == provider.ChunkReasoning {
			redacted = &chunks[i]
			break
		}
	}
	if redacted == nil {
		t.Fatalf("expected a reasoning chunk for redacted thinking, got %+v", chunks)
	}
	if redacted.Text != "" {
		t.Errorf("redacted reasoning chunk text = %q, want empty", redacted.Text)
	}
	if data, _ := redacted.Metadata["redactedData"].(string); data != "encrypted-blob-xyz" {
		t.Errorf("redactedData = %v, want encrypted-blob-xyz", redacted.Metadata["redactedData"])
	}
}

func TestParseSSE_RedactedThinking_ContextCancelled(t *testing.T) {
	// With a cancelled context and an unbuffered channel that has no reader,
	// TrySend must take the ctx.Done() branch and parseSSE must return without
	// blocking. Covers the cancellation path of the redacted_thinking case.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out := make(chan provider.StreamChunk)
	body := strings.NewReader(`data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"encrypted-blob"}}` + "\n\n")

	done := make(chan struct{})
	go func() {
		parseSSE(ctx, body, out, false)
		close(done)
	}()

	// With no reader, the unbuffered send is never ready, so TrySend must take
	// the cancelled-ctx branch and parseSSE must return (and close out).
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("parseSSE did not return after context cancellation")
	}

	// out is closed now; draining must complete without yielding a chunk.
	for chunk := range out {
		t.Errorf("expected no chunk after cancellation, got %+v", chunk)
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
	if _, hasTTL := cc["ttl"]; hasTTL {
		t.Errorf("cache_control should omit ttl by default, got %v", cc)
	}
}

func TestBuildRequest_SystemCacheControlTTL(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		System: "You are helpful.",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		PromptCaching: true,
		CacheTTL:      "1h",
	}, true)

	system := body["system"].([]map[string]any)
	cc, _ := system[0]["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control.type = %v, want ephemeral", cc["type"])
	}
	if cc["ttl"] != "1h" {
		t.Errorf("cache_control.ttl = %v, want 1h", cc["ttl"])
	}
}

// CacheTTL must not leak into the request when caching is off.
func TestBuildRequest_CacheTTLWithoutPromptCaching(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		System: "You are helpful.",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		CacheTTL: "1h",
	}, true)

	system := body["system"].([]map[string]any)
	if _, has := system[0]["cache_control"]; has {
		t.Errorf("cache_control set without PromptCaching: %v", system[0])
	}
}

func TestConvertMessages_PartLevelCacheTTL(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{
			Role:    provider.RoleUser,
			Content: []provider.Part{{Type: provider.PartText, Text: "hi", CacheControl: "ephemeral", CacheControlTTL: "1h"}},
		},
	})

	content := msgs[0]["content"].([]map[string]any)
	cc := content[0]["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control.type = %v, want ephemeral", cc["type"])
	}
	if cc["ttl"] != "1h" {
		t.Errorf("cache_control.ttl = %v, want 1h", cc["ttl"])
	}

	// Empty TTL must omit the ttl key (default 5m behavior preserved).
	msgs = convertMessages([]provider.Message{
		{
			Role:    provider.RoleUser,
			Content: []provider.Part{{Type: provider.PartText, Text: "hi", CacheControl: "ephemeral"}},
		},
	})
	cc = msgs[0]["content"].([]map[string]any)[0]["cache_control"].(map[string]any)
	if _, hasTTL := cc["ttl"]; hasTTL {
		t.Errorf("empty TTL should omit ttl key, got %v", cc)
	}
}

// A message-level breakpoint carries its own marker, so a part-level TTL on a
// part that has no part-level CacheControl must not alter it.
func TestConvertMessages_MessageLevelCacheControlIgnoresPartTTL(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{
			Role: provider.RoleUser,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "hi", CacheControlTTL: "1h"},
			},
			ProviderOptions: map[string]any{
				"anthropic": map[string]any{
					"cacheControl": map[string]any{"type": "ephemeral"},
				},
			},
		},
	})

	cc := msgs[0]["content"].([]map[string]any)[0]["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control.type = %v, want ephemeral", cc["type"])
	}
	if _, hasTTL := cc["ttl"]; hasTTL {
		t.Errorf("message-level marker must not gain a part TTL, got %v", cc)
	}
}

func TestEphemeralCacheControl(t *testing.T) {
	if cc := ephemeralCacheControl(""); cc["type"] != "ephemeral" || len(cc) != 1 {
		t.Errorf("ephemeralCacheControl(\"\") = %v, want {type:ephemeral} only", cc)
	}
	for _, ttl := range []string{"5m", "1h"} {
		cc := ephemeralCacheControl(ttl)
		if cc["type"] != "ephemeral" || cc["ttl"] != ttl {
			t.Errorf("ephemeralCacheControl(%q) = %v", ttl, cc)
		}
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
		{"none", map[string]any{"type": "none"}},
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
}

// Native structured output must be sent as output_config.format; the top-level
// output_format field is deprecated and rejected by the API. This exercises the
// full native path (injectNativeOutputFormat -> buildRequest).
func TestBuildRequest_NativeOutputFormat(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-6", opts: options{baseURL: defaultBaseURL}}

	schema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)
	params := provider.GenerateParams{
		Messages:       []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{Schema: schema},
		ProviderOptions: map[string]any{
			"structuredOutputMode": "outputFormat",
			"effort":               "high",
		},
	}

	if !m.useNativeOutputFormat(params) {
		t.Fatal("useNativeOutputFormat = false, want true for structuredOutputMode=outputFormat")
	}
	var err error
	params, err = injectNativeOutputFormat(params)
	if err != nil {
		t.Fatalf("injectNativeOutputFormat: %v", err)
	}
	body := m.buildRequest(params, false)

	// The deprecated top-level field must not be present.
	if _, ok := body["output_format"]; ok {
		t.Errorf("body has top-level output_format = %v, want it nested under output_config.format", body["output_format"])
	}

	oc, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("output_config = %v, want map", body["output_config"])
	}

	// effort and format coexist in the same output_config object.
	if oc["effort"] != "high" {
		t.Errorf("output_config.effort = %v, want high", oc["effort"])
	}

	format, ok := oc["format"].(map[string]any)
	if !ok {
		t.Fatalf("output_config.format = %v, want map", oc["format"])
	}
	if format["type"] != "json_schema" {
		t.Errorf("output_config.format.type = %v, want json_schema", format["type"])
	}
	got, _ := json.Marshal(format["schema"])
	var want, gotNorm any
	_ = json.Unmarshal(schema, &want)
	_ = json.Unmarshal(got, &gotNorm)
	wantJSON, _ := json.Marshal(want)
	if string(got) != string(wantJSON) {
		t.Errorf("output_config.format.schema = %s, want %s", got, wantJSON)
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

func TestConvertMessages_EmptySignedReasoning(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartReasoning, ProviderOptions: map[string]any{"signature": "sig-omitted"}},
			{Type: provider.PartText, Text: "answer"},
		}},
	})

	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 2 {
		t.Fatalf("content = %#v, want omitted thinking block and text", content)
	}
	if content[0]["type"] != "thinking" || content[0]["thinking"] != "" || content[0]["signature"] != "sig-omitted" {
		t.Fatalf("thinking block = %#v, want unchanged empty signed thinking", content[0])
	}
}

func TestConvertMessages_SkipsEmptyAndMergesSameRole(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartReasoning}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "one"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "two"}}},
	})
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v, want one merged user message", msgs)
	}
	content := msgs[0]["content"].([]map[string]any)
	if len(content) != 2 {
		t.Fatalf("content = %#v, want two text blocks", content)
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

func TestConvertMessages_FileRemoteRef(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartFile, RemoteRef: &provider.RemoteFileRef{ID: "file_abc123"}},
		}},
	})

	content := msgs[0]["content"].([]map[string]any)
	if content[0]["type"] != "document" {
		t.Errorf("type = %v, want document", content[0]["type"])
	}
	source, _ := content[0]["source"].(map[string]any)
	if source["type"] != "file" {
		t.Errorf("source.type = %v, want file", source["type"])
	}
	if source["file_id"] != "file_abc123" {
		t.Errorf("source.file_id = %v, want file_abc123", source["file_id"])
	}
}

func TestConvertMessages_FileRemoteRefWithCacheControl(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartFile, RemoteRef: &provider.RemoteFileRef{ID: "file_xyz"}, CacheControl: "ephemeral"},
		}},
	})

	content := msgs[0]["content"].([]map[string]any)
	if content[0]["type"] != "document" {
		t.Errorf("type = %v, want document", content[0]["type"])
	}
	if _, ok := content[0]["cache_control"]; !ok {
		t.Error("expected cache_control")
	}
}

func TestHasRemoteRef(t *testing.T) {
	if hasRemoteRef(nil) {
		t.Error("nil messages should return false")
	}
	if hasRemoteRef([]provider.Message{}) {
		t.Error("empty messages should return false")
	}
	if hasRemoteRef([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartText, Text: "hello"},
		}},
	}) {
		t.Error("messages without RemoteRef should return false")
	}
	if !hasRemoteRef([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartFile, RemoteRef: &provider.RemoteFileRef{ID: "file_1"}},
		}},
	}) {
		t.Error("messages with RemoteRef should return true")
	}
}

func TestChat_FileUploader(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"))
	uploader, ok := model.(provider.FileUploadCapableModel)
	if !ok {
		t.Fatal("chatModel should implement FileUploadCapableModel")
	}
	if uploader.FileUploader() == nil {
		t.Error("FileUploader() should return non-nil")
	}
}

func TestFileUploader_UploadFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/files" {
			t.Errorf("path = %q, want /v1/files", r.URL.Path)
		}
		if r.Method != "POST" {
			t.Errorf("method = %q, want POST", r.Method)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("x-api-key = %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != apiVersion {
			t.Errorf("anthropic-version = %q", r.Header.Get("anthropic-version"))
		}
		if r.Header.Get("anthropic-beta") != filesBetaHeader {
			t.Errorf("anthropic-beta = %q, want %q", r.Header.Get("anthropic-beta"), filesBetaHeader)
		}
		ct := r.Header.Get("Content-Type")
		if !strings.Contains(ct, "multipart/form-data") {
			t.Errorf("Content-Type = %q, want multipart/form-data", ct)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_xyz789","type":"file","size_bytes":123,"mime_type":"application/pdf","created_at":"2026-08-30T12:00:00Z","filename":"test.pdf"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("fake-pdf-content"),
		Filename:  "test.pdf",
		MediaType: "application/pdf",
		Purpose:   "assistants",
	})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
	if ref.ID != "file_xyz789" {
		t.Errorf("ref.ID = %q, want file_xyz789", ref.ID)
	}
	if ref.Filename != "test.pdf" {
		t.Errorf("ref.Filename = %q", ref.Filename)
	}
	if ref.MediaType != "application/pdf" {
		t.Errorf("ref.MediaType = %q", ref.MediaType)
	}
	if ref.Provider != "anthropic" {
		t.Errorf("ref.Provider = %q", ref.Provider)
	}
	if len(ref.Data) == 0 {
		t.Error("ref.Data should contain file bytes")
	}
}

func TestFileUploader_UploadFile_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"message":"bad request"}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("data"),
		Filename:  "test.pdf",
		MediaType: "application/pdf",
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestFileUploader_DeleteFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/files/file-to-delete" {
			t.Errorf("path = %q, want /v1/files/file-to-delete", r.URL.Path)
		}
		if r.Method != "DELETE" {
			t.Errorf("method = %q, want DELETE", r.Method)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("x-api-key = %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-beta") != filesBetaHeader {
			t.Errorf("anthropic-beta = %q", r.Header.Get("anthropic-beta"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file-to-delete","type":"file","deleted":true}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file-to-delete"})
	if err != nil {
		t.Fatalf("DeleteFile error: %v", err)
	}
}

func TestFileUploader_DeleteFile_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = fmt.Fprint(w, `{"error":{"message":"file not found"}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "nonexistent"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestFileUploader_UploadFile_EmptyMediaType_Anthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_mediatype","type":"file","size_bytes":3,"mime_type":"application/octet-stream","created_at":"2026-08-30T12:00:00Z","filename":"test.bin"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:   strings.NewReader("abc"),
		Filename: "test.bin",
	})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
	if ref.ID != "file_mediatype" {
		t.Errorf("ref.ID = %q", ref.ID)
	}
	if ref.MediaType == "" {
		t.Error("MediaType should be detected from content")
	}
}

func TestFileUploader_UploadFile_InvalidJSON_Anthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `not-json`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("data"),
		Filename:  "test.txt",
		MediaType: "text/plain",
	})
	if err == nil {
		t.Fatal("expected error for invalid JSON response")
	}
}

func TestFileUploader_UploadFile_ReadError_Anthropic(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    &failReader{},
		Filename:  "test.txt",
		MediaType: "text/plain",
	})
	if err == nil {
		t.Fatal("expected error for failing reader")
	}
}

func TestFileUploader_UploadFile_TokenError_Anthropic(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithTokenSource(failTokenSource{}))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("data"),
		Filename:  "test.txt",
		MediaType: "text/plain",
	})
	if err == nil {
		t.Fatal("expected error for token failure")
	}
}

func TestFileUploader_UploadFile_Headers_Anthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "value" {
			t.Errorf("X-Custom = %q", r.Header.Get("X-Custom"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file_headers","type":"file","size_bytes":3,"mime_type":"text/plain","created_at":"2026-08-30T12:00:00Z","filename":"test.txt"}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "value"}))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	ref, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("abc"),
		Filename:  "test.txt",
		MediaType: "text/plain",
	})
	if err != nil {
		t.Fatalf("UploadFile error: %v", err)
	}
	if ref.ID != "file_headers" {
		t.Errorf("ref.ID = %q", ref.ID)
	}
}

func TestFileUploader_UploadFile_HTTPClientError_Anthropic(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("data"),
		Filename:  "test.txt",
		MediaType: "text/plain",
	})
	if err == nil {
		t.Fatal("expected error for connection refused")
	}
}

func TestFileUploader_DeleteFile_TokenError_Anthropic(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithTokenSource(failTokenSource{}))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file-xyz"})
	if err == nil {
		t.Fatal("expected error for token failure")
	}
}

func TestFileUploader_DeleteFile_HTTPClientError_Anthropic(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file-xyz"})
	if err == nil {
		t.Fatal("expected error for connection refused")
	}
}

func TestFileUploader_DeleteFile_Headers_Anthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "value" {
			t.Errorf("X-Custom = %q", r.Header.Get("X-Custom"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"file-to-delete","type":"file","deleted":true}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL), WithHeaders(map[string]string{"X-Custom": "value"}))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file-to-delete"})
	if err != nil {
		t.Fatalf("DeleteFile error: %v", err)
	}
}

func TestFileUploader_UploadFile_InvalidURL_Anthropic(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL("://"))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	_, err := uploader.UploadFile(t.Context(), provider.FileUpload{
		Reader:    strings.NewReader("data"),
		Filename:  "test.txt",
		MediaType: "text/plain",
	})
	if err == nil {
		t.Fatal("expected error for invalid URL")
	}
}

func TestFileUploader_DeleteFile_InvalidURL_Anthropic(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL("://"))
	uploader := model.(provider.FileUploadCapableModel).FileUploader()

	err := uploader.DeleteFile(t.Context(), provider.RemoteFileRef{ID: "file-xyz"})
	if err == nil {
		t.Fatal("expected error for invalid URL")
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
		{"data:image/png;base64,iVBORw0KGgo=", "image/png", "iVBORw0KGgo=", true},
		{"data:application/pdf;base64,JVBERi0=", "application/pdf", "JVBERi0=", true},
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
	if !caps.FileUpload {
		t.Error("expected FileUpload=true")
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

func TestDoGenerate_PerRequestHeaderCannotOverrideAuth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("x-api-key"); got != "test-key" {
			t.Errorf("x-api-key = %q, want %q (per-request header must not override auth)", got, "test-key")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","model":"claude-sonnet-4-20250514","type":"message","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		// Caller attempts to override the auth header via a per-request header.
		Headers: map[string]string{"x-api-key": "attacker-key"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

func TestDoGenerate_PerRequestHeaderCannotOverrideAuthBearer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer my-token" {
			t.Errorf("Authorization = %q, want %q (per-request header must not override auth)", got, "Bearer my-token")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","model":"claude-sonnet-4-20250514","type":"message","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAuthMode(AuthBearer),
		WithTokenSource(provider.StaticToken("my-token")),
		WithBaseURL(server.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{"Authorization": "Bearer attacker-key"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
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

func TestBuildRequest_ThinkingAdaptiveDisplay(t *testing.T) {
	m := &chatModel{id: "claude-opus-4-8", opts: options{baseURL: defaultBaseURL}}
	body := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"thinking": map[string]any{"type": "adaptive", "display": "summarized"},
		},
	}, true)

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatal("thinking not in request body")
	}
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking.type = %v, want adaptive", thinking["type"])
	}
	if thinking["display"] != "summarized" {
		t.Errorf("thinking.display = %v, want summarized", thinking["display"])
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

func TestParseSSE_LargeEventPayload(t *testing.T) {
	// Regression test for issue #70: very long SSE data lines (e.g. large
	// tool-call argument deltas or reasoning blocks) must not fail with
	// "bufio.Scanner: token too long". The scanner backs off to a growable
	// bufio.Reader so lines of any size are accepted.
	largeText := strings.Repeat("x", 2*1024*1024) // 2 MiB
	event := map[string]any{
		"type":  "content_block_delta",
		"index": 0,
		"delta": map[string]any{"type": "text_delta", "text": largeText},
	}
	payload, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	stream := "data: " + string(payload) + "\n\n"

	out := make(chan provider.StreamChunk, 64)
	go parseSSE(t.Context(), strings.NewReader(stream), out, false)

	var gotText string
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			t.Fatalf("unexpected error chunk: %v", chunk.Error)
		}
		if chunk.Type == provider.ChunkText {
			gotText += chunk.Text
		}
	}
	if len(gotText) != len(largeText) {
		t.Errorf("got text len=%d, want %d", len(gotText), len(largeText))
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

// infiniteZeroReader yields zeros forever (never EOF). Used to drive the
// non-streaming response-size cap without allocating a huge buffer upfront.
type infiniteZeroReader struct{}

func (infiniteZeroReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

type oversizedBodyTransport struct{}

func (t *oversizedBodyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(infiniteZeroReader{}),
		Header:     make(http.Header),
	}, nil
}

// The non-streaming response body must be capped (64 MiB); an oversized body
// must yield a clear error instead of being read unbounded.
func TestDoGenerate_ResponseTooLarge(t *testing.T) {
	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("test-key"),
		WithBaseURL("http://localhost"),
		WithHTTPClient(&http.Client{Transport: &oversizedBodyTransport{}}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for oversized response body")
	}
	if !strings.Contains(err.Error(), "response body exceeds") {
		t.Errorf("expected 'response body exceeds' error, got: %v", err)
	}
}

// Error-response bodies read by doHTTP must be bounded (1 MiB) so a hostile
// error payload cannot exhaust memory; a large error body still yields a
// parsed HTTP error.
func TestDoHTTP_LargeErrorBody_Bounded(t *testing.T) {
	big := strings.Repeat("x", 2<<20) // 2 MiB > 1 MiB cap
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = fmt.Fprint(w, `{"error":{"type":"api_error","message":"`+big+`"}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error from 502 response")
	}
	if !strings.Contains(err.Error(), "api_error") {
		t.Errorf("expected api_error in error, got: %v", err)
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

		// Verify the schema is sent as output_config.format, not the deprecated
		// top-level output_format. Non-fatal assertions only: t.Fatal/panic in
		// this handler goroutine would reset the connection (client EOF).
		if _, ok := req["output_format"]; ok {
			t.Error("deprecated top-level output_format present in request")
		}
		oc, _ := req["output_config"].(map[string]any)
		if oc == nil {
			t.Error("output_config not present in request")
		}
		of, _ := oc["format"].(map[string]any)
		if of == nil {
			t.Error("output_config.format not present in request")
		} else if of["type"] != "json_schema" {
			t.Errorf("output_config.format.type = %v, want json_schema", of["type"])
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
	result, err := injectNativeOutputFormat(params)
	if err != nil {
		t.Fatalf("injectNativeOutputFormat: %v", err)
	}
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

// injectNativeOutputFormat with invalid schema bytes must surface an error
// rather than silently dropping the requested output mode.
func TestInjectNativeOutputFormat_InvalidSchema(t *testing.T) {
	params := provider.GenerateParams{
		ResponseFormat: &provider.ResponseFormat{
			Schema: json.RawMessage(`not valid json`),
		},
		ProviderOptions: map[string]any{"existing": "value"},
	}
	result, err := injectNativeOutputFormat(params)
	if err == nil {
		t.Fatal("expected an error for an invalid schema, got nil")
	}
	// ResponseFormat must not be cleared and output_format must not be set.
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
						"excludeTools":       []any{"bash"},
						"customUnknownField": "passthrough",
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
	// #16: claude-code-20250219 is no longer a default beta; a computer-use
	// request does not need it.
	if strings.Contains(gotBeta, "claude-code-20250219") {
		t.Errorf("beta header should not contain claude-code-20250219, got %q", gotBeta)
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
			name:  "tool_use_start",
			input: "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"fn\"}}\n",
		},
		{
			// server_tool_use start (line 710)
			name:  "server_tool_use_start",
			input: "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"t2\",\"name\":\"bash_code_execution\"}}\n",
		},
		{
			// text_delta (line 727)
			name:  "text_delta",
			input: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n",
		},
		{
			// thinking_delta (line 734)
			name:  "thinking_delta",
			input: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n",
		},
		{
			// signature_delta (line 741)
			name:  "signature_delta",
			input: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig123\"}}\n",
		},
		{
			// citations_delta (line 753)
			name:  "citations_delta",
			input: "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"citations_delta\",\"citation\":{\"url\":\"http://x\"}}}\n",
		},
		{
			// input_json_delta as text (RF mode handled separately -- normal tool delta, line 786)
			// This needs tool_use start first, which TrySends. Use pipe for that.
			// For pre-cancel: use isRFBlock=true path -- but parseSSE is called with isRFMode.
			// For RF mode text: just set isRFMode=true and the text TrySend at 768 fires.
			// But we can't control isRFMode here. Skip -- use a separate sub-test below.
			name:  "message_stop",
			input: "data: {\"type\":\"message_stop\"}\n",
		},
		{
			// message_delta stop_reason (line 828)
			name:  "message_delta",
			input: "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n",
		},
		{
			// error - overflow via ClassifyStreamError (line 916)
			name:  "error_overflow",
			input: "data: {\"type\":\"error\",\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"too long\"}}\n",
		},
		{
			// error - overflow via IsOverflow (line 937)
			name:  "error_overflow_msg",
			input: "data: {\"type\":\"error\",\"error\":{\"message\":\"prompt is too long\"}}\n",
		},
		{
			// error - generic (line 945)
			name:  "error_generic",
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
		cancel()                               // cancel before parsing
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
	result, err := injectNativeOutputFormat(params)
	if err != nil {
		t.Fatalf("injectNativeOutputFormat: %v", err)
	}
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

// TestParseSSE_ScannerError_EmitsChunkFinish verifies that when scanner.Err()
// fires (truncated stream), a ChunkFinish is emitted after the ChunkError.
func TestParseSSE_ScannerError_EmitsChunkFinish(t *testing.T) {
	// The stream has a message_start (providing input tokens), a text delta,
	// then a read error. The parser should emit: text chunk, error chunk, finish chunk.
	input := "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_err\",\"model\":\"sonnet-test\",\"usage\":{\"input_tokens\":10}}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n"

	r := &slowErrorReader{
		data: input,
		err:  fmt.Errorf("simulated read error"),
	}

	out := make(chan provider.StreamChunk, 16)
	parseSSE(t.Context(), r, out, false)

	var chunks []provider.StreamChunk
	for c := range out {
		chunks = append(chunks, c)
	}

	// Expect at least: text, error, finish.
	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 chunks (text, error, finish), got %d: %+v", len(chunks), chunks)
	}

	var errorIdx, finishIdx = -1, -1
	for i, c := range chunks {
		if c.Type == provider.ChunkError && errorIdx == -1 {
			errorIdx = i
		}
		if c.Type == provider.ChunkFinish {
			finishIdx = i
		}
	}
	if errorIdx == -1 {
		t.Fatal("no ChunkError found")
	}
	if finishIdx == -1 {
		t.Fatal("no ChunkFinish found after scanner error")
	}
	if finishIdx <= errorIdx {
		t.Errorf("ChunkFinish (index %d) should come after ChunkError (index %d)", finishIdx, errorIdx)
	}

	// Verify usage is propagated (input_tokens was in the stream).
	finish := chunks[finishIdx]
	if finish.Usage.InputTokens != 10 {
		t.Errorf("finish.Usage.InputTokens = %d, want 10", finish.Usage.InputTokens)
	}
	// Verify FinishReason is "error" on scanner error path.
	if finish.FinishReason != "error" {
		t.Errorf("finish.FinishReason = %q, want \"error\"", finish.FinishReason)
	}
	// Verify Response.Model is propagated from the message_start event.
	if finish.Response.Model != "sonnet-test" {
		t.Errorf("finish.Response.Model = %q, want \"sonnet-test\"", finish.Response.Model)
	}
}

// TestParseSSE_CleanEOF_EmitsChunkFinish verifies that when the stream ends
// without a message_stop event, a ChunkFinish is still emitted (clean-EOF path).
func TestParseSSE_CleanEOF_EmitsChunkFinish(t *testing.T) {
	input := "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_eof\",\"model\":\"sonnet-eof\",\"usage\":{\"input_tokens\":7}}}\n\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
	// Stream ends without message_stop (clean EOF).

	out := make(chan provider.StreamChunk, 16)
	parseSSE(t.Context(), strings.NewReader(input), out, false)

	var chunks []provider.StreamChunk
	for c := range out {
		chunks = append(chunks, c)
	}

	var finish *provider.StreamChunk
	for i := range chunks {
		if chunks[i].Type == provider.ChunkFinish {
			finish = &chunks[i]
		}
	}
	if finish == nil {
		t.Fatal("no ChunkFinish emitted on clean EOF")
	}
	if finish.Response.Model != "sonnet-eof" {
		t.Errorf("finish.Response.Model = %q, want \"sonnet-eof\"", finish.Response.Model)
	}
}

// TestDoStream_ContextCancel_NoDoubleClose verifies that cancelling the context
// does not cause a panic or double-close of resp.Body.
func TestDoStream_ContextCancel_NoDoubleClose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Flush a chunk then block indefinitely to simulate a stalled server.
		_, _ = fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n")
		w.(http.Flusher).Flush()
		// Block until client disconnects.
		<-r.Context().Done()
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("test-key"), WithBaseURL(server.URL))

	ctx, cancel := context.WithCancel(t.Context())
	result, err := model.DoStream(ctx, provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Cancel context while stream is in progress.
	cancel()

	// Drain the channel; this should not panic or deadlock.
	for range result.Stream {
	}
}

// --- Hook option tests (used by Vertex Anthropic adapter) ---

func TestWithAuthMode_Bearer(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"msg-1","type":"message","model":"claude-sonnet-4-20250514","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("my-token"),
		WithBaseURL(server.URL),
		WithAuthMode(AuthBearer),
		WithSkipEnvResolve(),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer my-token" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer my-token")
	}
}

func TestWithURLBuilder(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"msg-1","type":"message","model":"claude-sonnet-4-20250514","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithURLBuilder(func(baseURL, modelID string, streaming bool) string {
			if streaming {
				return baseURL + "/models/" + modelID + ":streamRawPredict"
			}
			return baseURL + "/models/" + modelID + ":rawPredict"
		}),
		WithSkipEnvResolve(),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/models/claude-sonnet-4-20250514:rawPredict" {
		t.Errorf("path = %q, want /models/claude-sonnet-4-20250514:rawPredict", gotPath)
	}
}

func TestWithURLBuilder_Streaming(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"type":"message_start","message":{"id":"msg-1","model":"claude-sonnet-4-20250514","usage":{"input_tokens":1}}}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`+"\n\n")
		fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithURLBuilder(func(baseURL, modelID string, streaming bool) string {
			return baseURL + "/models/" + modelID + ":streamRawPredict"
		}),
		WithSkipEnvResolve(),
	)
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
	if gotPath != "/models/claude-sonnet-4-20250514:streamRawPredict" {
		t.Errorf("path = %q, want :streamRawPredict suffix", gotPath)
	}
}

func TestWithBodyTransformer(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"msg-1","type":"message","model":"claude-sonnet-4-20250514","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithBodyTransformer(func(body map[string]any) map[string]any {
			delete(body, "model")
			body["anthropic_version"] = "vertex-2023-10-16"
			return body
		}),
		WithSkipEnvResolve(),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := gotBody["model"]; ok {
		t.Error("body should not contain 'model'")
	}
	if v := gotBody["anthropic_version"]; v != "vertex-2023-10-16" {
		t.Errorf("anthropic_version = %v, want vertex-2023-10-16", v)
	}
}

func TestWithErrorProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514",
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithErrorProvider("vertex-anthropic"),
		WithSkipEnvResolve(),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	// Verify it's an API error with the right status.
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *goai.APIError, got %T", err)
	}
	if apiErr.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", apiErr.StatusCode, http.StatusBadRequest)
	}
}

func TestWithSkipEnvResolve(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-key-should-not-be-used")
	t.Setenv("ANTHROPIC_BASE_URL", "http://env-url-should-not-be-used")

	model := Chat("claude-sonnet-4-20250514", WithSkipEnvResolve()).(*chatModel)

	// Should NOT have picked up env key.
	if model.opts.tokenSource != nil {
		t.Error("tokenSource should be nil when skipEnvResolve is set and no explicit key given")
	}
	// Should still have the default base URL, not the env one.
	if model.opts.baseURL != defaultBaseURL {
		t.Errorf("baseURL = %q, want %q", model.opts.baseURL, defaultBaseURL)
	}
}

// --- Server-side tool round-trip (web_search, code_execution, ...) ---

// TestChat_Generate_ServerToolResultRoundTrip verifies that web_search_tool_result
// (and similar server-executed tool result blocks) are captured into the
// matching ToolCall.Metadata so they survive a multi-turn round-trip.
// Regression for the issue where re-sending ResponseMessages containing a
// server_tool_use (e.g. web_search) without its inline result was rejected
// by the API as "tool_use ids were found without `tool_result` blocks".
func TestChat_Generate_ServerToolResultRoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "msg_srv1",
			"model": "sonnet-test-model",
			"type": "message",
			"content": [
				{"type": "text", "text": "Searching..."},
				{"type": "server_tool_use", "id": "srvtoolu_abc", "name": "web_search", "input": {"query": "go 1.26"}},
				{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_abc", "content": [
					{"type": "web_search_result", "url": "https://example.com", "title": "Go 1.26 release", "encrypted_content": "xxx"}
				]},
				{"type": "text", "text": "Go 1.26 was released in 2026."}
			],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 30, "output_tokens": 40}
		}`)
	}))
	defer server.Close()

	model := Chat("sonnet-test-model", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "what's new in go?"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
	tc := result.ToolCalls[0]
	if tc.ID != "srvtoolu_abc" || tc.Name != "web_search" {
		t.Errorf("ToolCall = %+v, want srvtoolu_abc/web_search", tc)
	}
	rb, ok := tc.Metadata["resultBlock"].(map[string]any)
	if !ok {
		t.Fatalf("ToolCall.Metadata[resultBlock] missing or wrong type: %T", tc.Metadata["resultBlock"])
	}
	if rb["type"] != "web_search_tool_result" {
		t.Errorf("resultBlock type = %v, want web_search_tool_result", rb["type"])
	}
	if rb["tool_use_id"] != "srvtoolu_abc" {
		t.Errorf("resultBlock tool_use_id = %v, want srvtoolu_abc", rb["tool_use_id"])
	}
}

// TestChat_Generate_ToolSearchResultRoundTrip verifies that the built-in tool
// search server tool (server_tool_use + tool_search_tool_result) is captured
// into the matching ToolCall.Metadata so it survives a multi-turn round-trip,
// like web_search and the other server-executed tool results. Unlike those,
// the tool_search_tool_result content is an object (not an array), which the
// verbatim round-trip must preserve unchanged.
func TestChat_Generate_ToolSearchResultRoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "msg_ts1",
			"model": "sonnet-test-model",
			"type": "message",
			"content": [
				{"type": "server_tool_use", "id": "srvtoolu_ts", "name": "tool_search_tool_regex", "input": {"query": "weather"}},
				{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_ts", "content": {"type": "tool_search_tool_search_result", "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}]}}
			],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 20, "output_tokens": 15}
		}`)
	}))
	defer server.Close()

	model := Chat("sonnet-test-model", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "weather in SF?"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
	tc := result.ToolCalls[0]
	if tc.ID != "srvtoolu_ts" || tc.Name != "tool_search_tool_regex" {
		t.Errorf("ToolCall = %+v, want srvtoolu_ts/tool_search_tool_regex", tc)
	}
	rb, ok := tc.Metadata["resultBlock"].(map[string]any)
	if !ok {
		t.Fatalf("ToolCall.Metadata[resultBlock] missing or wrong type: %T", tc.Metadata["resultBlock"])
	}
	if rb["type"] != "tool_search_tool_result" {
		t.Errorf("resultBlock type = %v, want tool_search_tool_result", rb["type"])
	}
	if rb["tool_use_id"] != "srvtoolu_ts" {
		t.Errorf("resultBlock tool_use_id = %v, want srvtoolu_ts", rb["tool_use_id"])
	}
}

// TestChat_Stream_ServerToolResultRoundTrip is the streaming counterpart to
// TestChat_Generate_ServerToolResultRoundTrip.
func TestChat_Stream_ServerToolResultRoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"id":"msg_srv2","model":"sonnet-test-model","usage":{"input_tokens":30}}}`+"\n\n")
		// Index 0: text
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Searching..."}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		// Index 1: server_tool_use
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_abc","name":"web_search"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"go 1.26\"}"}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":1}`+"\n\n")
		// Index 2: web_search_tool_result (full block in start, no deltas)
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":2,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_abc","content":[{"type":"web_search_result","url":"https://example.com","title":"Go 1.26 release","encrypted_content":"xxx"}]}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":2}`+"\n\n")
		// Index 3: text follow-up
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"Go 1.26 was released in 2026."}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":3}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":40}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("sonnet-test-model", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "what's new in go?"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var sawToolCall provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			sawToolCall = chunk
		}
	}
	if sawToolCall.ToolCallID != "srvtoolu_abc" {
		t.Fatalf("ChunkToolCall ID = %q, want srvtoolu_abc", sawToolCall.ToolCallID)
	}
	rb, ok := sawToolCall.Metadata["resultBlock"].(map[string]any)
	if !ok {
		t.Fatalf("ChunkToolCall.Metadata[resultBlock] missing: %v", sawToolCall.Metadata)
	}
	if rb["type"] != "web_search_tool_result" {
		t.Errorf("resultBlock type = %v, want web_search_tool_result", rb["type"])
	}
	if rb["tool_use_id"] != "srvtoolu_abc" {
		t.Errorf("resultBlock tool_use_id = %v, want srvtoolu_abc", rb["tool_use_id"])
	}
}

// TestParseSSE_ServerToolResult_UnknownToolUseID covers flushPendingCall's
// early return: a server tool result block whose tool_use_id has no matching
// pending server_tool_use. On content_block_stop the result-block branch calls
// flushPendingCall(currentResultUseID), which finds the id absent from
// pendingCalls and returns true immediately (no ChunkToolCall is emitted).
func TestParseSSE_ServerToolResult_UnknownToolUseID(t *testing.T) {
	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"ghost_id\",\"content\":[]}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n"

	out := make(chan provider.StreamChunk, 8)
	parseSSE(t.Context(), strings.NewReader(input), out, false)

	var chunks []provider.StreamChunk
	for c := range out {
		chunks = append(chunks, c)
	}
	for _, c := range chunks {
		if c.Type == provider.ChunkToolCall {
			t.Fatalf("unexpected ChunkToolCall for unknown tool_use_id ghost_id: %+v", c)
		}
	}
}

// TestParseSSE_ServerTool_PendingFlushedOnPlainTextBlock covers the
// content_block_start branch where a pending server_tool_use awaits its result
// block but the next block is a plain text block: the pending call is flushed
// (without a resultBlock attached) before the text block is processed.
func TestParseSSE_ServerTool_PendingFlushedOnPlainTextBlock(t *testing.T) {
	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"st1\",\"name\":\"web_search\"}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"x\\\"}\"}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n" +
		"data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"follow-up\"}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n" +
		"data: {\"type\":\"message_stop\"}\n"

	out := make(chan provider.StreamChunk, 16)
	parseSSE(t.Context(), strings.NewReader(input), out, false)

	var chunks []provider.StreamChunk
	for c := range out {
		chunks = append(chunks, c)
	}
	var sawToolCall bool
	for _, c := range chunks {
		if c.Type == provider.ChunkToolCall && c.ToolCallID == "st1" {
			sawToolCall = true
			if _, hasRB := c.Metadata["resultBlock"]; hasRB {
				t.Errorf("pending call flushed on plain text block unexpectedly has resultBlock: %+v", c.Metadata)
			}
		}
	}
	if !sawToolCall {
		t.Fatal("expected ChunkToolCall for pending server_tool_use st1 flushed on plain text block")
	}
}

// TestParseSSE_ServerTool_FlushAllPending_TrySendFail covers flushAllPending's
// failure path: a pending server_tool_use is flushed when the next block is a
// plain text block, but TrySend fails (cancelled context), so flushPendingCall
// returns false and flushAllPending aborts parseSSE.
func TestParseSSE_ServerTool_FlushAllPending_TrySendFail(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	out := make(chan provider.StreamChunk) // unbuffered

	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"st1\",\"name\":\"web_search\"}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n" +
		"data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n"

	done := make(chan struct{})
	go func() {
		parseSSE(ctx, strings.NewReader(input), out, false)
		close(done)
	}()

	<-out // server_tool_use start
	<-out // input_json_delta
	cancel()
	<-done
	for range out {
	}
}

// TestParseSSE_ServerTool_FlushOnMessageDelta covers the message_delta
// stop_reason branch: pending server_tool_use calls are flushed (emitting
// ChunkToolCall) before the ChunkStepFinish is signalled.
func TestParseSSE_ServerTool_FlushOnMessageDelta(t *testing.T) {
	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"st1\",\"name\":\"web_search\"}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n" +
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n" +
		"data: {\"type\":\"message_stop\"}\n"

	out := make(chan provider.StreamChunk, 16)
	parseSSE(t.Context(), strings.NewReader(input), out, false)

	var chunks []provider.StreamChunk
	for c := range out {
		chunks = append(chunks, c)
	}
	var toolIdx, finishIdx = -1, -1
	for i, c := range chunks {
		if c.Type == provider.ChunkToolCall && c.ToolCallID == "st1" {
			toolIdx = i
		}
		if c.Type == provider.ChunkStepFinish {
			finishIdx = i
		}
	}
	if toolIdx == -1 {
		t.Fatal("expected ChunkToolCall for st1 flushed on message_delta")
	}
	if finishIdx == -1 {
		t.Fatal("expected ChunkStepFinish")
	}
	if finishIdx <= toolIdx {
		t.Errorf("ChunkStepFinish (%d) should come after flushed ChunkToolCall (%d)", finishIdx, toolIdx)
	}
}

// TestParseSSE_ServerTool_FlushOnMessageStop covers the message_stop branch:
// pending server_tool_use calls are flushed (emitting ChunkToolCall) before the
// ChunkFinish is emitted.
func TestParseSSE_ServerTool_FlushOnMessageStop(t *testing.T) {
	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"st1\",\"name\":\"web_search\"}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n" +
		"data: {\"type\":\"message_stop\"}\n"

	out := make(chan provider.StreamChunk, 16)
	parseSSE(t.Context(), strings.NewReader(input), out, false)

	var chunks []provider.StreamChunk
	for c := range out {
		chunks = append(chunks, c)
	}
	var toolIdx, finishIdx = -1, -1
	for i, c := range chunks {
		if c.Type == provider.ChunkToolCall && c.ToolCallID == "st1" {
			toolIdx = i
		}
		if c.Type == provider.ChunkFinish {
			finishIdx = i
		}
	}
	if toolIdx == -1 {
		t.Fatal("expected ChunkToolCall for st1 flushed on message_stop")
	}
	if finishIdx == -1 {
		t.Fatal("expected ChunkFinish")
	}
	if finishIdx <= toolIdx {
		t.Errorf("ChunkFinish (%d) should come after flushed ChunkToolCall (%d)", finishIdx, toolIdx)
	}
}

// TestParseSSE_ServerTool_ResultStopFlushTrySendFail covers the failure branch
// of the content_block_stop result-block flush: flushPendingCall's TrySend
// fails (cancelled context), so parseSSE returns at the `return` inside
// `if !flushPendingCall(currentResultUseID)`.
func TestParseSSE_ServerTool_ResultStopFlushTrySendFail(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	out := make(chan provider.StreamChunk) // unbuffered

	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"st1\",\"name\":\"web_search\"}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n" +
		"data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"st1\",\"content\":[]}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n"

	done := make(chan struct{})
	go func() {
		parseSSE(ctx, strings.NewReader(input), out, false)
		close(done)
	}()

	<-out // server_tool_use start
	<-out // input_json_delta
	cancel()
	<-done
	for range out {
	}
}

// TestParseSSE_ServerTool_MessageDeltaFlushTrySendFail covers the failure branch
// of the message_delta stop_reason flush: flushAllPending's TrySend fails
// (cancelled context), so parseSSE returns at the `return` inside
// `if !flushAllPending()`.
func TestParseSSE_ServerTool_MessageDeltaFlushTrySendFail(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	out := make(chan provider.StreamChunk) // unbuffered

	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"st1\",\"name\":\"web_search\"}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n" +
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n"

	done := make(chan struct{})
	go func() {
		parseSSE(ctx, strings.NewReader(input), out, false)
		close(done)
	}()

	<-out // server_tool_use start
	<-out // input_json_delta
	cancel()
	<-done
	for range out {
	}
}

// TestParseSSE_ServerTool_MessageStopFlushTrySendFail covers the failure branch
// of the message_stop flush: flushAllPending's TrySend fails (cancelled
// context), so parseSSE returns at the `return` inside `if !flushAllPending()`.
func TestParseSSE_ServerTool_MessageStopFlushTrySendFail(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	out := make(chan provider.StreamChunk) // unbuffered

	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"st1\",\"name\":\"web_search\"}}\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n" +
		"data: {\"type\":\"content_block_stop\"}\n" +
		"data: {\"type\":\"message_stop\"}\n"

	done := make(chan struct{})
	go func() {
		parseSSE(ctx, strings.NewReader(input), out, false)
		close(done)
	}()

	<-out // server_tool_use start
	<-out // input_json_delta
	cancel()
	<-done
	for range out {
	}
}

// TestChat_Generate_ServerToolResult_UnknownToolUseID covers the parseResponse
// branch where a server_tool_result block references a tool_use_id that has no
// matching server_tool_use: the orphan result block is skipped via continue.
func TestChat_Generate_ServerToolResult_UnknownToolUseID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "msg_ghost",
			"model": "sonnet-test-model",
			"type": "message",
			"content": [
				{"type": "text", "text": "done"},
				{"type": "web_search_tool_result", "tool_use_id": "ghost_id", "content": []}
			],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 5, "output_tokens": 5}
		}`)
	}))
	defer server.Close()

	model := Chat("sonnet-test-model", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// The orphan result block for an unknown tool_use_id is skipped; no
	// ToolCall should be created from it.
	if len(result.ToolCalls) != 0 {
		t.Fatalf("ToolCalls = %d, want 0 (orphan result block skipped)", len(result.ToolCalls))
	}
	if result.Text != "done" {
		t.Errorf("Text = %q, want done", result.Text)
	}
}

// TestConvertMessages_ServerToolResultBlock verifies that an assistant
// PartToolCall with ProviderOptions["resultBlock"] re-emits the original
// server_tool_use + web_search_tool_result pair when sent back to the API.
func TestConvertMessages_ServerToolResultBlock(t *testing.T) {
	resultBlock := map[string]any{
		"type":        "web_search_tool_result",
		"tool_use_id": "srvtoolu_abc",
		"content": []any{
			map[string]any{"type": "web_search_result", "url": "https://example.com", "title": "Go 1.26"},
		},
	}
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "what's new?"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartText, Text: "Searching..."},
			{
				Type:            provider.PartToolCall,
				ToolCallID:      "srvtoolu_abc",
				ToolName:        "web_search",
				ToolInput:       json.RawMessage(`{"query":"go 1.26"}`),
				ProviderOptions: map[string]any{"resultBlock": resultBlock},
			},
			{Type: provider.PartText, Text: "Go 1.26 was released in 2026."},
		}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "thanks"}}},
	})

	if len(msgs) < 2 {
		t.Fatalf("got %d messages, want at least 2", len(msgs))
	}
	asst := msgs[1]["content"].([]map[string]any)
	// Find server_tool_use and assert the result block follows immediately.
	// (ReorderAssistantParts moves text before tool_use; what matters is the
	// tool_use → result adjacency required by the API.)
	stIdx := -1
	for i, p := range asst {
		if p["type"] == "server_tool_use" {
			stIdx = i
			break
		}
	}
	if stIdx < 0 {
		t.Fatalf("no server_tool_use block in assistant content: %+v", asst)
	}
	if asst[stIdx]["id"] != "srvtoolu_abc" {
		t.Errorf("server_tool_use id = %v, want srvtoolu_abc", asst[stIdx]["id"])
	}
	if stIdx+1 >= len(asst) {
		t.Fatalf("no block after server_tool_use; result missing")
	}
	rb := asst[stIdx+1]
	if rb["type"] != "web_search_tool_result" {
		t.Errorf("block after server_tool_use type = %v, want web_search_tool_result", rb["type"])
	}
	if rb["tool_use_id"] != "srvtoolu_abc" {
		t.Errorf("result tool_use_id = %v, want srvtoolu_abc", rb["tool_use_id"])
	}

	// No orphan tool_result should be injected for the server tool call --
	// its result is already inline in the assistant turn. An orphan would
	// surface as a tool_result block in a following user message.
	for i := 2; i < len(msgs); i++ {
		c := msgs[i]["content"].([]map[string]any)
		for _, p := range c {
			if p["type"] == "tool_result" && p["tool_use_id"] == "srvtoolu_abc" {
				t.Errorf("orphan tool_result injected for server tool srvtoolu_abc in msg[%d]: %+v", i, p)
			}
		}
	}
}

// End-to-end: the configured TTL must reach the wire on both the system block
// and the message breakpoint, and the 1h TTL must not require a beta header.
func TestDoGenerate_CacheTTLOnWire(t *testing.T) {
	var req map[string]any
	var betaHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		betaHeader = r.Header.Get("anthropic-beta")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		System: "You are helpful.",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{
				{Type: provider.PartText, Text: "hi", CacheControl: "ephemeral", CacheControlTTL: "1h"},
			}},
		},
		PromptCaching: true,
		CacheTTL:      "1h",
	})
	if err != nil {
		t.Fatal(err)
	}

	system := req["system"].([]any)[0].(map[string]any)
	if cc := system["cache_control"].(map[string]any); cc["ttl"] != "1h" {
		t.Errorf("system cache_control = %v, want ttl 1h", cc)
	}

	msg := req["messages"].([]any)[0].(map[string]any)
	part := msg["content"].([]any)[0].(map[string]any)
	if cc := part["cache_control"].(map[string]any); cc["ttl"] != "1h" {
		t.Errorf("part cache_control = %v, want ttl 1h", cc)
	}

	// 1h is GA -- it must not add a beta opt-in.
	if betaHeader != betaFeatures {
		t.Errorf("anthropic-beta = %q, want %q (1h TTL is GA)", betaHeader, betaFeatures)
	}
}

func TestAnthropicModelVersion(t *testing.T) {
	cases := []struct {
		id    string
		major int
		minor int
		ok    bool
	}{
		// Current naming, with and without a release-date suffix.
		{"claude-opus-4-7", 4, 7, true},
		{"claude-opus-4-7-20260101", 4, 7, true},
		{"claude-sonnet-4-6-20260310", 4, 6, true},
		{"claude-haiku-4-5-20251001", 4, 5, true},
		{"claude-opus-5", 5, 0, true},
		{"claude-fable-5", 5, 0, true},
		{"claude-mythos-5", 5, 0, true},
		{"claude-opus-5-1", 5, 1, true},
		// Multi-digit minor versions.
		{"claude-sonnet-4-10", 4, 10, true},
		// A trailing 8-digit run is a release date, not a minor version.
		{"claude-sonnet-4-20250514", 4, 0, true},
		{"claude-opus-4-20250514", 4, 0, true},
		// Bedrock reuses this provider with a prefixed id.
		{"anthropic.claude-opus-5", 5, 0, true},
		{"us.anthropic.claude-sonnet-4-6", 4, 6, true},
		// Vertex appends an @date suffix.
		{"claude-opus-4-5@20251101", 4, 5, true},
		// Legacy family-last naming carries no parseable version.
		{"claude-3-7-sonnet", 0, 0, false},
		{"claude-3-5-sonnet-20241022", 0, 0, false},
		{"anthropic.claude-3-5-sonnet-20241022-v2:0", 0, 0, false},
		{"not-a-claude-model", 0, 0, false},
		{"", 0, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			major, minor, ok := anthropicModelVersion(tc.id)
			if ok != tc.ok || major != tc.major || minor != tc.minor {
				t.Errorf("anthropicModelVersion(%q) = (%d, %d, %v), want (%d, %d, %v)",
					tc.id, major, minor, ok, tc.major, tc.minor, tc.ok)
			}
		})
	}
}

func TestSupportsThinking(t *testing.T) {
	cases := []struct {
		id   string
		want bool
	}{
		// Previously matched by the literal list -- must not regress.
		{"claude-3-7-sonnet", true},
		{"claude-3-7-sonnet-20250219", true},
		{"claude-sonnet-4-20250514", true},
		{"claude-opus-4-20250514", true},
		{"claude-sonnet-4-6", true},
		{"claude-opus-4-6", true},
		{"claude-opus-4-5", true},
		// Previously false despite supporting thinking.
		{"claude-haiku-4-5-20251001", true},
		{"claude-opus-4-7", true},
		{"claude-opus-4-8", true},
		{"claude-opus-5", true},
		{"claude-sonnet-5", true},
		{"claude-fable-5", true},
		{"claude-mythos-5", true},
		{"anthropic.claude-opus-5", true},
		// Pre-4 models have no thinking support.
		{"claude-3-5-sonnet-20241022", false},
		{"claude-3-haiku-20240307", false},
		{"not-a-claude-model", false},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			if got := supportsThinking(tc.id); got != tc.want {
				t.Errorf("supportsThinking(%q) = %v, want %v", tc.id, got, tc.want)
			}
		})
	}
}

func TestSupportsNativeOutputFormat(t *testing.T) {
	cases := []struct {
		id   string
		want bool
	}{
		// Documented direct-Claude API compatibility set.
		{"claude-sonnet-4-6", true},
		{"claude-sonnet-4-6-20260310", true},
		{"claude-opus-4-6", true},
		{"claude-sonnet-4-5", true},
		{"claude-opus-4-5", true},
		{"claude-opus-4-7", true},
		{"claude-opus-4-8", true},
		{"claude-haiku-4-5-20251001", true},
		{"claude-opus-5", true},
		{"claude-sonnet-5", true},
		{"claude-fable-5", true},
		{"claude-mythos-5", true},
		{"claude-mythos-preview", true},
		// Release-date and platform aliases.
		{"claude-opus-4-5@20251101", true},
		{"anthropic.claude-opus-5", true},
		{"us.anthropic.claude-sonnet-4-6", true},
		// Not on the documented list, despite a version >= 4.1.
		{"claude-opus-4-1", false},
		{"claude-opus-4-1-20250101", false},
		{"claude-opus-4-9", false},
		{"claude-opus-6", false},
		// The 4.0 release predates structured output, bare or dated.
		{"claude-sonnet-4-20250514", false},
		{"claude-opus-4-20250514", false},
		// Legacy family-last naming.
		{"claude-3-5-sonnet-20241022", false},
		{"claude-3-7-sonnet", false},
		{"not-a-claude-model", false},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			m := &chatModel{id: tc.id}
			if got := m.supportsNativeOutputFormat(); got != tc.want {
				t.Errorf("supportsNativeOutputFormat(%q) = %v, want %v", tc.id, got, tc.want)
			}
		})
	}
}

func TestSupportsNativeOutputFormat_Bedrock(t *testing.T) {
	bedrock := &chatModel{id: "x", opts: options{nativeOutputFormatModels: bedrockNativeOutputFormatModels}}
	cases := []struct {
		id   string
		want bool
	}{
		// Documented Bedrock subset.
		{"claude-opus-4-6", true},
		{"anthropic.claude-opus-4-6", true},
		{"claude-sonnet-4-6", true},
		{"us.anthropic.claude-sonnet-4-6", true},
		{"claude-sonnet-4-5", true},
		{"claude-opus-4-5", true},
		{"claude-haiku-4-5", true},
		// Not in the Bedrock subset even though the direct API supports them.
		{"claude-opus-5", false},
		{"claude-sonnet-5", false},
		{"claude-fable-5", false},
		{"claude-mythos-preview", false},
		{"claude-opus-4-7", false},
		{"claude-opus-4-8", false},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			bedrock.id = tc.id
			if got := bedrock.supportsNativeOutputFormat(); got != tc.want {
				t.Errorf("bedrock supportsNativeOutputFormat(%q) = %v, want %v", tc.id, got, tc.want)
			}
		})
	}
}

func TestSupportsNativeOutputFormat_Disabled(t *testing.T) {
	m := &chatModel{id: "claude-opus-5", opts: options{nativeOutputFormatModels: []string{}}}
	if m.supportsNativeOutputFormat() {
		t.Error("native structured output must be disabled when the adapter opts out")
	}
}

func TestTransformNativeOutputSchema(t *testing.T) {
	in := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"confidence"},
		"properties": map[string]any{
			"confidence": map[string]any{
				"type": "number", "minimum": 0.0, "maximum": 1.0,
			},
			"scenario": map[string]any{
				"type": "string", "minLength": 1.0, "format": "uuid",
			},
			"severity": map[string]any{
				"type": "string", "enum": []any{"high", "low"},
			},
			"tags": map[string]any{
				"type": "array", "minItems": 1.0, "uniqueItems": true,
				"items": map[string]any{"type": "string", "maxLength": 8.0},
			},
			"either": map[string]any{
				"anyOf": []any{
					map[string]any{"type": "integer", "multipleOf": 2.0},
					map[string]any{"type": "null"},
				},
			},
		},
	}

	gotVal, err := transformNativeOutputSchema(in)
	if err != nil {
		t.Fatalf("transformNativeOutputSchema: %v", err)
	}
	got, ok := gotVal.(map[string]any)
	if !ok {
		t.Fatalf("transformNativeOutputSchema returned %T, want map", gotVal)
	}

	// Supported constructs must survive untouched.
	props := got["properties"].(map[string]any)
	if props["confidence"].(map[string]any)["type"] != "number" {
		t.Errorf("confidence type lost: %#v", props["confidence"])
	}
	if props["scenario"].(map[string]any)["format"] != "uuid" {
		t.Errorf("string format must survive: %#v", props["scenario"])
	}
	if enum := props["severity"].(map[string]any)["enum"].([]any); len(enum) != 2 {
		t.Errorf("enum must survive: %#v", enum)
	}
	if got["additionalProperties"] != false {
		t.Errorf("additionalProperties:false must survive: %#v", got["additionalProperties"])
	}
	if req := got["required"].([]any); len(req) != 1 || req[0] != "confidence" {
		t.Errorf("required must survive: %#v", req)
	}
	if anyOf := props["either"].(map[string]any)["anyOf"].([]any); len(anyOf) != 2 {
		t.Errorf("anyOf must survive: %#v", anyOf)
	}
	// minItems 0/1 is supported and must be preserved, not dropped.
	if tags := props["tags"].(map[string]any); tags["minItems"] != float64(1) {
		t.Errorf("minItems:1 must survive: %#v", tags)
	}

	// Unsupported constraints must not appear as schema keys, but must be
	// recorded in a description rather than silently dropped.
	collected := collectDescriptions(got)
	for _, kw := range []string{"minimum", "maximum", "minLength", "uniqueItems", "multipleOf", "maxLength"} {
		if keyPresent(got, kw) {
			t.Errorf("unsupported keyword %q survived as a schema key", kw)
		}
		if !strings.Contains(collected, kw) {
			t.Errorf("unsupported keyword %q not recorded in a description", kw)
		}
	}

	// Input must not be mutated.
	origConf := in["properties"].(map[string]any)["confidence"].(map[string]any)
	if _, stillThere := origConf["minimum"]; !stillThere {
		t.Error("transformNativeOutputSchema mutated its input")
	}
}

// keyPresent reports whether the keyword appears as a schema key anywhere.
func keyPresent(node any, keyword string) bool {
	switch n := node.(type) {
	case map[string]any:
		if _, ok := n[keyword]; ok {
			return true
		}
		for _, v := range n {
			if keyPresent(v, keyword) {
				return true
			}
		}
	case []any:
		for _, v := range n {
			if keyPresent(v, keyword) {
				return true
			}
		}
	}
	return false
}

// collectDescriptions concatenates every "description" value in the schema.
func collectDescriptions(node any) string {
	var sb strings.Builder
	var walk func(any)
	walk = func(n any) {
		switch v := n.(type) {
		case map[string]any:
			if d, ok := v["description"].(string); ok {
				sb.WriteString(d)
				sb.WriteString("\n")
			}
			for _, sub := range v {
				walk(sub)
			}
		case []any:
			for _, sub := range v {
				walk(sub)
			}
		}
	}
	walk(node)
	return sb.String()
}

func TestBuildRequest_NativeOutputFormat_SanitisesSchema(t *testing.T) {
	m := &chatModel{id: "claude-opus-5", opts: options{baseURL: defaultBaseURL}}

	// A constrained schema of exactly the shape the API rejects.
	schema := json.RawMessage(`{"type":"object","additionalProperties":false,` +
		`"properties":{"confidence":{"type":"number","minimum":0,"maximum":1}},` +
		`"required":["confidence"]}`)
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat:  &provider.ResponseFormat{Schema: schema},
		ProviderOptions: map[string]any{"structuredOutputMode": "outputFormat"},
	}

	var err error
	params, err = injectNativeOutputFormat(params)
	if err != nil {
		t.Fatalf("injectNativeOutputFormat: %v", err)
	}
	body := m.buildRequest(params, false)

	raw, err := json.Marshal(body["output_config"])
	if err != nil {
		t.Fatal(err)
	}
	// The unsupported numeric constraints must not appear as schema keys.
	for _, kw := range []string{`"minimum":`, `"maximum":`} {
		if strings.Contains(string(raw), kw) {
			t.Errorf("output_config still carries %q: %s", kw, raw)
		}
	}
	// The schema body and format type must still be present.
	if !strings.Contains(string(raw), `"confidence"`) {
		t.Errorf("transformation dropped the schema body: %s", raw)
	}
	if !strings.Contains(string(raw), `"json_schema"`) {
		t.Errorf("output_config.format.type lost: %s", raw)
	}
}

func TestInjectNativeOutputFormat_InvalidSchema_Errors(t *testing.T) {
	params := provider.GenerateParams{
		ResponseFormat:  &provider.ResponseFormat{Schema: json.RawMessage(`{not json`)},
		ProviderOptions: map[string]any{"structuredOutputMode": "outputFormat"},
	}
	if _, err := injectNativeOutputFormat(params); err == nil {
		t.Error("expected an error for an invalid response format schema, got nil")
	}
}

// TestTransformNativeOutputSchema_PropertyNameCollision covers the case the
// first version of the sanitiser got wrong: a data-model field whose name
// happens to match a constraint keyword must survive, because keys inside
// "properties"/"$defs"/etc. are caller-chosen names, not validation keywords.
// Deleting them silently changed the requested shape and could leave
// "required" pointing at a property that no longer existed.
func TestTransformNativeOutputSchema_PropertyNameCollision(t *testing.T) {
	in := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"minimum", "maximum", "uniqueItems"},
		"properties": map[string]any{
			// Field names colliding with every stripped keyword.
			"minimum":     map[string]any{"type": "number", "minimum": 0.0},
			"maximum":     map[string]any{"type": "number", "maximum": 1.0},
			"uniqueItems": map[string]any{"type": "boolean"},
			"minLength":   map[string]any{"type": "integer", "minLength": 3.0},
		},
		"$defs": map[string]any{
			"multipleOf": map[string]any{"type": "string", "maxLength": 4.0},
		},
		"patternProperties": map[string]any{
			"^minItems$": map[string]any{"type": "string", "minLength": 1.0},
		},
	}

	gotVal, err := transformNativeOutputSchema(in)
	if err != nil {
		t.Fatalf("transformNativeOutputSchema: %v", err)
	}
	got := gotVal.(map[string]any)

	// Every property name survives.
	props := got["properties"].(map[string]any)
	for _, name := range []string{"minimum", "maximum", "uniqueItems", "minLength"} {
		if _, ok := props[name]; !ok {
			t.Errorf("property %q was deleted; keys inside properties are names, not keywords", name)
		}
	}
	if _, ok := got["$defs"].(map[string]any)["multipleOf"]; !ok {
		t.Error(`$defs entry "multipleOf" was deleted`)
	}

	// required stays satisfiable: every name in it still exists in properties.
	for _, r := range got["required"].([]any) {
		if _, ok := props[r.(string)]; !ok {
			t.Errorf("required references %q which is no longer in properties", r)
		}
	}

	// Constraints in genuine keyword position (siblings of "type") are not
	// carried as keys; they are folded into a description. Note the property
	// NAMES above legitimately collide with these keywords, so the check is
	// scoped to each property's own schema, not the whole tree.
	for _, name := range []string{"minimum", "maximum", "minLength"} {
		if _, ok := props[name].(map[string]any)[name]; ok {
			t.Errorf("constraint %q in keyword position inside property %q survived as a schema key", name, name)
		}
	}
	if _, ok := got["$defs"].(map[string]any)["multipleOf"].(map[string]any)["maxLength"]; ok {
		t.Error("maxLength constraint inside a $defs entry survived as a schema key")
	}
	// patternProperties is not a supported construct; it is recorded in the
	// description rather than preserved as a key.
	if _, ok := got["patternProperties"]; ok {
		t.Error("patternProperties must not survive as a schema key")
	}
	if !strings.Contains(collectDescriptions(got), "patternProperties") {
		t.Error("patternProperties not recorded in a description")
	}

	// A malformed name-keyed value must not panic.
	if _, err := transformNativeOutputSchema(map[string]any{"properties": "not-a-map"}); err == nil {
		t.Error("expected an error for a non-object properties value")
	}
}

// sseFrames joins event JSON documents into an SSE body.
func sseFrames(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\n\n")
	}
	return b.String()
}

// TestAccumulateStreamedMessage_Parity is the load-bearing test for the
// streaming transport: the same Message delivered as an event stream and as a
// single non-streaming response must parse to identical results. If this
// holds, DoGenerate's two transports are interchangeable by construction and
// no field can be silently lost in reassembly.
//
// Tool inputs in the complete fixtures are written in Go's canonical
// marshalling form (keys sorted, no whitespace) because the streamed side is
// re-marshalled from a decoded map; a semantically equal but differently
// formatted fixture would fail on json.RawMessage byte comparison.
func TestAccumulateStreamedMessage_Parity(t *testing.T) {
	tests := []struct {
		name     string
		stream   string
		complete string
	}{
		{
			name: "text only",
			stream: sseFrames(
				`{"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-5","type":"message","role":"assistant","content":[],"usage":{"input_tokens":15,"cache_read_input_tokens":5}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}`,
				`{"type":"message_stop"}`,
			),
			complete: `{"id":"msg_1","model":"claude-sonnet-5","type":"message","role":"assistant","content":[{"type":"text","text":"Hello world"}],"stop_reason":"end_turn","usage":{"input_tokens":15,"cache_read_input_tokens":5,"output_tokens":8}}`,
		},
		{
			name: "thinking block with signature is preserved verbatim",
			stream: sseFrames(
				`{"type":"message_start","message":{"id":"msg_2","model":"claude-opus-5","type":"message","role":"assistant","content":[],"usage":{"input_tokens":20}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me "}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"consider."}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIY"}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"AhgCIkAw"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer."}}`,
				`{"type":"content_block_stop","index":1}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":30,"output_tokens_details":{"thinking_tokens":22}}}`,
				`{"type":"message_stop"}`,
			),
			complete: `{"id":"msg_2","model":"claude-opus-5","type":"message","role":"assistant","content":[{"type":"thinking","thinking":"Let me consider.","signature":"EqQBCgIYAhgCIkAw"},{"type":"text","text":"Answer."}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":30,"output_tokens_details":{"thinking_tokens":22}}}`,
		},
		{
			name: "tool_use input reassembled from fragments",
			stream: sseFrames(
				`{"type":"message_start","message":{"id":"msg_3","model":"claude-sonnet-5","type":"message","role":"assistant","content":[],"usage":{"input_tokens":30}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read","input":{}}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":\"a.go\"}"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}`,
				`{"type":"message_stop"}`,
			),
			complete: `{"id":"msg_3","model":"claude-sonnet-5","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read","input":{"path":"a.go"}}],"stop_reason":"tool_use","usage":{"input_tokens":30,"output_tokens":12}}`,
		},
		{
			name: "redacted_thinking arrives complete in content_block_start",
			stream: sseFrames(
				`{"type":"message_start","message":{"id":"msg_4","model":"claude-opus-5","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"EncryptedPayload"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
				`{"type":"message_stop"}`,
			),
			complete: `{"id":"msg_4","model":"claude-opus-5","type":"message","role":"assistant","content":[{"type":"redacted_thinking","data":"EncryptedPayload"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":4}}`,
		},
		{
			name: "usage iterations and context_management",
			stream: sseFrames(
				`{"type":"message_start","message":{"id":"msg_5","model":"claude-sonnet-5","type":"message","role":"assistant","content":[],"usage":{"input_tokens":5}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3,"iterations":[{"type":"assistant","input_tokens":5,"output_tokens":3}]},"context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919"}]}}`,
				`{"type":"message_stop"}`,
			),
			complete: `{"id":"msg_5","model":"claude-sonnet-5","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3,"iterations":[{"type":"assistant","input_tokens":5,"output_tokens":3}]},"context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919"}]}}`,
		},
		{
			name: "citations arrive via citations_delta",
			stream: sseFrames(
				`{"type":"message_start","message":{"id":"msg_6","model":"claude-sonnet-5","type":"message","role":"assistant","content":[],"usage":{"input_tokens":5}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"See source."}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web","cited_text":"src","url":"https://example.com","title":"Example"}}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
				`{"type":"message_stop"}`,
			),
			complete: `{"id":"msg_6","model":"claude-sonnet-5","type":"message","role":"assistant","content":[{"type":"text","text":"See source.","citations":[{"type":"web","cited_text":"src","url":"https://example.com","title":"Example"}]}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":4}}`,
		},
		{
			name: "server tool use and its result block",
			stream: sseFrames(
				`{"type":"message_start","message":{"id":"msg_7","model":"claude-sonnet-5","type":"message","role":"assistant","content":[],"usage":{"input_tokens":5}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"toolu_web","name":"web_search","input":{}}}`,
				`{"type":"content_block_stop","index":0}`,
				`{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"toolu_web","content":[{"type":"text","text":"result"}]}}`,
				`{"type":"content_block_stop","index":1}`,
				`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}`,
				`{"type":"message_stop"}`,
			),
			complete: `{"id":"msg_7","model":"claude-sonnet-5","type":"message","role":"assistant","content":[{"type":"server_tool_use","id":"toolu_web","name":"web_search","input":{}},{"type":"web_search_tool_result","tool_use_id":"toolu_web","content":[{"type":"text","text":"result"}]}],"stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":4}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			accumulated, err := accumulateStreamedMessage(t.Context(), strings.NewReader(tt.stream))
			if err != nil {
				t.Fatalf("accumulateStreamedMessage: %v", err)
			}

			gotStreamed, err := parseResponse(accumulated)
			if err != nil {
				t.Fatalf("parseResponse(streamed): %v", err)
			}
			gotComplete, err := parseResponse([]byte(tt.complete))
			if err != nil {
				t.Fatalf("parseResponse(complete): %v", err)
			}

			if !reflect.DeepEqual(gotStreamed, gotComplete) {
				t.Errorf("streamed and non-streaming results differ\nstreamed: %#v\ncomplete: %#v\nreassembled JSON: %s",
					gotStreamed, gotComplete, accumulated)
			}
		})
	}
}

func TestAccumulateStreamedMessage_ErrorEvent(t *testing.T) {
	stream := sseFrames(
		`{"type":"message_start","message":{"id":"msg_e","model":"claude-sonnet-5","content":[]}}`,
		`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
	)

	body, err := accumulateStreamedMessage(t.Context(), strings.NewReader(stream))
	if err != nil {
		t.Fatalf("accumulateStreamedMessage: %v", err)
	}

	// The error envelope is handed to parseResponse unchanged so that error
	// classification stays in one place.
	if _, err = parseResponse(body); err == nil {
		t.Fatal("expected parseResponse to report the streamed error event")
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *goai.APIError, got %T: %v", err, err)
	}
	if !strings.Contains(apiErr.Message, "Overloaded") {
		t.Errorf("error message = %q, want it to mention Overloaded", apiErr.Message)
	}
}

func TestAccumulateStreamedMessage_NoMessageStart(t *testing.T) {
	if _, err := accumulateStreamedMessage(t.Context(), strings.NewReader("")); err == nil {
		t.Fatal("expected an error when the stream carries no message_start")
	}
}

func TestAccumulateStreamedMessage_TruncatedStream(t *testing.T) {
	// A stream cut off before message_stop is a protocol error: reporting the
	// partial text as a complete generation would hide a truncated response.
	stream := sseFrames(
		`{"type":"message_start","message":{"id":"msg_t","model":"claude-sonnet-5","type":"message","content":[],"usage":{"input_tokens":9}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
	)

	_, err := accumulateStreamedMessage(t.Context(), strings.NewReader(stream))
	if err == nil {
		t.Fatal("expected a protocol error for a stream truncated before message_stop")
	}
	if !strings.Contains(err.Error(), "message_stop") {
		t.Errorf("error = %q, want it to mention message_stop", err.Error())
	}
}

func TestAccumulateStreamedMessage_UnclosedBlock(t *testing.T) {
	// A stream that stops after starting a content block but never stops it is
	// malformed even if message_stop arrives.
	stream := sseFrames(
		`{"type":"message_start","message":{"id":"msg_b","model":"claude-sonnet-5","type":"message","content":[],"usage":{"input_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
		`{"type":"message_stop"}`,
	)
	_, err := accumulateStreamedMessage(t.Context(), strings.NewReader(stream))
	if err == nil {
		t.Fatal("expected a protocol error for an unclosed content block")
	}
	if !strings.Contains(err.Error(), "unclosed") {
		t.Errorf("error = %q, want it to mention an unclosed block", err.Error())
	}
}

func TestAccumulateStreamedMessage_MalformedFrame(t *testing.T) {
	// A frame that is not valid JSON is a protocol error, not a frame to skip.
	stream := sseFrames(
		`{"type":"message_start","message":{"id":"msg_m","model":"claude-sonnet-5","type":"message","content":[],"usage":{"input_tokens":1}}}`,
		`not json at all`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"fine"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	)

	_, err := accumulateStreamedMessage(t.Context(), strings.NewReader(stream))
	if err == nil {
		t.Fatal("expected a protocol error for a malformed stream frame")
	}
	if !strings.Contains(err.Error(), "malformed") {
		t.Errorf("error = %q, want it to mention a malformed frame", err.Error())
	}
}

func TestAccumulateStreamedMessage_UnparseableToolInput(t *testing.T) {
	// Fragments that never form valid JSON are a malformed stream and must not
	// be reported as a successful tool call with {} input.
	stream := sseFrames(
		`{"type":"message_start","message":{"id":"msg_u","model":"claude-sonnet-5","type":"message","content":[],"usage":{"input_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_9","name":"read","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	)

	_, err := accumulateStreamedMessage(t.Context(), strings.NewReader(stream))
	if err == nil {
		t.Fatal("expected a protocol error for unparseable tool input")
	}
	if !strings.Contains(err.Error(), "tool input") {
		t.Errorf("error = %q, want it to mention tool input", err.Error())
	}
}

func TestAccumulateStreamedMessage_BoundedIndex(t *testing.T) {
	// A hostile or corrupted index must be rejected, not trigger unbounded
	// allocation.
	for _, idx := range []string{"1000000000", "-1", "0.5"} {
		stream := sseFrames(
			`{"type":"message_start","message":{"id":"msg_i","model":"claude-sonnet-5","type":"message","content":[]}}`,
			`{"type":"content_block_start","index":`+idx+`,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_stop","index":`+idx+`}`,
			`{"type":"message_stop"}`,
		)
		if _, err := accumulateStreamedMessage(t.Context(), strings.NewReader(stream)); err == nil {
			t.Errorf("expected a protocol error for content block index %s", idx)
		}
	}
}

func TestWillThink(t *testing.T) {
	tests := []struct {
		name    string
		modelID string
		opts    map[string]any
		want    bool
	}{
		{name: "5.x thinks with no parameter at all", modelID: "claude-sonnet-5", want: true},
		{name: "opus 5 thinks by default", modelID: "claude-opus-5", want: true},
		{name: "fable 5 thinks by default", modelID: "claude-fable-5", want: true},
		{name: "4.8 only on request", modelID: "claude-opus-4-8", want: false},
		{
			name:    "4.8 with explicit thinking",
			modelID: "claude-opus-4-8",
			opts:    map[string]any{"thinking": map[string]any{"type": "adaptive"}},
			want:    true,
		},
		{
			name:    "4.7 with effort",
			modelID: "claude-opus-4-7",
			opts:    map[string]any{"effort": "medium"},
			want:    true,
		},
		{
			name:    "4.6 with budget_tokens thinking",
			modelID: "claude-sonnet-4-6",
			opts:    map[string]any{"thinking": map[string]any{"type": "enabled", "budgetTokens": 4000}},
			want:    true,
		},
		{name: "4.6 without thinking", modelID: "claude-sonnet-4-6", want: false},
		{
			name:    "model that cannot think ignores the option",
			modelID: "claude-3-5-sonnet-20241022",
			opts:    map[string]any{"thinking": map[string]any{"type": "enabled"}},
			want:    false,
		},
		{name: "bedrock-prefixed 5.x still detected", modelID: "us.anthropic.claude-sonnet-5", want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &chatModel{id: tt.modelID}
			got := m.willThink(provider.GenerateParams{ProviderOptions: tt.opts})
			if got != tt.want {
				t.Errorf("willThink(%q) = %v, want %v", tt.modelID, got, tt.want)
			}
		})
	}
}

func TestUseStreamingTransport_ExplicitOverride(t *testing.T) {
	tests := []struct {
		name    string
		modelID string
		opts    map[string]any
		want    bool
		wantErr bool
	}{
		{
			name:    "explicit false beats thinks-by-default",
			modelID: "claude-sonnet-5",
			opts:    map[string]any{"streamingTransport": false},
			want:    false,
		},
		{
			name:    "explicit true beats a non-thinking model",
			modelID: "claude-sonnet-4-6",
			opts:    map[string]any{"streamingTransport": true},
			want:    true,
		},
		{
			name:    "non-bool value is rejected",
			modelID: "claude-sonnet-5",
			opts:    map[string]any{"streamingTransport": "yes"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &chatModel{id: tt.modelID}
			got, err := m.useStreamingTransport(provider.GenerateParams{ProviderOptions: tt.opts})
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected an error for a non-boolean streamingTransport, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("useStreamingTransport: %v", err)
			}
			if got != tt.want {
				t.Errorf("useStreamingTransport = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestUseStreamingTransport_ProviderAware(t *testing.T) {
	// Auto-streaming is disabled for adapters that opt out (e.g. Bedrock,
	// whose streaming endpoint needs a different IAM permission), even for a
	// model that thinks by default.
	m := &chatModel{id: "claude-sonnet-5", opts: options{autoStreaming: false}}
	got, err := m.useStreamingTransport(provider.GenerateParams{})
	if err != nil {
		t.Fatalf("useStreamingTransport: %v", err)
	}
	if got {
		t.Error("auto-streaming must be disabled when the adapter opts out")
	}
	// An explicit override still wins.
	got, err = m.useStreamingTransport(provider.GenerateParams{ProviderOptions: map[string]any{"streamingTransport": true}})
	if err != nil {
		t.Fatalf("useStreamingTransport: %v", err)
	}
	if !got {
		t.Error("explicit streamingTransport=true must win over the adapter opt-out")
	}
}

// TestDoGenerate_UsesStreamingTransportForThinkingModels is the end-to-end
// check: a thinking model's DoGenerate must request stream:true, consume SSE,
// and still return a normal GenerateResult — including the thinking signature
// that provider.StreamChunk cannot carry.
func TestDoGenerate_UsesStreamingTransportForThinkingModels(t *testing.T) {
	var gotStream any
	var gotStreamingTransportOnWire bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotStream = body["stream"]
		_, gotStreamingTransportOnWire = body["streamingTransport"]

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseFrames(
			`{"type":"message_start","message":{"id":"msg_e2e","model":"claude-sonnet-5","type":"message","role":"assistant","content":[],"usage":{"input_tokens":11}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SigABC"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`,
			`{"type":"message_stop"}`,
		))
	}))
	defer server.Close()

	model := Chat("claude-sonnet-5", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	if gotStream != true {
		t.Errorf("request stream = %v, want true", gotStream)
	}
	if gotStreamingTransportOnWire {
		t.Error("streamingTransport leaked onto the wire; it is an SDK-internal key")
	}
	if result.Text != "done" {
		t.Errorf("Text = %q, want %q", result.Text, "done")
	}
	if result.Reasoning != "hmm" {
		t.Errorf("Reasoning = %q, want %q", result.Reasoning, "hmm")
	}
	if result.Usage.OutputTokens != 7 {
		t.Errorf("OutputTokens = %d, want 7", result.Usage.OutputTokens)
	}
	// The signature is the field a chunk-based reassembly would have dropped.
	if !strings.Contains(fmt.Sprint(result.ProviderMetadata), "SigABC") {
		t.Errorf("thinking signature missing from ProviderMetadata: %v", result.ProviderMetadata)
	}
}

func TestDoGenerate_NonThinkingModelStaysNonStreaming(t *testing.T) {
	var gotStream any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotStream = body["stream"]

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_ns","model":"claude-sonnet-4-6","type":"message","role":"assistant","content":[{"type":"text","text":"plain"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-6", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}

	if gotStream != false {
		t.Errorf("request stream = %v, want false for a non-thinking model", gotStream)
	}
	if result.Text != "plain" {
		t.Errorf("Text = %q, want %q", result.Text, "plain")
	}
}

func TestSupportsThinking_NewModels(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"claude-3-7-sonnet-20250219", true},
		{"claude-sonnet-4-20250514", true},
		{"claude-sonnet-4-6", true},
		{"claude-opus-4-20250514", true},
		{"claude-opus-4-8", true},
		{"claude-opus-4-7", true},
		{"claude-opus-4-6", true},
		{"claude-haiku-4-5", true},
		{"claude-sonnet-5", true},
		{"claude-opus-5", true},
		{"claude-3-5-haiku-20241022", false},
		{"claude-3-haiku-20240307", false},
		{"gpt-4o", false},
	}
	for _, c := range cases {
		if got := supportsThinking(c.model); got != c.want {
			t.Errorf("supportsThinking(%q) = %v, want %v", c.model, got, c.want)
		}
	}
}

// #14/#15/#16 -- feature-gated beta headers are only added when the feature is
// present, and claude-code-20250219 is never sent by default.
func TestCollectRequestBetas(t *testing.T) {
	// Plain request: no extra betas.
	if got := collectRequestBetas(map[string]any{"model": "x"}); len(got) != 0 {
		t.Errorf("plain request betas = %v, want none", got)
	}

	// context_management -> context-1m beta.
	if got := collectRequestBetas(map[string]any{"context_management": map[string]any{}}); len(got) != 1 || got[0] != betaContextManagement {
		t.Errorf("context_management betas = %v, want [%s]", got, betaContextManagement)
	}

	// speed -> fast-mode beta.
	if got := collectRequestBetas(map[string]any{"speed": "fast"}); len(got) != 1 || got[0] != betaFastMode {
		t.Errorf("speed betas = %v, want [%s]", got, betaFastMode)
	}

	// container -> claude-code beta.
	if got := collectRequestBetas(map[string]any{"container": map[string]any{}}); len(got) != 1 || got[0] != betaClaudeCode {
		t.Errorf("container betas = %v, want [%s]", got, betaClaudeCode)
	}

	// Multiple features combine.
	got := collectRequestBetas(map[string]any{
		"context_management": map[string]any{},
		"speed":              "fast",
	})
	if len(got) != 2 {
		t.Errorf("combined betas = %v, want 2", got)
	}
}

// #16 -- claude-code-20250219 must NOT be part of the default beta header.
func TestBetaFeatures_ExcludesClaudeCode(t *testing.T) {
	if strings.Contains(betaFeatures, betaClaudeCode) {
		t.Errorf("betaFeatures = %q must not contain %q", betaFeatures, betaClaudeCode)
	}
}

// #14 -- context_management request must carry the context-1m beta header.
func TestDoGenerate_ContextManagementBetaHeader(t *testing.T) {
	var betaHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		betaHeader = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"contextManagement": map[string]any{
				"edits": []any{map[string]any{"type": "clear_tool_uses_20250919"}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(betaHeader, betaContextManagement) {
		t.Errorf("anthropic-beta = %q, want it to contain %q", betaHeader, betaContextManagement)
	}
}

// #15 -- speed request must carry the fast-mode beta header.
func TestDoGenerate_SpeedBetaHeader(t *testing.T) {
	var betaHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		betaHeader = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"speed": "fast",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(betaHeader, betaFastMode) {
		t.Errorf("anthropic-beta = %q, want it to contain %q", betaHeader, betaFastMode)
	}
}

// #16 -- a plain request must NOT carry claude-code-20250219.
func TestDoGenerate_NoClaudeCodeBetaOnPlainRequest(t *testing.T) {
	var betaHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		betaHeader = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("claude-sonnet-4-20250514", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(betaHeader, betaClaudeCode) {
		t.Errorf("anthropic-beta = %q, must not contain %q on a plain request", betaHeader, betaClaudeCode)
	}
}

// #17 -- a PartImage carrying a RemoteRef emits a file-backed image block.
func TestConvertMessages_ImageRemoteRef(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartImage, RemoteRef: &provider.RemoteFileRef{ID: "file_img_123"}},
		}},
	})

	content := msgs[0]["content"].([]map[string]any)
	if content[0]["type"] != "image" {
		t.Fatalf("type = %v, want image", content[0]["type"])
	}
	source, _ := content[0]["source"].(map[string]any)
	if source["type"] != "file" {
		t.Errorf("source.type = %v, want file", source["type"])
	}
	if source["file_id"] != "file_img_123" {
		t.Errorf("source.file_id = %v, want file_img_123", source["file_id"])
	}
}

// #18 -- thinking enabled downgrades a forced tool_choice to auto.
func TestBuildRequest_ThinkingDowngradesForcedToolChoice(t *testing.T) {
	m := &chatModel{id: "claude-sonnet-4-20250514", opts: options{baseURL: defaultBaseURL}}

	// required -> auto
	body := m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "required",
		ProviderOptions: map[string]any{
			"thinking": map[string]any{"type": "enabled", "budgetTokens": float64(2000)},
		},
	}, false)
	tc, _ := body["tool_choice"].(map[string]any)
	if tc["type"] != "auto" {
		t.Errorf("tool_choice = %v, want auto (downgraded from required)", tc)
	}
	if body["thinking"] == nil {
		t.Error("thinking should still be set")
	}

	// specific tool -> auto
	body = m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "read_file",
		ProviderOptions: map[string]any{
			"thinking": map[string]any{"type": "enabled"},
		},
	}, false)
	tc, _ = body["tool_choice"].(map[string]any)
	if tc["type"] != "auto" {
		t.Errorf("tool_choice = %v, want auto (downgraded from specific tool)", tc)
	}

	// adaptive thinking must NOT downgrade.
	body = m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "required",
		ProviderOptions: map[string]any{
			"thinking": map[string]any{"type": "adaptive"},
		},
	}, false)
	tc, _ = body["tool_choice"].(map[string]any)
	if tc["type"] != "any" {
		t.Errorf("tool_choice = %v, want any (adaptive thinking must not downgrade)", tc)
	}
}
