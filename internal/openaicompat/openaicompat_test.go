package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// errorReader is an io.Reader that returns an error after some data.
type errorReader struct {
	data string
	pos  int
	err  error
}

func (r *errorReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, r.err
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	if r.pos >= len(r.data) {
		return n, r.err
	}
	return n, nil
}

var _ io.Reader = (*errorReader)(nil)

func TestBuildRequest_Basic(t *testing.T) {
	temp := 0.7
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
		},
		System:          "You are helpful.",
		MaxOutputTokens: 1000,
		Temperature:     &temp,
	}

	body := BuildRequest(params, "gpt-4o", true, RequestConfig{IncludeStreamOptions: true})

	if body["model"] != "gpt-4o" {
		t.Errorf("model = %v", body["model"])
	}
	if body["stream"] != true {
		t.Error("expected stream=true")
	}
	if body["max_tokens"] != 1000 {
		t.Errorf("max_tokens = %v", body["max_tokens"])
	}
	if body["temperature"] != 0.7 {
		t.Errorf("temperature = %v", body["temperature"])
	}

	so, ok := body["stream_options"].(map[string]any)
	if !ok {
		t.Fatal("missing stream_options")
	}
	if so["include_usage"] != true {
		t.Error("expected include_usage=true")
	}
}

func TestBuildRequest_WithTools(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read file"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "read_file", Description: "Read a file", InputSchema: schema},
		},
		ToolChoice: "auto",
	}

	body := BuildRequest(params, "gpt-4o", true, RequestConfig{})

	tools, ok := body["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %v", body["tools"])
	}
	fn := tools[0]["function"].(map[string]any)
	if fn["name"] != "read_file" {
		t.Errorf("tool name = %v", fn["name"])
	}
	if body["tool_choice"] != "auto" {
		t.Errorf("tool_choice = %v", body["tool_choice"])
	}
}

func TestBuildRequest_ToolChoiceSpecific(t *testing.T) {
	params := provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "read_file",
	}
	body := BuildRequest(params, "gpt-4o", true, RequestConfig{})

	tc, ok := body["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("expected map tool_choice, got %T", body["tool_choice"])
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "read_file" {
		t.Errorf("tool_choice function name = %v", fn["name"])
	}
}

func TestBuildRequest_StopSequences(t *testing.T) {
	params := provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		StopSequences: []string{"END", "STOP"},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})

	stop, ok := body["stop"].([]string)
	if !ok || len(stop) != 2 {
		t.Fatalf("expected 2 stop sequences, got %v", body["stop"])
	}
}

func TestBuildRequest_InvalidToolSchema(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools: []provider.ToolDefinition{
			{Name: "broken", Description: "broken tool", InputSchema: json.RawMessage(`not valid json`)},
		},
	}
	body := BuildRequest(params, "gpt-4o", true, RequestConfig{})

	tools, ok := body["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %v", body["tools"])
	}
	// Should fallback to empty object, not nil or invalid bytes
	fn := tools[0]["function"].(map[string]any)
	schema, ok := fn["parameters"].(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any parameters, got %T", fn["parameters"])
	}
	if len(schema) != 0 {
		t.Errorf("expected empty object fallback, got %v", schema)
	}
}

func TestBuildRequest_ExtraBody(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	}
	body := BuildRequest(params, "gpt-4o", true, RequestConfig{
		ExtraBody: map[string]any{"store": false, "custom_field": "value"},
	})

	if body["store"] != false {
		t.Errorf("store = %v", body["store"])
	}
	if body["custom_field"] != "value" {
		t.Errorf("custom_field = %v", body["custom_field"])
	}
}

func TestBuildRequest_ProviderOptions(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"reasoning_effort": "high", "prompt_cache_key": "abc"},
	}
	body := BuildRequest(params, "gpt-4o", true, RequestConfig{})

	if body["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v", body["reasoning_effort"])
	}
	if body["prompt_cache_key"] != "abc" {
		t.Errorf("prompt_cache_key = %v", body["prompt_cache_key"])
	}
}

func TestConvertMessages_SystemAndUser(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
	}
	result := ConvertMessages(msgs, "You are helpful.")

	if len(result) != 2 {
		t.Fatalf("got %d messages, want 2", len(result))
	}
	if result[0]["role"] != "system" {
		t.Errorf("first message role = %v", result[0]["role"])
	}
	if result[0]["content"] != "You are helpful." {
		t.Errorf("system content = %v", result[0]["content"])
	}
	if result[1]["role"] != "user" {
		t.Errorf("user message role = %v", result[1]["role"])
	}
}

func TestConvertMessages_ToolResults(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleTool,
			Content: []provider.Part{
				{Type: provider.PartToolResult, ToolCallID: "call_1", ToolOutput: "file contents"},
			},
		},
	}
	result := ConvertMessages(msgs, "")

	if len(result) != 1 {
		t.Fatalf("got %d messages, want 1", len(result))
	}
	if result[0]["role"] != "tool" {
		t.Errorf("role = %v", result[0]["role"])
	}
	if result[0]["tool_call_id"] != "call_1" {
		t.Errorf("tool_call_id = %v", result[0]["tool_call_id"])
	}
}

func TestConvertMessages_AssistantWithToolCalls(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "Let me read that."},
				{Type: provider.PartToolCall, ToolCallID: "call_1", ToolName: "read", ToolInput: json.RawMessage(`{"path":"main.go"}`)},
			},
		},
	}
	result := ConvertMessages(msgs, "")

	if len(result) != 1 {
		t.Fatalf("got %d messages, want 1", len(result))
	}
	tcs := result[0]["tool_calls"].([]map[string]any)
	if len(tcs) != 1 {
		t.Fatalf("got %d tool_calls, want 1", len(tcs))
	}
	fn := tcs[0]["function"].(map[string]any)
	if fn["name"] != "read" {
		t.Errorf("tool call name = %v", fn["name"])
	}
}

func TestConvertMessages_UserWithImage(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleUser,
			Content: []provider.Part{
				{Type: provider.PartText, Text: "What's in this image?"},
				{Type: provider.PartImage, URL: "data:image/png;base64,abc123"},
			},
		},
	}
	result := ConvertMessages(msgs, "")

	if len(result) != 1 {
		t.Fatalf("got %d messages, want 1", len(result))
	}
	content, ok := result[0]["content"].([]map[string]any)
	if !ok {
		t.Fatalf("expected content array, got %T", result[0]["content"])
	}
	if len(content) != 2 {
		t.Fatalf("got %d content parts, want 2", len(content))
	}
	if content[0]["type"] != "text" {
		t.Errorf("first part type = %v", content[0]["type"])
	}
	if content[1]["type"] != "image_url" {
		t.Errorf("second part type = %v", content[1]["type"])
	}
}

func TestParseStream_TextAndFinish(t *testing.T) {
	input := `data: {"id":"chatcmpl-stream-1","model":"gpt-4o-2024-08-06","choices":[{"delta":{"content":"Hello"},"index":0}]}
data: {"id":"chatcmpl-stream-1","model":"gpt-4o-2024-08-06","choices":[{"delta":{"content":" world"},"index":0}]}
data: {"id":"chatcmpl-stream-1","model":"gpt-4o-2024-08-06","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)

	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	// Should have: text "Hello", text " world", step_finish, finish
	if len(chunks) < 4 {
		t.Fatalf("got %d chunks, want at least 4", len(chunks))
	}

	if chunks[0].Type != provider.ChunkText || chunks[0].Text != "Hello" {
		t.Errorf("chunk[0] = %+v", chunks[0])
	}
	if chunks[1].Type != provider.ChunkText || chunks[1].Text != " world" {
		t.Errorf("chunk[1] = %+v", chunks[1])
	}
	if chunks[2].Type != provider.ChunkStepFinish {
		t.Errorf("chunk[2].Type = %v, want step_finish", chunks[2].Type)
	}
	if chunks[3].Type != provider.ChunkFinish {
		t.Errorf("chunk[3].Type = %v, want finish", chunks[3].Type)
	}
	if chunks[3].Usage.InputTokens != 10 || chunks[3].Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v", chunks[3].Usage)
	}
	if chunks[3].Response.ID != "chatcmpl-stream-1" {
		t.Errorf("Response.ID = %q, want %q", chunks[3].Response.ID, "chatcmpl-stream-1")
	}
	if chunks[3].Response.Model != "gpt-4o-2024-08-06" {
		t.Errorf("Response.Model = %q, want %q", chunks[3].Response.Model, "gpt-4o-2024-08-06")
	}
}

func TestParseStream_ToolCalls(t *testing.T) {
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":""}}]},"index":0}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]},"index":0}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"main.go\"}"}}]},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)

	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	// Find the tool call stream start
	var foundStart, foundCall bool
	for _, c := range chunks {
		if c.Type == provider.ChunkToolCallStreamStart {
			foundStart = true
			if c.ToolCallID != "call_1" || c.ToolName != "read" {
				t.Errorf("start chunk = %+v", c)
			}
		}
		if c.Type == provider.ChunkToolCall {
			foundCall = true
			if c.ToolInput != `{"path":"main.go"}` {
				t.Errorf("tool input = %q", c.ToolInput)
			}
		}
	}
	if !foundStart {
		t.Error("missing tool_call_streaming_start chunk")
	}
	if !foundCall {
		t.Error("missing tool_call chunk")
	}
}

func TestParseStream_Reasoning(t *testing.T) {
	input := `data: {"choices":[{"delta":{"reasoning_content":"Let me think..."},"index":0}]}
data: {"choices":[{"delta":{"content":"The answer is 42."},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)

	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	if chunks[0].Type != provider.ChunkReasoning || chunks[0].Text != "Let me think..." {
		t.Errorf("reasoning chunk = %+v", chunks[0])
	}
	if chunks[1].Type != provider.ChunkText || chunks[1].Text != "The answer is 42." {
		t.Errorf("text chunk = %+v", chunks[1])
	}
}

func TestParseStream_CachedTokens(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"hi"},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":60}}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)

	go ParseStream(t.Context(), scanner, out)

	var finishChunk provider.StreamChunk
	for chunk := range out {
		if chunk.Type == provider.ChunkFinish {
			finishChunk = chunk
		}
	}

	// InputTokens should be 100 - 60 = 40
	if finishChunk.Usage.InputTokens != 40 {
		t.Errorf("InputTokens = %d, want 40", finishChunk.Usage.InputTokens)
	}
	if finishChunk.Usage.CacheReadTokens != 60 {
		t.Errorf("CacheReadTokens = %d, want 60", finishChunk.Usage.CacheReadTokens)
	}
}

func TestParseStream_StreamError(t *testing.T) {
	input := `data: {"type":"error","error":{"code":"context_length_exceeded","message":"too long"}}
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)

	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	if chunks[0].Type != provider.ChunkError {
		t.Errorf("type = %v, want error", chunks[0].Type)
	}
}

func TestParseStream_ContextCancel(t *testing.T) {
	// Use a stream with data so the goroutine enters the loop before seeing cancel.
	input := `data: {"choices":[{"delta":{"content":"hi"},"index":0}]}
data: {"choices":[{"delta":{"content":"there"},"index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)

	ctx, cancel := context.WithCancel(t.Context())
	cancel() // cancel immediately

	// With TrySend, the goroutine exits cleanly when ctx is cancelled
	// instead of sending an error chunk (which could block if buffer is full).
	done := make(chan struct{})
	go func() {
		ParseStream(ctx, scanner, out)
		close(done)
	}()
	<-done

	// Goroutine should have exited without blocking. Any chunks received
	// are either text (if sent before cancel was noticed) or nothing.
	for chunk := range out {
		// Accept any chunk type - the key assertion is that the goroutine didn't leak.
		_ = chunk
	}
}

func TestParseStream_NonOverflowStreamError(t *testing.T) {
	input := `data: {"type":"error","error":{"code":"insufficient_quota","message":"quota exceeded"}}
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	if chunks[0].Type != provider.ChunkError {
		t.Errorf("type = %v, want error", chunks[0].Type)
	}
}

func TestParseStream_CachedTokensExceedPrompt(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"hi"},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":20}}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var finishChunk provider.StreamChunk
	for chunk := range out {
		if chunk.Type == provider.ChunkFinish {
			finishChunk = chunk
		}
	}
	if finishChunk.Usage.InputTokens != 0 {
		t.Errorf("InputTokens = %d, want 0 (clamped)", finishChunk.Usage.InputTokens)
	}
}

func TestParseStream_ScannerError(t *testing.T) {
	reader := &errorReader{
		data: "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"index\":0}]}\n",
		err:  fmt.Errorf("connection reset"),
	}
	scanner := sse.NewScanner(reader)
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	// Should get text chunk then error chunk
	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want at least 2", len(chunks))
	}
	lastChunk := chunks[len(chunks)-1]
	if lastChunk.Type != provider.ChunkError {
		t.Errorf("last chunk type = %v, want error", lastChunk.Type)
	}
}

func TestParseStream_ToolCallArgsWithoutID(t *testing.T) {
	// Tool call arguments arrive without a preceding ID chunk (edge case)
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	// Should not panic -- the nil active guard creates a new activeToolCall
	var foundToolCall bool
	for _, c := range chunks {
		if c.Type == provider.ChunkToolCall {
			foundToolCall = true
		}
	}
	if !foundToolCall {
		t.Error("expected tool call chunk from args without ID")
	}
}

func TestParseResponse_Basic(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-123",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Hello!"},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	result, err := ParseResponse(body)
	if err != nil {
		t.Fatalf("ParseResponse error: %v", err)
	}
	if result.Text != "Hello!" {
		t.Errorf("Text = %q", result.Text)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v", result.Usage)
	}
	if result.Response.ID != "chatcmpl-123" {
		t.Errorf("Response.ID = %q", result.Response.ID)
	}
}

func TestParseResponse_WithToolCalls(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-456",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "",
				"tool_calls": [{
					"id": "call_1",
					"type": "function",
					"function": {"name": "read_file", "arguments": "{\"path\":\"main.go\"}"}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 20, "completion_tokens": 15}
	}`)

	result, err := ParseResponse(body)
	if err != nil {
		t.Fatalf("ParseResponse error: %v", err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "read_file" {
		t.Errorf("tool call name = %q", result.ToolCalls[0].Name)
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
}

func TestParseResponse_CachedTokens(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-789",
		"model": "gpt-4o",
		"choices": [{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 80}}
	}`)

	result, err := ParseResponse(body)
	if err != nil {
		t.Fatalf("ParseResponse error: %v", err)
	}
	if result.Usage.InputTokens != 20 {
		t.Errorf("InputTokens = %d, want 20 (100-80)", result.Usage.InputTokens)
	}
	if result.Usage.CacheReadTokens != 80 {
		t.Errorf("CacheReadTokens = %d, want 80", result.Usage.CacheReadTokens)
	}
}

func TestParseResponse_InvalidJSON(t *testing.T) {
	_, err := ParseResponse([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseResponse_CachedTokensExceedPrompt(t *testing.T) {
	// Edge case: cached_tokens > prompt_tokens (shouldn't happen, but clamp to 0)
	body := []byte(`{
		"id": "x", "model": "gpt-4o",
		"choices": [{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 20}}
	}`)
	result, err := ParseResponse(body)
	if err != nil {
		t.Fatalf("ParseResponse error: %v", err)
	}
	if result.Usage.InputTokens != 0 {
		t.Errorf("InputTokens = %d, want 0 (clamped)", result.Usage.InputTokens)
	}
}

func TestParseResponse_EmptyChoices(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-0","model":"gpt-4o","choices":[]}`)
	result, err := ParseResponse(body)
	if err != nil {
		t.Fatalf("ParseResponse error: %v", err)
	}
	if result.Text != "" {
		t.Errorf("Text = %q, want empty", result.Text)
	}
}

func TestConvertMessages_SystemInMessages(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleSystem, Content: []provider.Part{{Type: provider.PartText, Text: "Be concise."}}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
	}
	result := ConvertMessages(msgs, "You are helpful.")

	// Should have: system (from param), system (from messages), user
	if len(result) != 3 {
		t.Fatalf("got %d messages, want 3", len(result))
	}
	if result[0]["content"] != "You are helpful." {
		t.Errorf("first system = %v", result[0]["content"])
	}
	if result[1]["content"] != "Be concise." {
		t.Errorf("second system = %v", result[1]["content"])
	}
}

func TestConvertMessages_AssistantTextOnly(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartText, Text: "Hello"},
			{Type: provider.PartText, Text: "World"},
		}},
	}
	result := ConvertMessages(msgs, "")
	if result[0]["content"] != "Hello\nWorld" {
		t.Errorf("content = %v, want 'Hello\\nWorld'", result[0]["content"])
	}
}

func TestBuildRequest_NoStreamOptions(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	}
	body := BuildRequest(params, "gpt-4o", true, RequestConfig{IncludeStreamOptions: false})
	if _, ok := body["stream_options"]; ok {
		t.Error("should not have stream_options when IncludeStreamOptions=false")
	}
}

func TestBuildRequest_NonStreaming(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{IncludeStreamOptions: true})
	if _, ok := body["stream_options"]; ok {
		t.Error("should not have stream_options when not streaming")
	}
}

func TestParseStream_EmptyChoices(t *testing.T) {
	input := `data: {"usage":{"prompt_tokens":5,"completion_tokens":0}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	// Should get just a finish chunk
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	if chunks[0].Type != provider.ChunkFinish {
		t.Errorf("type = %v, want finish", chunks[0].Type)
	}
}

func TestParseStream_InvalidJSON(t *testing.T) {
	// Non-JSON, non-error data should be skipped
	input := `data: not-json-at-all
data: {"choices":[{"delta":{"content":"ok"},"index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	// Should skip invalid JSON and get text + finish
	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2", len(chunks))
	}
	if chunks[0].Type != provider.ChunkText {
		t.Errorf("chunk[0].Type = %v", chunks[0].Type)
	}
}

func TestBuildRequest_Headers(t *testing.T) {
	// Covers lines 109-111: _headers injection when params.Headers is set.
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Headers: map[string]string{
			"X-Custom-Header": "test-value",
			"X-Another":       "another-value",
		},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})

	headers, ok := body["_headers"].(map[string]string)
	if !ok {
		t.Fatalf("_headers not set or wrong type: %T", body["_headers"])
	}
	if headers["X-Custom-Header"] != "test-value" {
		t.Errorf("X-Custom-Header = %q", headers["X-Custom-Header"])
	}
	if headers["X-Another"] != "another-value" {
		t.Errorf("X-Another = %q", headers["X-Another"])
	}
}

func TestBuildRequest_NoHeaders(t *testing.T) {
	// When Headers is nil, _headers should NOT be in body.
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if _, ok := body["_headers"]; ok {
		t.Error("_headers should not be set when Headers is nil")
	}
}

func TestBuildRequest_ResponseFormat(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test_schema",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})

	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format not set or wrong type: %T", body["response_format"])
	}
	if rf["type"] != "json_schema" {
		t.Errorf("response_format.type = %v, want json_schema", rf["type"])
	}
	js, ok := rf["json_schema"].(map[string]any)
	if !ok {
		t.Fatalf("json_schema not set or wrong type: %T", rf["json_schema"])
	}
	if js["name"] != "test_schema" {
		t.Errorf("json_schema.name = %v, want test_schema", js["name"])
	}
	// Default strictJsonSchema is false (matching Vercel).
	if js["strict"] != false {
		t.Errorf("json_schema.strict = %v, want false", js["strict"])
	}
	schema, ok := js["schema"].(map[string]any)
	if !ok {
		t.Fatalf("schema not a map: %T", js["schema"])
	}
	if schema["type"] != "object" {
		t.Errorf("schema.type = %v, want object", schema["type"])
	}
}

func TestBuildRequest_ResponseFormat_InvalidSchema(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "broken",
			Schema: json.RawMessage(`not valid json`),
		},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})

	// When schema is invalid JSON, falls back to json_object mode.
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatal("response_format should be set for json_object fallback")
	}
	if rf["type"] != "json_object" {
		t.Errorf("response_format.type = %v, want json_object", rf["type"])
	}
}

func TestBuildRequest_MaxCompletionTokensForReasoning(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		MaxOutputTokens: 2000,
		ProviderOptions: map[string]any{"reasoning_effort": "high"},
	}
	body := BuildRequest(params, "o3", false, RequestConfig{})

	if _, ok := body["max_tokens"]; ok {
		t.Error("max_tokens should be removed when reasoning_effort is set")
	}
	if body["max_completion_tokens"] != 2000 {
		t.Errorf("max_completion_tokens = %v, want 2000", body["max_completion_tokens"])
	}
}

func TestBuildRequest_TopP(t *testing.T) {
	topP := 0.95
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		TopP:     &topP,
	}
	body := BuildRequest(params, "gpt-4o", true, RequestConfig{})
	if body["top_p"] != 0.95 {
		t.Errorf("top_p = %v", body["top_p"])
	}
}

func TestConvertMessages_AssistantToolCallsNoText(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{Type: provider.PartToolCall, ToolCallID: "call_1", ToolName: "read", ToolInput: json.RawMessage(`{"path":"main.go"}`)},
			},
		},
	}
	result := ConvertMessages(msgs, "")

	if len(result) != 1 {
		t.Fatalf("got %d messages, want 1", len(result))
	}
	// Should have null content (not empty string) and tool_calls.
	// OpenAI requires content to be null or omitted when only tool_calls are present.
	if result[0]["content"] != nil {
		t.Errorf("content = %v, want nil (null JSON)", result[0]["content"])
	}
	tcs := result[0]["tool_calls"].([]map[string]any)
	if len(tcs) != 1 {
		t.Fatalf("got %d tool_calls, want 1", len(tcs))
	}
}

func TestParseStream_ToolCallFlushOnFinish(t *testing.T) {
	// Simulate incomplete JSON args that get flushed on finish
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":""}}]},"index":0}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	// Should have: start, tool_call (flushed on finish), step_finish, finish
	var foundToolCall bool
	for _, c := range chunks {
		if c.Type == provider.ChunkToolCall {
			foundToolCall = true
			if c.ToolInput != `{"path":` {
				t.Errorf("flushed tool input = %q", c.ToolInput)
			}
		}
	}
	if !foundToolCall {
		t.Error("expected flushed tool call chunk on finish")
	}
}

// =============================================================================
// BF.13 -- Vercel parity tests
// =============================================================================

// Item 6: image_url.detail field in message format.
func TestBF13_ImageURLDetail(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartText, Text: "What's in this image?"},
			{Type: provider.PartImage, URL: "data:image/png;base64,abc", Detail: "high"},
		}},
	}
	result := ConvertMessages(msgs, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	content, ok := result[0]["content"].([]map[string]any)
	if !ok {
		t.Fatalf("content not array, got %T", result[0]["content"])
	}
	if len(content) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(content))
	}
	imgPart := content[1]
	imgURL, ok := imgPart["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("image_url not map, got %T", imgPart["image_url"])
	}
	if imgURL["detail"] != "high" {
		t.Errorf("detail = %v, want high", imgURL["detail"])
	}
}

// Item 6: image_url.detail not set when empty.
func TestBF13_ImageURLNoDetail(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartText, Text: "What's in this image?"},
			{Type: provider.PartImage, URL: "data:image/png;base64,abc"},
		}},
	}
	result := ConvertMessages(msgs, "")
	content, ok := result[0]["content"].([]map[string]any)
	if !ok {
		t.Fatalf("content not array, got %T", result[0]["content"])
	}
	imgURL, ok := content[1]["image_url"].(map[string]any)
	if !ok {
		t.Fatal("image_url not map")
	}
	if _, hasDetail := imgURL["detail"]; hasDetail {
		t.Errorf("detail should not be set when empty")
	}
}

// Item 1: TotalTokens in non-streaming ParseResponse.
func TestBF13_TotalTokens_ParseResponse(t *testing.T) {
	body := []byte(`{
		"id":"x","model":"m",
		"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}
	}`)
	result, err := ParseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.TotalTokens != 120 {
		t.Errorf("TotalTokens = %d, want 120", result.Usage.TotalTokens)
	}
}

// Item 10: prediction tokens in ParseResponse.
func TestBF13_PredictionTokens_ParseResponse(t *testing.T) {
	body := []byte(`{
		"id":"x","model":"m",
		"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
		"usage":{
			"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,
			"completion_tokens_details":{"reasoning_tokens":5,"accepted_prediction_tokens":10,"rejected_prediction_tokens":3}
		}
	}`)
	result, err := ParseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.ReasoningTokens != 5 {
		t.Errorf("ReasoningTokens = %d, want 5", result.Usage.ReasoningTokens)
	}
	pm := result.ProviderMetadata["openai"]
	if pm == nil {
		t.Fatal("ProviderMetadata[openai] missing")
	}
	if pm["acceptedPredictionTokens"] != 10 {
		t.Errorf("acceptedPredictionTokens = %v, want 10", pm["acceptedPredictionTokens"])
	}
	if pm["rejectedPredictionTokens"] != 3 {
		t.Errorf("rejectedPredictionTokens = %v, want 3", pm["rejectedPredictionTokens"])
	}
}

// Item 11: url_citation in ParseResponse.
func TestBF13_URLCitation_ParseResponse(t *testing.T) {
	body := []byte(`{
		"id":"x","model":"m",
		"choices":[{
			"message":{
				"content":"text",
				"annotations":[{
					"type":"url_citation",
					"url_citation":{"url":"https://example.com","title":"Ex","start_index":0,"end_index":4}
				}]
			},
			"finish_reason":"stop"
		}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	result, err := ParseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Sources) != 1 {
		t.Fatalf("Sources = %d, want 1", len(result.Sources))
	}
	if result.Sources[0].URL != "https://example.com" {
		t.Errorf("URL = %q", result.Sources[0].URL)
	}
}

// Item 12: logprobs in ParseResponse.
func TestBF13_Logprobs_ParseResponse(t *testing.T) {
	body := []byte(`{
		"id":"x","model":"m",
		"choices":[{
			"message":{"content":"ok"},
			"logprobs":{"content":[{"token":"ok","logprob":-0.5}]},
			"finish_reason":"stop"
		}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	result, err := ParseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	pm := result.ProviderMetadata["openai"]
	if pm == nil {
		t.Fatal("ProviderMetadata[openai] missing")
	}
	if pm["logprobs"] == nil {
		t.Error("logprobs not extracted")
	}
}

// Item 17: tool call ID fallback in ParseResponse.
func TestBF13_ToolCallIDFallback_ParseResponse(t *testing.T) {
	body := []byte(`{
		"id":"x","model":"m",
		"choices":[{
			"message":{
				"content":"",
				"tool_calls":[{"id":"","type":"function","function":{"name":"read","arguments":"{}"}}]
			},
			"finish_reason":"tool_calls"
		}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	result, err := ParseResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
	id := result.ToolCalls[0].ID
	if !strings.HasPrefix(id, "call_") {
		t.Errorf("expected generated ID starting with call_, got %q", id)
	}
	if len(id) < 10 {
		t.Errorf("generated ID too short: %q", id)
	}
}

// Item 5: parallelToolCalls in BuildRequest.
func TestBF13_ParallelToolCalls_BuildRequest(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"parallelToolCalls": false},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["parallel_tool_calls"] != false {
		t.Errorf("parallel_tool_calls = %v, want false", body["parallel_tool_calls"])
	}
}

// Item 7: json_object mode in BuildRequest.
func TestBF13_JsonObjectMode_BuildRequest(t *testing.T) {
	params := provider.GenerateParams{
		Messages:       []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{Name: "test"},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatal("response_format not set")
	}
	if rf["type"] != "json_object" {
		t.Errorf("type = %v, want json_object", rf["type"])
	}
}

// Item 8: structuredOutputs=false in BuildRequest.
func TestBF13_StructuredOutputsFalse_BuildRequest(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test",
			Schema: json.RawMessage(`{"type":"object"}`),
		},
		ProviderOptions: map[string]any{"structuredOutputs": false},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatal("response_format not set")
	}
	if rf["type"] != "json_object" {
		t.Errorf("type = %v, want json_object", rf["type"])
	}
}

// Item 9: strictJsonSchema in BuildRequest.
func TestBF13_StrictJsonSchema_BuildRequest(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test",
			Schema: json.RawMessage(`{"type":"object"}`),
		},
		ProviderOptions: map[string]any{"strictJsonSchema": true},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	rf := body["response_format"].(map[string]any)
	js := rf["json_schema"].(map[string]any)
	if js["strict"] != true {
		t.Errorf("strict = %v, want true", js["strict"])
	}
}

// Item 8: structuredOutputs=false for tools -- strict not set.
func TestBF13_StructuredOutputsFalse_Tools(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools: []provider.ToolDefinition{
			{Name: "read", Description: "Read", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		ProviderOptions: map[string]any{"structuredOutputs": false},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	tools := body["tools"].([]map[string]any)
	fn := tools[0]["function"].(map[string]any)
	if _, hasStrict := fn["strict"]; hasStrict {
		t.Error("strict should not be set when structuredOutputs=false")
	}
}

// structuredOutputs with non-bool value should be ignored (keep default true).
func TestBuildRequest_StructuredOutputsNonBool(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools: []provider.ToolDefinition{
			{Name: "read", Description: "Read", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		ProviderOptions: map[string]any{"structuredOutputs": "yes"}, // non-bool
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	tools := body["tools"].([]map[string]any)
	fn := tools[0]["function"].(map[string]any)
	// Default structuredOutputs=true, so strict should be present (with default false).
	if fn["strict"] != false {
		t.Errorf("strict = %v, want false (default when structuredOutputs non-bool)", fn["strict"])
	}
}

// Item 9: strictJsonSchema in tool function parameters.
func TestBF13_StrictJsonSchema_Tools(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools: []provider.ToolDefinition{
			{Name: "read", Description: "Read", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		ProviderOptions: map[string]any{"strictJsonSchema": true},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	tools := body["tools"].([]map[string]any)
	fn := tools[0]["function"].(map[string]any)
	if fn["strict"] != true {
		t.Errorf("tool strict = %v, want true", fn["strict"])
	}
}

// Item 1: TotalTokens in streaming ParseStream.
func TestBF13_TotalTokens_ParseStream(t *testing.T) {
	sseData := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"index\":0}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n" +
		"data: [DONE]\n\n"

	reader := strings.NewReader(sseData)
	scanner := sse.NewScanner(reader)
	out := make(chan provider.StreamChunk, 32)
	go ParseStream(t.Context(), scanner, out)

	var usage provider.Usage
	for chunk := range out {
		if chunk.Type == provider.ChunkFinish {
			usage = chunk.Usage
		}
	}
	if usage.TotalTokens != 15 {
		t.Errorf("TotalTokens = %d, want 15", usage.TotalTokens)
	}
}

// =============================================================================
// applyProviderOptions coverage
// =============================================================================

func TestApplyProviderOptions_User(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"user": "user-123"},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["user"] != "user-123" {
		t.Errorf("user = %v, want user-123", body["user"])
	}
}

func TestApplyProviderOptions_LogprobsBool(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"logprobs": true},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["logprobs"] != true {
		t.Errorf("logprobs = %v, want true", body["logprobs"])
	}
	if body["top_logprobs"] != 0 {
		t.Errorf("top_logprobs = %v, want 0", body["top_logprobs"])
	}
}

func TestApplyProviderOptions_LogprobsInt(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"logprobs": 5},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["logprobs"] != true {
		t.Errorf("logprobs = %v, want true", body["logprobs"])
	}
	if body["top_logprobs"] != 5 {
		t.Errorf("top_logprobs = %v, want 5", body["top_logprobs"])
	}
}

func TestApplyProviderOptions_LogprobsFloat64(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"logprobs": float64(3)},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["logprobs"] != true {
		t.Errorf("logprobs = %v, want true", body["logprobs"])
	}
	if body["top_logprobs"] != int(3) {
		t.Errorf("top_logprobs = %v, want 3", body["top_logprobs"])
	}
}

func TestApplyProviderOptions_LogprobsBoolFalse(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"logprobs": false},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	// logprobs=false should not set anything
	if _, ok := body["logprobs"]; ok {
		t.Error("logprobs should not be set when false")
	}
}

func TestApplyProviderOptions_Metadata(t *testing.T) {
	meta := map[string]any{"key": "value"}
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"metadata": meta},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["metadata"] == nil {
		t.Error("metadata should be set")
	}
}

func TestApplyProviderOptions_SafetyIdentifier(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"safetyIdentifier": "safe-1"},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["safety_identifier"] != "safe-1" {
		t.Errorf("safety_identifier = %v, want safe-1", body["safety_identifier"])
	}
}

func TestApplyProviderOptions_Store(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"store": true},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["store"] != true {
		t.Errorf("store = %v, want true", body["store"])
	}
}

func TestApplyProviderOptions_ServiceTier(t *testing.T) {
	params := provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"serviceTier": "default"},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["service_tier"] != "default" {
		t.Errorf("service_tier = %v, want default", body["service_tier"])
	}
}

// =============================================================================
// ParseStream coverage -- streaming-specific branches
// =============================================================================

// Streaming: completion_tokens_details with reasoning + prediction tokens.
func TestParseStream_CompletionTokensDetails(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"hi"},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"completion_tokens_details":{"reasoning_tokens":8,"accepted_prediction_tokens":5,"rejected_prediction_tokens":2}}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var finishChunk provider.StreamChunk
	for chunk := range out {
		if chunk.Type == provider.ChunkFinish {
			finishChunk = chunk
		}
	}

	if finishChunk.Usage.ReasoningTokens != 8 {
		t.Errorf("ReasoningTokens = %d, want 8", finishChunk.Usage.ReasoningTokens)
	}

	pm, ok := finishChunk.Metadata["providerMetadata"].(map[string]map[string]any)
	if !ok {
		t.Fatal("providerMetadata missing from finish chunk")
	}
	oai := pm["openai"]
	if oai["acceptedPredictionTokens"] != 5 {
		t.Errorf("acceptedPredictionTokens = %v, want 5", oai["acceptedPredictionTokens"])
	}
	if oai["rejectedPredictionTokens"] != 2 {
		t.Errorf("rejectedPredictionTokens = %v, want 2", oai["rejectedPredictionTokens"])
	}
}

// Streaming: logprobs extraction.
func TestParseStream_Logprobs(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"hi"},"index":0,"logprobs":{"content":[{"token":"hi","logprob":-0.1}]}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var finishChunk provider.StreamChunk
	for chunk := range out {
		if chunk.Type == provider.ChunkFinish {
			finishChunk = chunk
		}
	}

	pm, ok := finishChunk.Metadata["providerMetadata"].(map[string]map[string]any)
	if !ok {
		t.Fatal("providerMetadata missing from finish chunk")
	}
	if pm["openai"]["logprobs"] == nil {
		t.Error("logprobs not extracted from stream")
	}
}

// Streaming: url_citation annotations.
func TestParseStream_URLCitation(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"text","annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"Example","start_index":0,"end_index":4}}]},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	var foundCitation bool
	for _, c := range chunks {
		if c.Metadata != nil {
			if src, ok := c.Metadata["source"].(provider.Source); ok {
				foundCitation = true
				if src.URL != "https://example.com" {
					t.Errorf("source URL = %q, want https://example.com", src.URL)
				}
				if src.Title != "Example" {
					t.Errorf("source Title = %q, want Example", src.Title)
				}
			}
		}
	}
	if !foundCitation {
		t.Error("url_citation annotation not found in stream chunks")
	}
}

// Streaming: tool call with no ID but with function name -- should generate fallback ID.
func TestParseStream_ToolCallFallbackID(t *testing.T) {
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read","arguments":""}}]},"index":0}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	var foundStart bool
	for _, c := range chunks {
		if c.Type == provider.ChunkToolCallStreamStart {
			foundStart = true
			if !strings.HasPrefix(c.ToolCallID, "call_") {
				t.Errorf("expected generated ID starting with call_, got %q", c.ToolCallID)
			}
			if c.ToolName != "read" {
				t.Errorf("ToolName = %q, want read", c.ToolName)
			}
		}
	}
	if !foundStart {
		t.Error("expected tool_call_streaming_start with generated fallback ID")
	}
}

// Streaming: providerMetadata in finish chunk when metadata is present.
func TestParseStream_ProviderMetadataInFinish(t *testing.T) {
	// Use logprobs to trigger providerMeta population, then check finish chunk has metadata.
	input := `data: {"choices":[{"delta":{"content":"hi"},"index":0,"logprobs":{"content":[]}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var finishChunk provider.StreamChunk
	for chunk := range out {
		if chunk.Type == provider.ChunkFinish {
			finishChunk = chunk
		}
	}

	if finishChunk.Metadata == nil {
		t.Fatal("expected metadata in finish chunk when providerMeta is populated")
	}
	pm, ok := finishChunk.Metadata["providerMetadata"].(map[string]map[string]any)
	if !ok {
		t.Fatal("providerMetadata has wrong type")
	}
	if pm["openai"] == nil {
		t.Error("openai provider metadata missing")
	}
}

// Non-streaming: top-level citations[] (xAI, Perplexity) → Sources populated.
func TestParseResponse_TopLevelCitations(t *testing.T) {
	body := `{
		"id": "resp-1",
		"model": "grok-2",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "The answer is 42."},
			"finish_reason": "stop"
		}],
		"citations": ["https://example.com/source1", "https://example.com/source2"],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`

	result, err := ParseResponse([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Sources) != 2 {
		t.Fatalf("len(Sources) = %d, want 2", len(result.Sources))
	}
	for i, s := range result.Sources {
		if s.Type != "url" {
			t.Errorf("Sources[%d].Type = %q, want url", i, s.Type)
		}
		if s.ID != fmt.Sprintf("citation_%d", i) {
			t.Errorf("Sources[%d].ID = %q, want citation_%d", i, s.ID, i)
		}
	}
	if result.Sources[0].URL != "https://example.com/source1" {
		t.Errorf("Sources[0].URL = %q", result.Sources[0].URL)
	}
	if result.Sources[1].URL != "https://example.com/source2" {
		t.Errorf("Sources[1].URL = %q", result.Sources[1].URL)
	}
}

// Non-streaming: both annotations AND top-level citations → both populated.
func TestParseResponse_AnnotationsAndCitations(t *testing.T) {
	body := `{
		"id": "resp-2",
		"model": "grok-2",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Answer",
				"annotations": [{
					"type": "url_citation",
					"url_citation": {"url": "https://ann.com", "title": "Ann", "start_index": 0, "end_index": 6}
				}]
			},
			"finish_reason": "stop"
		}],
		"citations": ["https://cit.com"],
		"usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
	}`

	result, err := ParseResponse([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	// 1 annotation + 1 citation = 2 sources.
	if len(result.Sources) != 2 {
		t.Fatalf("len(Sources) = %d, want 2", len(result.Sources))
	}
	if result.Sources[0].URL != "https://ann.com" {
		t.Errorf("Sources[0] (annotation) URL = %q", result.Sources[0].URL)
	}
	if result.Sources[1].URL != "https://cit.com" {
		t.Errorf("Sources[1] (citation) URL = %q", result.Sources[1].URL)
	}
}

// Streaming: top-level citations[] on final chunk → sources in finish metadata.
func TestParseStream_TopLevelCitations(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"Answer"},"index":0}]}
data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"citations":["https://source1.com","https://source2.com"],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}
data: [DONE]
`
	scanner := sse.NewScanner(strings.NewReader(input))
	out := make(chan provider.StreamChunk, 10)
	go ParseStream(t.Context(), scanner, out)

	var chunks []provider.StreamChunk
	for chunk := range out {
		chunks = append(chunks, chunk)
	}

	// Find finish chunk with sources metadata.
	var finishChunk provider.StreamChunk
	for _, c := range chunks {
		if c.Type == provider.ChunkFinish {
			finishChunk = c
		}
	}

	if finishChunk.Metadata == nil {
		t.Fatal("finish chunk has no metadata")
	}
	sources, ok := finishChunk.Metadata["sources"].([]provider.Source)
	if !ok {
		t.Fatal("finish chunk metadata missing 'sources'")
	}
	if len(sources) != 2 {
		t.Fatalf("len(sources) = %d, want 2", len(sources))
	}
	if sources[0].URL != "https://source1.com" || sources[0].Type != "url" {
		t.Errorf("sources[0] = %+v", sources[0])
	}
	if sources[1].URL != "https://source2.com" {
		t.Errorf("sources[1] = %+v", sources[1])
	}
}

// Non-streaming: empty citations[] → no Sources.
func TestParseResponse_EmptyCitations(t *testing.T) {
	body := `{
		"id": "resp-3",
		"model": "grok-2",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "hello"},
			"finish_reason": "stop"
		}],
		"citations": [],
		"usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
	}`

	result, err := ParseResponse([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Sources) != 0 {
		t.Errorf("len(Sources) = %d, want 0", len(result.Sources))
	}
}

func TestExtractTextContent(t *testing.T) {
	tests := []struct {
		name string
		raw  json.RawMessage
		want string
	}{
		{
			name: "string content",
			raw:  json.RawMessage(`"hello"`),
			want: "hello",
		},
		{
			name: "array with single text part",
			raw:  json.RawMessage(`[{"type":"text","text":"hello"}]`),
			want: "hello",
		},
		{
			name: "array with multiple text parts",
			raw:  json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`),
			want: "ab",
		},
		{
			name: "array with thinking part skipped",
			raw:  json.RawMessage(`[{"type":"thinking","text":"hmm"},{"type":"text","text":"hello"}]`),
			want: "hello",
		},
		{
			name: "null",
			raw:  nil,
			want: "",
		},
		{
			name: "empty",
			raw:  json.RawMessage(``),
			want: "",
		},
		{
			name: "invalid JSON",
			raw:  json.RawMessage(`{{{`),
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractTextContent(tt.raw)
			if got != tt.want {
				t.Errorf("extractTextContent() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBuildRequest_ProviderDefinedTool(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{
				Name:                "browser_search",
				ProviderDefinedType: "browser_search",
				ProviderDefinedOptions: map[string]any{
					"max_results": 10,
				},
			},
		},
	}

	body := BuildRequest(params, "gpt-oss-20b", false, RequestConfig{})

	tools, ok := body["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", body["tools"])
	}
	tool := tools[0]
	if tool["type"] != "browser_search" {
		t.Errorf("tool type = %v, want browser_search", tool["type"])
	}
	if tool["max_results"] != 10 {
		t.Errorf("max_results = %v", tool["max_results"])
	}
	// Should NOT have "function" wrapper.
	if _, ok := tool["function"]; ok {
		t.Error("provider-defined tool should not have function wrapper")
	}
}

func TestParseStream_ContextCancel_AllBranches(t *testing.T) {
	// Exercise every TrySend early-return path in ParseStream with a cancelled
	// context and unbuffered channel.  The stream contains:
	//   - stream error (context_overflow)
	//   - text content
	//   - reasoning content
	//   - annotation (url_citation)
	//   - tool call start + delta + complete JSON args
	//   - finish_reason with tool_calls (flush remaining)
	//   - step finish
	//   - scanner error path (via errorReader)
	//   - final ChunkFinish
	// We run separate sub-tests so each TrySend branch is the FIRST one hit.

	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "stream_error_overflow",
			input: "data: {\"type\":\"error\",\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"too long\"}}\n",
		},
		{
			name:  "stream_error_other",
			input: "data: {\"type\":\"error\",\"error\":{\"code\":\"insufficient_quota\",\"message\":\"quota exceeded\"}}\n",
		},
		{
			name:  "text_content",
			input: "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"index\":0}]}\ndata: [DONE]\n",
		},
		{
			name:  "reasoning_content",
			input: "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"},\"index\":0}]}\ndata: [DONE]\n",
		},
		{
			name:  "annotation",
			input: "data: {\"choices\":[{\"delta\":{\"annotations\":[{\"type\":\"url_citation\",\"url_citation\":{\"url\":\"http://x\",\"title\":\"T\",\"start_index\":0,\"end_index\":1}}]},\"index\":0}]}\ndata: [DONE]\n",
		},
		{
			name:  "tool_call_start",
			input: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"f\"}}]},\"index\":0}]}\ndata: [DONE]\n",
		},
		{
			// Tool call delta: first TrySend is ChunkToolCallDelta (line 474).
			// Achieved by having tool_calls with no ID/name (skips start TrySend)
			// but with arguments.
			name:  "tool_call_delta",
			input: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\"}}]},\"index\":0}]}\ndata: [DONE]\n",
		},
		{
			// Tool call complete JSON: first TrySend is ChunkToolCall (line 485).
			// Use valid complete JSON args with no ID (skips start TrySend).
			// ChunkToolCallDelta at 474 would fire first, so use buffered chan (see special handling below).
			name:  "tool_call_complete_json",
			input: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]},\"index\":0}]}\ndata: [DONE]\n",
		},
		{
			// Flush remaining at finish_reason=tool_calls (line 503).
			// Use a tool start (with ID) + partial args + finish. Tool start TrySend fires first.
			// Use buffered chan to pass tool start, then delta, then hit flush.
			name: "finish_reason_tool_calls_flush",
			input: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"f\"}}]},\"index\":0}]}\n" +
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"partial\"}}]},\"index\":0}]}\n" +
				"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\",\"index\":0}]}\ndata: [DONE]\n",
		},
		{
			name:  "step_finish",
			input: "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\ndata: [DONE]\n",
		},
		{
			// Final ChunkFinish TrySend at end of stream. Use a stream with no
			// choices (usage-only chunk) so no TrySend fires in the loop, then
			// the final ChunkFinish TrySend is the first one hit.
			name:  "final_finish_chunk",
			input: "data: {\"id\":\"x\",\"model\":\"m\",\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\ndata: [DONE]\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel() // cancel before calling ParseStream

			out := make(chan provider.StreamChunk) // unbuffered
			scanner := sse.NewScanner(strings.NewReader(tc.input))

			done := make(chan struct{})
			go func() {
				ParseStream(ctx, scanner, out)
				close(done)
			}()
			<-done

			// Drain any buffered chunks.
			for range out {
			}
		})
	}

	// Sub-test for scanner error path with cancelled context.
	t.Run("scanner_error", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()

		out := make(chan provider.StreamChunk) // unbuffered
		scanner := sse.NewScanner(&errorReader{
			data: "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"index\":0}]}\n",
			err:  fmt.Errorf("read error"),
		})

		done := make(chan struct{})
		go func() {
			ParseStream(ctx, scanner, out)
			close(done)
		}()
		<-done
		for range out {
		}
	})

	// Sub-tests for nested TrySend paths that require some sends to succeed
	// before the target TrySend fails. These use io.Pipe to control data flow
	// and cancel context between sends.

	t.Run("tool_call_complete_json_cancel", func(t *testing.T) {
		// Goal: cover line 490 (ChunkToolCall TrySend return).
		// Strategy: send tool call args with no ID (skip start TrySend),
		// args = "{}" (valid JSON). Delta TrySend at 474 is first, then
		// complete JSON TrySend at 485.
		// Use unbuffered chan: read the delta chunk (unblocking that TrySend),
		// then cancel so the next TrySend (complete JSON) fails.
		ctx, cancel := context.WithCancel(t.Context())

		out := make(chan provider.StreamChunk) // unbuffered
		input := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]},\"index\":0}]}\ndata: [DONE]\n"
		scanner := sse.NewScanner(strings.NewReader(input))

		done := make(chan struct{})
		go func() {
			ParseStream(ctx, scanner, out)
			close(done)
		}()

		<-out // receive the delta chunk (unblocks TrySend at 474)
		cancel()
		<-done
		for range out {
		}
	})

	t.Run("finish_flush_remaining_cancel", func(t *testing.T) {
		// Goal: cover line 508 (flush remaining tool args at finish_reason=tool_calls).
		// Strategy: send tool start + partial args + finish_reason=tool_calls.
		// Receive tool start and delta chunks, then cancel before flush.
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"f\"}}]},\"index\":0}]}\n" +
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"partial\"}}]},\"index\":0}]}\n" +
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\",\"index\":0}]}\ndata: [DONE]\n"
		scanner := sse.NewScanner(strings.NewReader(input))

		done := make(chan struct{})
		go func() {
			ParseStream(ctx, scanner, out)
			close(done)
		}()

		<-out // tool start
		<-out // delta
		cancel()
		<-done
		for range out {
		}
	})

	t.Run("scanner_error_cancel", func(t *testing.T) {
		// Goal: cover line 526 (scanner.Err() TrySend return).
		// Strategy: use errorReader that produces valid data then errors.
		// Receive the text chunk, then cancel before error chunk is sent.
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		scanner := sse.NewScanner(&errorReader{
			data: "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"index\":0}]}\n",
			err:  fmt.Errorf("read error"),
		})

		done := make(chan struct{})
		go func() {
			ParseStream(ctx, scanner, out)
			close(done)
		}()

		<-out // consume the text chunk
		cancel()
		<-done
		for range out {
		}
	})
}

func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		input string
		want  provider.FinishReason
	}{
		{"stop", provider.FinishStop},
		{"tool_calls", provider.FinishToolCalls},
		{"length", provider.FinishLength},
		{"content_filter", provider.FinishContentFilter},
		{"", provider.FinishOther},
		{"unknown_reason", provider.FinishOther},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := mapFinishReason(tt.input)
			if got != tt.want {
				t.Errorf("mapFinishReason(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestBuildRequest_NewOptions(t *testing.T) {
	topK := 10
	seed := 42
	freqPenalty := 0.5
	presPenalty := 0.3

	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		TopK:             &topK,
		Seed:             &seed,
		FrequencyPenalty: &freqPenalty,
		PresencePenalty:  &presPenalty,
	}

	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})

	if body["top_k"] != 10 {
		t.Errorf("top_k = %v, want 10", body["top_k"])
	}
	if body["seed"] != 42 {
		t.Errorf("seed = %v, want 42", body["seed"])
	}
	if body["frequency_penalty"] != 0.5 {
		t.Errorf("frequency_penalty = %v, want 0.5", body["frequency_penalty"])
	}
	if body["presence_penalty"] != 0.3 {
		t.Errorf("presence_penalty = %v, want 0.3", body["presence_penalty"])
	}
}

func TestBuildRequest_ProtectedKeysIgnoresModel(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"model":    "evil-model",
			"stream":   false,
			"messages": "should-not-override",
		},
	}
	body := BuildRequest(params, "gpt-4o", true, RequestConfig{})
	if body["model"] != "gpt-4o" {
		t.Errorf("model = %v, want gpt-4o (protectedKeys should prevent override)", body["model"])
	}
	if body["stream"] != true {
		t.Errorf("stream = %v, want true (protectedKeys should prevent override)", body["stream"])
	}
}

func TestBuildRequest_ProtectedKeysIgnoresTemperature(t *testing.T) {
	temp := 0.5
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Temperature: &temp,
		ProviderOptions: map[string]any{
			"temperature": 999.0,
			"max_tokens":  999,
			"tools":       "should-not-override",
		},
	}
	body := BuildRequest(params, "gpt-4o", false, RequestConfig{})
	if body["temperature"] != 0.5 {
		t.Errorf("temperature = %v, want 0.5 (protectedKeys should prevent override)", body["temperature"])
	}
}

// TestConvertMessages_MultipleToolResults verifies that a single RoleTool message
// with two PartToolResult parts produces two separate wire messages, one per result.
func TestConvertMessages_MultipleToolResults(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleTool,
			Content: []provider.Part{
				{Type: provider.PartToolResult, ToolCallID: "call_1", ToolOutput: "result one"},
				{Type: provider.PartToolResult, ToolCallID: "call_2", ToolOutput: "result two"},
			},
		},
	}
	result := ConvertMessages(msgs, "")

	if len(result) != 2 {
		t.Fatalf("got %d wire messages, want 2 (one per ToolResult part)", len(result))
	}

	for i, wm := range result {
		if wm["role"] != "tool" {
			t.Errorf("result[%d] role = %v, want tool", i, wm["role"])
		}
	}

	if result[0]["tool_call_id"] != "call_1" {
		t.Errorf("result[0] tool_call_id = %v, want call_1", result[0]["tool_call_id"])
	}
	if result[0]["content"] != "result one" {
		t.Errorf("result[0] content = %v, want \"result one\"", result[0]["content"])
	}
	if result[1]["tool_call_id"] != "call_2" {
		t.Errorf("result[1] tool_call_id = %v, want call_2", result[1]["tool_call_id"])
	}
	if result[1]["content"] != "result two" {
		t.Errorf("result[1] content = %v, want \"result two\"", result[1]["content"])
	}
}

// TestConvertMessages_EmptyContent verifies behavior when a user message has an
// empty content slice. The message is still appended to the output (no parts are
// skipped at the message level), and the resulting wire message has no "content"
// or "tool_calls" keys set.
func TestConvertMessages_EmptyContent(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{}},
	}
	result := ConvertMessages(msgs, "")

	// A message with no parts still produces one wire message.
	if len(result) != 1 {
		t.Fatalf("got %d wire messages, want 1", len(result))
	}
	if result[0]["role"] != "user" {
		t.Errorf("role = %v, want user", result[0]["role"])
	}
	// No content parts were collected, so "content" key should be absent.
	if _, ok := result[0]["content"]; ok {
		t.Errorf("expected no 'content' key for empty-content message, got %v", result[0]["content"])
	}
	// No tool calls either.
	if _, ok := result[0]["tool_calls"]; ok {
		t.Errorf("expected no 'tool_calls' key for empty-content message")
	}
}

// TestConvertMessages_ImageOnlyUser verifies that a user message containing only
// an image part (no text) produces a content array with exactly one image_url entry.
func TestConvertMessages_ImageOnlyUser(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleUser,
			Content: []provider.Part{
				{Type: provider.PartImage, URL: "https://example.com/image.png"},
			},
		},
	}
	result := ConvertMessages(msgs, "")

	if len(result) != 1 {
		t.Fatalf("got %d wire messages, want 1", len(result))
	}
	content, ok := result[0]["content"].([]map[string]any)
	if !ok {
		t.Fatalf("expected content array, got %T", result[0]["content"])
	}
	if len(content) != 1 {
		t.Fatalf("got %d content parts, want 1", len(content))
	}
	if content[0]["type"] != "image_url" {
		t.Errorf("content[0] type = %v, want image_url", content[0]["type"])
	}
	imgURL, ok := content[0]["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("expected image_url map, got %T", content[0]["image_url"])
	}
	if imgURL["url"] != "https://example.com/image.png" {
		t.Errorf("image url = %v, want https://example.com/image.png", imgURL["url"])
	}
}

// TestParseResponse_NullContent verifies that a response with "content": null
// is handled gracefully - extractTextContent returns empty string so result.Text
// should be empty rather than an error.
func TestParseResponse_NullContent(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-null",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": null},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
	}`)

	result, err := ParseResponse(body)
	if err != nil {
		t.Fatalf("ParseResponse error: %v", err)
	}
	if result.Text != "" {
		t.Errorf("Text = %q, want empty string for null content", result.Text)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want stop", result.FinishReason)
	}
}

// TestExtractTextContent_NullJSON directly tests that extractTextContent returns
// an empty string when given the JSON literal "null".
func TestExtractTextContent_NullJSON(t *testing.T) {
	got := extractTextContent(json.RawMessage("null"))
	if got != "" {
		t.Errorf("extractTextContent(null) = %q, want empty string", got)
	}
}

// TestConvertMessages_ToolCallsNoText verifies that an assistant message with
// tool calls but no text content does not produce "content":"" in the output.
// OpenAI requires content to be null or omitted, not an empty string.
func TestConvertMessages_ToolCallsNoText(t *testing.T) {
	msgs := []provider.Message{
		{
			Role: provider.RoleAssistant,
			Content: []provider.Part{
				{
					Type:       provider.PartToolCall,
					ToolCallID: "call_abc",
					ToolName:   "get_weather",
					ToolInput:  json.RawMessage(`{"location":"NYC"}`),
				},
			},
		},
	}

	result := ConvertMessages(msgs, "")
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	m := result[0]

	// Verify tool_calls are present.
	toolCalls, ok := m["tool_calls"]
	if !ok {
		t.Fatal("expected tool_calls key to be present")
	}
	tc := toolCalls.([]map[string]any)
	if len(tc) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tc))
	}

	// Verify content is NOT empty string.
	// It should be absent from the map (nil map lookup) or nil.
	content, hasContent := m["content"]
	if hasContent && content == "" {
		t.Error(`content = "", want null or absent; OpenAI rejects empty string with tool_calls`)
	}

	// Also verify via JSON serialization.
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	jsonStr := string(b)
	if strings.Contains(jsonStr, `"content":""`) {
		t.Errorf("marshaled JSON contains %q, want null or no content field; got: %s", `"content":""`, jsonStr)
	}
}
