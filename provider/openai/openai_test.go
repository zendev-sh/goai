package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// errorReader is an io.ReadCloser that returns an error on Read.
type errorReader struct{}

func (e *errorReader) Read(_ []byte) (int, error) { return 0, fmt.Errorf("read error") }
func (e *errorReader) Close() error                { return nil }

// chatCompletionsOpts forces the Chat Completions API path (not Responses API).
var chatCompletionsOpts = map[string]any{"useResponsesAPI": false}

// --- Chat Completions tests ---

func TestChat_ChatCompletions_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("unexpected auth: %s", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected content type: %s", r.Header.Get("Content-Type"))
		}

		// Verify request body.
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["model"] != "gpt-4o" {
			t.Errorf("model = %v, want gpt-4o", req["model"])
		}
		if req["stream"] != true {
			t.Errorf("stream = %v, want true", req["stream"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range result.Stream {
		chunks = append(chunks, chunk)
	}

	// text + text + step_finish + finish
	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 chunks, got %d", len(chunks))
	}
	if chunks[0].Text != "Hello" {
		t.Errorf("chunks[0].Text = %q, want %q", chunks[0].Text, "Hello")
	}
	if chunks[1].Text != " world" {
		t.Errorf("chunks[1].Text = %q, want %q", chunks[1].Text, " world")
	}
}

func TestChat_ChatCompletions_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["stream"] != false {
			t.Errorf("stream = %v, want false", req["stream"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "chatcmpl-123",
			"model": "gpt-4o",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello world"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want %q", result.Text, "Hello world")
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v", result.Usage)
	}
	if result.Response.ID != "chatcmpl-123" {
		t.Errorf("Response.ID = %q, want %q", result.Response.ID, "chatcmpl-123")
	}
}

func TestChat_ChatCompletions_ToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Tool call start
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","type":"function","function":{"name":"read","arguments":""}}]},"index":0}]}`+"\n\n")
		// Tool call args fragment 1
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]},"index":0}]}`+"\n\n")
		// Tool call args fragment 2
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]},"index":0}]}`+"\n\n")
		// Finish
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":10}}`+"\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read file"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCalls = append(toolCalls, chunk)
		}
	}

	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].ToolName != "read" {
		t.Errorf("ToolName = %q, want %q", toolCalls[0].ToolName, "read")
	}
	if toolCalls[0].ToolInput != `{"path":"a.txt"}` {
		t.Errorf("ToolInput = %q, want %q", toolCalls[0].ToolInput, `{"path":"a.txt"}`)
	}
}

func TestChat_ChatCompletions_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChat_ChatCompletions_CachedTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "chatcmpl-123",
			"model": "gpt-4o",
			"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}],
			"usage": {
				"prompt_tokens": 100,
				"completion_tokens": 10,
				"prompt_tokens_details": {"cached_tokens": 30}
			}
		}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}

	// InputTokens normalized: 100 - 30 = 70
	if result.Usage.InputTokens != 70 {
		t.Errorf("InputTokens = %d, want 70", result.Usage.InputTokens)
	}
	if result.Usage.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", result.Usage.CacheReadTokens)
	}
}

func TestChat_ChatCompletions_ReasoningContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{"reasoning_content":"thinking..."},"index":0}]}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"answer"},"index":0}]}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":10}}`+"\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotReasoning, gotText bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkReasoning && chunk.Text == "thinking..." {
			gotReasoning = true
		}
		if chunk.Type == provider.ChunkText && chunk.Text == "answer" {
			gotText = true
		}
	}
	if !gotReasoning {
		t.Error("missing reasoning chunk")
	}
	if !gotText {
		t.Error("missing text chunk")
	}
}

// --- Responses API tests ---

func TestChat_Responses_Stream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("unexpected path: %s, want /responses", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if req["model"] != "o3" {
			t.Errorf("model = %v, want o3", req["model"])
		}
		if _, ok := req["instructions"]; !ok {
			t.Error("missing instructions field")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"Hello"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":" world"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":10,"output_tokens":5}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		System: "be helpful",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	var finishReason provider.FinishReason
	var usage provider.Usage
	for chunk := range result.Stream {
		switch chunk.Type {
		case provider.ChunkText:
			texts = append(texts, chunk.Text)
		case provider.ChunkStepFinish:
			finishReason = chunk.FinishReason
		case provider.ChunkFinish:
			usage = chunk.Usage
		}
	}

	if len(texts) != 2 || texts[0] != "Hello" || texts[1] != " world" {
		t.Errorf("texts = %v", texts)
	}
	if finishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", finishReason, provider.FinishStop)
	}
	if usage.InputTokens != 10 || usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v", usage)
	}
}

func TestChat_Responses_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "resp-123",
			"model": "o3",
			"status": "completed",
			"output": [{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Hello world"}]
			}],
			"usage": {
				"input_tokens": 10,
				"output_tokens": 5,
				"output_tokens_details": {"reasoning_tokens": 3}
			}
		}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 {
		t.Errorf("Usage = %+v", result.Usage)
	}
	if result.Usage.ReasoningTokens != 3 {
		t.Errorf("ReasoningTokens = %d, want 3", result.Usage.ReasoningTokens)
	}
	if result.Response.ID != "resp-123" {
		t.Errorf("Response.ID = %q, want %q", result.Response.ID, "resp-123")
	}
}

func TestChat_Responses_ToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Tool call start
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"function_call","call_id":"tc1","name":"read"}}`+"\n\n")
		// Args delta
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"delta":"{\"path\":"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"delta":"\"a.txt\"}"}`+"\n\n")
		// Args done
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.done\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"arguments":"{\"path\":\"a.txt\"}"}`+"\n\n")
		// Item done
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0}`+"\n\n")
		// Completed
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":5,"output_tokens":10}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read file"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []provider.StreamChunk
	var finishReason provider.FinishReason
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCalls = append(toolCalls, chunk)
		}
		if chunk.Type == provider.ChunkStepFinish {
			finishReason = chunk.FinishReason
		}
	}

	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].ToolName != "read" {
		t.Errorf("ToolName = %q, want %q", toolCalls[0].ToolName, "read")
	}
	if toolCalls[0].ToolInput != `{"path":"a.txt"}` {
		t.Errorf("ToolInput = %q, want %q", toolCalls[0].ToolInput, `{"path":"a.txt"}`)
	}
	if finishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want %q", finishReason, provider.FinishToolCalls)
	}
}

func TestChat_Responses_Generate_ToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id": "resp-456",
			"model": "o3",
			"status": "completed",
			"output": [
				{"type": "function_call", "call_id": "tc1", "name": "read", "arguments": "{\"path\":\"a.txt\"}"}
			],
			"usage": {"input_tokens": 5, "output_tokens": 10}
		}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read file"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "read" {
		t.Errorf("ToolCall.Name = %q, want %q", result.ToolCalls[0].Name, "read")
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishToolCalls)
	}
}

func TestChat_Responses_Failed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.failed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"error":{"message":"context too long","code":"context_length_exceeded"}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotOverflow bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var coe *goai.ContextOverflowError
			if errors.As(chunk.Error, &coe) {
				gotOverflow = true
			}
		}
	}
	if !gotOverflow {
		t.Error("expected ContextOverflowError")
	}
}

func TestChat_Responses_Incomplete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"partial"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.incomplete\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":10,"output_tokens":100}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var finishReason provider.FinishReason
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkStepFinish {
			finishReason = chunk.FinishReason
		}
	}
	if finishReason != provider.FinishLength {
		t.Errorf("FinishReason = %q, want %q", finishReason, provider.FinishLength)
	}
}

func TestChat_Responses_Reasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"thinking..."}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"answer"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":5,"output_tokens":10,"output_tokens_details":{"reasoning_tokens":3}}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotReasoning, gotText bool
	var usage provider.Usage
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkReasoning && chunk.Text == "thinking..." {
			gotReasoning = true
		}
		if chunk.Type == provider.ChunkText && chunk.Text == "answer" {
			gotText = true
		}
		if chunk.Type == provider.ChunkFinish {
			usage = chunk.Usage
		}
	}
	if !gotReasoning {
		t.Error("missing reasoning chunk")
	}
	if !gotText {
		t.Error("missing text chunk")
	}
	if usage.ReasoningTokens != 3 {
		t.Errorf("ReasoningTokens = %d, want 3", usage.ReasoningTokens)
	}
}

// --- Model routing tests ---

func TestIsReasoningModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gpt-4o", false},
		{"gpt-4-turbo", false},
		{"gpt-3.5-turbo", false},
		{"gpt-5-chat-latest", false}, // gpt-5-chat is NOT reasoning per Vercel
		{"o1", true},
		{"o3", true},
		{"o3-mini", true},
		{"o4-mini", true},
		{"gpt-5", true},
		{"gpt-5.1-codex", true},
		{"gpt-5.2", true},
		{"codex-mini-latest", true},
		{"codex-mini", true},
		{"O3", true},        // case insensitive
		{"GPT-5.1", true},   // case insensitive
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got := isReasoningModel(tt.model)
			if got != tt.want {
				t.Errorf("isReasoningModel(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestShouldUseResponsesAPI(t *testing.T) {
	m := &chatModel{id: "gpt-4o"}

	// Default: all models use Responses API.
	params := provider.GenerateParams{}
	if !m.shouldUseResponsesAPI(params) {
		t.Error("expected gpt-4o to default to Responses API")
	}

	// Explicit opt-out.
	params.ProviderOptions = map[string]any{"useResponsesAPI": false}
	if m.shouldUseResponsesAPI(params) {
		t.Error("expected gpt-4o to use Chat Completions when opted out")
	}

	// Explicit opt-in.
	params.ProviderOptions = map[string]any{"useResponsesAPI": true}
	if !m.shouldUseResponsesAPI(params) {
		t.Error("expected gpt-4o to use Responses API when opted in")
	}
}

// --- Options tests ---

func TestChat_ModelID(t *testing.T) {
	model := Chat("gpt-4o", WithAPIKey("key"))
	if model.ModelID() != "gpt-4o" {
		t.Errorf("ModelID() = %q, want %q", model.ModelID(), "gpt-4o")
	}
}

func TestChat_Capabilities(t *testing.T) {
	// Standard model
	model := Chat("gpt-4o", WithAPIKey("key"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature {
		t.Error("gpt-4o should support temperature")
	}
	if caps.Reasoning {
		t.Error("gpt-4o should not be reasoning")
	}

	// Reasoning model
	model = Chat("o3", WithAPIKey("key"))
	caps = provider.ModelCapabilitiesOf(model)
	if caps.Temperature {
		t.Error("o3 should not support temperature")
	}
	if !caps.Reasoning {
		t.Error("o3 should be reasoning")
	}
}

func TestChat_NoAuth(t *testing.T) {
	model := Chat("gpt-4o") // no auth configured
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for missing auth")
	}
	if !strings.Contains(err.Error(), "no API key or token source") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestChat_CustomHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Custom") != "value" {
			t.Errorf("missing custom header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithHeaders(map[string]string{"X-Custom": "value"}),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

func TestChat_TokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer dynamic-token" {
			t.Errorf("unexpected auth: %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o",
		WithTokenSource(provider.StaticToken("dynamic-token")),
		WithBaseURL(server.URL),
	)
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

// --- Request body tests ---

func TestChat_ChatCompletions_RequestBody(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	temp := 0.7
	model := Chat("gpt-4o", WithAPIKey("key"), WithBaseURL(server.URL))
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		System: "be helpful",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		MaxOutputTokens: 100,
		Temperature:     &temp,
		Tools: []provider.ToolDefinition{{
			Name:        "read",
			Description: "Read a file",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
		ToolChoice:      "auto",
		ProviderOptions: chatCompletionsOpts,
	})

	if capturedBody["model"] != "gpt-4o" {
		t.Errorf("model = %v", capturedBody["model"])
	}
	if capturedBody["max_tokens"] != float64(100) {
		t.Errorf("max_tokens = %v", capturedBody["max_tokens"])
	}
	if capturedBody["temperature"] != 0.7 {
		t.Errorf("temperature = %v", capturedBody["temperature"])
	}
	if capturedBody["tool_choice"] != "auto" {
		t.Errorf("tool_choice = %v", capturedBody["tool_choice"])
	}
	tools, ok := capturedBody["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Errorf("tools = %v", capturedBody["tools"])
	}
}

func TestChat_Responses_RequestBody(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp-1","model":"o3","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		System: "be helpful",
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		MaxOutputTokens: 200,
		Tools: []provider.ToolDefinition{{
			Name:        "read",
			Description: "Read a file",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
		ToolChoice: "required",
	})

	if capturedBody["model"] != "o3" {
		t.Errorf("model = %v", capturedBody["model"])
	}
	if capturedBody["instructions"] != "be helpful" {
		t.Errorf("instructions = %v", capturedBody["instructions"])
	}
	if capturedBody["max_output_tokens"] != float64(200) {
		t.Errorf("max_output_tokens = %v", capturedBody["max_output_tokens"])
	}
	if capturedBody["tool_choice"] != "required" {
		t.Errorf("tool_choice = %v", capturedBody["tool_choice"])
	}
	// Responses API uses flat tool format.
	tools, ok := capturedBody["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", capturedBody["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["name"] != "read" {
		t.Errorf("tool.name = %v, want read", tool["name"])
	}
	// Flat format: name is top-level, not nested under "function".
	if _, hasFunction := tool["function"]; hasFunction {
		t.Error("Responses API tools should not have nested 'function' field")
	}

	// Input format (not messages).
	input, ok := capturedBody["input"].([]any)
	if !ok || len(input) != 1 {
		t.Fatalf("input = %v", capturedBody["input"])
	}
}

// --- Responses API message conversion tests ---

func TestConvertToResponsesInput(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleSystem, Content: []provider.Part{{Type: provider.PartText, Text: "system msg"}}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartText, Text: "I'll read that"},
			{Type: provider.PartToolCall, ToolCallID: "tc1", ToolName: "read", ToolInput: json.RawMessage(`{"path":"a.txt"}`)},
		}},
		{Role: provider.RoleTool, Content: []provider.Part{
			{Type: provider.PartToolResult, ToolCallID: "tc1", ToolName: "read", ToolOutput: "file contents"},
		}},
	}

	result := convertToResponsesInput(msgs)

	if len(result) != 5 { // system(developer) + user + assistant_message + function_call + function_call_output
		t.Fatalf("expected 5 items, got %d: %v", len(result), result)
	}

	// System → developer role
	if result[0]["role"] != "developer" {
		t.Errorf("result[0].role = %v, want developer", result[0]["role"])
	}

	// User message
	if result[1]["role"] != "user" {
		t.Errorf("result[1].role = %v, want user", result[1]["role"])
	}

	// Assistant text → message type
	if result[2]["type"] != "message" {
		t.Errorf("result[2].type = %v, want message", result[2]["type"])
	}

	// Tool call → function_call type
	if result[3]["type"] != "function_call" {
		t.Errorf("result[3].type = %v, want function_call", result[3]["type"])
	}
	if result[3]["name"] != "read" {
		t.Errorf("result[3].name = %v, want read", result[3]["name"])
	}

	// Tool result → function_call_output type
	if result[4]["type"] != "function_call_output" {
		t.Errorf("result[4].type = %v, want function_call_output", result[4]["type"])
	}
	if result[4]["call_id"] != "tc1" {
		t.Errorf("result[4].call_id = %v, want tc1", result[4]["call_id"])
	}
}

func TestConvertToResponsesInput_UserWithImage(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartText, Text: "describe this"},
			{Type: provider.PartImage, URL: "data:image/png;base64,abc"},
		}},
	}

	result := convertToResponsesInput(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 item, got %d", len(result))
	}

	content, ok := result[0]["content"].([]map[string]any)
	if !ok || len(content) != 2 {
		t.Fatalf("content = %v", result[0]["content"])
	}
	if content[0]["type"] != "input_text" {
		t.Errorf("content[0].type = %v, want input_text", content[0]["type"])
	}
	if content[1]["type"] != "input_image" {
		t.Errorf("content[1].type = %v, want input_image", content[1]["type"])
	}
}

// --- mapResponsesFinishReason tests ---

func TestMapResponsesFinishReason(t *testing.T) {
	tests := []struct {
		name             string
		eventType        string
		incompleteReason string
		hasFunctionCall  bool
		want             provider.FinishReason
	}{
		{"completed", "response.completed", "", false, provider.FinishStop},
		{"tool_calls", "response.completed", "", true, provider.FinishToolCalls},
		{"incomplete_length", "response.incomplete", "max_output_tokens", false, provider.FinishLength},
		{"incomplete_filter", "response.incomplete", "content_filter", false, provider.FinishContentFilter},
		{"incomplete_default", "response.incomplete", "", false, provider.FinishOther},
		{"incomplete_with_tools", "response.incomplete", "max_output_tokens", true, provider.FinishToolCalls},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mapResponsesFinishReason(tt.eventType, tt.incompleteReason, tt.hasFunctionCall)
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// --- Responses API error event tests ---

func TestChat_Responses_ErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: error\n")
		_, _ = fmt.Fprint(w, `data: {"message":"context overflow","code":"context_overflow"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotOverflow bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var coe *goai.ContextOverflowError
			if errors.As(chunk.Error, &coe) {
				gotOverflow = true
			}
		}
	}
	if !gotOverflow {
		t.Error("expected ContextOverflowError")
	}
}

func TestChat_Responses_Generate_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp-1","model":"o3","status":"failed","error":{"message":"context too long","code":"context_length_exceeded"}}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var coe *goai.ContextOverflowError
	if !errors.As(err, &coe) {
		t.Errorf("expected ContextOverflowError, got %T: %v", err, err)
	}
}

// --- Provider options passthrough ---

func TestChat_Responses_ProviderOptions(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp-1","model":"o3","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"serviceTier":     "default",
			"reasoning_effort": "medium",
		},
	})

	if capturedBody["service_tier"] != "default" {
		t.Errorf("service_tier = %v", capturedBody["service_tier"])
	}
	reasoning, ok := capturedBody["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning = %v", capturedBody["reasoning"])
	}
	if reasoning["effort"] != "medium" {
		t.Errorf("reasoning.effort = %v", reasoning["effort"])
	}
}

// --- Additional coverage tests ---

func TestChat_Responses_ProviderOptions_All(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp-1","model":"o3","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"parallelToolCalls": true,
			"truncation":        "auto",
			"include":           []string{"reasoning.encrypted_content"},
			"reasoning_summary":  "auto",
		},
	})

	if capturedBody["parallel_tool_calls"] != true {
		t.Errorf("parallel_tool_calls = %v", capturedBody["parallel_tool_calls"])
	}
	if capturedBody["truncation"] != "auto" {
		t.Errorf("truncation = %v", capturedBody["truncation"])
	}
	if capturedBody["include"] == nil {
		t.Error("missing include")
	}
	reasoning, ok := capturedBody["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning = %v", capturedBody["reasoning"])
	}
	if reasoning["summary"] != "auto" {
		t.Errorf("reasoning.summary = %v", reasoning["summary"])
	}
}

func TestBuildResponsesRequest_StopSequences(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{
		StopSequences: []string{"END", "STOP"},
	}, "o3", false)

	stops, ok := body["stop"].([]string)
	if !ok || len(stops) != 2 {
		t.Errorf("stop = %v", body["stop"])
	}
}

func TestBuildResponsesRequest_ToolChoiceSpecific(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{
		ToolChoice: "my_tool",
	}, "o3", false)

	tc, ok := body["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %v", body["tool_choice"])
	}
	if tc["type"] != "function" || tc["name"] != "my_tool" {
		t.Errorf("tool_choice = %v", tc)
	}
}

func TestBuildResponsesRequest_TemperatureAndTopP(t *testing.T) {
	temp := 0.5
	topP := 0.9
	body := buildResponsesRequest(provider.GenerateParams{
		Temperature: &temp,
		TopP:        &topP,
	}, "o3", false)

	if body["temperature"] != 0.5 {
		t.Errorf("temperature = %v", body["temperature"])
	}
	if body["top_p"] != 0.9 {
		t.Errorf("top_p = %v", body["top_p"])
	}
}

func TestConvertToResponsesInput_FileAttachment(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartText, Text: "read this PDF"},
			{Type: provider.PartFile, URL: "data:application/pdf;base64,abc", MediaType: "application/pdf", Filename: "doc.pdf"},
		}},
	}

	result := convertToResponsesInput(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 item, got %d", len(result))
	}

	content, ok := result[0]["content"].([]map[string]any)
	if !ok || len(content) != 2 {
		t.Fatalf("content = %v", result[0]["content"])
	}
	if content[1]["type"] != "input_file" {
		t.Errorf("content[1].type = %v, want input_file", content[1]["type"])
	}
	if content[1]["filename"] != "doc.pdf" {
		t.Errorf("content[1].filename = %v, want doc.pdf", content[1]["filename"])
	}
}

func TestConvertToResponsesInput_EmptyContent(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartText, Text: ""}, // empty text part
		}},
	}

	result := convertToResponsesInput(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 item, got %d", len(result))
	}
	// Falls back to plain text
	content, ok := result[0]["content"].([]map[string]any)
	if !ok {
		t.Fatalf("content = %v", result[0]["content"])
	}
	if content[0]["type"] != "input_text" {
		t.Errorf("content[0].type = %v, want input_text", content[0]["type"])
	}
}

func TestStreamResponses_RefusalDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.refusal.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"I cannot help with that"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":1,"output_tokens":5}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "bad"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotText bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText && chunk.Text == "I cannot help with that" {
			gotText = true
		}
	}
	if !gotText {
		t.Error("missing refusal text chunk")
	}
}

func TestStreamResponses_InsufficientQuota(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.failed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"error":{"message":"quota exceeded","code":"insufficient_quota"}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var apiErr *goai.APIError
			if errors.As(chunk.Error, &apiErr) && !apiErr.IsRetryable {
				gotError = true
			}
		}
	}
	if !gotError {
		t.Error("expected non-retryable APIError")
	}
}

func TestStreamResponses_UsageNotIncluded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.failed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"error":{"message":"not included","code":"usage_not_included"}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var apiErr *goai.APIError
			if errors.As(chunk.Error, &apiErr) {
				if !apiErr.IsRetryable {
					gotError = true
				}
			}
		}
	}
	if !gotError {
		t.Error("expected non-retryable APIError for usage_not_included")
	}
}

func TestStreamResponses_InvalidPrompt(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.failed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"error":{"message":"bad prompt","code":"invalid_prompt"}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var apiErr *goai.APIError
			if errors.As(chunk.Error, &apiErr) {
				if !apiErr.IsRetryable && apiErr.Message == "bad prompt" {
					gotError = true
				}
			}
		}
	}
	if !gotError {
		t.Error("expected non-retryable APIError for invalid_prompt")
	}
}

func TestStreamResponses_GenericFailedError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.failed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"error":{"message":"something went wrong","code":"unknown"}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected error chunk")
	}
}

func TestStreamResponses_ErrorWithMaxTokensCode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: error\n")
		_, _ = fmt.Fprint(w, `data: {"message":"too many tokens","code":"max_tokens"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotOverflow bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var coe *goai.ContextOverflowError
			if errors.As(chunk.Error, &coe) {
				gotOverflow = true
			}
		}
	}
	if !gotOverflow {
		t.Error("expected ContextOverflowError")
	}
}

func TestStreamResponses_GenericErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: error\n")
		_, _ = fmt.Fprint(w, `data: {"message":"server error","code":"internal"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var apiErr *goai.APIError
			if errors.As(chunk.Error, &apiErr) {
				if apiErr.Message == "server error" {
					gotError = true
				}
			}
		}
	}
	if !gotError {
		t.Error("expected APIError")
	}
}

func TestStreamResponses_FailedEmptyMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.failed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"error":{"message":"","code":"unknown"}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var apiErr *goai.APIError
			if errors.As(chunk.Error, &apiErr) {
				if apiErr.Message == "response failed" {
					gotError = true
				}
			}
		}
	}
	if !gotError {
		t.Error("expected APIError with default message")
	}
}

func TestStreamResponses_ErrorEmptyMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: error\n")
		_, _ = fmt.Fprint(w, `data: {"message":"","code":"internal"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotError bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			var apiErr *goai.APIError
			if errors.As(chunk.Error, &apiErr) {
				if apiErr.Message == "stream error" {
					gotError = true
				}
			}
		}
	}
	if !gotError {
		t.Error("expected APIError with default message")
	}
}

func TestParseResponsesResult_IncompleteStatus(t *testing.T) {
	body := `{
		"id": "resp-1",
		"model": "o3",
		"status": "incomplete",
		"output": [{"type": "message", "content": [{"type": "output_text", "text": "partial"}]}],
		"incomplete_details": {"reason": "content_filter"},
		"usage": {"input_tokens": 5, "output_tokens": 3}
	}`

	result, err := parseResponsesResult([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishContentFilter {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishContentFilter)
	}
	if result.Text != "partial" {
		t.Errorf("Text = %q, want %q", result.Text, "partial")
	}
}

func TestParseResponsesResult_GenericError(t *testing.T) {
	body := `{
		"id": "resp-1",
		"model": "o3",
		"status": "failed",
		"error": {"message": "some error", "code": "unknown"}
	}`

	_, err := parseResponsesResult([]byte(body))
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *goai.APIError
	if !errors.As(err, &apiErr) {
		t.Errorf("expected APIError, got %T: %v", err, err)
	} else if apiErr.Message != "some error" {
		t.Errorf("Message = %q, want %q", apiErr.Message, "some error")
	}
}

func TestParseResponsesResult_InvalidJSON(t *testing.T) {
	_, err := parseResponsesResult([]byte("not json"))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestParseResponsesResult_CachedTokens(t *testing.T) {
	body := `{
		"id": "resp-1",
		"model": "o3",
		"status": "completed",
		"output": [],
		"usage": {
			"input_tokens": 100,
			"output_tokens": 10,
			"input_tokens_details": {"cached_tokens": 30}
		}
	}`

	result, err := parseResponsesResult([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", result.Usage.CacheReadTokens)
	}
}

func TestChat_ChatCompletions_NetworkError(t *testing.T) {
	// Use a server that immediately closes
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			return
		}
		conn, _, _ := hj.Hijack()
		_ = conn.Close()
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestStreamResponses_CachedTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"ok"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":100,"output_tokens":10,"input_tokens_details":{"cached_tokens":30}}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var usage provider.Usage
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			usage = chunk.Usage
		}
	}
	if usage.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", usage.CacheReadTokens)
	}
}

func TestStreamResponses_ContentFilterIncomplete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.incomplete\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"incomplete_details":{"reason":"content_filter"},"usage":{"input_tokens":5,"output_tokens":0}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var finishReason provider.FinishReason
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkStepFinish {
			finishReason = chunk.FinishReason
		}
	}
	if finishReason != provider.FinishContentFilter {
		t.Errorf("FinishReason = %q, want %q", finishReason, provider.FinishContentFilter)
	}
}

func TestStreamResponses_DONE(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotFinish bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			gotFinish = true
		}
	}
	if !gotFinish {
		t.Error("expected finish chunk from [DONE]")
	}
}

func TestStreamResponses_ScannerError(t *testing.T) {
	// Directly test streamResponses with an error reader.
	out := make(chan provider.StreamChunk, 64)
	go streamResponses(t.Context(), &errorReader{}, out)

	var gotError bool
	for chunk := range out {
		if chunk.Type == provider.ChunkError {
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected error chunk from scanner error")
	}
}

func TestDoGenerateChatCompletions_ReadError(t *testing.T) {
	// Server that closes connection mid-body to trigger io.ReadAll error.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000") // lie about content length
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"ch`) // truncated
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: chatCompletionsOpts,
	})
	// Either read error or parse error -- both are valid
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDoGenerateResponses_ReadError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"id`) // truncated
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDoStreamResponses_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, `{"error":{"message":"server error"}}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDoGenerateResponses_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"message":"bad request"}}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestBuildResponsesRequest_NoSystem(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{}, "o3", false)
	if _, ok := body["instructions"]; ok {
		t.Error("instructions should not be set when system is empty")
	}
}

func TestStreamResponses_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel() // cancel immediately

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Write some data that will be read after context cancel
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"hello"}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	// Context already cancelled - doHTTP should fail
	_, err := model.DoStream(ctx, provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	// Should get an error because context is cancelled
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestBuildResponsesRequest_InvalidSchema(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{
		Tools: []provider.ToolDefinition{{
			Name:        "test",
			Description: "test tool",
			InputSchema: json.RawMessage(`invalid json`),
		}},
	}, "o3", false)

	tools, ok := body["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", body["tools"])
	}
	// Should fall back to empty object
	params := tools[0]["parameters"]
	if params == nil {
		t.Error("parameters should be empty object fallback, not nil")
	}
}

func TestStreamResponses_ContextCancelDuringStream(t *testing.T) {
	// With TrySend, the goroutine exits cleanly when ctx is cancelled.
	// This test verifies no hang or panic.
	ctx, cancel := context.WithCancel(t.Context())
	cancel() // cancel immediately

	input := "event: response.output_text.delta\ndata: {\"delta\":\"hello\"}\n\n" +
		"event: response.output_text.delta\ndata: {\"delta\":\"world\"}\n\n"

	out := make(chan provider.StreamChunk) // unbuffered
	done := make(chan struct{})
	go func() {
		streamResponses(ctx, io.NopCloser(strings.NewReader(input)), out)
		close(done)
	}()
	<-done
	for range out {
	}
}

func TestParseResponsesResult_IncompleteNoDetails(t *testing.T) {
	body := `{
		"id": "resp-1",
		"model": "o3",
		"status": "incomplete",
		"output": [],
		"usage": {"input_tokens": 5, "output_tokens": 3}
	}`

	result, err := parseResponsesResult([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	// Default to FinishOther when no details
	if result.FinishReason != provider.FinishOther {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishOther)
	}
}

func TestStreamResponses_ToolCallFlushOnComplete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"function_call","call_id":"tc1","name":"read"}}`+"\n\n")
		// Send partial args that don't form valid JSON
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"delta":"{\"path\":"}`+"\n\n")
		// Complete without function_call_arguments.done -- flush on response.completed
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCalls = append(toolCalls, chunk)
		}
	}
	// Partial args should be flushed on response.completed
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call (flushed), got %d", len(toolCalls))
	}
	if toolCalls[0].ToolInput != `{"path":` {
		t.Errorf("ToolInput = %q", toolCalls[0].ToolInput)
	}
}

func TestDoHTTP_RequestHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Request-Header"); got != "from-body" {
			t.Errorf("X-Request-Header = %q, want %q", got, "from-body")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	m := &chatModel{
		id:   "gpt-4o",
		opts: options{baseURL: server.URL, tokenSource: provider.StaticToken("key")},
	}
	body := map[string]any{
		"model":    "gpt-4o",
		"_headers": map[string]string{"X-Request-Header": "from-body"},
	}
	resp, err := m.doHTTP(t.Context(), server.URL+"/chat/completions", body)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
}

func TestWithHTTPClient(t *testing.T) {
	// Custom transport that adds a marker header to verify it was used.
	customTransport := &headerInjectTransport{
		base:   http.DefaultTransport,
		header: "X-Custom-Transport",
		value:  "injected",
	}
	customClient := &http.Client{Transport: customTransport}

	var gotHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Custom-Transport")
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithHTTPClient(customClient),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: "user", Content: []provider.Part{{Type: "text", Text: "hi"}}}},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotHeader != "injected" {
		t.Errorf("custom transport not used: X-Custom-Transport = %q, want %q", gotHeader, "injected")
	}
}

// headerInjectTransport is a test RoundTripper that injects a header.
type headerInjectTransport struct {
	base   http.RoundTripper
	header string
	value  string
}

func (t *headerInjectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set(t.header, t.value)
	return t.base.RoundTrip(req)
}

func TestWithHTTPClient_Default(t *testing.T) {
	// When no HTTPClient is set, should use http.DefaultClient (no panic).
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: "user", Content: []provider.Part{{Type: "text", Text: "hi"}}}},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestStreamResponses_DeltaWithoutOutputItemAdded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Send delta without a prior output_item.added -- hits the active == nil branch
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":5,"delta":"{\"x\":1}"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCalls = append(toolCalls, chunk)
		}
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].ToolInput != `{"x":1}` {
		t.Errorf("ToolInput = %q, want %q", toolCalls[0].ToolInput, `{"x":1}`)
	}
}

func TestStreamResponses_ArgumentsDoneFlush(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"function_call","call_id":"tc1","name":"read"}}`+"\n\n")
		// Send partial args that don't form valid JSON
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"delta":"{\"file\":"}`+"\n\n")
		// function_call_arguments.done should flush remaining
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.done\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCalls = append(toolCalls, chunk)
		}
	}
	// Should get one flush from arguments.done
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].ToolInput != `{"file":` {
		t.Errorf("ToolInput = %q", toolCalls[0].ToolInput)
	}
}

func TestParseResponsesResult_IncompleteMaxOutputTokens(t *testing.T) {
	body := `{
		"id": "resp-1",
		"model": "o3",
		"status": "incomplete",
		"incomplete_details": {"reason": "max_output_tokens"},
		"output": [{"type":"message","content":[{"type":"output_text","text":"partial"}]}],
		"usage": {"input_tokens": 10, "output_tokens": 50}
	}`

	result, err := parseResponsesResult([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if result.FinishReason != provider.FinishLength {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishLength)
	}
	if result.Text != "partial" {
		t.Errorf("Text = %q, want %q", result.Text, "partial")
	}
}

// --- Coverage gap tests ---

func TestBuildResponsesRequest_Headers(t *testing.T) {
	// Covers lines 83-85: _headers injection in buildResponsesRequest.
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{
			"X-Custom": "value1",
			"X-Other":  "value2",
		},
	}
	body := buildResponsesRequest(params, "o3", false)

	headers, ok := body["_headers"].(map[string]string)
	if !ok {
		t.Fatalf("_headers not set or wrong type: %T", body["_headers"])
	}
	if headers["X-Custom"] != "value1" {
		t.Errorf("X-Custom = %q, want %q", headers["X-Custom"], "value1")
	}
	if headers["X-Other"] != "value2" {
		t.Errorf("X-Other = %q, want %q", headers["X-Other"], "value2")
	}
}

func TestBuildResponsesRequest_NoHeaders(t *testing.T) {
	// When Headers is nil/empty, _headers should NOT be in body.
	body := buildResponsesRequest(provider.GenerateParams{}, "o3", false)
	if _, ok := body["_headers"]; ok {
		t.Error("_headers should not be set when Headers is empty")
	}
}

func TestApplyResponsesProviderOptions_PromptCacheKey(t *testing.T) {
	// Covers lines 119-121: promptCacheKey branch.
	body := map[string]any{}
	applyResponsesProviderOptions(body, map[string]any{
		"prompt_cache_key": "my-cache-key",
	})
	if body["prompt_cache_key"] != "my-cache-key" {
		t.Errorf("prompt_cache_key = %v, want %q", body["prompt_cache_key"], "my-cache-key")
	}
}

func TestApplyResponsesProviderOptions_TextVerbosity(t *testing.T) {
	body := map[string]any{}
	applyResponsesProviderOptions(body, map[string]any{
		"text_verbosity": "concise",
	})
	text, ok := body["text"].(map[string]any)
	if !ok {
		t.Fatalf("text = %v, want map", body["text"])
	}
	if text["verbosity"] != "concise" {
		t.Errorf("text.verbosity = %v, want %q", text["verbosity"], "concise")
	}
}

func TestGetOrCreateMap_ExistingMap(t *testing.T) {
	body := map[string]any{
		"reasoning": map[string]any{"effort": "high"},
	}
	m := getOrCreateMap(body, "reasoning")
	if m["effort"] != "high" {
		t.Errorf("effort = %v, want %q", m["effort"], "high")
	}
}

func TestStreamResponses_ReasoningWithCanonicalID(t *testing.T) {
	// Covers lines 316-319 (canonical ID from activeReasoning),
	// line 330 (response.reasoning_summary_part.added no-op),
	// lines 357-361 (output_item.added with type "reasoning"),
	// lines 416-422 (output_item.done cleanup for reasoning).
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		// 1. Reasoning item added -- sets canonical ID and currentReasoningIdx.
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"reasoning","id":"rs_canonical_001"}}`+"\n\n")

		// 2. Summary part added -- no-op event, just tracked.
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_part.added\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item_id":"rs_canonical_001","summary_index":0}`+"\n\n")

		// 3. Summary text delta -- should use canonical ID from activeReasoning.
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"item_id":"rs_rotated_999","summary_index":0,"delta":"Let me think..."}`+"\n\n")

		// 4. Another summary text delta with different summary_index.
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"item_id":"rs_rotated_999","summary_index":1,"delta":"More thinking"}`+"\n\n")

		// 5. Reasoning output item done -- cleanup.
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"reasoning"}}`+"\n\n")

		// 6. Text output.
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"The answer is 42."}`+"\n\n")

		// 7. Complete.
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":10,"output_tokens":20,"output_tokens_details":{"reasoning_tokens":8}}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var reasoningChunks []provider.StreamChunk
	var gotText bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkReasoning {
			reasoningChunks = append(reasoningChunks, chunk)
		}
		if chunk.Type == provider.ChunkText && chunk.Text == "The answer is 42." {
			gotText = true
		}
	}

	if len(reasoningChunks) != 2 {
		t.Fatalf("expected 2 reasoning chunks, got %d", len(reasoningChunks))
	}

	// First reasoning chunk should use canonical ID (rs_canonical_001), not the rotated one.
	meta0 := reasoningChunks[0].Metadata
	if meta0 == nil {
		t.Fatal("reasoning chunk[0] has no metadata")
	}
	rid0, _ := meta0["reasoningId"].(string)
	if rid0 != "rs_canonical_001:0" {
		t.Errorf("reasoningId[0] = %q, want %q", rid0, "rs_canonical_001:0")
	}
	if reasoningChunks[0].Text != "Let me think..." {
		t.Errorf("reasoning text[0] = %q", reasoningChunks[0].Text)
	}

	// Second reasoning chunk should also use canonical ID but summary_index=1.
	meta1 := reasoningChunks[1].Metadata
	rid1, _ := meta1["reasoningId"].(string)
	if rid1 != "rs_canonical_001:1" {
		t.Errorf("reasoningId[1] = %q, want %q", rid1, "rs_canonical_001:1")
	}

	if !gotText {
		t.Error("missing text chunk")
	}
}

func TestStreamResponses_FunctionCallOutputItemDone(t *testing.T) {
	// Covers line 416-417: response.output_item.done with type "function_call".
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		// Function call added.
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"function_call","call_id":"tc1","name":"read"}}`+"\n\n")

		// Args as valid JSON.
		_, _ = fmt.Fprint(w, "event: response.function_call_arguments.delta\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"delta":"{\"path\":\"a.go\"}"}`+"\n\n")

		// Function call done -- should clean up activeTools[0].
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"function_call"}}`+"\n\n")

		// Complete -- the tool should already be cleaned up, no double-flush.
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolCalls []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCall {
			toolCalls = append(toolCalls, chunk)
		}
	}
	// Should get exactly 1 tool call (from delta), not a second flush on completed.
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	if toolCalls[0].ToolInput != `{"path":"a.go"}` {
		t.Errorf("ToolInput = %q", toolCalls[0].ToolInput)
	}
}

func TestStreamResponses_ReasoningOutputItemDoneResetsIndex(t *testing.T) {
	// Verifies that after reasoning output_item.done, currentReasoningIdx resets
	// and subsequent reasoning_summary_text.delta without a new output_item.added
	// falls back to using item_id directly (no canonical override).
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		// Reasoning item added + done.
		_, _ = fmt.Fprint(w, "event: response.output_item.added\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"reasoning","id":"rs_first"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.output_item.done\n")
		_, _ = fmt.Fprint(w, `data: {"output_index":0,"item":{"type":"reasoning"}}`+"\n\n")

		// Another reasoning summary delta after the reasoning item is done --
		// currentReasoningIdx should be -1 now, so it uses the raw item_id.
		_, _ = fmt.Fprint(w, "event: response.reasoning_summary_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"item_id":"rs_orphan","summary_index":0,"delta":"orphan thought"}`+"\n\n")

		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var reasoningChunks []provider.StreamChunk
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkReasoning {
			reasoningChunks = append(reasoningChunks, chunk)
		}
	}

	if len(reasoningChunks) != 1 {
		t.Fatalf("expected 1 reasoning chunk, got %d", len(reasoningChunks))
	}
	rid, _ := reasoningChunks[0].Metadata["reasoningId"].(string)
	// Should use raw item_id since currentReasoningIdx was reset to -1.
	if rid != "rs_orphan:0" {
		t.Errorf("reasoningId = %q, want %q", rid, "rs_orphan:0")
	}
}

func TestBuildResponsesRequest_ResponseFormat(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test_schema",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
	}, "o3", false)

	text, ok := body["text"].(map[string]any)
	if !ok {
		t.Fatalf("text not set or wrong type: %T", body["text"])
	}
	format, ok := text["format"].(map[string]any)
	if !ok {
		t.Fatalf("format not set or wrong type: %T", text["format"])
	}
	if format["type"] != "json_schema" {
		t.Errorf("format.type = %v, want json_schema", format["type"])
	}
	if format["name"] != "test_schema" {
		t.Errorf("format.name = %v, want test_schema", format["name"])
	}
	// Default strictJsonSchema is false (matching Vercel).
	if format["strict"] != false {
		t.Errorf("format.strict = %v, want false", format["strict"])
	}
	schema, ok := format["schema"].(map[string]any)
	if !ok {
		t.Fatalf("schema not a map: %T", format["schema"])
	}
	if schema["type"] != "object" {
		t.Errorf("schema.type = %v, want object", schema["type"])
	}
}

func TestBuildResponsesRequest_ResponseFormat_InvalidSchema(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "broken",
			Schema: json.RawMessage(`not valid json`),
		},
	}, "o3", false)

	// When schema is invalid JSON, falls back to json_object mode.
	text, ok := body["text"].(map[string]any)
	if !ok {
		t.Fatal("text should be set for json_object fallback")
	}
	format, ok := text["format"].(map[string]any)
	if !ok {
		t.Fatal("format not set")
	}
	if format["type"] != "json_object" {
		t.Errorf("format.type = %v, want json_object", format["type"])
	}
}

// =============================================================================
// BF.13 -- Vercel parity tests
// =============================================================================

// Item 1: TotalTokens computation in non-streaming response.
func TestBF13_TotalTokens_ChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"x","model":"gpt-4o",
			"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}
		}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.TotalTokens != 120 {
		t.Errorf("TotalTokens = %d, want 120", result.Usage.TotalTokens)
	}
}

// Item 1: TotalTokens in Responses API.
func TestBF13_TotalTokens_Responses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"resp-1","model":"o3","status":"completed",
			"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],
			"usage":{"input_tokens":50,"output_tokens":30}
		}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.TotalTokens != 80 {
		t.Errorf("TotalTokens = %d, want 80", result.Usage.TotalTokens)
	}
}

// Item 2: store no longer hardcoded, configurable via ProviderOptions.
func TestBF13_StoreFromProviderOptions(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp-1","model":"gpt-4o","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"store": true},
	})

	if capturedBody["store"] != true {
		t.Errorf("store = %v, want true", capturedBody["store"])
	}
}

// Item 3: Default to Responses API for all models.
func TestBF13_DefaultResponsesAPI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("expected /responses, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp-1","model":"gpt-4o","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// Item 4: Responses API provider options (metadata, logprobs, user, etc.)
func TestBF13_ResponsesProviderOptions(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp-1","model":"o3","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("k"), WithBaseURL(server.URL))
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"metadata":           map[string]string{"session": "test"},
			"user":               "user-123",
			"safetyIdentifier":   "safe-id",
			"maxToolCalls":       10,
			"conversation":       "conv-123",
			"previousResponseId": "resp-0",
			"logprobs":           true,
		},
	})

	if capturedBody["metadata"] == nil {
		t.Error("metadata not set")
	}
	if capturedBody["user"] != "user-123" {
		t.Errorf("user = %v, want user-123", capturedBody["user"])
	}
	if capturedBody["safety_identifier"] != "safe-id" {
		t.Errorf("safety_identifier = %v", capturedBody["safety_identifier"])
	}
	if capturedBody["max_tool_calls"] != float64(10) {
		t.Errorf("max_tool_calls = %v", capturedBody["max_tool_calls"])
	}
	if capturedBody["conversation"] != "conv-123" {
		t.Errorf("conversation = %v", capturedBody["conversation"])
	}
	if capturedBody["previous_response_id"] != "resp-0" {
		t.Errorf("previous_response_id = %v", capturedBody["previous_response_id"])
	}
	if capturedBody["top_logprobs"] != float64(20) {
		t.Errorf("top_logprobs = %v, want 20", capturedBody["top_logprobs"])
	}
	// logprobs should add to include array
	includes, _ := capturedBody["include"].([]any)
	found := false
	for _, inc := range includes {
		if inc == "message.output_text.logprobs" {
			found = true
		}
	}
	if !found {
		t.Errorf("include should contain message.output_text.logprobs, got %v", includes)
	}
}

// Item 7: JSON object mode (schema-less).
func TestBF13_JsonObjectMode_Responses(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name: "test",
			// No schema -- should produce json_object mode.
		},
	}, "gpt-4o", false)

	text, ok := body["text"].(map[string]any)
	if !ok {
		t.Fatal("text not set")
	}
	format, ok := text["format"].(map[string]any)
	if !ok {
		t.Fatal("format not set")
	}
	if format["type"] != "json_object" {
		t.Errorf("format.type = %v, want json_object", format["type"])
	}
}

// Item 8: structuredOutputs toggle.
func TestBF13_StructuredOutputsToggle(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test",
			Schema: json.RawMessage(`{"type":"object"}`),
		},
		ProviderOptions: map[string]any{"structuredOutputs": false},
	}, "gpt-4o", false)

	text, ok := body["text"].(map[string]any)
	if !ok {
		t.Fatal("text not set")
	}
	format, ok := text["format"].(map[string]any)
	if !ok {
		t.Fatal("format not set")
	}
	// When structuredOutputs is false, falls back to json_object.
	if format["type"] != "json_object" {
		t.Errorf("format.type = %v, want json_object", format["type"])
	}
}

// Item 9: strictJsonSchema option.
func TestBF13_StrictJsonSchema(t *testing.T) {
	body := buildResponsesRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test",
			Schema: json.RawMessage(`{"type":"object"}`),
		},
		ProviderOptions: map[string]any{"strictJsonSchema": true},
	}, "gpt-4o", false)

	text := body["text"].(map[string]any)
	format := text["format"].(map[string]any)
	if format["strict"] != true {
		t.Errorf("format.strict = %v, want true", format["strict"])
	}
}

// Item 10: extract prediction tokens from completion_tokens_details.
func TestBF13_PredictionTokens_ChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"x","model":"gpt-4o",
			"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
			"usage":{
				"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,
				"completion_tokens_details":{"reasoning_tokens":5,"accepted_prediction_tokens":10,"rejected_prediction_tokens":3}
			}
		}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: chatCompletionsOpts,
	})
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

// Item 11: extract url_citation from Responses API non-streaming.
func TestBF13_URLCitation_Responses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"resp-1","model":"gpt-4o","status":"completed",
			"output":[{"type":"message","content":[{
				"type":"output_text","text":"According to...",
				"annotations":[{
					"type":"url_citation",
					"url_citation":{"url":"https://example.com","title":"Example","start_index":14,"end_index":20}
				}]
			}]}],
			"usage":{"input_tokens":10,"output_tokens":5}
		}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Sources) != 1 {
		t.Fatalf("Sources = %d, want 1", len(result.Sources))
	}
	s := result.Sources[0]
	if s.URL != "https://example.com" {
		t.Errorf("URL = %q", s.URL)
	}
	if s.Title != "Example" {
		t.Errorf("Title = %q", s.Title)
	}
	if s.StartIndex != 14 || s.EndIndex != 20 {
		t.Errorf("range = [%d,%d], want [14,20]", s.StartIndex, s.EndIndex)
	}
}

// Item 11: extract url_citation from Chat Completions non-streaming.
func TestBF13_URLCitation_ChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"x","model":"gpt-4o",
			"choices":[{
				"message":{
					"content":"According to...",
					"annotations":[{
						"type":"url_citation",
						"url_citation":{"url":"https://example.com","title":"Example","start_index":0,"end_index":10}
					}]
				},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
		}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: chatCompletionsOpts,
	})
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

// Item 12: extract logprobs from Chat Completions.
func TestBF13_Logprobs_ChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"x","model":"gpt-4o",
			"choices":[{
				"message":{"content":"ok"},
				"logprobs":{"content":[{"token":"ok","logprob":-0.5}]},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: chatCompletionsOpts,
	})
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

// Item 17: generate fallback tool call ID.
func TestBF13_ToolCallIDFallback_ChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Tool call without an ID.
		_, _ = fmt.Fprint(w, `{
			"id":"x","model":"gpt-4o",
			"choices":[{
				"message":{
					"content":"",
					"tool_calls":[{"id":"","type":"function","function":{"name":"read","arguments":"{}"}}]
				},
				"finish_reason":"tool_calls"
			}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: chatCompletionsOpts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
	if !strings.HasPrefix(result.ToolCalls[0].ID, "call_") {
		t.Errorf("expected generated ID starting with call_, got %q", result.ToolCalls[0].ID)
	}
	if len(result.ToolCalls[0].ID) < 10 {
		t.Errorf("generated ID too short: %q", result.ToolCalls[0].ID)
	}
}

// Item 5: parallelToolCalls via ProviderOptions in Chat Completions.
func TestBF13_ParallelToolCalls_ChatCompletions(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	_, _ = model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"useResponsesAPI": false, "parallelToolCalls": false},
	})

	if capturedBody["parallel_tool_calls"] != false {
		t.Errorf("parallel_tool_calls = %v, want false", capturedBody["parallel_tool_calls"])
	}
}

// Item 13: Image provider options passthrough.
func TestBF13_ImageProviderOptions(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		encoded := "aGVsbG8=" // base64 of "hello"
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("gpt-image-1", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "test",
		N:      1,
		ProviderOptions: map[string]any{
			"quality": "hd",
			"style":   "vivid",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if capturedBody["quality"] != "hd" {
		t.Errorf("quality = %v, want hd", capturedBody["quality"])
	}
	if capturedBody["style"] != "vivid" {
		t.Errorf("style = %v, want vivid", capturedBody["style"])
	}
}

// Item 14: extract revised_prompt in image metadata.
func TestBF13_ImageRevisedPrompt(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		encoded := "aGVsbG8="
		_, _ = fmt.Fprintf(w, `{"data":[{"b64_json":"%s","revised_prompt":"A beautiful cat sitting on a mat"}]}`, encoded)
	}))
	defer server.Close()

	model := Image("dall-e-3", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "cat", N: 1})
	if err != nil {
		t.Fatal(err)
	}
	pm := result.ProviderMetadata["openai"]
	if pm == nil {
		t.Fatal("ProviderMetadata[openai] missing")
	}
	images, ok := pm["images"].([]map[string]any)
	if !ok || len(images) != 1 {
		t.Fatalf("images metadata = %v", pm["images"])
	}
	if images[0]["revisedPrompt"] != "A beautiful cat sitting on a mat" {
		t.Errorf("revisedPrompt = %v", images[0]["revisedPrompt"])
	}
}

// Item 15: detect media type from response.
func TestBF13_ImageMediaTypeDetection(t *testing.T) {
	tests := []struct {
		format string
		want   string
	}{
		{"png", "image/png"},
		{"jpeg", "image/jpeg"},
		{"webp", "image/webp"},
		{"", "image/png"}, // default
	}
	for _, tt := range tests {
		got := detectMediaType(tt.format, "")
		if got != tt.want {
			t.Errorf("detectMediaType(%q) = %q, want %q", tt.format, got, tt.want)
		}
	}
}

// Item 16: embedding dimensions and user parameters.
func TestBF13_EmbeddingDimensionsAndUser(t *testing.T) {
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]any{{"embedding": []float64{0.1, 0.2}, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	defer server.Close()

	model := Embedding("text-embedding-3-small", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{
		ProviderOptions: map[string]any{
			"dimensions": 256,
			"user":       "user-123",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if capturedBody["dimensions"] != float64(256) {
		t.Errorf("dimensions = %v, want 256", capturedBody["dimensions"])
	}
	if capturedBody["user"] != "user-123" {
		t.Errorf("user = %v, want user-123", capturedBody["user"])
	}
}

// Item 6: image_url.detail field (tested via openaicompat -- see openaicompat_test.go).

// Item 3: Opt-out to Chat Completions via useResponsesAPI: false.
func TestBF13_OptOutResponsesAPI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected /chat/completions, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer server.Close()

	model := Chat("gpt-4o", WithAPIKey("k"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{"useResponsesAPI": false},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// Item 1: TotalTokens in streaming Responses API.
func TestBF13_TotalTokens_StreamResponses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: response.output_text.delta\n")
		_, _ = fmt.Fprint(w, `data: {"delta":"ok"}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: response.completed\n")
		_, _ = fmt.Fprint(w, `data: {"response":{"usage":{"input_tokens":40,"output_tokens":60}}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("o3", WithAPIKey("k"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var usage provider.Usage
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			usage = chunk.Usage
		}
	}
	if usage.TotalTokens != 100 {
		t.Errorf("TotalTokens = %d, want 100", usage.TotalTokens)
	}
}

// Verify goai.APIError and goai.ContextOverflowError still referenced properly.
func TestBF13_ErrorTypes(t *testing.T) {
	// Just verifying the import and usage.
	_ = goai.APIError{Message: "test"}
	_ = goai.ContextOverflowError{Message: "test"}
}

// --- 100% coverage gap tests ---

func TestDetectMediaType_GifAndJpg(t *testing.T) {
	// Covers image.go:147-148 (gif case) and jpg alias.
	if got := detectMediaType("gif", ""); got != "image/gif" {
		t.Errorf("detectMediaType(gif) = %q, want image/gif", got)
	}
	if got := detectMediaType("jpg", ""); got != "image/jpeg" {
		t.Errorf("detectMediaType(jpg) = %q, want image/jpeg", got)
	}
}

func TestApplyResponsesProviderOptions_InstructionsAndPreviousResponseId(t *testing.T) {
	// Covers responses.go:192-195 (instructions) and 196-199 (previousResponseId).
	body := map[string]any{}
	applyResponsesProviderOptions(body, map[string]any{
		"instructions":       "Be concise.",
		"previousResponseId": "resp_abc123",
	})
	if body["instructions"] != "Be concise." {
		t.Errorf("instructions = %v, want %q", body["instructions"], "Be concise.")
	}
	if body["previous_response_id"] != "resp_abc123" {
		t.Errorf("previous_response_id = %v, want %q", body["previous_response_id"], "resp_abc123")
	}
}

func TestApplyResponsesProviderOptions_LogprobsInt(t *testing.T) {
	// Covers responses.go:210-212 (logprobs as int).
	body := map[string]any{}
	applyResponsesProviderOptions(body, map[string]any{
		"logprobs": 5,
	})
	if body["top_logprobs"] != 5 {
		t.Errorf("top_logprobs = %v, want 5", body["top_logprobs"])
	}
	includes, ok := body["include"].([]string)
	if !ok || !slices.Contains(includes, "message.output_text.logprobs") {
		t.Errorf("include = %v, want to contain message.output_text.logprobs", body["include"])
	}
}

func TestApplyResponsesProviderOptions_LogprobsFloat64(t *testing.T) {
	// Covers responses.go:213-215 (logprobs as float64, e.g. from JSON unmarshal).
	body := map[string]any{}
	applyResponsesProviderOptions(body, map[string]any{
		"logprobs": float64(10),
	})
	if body["top_logprobs"] != 10 {
		t.Errorf("top_logprobs = %v, want 10", body["top_logprobs"])
	}
	includes, ok := body["include"].([]string)
	if !ok || !slices.Contains(includes, "message.output_text.logprobs") {
		t.Errorf("include = %v, want to contain message.output_text.logprobs", body["include"])
	}
}

func TestApplyResponsesProviderOptions_StoreWithReasoning(t *testing.T) {
	// Covers responses.go:243-245 (store=false + reasoning → encrypted_content include).
	body := map[string]any{}
	applyResponsesProviderOptions(body, map[string]any{
		"store":            false,
		"reasoning_effort": "high",
	})
	includes, ok := body["include"].([]string)
	if !ok || !slices.Contains(includes, "reasoning.encrypted_content") {
		t.Errorf("include = %v, want to contain reasoning.encrypted_content", body["include"])
	}
}

func TestApplyResponsesProviderOptions_UnknownKeysPassthrough(t *testing.T) {
	// Covers responses.go:249-251 (unknown keys passthrough).
	body := map[string]any{}
	applyResponsesProviderOptions(body, map[string]any{
		"custom_flag":    true,
		"another_option": "value",
	})
	if body["custom_flag"] != true {
		t.Errorf("custom_flag = %v, want true", body["custom_flag"])
	}
	if body["another_option"] != "value" {
		t.Errorf("another_option = %v, want %q", body["another_option"], "value")
	}
}

func TestParseResponsesResult_WithLogprobs(t *testing.T) {
	// Covers responses.go:821-825, 840, 877-879 (logprobs extraction + providerMetadata).
	body := `{
		"id": "resp-lp",
		"model": "o3",
		"status": "completed",
		"output": [{
			"type": "message",
			"content": [{
				"type": "output_text",
				"text": "hello",
				"logprobs": [{"token": "hello", "logprob": -0.5}]
			}]
		}],
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	result, err := parseResponsesResult([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello" {
		t.Errorf("Text = %q, want %q", result.Text, "hello")
	}
	pm := result.ProviderMetadata
	if pm == nil {
		t.Fatal("ProviderMetadata is nil")
	}
	openaiMeta := pm["openai"]
	if openaiMeta == nil {
		t.Fatal("ProviderMetadata[openai] is nil")
	}
	logprobs, ok := openaiMeta["logprobs"]
	if !ok || logprobs == nil {
		t.Error("logprobs not in provider metadata")
	}
}

func TestChat_EnvVarResolution(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	m := Chat("gpt-4o")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource should be set from OPENAI_API_KEY")
	}
}

func TestChat_EnvVarBaseURL(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	t.Setenv("OPENAI_BASE_URL", "https://custom.openai.com/v1")
	m := Chat("gpt-4o")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://custom.openai.com/v1" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestChat_EnvVarNotOverrideExplicit(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	t.Setenv("OPENAI_BASE_URL", "https://env.url")
	m := Chat("gpt-4o", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://explicit.url" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestBuildResponsesRequest_ProviderDefinedTool(t *testing.T) {
	params := provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "search the web"}}},
		},
		Tools: []provider.ToolDefinition{
			{
				Name:                "web_search",
				ProviderDefinedType: "web_search",
				ProviderDefinedOptions: map[string]any{
					"search_context_size": "high",
				},
			},
		},
	}

	body := buildResponsesRequest(params, "gpt-4o", false)

	tools, ok := body["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", body["tools"])
	}
	tool := tools[0]
	if tool["type"] != "web_search" {
		t.Errorf("tool type = %v, want web_search", tool["type"])
	}
	if tool["search_context_size"] != "high" {
		t.Errorf("search_context_size = %v", tool["search_context_size"])
	}
	// Should NOT have "function" or "name" keys (provider-defined tools have type only).
	if _, ok := tool["name"]; ok {
		t.Error("provider-defined tool should not have name in responses format")
	}
}

func TestStreamResponses_ContextCancel_AllBranches(t *testing.T) {
	// Exercise every TrySend early-return path in streamResponses.

	line := func(event, data string) string {
		return "event: " + event + "\ndata: " + data + "\n\n"
	}

	tests := []struct {
		name  string
		input string
	}{
		{
			// [DONE] ChunkFinish (line 431)
			name:  "done",
			input: "event: done\ndata: [DONE]\n\n",
		},
		{
			// response.output_text.delta (line 443)
			name:  "text_delta",
			input: line("response.output_text.delta", `{"delta":"hello"}`),
		},
		{
			// response.refusal.delta (line 453)
			name:  "refusal_delta",
			input: line("response.refusal.delta", `{"delta":"refused"}`),
		},
		{
			// response.reasoning_summary_text.delta (line 472)
			name:  "reasoning_delta",
			input: line("response.reasoning_summary_text.delta", `{"item_id":"r1","summary_index":0,"delta":"thinking"}`),
		},
		{
			// response.output_item.added function_call (line 505)
			name:  "function_call_start",
			input: line("response.output_item.added", `{"output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"fn"}}`),
		},
		{
			// response.completed step/finish (line 638)
			name:  "completed",
			input: line("response.completed", `{"response":{"id":"r1","model":"o3","usage":{"input_tokens":1,"output_tokens":1}}}`),
		},
		{
			// response.failed context_length_exceeded (line 667)
			name:  "failed_context_overflow",
			input: line("response.failed", `{"response":{"error":{"message":"too long","code":"context_length_exceeded"}}}`),
		},
		{
			// response.failed insufficient_quota (line 674)
			name:  "failed_quota",
			input: line("response.failed", `{"response":{"error":{"message":"quota","code":"insufficient_quota"}}}`),
		},
		{
			// response.failed usage_not_included (line 681)
			name:  "failed_usage",
			input: line("response.failed", `{"response":{"error":{"message":"no usage","code":"usage_not_included"}}}`),
		},
		{
			// response.failed invalid_prompt (line 688)
			name:  "failed_invalid_prompt",
			input: line("response.failed", `{"response":{"error":{"message":"bad prompt","code":"invalid_prompt"}}}`),
		},
		{
			// response.failed default (line 695)
			name:  "failed_default",
			input: line("response.failed", `{"response":{"error":{"message":"unknown","code":"some_code"}}}`),
		},
		{
			// error context_overflow (line 714)
			name:  "error_overflow",
			input: line("error", `{"message":"overflow","code":"context_overflow"}`),
		},
		{
			// error default (line 721)
			name:  "error_default",
			input: line("error", `{"message":"some error","code":"other"}`),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			out := make(chan provider.StreamChunk) // unbuffered
			done := make(chan struct{})
			go func() {
				streamResponses(ctx, io.NopCloser(strings.NewReader(tc.input)), out)
				close(done)
			}()
			<-done
			for range out {
			}
		})
	}

	// Nested: function_call_arguments.delta complete JSON (line 534)
	// requires function_call start TrySend (505) to succeed first.
	t.Run("function_call_args_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := line("response.output_item.added", `{"output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"fn"}}`) +
			line("response.function_call_arguments.delta", `{"output_index":0,"delta":"{}"}`)

		done := make(chan struct{})
		go func() {
			streamResponses(ctx, io.NopCloser(strings.NewReader(input)), out)
			close(done)
		}()

		<-out // function call start
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: function_call_arguments.done flush (line 553)
	t.Run("function_call_args_done_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := line("response.output_item.added", `{"output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"fn"}}`) +
			line("response.function_call_arguments.delta", `{"output_index":0,"delta":"{\"a\""}`) +
			line("response.function_call_arguments.done", `{"output_index":0}`)

		done := make(chan struct{})
		go func() {
			streamResponses(ctx, io.NopCloser(strings.NewReader(input)), out)
			close(done)
		}()

		<-out // function call start
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: response.completed flush remaining args (line 622)
	t.Run("completed_flush_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := line("response.output_item.added", `{"output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"fn"}}`) +
			line("response.function_call_arguments.delta", `{"output_index":0,"delta":"partial"}`) +
			line("response.completed", `{"response":{"id":"r1","model":"o3","usage":{"input_tokens":1,"output_tokens":1}}}`)

		done := make(chan struct{})
		go func() {
			streamResponses(ctx, io.NopCloser(strings.NewReader(input)), out)
			close(done)
		}()

		<-out // function call start
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: response.completed stepFinish (line 638) after text
	t.Run("completed_step_finish_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := line("response.output_text.delta", `{"delta":"hi"}`) +
			line("response.completed", `{"response":{"id":"r1","model":"o3","usage":{"input_tokens":1,"output_tokens":1}}}`)

		done := make(chan struct{})
		go func() {
			streamResponses(ctx, io.NopCloser(strings.NewReader(input)), out)
			close(done)
		}()

		<-out // text delta
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: response.completed ChunkFinish (line 644) -- stepFinish succeeds, ChunkFinish fails.
	t.Run("completed_chunk_finish_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := line("response.completed", `{"response":{"id":"r1","model":"o3","usage":{"input_tokens":1,"output_tokens":1}}}`)

		done := make(chan struct{})
		go func() {
			streamResponses(ctx, io.NopCloser(strings.NewReader(input)), out)
			close(done)
		}()

		<-out // step finish chunk
		cancel()
		<-done
		for range out {
		}
	})

	// Scanner error (line 736)
	t.Run("scanner_error_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		pr, pw := io.Pipe()
		go func() {
			_, _ = pw.Write([]byte("event: response.output_text.delta\ndata: {\"delta\":\"x\"}\n\n"))
			_ = pw.CloseWithError(fmt.Errorf("read error"))
		}()

		done := make(chan struct{})
		go func() {
			streamResponses(ctx, io.NopCloser(pr), out)
			close(done)
		}()

		<-out // text chunk
		cancel()
		<-done
		for range out {
		}
	})
}

// ---------------------------------------------------------------------------
// addIncludeKey
// ---------------------------------------------------------------------------

func TestAddIncludeKey_NilInclude(t *testing.T) {
	// When body has no "include" key, it should create a new slice.
	body := map[string]any{}
	addIncludeKey(body, "reasoning.encrypted_content")
	got, ok := body["include"].([]string)
	if !ok {
		t.Fatalf("include is %T, want []string", body["include"])
	}
	if len(got) != 1 || got[0] != "reasoning.encrypted_content" {
		t.Errorf("include = %v", got)
	}
}

func TestAddIncludeKey_ExistingStringSlice(t *testing.T) {
	// When body already has a []string include, it should append.
	body := map[string]any{
		"include": []string{"existing_key"},
	}
	addIncludeKey(body, "new_key")
	got := body["include"].([]string)
	if len(got) != 2 || got[0] != "existing_key" || got[1] != "new_key" {
		t.Errorf("include = %v", got)
	}
}

func TestAddIncludeKey_Duplicate(t *testing.T) {
	// Should not add duplicate keys.
	body := map[string]any{
		"include": []string{"reasoning.encrypted_content"},
	}
	addIncludeKey(body, "reasoning.encrypted_content")
	got := body["include"].([]string)
	if len(got) != 1 {
		t.Errorf("include = %v, want single entry (no duplicate)", got)
	}
}

func TestAddIncludeKey_AnySlice(t *testing.T) {
	// When body has []any (e.g., from JSON unmarshal), should convert string items.
	body := map[string]any{
		"include": []any{"existing_key"},
	}
	addIncludeKey(body, "new_key")
	got := body["include"].([]string)
	if len(got) != 2 || got[0] != "existing_key" || got[1] != "new_key" {
		t.Errorf("include = %v", got)
	}
}

func TestAddIncludeKey_AnySliceNonString(t *testing.T) {
	// Non-string items in []any should be skipped during conversion.
	body := map[string]any{
		"include": []any{42, "valid_key"},
	}
	addIncludeKey(body, "new_key")
	got := body["include"].([]string)
	if len(got) != 2 || got[0] != "valid_key" || got[1] != "new_key" {
		t.Errorf("include = %v", got)
	}
}

func TestAddIncludeKey_AnySliceDuplicate(t *testing.T) {
	// Duplicate detection should work with []any source.
	body := map[string]any{
		"include": []any{"reasoning.encrypted_content"},
	}
	addIncludeKey(body, "reasoning.encrypted_content")
	got := body["include"].([]string)
	if len(got) != 1 {
		t.Errorf("include = %v, want single entry", got)
	}
}
