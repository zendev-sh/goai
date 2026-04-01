package google

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
		if !strings.Contains(r.URL.Path, "gemini-2.5-flash") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if !strings.Contains(r.URL.RawQuery, "alt=sse") {
			t.Errorf("missing alt=sse query param")
		}
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("unexpected api key: %s", r.Header.Get("x-goog-api-key"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"modelVersion":"gemini-2.5-flash-preview-04-17","candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"modelVersion":"gemini-2.5-flash-preview-04-17","candidates":[{"content":{"parts":[{"text":" world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
	if len(chunks) < 4 {
		t.Fatalf("expected at least 4 chunks, got %d", len(chunks))
	}
	if chunks[0].Text != "Hello" {
		t.Errorf("chunks[0].Text = %q, want Hello", chunks[0].Text)
	}
	if chunks[1].Text != " world" {
		t.Errorf("chunks[1].Text = %q, want ' world'", chunks[1].Text)
	}
	// Last chunk should be finish with usage.
	finish := chunks[len(chunks)-1]
	if finish.Type != provider.ChunkFinish {
		t.Errorf("last chunk type = %s, want finish", finish.Type)
	}
	if finish.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", finish.Usage.InputTokens)
	}
	if finish.Usage.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", finish.Usage.OutputTokens)
	}
	if finish.Response.Model != "gemini-2.5-flash-preview-04-17" {
		t.Errorf("Response.Model = %q, want %q", finish.Response.Model, "gemini-2.5-flash-preview-04-17")
	}
}

func TestChat_Stream_ToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"read_file","args":{"path":"test.go"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read test.go"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotStart, gotCall bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkToolCallStreamStart {
			gotStart = true
			if chunk.ToolName != "read_file" {
				t.Errorf("ToolName = %q, want read_file", chunk.ToolName)
			}
			if chunk.ToolCallID != "call_read_file_0" {
				t.Errorf("ToolCallID = %q, want call_read_file_0", chunk.ToolCallID)
			}
		}
		if chunk.Type == provider.ChunkToolCall {
			gotCall = true
		}
	}
	if !gotStart {
		t.Error("expected tool_call_streaming_start")
	}
	if !gotCall {
		t.Error("expected tool_call")
	}
}

func TestChat_Stream_Reasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"Thinking...","thoughtSignature":"sig123"}]}}]}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"text":"Answer"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":8,"thoughtsTokenCount":3}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotReasoning, gotText bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkReasoning {
			gotReasoning = true
			if chunk.Text != "Thinking..." {
				t.Errorf("reasoning text = %q", chunk.Text)
			}
			if chunk.Metadata == nil || chunk.Metadata["google"] == nil {
				t.Error("expected thoughtSignature in metadata")
			}
		}
		if chunk.Type == provider.ChunkText {
			gotText = true
		}
		if chunk.Type == provider.ChunkFinish {
			// Thoughts should be separated from output.
			if chunk.Usage.ReasoningTokens != 3 {
				t.Errorf("ReasoningTokens = %d, want 3", chunk.Usage.ReasoningTokens)
			}
			if chunk.Usage.OutputTokens != 5 {
				t.Errorf("OutputTokens = %d, want 5 (8-3)", chunk.Usage.OutputTokens)
			}
		}
	}
	if !gotReasoning {
		t.Error("expected reasoning chunk")
	}
	if !gotText {
		t.Error("expected text chunk")
	}
}

func TestChat_Stream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"error":{"code":400,"message":"Bad request"}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
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

func TestChat_Stream_OverflowError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"error":{"code":400,"message":"input token count 200000 exceeds the maximum"}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	found := false
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkError {
			found = true
			var overflow *goai.ContextOverflowError
			if !errors.As(chunk.Error, &overflow) {
				t.Errorf("expected ContextOverflowError, got %T", chunk.Error)
			}
		}
	}
	if !found {
		t.Fatal("expected overflow error chunk")
	}
}

func TestChat_Stream_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"code":429,"message":"Rate limited"}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChat_Stream_CachedTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"cachedContentTokenCount":80}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	found := false
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			found = true
			if chunk.Usage.InputTokens != 20 {
				t.Errorf("InputTokens = %d, want 20 (100-80)", chunk.Usage.InputTokens)
			}
			if chunk.Usage.CacheReadTokens != 80 {
				t.Errorf("CacheReadTokens = %d, want 80", chunk.Usage.CacheReadTokens)
			}
		}
	}
	if !found {
		t.Fatal("expected finish chunk with usage data")
	}
}

func TestChat_Stream_ContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"text":"start"}]}}]}`+"\n\n")
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(t.Context())
	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(ctx, provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	for range result.Stream {
	}
}

func TestChat_Stream_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: not json\n\n")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var gotText bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText && chunk.Text == "ok" {
			gotText = true
		}
	}
	if !gotText {
		t.Error("expected text chunk despite invalid JSON line")
	}
}

func TestChat_Stream_EmptyCandidates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":0}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var textChunks int
	var stepFinishReason provider.FinishReason
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText {
			textChunks++
		}
		if chunk.Type == provider.ChunkStepFinish {
			stepFinishReason = chunk.FinishReason
		}
	}
	// The first SSE event has empty candidates - no text should be emitted for it.
	// The second event has candidates with text and STOP, so we expect text from it
	// and a step-finish with reason "stop".
	if textChunks == 0 {
		t.Error("expected at least one text chunk from the non-empty candidate event")
	}
	if stepFinishReason != provider.FinishStop {
		t.Errorf("step finish reason = %q, want %q", stepFinishReason, provider.FinishStop)
	}
}

func TestChat_Stream_ScannerError(t *testing.T) {
	out := make(chan provider.StreamChunk, 64)
	longLine := "data: " + strings.Repeat("x", 2*1024*1024) + "\n"
	go parseSSE(t.Context(), strings.NewReader(longLine), out)

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

// --- Non-streaming tests ---

func TestChat_Generate_TextResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.RawQuery, "alt=sse") {
			t.Error("non-streaming should not have alt=sse")
		}

		body, _ := io.ReadAll(r.Body)
		var req geminiRequestBody
		_ = json.Unmarshal(body, &req)

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"modelVersion":"gemini-2.5-flash-preview-04-17",
			"candidates":[{"content":{"parts":[{"text":"Hello world"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}
		}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want Hello world", result.Text)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %s, want stop", result.FinishReason)
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", result.Usage.InputTokens)
	}
	if result.Response.Model != "gemini-2.5-flash-preview-04-17" {
		t.Errorf("Response.Model = %q, want %q", result.Response.Model, "gemini-2.5-flash-preview-04-17")
	}
}

func TestChat_Generate_ToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"candidates":[{"content":{"parts":[{"functionCall":{"name":"read_file","args":{"path":"test.go"}}}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}
		}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "read"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %s, want tool_calls", result.FinishReason)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "read_file" {
		t.Errorf("ToolCall.Name = %q, want read_file", result.ToolCalls[0].Name)
	}
}

func TestChat_Generate_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"code":400,"message":"Bad request"}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChat_Generate_OverflowError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"error":{"code":400,"message":"input token count 200000 exceeds the maximum"}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var overflow *goai.ContextOverflowError
	if !errors.As(err, &overflow) {
		t.Errorf("expected ContextOverflowError, got %T: %v", err, err)
	}
}

func TestChat_Generate_EmptyCandidates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":0}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "" {
		t.Errorf("Text = %q, want empty", result.Text)
	}
}

func TestChat_Generate_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, "not json")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- Request building tests ---

func TestBuildRequest_System(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		System:   "Be helpful.",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})

	if body.SystemInstruction == nil {
		t.Fatal("SystemInstruction should be set")
	}
}

func TestBuildRequest_Tools(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Tools: []provider.ToolDefinition{
			{Name: "read_file", Description: "Read a file", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	})

	if body.Tools == nil {
		t.Fatal("Tools should be set")
	}
}

func TestBuildRequest_ThinkingLevel(t *testing.T) {
	// gemini-3 model should get thinkingLevel=high.
	m := &chatModel{id: "gemini-3-pro", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})

	config, ok := body.GenerationConfig.(map[string]any)
	if !ok {
		t.Fatal("GenerationConfig should be a map")
	}
	tc, ok := config["thinkingConfig"].(map[string]any)
	if !ok {
		t.Fatal("thinkingConfig should be a map")
	}
	if tc["thinkingLevel"] != "high" {
		t.Errorf("thinkingLevel = %v, want high", tc["thinkingLevel"])
	}
}

func TestBuildRequest_TemperatureAndStopSequences(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	temp := 0.7
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages:      []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		Temperature:   &temp,
		StopSequences: []string{"END"},
	})

	config := body.GenerationConfig.(map[string]any)
	if config["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", config["temperature"])
	}
	stops, ok := config["stopSequences"].([]string)
	if !ok || stops[0] != "END" {
		t.Errorf("stopSequences = %v, want [END]", config["stopSequences"])
	}
}

// --- Message conversion tests ---

func TestConvertMessages_RoleMapping(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		{Role: provider.RoleAssistant, Content: []provider.Part{{Type: provider.PartText, Text: "hello"}}},
		{Role: provider.RoleTool, Content: []provider.Part{{Type: provider.PartToolResult, ToolName: "test", ToolOutput: `"ok"`}}},
	})

	if msgs[0]["role"] != "user" {
		t.Errorf("user role = %v", msgs[0]["role"])
	}
	if msgs[1]["role"] != "model" {
		t.Errorf("assistant role = %v, want model", msgs[1]["role"])
	}
	if msgs[2]["role"] != "user" {
		t.Errorf("tool role = %v, want user", msgs[2]["role"])
	}
}

func TestConvertMessages_ToolCallAndResult(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolName: "read", ToolInput: json.RawMessage(`{"path":"a.go"}`)},
		}},
		{Role: provider.RoleTool, Content: []provider.Part{
			{Type: provider.PartToolResult, ToolName: "read", ToolOutput: `{"content":"hello"}`},
		}},
	})

	// Check function call.
	parts := msgs[0]["parts"].([]map[string]any)
	fc, ok := parts[0]["functionCall"].(map[string]any)
	if !ok {
		t.Fatal("expected functionCall")
	}
	if fc["name"] != "read" {
		t.Errorf("functionCall.name = %v, want read", fc["name"])
	}

	// Check function response (already an object -- should not be wrapped).
	parts = msgs[1]["parts"].([]map[string]any)
	fr, ok := parts[0]["functionResponse"].(map[string]any)
	if !ok {
		t.Fatal("expected functionResponse")
	}
	resp, ok := fr["response"].(map[string]any)
	if !ok {
		t.Fatal("expected response to be a map")
	}
	if resp["content"] != "hello" {
		t.Errorf("response.content = %v, want hello", resp["content"])
	}
}

func TestConvertMessages_ToolCallNilInput(t *testing.T) {
	// ToolInput is nil - args should default to {} (not nil/null).
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolName: "noop", ToolInput: nil},
		}},
	})

	parts := msgs[0]["parts"].([]map[string]any)
	fc, ok := parts[0]["functionCall"].(map[string]any)
	if !ok {
		t.Fatal("expected functionCall")
	}
	args, ok := fc["args"].(map[string]any)
	if !ok {
		t.Fatalf("expected args to be map[string]any, got %T", fc["args"])
	}
	if len(args) != 0 {
		t.Errorf("expected empty args, got %v", args)
	}
}

func TestConvertMessages_ToolResultNonObject(t *testing.T) {
	// String tool output should be wrapped in {result: ...}.
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleTool, Content: []provider.Part{
			{Type: provider.PartToolResult, ToolName: "test", ToolOutput: `"just a string"`},
		}},
	})

	parts := msgs[0]["parts"].([]map[string]any)
	fr := parts[0]["functionResponse"].(map[string]any)
	resp, ok := fr["response"].(map[string]any)
	if !ok {
		t.Fatal("expected wrapped response")
	}
	if resp["result"] != "just a string" {
		t.Errorf("response.result = %v, want 'just a string'", resp["result"])
	}
}

func TestConvertMessages_ToolResultInvalidJSON(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleTool, Content: []provider.Part{
			{Type: provider.PartToolResult, ToolName: "test", ToolOutput: "not json"},
		}},
	})

	parts := msgs[0]["parts"].([]map[string]any)
	fr := parts[0]["functionResponse"].(map[string]any)
	resp, ok := fr["response"].(map[string]any)
	if !ok {
		t.Fatal("expected wrapped response")
	}
	if resp["result"] != "not json" {
		t.Errorf("response.result = %v, want 'not json'", resp["result"])
	}
}

func TestConvertMessages_Image(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{
			{Type: provider.PartImage, URL: "data:image/png;base64,abc123"},
		}},
	})

	parts := msgs[0]["parts"].([]map[string]any)
	inline, ok := parts[0]["inlineData"].(map[string]any)
	if !ok {
		t.Fatal("expected inlineData")
	}
	if inline["mimeType"] != "image/png" {
		t.Errorf("mimeType = %v, want image/png", inline["mimeType"])
	}
}

func TestConvertMessages_ThoughtSignature(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartText, Text: "hello", ProviderOptions: map[string]any{
				"google": map[string]any{"thoughtSignature": "sig1"},
			}},
		}},
	})

	parts := msgs[0]["parts"].([]map[string]any)
	if parts[0]["thoughtSignature"] != "sig1" {
		t.Errorf("thoughtSignature = %v, want sig1", parts[0]["thoughtSignature"])
	}
}

func TestConvertMessages_SkipsSystem(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleSystem, Content: []provider.Part{{Type: provider.PartText, Text: "system"}}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
	})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
}

func TestConvertMessages_SkipsEmptyParts(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleUser, Content: []provider.Part{}},
		{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
	})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message (empty parts skipped), got %d", len(msgs))
	}
}

// --- SanitizeGeminiSchema tests ---

func TestSanitizeGeminiSchema_EnumToString(t *testing.T) {
	schema := map[string]any{
		"type": "integer",
		"enum": []any{1, 2, 3},
	}
	result := sanitizeGeminiSchema(schema)
	if result["type"] != "string" {
		t.Errorf("type = %v, want string", result["type"])
	}
	enum := result["enum"].([]any)
	if enum[0] != "1" {
		t.Errorf("enum[0] = %v, want '1'", enum[0])
	}
}

func TestSanitizeGeminiSchema_FilterRequired(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"name": map[string]any{"type": "string"}},
		"required":   []any{"name", "nonexistent"},
	}
	result := sanitizeGeminiSchema(schema)
	required := result["required"].([]any)
	if len(required) != 1 || required[0] != "name" {
		t.Errorf("required = %v, want [name]", required)
	}
}

func TestSanitizeGeminiSchema_ArrayDefaultType(t *testing.T) {
	schema := map[string]any{
		"type":  "array",
		"items": map[string]any{},
	}
	result := sanitizeGeminiSchema(schema)
	items := result["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("items.type = %v, want string", items["type"])
	}
}

func TestSanitizeGeminiSchema_ArrayNilItems(t *testing.T) {
	schema := map[string]any{
		"type": "array",
	}
	result := sanitizeGeminiSchema(schema)
	items := result["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("items.type = %v, want string", items["type"])
	}
}

func TestSanitizeGeminiSchema_RemovePropertiesFromNonObject(t *testing.T) {
	schema := map[string]any{
		"type":       "string",
		"properties": map[string]any{"x": "y"},
		"required":   []any{"x"},
	}
	result := sanitizeGeminiSchema(schema)
	if result["properties"] != nil {
		t.Error("properties should be removed from non-object type")
	}
	if result["required"] != nil {
		t.Error("required should be removed from non-object type")
	}
}

func TestSanitizeGeminiSchema_Nested(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{
				"type": "integer",
				"enum": []any{200, 404},
			},
		},
		"required": []any{"status"},
	}
	result := sanitizeGeminiSchema(schema)
	props := result["properties"].(map[string]any)
	status := props["status"].(map[string]any)
	if status["type"] != "string" {
		t.Errorf("nested enum type = %v, want string", status["type"])
	}
}

func TestSanitizeGeminiSchema_Nil(t *testing.T) {
	result := sanitizeImpl(nil)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestSanitizeGeminiSchema_Scalar(t *testing.T) {
	result := sanitizeImpl("hello")
	if result != "hello" {
		t.Errorf("expected hello, got %v", result)
	}
}

func TestSanitizeGeminiSchema_StripAdditionalProperties(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":                 "string",
				"additionalProperties": false,
			},
		},
		"additionalProperties": false,
	}
	result := sanitizeGeminiSchema(schema)
	if _, ok := result["additionalProperties"]; ok {
		t.Error("expected additionalProperties to be stripped at root")
	}
	props := result["properties"].(map[string]any)
	nameProp := props["name"].(map[string]any)
	if _, ok := nameProp["additionalProperties"]; ok {
		t.Error("expected additionalProperties to be stripped in nested property")
	}
}

// --- Option tests ---

func TestWithHTTPClient(t *testing.T) {
	customClient := &http.Client{}
	model := Chat("gemini-2.5-flash", WithAPIKey("key"), WithHTTPClient(customClient))
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
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash",
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithHeaders(map[string]string{"X-Custom": "value"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestNoTokenSource(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	model := Chat("gemini-2.5-flash")
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestCapabilities(t *testing.T) {
	model := Chat("gemini-2.5-flash", WithAPIKey("key"))
	caps := provider.ModelCapabilitiesOf(model)
	if !caps.Temperature || !caps.Reasoning || !caps.ToolCall || !caps.Attachment {
		t.Error("unexpected capabilities")
	}
	if caps.InputModalities.Video {
		t.Error("expected Video input modality to be false: no PartVideo type exists")
	}
	if !caps.InputModalities.PDF {
		t.Error("expected PDF input modality")
	}

	// Non-thinking model should not advertise Reasoning.
	model2 := Chat("gemini-1.5-flash", WithAPIKey("key"))
	caps2 := provider.ModelCapabilitiesOf(model2)
	if caps2.Reasoning {
		t.Error("gemini-1.5-flash should have Reasoning=false")
	}
}

func TestModelID(t *testing.T) {
	model := Chat("gemini-2.5-flash", WithAPIKey("key"))
	if model.ModelID() != "gemini-2.5-flash" {
		t.Errorf("ModelID() = %q", model.ModelID())
	}
}

func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		input string
		want  provider.FinishReason
	}{
		{"STOP", provider.FinishStop},
		{"MAX_TOKENS", provider.FinishLength},
		{"SAFETY", provider.FinishContentFilter},
		{"OTHER", provider.FinishOther},
	}
	for _, tt := range tests {
		if got := mapFinishReason(tt.input); got != tt.want {
			t.Errorf("mapFinishReason(%q) = %s, want %s", tt.input, got, tt.want)
		}
	}
}

func TestParseDataURL(t *testing.T) {
	tests := []struct {
		url  string
		ok   bool
		mime string
	}{
		{"data:image/png;base64,abc", true, "image/png"},
		{"https://example.com", false, ""},
		{"data:missing;abc", false, ""},
	}
	for _, tt := range tests {
		mime, _, ok := httpc.ParseDataURL(tt.url)
		if ok != tt.ok || mime != tt.mime {
			t.Errorf("ParseDataURL(%q) ok=%v mime=%q", tt.url, ok, mime)
		}
	}
}

func TestWithTokenSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-goog-api-key") != "dynamic-token" {
			t.Errorf("key = %q, want dynamic-token", r.Header.Get("x-goog-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
	}))
	defer server.Close()

	ts := provider.StaticToken("dynamic-token")
	model := Chat("gemini-2.5-flash", WithTokenSource(ts), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestBuildRequest_TopP(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	topP := 0.9
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		TopP:     &topP,
	})
	config := body.GenerationConfig.(map[string]any)
	if config["topP"] != 0.9 {
		t.Errorf("topP = %v, want 0.9", config["topP"])
	}
}

func TestBuildRequest_MaxOutputTokens(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages:        []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		MaxOutputTokens: 4096,
	})
	config := body.GenerationConfig.(map[string]any)
	if config["maxOutputTokens"] != 4096 {
		t.Errorf("maxOutputTokens = %v, want 4096", config["maxOutputTokens"])
	}
}

func TestChat_Generate_ErrorInJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"error":{"code":400,"message":"Bad request"}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
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
		t.Errorf("expected APIError, got %T", err)
	}
}

func TestDoGenerate_ReadError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "100")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, `{"id":"x"`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDoHTTP_ConnectionError(t *testing.T) {
	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL("http://127.0.0.1:1"))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected connection error")
	}
}

func TestChat_Generate_NegativeOutputTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// ThoughtsTokenCount > CandidatesTokenCount would give negative output.
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3,"thoughtsTokenCount":5}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.OutputTokens != 0 {
		t.Errorf("OutputTokens = %d, want 0 (clamped)", result.Usage.OutputTokens)
	}
}

func TestChat_Generate_NegativeInputTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"cachedContentTokenCount":10}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.InputTokens != 0 {
		t.Errorf("InputTokens = %d, want 0 (clamped)", result.Usage.InputTokens)
	}
}

func TestChat_Generate_NoFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Default should be stop.
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %s, want stop", result.FinishReason)
	}
}

func TestChat_Stream_NegativeTokens(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"thoughtsTokenCount":5,"cachedContentTokenCount":10}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkFinish {
			if chunk.Usage.InputTokens != 0 {
				t.Errorf("InputTokens = %d, want 0", chunk.Usage.InputTokens)
			}
			if chunk.Usage.OutputTokens != 0 {
				t.Errorf("OutputTokens = %d, want 0", chunk.Usage.OutputTokens)
			}
		}
	}
}

func TestConvertMessages_ReasoningWithSignature(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartReasoning, Text: "thinking...", ProviderOptions: map[string]any{
				"google": map[string]any{"thoughtSignature": "sig2"},
			}},
		}},
	})
	parts := msgs[0]["parts"].([]map[string]any)
	if parts[0]["thoughtSignature"] != "sig2" {
		t.Errorf("thoughtSignature = %v, want sig2", parts[0]["thoughtSignature"])
	}
}

func TestConvertMessages_ToolCallWithSignature(t *testing.T) {
	msgs := convertMessages([]provider.Message{
		{Role: provider.RoleAssistant, Content: []provider.Part{
			{Type: provider.PartToolCall, ToolName: "test", ToolInput: json.RawMessage(`{}`), ProviderOptions: map[string]any{
				"google": map[string]any{"thoughtSignature": "sig3"},
			}},
		}},
	})
	parts := msgs[0]["parts"].([]map[string]any)
	if parts[0]["thoughtSignature"] != "sig3" {
		t.Errorf("thoughtSignature = %v, want sig3", parts[0]["thoughtSignature"])
	}
}

func TestBuildRequest_ResponseFormat(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "test_schema",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
	})

	config, ok := body.GenerationConfig.(map[string]any)
	if !ok {
		t.Fatal("GenerationConfig should be a map")
	}
	if config["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType = %v, want application/json", config["responseMimeType"])
	}
	schema, ok := config["responseSchema"].(map[string]any)
	if !ok {
		t.Fatalf("responseSchema not a map: %T", config["responseSchema"])
	}
	if schema["type"] != "object" {
		t.Errorf("schema.type = %v, want object", schema["type"])
	}
}

func TestBuildRequest_ResponseFormat_InvalidSchema(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	_, err := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name:   "broken",
			Schema: json.RawMessage(`not valid json`),
		},
	})
	if err == nil {
		t.Fatal("expected error for invalid schema JSON")
	}
	if !strings.Contains(err.Error(), "invalid response schema") {
		t.Errorf("error = %v, want 'invalid response schema'", err)
	}
}

// --- BF.15: ProviderOptions tests ---

func TestBuildRequest_ThinkingConfigFromProviderOptions(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"thinkingConfig": map[string]any{
					"includeThoughts": true,
					"thinkingBudget":  2048,
				},
			},
		},
	})

	config := body.GenerationConfig.(map[string]any)
	tc, ok := config["thinkingConfig"].(map[string]any)
	if !ok {
		t.Fatal("thinkingConfig should be a map")
	}
	if tc["thinkingBudget"] != 2048 {
		t.Errorf("thinkingBudget = %v, want 2048", tc["thinkingBudget"])
	}
	// Should NOT have thinkingLevel since it was not in ProviderOptions.
	if _, ok := tc["thinkingLevel"]; ok {
		t.Error("thinkingLevel should not be set when overridden by ProviderOptions")
	}
}

func TestBuildRequest_ThinkingConfigDefault(t *testing.T) {
	// Without ProviderOptions, gemini-3 should still get thinkingLevel=high.
	m := &chatModel{id: "gemini-3-pro", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})

	config := body.GenerationConfig.(map[string]any)
	tc := config["thinkingConfig"].(map[string]any)
	if tc["thinkingLevel"] != "high" {
		t.Errorf("thinkingLevel = %v, want high", tc["thinkingLevel"])
	}
}

func TestBuildRequest_SafetySettings(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	safetySettings := []map[string]any{
		{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
		{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
	}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"safetySettings": safetySettings,
			},
		},
	})

	ss, ok := body.SafetySettings.([]map[string]any)
	if !ok {
		t.Fatalf("SafetySettings type = %T, want []map[string]any", body.SafetySettings)
	}
	if len(ss) != 2 {
		t.Fatalf("SafetySettings len = %d, want 2", len(ss))
	}
	if ss[0]["category"] != "HARM_CATEGORY_HATE_SPEECH" {
		t.Errorf("ss[0].category = %v", ss[0]["category"])
	}
	if ss[1]["threshold"] != "BLOCK_NONE" {
		t.Errorf("ss[1].threshold = %v", ss[1]["threshold"])
	}
}

func TestBuildRequest_ResponseModalities(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash-image-preview", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "draw a banana"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"responseModalities": []string{"TEXT", "IMAGE"},
			},
		},
	})

	config := body.GenerationConfig.(map[string]any)
	rm, ok := config["responseModalities"].([]string)
	if !ok {
		t.Fatalf("responseModalities type = %T, want []string", config["responseModalities"])
	}
	if len(rm) != 2 || rm[0] != "TEXT" || rm[1] != "IMAGE" {
		t.Errorf("responseModalities = %v, want [TEXT IMAGE]", rm)
	}
}

func TestBuildRequest_CachedContent(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"cachedContent": "cachedContents/abc123",
			},
		},
	})

	if body.CachedContent != "cachedContents/abc123" {
		t.Errorf("CachedContent = %v, want cachedContents/abc123", body.CachedContent)
	}
}

func TestBuildRequest_MediaResolution(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"mediaResolution": "MEDIA_RESOLUTION_HIGH",
			},
		},
	})

	config := body.GenerationConfig.(map[string]any)
	if config["mediaResolution"] != "MEDIA_RESOLUTION_HIGH" {
		t.Errorf("mediaResolution = %v, want MEDIA_RESOLUTION_HIGH", config["mediaResolution"])
	}
}

func TestBuildRequest_ImageConfig(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"imageConfig": map[string]any{
					"aspectRatio": "16:9",
					"imageSize":   "2K",
				},
			},
		},
	})

	config := body.GenerationConfig.(map[string]any)
	ic, ok := config["imageConfig"].(map[string]any)
	if !ok {
		t.Fatalf("imageConfig type = %T, want map", config["imageConfig"])
	}
	if ic["aspectRatio"] != "16:9" {
		t.Errorf("aspectRatio = %v, want 16:9", ic["aspectRatio"])
	}
	if ic["imageSize"] != "2K" {
		t.Errorf("imageSize = %v, want 2K", ic["imageSize"])
	}
}

func TestBuildRequest_RetrievalConfig(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"retrievalConfig": map[string]any{
					"latLng": map[string]any{
						"latitude":  37.7749,
						"longitude": -122.4194,
					},
				},
			},
		},
	})

	tc, ok := body.ToolConfig.(map[string]any)
	if !ok {
		t.Fatalf("ToolConfig type = %T, want map", body.ToolConfig)
	}
	rc, ok := tc["retrievalConfig"].(map[string]any)
	if !ok {
		t.Fatalf("retrievalConfig type = %T", tc["retrievalConfig"])
	}
	latLng, ok := rc["latLng"].(map[string]any)
	if !ok {
		t.Fatalf("latLng type = %T", rc["latLng"])
	}
	if latLng["latitude"] != 37.7749 {
		t.Errorf("latitude = %v, want 37.7749", latLng["latitude"])
	}
}

func TestBuildRequest_Labels(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"labels": map[string]any{
					"team":    "backend",
					"project": "myapp",
				},
			},
		},
	})

	labels, ok := body.Labels.(map[string]any)
	if !ok {
		t.Fatalf("Labels type = %T, want map", body.Labels)
	}
	if labels["team"] != "backend" {
		t.Errorf("labels.team = %v, want backend", labels["team"])
	}
	if labels["project"] != "myapp" {
		t.Errorf("labels.project = %v, want myapp", labels["project"])
	}
}

func TestBuildRequest_AudioTimestamp(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"audioTimestamp": true,
			},
		},
	})

	config := body.GenerationConfig.(map[string]any)
	if config["audioTimestamp"] != true {
		t.Errorf("audioTimestamp = %v, want true", config["audioTimestamp"])
	}
}

func TestBuildRequest_AudioTimestampFalse(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"audioTimestamp": false,
			},
		},
	})

	config := body.GenerationConfig.(map[string]any)
	if _, ok := config["audioTimestamp"]; ok {
		t.Error("audioTimestamp should not be set when false")
	}
}

func TestBuildRequest_ToolChoiceAuto(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "auto",
		Tools:      []provider.ToolDefinition{{Name: "test", InputSchema: json.RawMessage(`{}`)}},
	})

	tc, ok := body.ToolConfig.(map[string]any)
	if !ok {
		t.Fatalf("ToolConfig type = %T, want map", body.ToolConfig)
	}
	fcc, ok := tc["functionCallingConfig"].(map[string]any)
	if !ok {
		t.Fatalf("functionCallingConfig type = %T", tc["functionCallingConfig"])
	}
	if fcc["mode"] != "AUTO" {
		t.Errorf("mode = %v, want AUTO", fcc["mode"])
	}
}

func TestBuildRequest_ToolChoiceNone(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "none",
	})

	tc := body.ToolConfig.(map[string]any)
	fcc := tc["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "NONE" {
		t.Errorf("mode = %v, want NONE", fcc["mode"])
	}
}

func TestBuildRequest_ToolChoiceRequired(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "required",
	})

	tc := body.ToolConfig.(map[string]any)
	fcc := tc["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("mode = %v, want ANY", fcc["mode"])
	}
}

func TestBuildRequest_ToolChoiceSpecificTool(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "read_file",
		Tools:      []provider.ToolDefinition{{Name: "read_file", InputSchema: json.RawMessage(`{}`)}},
	})

	tc := body.ToolConfig.(map[string]any)
	fcc := tc["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("mode = %v, want ANY", fcc["mode"])
	}
	allowed, ok := fcc["allowedFunctionNames"].([]string)
	if !ok || len(allowed) != 1 || allowed[0] != "read_file" {
		t.Errorf("allowedFunctionNames = %v, want [read_file]", fcc["allowedFunctionNames"])
	}
}

func TestBuildRequest_ToolChoiceAndRetrievalConfig(t *testing.T) {
	// Both tool_choice and retrievalConfig should coexist in toolConfig.
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "auto",
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"retrievalConfig": map[string]any{
					"latLng": map[string]any{"latitude": 0.0, "longitude": 0.0},
				},
			},
		},
	})

	tc := body.ToolConfig.(map[string]any)
	if _, ok := tc["functionCallingConfig"]; !ok {
		t.Error("expected functionCallingConfig in toolConfig")
	}
	if _, ok := tc["retrievalConfig"]; !ok {
		t.Error("expected retrievalConfig in toolConfig")
	}
}

func TestBuildRequest_NoToolChoice(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})

	if body.ToolConfig != nil {
		t.Errorf("ToolConfig should be nil when no ToolChoice set, got %v", body.ToolConfig)
	}
}

func TestBuildRequest_NoProviderOptions(t *testing.T) {
	// Verify that buildRequest works fine with nil ProviderOptions.
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
	})

	if body.SafetySettings != nil {
		t.Error("SafetySettings should be nil")
	}
	if body.CachedContent != nil {
		t.Error("CachedContent should be nil")
	}
	if body.Labels != nil {
		t.Error("Labels should be nil")
	}
	config := body.GenerationConfig.(map[string]any)
	if _, ok := config["responseModalities"]; ok {
		t.Error("responseModalities should not be set")
	}
	if _, ok := config["mediaResolution"]; ok {
		t.Error("mediaResolution should not be set")
	}
}

func TestChat_Generate_ThoughtSignaturePreserved(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"candidates":[{"content":{"parts":[
				{"thought":true,"text":"thinking...","thoughtSignature":"sig_abc"},
				{"text":"Answer"}
			]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":8,"thoughtsTokenCount":3}
		}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "think"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "Answer" {
		t.Errorf("Text = %q, want Answer", result.Text)
	}

	// Check provider metadata contains thoughtSignature.
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata should not be nil")
	}
	gm, ok := result.ProviderMetadata["google"]
	if !ok {
		t.Fatal("expected google key in ProviderMetadata")
	}
	if gm["thoughtSignature"] != "sig_abc" {
		t.Errorf("thoughtSignature = %v, want sig_abc", gm["thoughtSignature"])
	}
}

func TestChat_Generate_NoThoughtSignature(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}
		}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderMetadata != nil {
		t.Errorf("ProviderMetadata should be nil when no thoughtSignature, got %v", result.ProviderMetadata)
	}
}

func TestChat_Generate_ProviderOptionsPassthrough(t *testing.T) {
	// Verify that provider options actually appear in the request body.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)

		// Check safety settings.
		if req["safetySettings"] == nil {
			t.Error("expected safetySettings in request body")
		}

		// Check cachedContent.
		if req["cachedContent"] != "cachedContents/xyz" {
			t.Errorf("cachedContent = %v, want cachedContents/xyz", req["cachedContent"])
		}

		// Check labels.
		labels, ok := req["labels"].(map[string]any)
		if !ok || labels["env"] != "test" {
			t.Errorf("labels = %v, want {env: test}", req["labels"])
		}

		// Check generationConfig fields.
		gc, ok := req["generationConfig"].(map[string]any)
		if !ok {
			t.Fatal("generationConfig missing")
		}
		if gc["mediaResolution"] != "MEDIA_RESOLUTION_LOW" {
			t.Errorf("mediaResolution = %v, want MEDIA_RESOLUTION_LOW", gc["mediaResolution"])
		}
		if gc["audioTimestamp"] != true {
			t.Errorf("audioTimestamp = %v, want true", gc["audioTimestamp"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"safetySettings": []map[string]any{
					{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
				},
				"cachedContent":   "cachedContents/xyz",
				"labels":          map[string]any{"env": "test"},
				"mediaResolution": "MEDIA_RESOLUTION_LOW",
				"audioTimestamp":  true,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestBuildRequest_AllProviderOptions(t *testing.T) {
	// Comprehensive test: set every ProviderOption and verify the request body.
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages:   []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ToolChoice: "required",
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"thinkingConfig": map[string]any{
					"includeThoughts": false,
					"thinkingLevel":   "minimal",
				},
				"safetySettings": []map[string]any{
					{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
				},
				"responseModalities": []string{"TEXT"},
				"cachedContent":      "cachedContents/test",
				"mediaResolution":    "MEDIA_RESOLUTION_MEDIUM",
				"imageConfig":        map[string]any{"aspectRatio": "1:1"},
				"retrievalConfig":    map[string]any{"latLng": map[string]any{"latitude": 0.0, "longitude": 0.0}},
				"labels":             map[string]any{"a": "b"},
				"audioTimestamp":     true,
			},
		},
	})

	// Verify all fields set correctly.
	config := body.GenerationConfig.(map[string]any)
	tc := config["thinkingConfig"].(map[string]any)
	if tc["thinkingLevel"] != "minimal" {
		t.Errorf("thinkingLevel = %v, want minimal", tc["thinkingLevel"])
	}
	if body.SafetySettings == nil {
		t.Error("SafetySettings should be set")
	}
	if config["responseModalities"] == nil {
		t.Error("responseModalities should be set")
	}
	if body.CachedContent != "cachedContents/test" {
		t.Errorf("CachedContent = %v", body.CachedContent)
	}
	if config["mediaResolution"] != "MEDIA_RESOLUTION_MEDIUM" {
		t.Errorf("mediaResolution = %v", config["mediaResolution"])
	}
	if config["imageConfig"] == nil {
		t.Error("imageConfig should be set")
	}
	if config["audioTimestamp"] != true {
		t.Error("audioTimestamp should be true")
	}
	if body.Labels == nil {
		t.Error("Labels should be set")
	}
	toolCfg := body.ToolConfig.(map[string]any)
	if toolCfg["functionCallingConfig"] == nil {
		t.Error("functionCallingConfig should be set")
	}
	if toolCfg["retrievalConfig"] == nil {
		t.Error("retrievalConfig should be set")
	}
}

func TestBuildRequest_ThinkingDisabled(t *testing.T) {
	// When thinkingConfig is set to false (bool), thinking should be disabled
	// and no default thinkingConfig should be injected.
	model := &chatModel{id: "gemini-3-flash"}
	body, _ := model.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"thinkingConfig": false,
			},
		},
	})

	config := body.GenerationConfig.(map[string]any)
	if _, hasTC := config["thinkingConfig"]; hasTC {
		t.Error("thinkingConfig should NOT be present when disabled via false")
	}
}

// --- Grounding metadata / citation tests ---

func TestChat_Generate_GroundingMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"modelVersion": "gemini-2.5-flash",
			"candidates": [{
				"content": {
					"parts": [{"text": "Grounded answer."}]
				},
				"finishReason": "STOP",
				"groundingMetadata": {
					"groundingChunks": [
						{"web": {"uri": "https://example.com/1", "title": "Source 1"}},
						{"web": {"uri": "https://example.com/2", "title": "Source 2"}},
						{"retrievedContext": {"uri": "gs://bucket/doc.pdf", "title": "Internal Doc"}}
					]
				}
			}],
			"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
		}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Sources) != 3 {
		t.Fatalf("len(Sources) = %d, want 3", len(result.Sources))
	}

	// Web sources.
	if result.Sources[0].Type != "url" || result.Sources[0].URL != "https://example.com/1" {
		t.Errorf("Sources[0] = %+v", result.Sources[0])
	}
	if result.Sources[0].Title != "Source 1" {
		t.Errorf("Sources[0].Title = %q", result.Sources[0].Title)
	}
	if result.Sources[0].ID != "grounding_0" {
		t.Errorf("Sources[0].ID = %q", result.Sources[0].ID)
	}
	if result.Sources[1].Type != "url" || result.Sources[1].URL != "https://example.com/2" {
		t.Errorf("Sources[1] = %+v", result.Sources[1])
	}

	// Retrieved context source.
	if result.Sources[2].Type != "document" || result.Sources[2].URL != "gs://bucket/doc.pdf" {
		t.Errorf("Sources[2] = %+v", result.Sources[2])
	}
	if result.Sources[2].Title != "Internal Doc" {
		t.Errorf("Sources[2].Title = %q", result.Sources[2].Title)
	}
}

func TestChat_Stream_GroundingMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"modelVersion":"gemini-2.5-flash","candidates":[{"content":{"parts":[{"text":"Grounded answer."}]},"finishReason":"STOP","groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com","title":"Example"}}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}`+"\n\n")
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var foundSource bool
	for chunk := range result.Stream {
		if chunk.Type == provider.ChunkText && chunk.Metadata != nil {
			if src, ok := chunk.Metadata["source"].(provider.Source); ok {
				foundSource = true
				if src.Type != "url" {
					t.Errorf("source.Type = %q, want url", src.Type)
				}
				if src.URL != "https://example.com" {
					t.Errorf("source.URL = %q", src.URL)
				}
				if src.Title != "Example" {
					t.Errorf("source.Title = %q", src.Title)
				}
			}
		}
	}
	if !foundSource {
		t.Error("grounding source not found in stream chunks")
	}
}

func TestChat_Generate_NoGroundingMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"modelVersion": "gemini-2.5-flash",
			"candidates": [{
				"content": {"parts": [{"text": "No grounding."}]},
				"finishReason": "STOP"
			}],
			"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}
		}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))
	result, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Sources) != 0 {
		t.Errorf("len(Sources) = %d, want 0", len(result.Sources))
	}
}

func TestChat_EnvVarResolution(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "env-key")
	m := Chat("gemini-2.5-flash")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource should be set from GOOGLE_GENERATIVE_AI_API_KEY")
	}
}

func TestChat_EnvVarBaseURL(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "env-key")
	t.Setenv("GOOGLE_GENERATIVE_AI_BASE_URL", "https://custom.googleapis.com")
	m := Chat("gemini-2.5-flash")
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://custom.googleapis.com" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestChat_EnvVarNotOverrideExplicit(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_BASE_URL", "https://env.url")
	m := Chat("gemini-2.5-flash", WithAPIKey("explicit"), WithBaseURL("https://explicit.url"))
	cm := m.(*chatModel)
	if cm.opts.baseURL != "https://explicit.url" {
		t.Errorf("baseURL = %q", cm.opts.baseURL)
	}
}

func TestBuildRequest_LegacyGoogleSearchProviderOption(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Tools []map[string]any `json:"tools"`
		}
		_ = json.Unmarshal(body, &req)

		// Should have one tool entry with "googleSearch" from legacy ProviderOptions.
		found := false
		for _, tool := range req.Tools {
			if _, ok := tool["googleSearch"]; ok {
				found = true
			}
		}
		if !found {
			t.Error("expected googleSearch tool from legacy ProviderOptions")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "search"}}},
		},
		ProviderOptions: map[string]any{
			"google": map[string]any{
				"google_search": map[string]any{
					"dynamicRetrievalConfig": map[string]any{
						"mode":             "MODE_DYNAMIC",
						"dynamicThreshold": 0.5,
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestBuildRequest_ProviderDefinedTool(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Tools []map[string]any `json:"tools"`
		}
		_ = json.Unmarshal(body, &req)

		// Expect 2 tool entries: one functionDeclarations, one googleSearch.
		if len(req.Tools) != 2 {
			t.Errorf("tools count = %d, want 2", len(req.Tools))
		}

		// First entry should be functionDeclarations.
		if _, ok := req.Tools[0]["functionDeclarations"]; !ok {
			t.Error("first tool should be functionDeclarations")
		}

		// Second entry should be googleSearch (provider-defined).
		if _, ok := req.Tools[1]["googleSearch"]; !ok {
			t.Errorf("second tool should be googleSearch, got keys: %v", req.Tools[1])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test"), WithBaseURL(server.URL))
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "search"}}},
		},
		Tools: []provider.ToolDefinition{
			{
				Name:        "get_weather",
				Description: "Get weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{}}`),
			},
			Tools.GoogleSearch(WithWebSearch()),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestParseSSE_ContextCancel_AllBranches(t *testing.T) {
	// Exercise every TrySend early-return path in parseSSE with a cancelled
	// context and unbuffered channel.

	tests := []struct {
		name  string
		input string
	}{
		{
			// ChunkToolCallStreamStart (line 611)
			name:  "tool_call_start",
			input: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"fn\",\"args\":{\"a\":1}}}]}}]}\n",
		},
		{
			// ChunkReasoning (line 629) -- thought part
			name:  "reasoning",
			input: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"thinking\",\"thought\":true}]}}]}\n",
		},
		{
			// ChunkText (line 633)
			name:  "text",
			input: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hello\"}]}}]}\n",
		},
		{
			// Grounding source (line 642)
			name:  "grounding_source",
			input: "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"groundingMetadata\":{\"groundingChunks\":[{\"web\":{\"uri\":\"http://x\",\"title\":\"T\"}}]}}]}\n",
		},
		{
			// ChunkStepFinish (line 655)
			name:  "step_finish",
			input: "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1}}\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			out := make(chan provider.StreamChunk) // unbuffered
			done := make(chan struct{})
			go func() {
				parseSSE(ctx, strings.NewReader(tc.input), out)
				close(done)
			}()
			<-done
			for range out {
			}
		})
	}

	// Nested: ChunkToolCall (line 619) requires start TrySend (611) to succeed first.
	t.Run("tool_call_complete_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"fn\",\"args\":{\"a\":1}}}]}}]}\n"

		done := make(chan struct{})
		go func() {
			parseSSE(ctx, strings.NewReader(input), out)
			close(done)
		}()

		<-out // receive tool start chunk
		cancel()
		<-done
		for range out {
		}
	})

	// Nested: ChunkStepFinish after grounding sources.
	t.Run("step_finish_after_source_cancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		out := make(chan provider.StreamChunk) // unbuffered

		input := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]},\"groundingMetadata\":{\"groundingChunks\":[{\"web\":{\"uri\":\"http://x\",\"title\":\"T\"}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1}}\n"

		done := make(chan struct{})
		go func() {
			parseSSE(ctx, strings.NewReader(input), out)
			close(done)
		}()

		<-out // text
		<-out // grounding source
		cancel()
		<-done
		for range out {
		}
	})
}

func TestBuildRequest_Tools_InvalidSchema(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	_, err := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "broken_tool", Description: "a tool with bad schema", InputSchema: json.RawMessage(`not valid json`)},
		},
	})
	if err == nil {
		t.Fatal("expected error for invalid tool schema, got nil")
	}
	if !strings.Contains(err.Error(), "invalid tool schema") {
		t.Errorf("error = %q, want 'invalid tool schema' in message", err.Error())
	}
}

// --- Coverage: DoGenerate buildRequest error path (line 131) ---

func TestDoGenerate_BuildRequestError(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL, tokenSource: provider.StaticToken("k")}}
	_, err := m.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "bad", InputSchema: json.RawMessage(`not json`)},
		},
	})
	if err == nil {
		t.Fatal("expected error from buildRequest")
	}
	if !strings.Contains(err.Error(), "invalid tool schema") {
		t.Errorf("error = %v", err)
	}
}

// --- Coverage: DoStream buildRequest error path (line 151) ---

func TestDoStream_BuildRequestError(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL, tokenSource: provider.StaticToken("k")}}
	_, err := m.DoStream(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Tools: []provider.ToolDefinition{
			{Name: "bad", InputSchema: json.RawMessage(`not json`)},
		},
	})
	if err == nil {
		t.Fatal("expected error from buildRequest")
	}
	if !strings.Contains(err.Error(), "invalid tool schema") {
		t.Errorf("error = %v", err)
	}
}

// --- Coverage: buildRequest ResponseFormat with no Schema (JSON mode only) ---

func TestBuildRequest_ResponseFormat_NoSchema(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	body, err := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		ResponseFormat: &provider.ResponseFormat{
			Name: "json_mode",
			// Schema intentionally nil/empty - pure JSON mode.
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	config := body.GenerationConfig.(map[string]any)
	if config["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType = %v, want application/json", config["responseMimeType"])
	}
	if _, ok := config["responseSchema"]; ok {
		t.Error("responseSchema should not be set when Schema is empty")
	}
}

// --- Coverage: mapFinishReason all values (83.3% → 100%) ---

func TestMapFinishReason_AllValues(t *testing.T) {
	tests := []struct {
		input string
		want  provider.FinishReason
	}{
		{"STOP", provider.FinishStop},
		{"MAX_TOKENS", provider.FinishLength},
		{"SAFETY", provider.FinishContentFilter},
		{"IMAGE_SAFETY", provider.FinishContentFilter},
		{"RECITATION", provider.FinishContentFilter},
		{"BLOCKLIST", provider.FinishContentFilter},
		{"PROHIBITED_CONTENT", provider.FinishContentFilter},
		{"SPII", provider.FinishContentFilter},
		{"MALFORMED_FUNCTION_CALL", provider.FinishError},
		{"UNKNOWN_REASON", provider.FinishOther},
		{"", provider.FinishOther},
	}
	for _, tt := range tests {
		if got := mapFinishReason(tt.input); got != tt.want {
			t.Errorf("mapFinishReason(%q) = %s, want %s", tt.input, got, tt.want)
		}
	}
}

// --- Coverage: doHTTP resolveToken error (no token source) ---

func TestDoHTTP_NoTokenSource(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: "http://localhost"}}
	_, err := m.doHTTP(t.Context(), "http://localhost/test", nil, nil)
	if err == nil {
		t.Fatal("expected error for missing token source")
	}
	if !strings.Contains(err.Error(), "resolving auth token") {
		t.Errorf("error = %v", err)
	}
}

// --- Coverage: doHTTP per-request headers ---

func TestDoHTTP_PerRequestHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Per-Request") != "yes" {
			t.Errorf("missing per-request header")
		}
		if r.Header.Get("X-Provider") != "google" {
			t.Errorf("missing provider header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`)
	}))
	defer srv.Close()

	model := Chat("gemini-2.5-flash",
		WithAPIKey("test-key"),
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{"X-Provider": "google"}),
	)
	_, err := model.DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
		Headers: map[string]string{"X-Per-Request": "yes"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// --- Coverage: Image env var resolution (line 49-52) ---

func TestImage_EnvVarAPIKey(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "env-image-key")
	model := Image("imagen-4.0-generate-001")
	im := model.(*imagenModel)
	if im.opts.tokenSource == nil {
		t.Error("tokenSource should be set from GOOGLE_GENERATIVE_AI_API_KEY")
	}
}

// --- Coverage: buildRequest TopK branch (line 313) ---

func TestBuildRequest_TopK(t *testing.T) {
	m := &chatModel{id: "gemini-2.5-flash", opts: options{baseURL: defaultBaseURL}}
	topK := 40
	body, _ := m.buildRequest(provider.GenerateParams{
		Messages: []provider.Message{{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}}},
		TopK:     &topK,
	})
	config := body.GenerationConfig.(map[string]any)
	if config["topK"] != 40 {
		t.Errorf("topK = %v, want 40", config["topK"])
	}
}

// --- Coverage: sanitizeImpl non-map path (line 919) ---

func TestSanitizeImpl_NonMap(t *testing.T) {
	result := sanitizeImpl("a string")
	if result != "a string" {
		t.Errorf("sanitizeImpl(string) = %v, want 'a string'", result)
	}
	result = sanitizeImpl(42)
	if result != 42 {
		t.Errorf("sanitizeImpl(int) = %v, want 42", result)
	}
}

func TestImage_EnvVarAPIKey_Gemini(t *testing.T) {
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "env-image-key")
	model := Image("gemini-2.5-flash-image")
	gm := model.(*geminiImageModel)
	if gm.opts.tokenSource == nil {
		t.Error("tokenSource should be set from GOOGLE_GENERATIVE_AI_API_KEY")
	}
}

func TestChat_EnvVarResolution_GeminiAPIKey(t *testing.T) {
	// GEMINI_API_KEY should work as fallback when GOOGLE_GENERATIVE_AI_API_KEY is not set.
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	m := Chat("gemini-2.5-flash")
	cm := m.(*chatModel)
	if cm.opts.tokenSource == nil {
		t.Error("tokenSource should be set from GEMINI_API_KEY")
	}
}

// TestParseSSE_OverflowError_ResponseBodyPopulated verifies that a
// ContextOverflowError emitted from a streaming SSE error response includes
// both Message and ResponseBody.
func TestParseSSE_OverflowError_ResponseBodyPopulated(t *testing.T) {
	overflowMsg := "context window exceeds limit"
	data := `{"error":{"code":400,"message":"` + overflowMsg + `","status":"INVALID_ARGUMENT"}}`
	sseInput := "data: " + data + "\n\n"

	out := make(chan provider.StreamChunk, 8)
	parseSSE(t.Context(), strings.NewReader(sseInput), out)

	var chunks []provider.StreamChunk
	for c := range out {
		chunks = append(chunks, c)
	}

	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}

	errChunk := chunks[0]
	if errChunk.Type != provider.ChunkError {
		t.Fatalf("chunk type = %v, want ChunkError", errChunk.Type)
	}

	var overflowErr *goai.ContextOverflowError
	if !errors.As(errChunk.Error, &overflowErr) {
		t.Fatalf("error type = %T, want *goai.ContextOverflowError", errChunk.Error)
	}
	if overflowErr.Message == "" {
		t.Error("ContextOverflowError.Message is empty")
	}
	if overflowErr.ResponseBody == "" {
		t.Error("ContextOverflowError.ResponseBody is empty; want the raw SSE data string")
	}
	if overflowErr.ResponseBody != data {
		t.Errorf("ResponseBody = %q, want %q", overflowErr.ResponseBody, data)
	}
}

// TestDoStream_ContextCancel_NoDoubleClose verifies that cancelling the context
// does not cause a panic or double-close of resp.Body.
func TestDoStream_ContextCancel_NoDoubleClose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Flush a chunk then block indefinitely to simulate a stalled server.
		_, _ = fmt.Fprint(w, `data: {"modelVersion":"gemini-2.5-flash","candidates":[{"content":{"parts":[{"text":"hi"}]}}]}`+"\n\n")
		w.(http.Flusher).Flush()
		// Block until client disconnects.
		<-r.Context().Done()
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))

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

// TestChat_PromptCachingIgnored verifies that passing PromptCaching=true to the Google
// provider succeeds (warning is written to stderr, not returned as error).
func TestChat_PromptCachingIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)
	}))
	defer server.Close()

	model := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(server.URL))

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
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`+"\n\n")
	}))
	defer streamServer.Close()

	streamModel := Chat("gemini-2.5-flash", WithAPIKey("test-key"), WithBaseURL(streamServer.URL))

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
