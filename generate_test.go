package goai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// mockModel implements provider.LanguageModel for testing.
type mockModel struct {
	id         string
	generateFn func(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error)
	streamFn   func(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error)
}

func (m *mockModel) ModelID() string { return m.id }
func (m *mockModel) DoGenerate(ctx context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
	return m.generateFn(ctx, params)
}
func (m *mockModel) DoStream(ctx context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
	return m.streamFn(ctx, params)
}
func (m *mockModel) Capabilities() provider.ModelCapabilities {
	return provider.ModelCapabilities{}
}

// streamFromChunks creates a StreamResult from pre-built chunks.
func streamFromChunks(chunks ...provider.StreamChunk) *provider.StreamResult {
	ch := make(chan provider.StreamChunk, len(chunks))
	for _, c := range chunks {
		ch <- c
	}
	close(ch)
	return &provider.StreamResult{Stream: ch}
}

// --- StreamText tests ---

func TestStreamText_Stream(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello"},
				provider.StreamChunk{Type: provider.ChunkText, Text: " world"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range stream.Stream() {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}
	if chunks[0].Text != "Hello" {
		t.Errorf("chunk[0].Text = %q, want %q", chunks[0].Text, "Hello")
	}
	if chunks[1].Text != " world" {
		t.Errorf("chunk[1].Text = %q, want %q", chunks[1].Text, " world")
	}
	if chunks[2].Type != provider.ChunkFinish {
		t.Errorf("chunk[2].Type = %q, want %q", chunks[2].Type, provider.ChunkFinish)
	}
}

func TestStreamText_StreamThenResult(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello"},
				provider.StreamChunk{Type: provider.ChunkText, Text: " world"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	// Consume stream first.
	for range stream.Stream() {
	}

	// Then get accumulated result.
	result := stream.Result()
	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want %q", result.Text, "Hello world")
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
	if result.TotalUsage.InputTokens != 10 {
		t.Errorf("TotalUsage.InputTokens = %d, want 10", result.TotalUsage.InputTokens)
	}
	if result.TotalUsage.OutputTokens != 5 {
		t.Errorf("TotalUsage.OutputTokens = %d, want 5", result.TotalUsage.OutputTokens)
	}
}

func TestStreamText_TextStream(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello"},
				provider.StreamChunk{Type: provider.ChunkReasoning, Text: "thinking..."},
				provider.StreamChunk{Type: provider.ChunkText, Text: " world"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	var texts []string
	for text := range stream.TextStream() {
		texts = append(texts, text)
	}

	// Should only get text chunks, not reasoning.
	if len(texts) != 2 {
		t.Fatalf("expected 2 text chunks, got %d", len(texts))
	}
	if texts[0] != "Hello" || texts[1] != " world" {
		t.Errorf("texts = %v, want [Hello, \" world\"]", texts)
	}
}

func TestStreamText_TextStreamThenResult(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "ok"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{OutputTokens: 1}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	for range stream.TextStream() {
	}

	result := stream.Result()
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
	if result.TotalUsage.OutputTokens != 1 {
		t.Errorf("TotalUsage.OutputTokens = %d, want 1", result.TotalUsage.OutputTokens)
	}
}

func TestStreamText_ResultOnly(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "result"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{OutputTokens: 1}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if result.Text != "result" {
		t.Errorf("Text = %q, want %q", result.Text, "result")
	}
	if len(result.Steps) != 1 {
		t.Fatalf("expected 1 step, got %d", len(result.Steps))
	}
	if result.Steps[0].Number != 1 {
		t.Errorf("Steps[0].Number = %d, want 1", result.Steps[0].Number)
	}
	if result.Steps[0].Text != "result" {
		t.Errorf("Steps[0].Text = %q, want %q", result.Steps[0].Text, "result")
	}
}

func TestStreamText_ToolCalls(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "reading file"},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "read", ToolInput: `{"path":"a.txt"}`},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc2", ToolName: "write", ToolInput: `{"path":"b.txt","content":"hi"}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("read and write"))
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if result.Text != "reading file" {
		t.Errorf("Text = %q, want %q", result.Text, "reading file")
	}
	if len(result.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "read" {
		t.Errorf("ToolCalls[0].Name = %q, want %q", result.ToolCalls[0].Name, "read")
	}
	if result.ToolCalls[1].Name != "write" {
		t.Errorf("ToolCalls[1].Name = %q, want %q", result.ToolCalls[1].Name, "write")
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishToolCalls)
	}

	// Verify tool call input can be unmarshaled.
	var input struct{ Path string }
	if err := json.Unmarshal(result.ToolCalls[0].Input, &input); err != nil {
		t.Fatalf("Unmarshal tool input: %v", err)
	}
	if input.Path != "a.txt" {
		t.Errorf("ToolCalls[0].Input.Path = %q, want %q", input.Path, "a.txt")
	}
}

func TestStreamText_EmptyStream(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if result.Text != "" {
		t.Errorf("Text = %q, want empty", result.Text)
	}
	if len(result.ToolCalls) != 0 {
		t.Errorf("expected 0 tool calls, got %d", len(result.ToolCalls))
	}
}

func TestStreamText_ErrorFromProvider(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, &APIError{Message: "rate limited", StatusCode: 429, IsRetryable: true}
		},
	}

	_, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.StatusCode != http.StatusTooManyRequests {
		t.Errorf("status = %d, want %d", apiErr.StatusCode, http.StatusTooManyRequests)
	}
}

func TestStreamText_ErrorChunk(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "partial"},
				provider.StreamChunk{Type: provider.ChunkError, Error: fmt.Errorf("stream error")},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	var gotError bool
	for chunk := range stream.Stream() {
		if chunk.Type == provider.ChunkError {
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected error chunk in stream")
	}

	// Text still accumulated up to the error.
	result := stream.Result()
	if result.Text != "partial" {
		t.Errorf("Text = %q, want %q", result.Text, "partial")
	}
}

func TestStreamText_StepFinishChunk(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hello"},
				provider.StreamChunk{Type: provider.ChunkStepFinish, FinishReason: provider.FinishStop},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{OutputTokens: 2}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	var chunks []provider.StreamChunk
	for chunk := range stream.Stream() {
		chunks = append(chunks, chunk)
	}

	// Verify step_finish chunk is passed through.
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}
	if chunks[1].Type != provider.ChunkStepFinish {
		t.Errorf("chunks[1].Type = %q, want %q", chunks[1].Type, provider.ChunkStepFinish)
	}
}

// --- GenerateText tests ---

func TestGenerateText(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "Hello world",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "Hello world" {
		t.Errorf("Text = %q, want %q", result.Text, "Hello world")
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishStop)
	}
	if result.TotalUsage.InputTokens != 10 || result.TotalUsage.OutputTokens != 5 {
		t.Errorf("TotalUsage = %+v, want InputTokens=10, OutputTokens=5", result.TotalUsage)
	}
	if len(result.Steps) != 1 {
		t.Fatalf("expected 1 step, got %d", len(result.Steps))
	}
	if result.Steps[0].Number != 1 {
		t.Errorf("Steps[0].Number = %d, want 1", result.Steps[0].Number)
	}
	if result.Steps[0].Text != "Hello world" {
		t.Errorf("Steps[0].Text = %q, want %q", result.Steps[0].Text, "Hello world")
	}
	if result.Steps[0].Usage.InputTokens != 10 {
		t.Errorf("Steps[0].Usage.InputTokens = %d, want 10", result.Steps[0].Usage.InputTokens)
	}
}

func TestGenerateText_WithToolCalls(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			if len(params.Tools) != 1 || params.Tools[0].Name != "read" {
				t.Errorf("tools not passed through: %+v", params.Tools)
			}
			return &provider.GenerateResult{
				ToolCalls: []provider.ToolCall{{
					ID:    "tc1",
					Name:  "read",
					Input: json.RawMessage(`{"path":"a.txt"}`),
				}},
				FinishReason: provider.FinishToolCalls,
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("read file"),
		WithTools(Tool{
			Name:        "read",
			Description: "Read a file",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "read" {
		t.Errorf("ToolCalls[0].Name = %q, want %q", result.ToolCalls[0].Name, "read")
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want %q", result.FinishReason, provider.FinishToolCalls)
	}
}

func TestGenerateText_Error(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, &APIError{Message: "bad request", StatusCode: 400}
		},
	}

	_, err := GenerateText(t.Context(), model, WithPrompt("hi"))
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- buildParams tests ---

func TestBuildParams_Prompt(t *testing.T) {
	opts := applyOptions(WithPrompt("hello"), WithSystem("be helpful"))
	params := buildParams(opts)

	if params.System != "be helpful" {
		t.Errorf("System = %q, want %q", params.System, "be helpful")
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	if params.Messages[0].Role != provider.RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", params.Messages[0].Role, provider.RoleUser)
	}
	if params.Messages[0].Content[0].Text != "hello" {
		t.Errorf("Messages[0].Content[0].Text = %q, want %q", params.Messages[0].Content[0].Text, "hello")
	}
}

func TestBuildParams_PromptPrependedToMessages(t *testing.T) {
	opts := applyOptions(
		WithPrompt("hello"),
		WithMessages(AssistantMessage("hi"), UserMessage("how are you")),
	)
	params := buildParams(opts)

	if len(params.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(params.Messages))
	}
	if params.Messages[0].Role != provider.RoleUser {
		t.Errorf("Messages[0].Role = %q, want user (prompt)", params.Messages[0].Role)
	}
	if params.Messages[0].Content[0].Text != "hello" {
		t.Errorf("Messages[0] should be prompt message")
	}
	if params.Messages[1].Role != provider.RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want assistant", params.Messages[1].Role)
	}
	if params.Messages[2].Role != provider.RoleUser {
		t.Errorf("Messages[2].Role = %q, want user", params.Messages[2].Role)
	}
}

func TestBuildParams_MessagesOnly(t *testing.T) {
	opts := applyOptions(WithMessages(UserMessage("hi")))
	params := buildParams(opts)

	if len(params.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(params.Messages))
	}
	if params.Messages[0].Content[0].Text != "hi" {
		t.Errorf("Messages[0].Content[0].Text = %q, want %q", params.Messages[0].Content[0].Text, "hi")
	}
}

func TestBuildParams_NoPromptNoMessages(t *testing.T) {
	opts := applyOptions()
	params := buildParams(opts)

	if len(params.Messages) != 0 {
		t.Errorf("expected 0 messages, got %d", len(params.Messages))
	}
}

func TestBuildParams_Tools(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	opts := applyOptions(WithTools(
		Tool{Name: "read", Description: "Read file", InputSchema: schema},
		Tool{Name: "write", Description: "Write file", InputSchema: schema},
	))
	params := buildParams(opts)

	if len(params.Tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(params.Tools))
	}
	if params.Tools[0].Name != "read" || params.Tools[0].Description != "Read file" {
		t.Errorf("Tools[0] = %+v", params.Tools[0])
	}
	if params.Tools[1].Name != "write" {
		t.Errorf("Tools[1].Name = %q, want %q", params.Tools[1].Name, "write")
	}
	// InputSchema should be preserved.
	if string(params.Tools[0].InputSchema) != `{"type":"object"}` {
		t.Errorf("Tools[0].InputSchema = %s", params.Tools[0].InputSchema)
	}
}

func TestBuildParams_ToolExecuteNotIncluded(t *testing.T) {
	opts := applyOptions(WithTools(Tool{
		Name:    "test",
		Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "", nil },
	}))
	params := buildParams(opts)

	// provider.ToolDefinition should not have Execute -- verify by checking fields exist.
	if len(params.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(params.Tools))
	}
	if params.Tools[0].Name != "test" {
		t.Errorf("Tools[0].Name = %q, want %q", params.Tools[0].Name, "test")
	}
}

func TestBuildParams_AllOptions(t *testing.T) {
	opts := applyOptions(
		WithMaxOutputTokens(100),
		WithTemperature(0.7),
		WithTopP(0.9),
		WithTopK(10),
		WithSeed(42),
		WithFrequencyPenalty(0.5),
		WithPresencePenalty(0.3),
		WithStopSequences("END", "STOP"),
		WithHeaders(map[string]string{"X-Custom": "val"}),
		WithProviderOptions(map[string]any{"key": "val"}),
		WithPromptCaching(true),
		WithToolChoice("required"),
	)
	params := buildParams(opts)

	if params.MaxOutputTokens != 100 {
		t.Errorf("MaxOutputTokens = %d, want 100", params.MaxOutputTokens)
	}
	if params.Temperature == nil || *params.Temperature != 0.7 {
		t.Error("Temperature not set correctly")
	}
	if params.TopP == nil || *params.TopP != 0.9 {
		t.Error("TopP not set correctly")
	}
	if params.TopK == nil || *params.TopK != 10 {
		t.Error("TopK not set correctly")
	}
	if params.Seed == nil || *params.Seed != 42 {
		t.Error("Seed not set correctly")
	}
	if params.FrequencyPenalty == nil || *params.FrequencyPenalty != 0.5 {
		t.Error("FrequencyPenalty not set correctly")
	}
	if params.PresencePenalty == nil || *params.PresencePenalty != 0.3 {
		t.Error("PresencePenalty not set correctly")
	}
	if len(params.StopSequences) != 2 || params.StopSequences[0] != "END" {
		t.Errorf("StopSequences = %v", params.StopSequences)
	}
	if params.Headers["X-Custom"] != "val" {
		t.Error("Headers not passed")
	}
	if params.ProviderOptions["key"] != "val" {
		t.Error("ProviderOptions not passed")
	}
	if !params.PromptCaching {
		t.Error("PromptCaching should be true")
	}
	if params.ToolChoice != "required" {
		t.Errorf("ToolChoice = %q, want %q", params.ToolChoice, "required")
	}
}

func TestStreamText_ParamsPassThrough(t *testing.T) {
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			capturedParams = params
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("hello"),
		WithSystem("be helpful"),
		WithMaxOutputTokens(100),
		WithTemperature(0.5),
	)
	if err != nil {
		t.Fatal(err)
	}
	stream.Result() // consume

	if capturedParams.System != "be helpful" {
		t.Errorf("System = %q, want %q", capturedParams.System, "be helpful")
	}
	if capturedParams.MaxOutputTokens != 100 {
		t.Errorf("MaxOutputTokens = %d, want 100", capturedParams.MaxOutputTokens)
	}
	if capturedParams.Temperature == nil || *capturedParams.Temperature != 0.5 {
		t.Error("Temperature not passed")
	}
	if len(capturedParams.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(capturedParams.Messages))
	}
}

func TestGenerateText_ParamsPassThrough(t *testing.T) {
	var capturedParams provider.GenerateParams
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			capturedParams = params
			return &provider.GenerateResult{Text: "ok"}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("hello"),
		WithSystem("be helpful"),
		WithToolChoice("auto"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if capturedParams.System != "be helpful" {
		t.Errorf("System = %q, want %q", capturedParams.System, "be helpful")
	}
	if capturedParams.ToolChoice != "auto" {
		t.Errorf("ToolChoice = %q, want %q", capturedParams.ToolChoice, "auto")
	}
}

func TestStreamText_WithUsageFields(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hi"},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Usage: provider.Usage{
						InputTokens:      100,
						OutputTokens:     50,
						ReasoningTokens:  10,
						CacheReadTokens:  20,
						CacheWriteTokens: 5,
					},
				},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	u := result.TotalUsage
	if u.InputTokens != 100 || u.OutputTokens != 50 || u.ReasoningTokens != 10 || u.CacheReadTokens != 20 || u.CacheWriteTokens != 5 {
		t.Errorf("TotalUsage = %+v", u)
	}
}

// --- Tool Loop tests ---

func TestGenerateText_ToolLoop_SingleStep(t *testing.T) {
	// Model requests a tool call, tool executes, model responds with text.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{{
						ID:    "tc1",
						Name:  "get_weather",
						Input: json.RawMessage(`{"city":"NYC"}`),
					}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			// Second call: model has tool results and responds with text.
			return &provider.GenerateResult{
				Text:         "The weather in NYC is sunny.",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("weather in NYC?"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "get_weather",
			Description: "Get weather for a city",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				var args struct{ City string }
				_ = json.Unmarshal(input, &args)
				return fmt.Sprintf("Sunny in %s", args.City), nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if callCount != 2 {
		t.Errorf("model called %d times, want 2", callCount)
	}
	if result.Text != "The weather in NYC is sunny." {
		t.Errorf("Text = %q", result.Text)
	}
	if len(result.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(result.Steps))
	}
	if result.Steps[0].Number != 1 || result.Steps[1].Number != 2 {
		t.Errorf("step numbers = %d, %d", result.Steps[0].Number, result.Steps[1].Number)
	}
	if result.TotalUsage.InputTokens != 30 || result.TotalUsage.OutputTokens != 15 {
		t.Errorf("TotalUsage = %+v", result.TotalUsage)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want stop", result.FinishReason)
	}
}

func TestGenerateText_ToolLoop_MultiStep(t *testing.T) {
	// 3 steps: tool call → tool call → final text.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			switch callCount {
			case 1:
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "step1", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			case 2:
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc2", Name: "step2", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 15, OutputTokens: 5},
				}, nil
			default:
				return &provider.GenerateResult{
					Text:         "done",
					FinishReason: provider.FinishStop,
					Usage:        provider.Usage{InputTokens: 20, OutputTokens: 3},
				}, nil
			}
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(
			Tool{Name: "step1", Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "result1", nil }},
			Tool{Name: "step2", Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "result2", nil }},
		),
	)
	if err != nil {
		t.Fatal(err)
	}

	if callCount != 3 {
		t.Errorf("model called %d times, want 3", callCount)
	}
	if len(result.Steps) != 3 {
		t.Fatalf("expected 3 steps, got %d", len(result.Steps))
	}
	if result.TotalUsage.InputTokens != 45 {
		t.Errorf("TotalUsage.InputTokens = %d, want 45", result.TotalUsage.InputTokens)
	}
}

func TestGenerateText_ToolLoop_MaxStepsReached(t *testing.T) {
	// Model always requests tool calls -- should stop at MaxSteps.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			return &provider.GenerateResult{
				ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", callCount), Name: "loop", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "loop",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "looping", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if callCount != 3 {
		t.Errorf("model called %d times, want 3 (MaxSteps)", callCount)
	}
	if len(result.Steps) != 3 {
		t.Fatalf("expected 3 steps, got %d", len(result.Steps))
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want tool_calls (MaxSteps hit)", result.FinishReason)
	}
}

func TestGenerateText_ToolLoop_OnStepFinish(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
		},
	}

	var stepResults []StepResult
	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnStepFinish(func(step StepResult) {
			stepResults = append(stepResults, step)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(stepResults) != 2 {
		t.Fatalf("OnStepFinish called %d times, want 2", len(stepResults))
	}
	if stepResults[0].Number != 1 || stepResults[1].Number != 2 {
		t.Errorf("step numbers = [%d, %d], want [1, 2]", stepResults[0].Number, stepResults[1].Number)
	}
	// Step 1 is a tool-call step; step 2 is the final answer.
	if stepResults[0].FinishReason != provider.FinishToolCalls {
		t.Errorf("step 1 FinishReason = %q, want %q", stepResults[0].FinishReason, provider.FinishToolCalls)
	}
	if stepResults[1].FinishReason != provider.FinishStop {
		t.Errorf("step 2 FinishReason = %q, want %q", stepResults[1].FinishReason, provider.FinishStop)
	}
}

func TestGenerateText_ToolLoop_ToolError(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "fail", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Verify error message was passed as tool result.
			for _, msg := range params.Messages {
				for _, p := range msg.Content {
					if p.Type == provider.PartToolResult && p.ToolOutput == "error: something went wrong" {
						return &provider.GenerateResult{Text: "handled error", FinishReason: provider.FinishStop}, nil
					}
				}
			}
			return &provider.GenerateResult{Text: "no error found", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "fail",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "", fmt.Errorf("something went wrong")
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "handled error" {
		t.Errorf("Text = %q, want 'handled error'", result.Text)
	}
}

func TestGenerateText_ToolLoop_UnknownTool(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "nonexistent", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Check tool result is "error: unknown tool".
			for _, msg := range params.Messages {
				for _, p := range msg.Content {
					if p.Type == provider.PartToolResult && p.ToolOutput == "error: unknown tool" {
						return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
					}
				}
			}
			return &provider.GenerateResult{Text: "unexpected", FinishReason: provider.FinishStop}, nil
		},
	}

	var capturedInfo ToolCallInfo
	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "known",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnToolCall(func(info ToolCallInfo) {
			capturedInfo = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "ok" {
		t.Errorf("Text = %q, want 'ok'", result.Text)
	}

	if capturedInfo.ToolName != "nonexistent" {
		t.Errorf("ToolCallInfo.ToolName = %q, want %q", capturedInfo.ToolName, "nonexistent")
	}
	if !errors.Is(capturedInfo.Error, ErrUnknownTool) {
		t.Errorf("ToolCallInfo.Error = %v, want ErrUnknownTool", capturedInfo.Error)
	}
}

func TestGenerateText_ToolLoop_NoExecuteNoLoop(t *testing.T) {
	// Tools without Execute should not trigger the loop.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			return &provider.GenerateResult{
				ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "read", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{
			Name:        "read",
			Description: "Read a file",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			// No Execute -- tool definitions only, loop managed externally.
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if callCount != 1 {
		t.Errorf("model called %d times, want 1 (no loop without Execute)", callCount)
	}
	if len(result.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
}

func TestGenerateText_ToolLoop_MultipleToolCalls(t *testing.T) {
	// Model requests 2 tool calls in one step.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "tc1", Name: "read", Input: json.RawMessage(`{"path":"a.txt"}`)},
						{ID: "tc2", Name: "read", Input: json.RawMessage(`{"path":"b.txt"}`)},
					},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 8},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "both files read",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 30, OutputTokens: 5},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("read a.txt and b.txt"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "read",
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				var args struct{ Path string }
				_ = json.Unmarshal(input, &args)
				return "contents of " + args.Path, nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "both files read" {
		t.Errorf("Text = %q", result.Text)
	}
	if len(result.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(result.Steps))
	}
	// First step had 2 tool calls.
	if len(result.Steps[0].ToolCalls) != 2 {
		t.Errorf("step 1 tool calls = %d, want 2", len(result.Steps[0].ToolCalls))
	}
}

func TestGenerateText_ToolLoop_MessagesPassedCorrectly(t *testing.T) {
	// Verify the conversation history grows correctly across steps.
	callCount := 0
	var messageCounts []int
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			messageCounts = append(messageCounts, len(params.Messages))
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "result", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Step 1: 1 message (user prompt).
	// Step 2: 1 (user) + 1 (assistant with tool_use) + 1 (tool result) = 3.
	if len(messageCounts) != 2 {
		t.Fatalf("model called %d times, want 2", len(messageCounts))
	}
	if messageCounts[0] != 1 {
		t.Errorf("step 1 messages = %d, want 1", messageCounts[0])
	}
	if messageCounts[1] != 3 {
		t.Errorf("step 2 messages = %d, want 3", messageCounts[1])
	}
}

func TestGenerateText_ToolLoop_ErrorOnSecondStep(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			return nil, &APIError{Message: "rate limited", StatusCode: 429}
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
	)
	if err == nil {
		t.Fatal("expected error on second step")
	}
}

func TestGenerateText_ToolLoop_TextAccumulation(t *testing.T) {
	// Text from multiple steps should be concatenated.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					Text:         "Step 1. ",
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "Step 2.",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "Step 1. Step 2." {
		t.Errorf("Text = %q, want 'Step 1. Step 2.'", result.Text)
	}
}

func TestBuildToolMap(t *testing.T) {
	// No tools → nil.
	if m := buildToolMap(nil); m != nil {
		t.Error("expected nil for no tools")
	}

	// Tools without Execute → nil.
	if m := buildToolMap([]Tool{{Name: "read"}}); m != nil {
		t.Error("expected nil for tools without Execute")
	}

	// Mixed.
	exec := func(_ context.Context, _ json.RawMessage) (string, error) { return "", nil }
	m := buildToolMap([]Tool{
		{Name: "read", Execute: exec},
		{Name: "write"},
	})
	if len(m) != 1 {
		t.Errorf("expected 1 executable tool, got %d", len(m))
	}
	if _, ok := m["read"]; !ok {
		t.Error("expected 'read' in map")
	}
}

func TestBuildTextResult_EmptySteps(t *testing.T) {
	usage := provider.Usage{InputTokens: 5, OutputTokens: 3}
	result := buildTextResult(nil, usage)
	if result.Text != "" {
		t.Errorf("Text = %q, want empty", result.Text)
	}
	if result.TotalUsage.InputTokens != 5 || result.TotalUsage.OutputTokens != 3 {
		t.Errorf("TotalUsage = %+v", result.TotalUsage)
	}
	if len(result.Steps) != 0 {
		t.Errorf("Steps = %d, want 0", len(result.Steps))
	}
}

func TestAddUsage(t *testing.T) {
	a := provider.Usage{InputTokens: 10, OutputTokens: 5, ReasoningTokens: 2, CacheReadTokens: 3, CacheWriteTokens: 1}
	b := provider.Usage{InputTokens: 20, OutputTokens: 10, ReasoningTokens: 3, CacheReadTokens: 7, CacheWriteTokens: 2}
	result := addUsage(a, b)
	if result.InputTokens != 30 || result.OutputTokens != 15 || result.ReasoningTokens != 5 || result.CacheReadTokens != 10 || result.CacheWriteTokens != 3 {
		t.Errorf("addUsage = %+v", result)
	}
}

func TestStreamText_WithTimeout_ErrorCancelsContext(t *testing.T) {
	// StreamText with Timeout: when DoStream fails, timeoutCancel must be called.
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, &APIError{Message: "bad request", StatusCode: 400}
		},
	}

	_, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithTimeout(5*time.Second),
	)
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.StatusCode != 400 {
		t.Errorf("StatusCode = %d, want 400", apiErr.StatusCode)
	}
}

func TestStreamText_WithTimeout_Success(t *testing.T) {
	// StreamText with Timeout: successful stream should work and goroutine
	// should call timeoutCancel when stream is consumed.
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hi"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{OutputTokens: 1}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithTimeout(5*time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if result.Text != "hi" {
		t.Errorf("Text = %q, want %q", result.Text, "hi")
	}
}

func TestStreamText_CtxCancelStopsConsume(t *testing.T) {
	// Verify that consume() returns promptly when context is cancelled,
	// even if the consumer has stopped reading from the rawOut channel.
	// The rawOut channel has capacity 64, so we need >64 chunks to fill
	// the buffer and trigger the blocking select where ctx.Done fires.
	ctx, cancel := context.WithCancel(t.Context())

	// Create a stream that produces many chunks (more than rawOut buffer of 64).
	ch := make(chan provider.StreamChunk, 1)
	go func() {
		defer close(ch)
		for range 500 {
			select {
			case ch <- provider.StreamChunk{Type: provider.ChunkText, Text: "x"}:
			case <-ctx.Done():
				return
			}
		}
		ch <- provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop}
	}()

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return &provider.StreamResult{Stream: ch}, nil
		},
	}

	stream, err := StreamText(ctx, model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	// Start consuming via Stream but don't read - let the buffer fill.
	rawCh := stream.Stream()

	// Read one chunk to ensure consume() goroutine is running.
	<-rawCh

	// Wait for the buffer to fill, then cancel context.
	// The consume goroutine will be blocked trying to send to the full rawOut channel.
	time.Sleep(20 * time.Millisecond)
	cancel()

	// The channel should close promptly.
	drained := 0
	for range rawCh {
		drained++
	}
	// If ctx.Done select works, we should get far fewer than 500 chunks.
	if drained >= 450 {
		t.Errorf("drained %d chunks - ctx.Done select did not trigger", drained)
	}
}

func TestGenerateText_WithResponseMetadata(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 5, OutputTokens: 1},
				Response:     provider.ResponseMetadata{ID: "resp-123", Model: "gpt-4o"},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
	if result.TotalUsage.InputTokens != 5 {
		t.Errorf("TotalUsage.InputTokens = %d, want 5", result.TotalUsage.InputTokens)
	}
	if result.Response.ID != "resp-123" {
		t.Errorf("Response.ID = %q, want %q", result.Response.ID, "resp-123")
	}
	if result.Response.Model != "gpt-4o" {
		t.Errorf("Response.Model = %q, want %q", result.Response.Model, "gpt-4o")
	}
}

func TestGenerateText_WithSources(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "The capital is Paris.",
				FinishReason: provider.FinishStop,
				Sources: []provider.Source{
					{Type: "url", URL: "https://example.com/paris", Title: "Paris Info", ID: "citation_0"},
					{Type: "url", URL: "https://wiki.org/france", Title: "France", ID: "citation_1"},
				},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model, WithPrompt("capital of France"))
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Sources) != 2 {
		t.Fatalf("Sources = %d, want 2", len(result.Sources))
	}
	if result.Sources[0].URL != "https://example.com/paris" {
		t.Errorf("Sources[0].URL = %q", result.Sources[0].URL)
	}
	if result.Sources[1].Title != "France" {
		t.Errorf("Sources[1].Title = %q", result.Sources[1].Title)
	}

	// Also check step-level sources.
	if len(result.Steps) != 1 {
		t.Fatalf("Steps = %d, want 1", len(result.Steps))
	}
	if len(result.Steps[0].Sources) != 2 {
		t.Errorf("Steps[0].Sources = %d, want 2", len(result.Steps[0].Sources))
	}
}

func TestStreamText_WithSources(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Paris"},
				provider.StreamChunk{
					Type: provider.ChunkText,
					Text: "",
					Metadata: map[string]any{
						"source": provider.Source{
							Type: "url", URL: "https://example.com/paris", Title: "Paris Info",
						},
					},
				},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Metadata: map[string]any{
						"sources": []provider.Source{
							{Type: "url", URL: "https://wiki.org", ID: "citation_0"},
						},
					},
				},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if result.Text != "Paris" {
		t.Errorf("Text = %q, want Paris", result.Text)
	}

	// Should have 2 sources: 1 from annotation + 1 from top-level citations.
	if len(result.Sources) != 2 {
		t.Fatalf("Sources = %d, want 2", len(result.Sources))
	}
	if result.Sources[0].URL != "https://example.com/paris" {
		t.Errorf("Sources[0].URL = %q", result.Sources[0].URL)
	}
	if result.Sources[1].URL != "https://wiki.org" {
		t.Errorf("Sources[1].URL = %q", result.Sources[1].URL)
	}
}

func TestBuildParams_ProviderDefinedTool(t *testing.T) {
	opts := applyOptions(
		WithTools(Tool{
			Name:                "computer",
			ProviderDefinedType: "computer_20250124",
			ProviderDefinedOptions: map[string]any{
				"display_width_px":  1920,
				"display_height_px": 1080,
			},
		}),
	)

	params := buildParams(opts)

	if len(params.Tools) != 1 {
		t.Fatalf("Tools = %d, want 1", len(params.Tools))
	}
	tool := params.Tools[0]
	if tool.ProviderDefinedType != "computer_20250124" {
		t.Errorf("ProviderDefinedType = %q, want computer_20250124", tool.ProviderDefinedType)
	}
	if tool.ProviderDefinedOptions["display_width_px"] != 1920 {
		t.Errorf("display_width_px = %v, want 1920", tool.ProviderDefinedOptions["display_width_px"])
	}
}

// --- Stream/TextStream mutual exclusion tests ---

func TestStreamText_StreamCalledTwice(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	ch1 := stream.Stream()
	ch2 := stream.Stream()

	// Second call should return the same channel as the first.
	if fmt.Sprintf("%p", ch1) != fmt.Sprintf("%p", ch2) {
		t.Error("expected Stream() called twice to return the same channel")
	}

	// Drain to avoid goroutine leak.
	for range ch1 {
	}
}

func TestStreamText_TextStreamCalledTwice(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	ch1 := stream.TextStream()
	ch2 := stream.TextStream()

	// Second call should return the same channel as the first.
	if fmt.Sprintf("%p", ch1) != fmt.Sprintf("%p", ch2) {
		t.Error("expected TextStream() called twice to return the same channel")
	}

	// Drain to avoid goroutine leak.
	for range ch1 {
	}
}

func TestStreamText_StreamAfterTextStream(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	// Consume via TextStream first.
	textCh := stream.TextStream()
	for range textCh {
	}

	// Now calling Stream() should return a closed channel.
	rawCh := stream.Stream()
	select {
	case _, ok := <-rawCh:
		if ok {
			t.Error("expected closed channel from Stream() after TextStream() consumed source")
		}
	case <-time.After(time.Second):
		t.Fatal("Stream() channel not closed within timeout")
	}
}

func TestStreamText_TextStreamAfterStream(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	// Consume via Stream first.
	rawCh := stream.Stream()
	for range rawCh {
	}

	// Now calling TextStream() should return a closed channel.
	textCh := stream.TextStream()
	select {
	case _, ok := <-textCh:
		if ok {
			t.Error("expected closed channel from TextStream() after Stream() consumed source")
		}
	case <-time.After(time.Second):
		t.Fatal("TextStream() channel not closed within timeout")
	}
}

func TestStreamText_TextStream_CtxCancelStopsConsume(t *testing.T) {
	// Verify that consume() returns promptly when context is cancelled
	// while using TextStream() (the textOut path in consume).
	// The textOut channel has capacity 64, so we need >64 chunks to fill
	// the buffer and trigger the blocking select where ctx.Done fires.
	ctx, cancel := context.WithCancel(t.Context())

	// Create a stream that produces many text chunks (more than textOut buffer of 64).
	ch := make(chan provider.StreamChunk, 1)
	go func() {
		defer close(ch)
		for range 500 {
			select {
			case ch <- provider.StreamChunk{Type: provider.ChunkText, Text: "x"}:
			case <-ctx.Done():
				return
			}
		}
		ch <- provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop}
	}()

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return &provider.StreamResult{Stream: ch}, nil
		},
	}

	stream, err := StreamText(ctx, model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	// Start consuming via TextStream but don't read - let the buffer fill.
	textCh := stream.TextStream()

	// Read one chunk to ensure consume() goroutine is running.
	<-textCh

	// Wait a bit for the buffer to fill, then cancel context.
	// The consume goroutine will be blocked trying to send to the full textOut channel.
	time.Sleep(20 * time.Millisecond)
	cancel()

	// The channel should close promptly.
	drained := 0
	for range textCh {
		drained++
	}
	// If ctx.Done select works on textOut, we should get far fewer than 500 chunks.
	if drained >= 450 {
		t.Errorf("drained %d text chunks - ctx.Done select on textOut did not trigger", drained)
	}
}

func TestStreamText_ErrNilOnSuccess(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hello"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}
	stream, err := StreamText(context.Background(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Result()
	if stream.Err() != nil {
		t.Errorf("expected nil Err(), got %v", stream.Err())
	}
}

func TestStreamText_ErrReturnsStreamError(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "partial"},
				provider.StreamChunk{Type: provider.ChunkError, Error: errors.New("stream broke")},
			), nil
		},
	}
	stream, err := StreamText(context.Background(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Result()
	if stream.Err() == nil {
		t.Fatal("expected non-nil Err()")
	}
	if stream.Err().Error() != "stream broke" {
		t.Errorf("unexpected error: %v", stream.Err())
	}
}

func TestExecuteTools_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel() // Cancel immediately.

	slowTool := Tool{
		Name:        "slow",
		Description: "a slow tool",
		InputSchema: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, input json.RawMessage) (string, error) {
			return "done", nil
		},
	}
	toolMap := map[string]Tool{"slow": slowTool}

	calls := []provider.ToolCall{
		{ID: "tc1", Name: "slow", Input: json.RawMessage(`{}`)},
		{ID: "tc2", Name: "slow", Input: json.RawMessage(`{}`)},
	}

	msgs := executeTools(ctx, calls, toolMap, 1, nil)
	// With a cancelled context, executeTools should return early (0 or fewer results).
	if len(msgs) >= 2 {
		t.Errorf("expected fewer than 2 messages (ctx cancelled), got %d", len(msgs))
	}
}

func TestConsume_ContextCancelledResultPath(t *testing.T) {
	// Create a stream that blocks, then cancel the context.
	// The Result() path (no rawOut, no textOut) should exit via ctx.Err() check.
	ctx, cancel := context.WithCancel(t.Context())

	ch := make(chan provider.StreamChunk, 10)
	// Send some chunks, then cancel before the channel closes.
	ch <- provider.StreamChunk{Type: provider.ChunkText, Text: "hello"}

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return &provider.StreamResult{Stream: ch}, nil
		},
	}

	stream, err := StreamText(ctx, model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	// Cancel context to trigger the ctx.Err() early exit in consume.
	cancel()
	// Close the channel so consume can drain.
	close(ch)

	result := stream.Result()
	// Result should have at most the text sent before cancel.
	_ = result
}
