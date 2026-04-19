package goai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
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

func TestStreamText_OnRequestCtx(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "ok"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}},
			), nil
		},
	}

	var gotCtx context.Context
	stream, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithOnRequest(func(info RequestInfo) {
			gotCtx = info.Ctx
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Result()
	if gotCtx == nil {
		t.Fatal("RequestInfo.Ctx was nil in StreamText single-step path")
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
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 400 {
		t.Errorf("StatusCode = %d, want 400", apiErr.StatusCode)
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

	// provider.ToolDefinition should not have Execute - verify by checking fields exist.
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
	// Model always requests tool calls - should stop at MaxSteps.
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
					if p.Type == provider.PartToolResult && p.ToolOutput == "error: goai: unknown tool" {
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
			// No Execute - tool definitions only, loop managed externally.
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

func TestStreamText_WithTimeout_ErrorReturnsAPIError(t *testing.T) {
	// StreamText with Timeout: when DoStream fails, error propagates correctly.
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

func TestExecuteToolsParallel_ContextCancelled(t *testing.T) {
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

	// With parallel execution, all tools run via goroutines and complete.
	// The cancelled context is passed to Execute, but the tool ignores it.
	msgs, _ := executeToolsParallel(ctx, calls, toolMap, 1, toolHooks{})
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages (parallel execution completes all), got %d", len(msgs))
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
	// Context was cancelled, so consume must exit early via ctx.Err().
	// Either the text chunk was drained before cancel (result.Text == "hello")
	// or the context error short-circuited before draining.
	if result.Text != "" && result.Text != "hello" {
		t.Errorf("Result.Text = %q, want either empty or %q (partial text before cancel)", result.Text, "hello")
	}
	// stream.Err() must reflect context cancellation.
	if err := stream.Err(); err == nil {
		t.Error("expected Err() to return context error, got nil")
	}
}

func TestStreamText_Error_OnResponseFired(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, &APIError{Message: "too many requests", StatusCode: 429, IsRetryable: false}
		},
	}

	var capturedResponse ResponseInfo
	_, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithMaxRetries(0),
		WithOnResponse(func(info ResponseInfo) {
			capturedResponse = info
		}),
	)
	if err == nil {
		t.Fatal("expected error from StreamText")
	}
	if capturedResponse.Error == nil {
		t.Error("OnResponse not called or ResponseInfo.Error is nil")
	}
	if capturedResponse.StatusCode != 429 {
		t.Errorf("ResponseInfo.StatusCode = %d, want 429", capturedResponse.StatusCode)
	}
}

// TestStreamText_OnResponse_StatusCodeOnStreamError verifies that when a
// ChunkError carrying an *APIError arrives during streaming, the OnResponse
// hook receives the correct StatusCode extracted via errors.As.
func TestStreamText_OnResponse_StatusCodeOnStreamError(t *testing.T) {
	apiErr := &APIError{StatusCode: 429, Message: "rate limited", IsRetryable: false}
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "partial"},
				provider.StreamChunk{Type: provider.ChunkError, Error: apiErr},
			), nil
		},
	}

	var capturedResponse ResponseInfo
	stream, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithMaxRetries(0),
		WithOnResponse(func(info ResponseInfo) {
			capturedResponse = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	// Consume the stream so the deferred OnResponse fires.
	stream.Result()

	if capturedResponse.StatusCode != 429 {
		t.Errorf("ResponseInfo.StatusCode = %d, want 429", capturedResponse.StatusCode)
	}
	if capturedResponse.Error == nil {
		t.Error("ResponseInfo.Error is nil, want non-nil")
	}
}

// TestStreamText_ProviderMetadata verifies that nested providerMetadata
// (OpenAI Chat Completions convention) flows from ChunkFinish.Metadata through
// consume() into TextResult.ProviderMetadata and StepResult.ProviderMetadata.
// The provider-level counterpart is TestParseStream_ProviderMetadataInFinish
// in internal/openaicompat/, which verifies the codec emits the right chunk shape.
func TestStreamText_ProviderMetadata(t *testing.T) {
	wantMeta := map[string]map[string]any{
		"openai": {"logprobs": []float64{-0.1, -0.2}},
	}

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hello"},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Usage:        provider.Usage{InputTokens: 5, OutputTokens: 1},
					Metadata: map[string]any{
						"providerMetadata": wantMeta,
					},
				},
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

	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata is nil, want non-nil")
	}
	openaiMeta, ok := result.ProviderMetadata["openai"]
	if !ok {
		t.Fatal("missing 'openai' key in ProviderMetadata")
	}
	if openaiMeta["logprobs"] == nil {
		t.Error("missing 'logprobs' in openai ProviderMetadata")
	}

	// Also check StepResult propagation.
	if len(result.Steps) == 0 {
		t.Fatal("no steps in result")
	}
	if result.Steps[0].ProviderMetadata == nil {
		t.Error("StepResult.ProviderMetadata is nil")
	}
}

func TestStreamText_ProviderMetadata_ResponseLevel(t *testing.T) {
	// Anthropic convention: flat metadata keys go into Response.ProviderMetadata.
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hi"},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Metadata: map[string]any{
						"iterations": 3,
						"container":  "abc",
					},
				},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("test"))
	if err != nil {
		t.Fatal(err)
	}
	for range stream.TextStream() {
	}
	result := stream.Result()

	if result.Response.ProviderMetadata == nil {
		t.Fatal("Response.ProviderMetadata is nil")
	}
	if result.Response.ProviderMetadata["iterations"] != 3 {
		t.Errorf("iterations = %v, want 3", result.Response.ProviderMetadata["iterations"])
	}
	if result.Response.ProviderMetadata["container"] != "abc" {
		t.Errorf("container = %v, want abc", result.Response.ProviderMetadata["container"])
	}
}

func TestStreamText_ProviderMetadata_BothConventions(t *testing.T) {
	// A chunk that carries both nested providerMetadata (OpenAI) and flat keys (Anthropic).
	wantNested := map[string]map[string]any{
		"openai": {"logprobs": true},
	}

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hi"},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Metadata: map[string]any{
						"providerMetadata": wantNested,
						"iterations":       5,
					},
				},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("test"))
	if err != nil {
		t.Fatal(err)
	}
	for range stream.TextStream() {
	}
	result := stream.Result()

	// Nested convention → TextResult.ProviderMetadata
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata is nil")
	}
	if _, ok := result.ProviderMetadata["openai"]; !ok {
		t.Error("missing 'openai' key in ProviderMetadata")
	}

	// Flat convention → Response.ProviderMetadata
	if result.Response.ProviderMetadata == nil {
		t.Fatal("Response.ProviderMetadata is nil")
	}
	if result.Response.ProviderMetadata["iterations"] != 5 {
		t.Errorf("iterations = %v, want 5", result.Response.ProviderMetadata["iterations"])
	}

	// Flat loop must NOT include the "providerMetadata" key.
	if _, ok := result.Response.ProviderMetadata["providerMetadata"]; ok {
		t.Error("providerMetadata key should not leak into Response.ProviderMetadata")
	}
}

func TestStreamText_ChunkStepFinish_ProviderMetadata(t *testing.T) {
	// ChunkStepFinish and ChunkFinish share the same case in consume().
	// Verify that metadata from ChunkStepFinish is captured and then
	// overwritten by ChunkFinish (last-write-wins, matching non-streaming).
	stepMeta := map[string]map[string]any{
		"openai": {"step": "first"},
	}
	finishMeta := map[string]map[string]any{
		"openai": {"step": "final"},
	}

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "step1"},
				provider.StreamChunk{
					Type:         provider.ChunkStepFinish,
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 5, OutputTokens: 2},
					Metadata: map[string]any{
						"providerMetadata": stepMeta,
					},
				},
				provider.StreamChunk{Type: provider.ChunkText, Text: " step2"},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Usage:        provider.Usage{InputTokens: 8, OutputTokens: 4},
					Metadata: map[string]any{
						"providerMetadata": finishMeta,
					},
				},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("multi-step"))
	if err != nil {
		t.Fatal(err)
	}
	for range stream.TextStream() {
	}
	result := stream.Result()

	// Final metadata should be from ChunkFinish (last-write-wins).
	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata is nil")
	}
	openaiMeta := result.ProviderMetadata["openai"]
	if openaiMeta["step"] != "final" {
		t.Errorf("step = %v, want 'final' (last-write-wins)", openaiMeta["step"])
	}
}

func TestStreamText_FlatMetadata_CacheTokens(t *testing.T) {
	// Bedrock puts cacheWriteInputTokens as a flat key on ChunkFinish.Metadata.
	// Verify it surfaces in Response.ProviderMetadata.
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "cached"},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Metadata: map[string]any{
						"cacheWriteInputTokens": 25,
					},
				},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model, WithPrompt("cache test"))
	if err != nil {
		t.Fatal(err)
	}
	for range stream.TextStream() {
	}
	result := stream.Result()

	if result.Response.ProviderMetadata == nil {
		t.Fatal("Response.ProviderMetadata is nil")
	}
	if result.Response.ProviderMetadata["cacheWriteInputTokens"] != 25 {
		t.Errorf("cacheWriteInputTokens = %v, want 25", result.Response.ProviderMetadata["cacheWriteInputTokens"])
	}
}

// TestGenerateText_ToolLoop_ToolChoiceRequiredResets verifies that
// WithToolChoice("required") is cleared after the first tool step so the model
// can produce a text response instead of looping on tool calls indefinitely.
func TestGenerateText_ToolLoop_ToolChoiceRequiredResets(t *testing.T) {
	step := 0
	var capturedToolChoices []string
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			step++
			capturedToolChoices = append(capturedToolChoices, params.ToolChoice)
			if step == 1 {
				return &provider.GenerateResult{
					FinishReason: provider.FinishToolCalls,
					ToolCalls: []provider.ToolCall{
						{ID: "c1", Name: "echo", Input: json.RawMessage(`{}`)},
					},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithTools(Tool{
			Name: "echo", Description: "echo",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithToolChoice("required"),
		WithMaxSteps(5),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "done" {
		t.Errorf("Text = %q, want done", result.Text)
	}
	if len(capturedToolChoices) != 2 {
		t.Fatalf("steps = %d, want 2", len(capturedToolChoices))
	}
	if capturedToolChoices[0] != "required" {
		t.Errorf("step 1 tool_choice = %q, want required", capturedToolChoices[0])
	}
	if capturedToolChoices[1] != "" {
		t.Errorf("step 2 tool_choice = %q, want empty (reset)", capturedToolChoices[1])
	}
}

func TestGenerateText_NilModel(t *testing.T) {
	_, err := GenerateText(t.Context(), nil, WithPrompt("hi"))
	if err == nil {
		t.Fatal("expected error for nil model")
	}
	if !strings.Contains(err.Error(), "model must not be nil") {
		t.Errorf("error = %q, want message containing %q", err.Error(), "model must not be nil")
	}
}

func TestStreamText_NilModel(t *testing.T) {
	_, err := StreamText(t.Context(), nil, WithPrompt("hi"))
	if err == nil {
		t.Fatal("expected error for nil model")
	}
	if !strings.Contains(err.Error(), "model must not be nil") {
		t.Errorf("error = %q, want message containing %q", err.Error(), "model must not be nil")
	}
}

func TestWithMaxSteps_ClampZero(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	_, err := GenerateText(t.Context(), model, WithPrompt("hi"), WithMaxSteps(0))
	if err != nil {
		t.Fatal(err)
	}
	// WithMaxSteps(0) should clamp to 1, so model is called exactly once.
	if callCount != 1 {
		t.Errorf("model called %d times, want 1 (MaxSteps clamped to 1)", callCount)
	}
}

func TestWithMaxRetries_ClampNegative(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			return nil, &APIError{Message: "service error", StatusCode: 503, IsRetryable: true}
		},
	}

	// Values below -1 are clamped to 0 (no retries).
	_, err := GenerateText(t.Context(), model, WithPrompt("hi"), WithMaxRetries(-2))
	if err == nil {
		t.Fatal("expected error")
	}
	if callCount != 1 {
		t.Errorf("model called %d times, want 1 (MaxRetries clamped to 0)", callCount)
	}
}

func TestWithMaxRetries_UnlimitedMinusOne(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount <= 3 {
				return nil, &APIError{Message: "service error", StatusCode: 503, IsRetryable: true}
			}
			return &provider.GenerateResult{Text: "ok"}, nil
		},
	}

	result, err := GenerateText(t.Context(), model, WithPrompt("hi"), WithMaxRetries(-1))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "ok" {
		t.Errorf("got text %q, want %q", result.Text, "ok")
	}
	if callCount != 4 {
		t.Errorf("model called %d times, want 4 (3 retries + 1 success)", callCount)
	}
}

func TestGenerateText_ContextCancelBetweenSteps(t *testing.T) {
	// Two-step tool loop: after step 1 tool execution, cancel the context.
	// The second model call should return the context error.
	ctx, cancel := context.WithCancel(t.Context())

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(ctx context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "work", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// On the second call, honor context cancellation.
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
				return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
			}
		},
	}

	_, err := GenerateText(ctx, model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithMaxRetries(0),
		WithTools(Tool{
			Name: "work",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				// Cancel the context inside tool execution, before step 2.
				cancel()
				return "result", nil
			},
		}),
	)
	if err == nil {
		t.Fatal("expected context error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

// --- StreamText Tool Loop tests ---

// TestStreamText_ToolLoop_TwoStep verifies a two-step tool loop via streaming:
// step 1 returns a tool call, step 2 returns text. Checks unified stream chunk
// order and Result() accumulation.
func TestStreamText_ToolLoop_TwoStep(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "Looking up..."},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "get_weather", ToolInput: `{"city":"NYC"}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}, Response: provider.ResponseMetadata{ID: "resp-1", Model: "test-model"}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Sunny in NYC"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 8}, Response: provider.ResponseMetadata{ID: "resp-2", Model: "test-model"}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("weather?"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				return "Sunny", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	var chunkTypes []provider.StreamChunkType
	for chunk := range stream.Stream() {
		chunkTypes = append(chunkTypes, chunk.Type)
	}

	// Expected order: ChunkText, ChunkToolCall, ChunkStepFinish(goai), ChunkText, ChunkStepFinish(goai), ChunkFinish
	want := []provider.StreamChunkType{
		provider.ChunkText, provider.ChunkToolCall, provider.ChunkStepFinish,
		provider.ChunkText, provider.ChunkStepFinish, provider.ChunkFinish,
	}
	if len(chunkTypes) != len(want) {
		t.Fatalf("chunk types = %v, want %v", chunkTypes, want)
	}
	for i := range want {
		if chunkTypes[i] != want[i] {
			t.Errorf("chunkTypes[%d] = %q, want %q", i, chunkTypes[i], want[i])
		}
	}

	result := stream.Result()
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}
	if result.Steps[0].Text != "Looking up..." {
		t.Errorf("Steps[0].Text = %q, want %q", result.Steps[0].Text, "Looking up...")
	}
	if result.Steps[1].Text != "Sunny in NYC" {
		t.Errorf("Steps[1].Text = %q, want %q", result.Steps[1].Text, "Sunny in NYC")
	}
	if result.TotalUsage.InputTokens != 30 || result.TotalUsage.OutputTokens != 13 {
		t.Errorf("TotalUsage = %+v, want InputTokens=30, OutputTokens=13", result.TotalUsage)
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want stop", result.FinishReason)
	}
	if result.Response.ID != "resp-2" {
		t.Errorf("Response.ID = %q, want resp-2 (last step wins)", result.Response.ID)
	}
	if result.Steps[0].Response.ID != "resp-1" {
		t.Errorf("Steps[0].Response.ID = %q, want resp-1", result.Steps[0].Response.ID)
	}
}

// TestStreamText_ToolLoop_MaxStepsReached verifies the loop stops at MaxSteps
// when the model keeps returning tool calls.
func TestStreamText_ToolLoop_MaxStepsReached(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: fmt.Sprintf("tc%d", n), ToolName: "loop", ToolInput: `{}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(2),
		WithTools(Tool{
			Name:    "loop",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "looping", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if int(callCount.Load()) != 2 {
		t.Errorf("model called %d times, want 2 (MaxSteps)", callCount.Load())
	}
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason = %q, want tool_calls", result.FinishReason)
	}
}

// TestStreamText_ToolLoop_ParallelExecution verifies that two tool calls in one
// step are both executed.
func TestStreamText_ToolLoop_ParallelExecution(t *testing.T) {
	var callCount atomic.Int32
	var execCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "work", ToolInput: `{"id":"a"}`},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc2", ToolName: "work", ToolInput: `{"id":"b"}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 3}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "work",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				execCount.Add(1)
				return "ok", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if execCount.Load() != 2 {
		t.Errorf("tools executed %d times, want 2", execCount.Load())
	}
	if result.Text != "done" {
		t.Errorf("Text = %q, want %q", result.Text, "done")
	}
}

// TestStreamText_ToolLoop_ContextCancel verifies clean shutdown when context
// is cancelled during step 1 drain. Uses a slow-producing channel so the
// cancel fires while drainStep is blocked trying to forward chunks.
func TestStreamText_ToolLoop_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())

	// Create a stream that produces many chunks (more than out buffer of 64).
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

	stream, err := StreamText(ctx, model,
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

	// Start consuming via Stream but stop reading to let the out buffer fill.
	rawCh := stream.Stream()
	<-rawCh // Read one chunk.

	// Let the out buffer fill, then cancel.
	time.Sleep(20 * time.Millisecond)
	cancel()

	// Drain remaining chunks.
	for range rawCh {
	}

	if stream.Err() == nil {
		t.Fatal("expected non-nil Err()")
	}
	if !errors.Is(stream.Err(), context.Canceled) {
		t.Errorf("Err() = %v, want context.Canceled", stream.Err())
	}
}

// TestStreamText_ToolLoop_HookOrdering verifies the sequence of hook calls
// for a 2-step tool loop including OnToolCallStart:
// OnRequest -> OnResponse -> OnStepFinish -> OnToolCallStart -> OnToolCall -> OnRequest -> ...
func TestStreamText_ToolLoop_HookOrdering(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 3}},
			), nil
		},
	}

	var mu sync.Mutex
	var sequence []string
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnRequest(func(_ RequestInfo) {
			mu.Lock()
			sequence = append(sequence, "OnRequest")
			mu.Unlock()
		}),
		WithOnResponse(func(_ ResponseInfo) {
			mu.Lock()
			sequence = append(sequence, "OnResponse")
			mu.Unlock()
		}),
		WithOnStepFinish(func(_ StepResult) {
			mu.Lock()
			sequence = append(sequence, "OnStepFinish")
			mu.Unlock()
		}),
		WithOnToolCallStart(func(_ ToolCallStartInfo) {
			mu.Lock()
			sequence = append(sequence, "OnToolCallStart")
			mu.Unlock()
		}),
		WithOnToolCall(func(_ ToolCallInfo) {
			mu.Lock()
			sequence = append(sequence, "OnToolCall")
			mu.Unlock()
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Consume fully.
	result := stream.Result()
	_ = result

	// Expected: drain -> OnResponse -> OnStepFinish -> OnToolCallStart -> OnToolCall -> next step
	want := []string{
		"OnRequest", "OnResponse", "OnStepFinish", "OnToolCallStart", "OnToolCall", // step 1
		"OnRequest", "OnResponse", "OnStepFinish", // step 2
	}
	mu.Lock()
	defer mu.Unlock()
	if len(sequence) != len(want) {
		t.Fatalf("sequence = %v, want %v", sequence, want)
	}
	for i := range want {
		if sequence[i] != want[i] {
			t.Errorf("sequence[%d] = %q, want %q", i, sequence[i], want[i])
		}
	}
}

// TestStreamText_ToolLoop_ToolError verifies that a tool returning an error
// passes the error message to the model and the loop continues.
func TestStreamText_ToolLoop_ToolError(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "fail", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			// Verify the error was passed as tool result.
			for _, msg := range params.Messages {
				for _, p := range msg.Content {
					if p.Type == provider.PartToolResult && p.ToolOutput == "error: something went wrong" {
						return streamFromChunks(
							provider.StreamChunk{Type: provider.ChunkText, Text: "handled error"},
							provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 3}},
						), nil
					}
				}
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "no error found"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
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

	result := stream.Result()
	if !strings.Contains(result.Text, "handled error") {
		t.Errorf("Text = %q, want containing 'handled error'", result.Text)
	}
}

// TestStreamText_ToolLoop_ResultAccumulation verifies multi-step Result()
// accumulation: Steps length, per-step text/usage, and aggregated TotalUsage.
func TestStreamText_ToolLoop_ResultAccumulation(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "Step1."},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Step2."},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 8}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
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

	result := stream.Result()
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}
	if result.Steps[0].Number != 1 {
		t.Errorf("Steps[0].Number = %d, want 1", result.Steps[0].Number)
	}
	if result.Steps[1].Number != 2 {
		t.Errorf("Steps[1].Number = %d, want 2", result.Steps[1].Number)
	}
	if result.Steps[0].Text != "Step1." {
		t.Errorf("Steps[0].Text = %q, want %q", result.Steps[0].Text, "Step1.")
	}
	if result.Steps[1].Text != "Step2." {
		t.Errorf("Steps[1].Text = %q, want %q", result.Steps[1].Text, "Step2.")
	}
	if result.Steps[0].Usage.InputTokens != 10 || result.Steps[0].Usage.OutputTokens != 5 {
		t.Errorf("Steps[0].Usage = %+v", result.Steps[0].Usage)
	}
	if result.Steps[1].Usage.InputTokens != 20 || result.Steps[1].Usage.OutputTokens != 8 {
		t.Errorf("Steps[1].Usage = %+v", result.Steps[1].Usage)
	}
	if result.TotalUsage.InputTokens != 30 || result.TotalUsage.OutputTokens != 13 {
		t.Errorf("TotalUsage = %+v, want InputTokens=30, OutputTokens=13", result.TotalUsage)
	}
	if result.Text != "Step1.Step2." {
		t.Errorf("Text = %q, want %q", result.Text, "Step1.Step2.")
	}
}

// TestStreamText_ToolLoop_TextStreamPath verifies consuming via TextStream()
// flows text from all steps and Result() returns correct data afterward.
func TestStreamText_ToolLoop_TextStreamPath(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "A"},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 5, OutputTokens: 3}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "B"},
				provider.StreamChunk{Type: provider.ChunkText, Text: "C"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
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

	var texts []string
	for text := range stream.TextStream() {
		texts = append(texts, text)
	}

	// Text from both steps should flow through.
	joined := strings.Join(texts, "")
	if joined != "ABC" {
		t.Errorf("TextStream joined = %q, want %q", joined, "ABC")
	}

	result := stream.Result()
	if result.Text != "ABC" {
		t.Errorf("Result().Text = %q, want %q", result.Text, "ABC")
	}
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}
	if result.TotalUsage.InputTokens != 15 {
		t.Errorf("TotalUsage.InputTokens = %d, want 15", result.TotalUsage.InputTokens)
	}
}

// TestStreamText_ToolLoop_ToolPanic verifies that a panicking tool Execute
// does not crash the process and an error appears in the tool result message.
func TestStreamText_ToolLoop_ToolPanic(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "panicker", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			// Verify panic was captured in tool result message.
			for _, msg := range params.Messages {
				for _, p := range msg.Content {
					if p.Type == provider.PartToolResult && strings.Contains(p.ToolOutput, "panicked") {
						return streamFromChunks(
							provider.StreamChunk{Type: provider.ChunkText, Text: "panic handled"},
							provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 3}},
						), nil
					}
				}
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "no panic found"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "panicker",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				panic("tool exploded")
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if !strings.Contains(result.Text, "panic handled") {
		t.Errorf("Text = %q, want containing 'panic handled'", result.Text)
	}
}

// TestStreamText_ToolLoop_UnknownTool verifies that a tool call referencing
// an unknown tool fires OnToolCall with ErrUnknownTool.
func TestStreamText_ToolLoop_UnknownTool(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "nonexistent", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			// Check tool result has unknown tool error.
			for _, msg := range params.Messages {
				for _, p := range msg.Content {
					if p.Type == provider.PartToolResult && p.ToolOutput == "error: goai: unknown tool" {
						return streamFromChunks(
							provider.StreamChunk{Type: provider.ChunkText, Text: "ok"},
							provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
						), nil
					}
				}
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "unexpected"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	var capturedInfo ToolCallInfo
	stream, err := StreamText(t.Context(), model,
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

	result := stream.Result()
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
	if capturedInfo.ToolName != "nonexistent" {
		t.Errorf("ToolCallInfo.ToolName = %q, want %q", capturedInfo.ToolName, "nonexistent")
	}
	if !errors.Is(capturedInfo.Error, ErrUnknownTool) {
		t.Errorf("ToolCallInfo.Error = %v, want ErrUnknownTool", capturedInfo.Error)
	}
}

// TestStreamText_ToolLoop_ToolChoiceReset verifies that ToolChoice is cleared
// after step 1 so the model can freely respond on subsequent steps.
func TestStreamText_ToolLoop_ToolChoiceReset(t *testing.T) {
	var mu sync.Mutex
	var capturedToolChoices []string
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			mu.Lock()
			capturedToolChoices = append(capturedToolChoices, params.ToolChoice)
			mu.Unlock()
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "echo", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 15, OutputTokens: 3}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithToolChoice("required"),
		WithTools(Tool{
			Name:    "echo",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	_ = result
	mu.Lock()
	defer mu.Unlock()
	if len(capturedToolChoices) != 2 {
		t.Fatalf("steps = %d, want 2", len(capturedToolChoices))
	}
	if capturedToolChoices[0] != "required" {
		t.Errorf("step 1 tool_choice = %q, want required", capturedToolChoices[0])
	}
	if capturedToolChoices[1] != "" {
		t.Errorf("step 2 tool_choice = %q, want empty (reset)", capturedToolChoices[1])
	}
}

// TestStreamText_ToolLoop_Step1DoStreamFailure verifies that when DoStream
// fails on step 1, StreamText returns (nil, error).
func TestStreamText_ToolLoop_Step1DoStreamFailure(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, &APIError{Message: "bad request", StatusCode: 400}
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
	)
	if stream != nil {
		t.Error("expected nil stream on step 1 failure")
	}
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 400 {
		t.Errorf("StatusCode = %d, want 400", apiErr.StatusCode)
	}
}

// TestStreamText_ToolLoop_SingleStepDispatch verifies that MaxSteps=1 with
// executable tools dispatches to the single-step path and tools are NOT executed.
func TestStreamText_ToolLoop_SingleStepDispatch(t *testing.T) {
	var execCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(1),
		WithTools(Tool{
			Name: "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				execCount.Add(1)
				return "ok", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if execCount.Load() != 0 {
		t.Errorf("tool executed %d times, want 0 (single-step, no loop)", execCount.Load())
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "tool" {
		t.Errorf("ToolCalls[0].Name = %q, want %q", result.ToolCalls[0].Name, "tool")
	}
}

// TestStreamText_ToolLoop_OnToolCallStart verifies that OnToolCallStart fires
// before Execute with correct fields.
func TestStreamText_ToolLoop_OnToolCallStart(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "myTool", ToolInput: `{"key":"val"}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 15, OutputTokens: 3}},
			), nil
		},
	}

	var startInfo ToolCallStartInfo
	var startFiredBeforeExec atomic.Bool
	var execFired atomic.Bool
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "myTool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				if !execFired.Load() {
					// Check if OnToolCallStart already fired.
					startFiredBeforeExec.Store(startInfo.ToolCallID != "")
				}
				execFired.Store(true)
				return "result", nil
			},
		}),
		WithOnToolCallStart(func(info ToolCallStartInfo) {
			startInfo = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	_ = stream.Result()
	if startInfo.ToolCallID != "tc1" {
		t.Errorf("ToolCallID = %q, want tc1", startInfo.ToolCallID)
	}
	if startInfo.ToolName != "myTool" {
		t.Errorf("ToolName = %q, want myTool", startInfo.ToolName)
	}
	if startInfo.Step != 1 {
		t.Errorf("Step = %d, want 1", startInfo.Step)
	}
	if string(startInfo.Input) != `{"key":"val"}` {
		t.Errorf("Input = %q, want %q", string(startInfo.Input), `{"key":"val"}`)
	}
	if !startFiredBeforeExec.Load() {
		t.Error("OnToolCallStart did not fire before Execute")
	}
}

// TestStreamText_ToolLoop_ChunkStepFinishDisambiguation verifies that
// provider-internal ChunkStepFinish (without stepSource=goai) IS forwarded
// to the consumer and does NOT have Metadata["stepSource"]="goai".
func TestStreamText_ToolLoop_ChunkStepFinishDisambiguation(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "thinking..."},
					// Provider-internal step finish (e.g., Anthropic thinking step)
					provider.StreamChunk{Type: provider.ChunkStepFinish, FinishReason: provider.FinishStop,
						Metadata: map[string]any{"internal": true}},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 15, OutputTokens: 3}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
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

	var providerStepFinishes []provider.StreamChunk
	var goaiStepFinishes []provider.StreamChunk
	for chunk := range stream.Stream() {
		if chunk.Type == provider.ChunkStepFinish {
			if stepSource, _ := chunk.Metadata["stepSource"].(string); stepSource == "goai" {
				goaiStepFinishes = append(goaiStepFinishes, chunk)
			} else {
				providerStepFinishes = append(providerStepFinishes, chunk)
			}
		}
	}

	// Provider step finish should be forwarded (1 from step 1 provider stream).
	if len(providerStepFinishes) != 1 {
		t.Errorf("provider ChunkStepFinish count = %d, want 1", len(providerStepFinishes))
	}
	if providerStepFinishes[0].Metadata["internal"] != true {
		t.Error("provider ChunkStepFinish missing internal metadata")
	}

	// GoAI step finishes: 2 (one per step in the loop).
	if len(goaiStepFinishes) != 2 {
		t.Errorf("goai ChunkStepFinish count = %d, want 2", len(goaiStepFinishes))
	}
}

// TestStreamText_ToolLoop_ReasoningExclusion verifies that ChunkReasoning
// content is included in Result().Text (backward compat) but excluded from
// Result().Steps[n].Text.
func TestStreamText_ToolLoop_ReasoningExclusion(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkReasoning, Text: "thinking..."},
					provider.StreamChunk{Type: provider.ChunkText, Text: "answer"},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "final"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 3}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
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

	result := stream.Result()
	// Global text includes reasoning (backward compat).
	if result.Text != "thinking...answerfinal" {
		t.Errorf("Text = %q, want %q", result.Text, "thinking...answerfinal")
	}
	// Step text excludes reasoning.
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}
	if result.Steps[0].Text != "answer" {
		t.Errorf("Steps[0].Text = %q, want %q (reasoning excluded)", result.Steps[0].Text, "answer")
	}
	if result.Steps[1].Text != "final" {
		t.Errorf("Steps[1].Text = %q, want %q", result.Steps[1].Text, "final")
	}
}

// TestStreamText_ToolLoop_PartialStepsOnError verifies that when DoStream
// fails on step 2, Result().Steps has step 1 data and Err() returns the error.
func TestStreamText_ToolLoop_PartialStepsOnError(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "step1"},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return nil, &APIError{Message: "server error", StatusCode: 500}
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithMaxRetries(0),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if stream.Err() == nil {
		t.Fatal("expected non-nil Err()")
	}
	var apiErr *APIError
	if !errors.As(stream.Err(), &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", stream.Err(), stream.Err())
	}
	if apiErr.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", apiErr.StatusCode)
	}
	// Step 1 data should still be present.
	if len(result.Steps) < 1 {
		t.Fatal("expected at least 1 step in partial result")
	}
	if result.Steps[0].Text != "step1" {
		t.Errorf("Steps[0].Text = %q, want %q", result.Steps[0].Text, "step1")
	}
}

// TestStreamText_ToolLoop_NoExecuteNoLoop verifies that tools without Execute
// functions + MaxSteps=3 dispatches to the single-step path.
func TestStreamText_ToolLoop_NoExecuteNoLoop(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			callCount.Add(1)
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "read", ToolInput: `{}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "read",
			Description: "Read a file",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			// No Execute: definition only.
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if callCount.Load() != 1 {
		t.Errorf("model called %d times, want 1 (no loop without Execute)", callCount.Load())
	}
	if len(result.ToolCalls) != 1 {
		t.Errorf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
}

// TestStreamText_SingleStep_ReasoningExclusion verifies that in the single-step
// path (MaxSteps=1, no tool loop), ChunkReasoning is included in Result().Text
// but excluded from Steps[0].Text and OnStepFinish's StepResult.Text.
func TestStreamText_SingleStep_ReasoningExclusion(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkReasoning, Text: "thinking..."},
				provider.StreamChunk{Type: provider.ChunkText, Text: "answer"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	var hookStepText string
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithOnStepFinish(func(step StepResult) {
			hookStepText = step.Text
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	// Global text includes reasoning (backward compat).
	if result.Text != "thinking...answer" {
		t.Errorf("Text = %q, want %q", result.Text, "thinking...answer")
	}
	// Step text excludes reasoning.
	if len(result.Steps) != 1 {
		t.Fatalf("Steps = %d, want 1", len(result.Steps))
	}
	if result.Steps[0].Text != "answer" {
		t.Errorf("Steps[0].Text = %q, want %q (reasoning excluded)", result.Steps[0].Text, "answer")
	}
	// OnStepFinish hook also excludes reasoning.
	if hookStepText != "answer" {
		t.Errorf("OnStepFinish Text = %q, want %q (reasoning excluded)", hookStepText, "answer")
	}
}

// TestStreamText_ToolLoop_OnToolCallStartPanic verifies that a panicking
// OnToolCallStart hook is safely recovered: the process does not crash, the
// tool result message contains the panic error, OnToolCall does NOT fire
// (because Execute never ran), and the model can respond on step 2.
func TestStreamText_ToolLoop_OnToolCallStartPanic(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "boom", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			// Step 2: verify the tool result contains the panic error.
			for _, msg := range params.Messages {
				for _, p := range msg.Content {
					if p.Type == provider.PartToolResult && strings.Contains(p.ToolOutput, "OnToolCallStart hook") && strings.Contains(p.ToolOutput, "panicked") {
						return streamFromChunks(
							provider.StreamChunk{Type: provider.ChunkText, Text: "recovered from hook panic"},
							provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 5}},
						), nil
					}
				}
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "no panic found"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	var onToolCallFired atomic.Bool
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "boom",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "should not run", nil
			},
		}),
		WithOnToolCallStart(func(_ ToolCallStartInfo) {
			panic("hook exploded")
		}),
		WithOnToolCall(func(_ ToolCallInfo) {
			onToolCallFired.Store(true)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if !strings.Contains(result.Text, "recovered from hook panic") {
		t.Errorf("Text = %q, want containing 'recovered from hook panic'", result.Text)
	}
	if onToolCallFired.Load() {
		t.Error("OnToolCall should NOT fire when OnToolCallStart panics (Execute never ran)")
	}
}

// TestStreamText_ToolLoop_Step2Retry verifies that withRetry works on step 2+
// DoStream calls: step 1 succeeds with a tool call, step 2 fails once with a
// retryable 500 error then succeeds on retry.
func TestStreamText_ToolLoop_Step2Retry(t *testing.T) {
	var doStreamCalls atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := doStreamCalls.Add(1)
			switch n {
			case 1:
				// Step 1: tool call.
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "myTool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			case 2:
				// Step 2, attempt 1: retryable error.
				return nil, &APIError{StatusCode: 500, Message: "server error", IsRetryable: true}
			default:
				// Step 2, attempt 2 (retry): success.
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "step2 ok"},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 3}},
				), nil
			}
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithMaxRetries(1),
		WithTools(Tool{
			Name: "myTool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "tool result", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if result.Text != "step2 ok" {
		t.Errorf("Text = %q, want %q", result.Text, "step2 ok")
	}
	if got := doStreamCalls.Load(); got != 3 {
		t.Errorf("DoStream call count = %d, want 3 (step1 + step2 fail + step2 retry)", got)
	}
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want stop", result.FinishReason)
	}
}

// TestStreamText_ToolLoop_UnknownToolOnToolCallStartPanic verifies that when
// an unknown tool's OnToolCallStart hook panics, it is silently recovered and
// OnToolCall still fires with ErrUnknownTool (independent recover wrapping for
// unknown tools).
func TestStreamText_ToolLoop_UnknownToolOnToolCallStartPanic(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, params provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "nonexistent_tool", ToolInput: `{"x":1}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			// Step 2: the model sees the error and responds.
			for _, msg := range params.Messages {
				for _, p := range msg.Content {
					if p.Type == provider.PartToolResult {
						return streamFromChunks(
							provider.StreamChunk{Type: provider.ChunkText, Text: "handled"},
							provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
						), nil
					}
				}
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "unexpected"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	var onToolCallFired atomic.Bool
	var capturedInfo ToolCallInfo
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "other_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnToolCallStart(func(_ ToolCallStartInfo) {
			panic("start hook boom")
		}),
		WithOnToolCall(func(info ToolCallInfo) {
			onToolCallFired.Store(true)
			capturedInfo = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if result.Text != "handled" {
		t.Errorf("Text = %q, want %q", result.Text, "handled")
	}
	if !onToolCallFired.Load() {
		t.Fatal("OnToolCall did not fire for unknown tool (expected it to fire even after OnToolCallStart panic)")
	}
	if !errors.Is(capturedInfo.Error, ErrUnknownTool) {
		t.Errorf("ToolCallInfo.Error = %v, want ErrUnknownTool", capturedInfo.Error)
	}
}

// TestGenerateText_ToolLoop_OnToolCallStart verifies that OnToolCallStart fires
// correctly in GenerateText (non-streaming) tool loop with correct fields and
// fires before Execute.
func TestGenerateText_ToolLoop_OnToolCallStart(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "myTool", Input: json.RawMessage(`{"key":"val"}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
		},
	}

	var startInfo ToolCallStartInfo
	var startFiredBeforeExec atomic.Bool
	var execFired atomic.Bool
	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "myTool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				if !execFired.Load() {
					// Check if OnToolCallStart already fired.
					startFiredBeforeExec.Store(startInfo.ToolCallID != "")
				}
				execFired.Store(true)
				return "result", nil
			},
		}),
		WithOnToolCallStart(func(info ToolCallStartInfo) {
			startInfo = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if result.Text != "done" {
		t.Errorf("Text = %q, want %q", result.Text, "done")
	}
	if startInfo.ToolCallID != "tc1" {
		t.Errorf("ToolCallID = %q, want tc1", startInfo.ToolCallID)
	}
	if startInfo.ToolName != "myTool" {
		t.Errorf("ToolName = %q, want myTool", startInfo.ToolName)
	}
	if startInfo.Step != 1 {
		t.Errorf("Step = %d, want 1", startInfo.Step)
	}
	if string(startInfo.Input) != `{"key":"val"}` {
		t.Errorf("Input = %q, want %q", string(startInfo.Input), `{"key":"val"}`)
	}
	if !startFiredBeforeExec.Load() {
		t.Error("OnToolCallStart did not fire before Execute")
	}
}

// TestStreamText_ToolLoop_ChunkToolCallDeltaForwarded verifies that
// ChunkToolCallStreamStart and ChunkToolCallDelta chunks are forwarded to the
// consumer, and that drainStep only accumulates the final ChunkToolCall (not deltas).
func TestStreamText_ToolLoop_ChunkToolCallDeltaForwarded(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCallStreamStart, ToolCallID: "tc1", ToolName: "myTool"},
					provider.StreamChunk{Type: provider.ChunkToolCallDelta, ToolCallID: "tc1", ToolInput: `{"key":`},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "myTool", ToolInput: `{"key":"val"}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 15, OutputTokens: 3}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "myTool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "result", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	var chunkTypes []provider.StreamChunkType
	for chunk := range stream.Stream() {
		chunkTypes = append(chunkTypes, chunk.Type)
	}

	// Expected: StreamStart, Delta, ToolCall, StepFinish(goai), Text, StepFinish(goai), Finish
	want := []provider.StreamChunkType{
		provider.ChunkToolCallStreamStart,
		provider.ChunkToolCallDelta,
		provider.ChunkToolCall,
		provider.ChunkStepFinish,
		provider.ChunkText,
		provider.ChunkStepFinish,
		provider.ChunkFinish,
	}
	if len(chunkTypes) != len(want) {
		t.Fatalf("chunk types = %v, want %v", chunkTypes, want)
	}
	for i := range want {
		if chunkTypes[i] != want[i] {
			t.Errorf("chunkTypes[%d] = %q, want %q", i, chunkTypes[i], want[i])
		}
	}

	result := stream.Result()
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}
	// drainStep should only accumulate the ChunkToolCall, not the delta.
	if len(result.Steps[0].ToolCalls) != 1 {
		t.Fatalf("Steps[0].ToolCalls = %d, want 1", len(result.Steps[0].ToolCalls))
	}
	if result.Steps[0].ToolCalls[0].ID != "tc1" {
		t.Errorf("ToolCalls[0].ID = %q, want tc1", result.Steps[0].ToolCalls[0].ID)
	}
	if result.Steps[1].Text != "done" {
		t.Errorf("Steps[1].Text = %q, want done", result.Steps[1].Text)
	}
}

// TestStreamText_ToolLoop_DrainStepMetadata verifies that drainStep correctly
// extracts sources, providerMetadata, and response providerMetadata from
// ChunkStepFinish and ChunkFinish metadata during multi-step streaming.
func TestStreamText_ToolLoop_DrainStepMetadata(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-meta",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "call"},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{
						Type:         provider.ChunkStepFinish,
						FinishReason: provider.FinishToolCalls,
						Usage:        provider.Usage{InputTokens: 5, OutputTokens: 3},
						Metadata: map[string]any{
							"sources":          []provider.Source{{URL: "http://example.com"}},
							"providerMetadata": map[string]map[string]any{"openai": {"logprobs": true}},
							"cacheTokens":      42,
						},
					},
					provider.StreamChunk{
						Type:         provider.ChunkFinish,
						FinishReason: provider.FinishToolCalls,
						Usage:        provider.Usage{InputTokens: 5, OutputTokens: 3},
						Response:     provider.ResponseMetadata{ID: "r1", Model: "m"},
						Metadata: map[string]any{
							"sources":          []provider.Source{{URL: "http://example.com"}},
							"providerMetadata": map[string]map[string]any{"openai": {"logprobs": true}},
							"cacheTokens":      42,
						},
					},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{
					Type:         provider.ChunkFinish,
					FinishReason: provider.FinishStop,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 4},
					Response:     provider.ResponseMetadata{ID: "r2", Model: "m"},
					Metadata: map[string]any{
						"sources":          []provider.Source{{URL: "http://example2.com"}},
						"providerMetadata": map[string]map[string]any{"openai": {"logprobs": false}},
					},
				},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
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

	result := stream.Result()
	if stream.Err() != nil {
		t.Fatalf("unexpected error: %v", stream.Err())
	}
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}

	// Step 1: sources from ChunkFinish metadata (last value wins).
	if len(result.Steps[0].Sources) == 0 {
		t.Fatal("Steps[0].Sources is empty, want at least one source")
	}
	if result.Steps[0].Sources[0].URL != "http://example.com" {
		t.Errorf("Steps[0].Sources[0].URL = %q, want http://example.com", result.Steps[0].Sources[0].URL)
	}

	// Step 1: providerMetadata from ChunkFinish metadata.
	if result.Steps[0].ProviderMetadata == nil {
		t.Fatal("Steps[0].ProviderMetadata is nil")
	}
	if val, ok := result.Steps[0].ProviderMetadata["openai"]["logprobs"].(bool); !ok || !val {
		t.Errorf("Steps[0].ProviderMetadata[openai][logprobs] = %v, want true", result.Steps[0].ProviderMetadata["openai"]["logprobs"])
	}

	// Step 1: Response.ProviderMetadata has cacheTokens from flat metadata.
	if result.Steps[0].Response.ProviderMetadata == nil {
		t.Fatal("Steps[0].Response.ProviderMetadata is nil")
	}
	if ct, ok := result.Steps[0].Response.ProviderMetadata["cacheTokens"].(int); !ok || ct != 42 {
		t.Errorf("Steps[0].Response.ProviderMetadata[cacheTokens] = %v, want 42", result.Steps[0].Response.ProviderMetadata["cacheTokens"])
	}

	// Final response has ID from step 2.
	if result.Response.ID != "r2" {
		t.Errorf("Response.ID = %q, want r2", result.Response.ID)
	}

	// Step 2: providerMetadata from ChunkFinish metadata (via drainStep, then
	// GoAI ChunkStepFinish forwarded to consume).
	if result.Steps[1].ProviderMetadata == nil {
		t.Fatal("Steps[1].ProviderMetadata is nil")
	}
	if val, ok := result.Steps[1].ProviderMetadata["openai"]["logprobs"].(bool); !ok || val {
		t.Errorf("Steps[1].ProviderMetadata[openai][logprobs] = %v, want false", result.Steps[1].ProviderMetadata["openai"]["logprobs"])
	}
}

// TestStreamText_ToolLoop_EmptyStepGuard verifies that the empty step guard
// (line 570) prevents phantom steps when a provider stream emits only a
// ChunkError then closes (no text, no tool calls, no finish reason).
func TestStreamText_ToolLoop_EmptyStepGuard(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-guard",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				// Step 1: only emits ChunkError then closes (no ChunkFinish with finishReason).
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkError, Error: fmt.Errorf("transient failure")},
				), nil
			}
			// Should never be reached due to empty step guard breaking the loop.
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "should not appear"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
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

	// Consume the stream.
	var gotError bool
	for chunk := range stream.Stream() {
		if chunk.Type == provider.ChunkError {
			gotError = true
		}
	}
	if !gotError {
		t.Error("expected ChunkError in stream")
	}

	result := stream.Result()
	// The stream should have the error forwarded but no phantom step.
	if len(result.Steps) != 0 {
		t.Errorf("Steps = %d, want 0 (empty step guard should prevent phantom step)", len(result.Steps))
	}
	if result.FinishReason != "" {
		t.Errorf("FinishReason = %q, want empty", result.FinishReason)
	}
}

// TestStreamText_ToolLoop_Step2DoStreamErrorWithOnResponse verifies that when
// step 2 DoStream fails, the OnResponse hook fires with Error and StatusCode,
// and the stream carries the error.
func TestStreamText_ToolLoop_Step2DoStreamErrorWithOnResponse(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-hook-err",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "step1"},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return nil, &APIError{StatusCode: 429, Message: "rate limited"}
		},
	}

	var mu sync.Mutex
	var capturedResp []ResponseInfo

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithMaxRetries(0),
		WithOnResponse(func(info ResponseInfo) {
			mu.Lock()
			capturedResp = append(capturedResp, info)
			mu.Unlock()
		}),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Consume stream.
	for range stream.Stream() {
	}

	if stream.Err() == nil {
		t.Fatal("expected non-nil Err()")
	}
	var apiErr *APIError
	if !errors.As(stream.Err(), &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", stream.Err(), stream.Err())
	}
	if apiErr.StatusCode != 429 {
		t.Errorf("StatusCode = %d, want 429", apiErr.StatusCode)
	}

	// OnResponse should have fired at least twice: once for step 1 success, once for step 2 error.
	mu.Lock()
	defer mu.Unlock()
	if len(capturedResp) < 2 {
		t.Fatalf("OnResponse fired %d times, want >= 2", len(capturedResp))
	}
	// Find the error response.
	var foundErr bool
	for _, r := range capturedResp {
		if r.Error != nil && r.StatusCode == 429 {
			foundErr = true
			break
		}
	}
	if !foundErr {
		t.Error("OnResponse never fired with Error and StatusCode=429")
	}

	// Step 1 data should still be present.
	result := stream.Result()
	if len(result.Steps) < 1 {
		t.Fatal("expected at least 1 step in partial result")
	}
	if result.Steps[0].Text != "step1" {
		t.Errorf("Steps[0].Text = %q, want step1", result.Steps[0].Text)
	}
}

// TestStreamText_ToolLoop_DrainStepChunkReasoningSource verifies that
// ChunkReasoning with a source in its Metadata is captured by drainStep.
func TestStreamText_ToolLoop_DrainStepChunkReasoningSource(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-reasoning-src",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{
						Type: provider.ChunkReasoning,
						Text: "thinking about it",
						Metadata: map[string]any{
							"source": provider.Source{URL: "http://reasoning-source.com", Type: "url"},
						},
					},
					provider.StreamChunk{Type: provider.ChunkText, Text: "answer"},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 15, OutputTokens: 3}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
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

	result := stream.Result()
	if stream.Err() != nil {
		t.Fatalf("unexpected error: %v", stream.Err())
	}
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}

	// Step 1 should have the reasoning source.
	if len(result.Steps[0].Sources) == 0 {
		t.Fatal("Steps[0].Sources is empty, want reasoning source")
	}
	foundReasoningSrc := false
	for _, s := range result.Steps[0].Sources {
		if s.URL == "http://reasoning-source.com" {
			foundReasoningSrc = true
			break
		}
	}
	if !foundReasoningSrc {
		t.Errorf("Steps[0].Sources = %v, want source with URL http://reasoning-source.com", result.Steps[0].Sources)
	}

	// Reasoning text should NOT be in Step text.
	if strings.Contains(result.Steps[0].Text, "thinking") {
		t.Errorf("Steps[0].Text = %q, should not contain reasoning text", result.Steps[0].Text)
	}
	if result.Steps[0].Text != "answer" {
		t.Errorf("Steps[0].Text = %q, want answer", result.Steps[0].Text)
	}
}

// --- Empty prompt validation tests ---

func TestGenerateText_EmptyPrompt(t *testing.T) {
	model := &mockModel{id: "test"}
	_, err := GenerateText(t.Context(), model)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "prompt or messages must not be empty") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestStreamText_EmptyPrompt(t *testing.T) {
	model := &mockModel{id: "test"}
	_, err := StreamText(t.Context(), model)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "prompt or messages must not be empty") {
		t.Errorf("unexpected error: %v", err)
	}
}

// --- Hook panic recovery tests ---

func TestStreamText_OnStepFinishPanic(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hello"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithOnStepFinish(func(_ StepResult) {
			panic("hook panic")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	for range stream.Stream() {
	}

	result := stream.Result()
	if stream.Err() != nil {
		t.Fatalf("unexpected error: %v", stream.Err())
	}
	if result.Text != "hello" {
		t.Errorf("Text = %q, want %q", result.Text, "hello")
	}
}

func TestStreamText_OnResponsePanic(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hello"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithOnResponse(func(_ ResponseInfo) {
			panic("hook panic")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	for range stream.Stream() {
	}

	result := stream.Result()
	if stream.Err() != nil {
		t.Fatalf("unexpected error: %v", stream.Err())
	}
	if result.Text != "hello" {
		t.Errorf("Text = %q, want %q", result.Text, "hello")
	}
}

func TestGenerateText_OnStepFinishPanic(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithOnStepFinish(func(_ StepResult) {
			panic("hook panic")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

func TestGenerateText_OnResponsePanic(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithOnResponse(func(_ ResponseInfo) {
			panic("hook panic")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want %q", result.Text, "ok")
	}
}

// --- buildToolMap empty name test ---

func TestBuildToolMap_EmptyToolName(t *testing.T) {
	called := false
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			called = true
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithTools(
			Tool{
				Name:    "",
				Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "bad", nil },
			},
			Tool{
				Name:    "valid",
				Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
			},
		),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Error("model generateFn was not called")
	}
	if result.Text != "done" {
		t.Errorf("Text = %q, want %q", result.Text, "done")
	}
}

// TestStreamText_ToolLoop_OnRequestPanic verifies that a panicking OnRequest
// hook on step 2+ is safely recovered: the process does not crash and the
// stream still delivers the final text from step 2.
func TestStreamText_ToolLoop_OnRequestPanic(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "mytool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "done"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 4}},
			), nil
		},
	}

	// hookCall counts OnRequest invocations; the hook only panics on step 2+
	// because step 1 fires OnRequest outside the goroutine without recovery.
	var hookCall atomic.Int32
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "mytool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "result", nil
			},
		}),
		WithOnRequest(func(_ RequestInfo) {
			if hookCall.Add(1) >= 2 {
				panic("onrequest hook exploded on step 2+")
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if !strings.Contains(result.Text, "done") {
		t.Errorf("Text = %q, want containing 'done'", result.Text)
	}
}

// TestStreamText_ToolLoop_OnResponsePanicOnError verifies that a panicking
// OnResponse hook fired after a step 2+ DoStream error is safely recovered
// and the stream reports the underlying error via stream.Err().
func TestStreamText_ToolLoop_OnResponsePanicOnError(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "mytool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return nil, errors.New("step 2 failed")
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "mytool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "result", nil
			},
		}),
		WithOnResponse(func(_ ResponseInfo) {
			panic("onresponse hook exploded")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	stream.Result()
	if stream.Err() == nil {
		t.Error("expected non-nil stream error from step 2 DoStream failure")
	}
}

// TestStreamText_ToolLoop_OnResponsePanicOnSuccess verifies that a panicking
// OnResponse hook fired after a successful step 2+ drain is safely recovered
// and the stream still delivers the final text.
func TestStreamText_ToolLoop_OnResponsePanicOnSuccess(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "mytool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "final"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "mytool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "result", nil
			},
		}),
		WithOnResponse(func(_ ResponseInfo) {
			panic("onresponse hook exploded")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if !strings.Contains(result.Text, "final") {
		t.Errorf("Text = %q, want containing 'final'", result.Text)
	}
}

// TestStreamText_ToolLoop_OnStepFinishPanic verifies that a panicking
// OnStepFinish hook on step 2+ is safely recovered and the stream still
// delivers the final text.
// TestOnToolCall_PanicIsolation verifies that a panicking OnToolCall hook
// does not prevent a second hook from firing.
func TestOnToolCall_PanicIsolation(t *testing.T) {
	var secondFired atomic.Bool
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "my_tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 8},
			}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "my_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnToolCall(func(_ ToolCallInfo) { panic("hook1 panic") }),
		WithOnToolCall(func(_ ToolCallInfo) { secondFired.Store(true) }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !secondFired.Load() {
		t.Error("second OnToolCall hook should fire even if first panics")
	}
}

// TestOnToolCallStart_PanicIsolation verifies that a panicking OnToolCallStart hook
// does not prevent a second hook from firing, and the tool does not execute.
func TestOnToolCallStart_PanicIsolation(t *testing.T) {
	var secondFired atomic.Bool
	var toolExecuted atomic.Bool
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "my_tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 8},
			}, nil
		},
	}

	_, _ = GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "my_tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				toolExecuted.Store(true)
				return "ok", nil
			},
		}),
		WithOnToolCallStart(func(_ ToolCallStartInfo) { panic("hook1 panic") }),
		WithOnToolCallStart(func(_ ToolCallStartInfo) { secondFired.Store(true) }),
	)
	if !secondFired.Load() {
		t.Error("second OnToolCallStart hook should fire even if first panics")
	}
	if toolExecuted.Load() {
		t.Error("tool should NOT execute when OnToolCallStart panics")
	}
}

// TestGenerateText_OnResponse_PanicIsolation verifies that in the GenerateText
// multi-step loop, a panicking OnResponse hook does not prevent the second hook
// from firing (recover-wrapped path).
func TestGenerateText_OnResponse_PanicIsolation(t *testing.T) {
	var secondFired atomic.Bool
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithOnResponse(func(_ ResponseInfo) { panic("hook1 panic") }),
		WithOnResponse(func(_ ResponseInfo) { secondFired.Store(true) }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Errorf("Text = %q, want ok", result.Text)
	}
	if !secondFired.Load() {
		t.Error("second OnResponse hook should fire even if first panics (GenerateText multi-step is recover-wrapped)")
	}
}

func TestStreamText_ToolLoop_OnStepFinishPanic(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "mytool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "final"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "mytool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "result", nil
			},
		}),
		WithOnStepFinish(func(_ StepResult) {
			panic("onstepfinish hook exploded")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if !strings.Contains(result.Text, "final") {
		t.Errorf("Text = %q, want containing 'final'", result.Text)
	}
}

// TestGenerateText_ToolLoop_EmptyFinishReason verifies that tool calls with
// empty finish_reason (common in MiniMax, Azure MaaS deepseek) are still executed.
func TestGenerateText_ToolLoop_EmptyFinishReason(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{{
						ID:    "tc1",
						Name:  "get_weather",
						Input: json.RawMessage(`{"city":"NYC"}`),
					}},
					FinishReason: "", // Bug: provider sends empty instead of "tool-calls"
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "Sunny in NYC",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
			}, nil
		},
	}

	toolExecuted := false
	result, err := GenerateText(t.Context(), model,
		WithPrompt("weather?"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				toolExecuted = true
				return "Sunny", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !toolExecuted {
		t.Error("tool was not executed despite tool calls being present")
	}
	if callCount != 2 {
		t.Errorf("model called %d times, want 2", callCount)
	}
	if result.Text != "Sunny in NYC" {
		t.Errorf("Text = %q, want %q", result.Text, "Sunny in NYC")
	}
}

// TestGenerateText_ToolLoop_StopFinishReason verifies that tool calls with
// finish_reason="stop" (instead of "tool-calls") are still executed.
func TestGenerateText_ToolLoop_StopFinishReason(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{{
						ID:    "tc1",
						Name:  "get_weather",
						Input: json.RawMessage(`{"city":"NYC"}`),
					}},
					FinishReason: provider.FinishStop, // Bug: provider sends "stop" instead of "tool-calls"
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "Done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
			}, nil
		},
	}

	toolExecuted := false
	result, err := GenerateText(t.Context(), model,
		WithPrompt("weather?"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				toolExecuted = true
				return "Sunny", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !toolExecuted {
		t.Error("tool was not executed despite tool calls being present")
	}
	if callCount != 2 {
		t.Errorf("model called %d times, want 2", callCount)
	}
	if result.Text != "Done" {
		t.Errorf("Text = %q", result.Text)
	}
}

// TestStreamText_ToolLoop_EmptyFinishReason verifies that streaming tool calls
// with empty finish_reason are still executed.
func TestStreamText_ToolLoop_EmptyFinishReason(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: "Looking up..."},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "get_weather", ToolInput: `{"city":"NYC"}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: "", Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}}, // empty finish_reason
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Sunny in NYC"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 8}},
			), nil
		},
	}

	toolExecuted := false
	stream, err := StreamText(t.Context(), model,
		WithPrompt("weather?"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				toolExecuted = true
				return "Sunny", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	for range stream.Stream() {
	}

	if !toolExecuted {
		t.Error("tool was not executed despite tool calls being present in stream")
	}
	result := stream.Result()
	if len(result.Steps) != 2 {
		t.Fatalf("Steps = %d, want 2", len(result.Steps))
	}
	if !strings.Contains(result.Text, "Sunny in NYC") {
		t.Errorf("Text = %q, want containing %q", result.Text, "Sunny in NYC")
	}
}

// TestToolCallIDFromContext verifies that the tool call ID is available
// inside the Execute function via ToolCallIDFromContext.
func TestToolCallIDFromContext(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{{
						ID:    "call_abc123",
						Name:  "my_tool",
						Input: json.RawMessage(`{}`),
					}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "Done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
			}, nil
		},
	}

	var capturedID string
	_, err := GenerateText(t.Context(), model,
		WithPrompt("test"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "my_tool",
			Description: "A test tool",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute: func(ctx context.Context, _ json.RawMessage) (string, error) {
				capturedID = ToolCallIDFromContext(ctx)
				return "ok", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if capturedID != "call_abc123" {
		t.Errorf("ToolCallIDFromContext = %q, want %q", capturedID, "call_abc123")
	}
}

// TestToolCallIDFromContext_Empty verifies that ToolCallIDFromContext
// returns empty string when called outside of tool execution.
func TestToolCallIDFromContext_Empty(t *testing.T) {
	id := ToolCallIDFromContext(t.Context())
	if id != "" {
		t.Errorf("ToolCallIDFromContext = %q, want empty", id)
	}
}

// --- ResponseMessages tests ---

// TestGenerateText_ResponseMessages_SingleStep verifies that a simple text
// generation (no tools) produces ResponseMessages with a single assistant
// message containing the generated text.
func TestGenerateText_ResponseMessages_SingleStep(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "Hello, world!",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 5, OutputTokens: 3},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	if len(result.ResponseMessages) != 1 {
		t.Fatalf("ResponseMessages length = %d, want 1", len(result.ResponseMessages))
	}

	msg := result.ResponseMessages[0]
	if msg.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", msg.Role, provider.RoleAssistant)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("ResponseMessages[0].Content length = %d, want 1", len(msg.Content))
	}
	if msg.Content[0].Type != provider.PartText {
		t.Errorf("ResponseMessages[0].Content[0].Type = %q, want %q", msg.Content[0].Type, provider.PartText)
	}
	if msg.Content[0].Text != "Hello, world!" {
		t.Errorf("ResponseMessages[0].Content[0].Text = %q, want %q", msg.Content[0].Text, "Hello, world!")
	}
}

// TestGenerateText_ResponseMessages_ToolLoop verifies that a two-step tool loop
// produces the correct sequence of response messages: assistant (tool calls),
// tool result, assistant (final text).
func TestGenerateText_ResponseMessages_ToolLoop(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, params provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "call-1", Name: "weather", Input: json.RawMessage(`{"city":"NYC"}`)},
					},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "The weather in NYC is sunny.",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 10},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("What's the weather in NYC?"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "Sunny, 72F", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: assistant (tool call), tool result, assistant (final text) = 3 messages
	if len(result.ResponseMessages) != 3 {
		t.Fatalf("ResponseMessages length = %d, want 3", len(result.ResponseMessages))
	}

	// Message 0: assistant with tool call
	assistantToolCall := result.ResponseMessages[0]
	if assistantToolCall.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", assistantToolCall.Role, provider.RoleAssistant)
	}
	hasToolCallPart := false
	for _, part := range assistantToolCall.Content {
		if part.Type == provider.PartToolCall {
			hasToolCallPart = true
			if part.ToolCallID != "call-1" {
				t.Errorf("tool call part ToolCallID = %q, want %q", part.ToolCallID, "call-1")
			}
			if part.ToolName != "weather" {
				t.Errorf("tool call part ToolName = %q, want %q", part.ToolName, "weather")
			}
			if string(part.ToolInput) != `{"city":"NYC"}` {
				t.Errorf("tool call ToolInput = %s, want %s", part.ToolInput, `{"city":"NYC"}`)
			}
		}
	}
	if !hasToolCallPart {
		t.Error("ResponseMessages[0] missing PartToolCall")
	}
	if len(assistantToolCall.Content) != 1 {
		t.Fatalf("expected exactly 1 part in assistant tool call message, got %d", len(assistantToolCall.Content))
	}

	// Message 1: tool result
	toolResult := result.ResponseMessages[1]
	if toolResult.Role != provider.RoleTool {
		t.Errorf("ResponseMessages[1].Role = %q, want %q", toolResult.Role, provider.RoleTool)
	}
	if len(toolResult.Content) < 1 {
		t.Fatalf("ResponseMessages[1].Content length = %d, want >= 1", len(toolResult.Content))
	}
	toolResultPart := toolResult.Content[0]
	if toolResultPart.Type != provider.PartToolResult {
		t.Errorf("ResponseMessages[1].Content[0].Type = %q, want %q", toolResultPart.Type, provider.PartToolResult)
	}
	if toolResultPart.ToolCallID != "call-1" {
		t.Errorf("tool result ToolCallID = %q, want %q", toolResultPart.ToolCallID, "call-1")
	}
	if toolResultPart.ToolName != "weather" {
		t.Errorf("tool result ToolName = %q, want %q", toolResultPart.ToolName, "weather")
	}
	if toolResultPart.ToolOutput != "Sunny, 72F" {
		t.Errorf("tool result ToolOutput = %q, want %q", toolResultPart.ToolOutput, "Sunny, 72F")
	}

	// Message 2: assistant with final text
	finalAssistant := result.ResponseMessages[2]
	if finalAssistant.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[2].Role = %q, want %q", finalAssistant.Role, provider.RoleAssistant)
	}
	hasTextPart := false
	for _, part := range finalAssistant.Content {
		if part.Type == provider.PartText {
			hasTextPart = true
			if part.Text != "The weather in NYC is sunny." {
				t.Errorf("final text part = %q, want %q", part.Text, "The weather in NYC is sunny.")
			}
		}
	}
	if !hasTextPart {
		t.Error("ResponseMessages[2] missing PartText with final text")
	}
}

// TestGenerateText_ResponseMessages_MultipleToolCalls verifies that when the
// model returns multiple parallel tool calls in a single step, all calls and
// their results appear in ResponseMessages.
func TestGenerateText_ResponseMessages_MultipleToolCalls(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "call-a", Name: "weather", Input: json.RawMessage(`{"city":"NYC"}`)},
						{ID: "call-b", Name: "weather", Input: json.RawMessage(`{"city":"LA"}`)},
					},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "NYC is sunny, LA is cloudy.",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 30, OutputTokens: 10},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("weather in NYC and LA?"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				var args struct{ City string }
				if err := json.Unmarshal(input, &args); err != nil {
					return "", err
				}
				if args.City == "NYC" {
					return "Sunny", nil
				}
				return "Cloudy", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: assistant (2 tool calls), tool results, assistant (final text) = 3 messages
	if len(result.ResponseMessages) != 3 {
		t.Fatalf("ResponseMessages length = %d, want 3", len(result.ResponseMessages))
	}

	// Message 0: assistant with both tool calls
	assistantMsg := result.ResponseMessages[0]
	if assistantMsg.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", assistantMsg.Role, provider.RoleAssistant)
	}
	toolCallParts := 0
	toolCallIDs := map[string]bool{}
	for _, part := range assistantMsg.Content {
		if part.Type == provider.PartToolCall {
			toolCallParts++
			toolCallIDs[part.ToolCallID] = true
		}
	}
	if toolCallParts != 2 {
		t.Errorf("assistant message has %d tool call parts, want 2", toolCallParts)
	}
	if !toolCallIDs["call-a"] || !toolCallIDs["call-b"] {
		t.Errorf("tool call IDs = %v, want call-a and call-b", toolCallIDs)
	}

	// Message 1: tool results for both calls
	toolMsg := result.ResponseMessages[1]
	if toolMsg.Role != provider.RoleTool {
		t.Errorf("ResponseMessages[1].Role = %q, want %q", toolMsg.Role, provider.RoleTool)
	}
	toolResultParts := 0
	resultIDs := map[string]bool{}
	resultOutputs := map[string]string{}
	for _, part := range toolMsg.Content {
		if part.Type == provider.PartToolResult {
			toolResultParts++
			resultIDs[part.ToolCallID] = true
			resultOutputs[part.ToolCallID] = part.ToolOutput
			if part.ToolName != "weather" {
				t.Errorf("tool result ToolName = %q, want %q", part.ToolName, "weather")
			}
		}
	}
	if toolResultParts != 2 {
		t.Errorf("tool message has %d tool result parts, want 2", toolResultParts)
	}
	if !resultIDs["call-a"] || !resultIDs["call-b"] {
		t.Errorf("tool result IDs = %v, want call-a and call-b", resultIDs)
	}
	if resultOutputs["call-a"] != "Sunny" {
		t.Errorf("tool result for call-a ToolOutput = %q, want %q", resultOutputs["call-a"], "Sunny")
	}
	if resultOutputs["call-b"] != "Cloudy" {
		t.Errorf("tool result for call-b ToolOutput = %q, want %q", resultOutputs["call-b"], "Cloudy")
	}

	// Message 2: assistant with final text
	finalMsg := result.ResponseMessages[2]
	if finalMsg.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[2].Role = %q, want %q", finalMsg.Role, provider.RoleAssistant)
	}
	hasText := false
	for _, part := range finalMsg.Content {
		if part.Type == provider.PartText && part.Text == "NYC is sunny, LA is cloudy." {
			hasText = true
		}
	}
	if !hasText {
		t.Error("ResponseMessages[2] missing expected final text")
	}
}

// TestStreamText_ResponseMessages_ToolLoop verifies that StreamText produces
// the same ResponseMessages structure as GenerateText for a two-step tool loop.
func TestStreamText_ResponseMessages_ToolLoop(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "get_weather", ToolInput: `{"city":"NYC"}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Sunny in NYC"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 8}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("weather?"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "Sunny", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Drain the stream.
	for range stream.Stream() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}

	result := stream.Result()

	// Expected: assistant (tool call), tool result, assistant (final text) = 3 messages
	if len(result.ResponseMessages) != 3 {
		t.Fatalf("ResponseMessages length = %d, want 3", len(result.ResponseMessages))
	}

	// Message 0: assistant with tool call
	if result.ResponseMessages[0].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", result.ResponseMessages[0].Role, provider.RoleAssistant)
	}
	hasToolCall := false
	for _, part := range result.ResponseMessages[0].Content {
		if part.Type == provider.PartToolCall {
			hasToolCall = true
			if part.ToolCallID != "tc1" {
				t.Errorf("tool call ToolCallID = %q, want %q", part.ToolCallID, "tc1")
			}
			if part.ToolName != "get_weather" {
				t.Errorf("tool call ToolName = %q, want %q", part.ToolName, "get_weather")
			}
			if string(part.ToolInput) != `{"city":"NYC"}` {
				t.Errorf("tool call ToolInput = %s, want %s", part.ToolInput, `{"city":"NYC"}`)
			}
		}
	}
	if !hasToolCall {
		t.Error("ResponseMessages[0] missing PartToolCall")
	}
	if len(result.ResponseMessages[0].Content) != 1 {
		t.Fatalf("expected exactly 1 part in assistant tool call message, got %d", len(result.ResponseMessages[0].Content))
	}

	// Message 1: tool result
	if result.ResponseMessages[1].Role != provider.RoleTool {
		t.Errorf("ResponseMessages[1].Role = %q, want %q", result.ResponseMessages[1].Role, provider.RoleTool)
	}
	if len(result.ResponseMessages[1].Content) < 1 {
		t.Fatalf("ResponseMessages[1].Content length = %d, want >= 1", len(result.ResponseMessages[1].Content))
	}
	trp := result.ResponseMessages[1].Content[0]
	if trp.Type != provider.PartToolResult {
		t.Errorf("tool result Type = %q, want %q", trp.Type, provider.PartToolResult)
	}
	if trp.ToolCallID != "tc1" {
		t.Errorf("tool result ToolCallID = %q, want %q", trp.ToolCallID, "tc1")
	}
	if trp.ToolName != "get_weather" {
		t.Errorf("tool result ToolName = %q, want %q", trp.ToolName, "get_weather")
	}
	if trp.ToolOutput != "Sunny" {
		t.Errorf("tool result ToolOutput = %q, want %q", trp.ToolOutput, "Sunny")
	}

	// Message 2: assistant with final text
	if result.ResponseMessages[2].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[2].Role = %q, want %q", result.ResponseMessages[2].Role, provider.RoleAssistant)
	}
	hasText := false
	for _, part := range result.ResponseMessages[2].Content {
		if part.Type == provider.PartText && part.Text == "Sunny in NYC" {
			hasText = true
		}
	}
	if !hasText {
		t.Error("ResponseMessages[2] missing expected final text 'Sunny in NYC'")
	}
}

// TestGenerateText_ResponseMessages_NoToolExecute verifies that when tools are
// provided without Execute functions, ResponseMessages contains a single
// assistant message with both text and tool call parts.
func TestGenerateText_ResponseMessages_NoToolExecute(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text: "I'll call the tool.",
				ToolCalls: []provider.ToolCall{
					{ID: "call-1", Name: "search", Input: json.RawMessage(`{"q":"golang"}`)},
				},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 8},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("search for golang"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "search",
			Description: "Search the web",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
			// No Execute function - tool loop should not proceed.
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// With no Execute, the loop stops after one step. ResponseMessages should
	// contain a single assistant message with both text and tool call parts.
	if len(result.ResponseMessages) != 1 {
		t.Fatalf("ResponseMessages length = %d, want 1", len(result.ResponseMessages))
	}

	msg := result.ResponseMessages[0]
	if msg.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", msg.Role, provider.RoleAssistant)
	}

	hasText := false
	hasToolCall := false
	for _, part := range msg.Content {
		switch part.Type {
		case provider.PartText:
			hasText = true
			if part.Text != "I'll call the tool." {
				t.Errorf("text part = %q, want %q", part.Text, "I'll call the tool.")
			}
		case provider.PartToolCall:
			hasToolCall = true
			if part.ToolCallID != "call-1" {
				t.Errorf("tool call ToolCallID = %q, want %q", part.ToolCallID, "call-1")
			}
			if part.ToolName != "search" {
				t.Errorf("tool call ToolName = %q, want %q", part.ToolName, "search")
			}
		}
	}
	if !hasText {
		t.Error("ResponseMessages[0] missing PartText")
	}
	if !hasToolCall {
		t.Error("ResponseMessages[0] missing PartToolCall")
	}
}

// TestGenerateText_ResponseMessages_ThreeStepToolLoop verifies that a three-step
// tool loop (tool A -> tool B -> final text) produces the correct sequence of
// response messages: assistant(A), tool(A), assistant(B), tool(B), assistant(text).
func TestGenerateText_ResponseMessages_ThreeStepToolLoop(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			switch callCount {
			case 1:
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "call-a", Name: "lookup", Input: json.RawMessage(`{"key":"alpha"}`)},
					},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			case 2:
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "call-b", Name: "lookup", Input: json.RawMessage(`{"key":"beta"}`)},
					},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 20, OutputTokens: 5},
				}, nil
			default:
				return &provider.GenerateResult{
					Text:         "Done with both lookups.",
					FinishReason: provider.FinishStop,
					Usage:        provider.Usage{InputTokens: 30, OutputTokens: 10},
				}, nil
			}
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("look up alpha then beta"),
		WithMaxSteps(5),
		WithTools(Tool{
			Name:        "lookup",
			Description: "Look up a key",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"key":{"type":"string"}}}`),
			Execute: func(_ context.Context, input json.RawMessage) (string, error) {
				var args struct{ Key string }
				if err := json.Unmarshal(input, &args); err != nil {
					return "", err
				}
				return "result-" + args.Key, nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: assistant(call-a), tool(result-a), assistant(call-b), tool(result-b), assistant(text) = 5 messages
	if len(result.ResponseMessages) != 5 {
		t.Fatalf("ResponseMessages length = %d, want 5", len(result.ResponseMessages))
	}

	// Message 0: assistant with tool call A
	if result.ResponseMessages[0].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", result.ResponseMessages[0].Role, provider.RoleAssistant)
	}
	if len(result.ResponseMessages[0].Content) == 0 {
		t.Fatalf("ResponseMessages[0].Content is empty")
	}
	if result.ResponseMessages[0].Content[0].Type != provider.PartToolCall || result.ResponseMessages[0].Content[0].ToolCallID != "call-a" {
		t.Errorf("ResponseMessages[0] should contain tool call A, got %+v", result.ResponseMessages[0].Content)
	}

	// Message 1: tool result for A
	if result.ResponseMessages[1].Role != provider.RoleTool {
		t.Errorf("ResponseMessages[1].Role = %q, want %q", result.ResponseMessages[1].Role, provider.RoleTool)
	}
	if len(result.ResponseMessages[1].Content) == 0 {
		t.Fatalf("ResponseMessages[1].Content is empty")
	}
	if result.ResponseMessages[1].Content[0].ToolOutput != "result-alpha" {
		t.Errorf("ResponseMessages[1] tool output = %q, want %q", result.ResponseMessages[1].Content[0].ToolOutput, "result-alpha")
	}

	// Message 2: assistant with tool call B
	if result.ResponseMessages[2].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[2].Role = %q, want %q", result.ResponseMessages[2].Role, provider.RoleAssistant)
	}
	if len(result.ResponseMessages[2].Content) == 0 {
		t.Fatalf("ResponseMessages[2].Content is empty")
	}
	if result.ResponseMessages[2].Content[0].Type != provider.PartToolCall || result.ResponseMessages[2].Content[0].ToolCallID != "call-b" {
		t.Errorf("ResponseMessages[2] should contain tool call B, got %+v", result.ResponseMessages[2].Content)
	}

	// Message 3: tool result for B
	if result.ResponseMessages[3].Role != provider.RoleTool {
		t.Errorf("ResponseMessages[3].Role = %q, want %q", result.ResponseMessages[3].Role, provider.RoleTool)
	}
	if len(result.ResponseMessages[3].Content) == 0 {
		t.Fatalf("ResponseMessages[3].Content is empty")
	}
	if result.ResponseMessages[3].Content[0].ToolOutput != "result-beta" {
		t.Errorf("ResponseMessages[3] tool output = %q, want %q", result.ResponseMessages[3].Content[0].ToolOutput, "result-beta")
	}

	// Message 4: assistant with final text
	if result.ResponseMessages[4].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[4].Role = %q, want %q", result.ResponseMessages[4].Role, provider.RoleAssistant)
	}
	hasText := false
	for _, part := range result.ResponseMessages[4].Content {
		if part.Type == provider.PartText && part.Text == "Done with both lookups." {
			hasText = true
		}
	}
	if !hasText {
		t.Error("ResponseMessages[4] missing expected final text")
	}
}

// TestGenerateText_ResponseMessages_ToolError verifies that when a tool's Execute
// function returns an error, the error text still appears in ResponseMessages as
// a tool result message.
func TestGenerateText_ResponseMessages_ToolError(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "call-err", Name: "flaky", Input: json.RawMessage(`{}`)},
					},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "I see the tool failed.",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 20, OutputTokens: 8},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("try the flaky tool"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "flaky",
			Description: "A tool that fails",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "", fmt.Errorf("something went wrong")
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: assistant(tool call), tool(error result), assistant(final text) = 3 messages
	if len(result.ResponseMessages) != 3 {
		t.Fatalf("ResponseMessages length = %d, want 3", len(result.ResponseMessages))
	}

	// Message 0: assistant with tool call
	if result.ResponseMessages[0].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", result.ResponseMessages[0].Role, provider.RoleAssistant)
	}
	if len(result.ResponseMessages[0].Content) == 0 {
		t.Fatalf("ResponseMessages[0].Content is empty")
	}
	if result.ResponseMessages[0].Content[0].Type != provider.PartToolCall {
		t.Errorf("ResponseMessages[0].Content[0].Type = %q, want %q", result.ResponseMessages[0].Content[0].Type, provider.PartToolCall)
	}
	if result.ResponseMessages[0].Content[0].ToolCallID != "call-err" {
		t.Errorf("ResponseMessages[0].Content[0].ToolCallID = %q, want %q", result.ResponseMessages[0].Content[0].ToolCallID, "call-err")
	}
	if result.ResponseMessages[0].Content[0].ToolName != "flaky" {
		t.Errorf("ResponseMessages[0].Content[0].ToolName = %q, want %q", result.ResponseMessages[0].Content[0].ToolName, "flaky")
	}

	// Message 1: tool result with error text
	toolResult := result.ResponseMessages[1]
	if toolResult.Role != provider.RoleTool {
		t.Errorf("ResponseMessages[1].Role = %q, want %q", toolResult.Role, provider.RoleTool)
	}
	if len(toolResult.Content) < 1 {
		t.Fatalf("ResponseMessages[1].Content length = %d, want >= 1", len(toolResult.Content))
	}
	trp := toolResult.Content[0]
	if trp.Type != provider.PartToolResult {
		t.Errorf("tool result Type = %q, want %q", trp.Type, provider.PartToolResult)
	}
	if trp.ToolCallID != "call-err" {
		t.Errorf("tool result ToolCallID = %q, want %q", trp.ToolCallID, "call-err")
	}
	if trp.ToolOutput != "error: something went wrong" {
		t.Errorf("tool result ToolOutput = %q, want %q", trp.ToolOutput, "error: something went wrong")
	}

	// Message 2: assistant with final text
	if result.ResponseMessages[2].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[2].Role = %q, want %q", result.ResponseMessages[2].Role, provider.RoleAssistant)
	}
	hasText := false
	for _, part := range result.ResponseMessages[2].Content {
		if part.Type == provider.PartText && part.Text == "I see the tool failed." {
			hasText = true
		}
	}
	if !hasText {
		t.Error("ResponseMessages[2] missing expected final text")
	}
}

// TestGenerateText_ResponseMessages_MaxStepsExhausted verifies that when MaxSteps
// is reached with the model still requesting tool calls, ResponseMessages contains
// all tool round-trips plus the final assistant message with tool call parts.
// With MaxSteps=2, both steps execute tools. The delta captures all round-trips,
// and the final assistant message reflects the last step's tool calls.
func TestGenerateText_ResponseMessages_MaxStepsExhausted(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			switch callCount {
			case 1:
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "call-1", Name: "compute", Input: json.RawMessage(`{"n":1}`)},
					},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
				}, nil
			default:
				// Step 2: model returns another tool call (MaxSteps=2, so loop exits after this step)
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "call-2", Name: "compute", Input: json.RawMessage(`{"n":2}`)},
					},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 20, OutputTokens: 5},
				}, nil
			}
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("compute stuff"),
		WithMaxSteps(2),
		WithTools(Tool{
			Name:        "compute",
			Description: "Compute something",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer"}}}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "computed", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// With MaxSteps=2, both steps execute tools and append round-trips.
	// Delta: [assistant(call-1), tool(result-1), assistant(call-2), tool(result-2)]
	// The last step's assistant message is already in the delta (appendToolRoundTrip ran),
	// so buildResponseMessages does NOT append a duplicate. Total: 4 messages.
	if len(result.ResponseMessages) != 4 {
		t.Fatalf("ResponseMessages length = %d, want 4", len(result.ResponseMessages))
	}

	// Message 0: assistant with tool call 1
	if result.ResponseMessages[0].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", result.ResponseMessages[0].Role, provider.RoleAssistant)
	}
	if len(result.ResponseMessages[0].Content) == 0 {
		t.Fatalf("ResponseMessages[0].Content is empty")
	}
	if result.ResponseMessages[0].Content[0].Type != provider.PartToolCall || result.ResponseMessages[0].Content[0].ToolCallID != "call-1" {
		t.Errorf("ResponseMessages[0] should contain tool call 1")
	}

	// Message 1: tool result for call-1
	if result.ResponseMessages[1].Role != provider.RoleTool {
		t.Errorf("ResponseMessages[1].Role = %q, want %q", result.ResponseMessages[1].Role, provider.RoleTool)
	}
	if len(result.ResponseMessages[1].Content) == 0 {
		t.Fatalf("ResponseMessages[1].Content is empty")
	}
	if result.ResponseMessages[1].Content[0].ToolOutput != "computed" {
		t.Errorf("ResponseMessages[1] tool output = %q, want %q", result.ResponseMessages[1].Content[0].ToolOutput, "computed")
	}

	// Message 2: assistant with tool call 2
	if result.ResponseMessages[2].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[2].Role = %q, want %q", result.ResponseMessages[2].Role, provider.RoleAssistant)
	}
	if len(result.ResponseMessages[2].Content) == 0 {
		t.Fatalf("ResponseMessages[2].Content is empty")
	}
	if result.ResponseMessages[2].Content[0].Type != provider.PartToolCall || result.ResponseMessages[2].Content[0].ToolCallID != "call-2" {
		t.Errorf("ResponseMessages[2] should contain tool call 2")
	}

	// Message 3: tool result for call-2 (executed before loop exits)
	if result.ResponseMessages[3].Role != provider.RoleTool {
		t.Errorf("ResponseMessages[3].Role = %q, want %q", result.ResponseMessages[3].Role, provider.RoleTool)
	}
	if len(result.ResponseMessages[3].Content) == 0 {
		t.Fatalf("ResponseMessages[3].Content is empty")
	}
	if result.ResponseMessages[3].Content[0].ToolOutput != "computed" {
		t.Errorf("ResponseMessages[3] tool output = %q, want %q", result.ResponseMessages[3].Content[0].ToolOutput, "computed")
	}
}

// TestStreamText_ResponseMessages_MaxStepsExhausted verifies that when MaxSteps
// is exhausted in streaming mode, ResponseMessages does not contain a duplicate
// assistant message for the last step.
func TestStreamText_ResponseMessages_MaxStepsExhausted(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			switch n {
			case 1:
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "call-1", ToolName: "compute", ToolInput: `{"n":1}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			default:
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "call-2", ToolName: "compute", ToolInput: `{"n":2}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 20, OutputTokens: 5}},
				), nil
			}
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("compute stuff"),
		WithMaxSteps(2),
		WithTools(Tool{
			Name:        "compute",
			Description: "Compute something",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer"}}}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "computed", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Drain the stream.
	for range stream.Stream() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}

	result := stream.Result()

	// With MaxSteps=2, both steps return tool calls. The delta contains all
	// round-trips: [assistant(call-1), tool(result-1), assistant(call-2), tool(result-2)].
	// No duplicate assistant message should be appended. Total: 4 messages.
	if len(result.ResponseMessages) != 4 {
		t.Fatalf("ResponseMessages length = %d, want 4", len(result.ResponseMessages))
	}

	// Message 0: assistant with tool call 1
	if result.ResponseMessages[0].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", result.ResponseMessages[0].Role, provider.RoleAssistant)
	}
	if len(result.ResponseMessages[0].Content) == 0 {
		t.Fatalf("ResponseMessages[0].Content is empty")
	}
	if result.ResponseMessages[0].Content[0].Type != provider.PartToolCall || result.ResponseMessages[0].Content[0].ToolCallID != "call-1" {
		t.Errorf("ResponseMessages[0] should contain tool call 1")
	}

	// Message 1: tool result for call-1
	if result.ResponseMessages[1].Role != provider.RoleTool {
		t.Errorf("ResponseMessages[1].Role = %q, want %q", result.ResponseMessages[1].Role, provider.RoleTool)
	}
	if len(result.ResponseMessages[1].Content) == 0 {
		t.Fatalf("ResponseMessages[1].Content is empty")
	}
	if result.ResponseMessages[1].Content[0].ToolOutput != "computed" {
		t.Errorf("ResponseMessages[1] tool output = %q, want %q", result.ResponseMessages[1].Content[0].ToolOutput, "computed")
	}

	// Message 2: assistant with tool call 2
	if result.ResponseMessages[2].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[2].Role = %q, want %q", result.ResponseMessages[2].Role, provider.RoleAssistant)
	}
	if len(result.ResponseMessages[2].Content) == 0 {
		t.Fatalf("ResponseMessages[2].Content is empty")
	}
	if result.ResponseMessages[2].Content[0].Type != provider.PartToolCall || result.ResponseMessages[2].Content[0].ToolCallID != "call-2" {
		t.Errorf("ResponseMessages[2] should contain tool call 2")
	}

	// Message 3: tool result for call-2
	if result.ResponseMessages[3].Role != provider.RoleTool {
		t.Errorf("ResponseMessages[3].Role = %q, want %q", result.ResponseMessages[3].Role, provider.RoleTool)
	}
	if len(result.ResponseMessages[3].Content) == 0 {
		t.Fatalf("ResponseMessages[3].Content is empty")
	}
	if result.ResponseMessages[3].Content[0].ToolOutput != "computed" {
		t.Errorf("ResponseMessages[3] tool output = %q, want %q", result.ResponseMessages[3].Content[0].ToolOutput, "computed")
	}
}

// TestStreamText_ResponseMessages_WithReasoning verifies that ResponseMessages
// includes reasoning parts in assistant messages, with reasoning appearing before
// text parts (matching appendToolRoundTrip ordering).
func TestStreamText_ResponseMessages_WithReasoning(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkReasoning, Text: "let me think..."},
					provider.StreamChunk{Type: provider.ChunkText, Text: "I'll call the tool"},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "lookup", ToolInput: `{"q":"test"}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "final answer"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20, OutputTokens: 8}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("reason about this"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:        "lookup",
			Description: "Look something up",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "found it", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Drain the stream.
	for range stream.Stream() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}

	result := stream.Result()

	// Expected: assistant (reasoning + text + tool call), tool result, assistant (final text) = 3 messages
	if len(result.ResponseMessages) != 3 {
		t.Fatalf("ResponseMessages length = %d, want 3", len(result.ResponseMessages))
	}

	// Message 0: assistant with reasoning, text, and tool call
	msg0 := result.ResponseMessages[0]
	if msg0.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", msg0.Role, provider.RoleAssistant)
	}
	// Verify ordering: reasoning parts appear before text and tool call parts.
	if len(msg0.Content) != 3 {
		t.Fatalf("ResponseMessages[0].Content length = %d, want 3", len(msg0.Content))
	}
	// First part should be reasoning.
	if msg0.Content[0].Type != provider.PartReasoning {
		t.Errorf("ResponseMessages[0].Content[0].Type = %q, want %q", msg0.Content[0].Type, provider.PartReasoning)
	}
	if msg0.Content[0].Text != "let me think..." {
		t.Errorf("ResponseMessages[0].Content[0].Text = %q, want %q", msg0.Content[0].Text, "let me think...")
	}
	// Second part should be text.
	if msg0.Content[1].Type != provider.PartText {
		t.Errorf("ResponseMessages[0].Content[1].Type = %q, want %q", msg0.Content[1].Type, provider.PartText)
	}
	if msg0.Content[1].Text != "I'll call the tool" {
		t.Errorf("ResponseMessages[0].Content[1].Text = %q, want %q", msg0.Content[1].Text, "I'll call the tool")
	}
	// Third part should be tool call.
	if msg0.Content[2].Type != provider.PartToolCall {
		t.Errorf("ResponseMessages[0].Content[2].Type = %q, want %q", msg0.Content[2].Type, provider.PartToolCall)
	}
	if msg0.Content[2].ToolCallID != "tc1" {
		t.Errorf("ResponseMessages[0].Content[2].ToolCallID = %q, want %q", msg0.Content[2].ToolCallID, "tc1")
	}

	// Message 1: tool result
	if result.ResponseMessages[1].Role != provider.RoleTool {
		t.Errorf("ResponseMessages[1].Role = %q, want %q", result.ResponseMessages[1].Role, provider.RoleTool)
	}
	if len(result.ResponseMessages[1].Content) == 0 {
		t.Fatalf("ResponseMessages[1].Content is empty")
	}
	if result.ResponseMessages[1].Content[0].ToolOutput != "found it" {
		t.Errorf("ResponseMessages[1] tool output = %q, want %q", result.ResponseMessages[1].Content[0].ToolOutput, "found it")
	}

	// Message 2: final assistant with text
	msg2 := result.ResponseMessages[2]
	if msg2.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[2].Role = %q, want %q", msg2.Role, provider.RoleAssistant)
	}
	hasText := false
	for _, part := range msg2.Content {
		if part.Type == provider.PartText && part.Text == "final answer" {
			hasText = true
		}
	}
	if !hasText {
		t.Error("ResponseMessages[2] missing final text")
	}
}

// TestStreamText_ResponseMessages_SingleStep verifies that a simple single-step
// streaming text generation produces a ResponseMessages with one assistant message.
// This tests the buildResult() fallback path where ts.responseMessages is nil.
func TestStreamText_ResponseMessages_SingleStep(t *testing.T) {
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "Hello, world!"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 5, OutputTokens: 3}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("say hi"),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Drain the stream.
	for range stream.Stream() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}

	result := stream.Result()

	// Single-step: one assistant message with the text.
	if len(result.ResponseMessages) != 1 {
		t.Fatalf("ResponseMessages length = %d, want 1", len(result.ResponseMessages))
	}
	if result.ResponseMessages[0].Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", result.ResponseMessages[0].Role, provider.RoleAssistant)
	}
	hasText := false
	for _, part := range result.ResponseMessages[0].Content {
		if part.Type == provider.PartText && part.Text == "Hello, world!" {
			hasText = true
		}
	}
	if !hasText {
		t.Error("ResponseMessages[0] missing expected text")
	}
}

// TestStreamText_ResponseMessages_SingleStepReasoning verifies that a single-step
// streaming response with reasoning (no tools, MaxSteps=1) produces a
// ResponseMessages with one assistant message containing PartReasoning then PartText.
// This tests the buildResult() single-step fallback with reasoning.
func TestStreamText_ResponseMessages_SingleStepReasoning(t *testing.T) {
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkReasoning, Text: "thinking..."},
				// Second reasoning chunk with signature metadata (like Anthropic/Bedrock).
				provider.StreamChunk{Type: provider.ChunkReasoning, Text: "", Metadata: map[string]any{"signature": "sig-abc"}},
				provider.StreamChunk{Type: provider.ChunkText, Text: "answer"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("think about this"),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Drain the stream.
	for range stream.Stream() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}

	result := stream.Result()

	// Single-step with reasoning: one assistant message.
	if len(result.ResponseMessages) != 1 {
		t.Fatalf("ResponseMessages length = %d, want 1", len(result.ResponseMessages))
	}

	msg := result.ResponseMessages[0]
	if msg.Role != provider.RoleAssistant {
		t.Errorf("ResponseMessages[0].Role = %q, want %q", msg.Role, provider.RoleAssistant)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("ResponseMessages[0].Content length = %d, want 2", len(msg.Content))
	}

	// First part: consolidated reasoning (two chunks merged into one with metadata).
	if msg.Content[0].Type != provider.PartReasoning {
		t.Errorf("ResponseMessages[0].Content[0].Type = %q, want %q", msg.Content[0].Type, provider.PartReasoning)
	}
	if msg.Content[0].Text != "thinking..." {
		t.Errorf("ResponseMessages[0].Content[0].Text = %q, want %q", msg.Content[0].Text, "thinking...")
	}
	// Verify ProviderOptions contains merged metadata (signature from second chunk).
	if msg.Content[0].ProviderOptions == nil {
		t.Fatal("ResponseMessages[0].Content[0].ProviderOptions is nil, want signature metadata")
	}
	if sig, ok := msg.Content[0].ProviderOptions["signature"].(string); !ok || sig != "sig-abc" {
		t.Errorf("reasoning ProviderOptions[signature] = %v, want %q", msg.Content[0].ProviderOptions["signature"], "sig-abc")
	}

	// Second part: text.
	if msg.Content[1].Type != provider.PartText {
		t.Errorf("ResponseMessages[0].Content[1].Type = %q, want %q", msg.Content[1].Type, provider.PartText)
	}
	if msg.Content[1].Text != "answer" {
		t.Errorf("ResponseMessages[0].Content[1].Text = %q, want %q", msg.Content[1].Text, "answer")
	}
}

// TestStreamText_ResponseMessages_MidLoopError verifies that when DoStream fails
// on step 2 of a tool loop, the stream reports an error and ResponseMessages is
// either nil or contains only a minimal fallback (not the full tool round-trip).
func TestStreamText_ResponseMessages_MidLoopError(t *testing.T) {
	var callCount atomic.Int32
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			if n == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 10, OutputTokens: 5}},
				), nil
			}
			return nil, &APIError{Message: "server error", StatusCode: 500}
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithMaxRetries(0),
		WithTools(Tool{
			Name:        "tool",
			Description: "A tool",
			InputSchema: json.RawMessage(`{"type":"object"}`),
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "ok", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Drain the stream to trigger the error.
	for range stream.Stream() {
	}

	// Verify stream has an error.
	if stream.Err() == nil {
		t.Fatal("expected non-nil Err()")
	}
	var apiErr *APIError
	if !errors.As(stream.Err(), &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", stream.Err(), stream.Err())
	}
	if apiErr.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", apiErr.StatusCode)
	}

	// ResponseMessages should be minimal: at most one assistant message with partial
	// tool call data. The full round-trip (assistant + tool results) from step 1 is lost
	// because the goroutine returns without setting ts.responseMessages on error.
	result := stream.Result()
	if len(result.ResponseMessages) > 1 {
		t.Errorf("expected ResponseMessages to have at most 1 message on mid-loop error, got %d", len(result.ResponseMessages))
	}
	// If present, the single message should be an assistant message (not a tool result).
	for _, msg := range result.ResponseMessages {
		if msg.Role != provider.RoleAssistant {
			t.Errorf("expected only assistant messages in error ResponseMessages, got role %q", msg.Role)
		}
	}
}

// TestGenerateText_ResponseMessages_EmptyResponse verifies that ResponseMessages
// is nil when the model returns empty text and no tool calls.
func TestGenerateText_ResponseMessages_EmptyResponse(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 5, OutputTokens: 0},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model, WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}

	if result.ResponseMessages != nil {
		t.Errorf("ResponseMessages = %v, want nil for empty response", result.ResponseMessages)
	}
}
