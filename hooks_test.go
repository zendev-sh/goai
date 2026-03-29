package goai

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestWithOnRequest(t *testing.T) {
	var captured RequestInfo
	model := &mockModel{
		id: "test-model",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 5}}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithTools(Tool{Name: "read", InputSchema: json.RawMessage(`{}`)}),
		WithOnRequest(func(info RequestInfo) {
			captured = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if captured.Model != "test-model" {
		t.Errorf("Model = %q, want test-model", captured.Model)
	}
	if captured.MessageCount != 1 {
		t.Errorf("MessageCount = %d, want 1", captured.MessageCount)
	}
	if captured.ToolCount != 1 {
		t.Errorf("ToolCount = %d, want 1", captured.ToolCount)
	}
	if captured.Timestamp.IsZero() {
		t.Error("Timestamp should not be zero")
	}
}

func TestWithOnResponse_Success(t *testing.T) {
	var captured ResponseInfo
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 10, OutputTokens: 5},
			}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithOnResponse(func(info ResponseInfo) {
			captured = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if captured.Latency <= 0 {
		t.Error("Latency should be > 0")
	}
	if captured.Usage.InputTokens != 10 {
		t.Errorf("Usage.InputTokens = %d, want 10", captured.Usage.InputTokens)
	}
	if captured.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason = %q, want stop", captured.FinishReason)
	}
	if captured.Error != nil {
		t.Errorf("Error should be nil, got %v", captured.Error)
	}
}

func TestWithOnResponse_Error(t *testing.T) {
	var captured ResponseInfo
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, &APIError{Message: "bad request", StatusCode: 400}
		},
	}

	_, _ = GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithMaxRetries(0),
		WithOnResponse(func(info ResponseInfo) {
			captured = info
		}),
	)

	if captured.Error == nil {
		t.Error("Error should be non-nil")
	}
	if captured.StatusCode != 400 {
		t.Errorf("StatusCode = %d, want 400", captured.StatusCode)
	}
}

func TestWithOnToolCall_Success(t *testing.T) {
	var captured []ToolCallInfo
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "read", Input: json.RawMessage(`{"path":"a.txt"}`)}},
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
			Name: "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "file contents", nil
			},
		}),
		WithOnToolCall(func(info ToolCallInfo) {
			captured = append(captured, info)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(captured) != 1 {
		t.Fatalf("OnToolCall called %d times, want 1", len(captured))
	}
	if captured[0].ToolCallID != "tc1" {
		t.Errorf("ToolCallID = %q, want tc1", captured[0].ToolCallID)
	}
	if captured[0].ToolName != "read" {
		t.Errorf("ToolName = %q, want read", captured[0].ToolName)
	}
	if captured[0].Step != 1 {
		t.Errorf("Step = %d, want 1", captured[0].Step)
	}
	if string(captured[0].Input) != `{"path":"a.txt"}` {
		t.Errorf("Input = %s, want {\"path\":\"a.txt\"}", captured[0].Input)
	}
	if captured[0].Output != "file contents" {
		t.Errorf("Output = %q, want \"file contents\"", captured[0].Output)
	}
	if captured[0].OutputObject != nil {
		t.Errorf("OutputObject should be nil for non-JSON output, got %v", captured[0].OutputObject)
	}
	if captured[0].Duration < 0 {
		t.Error("Duration should be >= 0")
	}
	if captured[0].Error != nil {
		t.Errorf("Error should be nil, got %v", captured[0].Error)
	}
}

func TestWithOnToolCall_Error(t *testing.T) {
	var captured ToolCallInfo
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "fail", Input: json.RawMessage(`{}`)}},
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
			Name: "fail",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "", fmt.Errorf("tool failed")
			},
		}),
		WithOnToolCall(func(info ToolCallInfo) {
			captured = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if captured.Error == nil {
		t.Error("Error should be non-nil")
	}
	if captured.Error.Error() != "tool failed" {
		t.Errorf("Error = %q, want 'tool failed'", captured.Error)
	}
}

func TestWithOnToolCall_JSONOutput(t *testing.T) {
	var captured ToolCallInfo
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "info", Input: json.RawMessage(`{}`)}},
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
			Name: "info",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return `{"temperature":22,"city":"Tokyo"}`, nil
			},
		}),
		WithOnToolCall(func(info ToolCallInfo) {
			captured = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if captured.OutputObject == nil {
		t.Fatal("OutputObject should be non-nil for JSON output")
	}
	m, ok := captured.OutputObject.(map[string]any)
	if !ok {
		t.Fatalf("OutputObject type = %T, want map[string]any", captured.OutputObject)
	}
	if m["city"] != "Tokyo" {
		t.Errorf("OutputObject[city] = %v, want Tokyo", m["city"])
	}
}

func TestWithOnToolCall_UnknownTool(t *testing.T) {
	var captured ToolCallInfo
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "nonexistent", Input: json.RawMessage(`{}`)}},
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
			Name:    "known",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnToolCall(func(info ToolCallInfo) {
			captured = info
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if captured.ToolName != "nonexistent" {
		t.Errorf("ToolName = %q", captured.ToolName)
	}
	if captured.Error == nil {
		t.Error("expected error for unknown tool")
	}
}

func TestHooks_MultipleSteps(t *testing.T) {
	// Verify hooks fire on each step of a multi-step tool loop.
	var reqCount, respCount, toolCount int
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 10},
				}, nil
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 20}}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnRequest(func(_ RequestInfo) { reqCount++ }),
		WithOnResponse(func(_ ResponseInfo) { respCount++ }),
		WithOnToolCall(func(_ ToolCallInfo) { toolCount++ }),
	)
	if err != nil {
		t.Fatal(err)
	}

	if reqCount != 2 {
		t.Errorf("OnRequest called %d times, want 2", reqCount)
	}
	if respCount != 2 {
		t.Errorf("OnResponse called %d times, want 2", respCount)
	}
	if toolCount != 1 {
		t.Errorf("OnToolCall called %d times, want 1", toolCount)
	}
}

func TestWithOnRequest_SystemMessageInMessages(t *testing.T) {
	// requestMessages prepends the system message to RequestInfo.Messages
	// so hooks always see the full conversation including the system prompt.
	var captured RequestInfo
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithSystem("you are a helper"),
		WithPrompt("hello"),
		WithOnRequest(func(info RequestInfo) { captured = info }),
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(captured.Messages) < 2 {
		t.Fatalf("Messages len = %d, want >= 2 (system + user)", len(captured.Messages))
	}
	if captured.Messages[0].Role != provider.RoleSystem {
		t.Errorf("Messages[0].Role = %q, want system", captured.Messages[0].Role)
	}
	if captured.Messages[1].Role != provider.RoleUser {
		t.Errorf("Messages[1].Role = %q, want user", captured.Messages[1].Role)
	}
}
