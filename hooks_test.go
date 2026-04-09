package goai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
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
	if !captured.StartTime.IsZero() {
		t.Error("StartTime should be zero for unknown tools")
	}
	if captured.Duration != 0 {
		t.Error("Duration should be zero for unknown tools")
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

// --- OnBeforeToolExecute tests ---

func TestOnBeforeToolExecute_PassThrough(t *testing.T) {
	// Hook returns Skip=false → tool executes normally.
	var hookCalled bool
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

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "file contents", nil
			},
		}),
		WithOnBeforeToolExecute(func(info BeforeToolExecuteInfo) BeforeToolExecuteResult {
			hookCalled = true
			if info.ToolName != "read" {
				t.Errorf("ToolName = %q, want read", info.ToolName)
			}
			if info.ToolCallID != "tc1" {
				t.Errorf("ToolCallID = %q, want tc1", info.ToolCallID)
			}
			return BeforeToolExecuteResult{} // Skip=false
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !hookCalled {
		t.Error("OnBeforeToolExecute was not called")
	}
	if result.Text != "done" {
		t.Errorf("Text = %q, want done", result.Text)
	}
}

func TestOnBeforeToolExecute_Skip(t *testing.T) {
	// Hook returns Skip=true → tool does NOT execute, synthetic result used.
	toolExecuted := false
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "bash", Input: json.RawMessage(`{"cmd":"rm -rf /"}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Step 2: verify the tool result contains the synthetic "permission denied".
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult && part.ToolOutput == "permission denied" {
						return &provider.GenerateResult{Text: "ok, I won't do that", FinishReason: provider.FinishStop}, nil
					}
				}
			}
			return &provider.GenerateResult{Text: "unexpected", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "bash",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				toolExecuted = true
				return "executed!", nil
			},
		}),
		WithOnBeforeToolExecute(func(_ BeforeToolExecuteInfo) BeforeToolExecuteResult {
			return BeforeToolExecuteResult{Skip: true, Result: "permission denied"}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if toolExecuted {
		t.Error("tool should NOT have executed when Skip=true")
	}
	if result.Text != "ok, I won't do that" {
		t.Errorf("Text = %q, want 'ok, I won't do that'", result.Text)
	}
}

func TestOnBeforeToolExecute_SkipWithError(t *testing.T) {
	// Hook returns Skip=true with Error → tool result is an error message.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "write", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Check the tool result contains error.
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult && part.ToolOutput == "error: doom loop detected" {
						return &provider.GenerateResult{Text: "stopped", FinishReason: provider.FinishStop}, nil
					}
				}
			}
			return &provider.GenerateResult{Text: "unexpected", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "write",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeToolExecute(func(_ BeforeToolExecuteInfo) BeforeToolExecuteResult {
			return BeforeToolExecuteResult{Skip: true, Error: fmt.Errorf("doom loop detected")}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "stopped" {
		t.Errorf("Text = %q, want stopped", result.Text)
	}
}

// --- OnAfterToolExecute tests ---

func TestOnAfterToolExecute_ModifyOutput(t *testing.T) {
	// Hook modifies tool output (e.g., secret scanning).
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "read", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Check the tool result was modified by hook.
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult && part.ToolOutput == "REDACTED" {
						return &provider.GenerateResult{Text: "got redacted", FinishReason: provider.FinishStop}, nil
					}
				}
			}
			return &provider.GenerateResult{Text: "unexpected", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "secret-api-key-12345", nil
			},
		}),
		WithOnAfterToolExecute(func(info AfterToolExecuteInfo) AfterToolExecuteResult {
			if info.ToolName != "read" {
				t.Errorf("ToolName = %q, want read", info.ToolName)
			}
			if info.Output != "secret-api-key-12345" {
				t.Errorf("Output = %q, want original", info.Output)
			}
			return AfterToolExecuteResult{Output: "REDACTED"}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "got redacted" {
		t.Errorf("Text = %q, want 'got redacted'", result.Text)
	}
}

func TestOnAfterToolExecute_NoModify(t *testing.T) {
	// Hook returns empty result → original output preserved.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "read", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult && part.ToolOutput == "original output" {
						return &provider.GenerateResult{Text: "preserved", FinishReason: provider.FinishStop}, nil
					}
				}
			}
			return &provider.GenerateResult{Text: "unexpected", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "original output", nil
			},
		}),
		WithOnAfterToolExecute(func(_ AfterToolExecuteInfo) AfterToolExecuteResult {
			return AfterToolExecuteResult{} // empty → no change
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "preserved" {
		t.Errorf("Text = %q, want preserved", result.Text)
	}
}

// --- OnBeforeStep tests ---

func TestOnBeforeStep_InjectMessages(t *testing.T) {
	// Hook injects extra user messages before step 2.
	callCount := 0
	var step2Messages []provider.Message
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			step2Messages = p.Messages
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(info BeforeStepInfo) BeforeStepResult {
			if info.Step != 2 {
				t.Errorf("Step = %d, want 2", info.Step)
			}
			return BeforeStepResult{
				ExtraMessages: []provider.Message{
					{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "injected message"}}},
				},
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Verify injected message is in the conversation.
	found := false
	for _, msg := range step2Messages {
		for _, part := range msg.Content {
			if part.Type == provider.PartText && part.Text == "injected message" {
				found = true
			}
		}
	}
	if !found {
		t.Error("injected message not found in step 2 messages")
	}
}

func TestOnBeforeStep_Stop(t *testing.T) {
	// Hook returns Stop=true → loop terminates before step 2 LLM call.
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
			return &provider.GenerateResult{Text: "should not reach", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			return BeforeStepResult{Stop: true}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if callCount != 1 {
		t.Errorf("LLM called %d times, want 1 (stop before step 2)", callCount)
	}
	// Result should have the tool calls from step 1.
	if len(result.ToolCalls) != 1 {
		t.Errorf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
}

func TestOnBeforeStep_NotCalledOnStep1(t *testing.T) {
	// OnBeforeStep is only called for step 2+, NOT step 1.
	hookCalled := false
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "hello", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			hookCalled = true
			return BeforeStepResult{}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if hookCalled {
		t.Error("OnBeforeStep should NOT be called when there's only 1 step")
	}
}

// --- StreamText variants ---

func TestOnBeforeToolExecute_StreamText(t *testing.T) {
	// Same as GenerateText skip test but via StreamText.
	toolExecuted := false
	callCount := 0
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			callCount++
			if callCount == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "bash", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "skipped"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "bash",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				toolExecuted = true
				return "ran", nil
			},
		}),
		WithOnBeforeToolExecute(func(_ BeforeToolExecuteInfo) BeforeToolExecuteResult {
			return BeforeToolExecuteResult{Skip: true, Result: "denied"}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	result := stream.Result()
	if toolExecuted {
		t.Error("tool should NOT have executed")
	}
	if result.Text != "skipped" {
		t.Errorf("Text = %q, want skipped", result.Text)
	}
}

func TestOnBeforeStep_StreamText_Stop(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			callCount++
			if callCount == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "should not reach"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			return BeforeStepResult{Stop: true}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	stream.Result() // consume
	if callCount != 1 {
		t.Errorf("LLM called %d times, want 1", callCount)
	}
}

// --- Test 7: Multiple parallel tool calls with partial skip ---

func TestOnBeforeToolExecute_PartialSkip(t *testing.T) {
	// Model returns 2 tool calls in step 1. Hook skips "dangerous" but allows "safe".
	// Verify: "dangerous" NOT executed, "safe" IS executed, LLM receives both results.
	dangerousExecuted := false
	safeExecuted := false
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls: []provider.ToolCall{
						{ID: "tc1", Name: "dangerous", Input: json.RawMessage(`{}`)},
						{ID: "tc2", Name: "safe", Input: json.RawMessage(`{}`)},
					},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Step 2: verify both tool results are in messages.
			var foundDenied, foundSafeOutput bool
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult {
						if part.ToolOutput == "permission denied" {
							foundDenied = true
						}
						if part.ToolOutput == "safe result" {
							foundSafeOutput = true
						}
					}
				}
			}
			if !foundDenied {
				t.Error("step 2: expected 'permission denied' tool result for dangerous")
			}
			if !foundSafeOutput {
				t.Error("step 2: expected 'safe result' tool result for safe")
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(
			Tool{
				Name: "dangerous",
				Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
					dangerousExecuted = true
					return "dangerous output", nil
				},
			},
			Tool{
				Name: "safe",
				Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
					safeExecuted = true
					return "safe result", nil
				},
			},
		),
		WithOnBeforeToolExecute(func(info BeforeToolExecuteInfo) BeforeToolExecuteResult {
			if info.ToolName == "dangerous" {
				return BeforeToolExecuteResult{Skip: true, Result: "permission denied"}
			}
			return BeforeToolExecuteResult{} // pass-through
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if dangerousExecuted {
		t.Error("dangerous tool should NOT have executed")
	}
	if !safeExecuted {
		t.Error("safe tool should have executed")
	}
	if result.Text != "done" {
		t.Errorf("Text = %q, want done", result.Text)
	}
}

// --- Test 8: OnAfterToolExecute when tool returns error ---

func TestOnAfterToolExecute_ToolError(t *testing.T) {
	// Tool returns error. Hook receives the error in info.Error.
	// Hook returns AfterToolExecuteResult{Output: "modified error output"}.
	var capturedInfo AfterToolExecuteInfo
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "flaky", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Step 2: check the tool result contains the modified error output.
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult && part.ToolOutput == "error: timeout" {
						return &provider.GenerateResult{Text: "handled error", FinishReason: provider.FinishStop}, nil
					}
				}
			}
			return &provider.GenerateResult{Text: "unexpected", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "flaky",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "partial output", fmt.Errorf("timeout")
			},
		}),
		WithOnAfterToolExecute(func(info AfterToolExecuteInfo) AfterToolExecuteResult {
			capturedInfo = info
			return AfterToolExecuteResult{} // don't modify --let the error pass through
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Verify hook received the error and partial output.
	if capturedInfo.Error == nil {
		t.Error("hook should have received non-nil Error")
	} else if capturedInfo.Error.Error() != "timeout" {
		t.Errorf("hook Error = %q, want 'timeout'", capturedInfo.Error)
	}
	if capturedInfo.Output != "partial output" {
		t.Errorf("hook Output = %q, want 'partial output'", capturedInfo.Output)
	}
	if result.Text != "handled error" {
		t.Errorf("Text = %q, want 'handled error'", result.Text)
	}
}

// --- Test 9: OnBeforeToolExecute panic recovery ---

func TestOnBeforeToolExecute_PanicRecovery(t *testing.T) {
	// Hook panics. Tool should NOT execute. Error result sent to LLM.
	toolExecuted := false
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "read", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Step 2: check that the tool result contains "panicked".
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult {
						if strings.Contains(part.ToolOutput, "panicked") {
							return &provider.GenerateResult{Text: "recovered", FinishReason: provider.FinishStop}, nil
						}
					}
				}
			}
			return &provider.GenerateResult{Text: "unexpected", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				toolExecuted = true
				return "should not run", nil
			},
		}),
		WithOnBeforeToolExecute(func(_ BeforeToolExecuteInfo) BeforeToolExecuteResult {
			panic("kaboom")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if toolExecuted {
		t.Error("tool should NOT have executed when hook panicked")
	}
	if result.Text != "recovered" {
		t.Errorf("Text = %q, want 'recovered'", result.Text)
	}
}

// --- Test 10: OnAfterToolExecute in StreamText ---

func TestOnAfterToolExecute_StreamText(t *testing.T) {
	// Same as GenerateText test but via StreamText. Verify output modification works.
	callCount := 0
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, p provider.GenerateParams) (*provider.StreamResult, error) {
			callCount++
			if callCount == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "read", ToolInput: "{}"},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls},
				), nil
			}
			// Step 2: check tool result contains "REDACTED".
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult && part.ToolOutput == "REDACTED" {
						return streamFromChunks(
							provider.StreamChunk{Type: provider.ChunkText, Text: "redacted ok"},
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

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "secret-data", nil
			},
		}),
		WithOnAfterToolExecute(func(info AfterToolExecuteInfo) AfterToolExecuteResult {
			if info.Output == "secret-data" {
				return AfterToolExecuteResult{Output: "REDACTED"}
			}
			return AfterToolExecuteResult{}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	result := stream.Result()
	if result.Text != "redacted ok" {
		t.Errorf("Text = %q, want 'redacted ok'", result.Text)
	}
}

// --- Test 11: OnBeforeToolExecute skip fires OnToolCall with Skipped=true ---

func TestOnBeforeToolExecute_SkipFiresOnToolCallWithSkipped(t *testing.T) {
	// When hook skips, OnToolCall still fires with Skipped=true.
	var captured []ToolCallInfo
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "blocked", Input: json.RawMessage(`{"x":1}`)}},
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
			Name: "blocked",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				t.Error("tool should NOT have executed")
				return "nope", nil
			},
		}),
		WithOnBeforeToolExecute(func(_ BeforeToolExecuteInfo) BeforeToolExecuteResult {
			return BeforeToolExecuteResult{Skip: true, Result: "skipped by policy"}
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
	if !captured[0].Skipped {
		t.Error("Skipped should be true")
	}
	if captured[0].ToolCallID != "tc1" {
		t.Errorf("ToolCallID = %q, want tc1", captured[0].ToolCallID)
	}
	if captured[0].ToolName != "blocked" {
		t.Errorf("ToolName = %q, want blocked", captured[0].ToolName)
	}
	if captured[0].Output != "skipped by policy" {
		t.Errorf("Output = %q, want 'skipped by policy'", captured[0].Output)
	}
	if captured[0].StartTime.IsZero() {
		t.Error("StartTime should be non-zero")
	}
	if captured[0].Duration != 0 {
		t.Errorf("Duration = %v, want 0 for skipped tools", captured[0].Duration)
	}
}

// --- Finding 7: OnBeforeStep panic recovery ---

func TestOnBeforeStep_PanicRecovery_GenerateText(t *testing.T) {
	// OnBeforeStep panics --should be recovered, step proceeds normally.
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
			return &provider.GenerateResult{Text: "done after panic", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			panic("step panic!")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if callCount != 2 {
		t.Errorf("LLM called %d times, want 2 (panic recovered, step proceeded)", callCount)
	}
	if result.Text != "done after panic" {
		t.Errorf("Text = %q, want 'done after panic'", result.Text)
	}
}

func TestOnBeforeStep_PanicRecovery_StreamText(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			callCount++
			if callCount == 1 {
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "tool", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls},
				), nil
			}
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "recovered"},
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
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			panic("stream step panic!")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	result := stream.Result()
	if callCount != 2 {
		t.Errorf("LLM called %d times, want 2", callCount)
	}
	if result.Text != "recovered" {
		t.Errorf("Text = %q, want recovered", result.Text)
	}
}

// --- Finding 8: OnAfterToolExecute panic recovery ---

func TestOnAfterToolExecute_PanicRecovery(t *testing.T) {
	// OnAfterToolExecute panics --original tool result preserved.
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "read", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Check the original tool output survived the panic.
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult && part.ToolOutput == "original output" {
						return &provider.GenerateResult{Text: "preserved", FinishReason: provider.FinishStop}, nil
					}
				}
			}
			return &provider.GenerateResult{Text: "lost", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name: "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
				return "original output", nil
			},
		}),
		WithOnAfterToolExecute(func(_ AfterToolExecuteInfo) AfterToolExecuteResult {
			panic("after-execute panic!")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "preserved" {
		t.Errorf("Text = %q, want preserved (original output should survive panic)", result.Text)
	}
}

// --- Finding 9: OnBeforeStep in GenerateObject ---

func TestOnBeforeStep_GenerateObject(t *testing.T) {
	// Verify OnBeforeStep fires in GenerateObject's tool loop.
	var hookCalled bool
	var hookStep int
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
			return &provider.GenerateResult{
				Text:         `{"name":"test","value":42}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	type TestObj struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	result, err := GenerateObject[TestObj](t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(info BeforeStepInfo) BeforeStepResult {
			hookCalled = true
			hookStep = info.Step
			return BeforeStepResult{}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !hookCalled {
		t.Error("OnBeforeStep was not called in GenerateObject")
	}
	if hookStep != 2 {
		t.Errorf("Step = %d, want 2", hookStep)
	}
	if result.Object.Name != "test" {
		t.Errorf("Object.Name = %q, want test", result.Object.Name)
	}
}

func TestOnBeforeStep_GenerateObject_Stop(t *testing.T) {
	// OnBeforeStep Stop=true in GenerateObject terminates the loop.
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
			return &provider.GenerateResult{
				Text:         `{"name":"stopped","value":0}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	type TestObj struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	_, err := GenerateObject[TestObj](t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			return BeforeStepResult{Stop: true}
		}),
	)
	// Stop before step 2 means no structured output produced -- expect MaxSteps error.
	if err == nil {
		t.Fatal("expected error when Stop prevents structured output")
	}
	if callCount != 1 {
		t.Errorf("LLM called %d times, want 1 (Stop before step 2)", callCount)
	}
}

func TestOnBeforeStep_GenerateObject_InjectMessages(t *testing.T) {
	// OnBeforeStep injects messages in GenerateObject.
	callCount := 0
	var step2MsgCount int
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			step2MsgCount = len(p.Messages)
			return &provider.GenerateResult{
				Text:         `{"name":"injected","value":99}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	type TestObj struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	result, err := GenerateObject[TestObj](t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			return BeforeStepResult{
				ExtraMessages: []provider.Message{
					{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "extra context"}}},
				},
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Object.Name != "injected" {
		t.Errorf("Object.Name = %q, want injected", result.Object.Name)
	}
	// Step 2 should have more messages than step 1 (original + assistant + tool + injected).
	if step2MsgCount < 4 {
		t.Errorf("step 2 message count = %d, want >= 4 (with injected message)", step2MsgCount)
	}
}

func TestOnBeforeStep_GenerateObject_PanicRecovery(t *testing.T) {
	// OnBeforeStep panics in GenerateObject -- recovered, step proceeds.
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
			return &provider.GenerateResult{
				Text:         `{"name":"recovered","value":1}`,
				FinishReason: provider.FinishStop,
			}, nil
		},
	}

	type TestObj struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	result, err := GenerateObject[TestObj](t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			panic("object step panic!")
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if callCount != 2 {
		t.Errorf("LLM called %d times, want 2 (panic recovered)", callCount)
	}
	if result.Object.Name != "recovered" {
		t.Errorf("Object.Name = %q, want recovered", result.Object.Name)
	}
}

// --- Finding 5: Skip=true with both Result and Error set ---

func TestOnBeforeToolExecute_SkipWithResultAndError(t *testing.T) {
	// When Skip=true with both Result and Error, Error takes precedence.
	// Result is ignored --LLM receives "error: <message>".
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			// Verify the tool result is the error, not the Result string.
			foundToolResult := false
			for _, msg := range p.Messages {
				for _, part := range msg.Content {
					if part.Type == provider.PartToolResult {
						foundToolResult = true
						if part.ToolOutput == "error: access denied" {
							return &provider.GenerateResult{Text: "error wins", FinishReason: provider.FinishStop}, nil
						}
						if part.ToolOutput == "some context" {
							t.Error("Result string should be ignored when Error is set, but it was sent to LLM")
						}
					}
				}
			}
			if !foundToolResult {
				t.Error("no tool result message found in step 2 -- tool result was dropped entirely")
			}
			return &provider.GenerateResult{Text: "unexpected", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "should not run", nil },
		}),
		WithOnBeforeToolExecute(func(_ BeforeToolExecuteInfo) BeforeToolExecuteResult {
			return BeforeToolExecuteResult{
				Skip:   true,
				Result: "some context",            // should be ignored
				Error:  fmt.Errorf("access denied"), // should win
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "error wins" {
		t.Errorf("Text = %q, want 'error wins'", result.Text)
	}
}

// --- Finding D: Stop=true + ExtraMessages → ExtraMessages ignored ---

func TestOnBeforeStep_StopIgnoresExtraMessages(t *testing.T) {
	// When Stop=true and ExtraMessages are set, ExtraMessages are ignored.
	callCount := 0
	var step2Messages []provider.Message
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "tool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			step2Messages = p.Messages
			return &provider.GenerateResult{Text: "step2", FinishReason: provider.FinishStop}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(_ BeforeStepInfo) BeforeStepResult {
			return BeforeStepResult{
				Stop: true,
				ExtraMessages: []provider.Message{
					{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "should be ignored"}}},
				},
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	// Stop=true → loop should have stopped before step 2, LLM called only once.
	if callCount != 1 {
		t.Errorf("LLM called %d times, want 1 (Stop=true should prevent step 2)", callCount)
	}
	// step2Messages should be nil (never reached step 2).
	if step2Messages != nil {
		t.Error("step 2 should not have been reached")
	}
	// Result should have tool calls from step 1 (loop stopped before step 2).
	if len(result.ToolCalls) != 1 {
		t.Errorf("ToolCalls = %d, want 1", len(result.ToolCalls))
	}
}

// --- Ctx field tests ---

func TestOnBeforeToolExecute_CtxHasToolCallID(t *testing.T) {
	// Verify info.Ctx is non-nil and carries the tool call ID.
	var capturedCtx context.Context
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc-abc", Name: "read", Input: json.RawMessage(`{}`)}},
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
			Name:    "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeToolExecute(func(info BeforeToolExecuteInfo) BeforeToolExecuteResult {
			capturedCtx = info.Ctx
			return BeforeToolExecuteResult{}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if capturedCtx == nil {
		t.Fatal("BeforeToolExecuteInfo.Ctx should not be nil")
	}
	if id := ToolCallIDFromContext(capturedCtx); id != "tc-abc" {
		t.Errorf("ToolCallIDFromContext = %q, want tc-abc", id)
	}
}

func TestOnAfterToolExecute_CtxHasToolCallID(t *testing.T) {
	var capturedCtx context.Context
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc-xyz", Name: "read", Input: json.RawMessage(`{}`)}},
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
			Name:    "read",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnAfterToolExecute(func(info AfterToolExecuteInfo) AfterToolExecuteResult {
			capturedCtx = info.Ctx
			return AfterToolExecuteResult{}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if capturedCtx == nil {
		t.Fatal("AfterToolExecuteInfo.Ctx should not be nil")
	}
	if id := ToolCallIDFromContext(capturedCtx); id != "tc-xyz" {
		t.Errorf("ToolCallIDFromContext = %q, want tc-xyz", id)
	}
}

type ctxKey struct{}

func TestOnBeforeStep_CtxInheritsFromCaller(t *testing.T) {
	// Verify info.Ctx inherits from the caller's context (not context.Background).
	var capturedCtx context.Context
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

	// Inject a custom value into the caller context.
	callerCtx := context.WithValue(t.Context(), ctxKey{}, "caller-marker")

	_, err := GenerateText(callerCtx, model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "tool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnBeforeStep(func(info BeforeStepInfo) BeforeStepResult {
			capturedCtx = info.Ctx
			return BeforeStepResult{}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if capturedCtx == nil {
		t.Fatal("BeforeStepInfo.Ctx should not be nil")
	}
	// Verify the context inherits from the caller (not a fresh context.Background).
	if v, ok := capturedCtx.Value(ctxKey{}).(string); !ok || v != "caller-marker" {
		t.Error("BeforeStepInfo.Ctx should inherit from caller context, but custom value not found")
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
