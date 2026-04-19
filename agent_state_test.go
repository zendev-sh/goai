package goai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/goleak"

	"github.com/zendev-sh/goai/provider"
)

// validateReplayStrict enforces Anthropic-style replay invariants on a
// ResponseMessages transcript:
//
//  1. Count: every tool_use ID has exactly one tool_result with the same ID.
//  2. Role correctness: tool_result parts appear only in RoleTool messages;
//     tool_use parts appear only in RoleAssistant messages.
//  3. Adjacency: the tool_result message(s) for an assistant's tool_use
//     message appear in the *immediately next* message(s) - before any
//     following RoleAssistant / RoleUser message.
//  4. Ordering within turn: all tool_results for a given assistant turn
//     appear before the next RoleAssistant message.
//  5. No dangling tool_result: every tool_result has a preceding tool_use.
//
// Returns the first violation found or nil when the transcript is sound.
func validateReplayStrict(msgs []provider.Message) error {
	// Invariant 2a + 5 + 1-count precheck: collect tool_use ids (assistant)
	// and tool_result ids (tool), then cross-check.
	assistantUses := map[string]int{}     // id -> count
	toolResultsByID := map[string]int{}    // id -> count
	for i, m := range msgs {
		for _, p := range m.Content {
			if p.Type == provider.PartToolCall {
				if m.Role != provider.RoleAssistant {
					return fmt.Errorf("msg[%d]: tool_use in non-assistant role %q", i, m.Role)
				}
				assistantUses[p.ToolCallID]++
			}
			if p.Type == provider.PartToolResult {
				if m.Role != provider.RoleTool {
					return fmt.Errorf("msg[%d]: tool_result in non-tool role %q", i, m.Role)
				}
				toolResultsByID[p.ToolCallID]++
			}
		}
	}
	// Invariant 5: every tool_result has a preceding tool_use.
	for id := range toolResultsByID {
		if _, ok := assistantUses[id]; !ok {
			return fmt.Errorf("dangling tool_result with no matching tool_use: %q", id)
		}
	}
	// Invariant 1 (count): exactly-one tool_result per tool_use.
	for id, uses := range assistantUses {
		if uses != 1 {
			return fmt.Errorf("tool_use id %q appears %d times (want 1)", id, uses)
		}
		if toolResultsByID[id] != 1 {
			return fmt.Errorf("tool_use id %q has %d tool_results (want 1)", id, toolResultsByID[id])
		}
	}
	// Invariants 3 + 4: for each assistant message with tool_use parts, the
	// next message(s) must be RoleTool and together cover all tool_use ids,
	// before any subsequent RoleAssistant / RoleUser message.
	for i, m := range msgs {
		if m.Role != provider.RoleAssistant {
			continue
		}
		pendingIDs := map[string]bool{}
		for _, p := range m.Content {
			if p.Type == provider.PartToolCall {
				pendingIDs[p.ToolCallID] = true
			}
		}
		if len(pendingIDs) == 0 {
			continue
		}
		// Walk forward. Next message must be RoleTool; keep consuming
		// RoleTool messages until all ids covered. No RoleAssistant /
		// RoleUser is allowed before all ids are covered.
		j := i + 1
		for len(pendingIDs) > 0 {
			if j >= len(msgs) {
				// Transcript ended while tool_use ids are still uncovered. This
				// is distinct from "mismatching id" (handled below) - here the
				// transcript is simply truncated / incomplete.
				return fmt.Errorf("assistant msg[%d]: transcript ends with %d unmatched tool_use id(s) (missing: %v)", i, len(pendingIDs), keysOf(pendingIDs))
			}
			next := msgs[j]
			if next.Role != provider.RoleTool {
				return fmt.Errorf("assistant msg[%d]: expected RoleTool at msg[%d] to cover pending tool_use ids, got role %q (missing: %v)", i, j, next.Role, keysOf(pendingIDs))
			}
			for _, p := range next.Content {
				if p.Type == provider.PartToolResult {
					if !pendingIDs[p.ToolCallID] {
						return fmt.Errorf("tool msg[%d] pos %d: tool_result id %q does not match any pending tool_use from assistant msg[%d]", j, i, p.ToolCallID, i)
					}
					delete(pendingIDs, p.ToolCallID)
				}
			}
			j++
		}
	}
	return nil
}

func keysOf(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// TestWithStopWhen_NeverTrue_FullLoop: predicate always false; loop runs until
// the model returns a step without tool calls.
func TestWithStopWhen_NeverTrue_FullLoop(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			if n < 3 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	var checks atomic.Int32
	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithStopWhen(func(steps []StepResult) bool {
			checks.Add(1)
			return false
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Steps) != 3 {
		t.Fatalf("steps=%d want 3", len(result.Steps))
	}
	if result.FinishReason != provider.FinishStop {
		t.Errorf("FinishReason=%q want stop", result.FinishReason)
	}
	// Predicate is evaluated only after steps that had tool calls (so step 1 & step 2),
	// not after the terminal step. We expect at least 2 evaluations.
	if checks.Load() < 2 {
		t.Errorf("predicate evaluated %d times, want >= 2", checks.Load())
	}
}

// TestWithStopWhen_TrueAfterStep1_ExitsCleanly: predicate returns true after
// the first step; loop exits with Steps[0] populated and no goroutine leaks.
func TestWithStopWhen_TrueAfterStep1_ExitsCleanly(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	var toolInvokes atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			return &provider.GenerateResult{
				ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) {
			toolInvokes.Add(1)
			return "ok", nil
		}}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 1 }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Steps) != 1 {
		t.Fatalf("steps=%d want 1", len(result.Steps))
	}
	// Vercel-parity StopWhen placement: step-1's tools run BEFORE the
	// predicate evaluation, so the one tool call is invoked exactly once.
	if callCount.Load() != 1 {
		t.Errorf("callCount=%d want 1 (step 2 must not issue DoGenerate)", callCount.Load())
	}
	if toolInvokes.Load() != 1 {
		t.Errorf("toolInvokes=%d want 1 (step-1 tool executes before StopWhen break)", toolInvokes.Load())
	}
	// Natural finish reason from step 1 (tool_calls) is preserved.
	if result.FinishReason != provider.FinishToolCalls {
		t.Errorf("FinishReason=%q want %q", result.FinishReason, provider.FinishToolCalls)
	}
	if result.StepsExhausted {
		t.Errorf("StepsExhausted=true; expected false for StopWhen-break under MaxSteps")
	}
	// ResponseMessages must contain the paired assistant(tool_use) +
	// tool(tool_result) for step 1 so the transcript is replay-safe against
	// Anthropic/OpenAI (which reject a dangling tool_use).
	if len(result.ResponseMessages) != 2 {
		t.Fatalf("ResponseMessages len=%d want 2 (assistant+tool pair); got=%#v", len(result.ResponseMessages), result.ResponseMessages)
	}
	if result.ResponseMessages[0].Role != provider.RoleAssistant {
		t.Errorf("msg[0].Role=%q want assistant", result.ResponseMessages[0].Role)
	}
	if result.ResponseMessages[1].Role != provider.RoleTool {
		t.Errorf("msg[1].Role=%q want tool", result.ResponseMessages[1].Role)
	}
}

// TestWithStopWhen_TrueAfterStep2_ResponseMessagesIntact: verifies that both
// step 1 and step 2 contribute their assistant/tool-result messages to
// ResponseMessages.
func TestWithStopWhen_TrueAfterStep2_ResponseMessagesIntact(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			return &provider.GenerateResult{
				Text:         fmt.Sprintf("step-%d-text", n),
				ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "tool-out", nil }}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 2 }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Steps) != 2 {
		t.Fatalf("steps=%d want 2", len(result.Steps))
	}
	if len(result.ResponseMessages) == 0 {
		t.Fatalf("ResponseMessages is empty")
	}
	// Vercel-parity StopWhen placement: the predicate fires AFTER the
	// current step's tools have executed, so both step-1 and step-2 have
	// fully paired assistant(tool_use) + tool(tool_result) messages in the
	// delta. The transcript is safe to replay against strict providers
	// (Anthropic, OpenAI) that reject dangling tool_use.
	//   1. assistant(step-1-text + tc1)
	//   2. tool(result for tc1)
	//   3. assistant(step-2-text + tc2)
	//   4. tool(result for tc2)
	if len(result.ResponseMessages) != 4 {
		t.Fatalf("ResponseMessages len=%d want 4; got=%#v", len(result.ResponseMessages), result.ResponseMessages)
	}
	// Both step-1 and step-2 assistant text must be present.
	foundStep1, foundStep2 := false, false
	for _, m := range result.ResponseMessages {
		for _, p := range m.Content {
			if p.Text == "step-1-text" {
				foundStep1 = true
			}
			if p.Text == "step-2-text" {
				foundStep2 = true
			}
		}
	}
	if !foundStep1 {
		t.Errorf("step-1 assistant text missing from ResponseMessages")
	}
	if !foundStep2 {
		t.Errorf("step-2 assistant text missing from ResponseMessages (Gap 1 regressed)")
	}
	// StepsExhausted must remain false even though we stopped with pending tool
	// calls; the stop was driven by the predicate, not MaxSteps exhaustion.
	if result.StepsExhausted {
		t.Errorf("StepsExhausted=true; want false for StopWhen break")
	}
}

// TestWithStopWhen_OnBeforeStepStop_ResponseMessagesIntact verifies the
// OnBeforeStep.Stop path: because OnBeforeStep fires at the START of the
// next step (AFTER appendToolRoundTrip ran for the previous step), the last
// step's assistant AND tool-result are already in the delta. We must NOT
// duplicate them.
func TestWithStopWhen_OnBeforeStepStop_ResponseMessagesIntact(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			return &provider.GenerateResult{
				Text:         fmt.Sprintf("step-%d-text", n),
				ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	// OnBeforeStep fires at step 2+; stop immediately so LLM call #2 never issues.
	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "tool-out", nil }}),
		WithOnBeforeStep(func(BeforeStepInfo) BeforeStepResult { return BeforeStepResult{Stop: true} }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Steps) != 1 {
		t.Fatalf("steps=%d want 1", len(result.Steps))
	}
	if callCount.Load() != 1 {
		t.Errorf("callCount=%d want 1 (step 2 must not issue DoGenerate)", callCount.Load())
	}
	// Expected: assistant(step1) + tool-result(step1) = 2 messages. Must NOT
	// be duplicated; appendToolRoundTrip already ran for step 1.
	if len(result.ResponseMessages) != 2 {
		t.Fatalf("ResponseMessages len=%d want 2 (no duplication); got=%#v", len(result.ResponseMessages), result.ResponseMessages)
	}
	if result.StepsExhausted {
		t.Errorf("StepsExhausted=true; want false for OnBeforeStep.Stop")
	}
}

// TestWithStopWhen_AtMaxSteps_StepsExhaustedFalse: the predicate fires on
// exactly the MaxSteps step with pending tool calls. The post-loop derivation
// must NOT flip StepsExhausted=true, because the loop was stopped by the
// user predicate, not by MaxSteps exhaustion.
func TestWithStopWhen_AtMaxSteps_StepsExhaustedFalse(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			return &provider.GenerateResult{
				Text:         fmt.Sprintf("s%d", n),
				ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(2),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 2 }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Steps) != 2 {
		t.Fatalf("steps=%d want 2", len(result.Steps))
	}
	if result.StepsExhausted {
		t.Errorf("StepsExhausted=true; want false (StopWhen break at MaxSteps boundary)")
	}
}

// TestStreamText_StopWhenMidLoop_CleanClose: streaming version - StopWhen
// true after step 1 closes the stream cleanly with no leaks and a final
// ChunkFinish carrying step 1's natural finish reason.
func TestStreamText_StopWhenMidLoop_CleanClose(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test-stream",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: fmt.Sprintf("step%d", n)},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: fmt.Sprintf("tc%d", n), ToolName: "t", ToolInput: `{}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}},
			), nil
		},
	}

	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 1 }),
	)
	if err != nil {
		t.Fatal(err)
	}

	var finalFinish provider.FinishReason
	var sawFinish bool
	for chunk := range stream.Stream() {
		if chunk.Type == provider.ChunkFinish {
			finalFinish = chunk.FinishReason
			sawFinish = true
		}
	}

	if !sawFinish {
		t.Fatal("never observed ChunkFinish")
	}
	if finalFinish != provider.FinishToolCalls {
		t.Errorf("final ChunkFinish.FinishReason=%q want %q", finalFinish, provider.FinishToolCalls)
	}

	result := stream.Result()
	if len(result.Steps) != 1 {
		t.Errorf("Steps=%d want 1", len(result.Steps))
	}
	if callCount.Load() != 1 {
		t.Errorf("callCount=%d want 1", callCount.Load())
	}
	if err := stream.Err(); err != nil {
		t.Errorf("stream.Err()=%v", err)
	}
}

// TestAgentState_Observe_TransitionsCorrectly: verifies state transitions
// across a 2-step tool loop: Starting -> LLMInFlight(1) -> ToolExecuting(1)
// -> LLMInFlight(2) -> Idle.
func TestAgentState_Observe_TransitionsCorrectly(t *testing.T) {
	defer goleak.VerifyNone(t)

	var state AgentState

	// Record observed transitions via a tool Execute hook.
	var observations []struct {
		kind StepKind
		step int
	}
	var mu sync.Mutex
	record := func(label string) {
		k, s := state.Observe()
		mu.Lock()
		observations = append(observations, struct {
			kind StepKind
			step int
		}{k, s})
		mu.Unlock()
		_ = label
	}

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			// At time of LLM call, we should see LLMInFlight with step==n.
			k, s := state.Observe()
			if k != StepLLMInFlight {
				t.Errorf("during DoGenerate step %d: observed kind=%v want LLMInFlight", n, k)
			}
			if s != int(n) {
				t.Errorf("during DoGenerate step %d: observed step=%d want %d", n, s, n)
			}
			if n < 2 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "t", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	// OnStepFinish fires AFTER StateRef transitions to StepStepFinished
	// (see Gap 4 fix). Capture the state observed from inside OnStepFinish
	// to assert the post-LLM / pre-tool window is visible to pollers.
	var stepFinishedObservations []StepKind
	var sfMu sync.Mutex
	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithStateRef(&state),
		WithOnStepFinish(func(StepResult) {
			k, _ := state.Observe()
			sfMu.Lock()
			stepFinishedObservations = append(stepFinishedObservations, k)
			sfMu.Unlock()
		}),
		WithTools(Tool{
			Name: "t",
			Execute: func(context.Context, json.RawMessage) (string, error) {
				record("during-tool-execute")
				return "ok", nil
			},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// After return, state must be Idle.
	k, _ := state.Observe()
	if k != StepIdle {
		t.Errorf("after return: kind=%v want Idle", k)
	}

	// While tool was executing we must have observed ToolExecuting.
	mu.Lock()
	defer mu.Unlock()
	if len(observations) == 0 {
		t.Fatal("no tool-execute observations recorded")
	}
	for _, obs := range observations {
		if obs.kind != StepToolExecuting {
			t.Errorf("tool-execute observation kind=%v want ToolExecuting", obs.kind)
		}
	}

	// Gap 4: StepStepFinished must be observable between DoGenerate return
	// and tool execution. OnStepFinish fires exactly in that window.
	sfMu.Lock()
	defer sfMu.Unlock()
	if len(stepFinishedObservations) == 0 {
		t.Fatal("no OnStepFinish observations recorded")
	}
	for i, k := range stepFinishedObservations {
		if k != StepStepFinished {
			t.Errorf("OnStepFinish obs[%d] kind=%v want StepStepFinished", i, k)
		}
	}
}

// TestAgentState_RaceFree: concurrent readers + one writer via goai's tool
// loop. Run with `-race` to validate atomic access.
func TestAgentState_RaceFree(t *testing.T) {
	defer goleak.VerifyNone(t)

	var state AgentState
	stopPoll := make(chan struct{})
	var pollWG sync.WaitGroup
	for range 8 {
		pollWG.Go(func() {
			for {
				select {
				case <-stopPoll:
					return
				default:
					_, _ = state.Observe()
				}
			}
		})
	}

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			if n < 3 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
				}, nil
			}
			return &provider.GenerateResult{
				Text:         "done",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithStateRef(&state),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) {
			// Let pollers spin briefly during tool execution.
			time.Sleep(2 * time.Millisecond)
			return "ok", nil
		}}),
	)
	close(stopPoll)
	pollWG.Wait()

	if err != nil {
		t.Fatal(err)
	}
	k, _ := state.Observe()
	if k != StepIdle {
		t.Errorf("post-run kind=%v want Idle", k)
	}
}

// TestWithStopWhen_NilPredicate_NoOp: explicit nil is safe and behaves as if
// the option were never set.
func TestWithStopWhen_NilPredicate_NoOp(t *testing.T) {
	defer goleak.VerifyNone(t)

	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}
	_, err := GenerateText(t.Context(), model, WithPrompt("go"), WithStopWhen(nil))
	if err != nil {
		t.Fatal(err)
	}
}

// TestWithStopWhen_Panic_Recovered: a panicking predicate is treated as "do
// not stop" and logged; loop continues normally.
func TestWithStopWhen_Panic_Recovered(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			if n < 2 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "t", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
					Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
				}, nil
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}}, nil
		},
	}
	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithStopWhen(func(steps []StepResult) bool { panic(errors.New("boom")) }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if callCount.Load() != 2 {
		t.Errorf("callCount=%d want 2 (loop should have proceeded past panic)", callCount.Load())
	}
}

// TestWithStopWhen_ReplaySafety verifies the ResponseMessages produced on a
// StopWhen break is a valid replay transcript: every tool_use has a paired
// tool_result. We simulate replay by appending a user message and running
// GenerateText again through a strict mock that rejects dangling tool_use.
func TestWithStopWhen_ReplaySafety(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			return &provider.GenerateResult{
				Text:         fmt.Sprintf("s%d", n),
				ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}
	first, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 1 }),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Strict Anthropic-style replay validator: enforces 5 invariants --
	// count, role correctness, adjacency, ordering, no-dangling. See
	// validateReplayStrict above.
	if err := validateReplayStrict(first.ResponseMessages); err != nil {
		t.Fatalf("ResponseMessages are not replay-safe: %v", err)
	}

	// Second call: simulate "continue conversation" with a strict mock that
	// runs the validator on the incoming message list.
	var innerErr error
	strict := &mockModel{
		id: "strict",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			if err := validateReplayStrict(p.Messages); err != nil {
				innerErr = err
				return nil, err
			}
			return &provider.GenerateResult{
				Text:         "ok",
				FinishReason: provider.FinishStop,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}
	msgs := append([]provider.Message{}, first.ResponseMessages...)
	msgs = append(msgs, provider.Message{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "continue"}}})

	_, err = GenerateText(t.Context(), strict,
		WithMessages(msgs...),
	)
	if err != nil {
		t.Fatalf("replay GenerateText failed: %v (innerErr=%v)", err, innerErr)
	}
}

// TestAgentState_ValidationError_StillIdle verifies that a pre-loop
// validation error (empty prompt/messages) still transitions StateRef to
// StepIdle so pollers waiting on it do not deadlock.
func TestAgentState_ValidationError_StillIdle(t *testing.T) {
	defer goleak.VerifyNone(t)

	var state AgentState
	_, err := GenerateText(t.Context(), &mockModel{id: "t"}, WithStateRef(&state))
	if err == nil {
		t.Fatal("expected validation error, got nil")
	}
	k, _ := state.Observe()
	if k != StepIdle {
		t.Errorf("GenerateText validation-error: state=%v want StepIdle", k)
	}

	var state2 AgentState
	_, err = StreamText(t.Context(), &mockModel{id: "t"}, WithStateRef(&state2))
	if err == nil {
		t.Fatal("expected validation error, got nil")
	}
	k2, _ := state2.Observe()
	if k2 != StepIdle {
		t.Errorf("StreamText validation-error: state=%v want StepIdle", k2)
	}
}

// TestWithStopWhen_OnBeforeStep_MarksHookStopped verifies FIX 2: when
// OnBeforeStep returns Stop=true the post-loop StepsExhausted derivation
// treats it like a WithStopWhen break (hookStopped semantics), so a Stop=true
// at exactly the MaxSteps boundary does NOT flip StepsExhausted=true.
func TestWithStopWhen_OnBeforeStep_MarksHookStopped(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			return &provider.GenerateResult{
				ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	var finalInfo FinishInfo
	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(2),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithOnBeforeStep(func(info BeforeStepInfo) BeforeStepResult {
			if info.Step == 2 {
				return BeforeStepResult{Stop: true}
			}
			return BeforeStepResult{}
		}),
		WithOnFinish(func(fi FinishInfo) { finalInfo = fi }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.StepsExhausted {
		t.Errorf("StepsExhausted=true; want false (OnBeforeStep.Stop at MaxSteps boundary)")
	}
	if finalInfo.StoppedBy != provider.StopCauseBeforeStep {
		t.Errorf("FinishInfo.StoppedBy=%q want %q", finalInfo.StoppedBy, provider.StopCauseBeforeStep)
	}
}

// TestWithStopWhen_StoppedByCause exercises FIX 5: the StopCause signal is
// surfaced on FinishInfo and the final ChunkFinish for each termination
// kind (natural, predicate, max-steps, before-step).
func TestWithStopWhen_StoppedByCause(t *testing.T) {
	defer goleak.VerifyNone(t)

	mkModel := func(cc *atomic.Int32, steps int) *mockModel {
		return &mockModel{
			id: "test",
			generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
				n := cc.Add(1)
				if int(n) < steps {
					return &provider.GenerateResult{
						ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "t", Input: json.RawMessage(`{}`)}},
						FinishReason: provider.FinishToolCalls,
						Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
					}, nil
				}
				return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}}, nil
			},
		}
	}

	type tc struct {
		name  string
		run   func() FinishInfo
		stream bool
		want  provider.StopCause
	}
	cases := []tc{
		{
			name: "natural",
			want: provider.StopCauseNatural,
			run: func() FinishInfo {
				var cc atomic.Int32
				var info FinishInfo
				_, _ = GenerateText(t.Context(), mkModel(&cc, 2),
					WithPrompt("go"), WithMaxSteps(5),
					WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
					WithOnFinish(func(fi FinishInfo) { info = fi }),
				)
				return info
			},
		},
		{
			name: "predicate",
			want: provider.StopCausePredicate,
			run: func() FinishInfo {
				var cc atomic.Int32
				var info FinishInfo
				_, _ = GenerateText(t.Context(), mkModel(&cc, 10),
					WithPrompt("go"), WithMaxSteps(10),
					WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
					WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 1 }),
					WithOnFinish(func(fi FinishInfo) { info = fi }),
				)
				return info
			},
		},
		{
			name: "max-steps",
			want: provider.StopCauseMaxSteps,
			run: func() FinishInfo {
				var cc atomic.Int32
				var info FinishInfo
				_, _ = GenerateText(t.Context(), mkModel(&cc, 100),
					WithPrompt("go"), WithMaxSteps(2),
					WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
					WithOnFinish(func(fi FinishInfo) { info = fi }),
				)
				return info
			},
		},
		{
			name: "before-step",
			want: provider.StopCauseBeforeStep,
			run: func() FinishInfo {
				var cc atomic.Int32
				var info FinishInfo
				_, _ = GenerateText(t.Context(), mkModel(&cc, 10),
					WithPrompt("go"), WithMaxSteps(5),
					WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
					WithOnBeforeStep(func(info BeforeStepInfo) BeforeStepResult {
						if info.Step == 2 {
							return BeforeStepResult{Stop: true}
						}
						return BeforeStepResult{}
					}),
					WithOnFinish(func(fi FinishInfo) { info = fi }),
				)
				return info
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := c.run()
			if got.StoppedBy != c.want {
				t.Errorf("FinishInfo.StoppedBy=%q want %q", got.StoppedBy, c.want)
			}
		})
	}

	// Streaming parity: verify the final ChunkFinish carries StoppedBy for a
	// predicate-driven break.
	t.Run("stream-predicate", func(t *testing.T) {
		defer goleak.VerifyNone(t)
		var cc atomic.Int32
		streamModel := &mockModel{
			id: "test-stream",
			streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
				n := cc.Add(1)
				return streamFromChunks(
					provider.StreamChunk{Type: provider.ChunkText, Text: fmt.Sprintf("s%d", n)},
					provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: fmt.Sprintf("tc%d", n), ToolName: "t", ToolInput: `{}`},
					provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}},
				), nil
			},
		}
		stream, err := StreamText(t.Context(), streamModel,
			WithPrompt("go"), WithMaxSteps(10),
			WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
			WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 1 }),
		)
		if err != nil {
			t.Fatal(err)
		}
		var lastFinish provider.StreamChunk
		for chunk := range stream.Stream() {
			if chunk.Type == provider.ChunkFinish {
				lastFinish = chunk
			}
		}
		_ = stream.Result()
		if lastFinish.StoppedBy != provider.StopCausePredicate {
			t.Errorf("final ChunkFinish.StoppedBy=%q want %q", lastFinish.StoppedBy, provider.StopCausePredicate)
		}
	})
}

// TestWithStopWhen_PredicateSeesToolResults verifies FIX 7: the StopCondition
// predicate sees ToolResults populated on the most recent step (not only
// ToolCalls). The predicate stops iff the last step's tool result contains a
// magic string. The model returns a tool call each step; the tool echoes an
// input-driven token so the predicate can branch on outputs.
func TestWithStopWhen_PredicateSeesToolResults(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			return &provider.GenerateResult{
				Text:         fmt.Sprintf("s%d", n),
				ToolCalls:    []provider.ToolCall{{ID: fmt.Sprintf("tc%d", n), Name: "echo", Input: json.RawMessage(fmt.Sprintf(`{"v":%d}`, n))}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}

	// Tool returns "STOP" on step 2 input, otherwise "continue".
	echoTool := Tool{
		Name: "echo",
		Execute: func(_ context.Context, in json.RawMessage) (string, error) {
			var v struct{ V int }
			_ = json.Unmarshal(in, &v)
			if v.V == 2 {
				return "STOP-NOW", nil
			}
			return "continue", nil
		},
	}

	var observed [][]provider.ToolResult
	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(echoTool),
		WithStopWhen(func(steps []StepResult) bool {
			// Snapshot the last step's tool results so we can audit the
			// predicate actually saw them.
			last := steps[len(steps)-1]
			observed = append(observed, append([]provider.ToolResult(nil), last.ToolResults...))
			for _, r := range last.ToolResults {
				if r.Output == "STOP-NOW" {
					return true
				}
			}
			return false
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Steps) != 2 {
		t.Fatalf("steps=%d want 2 (should stop after step 2)", len(result.Steps))
	}
	// Verify StepResult.ToolResults was populated on the final step.
	last := result.Steps[len(result.Steps)-1]
	if len(last.ToolResults) != 1 {
		t.Fatalf("step[last].ToolResults len=%d want 1", len(last.ToolResults))
	}
	if last.ToolResults[0].Output != "STOP-NOW" {
		t.Errorf("ToolResults[0].Output=%q want STOP-NOW", last.ToolResults[0].Output)
	}
	if last.ToolResults[0].ToolCallID != "tc2" {
		t.Errorf("ToolResults[0].ToolCallID=%q want tc2", last.ToolResults[0].ToolCallID)
	}
	if last.ToolResults[0].IsError {
		t.Errorf("ToolResults[0].IsError=true want false")
	}
	// Predicate saw both steps' outputs.
	if len(observed) != 2 {
		t.Errorf("predicate fired %d times, want 2", len(observed))
	}
}

// TestWithStopWhen_PredicateSeesToolResults_Stream is the streaming-path
// analog: verifies that StreamText also exposes ToolResults to the predicate
// before the stop decision, matching the sync path.
func TestWithStopWhen_PredicateSeesToolResults_Stream(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: fmt.Sprintf("s%d", n)},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: fmt.Sprintf("tc%d", n), ToolName: "echo", ToolInput: fmt.Sprintf(`{"v":%d}`, n)},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}},
			), nil
		},
	}
	echoTool := Tool{
		Name: "echo",
		Execute: func(_ context.Context, in json.RawMessage) (string, error) {
			var v struct{ V int }
			_ = json.Unmarshal(in, &v)
			if v.V == 2 {
				return "STOP-NOW", nil
			}
			return "continue", nil
		},
	}
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(echoTool),
		WithStopWhen(func(steps []StepResult) bool {
			last := steps[len(steps)-1]
			for _, r := range last.ToolResults {
				if r.Output == "STOP-NOW" {
					return true
				}
			}
			return false
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	for range stream.Stream() {
	}
	res := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) != 2 {
		t.Fatalf("stream steps=%d want 2", len(res.Steps))
	}
	last := res.Steps[len(res.Steps)-1]
	if len(last.ToolResults) != 1 || last.ToolResults[0].Output != "STOP-NOW" {
		t.Errorf("stream step[last].ToolResults=%#v want one STOP-NOW entry", last.ToolResults)
	}
}

// TestWithStopWhen_ReplaySafety_ParallelTools verifies that when a single step
// requests MULTIPLE tool calls in parallel and the predicate fires after that
// step, every tool_use has a paired tool_result in ResponseMessages (strict
// Anthropic-style replay validation). Also asserts ToolResults has one entry
// per call in the same order.
func TestWithStopWhen_ReplaySafety_ParallelTools(t *testing.T) {
	defer goleak.VerifyNone(t)

	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text: "parallel",
				ToolCalls: []provider.ToolCall{
					{ID: "tcA", Name: "t", Input: json.RawMessage(`{"n":1}`)},
					{ID: "tcB", Name: "t", Input: json.RawMessage(`{"n":2}`)},
					{ID: "tcC", Name: "t", Input: json.RawMessage(`{"n":3}`)},
				},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}
	first, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{Name: "t", Execute: func(_ context.Context, in json.RawMessage) (string, error) {
			return "ok:" + string(in), nil
		}}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 1 }),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Assert ToolResults parity with ToolCalls, element-for-element.
	last := first.Steps[len(first.Steps)-1]
	if len(last.ToolResults) != 3 {
		t.Fatalf("ToolResults len=%d want 3", len(last.ToolResults))
	}
	for i, want := range []string{"tcA", "tcB", "tcC"} {
		if last.ToolResults[i].ToolCallID != want {
			t.Errorf("ToolResults[%d].ToolCallID=%q want %q", i, last.ToolResults[i].ToolCallID, want)
		}
		if last.ToolResults[i].IsError {
			t.Errorf("ToolResults[%d].IsError=true", i)
		}
	}

	if err := validateReplayStrict(first.ResponseMessages); err != nil {
		t.Fatalf("parallel-tool ResponseMessages not replay-safe: %v", err)
	}
	// Sanity: replay against a strict mock that validates incoming messages.
	var innerErr error
	strict := &mockModel{
		id: "strict",
		generateFn: func(_ context.Context, p provider.GenerateParams) (*provider.GenerateResult, error) {
			if err := validateReplayStrict(p.Messages); err != nil {
				innerErr = err
				return nil, err
			}
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}}, nil
		},
	}
	msgs := append([]provider.Message{}, first.ResponseMessages...)
	msgs = append(msgs, provider.Message{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "go on"}}})
	if _, err := GenerateText(t.Context(), strict, WithMessages(msgs...)); err != nil {
		t.Fatalf("parallel-tool replay failed: %v (inner=%v)", err, innerErr)
	}
}

// TestWithStopWhen_ReplaySafety_MultiRound verifies replay safety across
// MULTIPLE rounds of parallel tool calls before the predicate fires. Three
// rounds, each with 2 parallel tool calls; predicate stops after round 3.
// The full transcript must have all 6 tool_use IDs paired.
func TestWithStopWhen_ReplaySafety_MultiRound(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			n := callCount.Add(1)
			return &provider.GenerateResult{
				Text: fmt.Sprintf("round %d", n),
				ToolCalls: []provider.ToolCall{
					{ID: fmt.Sprintf("r%d-a", n), Name: "t", Input: json.RawMessage(`{}`)},
					{ID: fmt.Sprintf("r%d-b", n), Name: "t", Input: json.RawMessage(`{}`)},
				},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}
	first, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 3 }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Steps) != 3 {
		t.Fatalf("steps=%d want 3", len(first.Steps))
	}
	// Every step must carry ToolResults (len==2).
	for i, s := range first.Steps {
		if len(s.ToolResults) != 2 {
			t.Errorf("step[%d].ToolResults len=%d want 2", i, len(s.ToolResults))
		}
	}

	if err := validateReplayStrict(first.ResponseMessages); err != nil {
		t.Fatalf("multi-round ResponseMessages not replay-safe: %v", err)
	}
}

// TestWithStopWhen_ReplaySafety_ParallelTools_Stream is the streaming-path
// analog of TestWithStopWhen_ReplaySafety_ParallelTools: a single step with
// multiple parallel tool calls, predicate stops after that step. Enforces
// the full Anthropic-style invariants via validateReplayStrict on
// TextResult.ResponseMessages drained from stream.Result().
func TestWithStopWhen_ReplaySafety_ParallelTools_Stream(t *testing.T) {
	defer goleak.VerifyNone(t)

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "parallel"},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tcA", ToolName: "t", ToolInput: `{"n":1}`},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tcB", ToolName: "t", ToolInput: `{"n":2}`},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tcC", ToolName: "t", ToolInput: `{"n":3}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}},
			), nil
		},
	}
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),
		WithTools(Tool{Name: "t", Execute: func(_ context.Context, in json.RawMessage) (string, error) {
			return "ok:" + string(in), nil
		}}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 1 }),
	)
	if err != nil {
		t.Fatal(err)
	}
	for range stream.Stream() {
	}
	res := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	// Assert element-for-element ToolResults parity on the last step.
	last := res.Steps[len(res.Steps)-1]
	if len(last.ToolResults) != 3 {
		t.Fatalf("stream ToolResults len=%d want 3", len(last.ToolResults))
	}
	for i, want := range []string{"tcA", "tcB", "tcC"} {
		if last.ToolResults[i].ToolCallID != want {
			t.Errorf("stream ToolResults[%d].ToolCallID=%q want %q", i, last.ToolResults[i].ToolCallID, want)
		}
	}
	if err := validateReplayStrict(res.ResponseMessages); err != nil {
		t.Fatalf("stream parallel-tool ResponseMessages not replay-safe: %v", err)
	}
}

// TestWithStopWhen_ReplaySafety_MultiRound_Stream is the streaming-path
// analog of TestWithStopWhen_ReplaySafety_MultiRound: three rounds of two
// parallel tool calls each, predicate stops after round 3. The full
// transcript (6 tool_use IDs) must pass validateReplayStrict.
func TestWithStopWhen_ReplaySafety_MultiRound_Stream(t *testing.T) {
	defer goleak.VerifyNone(t)

	var callCount atomic.Int32
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			n := callCount.Add(1)
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: fmt.Sprintf("round %d", n)},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: fmt.Sprintf("r%d-a", n), ToolName: "t", ToolInput: `{}`},
				provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: fmt.Sprintf("r%d-b", n), ToolName: "t", ToolInput: `{}`},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls, Usage: provider.Usage{InputTokens: 1, OutputTokens: 1}},
			), nil
		},
	}
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(10),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithStopWhen(func(steps []StepResult) bool { return len(steps) >= 3 }),
	)
	if err != nil {
		t.Fatal(err)
	}
	for range stream.Stream() {
	}
	res := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) != 3 {
		t.Fatalf("stream steps=%d want 3", len(res.Steps))
	}
	for i, s := range res.Steps {
		if len(s.ToolResults) != 2 {
			t.Errorf("stream step[%d].ToolResults len=%d want 2", i, len(s.ToolResults))
		}
	}
	if err := validateReplayStrict(res.ResponseMessages); err != nil {
		t.Fatalf("stream multi-round ResponseMessages not replay-safe: %v", err)
	}
}

// TestStopCauseEmpty_StreamEmptyProviderResponse verifies FIX 9: when the
// provider closes its stream without emitting any chunks (no text, no tool
// calls, no finish_reason), goai exits the tool loop with StopCauseEmpty --
// NOT StopCauseAbort (which is reserved for real error paths).
func TestStopCauseEmpty_StreamEmptyProviderResponse(t *testing.T) {
	defer goleak.VerifyNone(t)

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			// Close immediately with no chunks at all.
			ch := make(chan provider.StreamChunk)
			close(ch)
			return &provider.StreamResult{Stream: ch}, nil
		},
	}
	var finishInfo FinishInfo
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{Name: "t", Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil }}),
		WithOnFinish(func(fi FinishInfo) { finishInfo = fi }),
	)
	if err != nil {
		t.Fatal(err)
	}
	var lastFinish provider.StreamChunk
	for chunk := range stream.Stream() {
		if chunk.Type == provider.ChunkFinish {
			lastFinish = chunk
		}
	}
	_ = stream.Result()
	if lastFinish.StoppedBy != provider.StopCauseEmpty {
		t.Errorf("final ChunkFinish.StoppedBy=%q want %q", lastFinish.StoppedBy, provider.StopCauseEmpty)
	}
	if finishInfo.StoppedBy != provider.StopCauseEmpty {
		t.Errorf("FinishInfo.StoppedBy=%q want %q", finishInfo.StoppedBy, provider.StopCauseEmpty)
	}
}

// TestStopCauseNoExecutableTools verifies FIX 11: when the model returns tool
// calls but no tool in the configured set has an Execute function, the sync
// loop exits cleanly with StopCauseNoExecutableTools - distinguishing it from
// StopCauseNatural ("model stopped on its own").
//
// Emission scope: StopCauseNoExecutableTools is sync-only (GenerateText).
// The streaming tool-loop (streamWithToolLoop) is only entered when at least
// one executable tool is configured, so the cause is unreachable from that
// path; a streaming call with zero executable tools takes the single-shot
// path and reports StopCauseNatural. This is intentional and documented on
// the StopCauseNoExecutableTools constant godoc.
func TestStopCauseNoExecutableTools(t *testing.T) {
	defer goleak.VerifyNone(t)

	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{
				Text:         "calling",
				ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "declared", Input: json.RawMessage(`{}`)}},
				FinishReason: provider.FinishToolCalls,
				Usage:        provider.Usage{InputTokens: 1, OutputTokens: 1},
			}, nil
		},
	}
	var fi FinishInfo
	// Tool is declared (for model awareness) but has no Execute - so toolMap
	// is empty. goai must exit with NoExecutableTools, not Natural.
	_, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{Name: "declared"}), // no Execute
		WithOnFinish(func(info FinishInfo) { fi = info }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if fi.StoppedBy != provider.StopCauseNoExecutableTools {
		t.Errorf("FinishInfo.StoppedBy=%q want %q", fi.StoppedBy, provider.StopCauseNoExecutableTools)
	}
}

// TestStopCauseNatural_MaxStepsOne_EmptyResponse_SingleShot locks in the
// documented behavior (FIX 32 / FIX 38) for the streaming single-shot
// branch when the provider emits zero chunks: the loop exits cleanly
// with StopCauseNatural - NOT StopCauseEmpty (which is reserved for the
// multi-step tool-loop path; see provider.StopCauseEmpty godoc), NOT
// StopCauseMaxSteps, NOT StopCauseAbort - and leaks no goroutines.
//
// MaxSteps=1 (or zero executable tools) bypasses streamWithToolLoop
// entirely, so the classification is whatever the single-shot defer
// hardcodes. Today that defer emits StopCauseNatural unconditionally.
// A latent semantic TODO (should empty single-shot streams emit
// StopCauseEmpty for symmetry with the multi-step path?) is tracked
// below; if the defer is ever changed, this test breaks and the change
// must be reviewed jointly with provider/types.go StopCauseEmpty godoc
// and the plan doc §1.5.
//
// Originally named TestStopCauseEmpty_MaxStepsOne_EmptyResponse --
// renamed FIX 37 to match the actual asserted behavior; the old name
// implied StopCauseEmpty was expected here, which contradicted the
// assertion body.
func TestStopCauseNatural_MaxStepsOne_EmptyResponse_SingleShot(t *testing.T) {
	defer goleak.VerifyNone(t)

	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			ch := make(chan provider.StreamChunk)
			close(ch)
			return &provider.StreamResult{Stream: ch}, nil
		},
	}
	var finishInfo FinishInfo
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(1),
		// No tools: forces the MaxSteps=1 single-shot branch.
		WithOnFinish(func(fi FinishInfo) { finishInfo = fi }),
	)
	if err != nil {
		t.Fatal(err)
	}
	var lastFinish provider.StreamChunk
	chunkCount := 0
	for chunk := range stream.Stream() {
		chunkCount++
		if chunk.Type == provider.ChunkFinish {
			lastFinish = chunk
		}
	}
	res := stream.Result()
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream Err: %v", err)
	}
	// The single-shot path does not classify "empty" streams - only the
	// multi-step loop emits StopCauseEmpty. In MaxSteps=1 the loop is
	// bypassed entirely, so StoppedBy on ChunkFinish (the provider's
	// chunk, absent in this case) stays empty and OnFinish fires with
	// StopCauseNatural (the default for single-shot streaming; hardcoded
	// in generate.go ~line 280).
	//
	// FIX 32: positive assertion locks in current behavior.
	// TODO(stopcause-semantics): StopCauseNatural semantically means
	// "model stopped on its own"; an empty-chunk stream is arguably
	// better classified as StopCauseEmpty here too. Keeping the
	// single-shot path simple for now - revisit if consumers report
	// ambiguity. If this assertion breaks, also review the
	// hardcoded StopCauseNatural at generate.go `fireOnFinish`.
	if finishInfo.StoppedBy != provider.StopCauseNatural {
		t.Errorf("FinishInfo.StoppedBy=%q; want %q (single-shot empty stream hardcodes natural)",
			finishInfo.StoppedBy, provider.StopCauseNatural)
	}
	// Defense-in-depth: the specific wrong classifications we must never emit.
	if finishInfo.StoppedBy == provider.StopCauseMaxSteps {
		t.Errorf("FinishInfo.StoppedBy=max-steps; must not be overwritten on empty single-shot")
	}
	if finishInfo.StoppedBy == provider.StopCauseAbort {
		t.Errorf("FinishInfo.StoppedBy=abort on empty single-shot; want non-abort (clean exit)")
	}
	// lastFinish is the zero StreamChunk because the provider emitted none;
	// chunkCount must be 0.
	if chunkCount != 0 {
		t.Errorf("chunkCount=%d; empty stream should emit 0 chunks, got lastFinish=%+v", chunkCount, lastFinish)
	}
	if len(res.Steps) != 0 {
		t.Errorf("len(Steps)=%d; MaxSteps=1 empty single-shot should produce 0 steps", len(res.Steps))
	}
}

// TestAgentState_SetNegativeStepPanics verifies FIX 26: AgentState.set
// panics on a negative step rather than silently clamping to 0. Silent
// clamps hid real bugs in earlier iterations; a panic makes the internal
// invariant violation loud. Callers inside goai never pass negatives.
func TestAgentState_SetNegativeStepPanics(t *testing.T) {
	var s AgentState
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("AgentState.set(-1) did not panic")
		}
		// FIX 31: panic value is now an error (was a string). Using error
		// allows errors.Is / errors.As at the recovery site; more idiomatic.
		err, ok := r.(error)
		if !ok {
			t.Fatalf("panic value is not an error: %T %v", r, r)
		}
		if !strings.Contains(err.Error(), "negative step") {
			t.Errorf("panic message = %q; want it to contain %q", err.Error(), "negative step")
		}
	}()
	s.set(StepLLMInFlight, -1)
}

// TestAgentState_NilModel_StillIdle verifies FIX 27: GenerateText and
// StreamText must transition a caller-provided AgentState to StepIdle even
// when they early-return on nil model. Otherwise a poller waiting for
// StepIdle would deadlock.
func TestAgentState_NilModel_StillIdle_Generate(t *testing.T) {
	defer goleak.VerifyNone(t)
	var state AgentState
	_, err := GenerateText(t.Context(), nil,
		WithPrompt("hi"),
		WithStateRef(&state),
	)
	if err == nil {
		t.Fatal("GenerateText(nil model) must return error")
	}
	if !strings.Contains(err.Error(), "model must not be nil") {
		t.Fatalf("unexpected err: %v", err)
	}
	kind, _ := state.Observe()
	if kind != StepIdle {
		t.Errorf("state=%v; want StepIdle after nil-model early return", kind)
	}
}

func TestAgentState_NilModel_StillIdle_Stream(t *testing.T) {
	defer goleak.VerifyNone(t)
	var state AgentState
	_, err := StreamText(t.Context(), nil,
		WithPrompt("hi"),
		WithStateRef(&state),
	)
	if err == nil {
		t.Fatal("StreamText(nil model) must return error")
	}
	if !strings.Contains(err.Error(), "model must not be nil") {
		t.Fatalf("unexpected err: %v", err)
	}
	kind, _ := state.Observe()
	if kind != StepIdle {
		t.Errorf("state=%v; want StepIdle after nil-model early return", kind)
	}
}

// TestAgentState_EmptyPrompt_StillIdle covers the other pre-loop early
// return (prompt/messages validation). FIX 27 already handled this path;
// we lock it in with an explicit test.
func TestAgentState_EmptyPrompt_StillIdle_Generate(t *testing.T) {
	defer goleak.VerifyNone(t)
	var state AgentState
	model := &mockModel{id: "m"}
	_, err := GenerateText(t.Context(), model, WithStateRef(&state))
	if err == nil {
		t.Fatal("GenerateText(empty prompt) must return error")
	}
	kind, _ := state.Observe()
	if kind != StepIdle {
		t.Errorf("state=%v; want StepIdle after empty-prompt early return", kind)
	}
}

func TestAgentState_EmptyPrompt_StillIdle_Stream(t *testing.T) {
	defer goleak.VerifyNone(t)
	var state AgentState
	model := &mockModel{id: "m"}
	_, err := StreamText(t.Context(), model, WithStateRef(&state))
	if err == nil {
		t.Fatal("StreamText(empty prompt) must return error")
	}
	kind, _ := state.Observe()
	if kind != StepIdle {
		t.Errorf("state=%v; want StepIdle after empty-prompt early return", kind)
	}
}

// TestStopSafe_AliasingContract documents FIX 30 as refined by FIX 36:
// stopSafe passes a shallow clone to the predicate. We assert TWO things
// about the resulting safety surface, so the test is actually
// load-bearing (rather than just re-asserting stdlib slices.Clone):
//
//  1. Top-level mutations (zero-out / reslice / append) are isolated.
//     This is the safety guarantee stopSafe provides on purpose.
//  2. Nested-slice mutations (StepResult.ToolCalls / ToolResults
//     element-writes) DO propagate to goai's internal state. This is
//     documented as a contract violation in WithStopWhen's godoc; the
//     test pins the actual behavior so a future deep-clone refactor
//     updates both the test and the godoc together.
//
// Without assertion #2 this test was tautological - it only re-tested
// stdlib `slices.Clone`'s documented behavior. The hazardous case (#2)
// is now exercised explicitly so the contract is visible from the
// test suite and accidental deep-clone regressions are caught.
func TestStopSafe_AliasingContract_ShallowCloneIsolatesTopLevel(t *testing.T) {
	// Populate nested slices with sentinel payloads so we can check
	// whether a predicate mutation to nested elements leaked back.
	originalCall := provider.ToolCall{ID: "call_1", Name: "orig", Input: json.RawMessage(`{}`)}
	originalResult := provider.ToolResult{ToolCallID: "call_1", ToolName: "orig", Output: "original"}
	internal := []StepResult{
		{
			Number:      1,
			Text:        "a",
			ToolCalls:   []provider.ToolCall{originalCall},
			ToolResults: []provider.ToolResult{originalResult},
		},
		{Number: 2, Text: "b"},
	}
	called := false
	pred := StopCondition(func(steps []StepResult) bool {
		called = true
		// (1) Top-level header mutations - must NOT propagate.
		for i := range steps {
			// Copy before zeroing so the nested-slice pointer we need
			// in step (2) is still reachable via the clone element.
			_ = steps[i]
		}
		// Reslice / append / zero; these operations may only affect
		// the predicate's local view of the slice header, never the
		// caller's `internal`.
		steps[0].Text = "CLOBBER_TOP_LEVEL"
		steps = append(steps[:0], StepResult{Number: 999, Text: "clobber"})
		_ = steps
		return false
	})
	stop := stopSafe(pred, internal)
	if stop {
		t.Fatalf("stopSafe returned true; predicate returned false")
	}
	if !called {
		t.Fatalf("predicate not invoked")
	}
	// (1a) Len untouched - resize did NOT leak.
	if len(internal) != 2 {
		t.Fatalf("internal len=%d; want 2 (shallow-clone did not isolate resize)", len(internal))
	}
	// (1b) The top-level field write - into a StepResult the clone
	// copied by value - must NOT be visible on internal.
	if internal[0].Number != 1 || internal[0].Text != "a" {
		t.Errorf("internal[0]=%+v; want {Number:1 Text:a} - top-level mutation leaked", internal[0])
	}
	if internal[1].Number != 2 || internal[1].Text != "b" {
		t.Errorf("internal[1]=%+v; want {Number:2 Text:b} - top-level mutation leaked", internal[1])
	}

	// (2) Nested-slice aliasing - hazardous contract. Run a SECOND
	// predicate invocation that intentionally mutates a nested slice
	// element. Assert mutation DOES propagate. This proves the
	// aliasing is real and motivates the WithStopWhen godoc "treat
	// contents as read-only" warning.
	hazardous := StopCondition(func(steps []StepResult) bool {
		if len(steps) > 0 && len(steps[0].ToolResults) > 0 {
			// Mutate the element of the nested slice (not the header).
			// The clone shares this backing array - the write is
			// visible on the caller side.
			steps[0].ToolResults[0].Output = "CORRUPTED_BY_PREDICATE"
		}
		return false
	})
	if stopSafe(hazardous, internal) {
		t.Fatalf("hazardous predicate returned false but stopSafe reported true")
	}
	if got := internal[0].ToolResults[0].Output; got != "CORRUPTED_BY_PREDICATE" {
		t.Fatalf("nested mutation did NOT propagate (got Output=%q); stopSafe must still use SHALLOW clone per FIX 30 / FIX 36 godoc. If you intentionally moved to deep clone, update this test AND provider/WithStopWhen godoc.", got)
	}
	// Reset for test hygiene - do NOT leave the corrupted sentinel
	// visible to any later assertions in this function.
	internal[0].ToolResults[0].Output = originalResult.Output
}

// ---------------------------------------------------------------------------
// FIX 34 - StreamText single-shot StateRef wiring tests.
// ---------------------------------------------------------------------------

// TestAgentState_StreamText_SingleShot_ReachesIdle exercises the single-shot
// streaming path (MaxSteps=1, no executable tools) and asserts the
// AgentState transitions through StepStarting → StepLLMInFlight → StepIdle
// without deadlocking pollers. Before FIX 34 the single-shot path never
// touched StateRef, leaving pollers stuck at (StepStarting, 0) indefinitely.
func TestAgentState_StreamText_SingleShot_ReachesIdle(t *testing.T) {
	defer goleak.VerifyNone(t)
	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hi"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}
	var state AgentState
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(1),
		WithStateRef(&state),
	)
	if err != nil {
		t.Fatal(err)
	}
	// Consume the stream to close.
	for range stream.Stream() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream Err: %v", err)
	}
	kind, step := state.Observe()
	if kind != StepIdle {
		t.Errorf("state kind=%v; want StepIdle after single-shot drain", kind)
	}
	if step != 1 {
		t.Errorf("state step=%d; want 1 (single step completed)", step)
	}
}

// TestAgentState_StreamText_SingleShot_EmptyStream_ReachesIdle is the
// empty-stream companion to FIX 34. A provider that emits zero chunks still
// must leave AgentState at StepIdle (otherwise pollers deadlock).
func TestAgentState_StreamText_SingleShot_EmptyStream_ReachesIdle(t *testing.T) {
	defer goleak.VerifyNone(t)
	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			ch := make(chan provider.StreamChunk)
			close(ch)
			return &provider.StreamResult{Stream: ch}, nil
		},
	}
	var state AgentState
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(1),
		WithStateRef(&state),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Result()
	kind, _ := state.Observe()
	if kind != StepIdle {
		t.Errorf("state kind=%v; want StepIdle after empty single-shot stream", kind)
	}
}

// TestAgentState_StreamText_SingleShot_DoStreamError_ReachesIdle covers the
// other inline return path in FIX 34: DoStream failing before the consume
// goroutine starts. Pollers waiting for StepIdle must still be released.
func TestAgentState_StreamText_SingleShot_DoStreamError_ReachesIdle(t *testing.T) {
	defer goleak.VerifyNone(t)
	wantErr := errors.New("upstream boom")
	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, wantErr
		},
	}
	var state AgentState
	_, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(1),
		WithStateRef(&state),
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err=%v; want wrapping wantErr", err)
	}
	kind, step := state.Observe()
	if kind != StepIdle {
		t.Errorf("state kind=%v; want StepIdle after DoStream error", kind)
	}
	// FIX 47: step=1 (not 0) - StreamText advertised (StepLLMInFlight, 1)
	// before the DoStream call; the post-error StepIdle must preserve the
	// same step count to keep the step counter monotonically non-decreasing.
	if step != 1 {
		t.Errorf("state step=%d; want 1 (monotonic: LLMInFlight=1 → Idle=1)", step)
	}
}

// TestAgentState_StreamText_SingleShot_NoTools_ReachesIdle exercises the
// alternative trigger for the single-shot path: tools are supplied but none
// has an Execute function, which also bypasses streamWithToolLoop.
func TestAgentState_StreamText_SingleShot_NoTools_ReachesIdle(t *testing.T) {
	defer goleak.VerifyNone(t)
	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "ok"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}
	var state AgentState
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(5),                   // >1, but toolMap will be empty
		WithTools(Tool{Name: "declared"}), // no Execute - bypasses tool loop
		WithStateRef(&state),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.Result()
	kind, _ := state.Observe()
	if kind != StepIdle {
		t.Errorf("state kind=%v; want StepIdle (single-shot path via no-executable-tools)", kind)
	}
}

// ---------------------------------------------------------------------------
// FIX 35 - GenerateObject / StreamObject emit WithStateRef warning.
// ---------------------------------------------------------------------------

// TestGenerateObject_StateRefIgnored_Warns verifies FIX 35: passing
// WithStateRef to GenerateObject emits a one-shot stderr warning (per
// process per entry point) and otherwise proceeds as if StateRef were
// absent. This mirrors the FIX 12 / FIX 21 pattern used for WithStopWhen.
func TestGenerateObject_StateRefIgnored_Warns(t *testing.T) {
	// Reset the once flag so this test sees a fresh atomic.Bool.
	// Pattern matches FIX 33 caveat: tests that reset these flags do
	// NOT use t.Parallel().
	generateObjectStateRefWarned.Store(false)
	var captured strings.Builder
	orig := warnStateRefIgnoredForObject
	t.Cleanup(func() { warnStateRefIgnoredForObject = orig })
	warnStateRefIgnoredForObject = func(fn string) {
		fmt.Fprintf(&captured, "warn:%s\n", fn)
	}
	// Not invoking the real function - just verify that GenerateObject
	// reaches the warn site. A mock model that returns an error keeps
	// the test hermetic (no real provider call / schema handling).
	model := &mockModel{
		id: "t",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, errors.New("mock short-circuit")
		},
	}
	var state AgentState
	type payload struct {
		X int `json:"x"`
	}
	_, _ = GenerateObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStateRef(&state),
	)
	if !strings.Contains(captured.String(), "warn:GenerateObject") {
		t.Errorf("expected warn:GenerateObject; got %q", captured.String())
	}
	// Second call with the same swapped warn: our replacement fires
	// every call (no once-check inside the mock), which is fine - the
	// real defaultWarnStateRefIgnoredForObject uses CompareAndSwap so
	// it fires at most once per process. We cover that in the next
	// sub-test below using the default warn.
	captured.Reset()
	_, _ = GenerateObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStateRef(&state),
	)
	if !strings.Contains(captured.String(), "warn:GenerateObject") {
		t.Errorf("expected warn:GenerateObject on second call too (test double fires every call); got %q", captured.String())
	}
}

// TestStreamObject_StateRefIgnored_Warns mirrors the GenerateObject test for
// the streaming structured-output entry point.
func TestStreamObject_StateRefIgnored_Warns(t *testing.T) {
	streamObjectStateRefWarned.Store(false)
	var captured strings.Builder
	orig := warnStateRefIgnoredForObject
	t.Cleanup(func() { warnStateRefIgnoredForObject = orig })
	warnStateRefIgnoredForObject = func(fn string) {
		fmt.Fprintf(&captured, "warn:%s\n", fn)
	}
	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, errors.New("mock short-circuit")
		},
	}
	var state AgentState
	type payload struct {
		X int `json:"x"`
	}
	_, _ = StreamObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStateRef(&state),
	)
	if !strings.Contains(captured.String(), "warn:StreamObject") {
		t.Errorf("expected warn:StreamObject; got %q", captured.String())
	}
}

// TestAgentState_Observe_NilReceiver covers the nil-receiver safety path
// (documented contract: a nil *AgentState Observe returns the zero state).
// Spike FIX 41 coverage pass - previously uncovered.
func TestAgentState_Observe_NilReceiver(t *testing.T) {
	var s *AgentState
	kind, step := s.Observe()
	if kind != StepStarting || step != 0 {
		t.Errorf("nil-receiver Observe()=(%v,%d); want (StepStarting,0)", kind, step)
	}
}

// TestStepKind_String covers the StepKind.String() humanizer. Spike
// FIX 41 coverage pass - previously at 0%.
func TestStepKind_String(t *testing.T) {
	cases := map[StepKind]string{
		StepStarting:      "starting",
		StepLLMInFlight:   "llm-in-flight",
		StepStepFinished:  "step-finished",
		StepToolExecuting: "tool-executing",
		StepIdle:          "idle",
		StepKind(999):     "unknown",
	}
	for k, want := range cases {
		if got := k.String(); got != want {
			t.Errorf("StepKind(%d).String()=%q; want %q", int(k), got, want)
		}
	}
}

// TestDefaultWarnStateRefIgnored_AllBranches covers StreamObject + default
// fn-name branches of defaultWarnStateRefIgnoredForObject to complete
// coverage of the warn switch. GenerateObject is covered by the sibling
// TestGenerateObject_StateRefWarn_OnceDefault.
func TestDefaultWarnStateRefIgnored_AllBranches(t *testing.T) {
	streamObjectStateRefWarned.Store(false)
	origStderr := osStderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	osStderr = w
	t.Cleanup(func() { osStderr = origStderr })

	// StreamObject branch - CAS once.
	defaultWarnStateRefIgnoredForObject("StreamObject")
	defaultWarnStateRefIgnoredForObject("StreamObject")
	// Default branch (unreachable in production, guarded in tests) --
	// fires unconditionally.
	defaultWarnStateRefIgnoredForObject("UnknownEntryPoint")
	_ = w.Close()

	var buf strings.Builder
	_, _ = io.Copy(&buf, r)
	_ = r.Close()
	got := buf.String()
	if n := strings.Count(got, "WithStateRef is not supported in StreamObject"); n != 1 {
		t.Errorf("StreamObject warn count=%d; want 1", n)
	}
	if !strings.Contains(got, "WithStateRef is not supported in UnknownEntryPoint") {
		t.Errorf("default-branch warn missing; got %q", got)
	}
}

// TestGenerateObject_StateRefWarn_OnceDefault exercises the real
// defaultWarnStateRefIgnoredForObject path and verifies the CAS-guarded
// once-per-process semantics by redirecting osStderr to a pipe.
func TestGenerateObject_StateRefWarn_OnceDefault(t *testing.T) {
	// Reset both once flags so counters start clean. Matches FIX 33
	// reset caveat: no t.Parallel() on this or its siblings.
	generateObjectStateRefWarned.Store(false)
	streamObjectStateRefWarned.Store(false)

	// Swap osStderr for a pipe so we can capture writes.
	origStderr := osStderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	osStderr = w
	t.Cleanup(func() { osStderr = origStderr })

	// Call the default warn twice; second call must be a no-op.
	defaultWarnStateRefIgnoredForObject("GenerateObject")
	defaultWarnStateRefIgnoredForObject("GenerateObject")
	_ = w.Close()

	var buf strings.Builder
	_, _ = io.Copy(&buf, r)
	_ = r.Close()

	got := buf.String()
	want := "goai: WithStateRef is not supported in GenerateObject"
	if !strings.Contains(got, want) {
		t.Fatalf("warning missing; got %q want substring %q", got, want)
	}
	// Exactly one occurrence - CompareAndSwap blocks the second.
	if n := strings.Count(got, want); n != 1 {
		t.Errorf("warning fired %d times; want 1 (CAS-guarded once)", n)
	}
}

// ---------------------------------------------------------------------------
// FIX 44 - ctx-cancel mid-stream on single-shot + StateRef reaches Idle.
// ---------------------------------------------------------------------------

// TestAgentState_StreamText_SingleShot_CtxCancel_ReachesIdle verifies that
// when the caller cancels ctx mid-stream on a single-shot StreamText call
// wired with WithStateRef, the consume goroutine still terminates cleanly
// and publishes StepIdle so pollers are not stuck at StepLLMInFlight. No
// goroutine leak.
func TestAgentState_StreamText_SingleShot_CtxCancel_ReachesIdle(t *testing.T) {
	defer goleak.VerifyNone(t)

	// Provider stream emits chunks then closes. We use Result() (nil/nil
	// branch in consume) so consume checks ctx.Err() after each chunk.
	// Cancel ctx before consume runs; when consume processes the first
	// chunk and hits its post-switch ctx check, it returns - advancing
	// StateRef to StepIdle.
	providerCh := make(chan provider.StreamChunk, 2)
	providerCh <- provider.StreamChunk{Type: provider.ChunkText, Text: "hi"}
	providerCh <- provider.StreamChunk{Type: provider.ChunkText, Text: "post-cancel"}
	close(providerCh)

	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return &provider.StreamResult{Stream: providerCh}, nil
		},
	}

	ctx, cancel := context.WithCancel(t.Context())
	var state AgentState
	stream, err := StreamText(ctx, model,
		WithPrompt("go"),
		WithMaxSteps(1),
		WithStateRef(&state),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Cancel ctx before Result() starts consume. consume's per-iteration
	// nil/nil branch will observe ctx.Err() != nil after processing the
	// first chunk and return early, exercising the mid-stream-cancel
	// StepIdle transition.
	cancel()
	_ = stream.Result()
	// Err must surface the ctx error.
	if gotErr := stream.Err(); !errors.Is(gotErr, context.Canceled) {
		t.Errorf("stream.Err()=%v; want context.Canceled", gotErr)
	}

	kind, step := state.Observe()
	if kind != StepIdle {
		t.Errorf("state kind=%v; want StepIdle after ctx-cancel mid-stream", kind)
	}
	if step != 1 {
		t.Errorf("state step=%d; want 1 (monotonic: StepLLMInFlight=1 → StepIdle=1)", step)
	}
}

// ---------------------------------------------------------------------------
// FIX 45 - mid-stream ChunkError on single-shot still reaches Idle.
// ---------------------------------------------------------------------------

// TestAgentState_StreamText_SingleShot_ChunkError_ReachesIdle verifies that
// a provider emitting a ChunkError mid-stream leaves AgentState at
// StepIdle and does not leak the consume goroutine.
func TestAgentState_StreamText_SingleShot_ChunkError_ReachesIdle(t *testing.T) {
	defer goleak.VerifyNone(t)

	boom := errors.New("mid-stream boom")
	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "partial"},
				provider.StreamChunk{Type: provider.ChunkError, Error: boom},
				// No ChunkFinish - provider aborted.
			), nil
		},
	}

	var state AgentState
	stream, err := StreamText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(1),
		WithStateRef(&state),
	)
	if err != nil {
		t.Fatal(err)
	}
	for range stream.Stream() {
	}
	// stream.Err() should surface the mid-stream error.
	if gotErr := stream.Err(); !errors.Is(gotErr, boom) {
		t.Errorf("stream.Err()=%v; want %v", gotErr, boom)
	}
	kind, step := state.Observe()
	if kind != StepIdle {
		t.Errorf("state kind=%v; want StepIdle after mid-stream ChunkError", kind)
	}
	if step != 1 {
		t.Errorf("state step=%d; want 1 (monotonic; single-shot stream step is 1)", step)
	}
}

// ---------------------------------------------------------------------------
// FIX 46 - combined WithStopWhen + WithStateRef independent warnings.
// ---------------------------------------------------------------------------

// TestGenerateObject_BothWarnings_FireOnceEach_WhenBothPassed verifies that
// passing BOTH WithStopWhen and WithStateRef in the SAME call emits BOTH
// warnings once each (idempotent across a second call). Compare with
// TestGenerateObject_Warnings_LatchesAreIndependent below, which exercises
// cross-latch independence (one option at a time across calls).
func TestGenerateObject_BothWarnings_FireOnceEach_WhenBothPassed(t *testing.T) {
	// Reset both latches (FIX 33 caveat - no t.Parallel).
	generateObjectStopWhenWarned.Store(false)
	generateObjectStateRefWarned.Store(false)

	origStderr := osStderr
	r, w, perr := os.Pipe()
	if perr != nil {
		t.Fatalf("pipe: %v", perr)
	}
	osStderr = w
	t.Cleanup(func() { osStderr = origStderr })

	model := &mockModel{
		id: "t",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, errors.New("mock short-circuit")
		},
	}
	var state AgentState
	type payload struct {
		X int `json:"x"`
	}

	// First call: both warnings fire.
	_, _ = GenerateObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStopWhen(func([]StepResult) bool { return false }),
		WithStateRef(&state),
	)
	// Second call: both latches already tripped - zero additional warnings.
	_, _ = GenerateObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStopWhen(func([]StepResult) bool { return false }),
		WithStateRef(&state),
	)

	_ = w.Close()
	var buf strings.Builder
	_, _ = io.Copy(&buf, r)
	_ = r.Close()
	got := buf.String()

	stopWhenWant := "goai: WithStopWhen is not supported in GenerateObject"
	stateRefWant := "goai: WithStateRef is not supported in GenerateObject"
	if n := strings.Count(got, stopWhenWant); n != 1 {
		t.Errorf("WithStopWhen warning fired %d times across 2 calls; want 1; full stderr=%q", n, got)
	}
	if n := strings.Count(got, stateRefWant); n != 1 {
		t.Errorf("WithStateRef warning fired %d times across 2 calls; want 1; full stderr=%q", n, got)
	}
}

// TestStreamObject_BothWarnings_FireOnceEach_WhenBothPassed is the StreamObject
// twin of the GenerateObject both-warnings test above.
func TestStreamObject_BothWarnings_FireOnceEach_WhenBothPassed(t *testing.T) {
	streamObjectStopWhenWarned.Store(false)
	streamObjectStateRefWarned.Store(false)

	origStderr := osStderr
	r, w, perr := os.Pipe()
	if perr != nil {
		t.Fatalf("pipe: %v", perr)
	}
	osStderr = w
	t.Cleanup(func() { osStderr = origStderr })

	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, errors.New("mock short-circuit")
		},
	}
	var state AgentState
	type payload struct {
		X int `json:"x"`
	}

	_, _ = StreamObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStopWhen(func([]StepResult) bool { return false }),
		WithStateRef(&state),
	)
	_, _ = StreamObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStopWhen(func([]StepResult) bool { return false }),
		WithStateRef(&state),
	)

	_ = w.Close()
	var buf strings.Builder
	_, _ = io.Copy(&buf, r)
	_ = r.Close()
	got := buf.String()

	stopWhenWant := "goai: WithStopWhen is not supported in StreamObject"
	stateRefWant := "goai: WithStateRef is not supported in StreamObject"
	if n := strings.Count(got, stopWhenWant); n != 1 {
		t.Errorf("WithStopWhen warning fired %d times across 2 calls; want 1; full stderr=%q", n, got)
	}
	if n := strings.Count(got, stateRefWant); n != 1 {
		t.Errorf("WithStateRef warning fired %d times across 2 calls; want 1; full stderr=%q", n, got)
	}
}

// ---------------------------------------------------------------------------
// FIX 51 - cross-latch independence. The StopWhen latch must NOT affect the
// StateRef latch and vice versa when the options are passed one at a time
// across a sequence of calls.
// ---------------------------------------------------------------------------

// TestGenerateObject_Warnings_LatchesAreIndependent proves each of the two
// once-latches (StopWhen, StateRef) advances on its own and does not
// cross-trip the other. Sequence: call 1 with only WithStopWhen → exactly
// one "WithStopWhen" warning (no "WithStateRef"). Call 2 with only
// WithStateRef → exactly one "WithStateRef" warning (no additional
// "WithStopWhen"). Call 3 with both → zero additional warnings.
func TestGenerateObject_Warnings_LatchesAreIndependent(t *testing.T) {
	generateObjectStopWhenWarned.Store(false)
	generateObjectStateRefWarned.Store(false)

	origStderr := osStderr
	r, w, perr := os.Pipe()
	if perr != nil {
		t.Fatalf("pipe: %v", perr)
	}
	osStderr = w
	t.Cleanup(func() { osStderr = origStderr })

	model := &mockModel{
		id: "t",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return nil, errors.New("mock short-circuit")
		},
	}
	var state AgentState
	type payload struct {
		X int `json:"x"`
	}

	// Call 1: WithStopWhen only.
	_, _ = GenerateObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStopWhen(func([]StepResult) bool { return false }),
	)
	// Call 2: WithStateRef only (StopWhen latch already tripped; StateRef latch fresh).
	_, _ = GenerateObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStateRef(&state),
	)
	// Call 3: both - both latches now tripped; no new warnings.
	_, _ = GenerateObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStopWhen(func([]StepResult) bool { return false }),
		WithStateRef(&state),
	)

	_ = w.Close()
	var buf strings.Builder
	_, _ = io.Copy(&buf, r)
	_ = r.Close()
	got := buf.String()

	stopWhenWant := "goai: WithStopWhen is not supported in GenerateObject"
	stateRefWant := "goai: WithStateRef is not supported in GenerateObject"
	if n := strings.Count(got, stopWhenWant); n != 1 {
		t.Errorf("WithStopWhen warning fired %d times across 3 calls; want 1; full stderr=%q", n, got)
	}
	if n := strings.Count(got, stateRefWant); n != 1 {
		t.Errorf("WithStateRef warning fired %d times across 3 calls; want 1; full stderr=%q", n, got)
	}
}

// TestStreamObject_Warnings_LatchesAreIndependent is the StreamObject twin.
func TestStreamObject_Warnings_LatchesAreIndependent(t *testing.T) {
	streamObjectStopWhenWarned.Store(false)
	streamObjectStateRefWarned.Store(false)

	origStderr := osStderr
	r, w, perr := os.Pipe()
	if perr != nil {
		t.Fatalf("pipe: %v", perr)
	}
	osStderr = w
	t.Cleanup(func() { osStderr = origStderr })

	model := &mockModel{
		id: "t",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return nil, errors.New("mock short-circuit")
		},
	}
	var state AgentState
	type payload struct {
		X int `json:"x"`
	}

	_, _ = StreamObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStopWhen(func([]StepResult) bool { return false }),
	)
	_, _ = StreamObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStateRef(&state),
	)
	_, _ = StreamObject[payload](t.Context(), model,
		WithPrompt("go"),
		WithStopWhen(func([]StepResult) bool { return false }),
		WithStateRef(&state),
	)

	_ = w.Close()
	var buf strings.Builder
	_, _ = io.Copy(&buf, r)
	_ = r.Close()
	got := buf.String()

	stopWhenWant := "goai: WithStopWhen is not supported in StreamObject"
	stateRefWant := "goai: WithStateRef is not supported in StreamObject"
	if n := strings.Count(got, stopWhenWant); n != 1 {
		t.Errorf("WithStopWhen warning fired %d times across 3 calls; want 1; full stderr=%q", n, got)
	}
	if n := strings.Count(got, stateRefWant); n != 1 {
		t.Errorf("WithStateRef warning fired %d times across 3 calls; want 1; full stderr=%q", n, got)
	}
}

// ---------------------------------------------------------------------------
// FIX 47 - step counter monotonicity on error after StepLLMInFlight.
// ---------------------------------------------------------------------------

// TestAgentState_StreamText_ErrorAfterLLMInFlight_MonotonicStep pins the
// FIX 47 invariant: once goai has advertised (StepLLMInFlight, N) via the
// StateRef, no subsequent store may regress the step counter below N.
// Before FIX 47, the initial DoStream error path published (StepIdle, 0)
// immediately after (StepLLMInFlight, 1), so a racing poller could observe
// step 1 → 0.
func TestAgentState_StreamText_ErrorAfterLLMInFlight_MonotonicStep(t *testing.T) {
	defer goleak.VerifyNone(t)

	// Single-shot path (MaxSteps=1, no tools).
	t.Run("single-shot/DoStream-error", func(t *testing.T) {
		model := &mockModel{
			id: "t",
			streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
				return nil, errors.New("upstream refused")
			},
		}
		var state AgentState
		_, err := StreamText(t.Context(), model,
			WithPrompt("go"),
			WithMaxSteps(1),
			WithStateRef(&state),
		)
		if err == nil {
			t.Fatal("expected error from failing DoStream")
		}
		kind, step := state.Observe()
		if kind != StepIdle {
			t.Errorf("state kind=%v; want StepIdle", kind)
		}
		if step != 1 {
			t.Errorf("state step=%d; want 1 (monotonic: LLMInFlight=1 → Idle=1, no regression to 0)", step)
		}
	})

	// Multi-step tool-loop path: initial DoStream failure.
	t.Run("tool-loop/initial-DoStream-error", func(t *testing.T) {
		model := &mockModel{
			id: "t",
			streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
				return nil, errors.New("upstream refused")
			},
		}
		var state AgentState
		_, err := StreamText(t.Context(), model,
			WithPrompt("go"),
			WithMaxSteps(5),
			WithTools(Tool{
				Name:    "noop",
				Execute: func(_ context.Context, _ json.RawMessage) (string, error) { return "", nil },
			}),
			WithStateRef(&state),
		)
		if err == nil {
			t.Fatal("expected error from failing DoStream")
		}
		kind, step := state.Observe()
		if kind != StepIdle {
			t.Errorf("state kind=%v; want StepIdle", kind)
		}
		if step != 1 {
			t.Errorf("state step=%d; want 1 (monotonic)", step)
		}
	})

	// GenerateText: DoGenerate error on step 1 must NOT regress step 1 → 0.
	t.Run("generate-text/DoGenerate-error", func(t *testing.T) {
		model := &mockModel{
			id: "t",
			generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
				return nil, errors.New("upstream refused")
			},
		}
		var state AgentState
		_, err := GenerateText(t.Context(), model,
			WithPrompt("go"),
			WithMaxSteps(1),
			WithStateRef(&state),
		)
		if err == nil {
			t.Fatal("expected error from failing DoGenerate")
		}
		kind, step := state.Observe()
		if kind != StepIdle {
			t.Errorf("state kind=%v; want StepIdle", kind)
		}
		if step != 1 {
			t.Errorf("state step=%d; want 1 (monotonic: LLMInFlight=1 → Idle=1)", step)
		}
	})

	// Aggressive stress: many concurrent pollers must never observe step
	// going backwards across the LLMInFlight → Idle transition.
	t.Run("race/many-pollers-observe-monotonic-step", func(t *testing.T) {
		for trial := 0; trial < 50; trial++ {
			model := &mockModel{
				id: "t",
				streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
					return nil, errors.New("boom")
				},
			}
			var state AgentState
			var pollerWG sync.WaitGroup
			stop := make(chan struct{})
			var violated atomic.Bool
			for i := 0; i < 4; i++ {
				pollerWG.Add(1)
				go func() {
					defer pollerWG.Done()
					var lastStep int
					for {
						select {
						case <-stop:
							return
						default:
						}
						_, s := state.Observe()
						if s < lastStep {
							violated.Store(true)
							return
						}
						lastStep = s
					}
				}()
			}
			_, _ = StreamText(t.Context(), model,
				WithPrompt("go"),
				WithMaxSteps(1),
				WithStateRef(&state),
			)
			close(stop)
			pollerWG.Wait()
			if violated.Load() {
				t.Fatalf("trial %d: poller observed step regression", trial)
			}
		}
	})

	// FIX 50: the single-shot (MaxSteps=1) race subtest above is tautological
	// for the load-bearing monotonicity path - a single-shot failure has
	// highestInflightStep == len(steps) (both effectively 1), so the
	// `highestInflightStep > finalStep` branch in the StepIdle defer is
	// never exercised. This subtest exercises the genuine mid-loop case:
	// step 1 succeeds with a tool call (steps gets length 1), then step 2's
	// DoStream errors BEFORE a StepResult is appended (len(steps) stays 1,
	// highestInflightStep reaches 2). Pollers must observe step monotonically
	// advance from 1 → 2 at Idle, never regressing.
	t.Run("race/mid-loop-error-monotonic-step", func(t *testing.T) {
		for trial := 0; trial < 50; trial++ {
			var callCount atomic.Int32
			model := &mockModel{
				id: "t",
				streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
					n := callCount.Add(1)
					if n == 1 {
						// Step 1: success with one tool call → tool executes,
						// loop proceeds to step 2.
						return streamFromChunks(
							provider.StreamChunk{Type: provider.ChunkToolCall, ToolCallID: "tc1", ToolName: "ping", ToolInput: `{}`},
							provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishToolCalls},
						), nil
					}
					// Step 2+: DoStream itself fails (error before any StepResult).
					return nil, errors.New("step2 upstream refused")
				},
			}
			var state AgentState
			var pollerWG sync.WaitGroup
			stop := make(chan struct{})
			var violated atomic.Bool
			var maxObserved atomic.Int32
			for i := 0; i < 8; i++ {
				pollerWG.Add(1)
				go func() {
					defer pollerWG.Done()
					var lastStep int
					for {
						select {
						case <-stop:
							return
						default:
						}
						_, s := state.Observe()
						if s < lastStep {
							violated.Store(true)
							return
						}
						if int32(s) > maxObserved.Load() {
							maxObserved.Store(int32(s))
						}
						lastStep = s
					}
				}()
			}
			stream, err := StreamText(t.Context(), model,
				WithPrompt("go"),
				WithMaxSteps(3),
				WithTools(Tool{
					Name:        "ping",
					Description: "ping",
					InputSchema: json.RawMessage(`{"type":"object"}`),
					Execute: func(_ context.Context, _ json.RawMessage) (string, error) {
						return "pong", nil
					},
				}),
				WithStateRef(&state),
			)
			// StreamText returns (stream, nil) even if a subsequent step
			// errors - the error surfaces via ChunkError inside the stream.
			if err != nil {
				// Unexpected: step 1 succeeded so StreamText should have
				// returned a live stream.
				close(stop)
				pollerWG.Wait()
				t.Fatalf("trial %d: unexpected initial StreamText error: %v", trial, err)
			}
			// Drain the stream so the background goroutine completes and
			// publishes StepIdle. Without this, the tool-loop goroutine is
			// still running when we check final state below.
			for range stream.Stream() {
			}
			// Additionally wait for the done signal - raw channel close
			// fires BEFORE StepIdle store (documented in WithStateRef godoc).
			_ = stream.Err()
			close(stop)
			pollerWG.Wait()
			if violated.Load() {
				t.Fatalf("trial %d: poller observed step regression across LLMInFlight→Idle transition", trial)
			}
			kind, finalStep := state.Observe()
			if kind != StepIdle {
				t.Errorf("trial %d: final kind=%v; want StepIdle", trial, kind)
			}
			// FIX 47 invariant: finalStep must reflect the highest
			// announced in-flight step (2), NOT len(steps) (which is 1
			// because step 2 erred before a StepResult was appended).
			if finalStep < 2 {
				t.Errorf("trial %d: finalStep=%d; want >= 2 (highestInflightStep tracked step-2 announcement)", trial, finalStep)
			}
			// Sanity: pollers must have seen the advance to step 2 at
			// some point (not merely read it once at exit).
			if maxObserved.Load() < 2 {
				t.Errorf("trial %d: maxObserved=%d; pollers never saw step 2 advancing", trial, maxObserved.Load())
			}
		}
	})
}
