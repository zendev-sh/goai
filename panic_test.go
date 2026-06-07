package goai

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

func TestPanicError_ErrorAndUnwrap(t *testing.T) {
	cause := errors.New("boom")
	pe := &PanicError{Phase: "OnFinish", Value: cause}
	// Error() identifies the phase only; the raw value is omitted to avoid
	// leaking sensitive data into logged error strings.
	if got := pe.Error(); got != "goai: panic in OnFinish" {
		t.Errorf("Error() = %q, want %q", got, "goai: panic in OnFinish")
	}
	if strings.Contains(pe.Error(), "boom") {
		t.Error("Error() must not contain the raw panic value")
	}
	if !errors.Is(pe, cause) {
		t.Error("errors.Is(pe, cause) = false; Unwrap should expose an error panic value")
	}

	// A non-error panic value unwraps to nil.
	peStr := &PanicError{Phase: "OnStepFinish", Value: "plain string"}
	if peStr.Unwrap() != nil {
		t.Errorf("Unwrap() = %v, want nil for non-error value", peStr.Unwrap())
	}
}

// TestOnPanic_FiresWithInfo verifies the OnPanic hook receives PanicInfo with
// the phase, value, and a non-empty stack for a propagate-fatal hook.
func TestOnPanic_FiresWithInfo(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	var got PanicInfo
	var fired atomic.Int32
	_, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithOnStepFinish(func(_ StepResult) { panic("kaboom") }),
		WithOnPanic(func(info PanicInfo) {
			got = info
			fired.Add(1)
		}),
	)

	var pe *PanicError
	if !errors.As(err, &pe) {
		t.Fatalf("err = %v, want *PanicError", err)
	}
	if fired.Load() != 1 {
		t.Fatalf("OnPanic fired %d times, want 1", fired.Load())
	}
	if got.Phase != "OnStepFinish" {
		t.Errorf("Phase = %q, want OnStepFinish", got.Phase)
	}
	if got.Value != "kaboom" {
		t.Errorf("Value = %v, want kaboom", got.Value)
	}
	if len(got.Stack) == 0 {
		t.Error("Stack is empty, want a captured goroutine stack")
	}
}

// TestOnPanic_FiresForToolPath verifies OnPanic fires for a tool Execute panic
// even though the tool path stays resilient (no *PanicError propagation).
func TestOnPanic_FiresForToolPath(t *testing.T) {
	callCount := 0
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			callCount++
			if callCount == 1 {
				return &provider.GenerateResult{
					ToolCalls:    []provider.ToolCall{{ID: "tc1", Name: "boomtool", Input: json.RawMessage(`{}`)}},
					FinishReason: provider.FinishToolCalls,
				}, nil
			}
			return &provider.GenerateResult{Text: "done", FinishReason: provider.FinishStop}, nil
		},
	}

	var got PanicInfo
	var fired atomic.Int32
	result, err := GenerateText(t.Context(), model,
		WithPrompt("go"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "boomtool",
			Execute: func(_ context.Context, _ json.RawMessage) (string, error) { panic("tool boom") },
		}),
		WithOnPanic(func(info PanicInfo) {
			got = info
			fired.Add(1)
		}),
	)
	// Tool panic does NOT propagate: the loop continues and returns normally.
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "done" {
		t.Errorf("Text = %q, want done (loop should continue past a tool panic)", result.Text)
	}
	if fired.Load() != 1 {
		t.Fatalf("OnPanic fired %d times, want 1", fired.Load())
	}
	if got.Phase != "tool:boomtool" {
		t.Errorf("Phase = %q, want tool:boomtool", got.Phase)
	}
	if got.Value != "tool boom" {
		t.Errorf("Value = %v, want 'tool boom'", got.Value)
	}
}

// TestOnPanic_HandlerPanicIsContained verifies that a panic inside an OnPanic
// callback itself is swallowed and does not disrupt the *PanicError propagation.
func TestOnPanic_HandlerPanicIsContained(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	_, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithOnFinish(func(_ FinishInfo) { panic("original") }),
		WithOnPanic(func(_ PanicInfo) { panic("handler also panics") }),
	)

	var pe *PanicError
	if !errors.As(err, &pe) {
		t.Fatalf("err = %v, want *PanicError", err)
	}
	if pe.Phase != "OnFinish" || pe.Value != "original" {
		t.Errorf("PanicError = {%q, %v}, want {OnFinish, original}", pe.Phase, pe.Value)
	}
}

// --- Direct unit tests for the panic helpers' defensive branches ---

// callHook should re-panic an already-wrapped *PanicError as-is, without firing
// OnPanic a second time.
func TestCallHook_NestedPanicErrorPassesThrough(t *testing.T) {
	orig := &PanicError{Phase: "Inner", Value: "x"}
	var fired int
	onPanic := []func(PanicInfo){func(PanicInfo) { fired++ }}

	defer func() {
		r := recover()
		if r != orig {
			t.Fatalf("re-panicked %v, want the original *PanicError", r)
		}
		if fired != 0 {
			t.Errorf("OnPanic fired %d times, want 0 (already wrapped)", fired)
		}
	}()
	callHook(onPanic, "Outer", func() { panic(orig) })
}

// recoverToError converts a *PanicError to the error pointer as-is, and wraps
// any other panic value as a *PanicError(phase="internal") rather than letting
// it escape as an uncaught crash.
func TestRecoverToError(t *testing.T) {
	// *PanicError -> assigned, phase preserved.
	func() {
		var err error
		defer func() {
			var pe *PanicError
			if !errors.As(err, &pe) {
				t.Fatalf("err = %v, want *PanicError", err)
			}
			if pe.Phase != "X" {
				t.Errorf("Phase = %q, want X", pe.Phase)
			}
		}()
		defer recoverToError(nil, &err)
		panic(&PanicError{Phase: "X", Value: "y"})
	}()

	// Non-*PanicError -> wrapped as internal, OnPanic fired, no re-panic escapes.
	func() {
		var err error
		var fired int
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("recoverToError must not re-panic; got %v", r)
			}
			var pe *PanicError
			if !errors.As(err, &pe) {
				t.Fatalf("err = %v, want *PanicError", err)
			}
			if pe.Phase != "internal" || pe.Value != "raw" {
				t.Errorf("PanicError = {%q, %v}, want {internal, raw}", pe.Phase, pe.Value)
			}
			if fired != 1 {
				t.Errorf("OnPanic fired %d times, want 1", fired)
			}
		}()
		defer recoverToError([]func(PanicInfo){func(PanicInfo) { fired++ }}, &err)
		panic("raw")
	}()
}

// recoverToStreamErr stores a *PanicError directly and wraps a raw panic value
// (firing OnPanic) so a goroutine panic never escapes.
func TestRecoverToStreamErr(t *testing.T) {
	// *PanicError -> stored verbatim, OnPanic not re-fired.
	func() {
		var got error
		var fired int
		defer func() {
			var pe *PanicError
			if !errors.As(got, &pe) || pe.Phase != "X" {
				t.Errorf("got = %v, want *PanicError{Phase:X}", got)
			}
			if fired != 0 {
				t.Errorf("OnPanic fired %d times, want 0", fired)
			}
		}()
		defer recoverToStreamErr([]func(PanicInfo){func(PanicInfo) { fired++ }}, "stream", func(e error) { got = e })
		panic(&PanicError{Phase: "X", Value: "y"})
	}()

	// Raw value -> wrapped with the fallback phase, OnPanic fired.
	func() {
		var got error
		var fired int
		defer func() {
			var pe *PanicError
			if !errors.As(got, &pe) {
				t.Fatalf("got = %v, want *PanicError", got)
			}
			if pe.Phase != "stream" || pe.Value != "raw" {
				t.Errorf("PanicError = {%q, %v}, want {stream, raw}", pe.Phase, pe.Value)
			}
			if fired != 1 {
				t.Errorf("OnPanic fired %d times, want 1", fired)
			}
		}()
		defer recoverToStreamErr([]func(PanicInfo){func(PanicInfo) { fired++ }}, "stream", func(e error) { got = e })
		panic("raw")
	}()
}

// TestStreamText_Step1OnRequestPanic verifies that a panic in the synchronous
// step-1 OnRequest hook (single-step StreamText) is surfaced as a *PanicError
// returned by StreamText, not propagated raw into the caller's goroutine.
func TestStreamText_Step1OnRequestPanic(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hi"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	var fired int
	_, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithOnRequest(func(_ RequestInfo) { panic("step1 onrequest boom") }),
		WithOnPanic(func(_ PanicInfo) { fired++ }),
	)
	var pe *PanicError
	if !errors.As(err, &pe) {
		t.Fatalf("err = %v, want *PanicError", err)
	}
	if pe.Phase != "OnRequest" {
		t.Errorf("Phase = %q, want OnRequest", pe.Phase)
	}
	if fired != 1 {
		t.Errorf("OnPanic fired %d times, want 1", fired)
	}
}

// TestStreamText_ToolLoop_Step1OnRequestPanic verifies the same for the
// multi-step path (streamWithToolLoop), where step-1 hooks also run
// synchronously in the caller's goroutine.
func TestStreamText_ToolLoop_Step1OnRequestPanic(t *testing.T) {
	model := &mockModel{
		id: "test",
		streamFn: func(_ context.Context, _ provider.GenerateParams) (*provider.StreamResult, error) {
			return streamFromChunks(
				provider.StreamChunk{Type: provider.ChunkText, Text: "hi"},
				provider.StreamChunk{Type: provider.ChunkFinish, FinishReason: provider.FinishStop},
			), nil
		},
	}

	_, err := StreamText(t.Context(), model,
		WithPrompt("hi"),
		WithMaxSteps(3),
		WithTools(Tool{
			Name:    "t",
			Execute: func(context.Context, json.RawMessage) (string, error) { return "ok", nil },
		}),
		WithOnRequest(func(_ RequestInfo) { panic("step1 onrequest boom") }),
	)
	var pe *PanicError
	if !errors.As(err, &pe) {
		t.Fatalf("err = %v, want *PanicError", err)
	}
	if pe.Phase != "OnRequest" {
		t.Errorf("Phase = %q, want OnRequest", pe.Phase)
	}
}

// TestOnPanic_MultipleObserversInOrder verifies all registered OnPanic hooks fire.
func TestOnPanic_MultipleObserversInOrder(t *testing.T) {
	model := &mockModel{
		id: "test",
		generateFn: func(_ context.Context, _ provider.GenerateParams) (*provider.GenerateResult, error) {
			return &provider.GenerateResult{Text: "ok", FinishReason: provider.FinishStop}, nil
		},
	}

	var order []string
	_, err := GenerateText(t.Context(), model,
		WithPrompt("hi"),
		WithOnStepFinish(func(_ StepResult) { panic("x") }),
		WithOnPanic(func(_ PanicInfo) { order = append(order, "a") }),
		WithOnPanic(func(_ PanicInfo) { order = append(order, "b") }),
	)
	if err == nil {
		t.Fatal("want *PanicError, got nil")
	}
	if strings.Join(order, ",") != "a,b" {
		t.Errorf("OnPanic order = %v, want [a b]", order)
	}
}
