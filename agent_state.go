package goai

import (
	"fmt"
	"sync/atomic"
)

// StepKind identifies the lifecycle phase of the tool loop at a given moment.
//
// AgentState (observed via Observe) transitions through these kinds as the
// goai tool loop progresses. Consumers poll from a separate goroutine to
// decide whether it is safe to inject work / send wake signals without
// interrupting an in-flight LLM call or tool execution.
type StepKind int

const (
	// StepStarting is the initial state before the first LLM call begins.
	StepStarting StepKind = iota

	// StepLLMInFlight indicates an LLM request is in flight (sync or stream).
	// For StreamText this covers the period from DoStream dispatch until the
	// last chunk of the step has been drained.
	StepLLMInFlight

	// StepStepFinished indicates the LLM call for the current step has fully
	// returned (DoGenerate returned / the stream's ChunkStepFinish has been
	// emitted) and the step result has been recorded, but tool execution has
	// not yet begun and loop-termination predicates (WithStopWhen,
	// OnBeforeStep) have not yet decided whether to continue. This state
	// exists so that pollers observing between phases cannot mistake the
	// post-LLM / pre-tool window for "LLM still in flight".
	StepStepFinished

	// StepToolExecuting indicates one or more tool Execute functions are
	// running. It is entered after the LLM's stepN response is fully drained
	// and exited when all parallel tool calls return.
	StepToolExecuting

	// StepIdle indicates the tool loop has terminated (either naturally, by
	// MaxSteps exhaustion, by OnBeforeStep.Stop, or by WithStopWhen returning
	// true). Once StepIdle is observed, the state will not transition again.
	StepIdle
)

// String returns a human-readable name for the kind.
func (k StepKind) String() string {
	switch k {
	case StepStarting:
		return "starting"
	case StepLLMInFlight:
		return "llm-in-flight"
	case StepStepFinished:
		return "step-finished"
	case StepToolExecuting:
		return "tool-executing"
	case StepIdle:
		return "idle"
	default:
		return "unknown"
	}
}

// AgentState is a race-free observable handle into the tool-loop's lifecycle
// state. Zero value is usable and reports (StepStarting, 0) until goai begins
// mutating it.
//
// Concurrency: goai writes atomically from the tool-loop goroutine; consumers
// may poll Observe concurrently from any number of goroutines.
//
// Layout: we pack (step:uint32, kind:uint32) into a single atomic.Uint64 so
// that both fields advance together without tearing.
type AgentState struct {
	// packed holds step in the high 32 bits and kind in the low 32 bits.
	packed atomic.Uint64
}

// Observe returns the current (kind, step) of the agent.
//
// Step semantics:
//   - Before first LLM call: step == 0.
//   - During StepLLMInFlight / StepStepFinished / StepToolExecuting:
//     step == the 1-indexed current step number.
//   - At StepIdle: step == the highest step announced via StepLLMInFlight
//     (FIX 47 monotonicity invariant). NOTE: at StepIdle, `step` may EXCEED
//     `len(TextResult.Steps)` when an in-flight step errored before being
//     appended to the step list - the counter was advanced for
//     StepLLMInFlight but the StepResult was never recorded because the
//     error short-circuited the step. Consumers tracking "how many steps
//     completed" should use `len(result.Steps)`, not `state.Observe()`'s
//     step return.
func (s *AgentState) Observe() (kind StepKind, step int) {
	if s == nil {
		return StepStarting, 0
	}
	v := s.packed.Load()
	return StepKind(uint32(v)), int(uint32(v >> 32))
}

// set stores (kind, step) atomically. A negative step is a programming
// error (goai's internal tool loop never passes one) and panics loudly
// rather than silently clamping - silent clamps hid real bugs in earlier
// iterations. Exported API callers cannot reach this function; only goai
// itself calls set.
func (s *AgentState) set(kind StepKind, step int) {
	if s == nil {
		return
	}
	if step < 0 {
		// Panic with an error value (not a string) so recoverers can use
		// errors.Is / errors.As and so the standard panic-recovery style
		// (r, ok := r.(error)) works. The Sprintf form loses type information.
		panic(fmt.Errorf("goai: negative step %d in AgentState.set (internal invariant violated)", step))
	}
	packed := (uint64(uint32(step)) << 32) | uint64(uint32(kind))
	s.packed.Store(packed)
}

// WithStateRef exposes goai's tool-loop lifecycle state via an
// externally-owned AgentState. The caller allocates the value and retains a
// pointer; goai mutates it atomically as the loop progresses.
//
// Typical use (zenflow poller):
//
//	var state goai.AgentState
//	go pollLoop(&state)
//	_, err := goai.GenerateText(ctx, model, goai.WithStateRef(&state), ...)
//
// ref must be non-nil; nil refs are ignored.
//
// Supported entry points: GenerateText and StreamText (both single-shot
// and multi-step tool-loop paths). WithStateRef is NOT supported by
// GenerateObject or StreamObject - those functions do not run a
// multi-step loop worth polling, so the AgentState would remain at its
// zero value (StepStarting, 0) indefinitely. Passing WithStateRef to
// either function emits a one-shot stderr warning per process per entry
// point and is otherwise a no-op (FIX 35).
//
// Warning semantics (FIX 48). The stderr warning fires ONCE per process
// per entry point: GenerateObject and StreamObject each own an
// independent atomic.Bool latch. A long-lived process that calls
// GenerateObject with WithStateRef a thousand times sees exactly ONE
// warning for GenerateObject; if that same process later calls
// StreamObject with WithStateRef it sees one additional warning for
// StreamObject. Tests may reset these latches in a hermetic way (see
// object.go FIX 33 caveat) - production code must not.
//
// Observation semantics (FIX 43). StateRef provides an atomic read of the
// (kind, step) pair, but reading StepIdle does NOT establish happens-before
// on other TextStream / TextResult fields (streamErr, ResponseMessages,
// Steps, final Usage, etc.). A poller that observes StepIdle and
// immediately reads stream.Err() or the TextResult return value without
// additional synchronization races against goai's last writes.
//
// Pollers that observe StepIdle MUST still wait on conventional sync
// before reading result data. The defer-chain ordering of the raw
// Stream() channel close relative to the StepIdle publish differs
// between the single-shot and multi-step StreamText paths, so users
// MUST NOT rely on raw channel close as a StepIdle sync point:
//
//   - StreamText, single-shot path (MaxSteps==1 OR no tools bound):
//     the consume goroutine's defer chain closes the raw Stream()
//     channel BEFORE publishing the StepIdle atomic store. A poller
//     that waits only on raw channel close and then reads
//     state.Observe() may still see StepLLMInFlight / StepToolExecuting.
//   - StreamText, multi-step path (MaxSteps>1 with tools): StepIdle is
//     published by the tool-loop goroutine's defer, while the raw
//     output channel is closed by the same goroutine in a separate
//     defer. Callers should not depend on inter-defer ordering here.
//   - StreamText, BOTH paths: call stream.Err() or stream.Result() to
//     reliably observe StepIdle. These block on the internal doneCh
//     which closes AFTER the StepIdle atomic store - so a caller that
//     sees Err()/Result() return is guaranteed to observe StepIdle on
//     a subsequent state.Observe().
//   - GenerateText: use the TextResult returned by GenerateText - by the
//     time GenerateText returns, StepIdle and result fields are
//     synchronized via the function-return happens-before edge.
//
// Use StateRef for "should I wake up / inject work?" polling decisions;
// use conventional sync (channel close, function return) for reading
// result data.
func WithStateRef(ref *AgentState) Option {
	return func(o *options) { o.StateRef = ref }
}

// StopCondition is a composable predicate evaluated AFTER each step of the
// tool loop fully completes - including tool execution for that step --
// and BEFORE deciding whether to issue the next LLM call. Returning true
// causes the loop to exit cleanly using the last completed step's natural
// FinishReason (no synthetic reason is emitted).
//
// The predicate receives the full list of completed steps. Each StepResult
// exposes both ToolCalls (requests the model made this step) and
// ToolResults (outputs produced by executeToolsParallel for those calls);
// ToolResults are populated in element-for-element order with ToolCalls and
// are visible to the predicate because the predicate fires AFTER tool
// execution. Predicates may branch on either - e.g. "stop after the first
// step where any tool result is empty":
//
//	goai.WithStopWhen(func(steps []goai.StepResult) bool {
//	    last := steps[len(steps)-1]
//	    for _, r := range last.ToolResults {
//	        if r.Output == "" && !r.IsError {
//	            return true
//	        }
//	    }
//	    return false
//	})
//
// This mirrors Vercel AI SDK's StopCondition placement
// (packages/ai/src/generate-text/generate-text.ts and stop-condition.ts):
// the predicate gates the NEXT iteration, not the tool exec of the current
// iteration. Consumers can compose multiple predicates with a standard OR:
//
//	goai.WithStopWhen(func(steps []goai.StepResult) bool {
//	    return pred1(steps) || pred2(steps)
//	})
type StopCondition func(steps []StepResult) bool

// WithStopWhen registers a stop predicate. It is evaluated after the
// current step's LLM call AND its tool executions complete (tool-result
// messages already folded into the running message list), and BEFORE the
// next DoGenerate/DoStream is issued. This matches Vercel AI SDK's
// placement (packages/ai/src/generate-text/generate-text.ts).
//
// Consequences of this placement:
//
//   - If the predicate returns true on a step whose LLM response contained
//     tool calls, those tool calls ARE executed before the break. Both
//     the assistant message (with tool_use parts) and the paired
//     tool-result message for the last step are included in
//     ResponseMessages, making the transcript safe to replay against
//     strict providers (Anthropic, OpenAI) that reject dangling tool_use.
//   - The loop's FinishReason is the last completed step's natural reason
//     (typically FinishToolCalls when stopping mid-loop).
//   - StepsExhausted is NOT set by a predicate-driven break even when the
//     break coincides with step == MaxSteps.
//
// If predicate is nil, the option is a no-op. If called multiple times the
// last value wins. Panics in the predicate are recovered and logged; they
// are treated as "do not stop".
//
// Aliasing contract (FIX 30 / FIX 36). The predicate receives a SHALLOW
// clone of goai's internal steps slice: the top-level slice header is
// safe to reslice / append / zero (those mutations stay local). Nested
// slices (StepResult.ToolCalls, StepResult.ToolResults,
// StepResult.Content) are ALIASED into goai's internal state - writing
// to an element of those nested slices WILL corrupt goai's internal
// record and may cause incorrect ResponseMessages or undefined
// downstream behavior. Predicates MUST treat StepResult contents as
// read-only. goai does NOT enforce this via deep-clone (prohibitive
// per-step cost for a feature that is not a supported use case).
// TestStopSafe_AliasingContract_ShallowCloneIsolatesTopLevel locks in
// both halves of the contract.
//
// WithStopWhen is ignored by GenerateObject and StreamObject (those
// functions use their own fixed exit conditions tied to structured output
// parsing). A one-shot stderr warning is emitted the first time WithStopWhen
// is passed to those paths.
func WithStopWhen(predicate StopCondition) Option {
	return func(o *options) { o.StopWhen = predicate }
}
