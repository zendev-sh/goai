package goai

import (
	"context"
	"encoding/json"
	"time"

	"github.com/zendev-sh/goai/provider"
)

// RequestInfo is passed to the OnRequest hook before a generation call.
type RequestInfo struct {
	// Ctx is the caller's context for this generation call.
	// Observability hooks can use this for span parenting (e.g., creating
	// child spans under the caller's existing trace context).
	Ctx context.Context

	// Model is the model ID.
	Model string

	// MessageCount is the number of messages in the request.
	MessageCount int

	// ToolCount is the number of tools available.
	ToolCount int

	// Timestamp is when the request was initiated.
	Timestamp time.Time

	// Messages is the full conversation history sent to the model for this call.
	// If a system prompt is set, it is prepended as the first message with role "system".
	// For step-1 this is [system, user]. For step-2+ it also includes assistant
	// tool-call messages and tool result messages from prior steps.
	Messages []provider.Message
}

// ResponseInfo is passed to the OnResponse hook after a generation call completes.
type ResponseInfo struct {
	// Latency is the time from request to response.
	Latency time.Duration

	// Usage is the token consumption for this call.
	Usage provider.Usage

	// FinishReason indicates why generation stopped.
	FinishReason provider.FinishReason

	// Error is non-nil if the API call failed (DoGenerate/DoStream returned error)
	// or if a mid-stream ChunkError occurred (in StreamText/StreamObject consume goroutine).
	// For streaming success paths, check stream.Err() for the definitive error status.
	Error error

	// StatusCode is the HTTP status code (0 if not applicable).
	StatusCode int
}

// ToolCallInfo is passed to the OnToolCall hook after a tool executes.
// It contains the full tool Input and Output, which may include sensitive data.
// Consumers that log or export hook data should sanitize accordingly.
type ToolCallInfo struct {
	// ToolCallID is the provider-assigned identifier for this tool call.
	ToolCallID string

	// ToolName is the name of the tool that was called.
	ToolName string

	// Step is the 1-based index of the generation step in which this tool was called.
	Step int

	// Input is the raw JSON arguments passed to the tool.
	// Replaces the former InputSize int field --use len(Input) for byte size.
	Input json.RawMessage

	// Output is the string result returned by the tool.
	// Typically empty when Error is non-nil, but tools may return both output and error
	// (e.g., partial output on timeout). Empty for unknown tools (ErrUnknownTool).
	// Note: when OnBeforeToolExecute skips with both Result and Error set,
	// Output contains the Result string but the LLM receives "error: <message>"
	// (Error takes precedence in the message sent to the model).
	Output string

	// OutputObject is the parsed JSON value of Output when the tool returned valid JSON.
	// Nil if the output is not valid JSON or the tool returned an error.
	// The dynamic type follows json.Unmarshal rules: JSON objects become map[string]any,
	// arrays become []any, numbers become float64, booleans become bool, strings become string.
	OutputObject any

	// StartTime is when the tool execution began.
	// Zero for unknown tools (ErrUnknownTool). For skipped tools (Skipped=true),
	// reflects when the skip decision was made, not execution start.
	StartTime time.Time

	// Duration measures time from before Execute to after OnAfterToolExecute
	// (if registered). Includes both tool execution and after-hook overhead.
	Duration time.Duration

	// Skipped is true when the tool execution was skipped by OnBeforeToolExecute.
	// When Skipped is true, StartTime reflects when the skip decision was made
	// (not when execution started) and Duration is zero.
	Skipped bool

	// Error is non-nil if the tool execution failed.
	Error error

	// Metadata is opaque consumer data set by OnAfterToolExecute.
	// Nil if no OnAfterToolExecute hook is registered or if the hook
	// returned nil Metadata.
	Metadata map[string]any
}

// WithOnStepFinish adds a callback invoked after each generation step completes.
// Multiple callbacks are called in registration order. Each callback is individually
// panic-recovered in all paths (GenerateText, StreamText, GenerateObject, StreamObject).
// A panic in one callback does not prevent subsequent callbacks from firing.
//
// Timing: OnStepFinish fires after the LLM call for the step returns and
// BEFORE any tool executions for that step run. The StepResult passed to
// the hook therefore always carries an EMPTY ToolResults slice - even in
// the sync (GenerateText) path. To observe tool results as they complete,
// use OnToolCall or OnAfterToolExecute; to see the full populated
// StepResult.ToolResults after tools ran, inspect the TextResult.Steps
// slice returned by GenerateText or stream.Result().
func WithOnStepFinish(fn func(StepResult)) Option {
	return func(o *options) { o.OnStepFinish = append(o.OnStepFinish, fn) }
}

// FinishInfo is passed to the OnFinish hook after all generation steps complete.
// It fires exactly once per GenerateText/StreamText/GenerateObject/StreamObject call,
// after the last OnStepFinish. For StreamText/StreamObject, it fires in the
// stream goroutine.
type FinishInfo struct {
	// StepsExhausted is true when MaxSteps was reached while the model still
	// requested tool calls. This is the authoritative signal for "max_steps"
	// termination - it is not available from any per-step hook.
	StepsExhausted bool

	// TotalSteps is the number of generation steps executed.
	TotalSteps int

	// TotalUsage is the aggregated token usage across all steps.
	TotalUsage provider.Usage

	// FinishReason from the last step.
	FinishReason provider.FinishReason

	// StoppedBy classifies how the loop terminated: natural model exit,
	// MaxSteps exhaustion, WithStopWhen predicate, OnBeforeStep.Stop,
	// an error-driven abort, an empty provider stream, or a model
	// requesting tool calls with no executable tools configured.
	// Possible values: "natural", "max-steps", "predicate", "before-step",
	// "abort", "empty" (streaming only), "no-executable-tools" (sync only).
	// Useful when FinishReason alone is ambiguous (e.g. a StopWhen break
	// on a tool-calls step reports FinishToolCalls but
	// StoppedBy=="predicate"). May be empty on pre-loop error paths.
	StoppedBy provider.StopCause
}

// WithOnFinish adds a callback invoked once after all generation steps complete.
// Multiple callbacks are called in registration order. Each callback is individually
// panic-recovered so a panic in one does not prevent subsequent callbacks from firing.
//
// This hook fires in all code paths:
//   - GenerateText: after the tool loop exits (natural, max_steps, or hook_stopped)
//   - StreamText: in the stream goroutine, after the tool loop exits
//   - GenerateObject: after the tool loop exits (including max_steps error path)
//   - StreamObject: in the consume goroutine, after the stream completes
//
// It does NOT fire when DoGenerate/DoStream returns a provider error (API failure,
// context cancelled). In those cases, the OnResponse hook receives the error.
// Note: GenerateObject's max_steps error IS a goai-level error, not a provider error,
// so OnFinish fires before that error is returned.
// For StreamText/StreamObject, check stream.Err() for definitive error status.
func WithOnFinish(fn func(FinishInfo)) Option {
	return func(o *options) { o.OnFinish = append(o.OnFinish, fn) }
}

// WithOnRequest adds a callback invoked before each model call.
// Multiple callbacks are called in registration order.
// In GenerateText, GenerateObject, and StreamText's first step, panics propagate
// to the caller (a panic in one callback prevents subsequent callbacks from firing).
// In StreamText step 2+ (goroutine), each callback is individually panic-recovered.
func WithOnRequest(fn func(RequestInfo)) Option {
	return func(o *options) { o.OnRequest = append(o.OnRequest, fn) }
}

// WithOnResponse adds a callback invoked after each model call completes.
// Multiple callbacks are called in registration order.
// Each callback is individually panic-recovered in GenerateText (all steps),
// all StreamText success paths (single-step consume goroutine and multi-step goroutine),
// GenerateObject (all steps), StreamObject's success path (consume goroutine),
// and StreamObject's error path.
// In StreamText's first-step error path, panics propagate to the caller.
func WithOnResponse(fn func(ResponseInfo)) Option {
	return func(o *options) { o.OnResponse = append(o.OnResponse, fn) }
}

// WithOnToolCall adds a callback invoked after each tool execution.
// Multiple callbacks are called in registration order. Each callback is individually
// panic-recovered so a panic in one does not prevent others from firing.
func WithOnToolCall(fn func(ToolCallInfo)) Option {
	return func(o *options) { o.OnToolCall = append(o.OnToolCall, fn) }
}

// ToolCallStartInfo is passed to the OnToolCallStart hook before a tool executes.
type ToolCallStartInfo struct {
	// ToolCallID is the provider-assigned identifier for this tool call.
	ToolCallID string

	// ToolName is the name of the tool about to execute.
	ToolName string

	// Step is the 1-based index of the generation step in which this tool was called.
	Step int

	// Input is the raw JSON arguments that will be passed to the tool.
	Input json.RawMessage
}

// WithOnToolCallStart adds a callback invoked before each tool execution.
// Multiple callbacks are called in registration order. Each callback is individually
// panic-recovered so a panic in one does not prevent others from firing. If any callback
// panics, the tool does not execute.
func WithOnToolCallStart(fn func(ToolCallStartInfo)) Option {
	return func(o *options) { o.OnToolCallStart = append(o.OnToolCallStart, fn) }
}

// BeforeToolExecuteInfo is passed to the OnBeforeToolExecute hook before a tool's
// Execute function runs. The hook can inspect the call and optionally skip execution.
type BeforeToolExecuteInfo struct {
	// Ctx is the tool execution context (with tool call ID injected).
	// Use this for passing cancellation, timeouts, or tracing context
	// to any work done inside the hook (e.g., spawning child agents).
	Ctx context.Context

	// ToolCallID is the provider-assigned identifier for this tool call.
	ToolCallID string

	// ToolName is the name of the tool about to execute.
	ToolName string

	// Step is the 1-based index of the generation step.
	Step int

	// Input is the raw JSON arguments that will be passed to the tool.
	Input json.RawMessage
}

// BeforeToolExecuteResult controls what happens after OnBeforeToolExecute runs.
type BeforeToolExecuteResult struct {
	// Skip, when true, prevents the tool's Execute function from running.
	// The Result string is used as the tool output sent back to the LLM.
	// If Error is also set, it is reported as a tool error instead.
	Skip bool

	// Result is the synthetic tool output when Skip is true.
	// Ignored when Skip is false.
	// When Error is also non-nil, Result is ignored for the LLM message --only the
	// error is sent (as "error: <message>"). However, OnToolCall still receives Result
	// in its Output field for observability.
	// When Skip is true and both Result and Error are empty/nil, an empty string
	// is sent as the tool output to the LLM. Set Result to a meaningful string
	// to signal the skip reason to the model.
	Result string

	// Error, when non-nil and Skip is true, is reported as the tool error.
	// Result is ignored when Error is set --the LLM receives "error: <message>".
	// Ignored when Skip is false.
	Error error

	// Ctx, when non-nil, replaces the context passed to tool.Execute.
	// Use this to inject per-tool values (auth tokens, session IDs, timeouts)
	// or to wrap the context with context.WithTimeout for per-tool deadlines.
	// Nil means use the default toolCtx (parent ctx + tool call ID).
	// Ignored when Skip is true.
	Ctx context.Context

	// Input, when non-nil, replaces the tool arguments passed to Execute.
	// Use for input normalization, PII redaction, or secret injection.
	// Nil means use the original input from the LLM.
	// Ignored when Skip is true.
	Input json.RawMessage
}

// WithOnBeforeToolExecute adds a callback invoked before each known tool's Execute function.
// Not called for unknown tools (which always fail with ErrUnknownTool).
// The callback can inspect the tool call and return a BeforeToolExecuteResult to skip
// execution (e.g., for permission checks, rate limiting, or doom-loop detection).
// Only one callback is supported; setting a second replaces the first.
// The callback is panic-recovered; a panic skips the tool with an error result.
func WithOnBeforeToolExecute(fn func(BeforeToolExecuteInfo) BeforeToolExecuteResult) Option {
	return func(o *options) { o.OnBeforeToolExecute = fn }
}

// AfterToolExecuteInfo is passed to the OnAfterToolExecute hook after a tool's
// Execute function completes, before the result is sent to the LLM.
type AfterToolExecuteInfo struct {
	// Ctx is the tool execution context (with tool call ID injected).
	Ctx context.Context

	// ToolCallID is the provider-assigned identifier for this tool call.
	ToolCallID string

	// ToolName is the name of the tool that executed.
	ToolName string

	// Step is the 1-based index of the generation step.
	Step int

	// Input is the raw JSON arguments passed to the tool.
	Input json.RawMessage

	// Output is the string result returned by the tool's Execute function.
	Output string

	// Error is non-nil if the tool's Execute function returned an error.
	Error error
}

// AfterToolExecuteResult allows the hook to modify the tool output before it
// is sent to the LLM.
type AfterToolExecuteResult struct {
	// Output replaces the tool's original output. If empty and the original
	// output was non-empty, the original is preserved (use a space to force empty).
	Output string

	// Error, when non-nil, replaces the tool's original error.
	// A nil value preserves the original error (cannot clear an error to nil).
	Error error

	// Metadata is opaque consumer data attached to the tool call result.
	// Passed through to ToolCallInfo.Metadata in the OnToolCall hook.
	// Use for tool titles, display tags, scanner flags, or any consumer-specific
	// data that should travel alongside the tool result.
	// Nil means no metadata.
	Metadata map[string]any
}

// WithOnAfterToolExecute adds a callback invoked after each tool's Execute function,
// before the result is sent to the LLM. The callback can modify the output
// (e.g., for secret scanning, truncation, or output transformation).
// Not called when OnBeforeToolExecute skips execution or for unknown tools.
// Use OnToolCall with Skipped=true to observe skipped tool results.
// Only one callback is supported; setting a second replaces the first.
// The callback is panic-recovered; a panic preserves the original tool result.
func WithOnAfterToolExecute(fn func(AfterToolExecuteInfo) AfterToolExecuteResult) Option {
	return func(o *options) { o.OnAfterToolExecute = fn }
}

// BeforeStepInfo is passed to the OnBeforeStep hook before each LLM call in
// a multi-step tool loop (step 2+). It is NOT called before step 1.
type BeforeStepInfo struct {
	// Ctx is the generation context. Use for cancellation checks or
	// passing context to external calls (e.g., inbox drain, context fetch).
	Ctx context.Context

	// Step is the 1-based step number about to execute.
	Step int

	// Messages is the current conversation history that will be sent to the LLM.
	// This is a shallow clone of the internal messages slice --the slice header is
	// independent but Message fields containing reference types (e.g., Content []Part)
	// share the underlying arrays with the live conversation. Do not mutate message
	// Content directly; use ExtraMessages to inject new messages.
	Messages []provider.Message
}

// BeforeStepResult controls what happens before the next LLM call.
type BeforeStepResult struct {
	// ExtraMessages are appended to the conversation before the LLM call.
	// Use this to inject inter-agent messages, context updates, etc.
	// Ignored when Stop is true.
	ExtraMessages []provider.Message

	// Stop, when true, terminates the tool loop before the next LLM call.
	// The current accumulated result is returned as-is.
	// When Stop is true, ExtraMessages are ignored (Stop takes precedence).
	Stop bool
}

// WithOnBeforeStep adds a callback invoked before each LLM call in a multi-step
// tool loop (step 2+, after tool execution). The callback can inject additional
// messages or stop the loop early.
// Only one callback is supported; setting a second replaces the first.
// The callback is panic-recovered; a panic is logged and the step proceeds normally.
func WithOnBeforeStep(fn func(BeforeStepInfo) BeforeStepResult) Option {
	return func(o *options) { o.OnBeforeStep = fn }
}

// WrapOnBeforeToolExecute returns an Option that wraps the existing OnBeforeToolExecute hook.
// The wrapper function receives the current hook (may be nil if none is registered) and
// returns a replacement. This enables observability packages to add side-effects without
// replacing the user's hook.
//
// The wrapper MUST return the user hook's result verbatim when delegating - accidentally
// zeroing the struct suppresses tool executions or input overrides. When the existing
// hook is nil, return a zero BeforeToolExecuteResult (no skip, no override).
//
// Ordering hazard: the wrapper captures the existing hook at option-apply time.
// If a user calls WithOnBeforeToolExecute AFTER this wrapper is applied, the user's
// hook overwrites the wrapper entirely. Callers using this for observability should
// apply it after user hooks (e.g., WithTracing should be the last option).
func WrapOnBeforeToolExecute(wrapper func(existing func(BeforeToolExecuteInfo) BeforeToolExecuteResult) func(BeforeToolExecuteInfo) BeforeToolExecuteResult) Option {
	return func(o *options) {
		o.OnBeforeToolExecute = wrapper(o.OnBeforeToolExecute)
	}
}

// WrapOnAfterToolExecute returns an Option that wraps the existing OnAfterToolExecute hook.
// The wrapper function receives the current hook (may be nil) and returns a replacement.
// The wrapper MUST return the user hook's result verbatim - an empty AfterToolExecuteResult
// has special semantics (empty Output preserves original, nil Error preserves original).
// When the existing hook is nil, return a zero AfterToolExecuteResult.
//
// Ordering hazard: same as WrapOnBeforeToolExecute - apply after user hooks.
func WrapOnAfterToolExecute(wrapper func(existing func(AfterToolExecuteInfo) AfterToolExecuteResult) func(AfterToolExecuteInfo) AfterToolExecuteResult) Option {
	return func(o *options) {
		o.OnAfterToolExecute = wrapper(o.OnAfterToolExecute)
	}
}

// WrapOnBeforeStep returns an Option that wraps the existing OnBeforeStep hook.
// The wrapper function receives the current hook (may be nil) and returns a replacement.
// The wrapper MUST forward the user hook's Stop and ExtraMessages fields verbatim.
// When the existing hook is nil, return a zero BeforeStepResult (no stop, no messages).
//
// Ordering hazard: same as WrapOnBeforeToolExecute - apply after user hooks.
func WrapOnBeforeStep(wrapper func(existing func(BeforeStepInfo) BeforeStepResult) func(BeforeStepInfo) BeforeStepResult) Option {
	return func(o *options) {
		o.OnBeforeStep = wrapper(o.OnBeforeStep)
	}
}
