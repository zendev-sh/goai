---
title: Agent State & Stop Conditions
description: "Observe and control GoAI's tool loop lifecycle. AgentState, StepKind, WithStopWhen, and terminal states for long-running agents."
---

# Agent State & Stop Conditions

GoAI exposes the tool loop's lifecycle through two composable primitives:

- `AgentState` + `WithStateRef` - race-free observable handle into the current loop phase. Pollers in another goroutine can decide when to inject work or signal a wake.
- `WithStopWhen` - stop predicate evaluated after each step's LLM call and tool execution. Mirrors Vercel AI SDK's `StopCondition` placement.

Both apply to `GenerateText` and `StreamText`. `GenerateObject` / `StreamObject` ignore them.

## StepKind Lifecycle

A running tool loop advances through these kinds:

| Kind                | Meaning                                                              |
| ------------------- | -------------------------------------------------------------------- |
| `StepStarting`      | Initial state before the first LLM call.                             |
| `StepLLMInFlight`   | LLM request is in flight (sync or stream).                           |
| `StepStepFinished`  | LLM call returned, step result recorded, tool execution not started. |
| `StepToolExecuting` | One or more tool `Execute` functions are running.                    |
| `StepIdle`          | Tool loop terminated internally. Parking state, NOT terminal.        |
| `StepDone`          | Terminal. Runner completed naturally.                                |
| `StepCancelled`     | Terminal. Runner exited due to context cancellation.                 |
| `StepError`         | Terminal. Runner returned an error or recovered a panic.             |

`StepIdle` is reached when the loop finishes (naturally, by `MaxSteps`, by `OnBeforeStep.Stop`, or by `WithStopWhen`). The three terminal kinds are set by the consumer that owns the `AgentState` lifetime - goai itself never writes them.

`StepKind.IsTerminal()` reports whether a kind is one of `StepDone` / `StepCancelled` / `StepError`.

## Observing State

Allocate an `AgentState` and pass a pointer via `WithStateRef`:

```go
var state goai.AgentState

go func() {
    ticker := time.NewTicker(50 * time.Millisecond)
    defer ticker.Stop()
    for range ticker.C {
        kind, step := state.Observe()
        log.Printf("agent phase=%s step=%d", kind, step)
        if kind.IsTerminal() {
            return
        }
    }
}()

result, err := goai.GenerateText(ctx, model,
    goai.WithStateRef(&state),
    goai.WithMaxSteps(5),
    goai.WithTools(tools...),
    goai.WithPrompt("..."),
)
```

`AgentState.Observe()` returns `(kind, step)` atomically. Zero value is usable: a fresh `AgentState` reports `(StepStarting, 0)` until goai begins mutating it.

**Step semantics:**

- Before first LLM call: `step == 0`.
- During `StepLLMInFlight` / `StepStepFinished` / `StepToolExecuting`: `step` is the 1-indexed current step number.
- At `StepIdle`: `step` is the highest announced step. It may exceed `len(result.Steps)` if an in-flight step errored before being appended. Use `len(result.Steps)` when you need completed-step count.

## Observation Hazards

`StateRef` gives you an atomic read of `(kind, step)`, but observing `StepIdle` does NOT establish happens-before on other result fields (`streamErr`, `ResponseMessages`, `Steps`, final `Usage`). Treat it as a "wake eligible" signal only.

Safe sync points:

- **GenerateText**: the returned `TextResult` is synchronized with the function return.
- **StreamText**: call `stream.Err()` or `stream.Result()`. Both block on the internal doneCh which closes AFTER the `StepIdle` store, so once they return you will observe `StepIdle` on a subsequent `Observe()`. Do NOT rely on the raw `Stream()` channel close alone - defer ordering differs between the single-shot and multi-step paths.

## Terminal States

Terminal kinds are sticky: the first `SetTerminal` wins, subsequent calls are no-ops returning `false`. The step counter is preserved across the transition.

```go
func (s *AgentState) SetTerminal(kind StepKind) bool
```

Typical usage in a consumer that owns the runner:

```go
func (r *Runner) Run(ctx context.Context) (err error) {
    defer func() {
        switch {
        case recover() != nil:
            r.state.SetTerminal(goai.StepError)
        case ctx.Err() != nil:
            r.state.SetTerminal(goai.StepCancelled)
        case err != nil:
            r.state.SetTerminal(goai.StepError)
        default:
            r.state.SetTerminal(goai.StepDone)
        }
    }()
    _, err = goai.GenerateText(ctx, r.model, goai.WithStateRef(&r.state), ...)
    return err
}
```

Constraints:

- Passing a non-terminal kind (e.g. `StepIdle`) panics.
- `nil` receiver is a no-op and returns `false`.
- Only one writer should call `SetTerminal` per `AgentState` (the owning consumer). goai's own hooks never call it.

## WithStopWhen

A stop predicate evaluated AFTER the current step's LLM call AND its tool executions complete, and BEFORE the next LLM call. Returning `true` exits the loop cleanly using the last completed step's natural `FinishReason` (no synthetic reason is emitted).

```go
goai.WithStopWhen(func(steps []goai.StepResult) bool {
    last := steps[len(steps)-1]
    for _, r := range last.ToolResults {
        if r.Output == "" && !r.IsError {
            return true
        }
    }
    return false
})
```

Placement consequences (matches Vercel AI SDK):

- If the predicate breaks on a step whose response contained tool calls, **those tool calls ARE executed before the break**. Both the assistant message and the paired tool-result message are included in `ResponseMessages`, keeping the transcript replay-safe against strict providers (Anthropic, OpenAI).
- The loop's `FinishReason` is the last completed step's natural reason (typically `FinishToolCalls` when stopping mid-loop).
- `StepsExhausted` is NOT set by a predicate break, even when it coincides with `step == MaxSteps`.

### Stream finish reason

For `StreamText`, the final `ChunkFinish.StoppedBy` carries `StopCausePredicate` when a predicate-driven break fires. Use this to distinguish predicate breaks from natural finishes in streaming consumers.

### Composition

Multiple predicates compose with a standard OR:

```go
goai.WithStopWhen(func(steps []goai.StepResult) bool {
    return stopOnEmptyToolResult(steps) || stopAfterSpecificTool(steps)
})
```

### Aliasing contract

The predicate receives a SHALLOW clone of goai's internal steps slice:

- **Top-level slice**: safe to reslice / append / zero. Mutations stay local.
- **Nested slices** (`StepResult.ToolCalls`, `StepResult.ToolResults`, `StepResult.Content`): aliased into goai's internal state. Writing to an element WILL corrupt the internal record and may produce incorrect `ResponseMessages`.

Treat `StepResult` contents as read-only. goai does not enforce this via deep-clone (prohibitive per-step cost).

### Panics

Panics inside the predicate are recovered and logged. They are treated as "do not stop" so the loop continues.

## Scope

| Function         | `WithStateRef` | `WithStopWhen` |
| ---------------- | -------------- | -------------- |
| `GenerateText`   | supported      | supported      |
| `StreamText`     | supported      | supported      |
| `GenerateObject` | warns, no-op   | warns, no-op   |
| `StreamObject`   | warns, no-op   | warns, no-op   |

`GenerateObject` and `StreamObject` run with fixed exit conditions tied to structured output parsing. Passing either option emits a one-shot stderr warning per process per entry point and is otherwise ignored.
