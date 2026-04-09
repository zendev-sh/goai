---
title: Lifecycle Hooks
description: "Intercept and control GoAI's tool loop with lifecycle hooks. Permission gates, secret scanning, output transformation, and loop control."
---

# Lifecycle Hooks

GoAI provides eight lifecycle hooks that let you observe, intercept, and control the generation process without modifying core SDK code.

## Hook Categories

Hooks fall into two categories:

| Category | Hooks | Callbacks | Purpose |
|----------|-------|-----------|---------|
| **Observability** | `OnRequest`, `OnResponse`, `OnToolCallStart`, `OnToolCall`, `OnStepFinish` | Multiple (append) | Logging, metrics, tracing |
| **Interceptor** | `OnBeforeToolExecute`, `OnAfterToolExecute`, `OnBeforeStep` | Single (replace) | Permission, transformation, control flow |

Observability hooks are fire-and-forget: they receive data but cannot change behavior. Interceptor hooks return values that control execution.

## Execution Order

During a multi-step tool loop, hooks fire in this order:

```
Step 1 --LLM call:
  OnRequest            → before LLM call
  LLM DoGenerate/DoStream
  OnResponse           → after LLM call
  OnStepFinish         → step result ready (includes ToolCalls from LLM)

Step 1 --Tool execution (if LLM requested tools):
  OnToolCallStart      → before each tool (parallel)
  OnBeforeToolExecute  → can skip tool
  tool.Execute()       → actual execution
  OnAfterToolExecute   → can modify output
  OnToolCall           → after each tool (parallel)

Step 2+ --Transition and next LLM call:
  OnBeforeStep         → can inject messages or stop loop
  OnRequest            → before LLM call
  ... (same pattern as Step 1)
```

Note: "step" = one LLM call. Tool execution happens between steps -- results feed into the next step's messages. By default, tools within a step execute in parallel. Use `WithSequentialToolExecution()` to force one-at-a-time execution when tools share non-thread-safe resources or when execution order matters.

## Interceptor Hooks

### OnBeforeToolExecute --Permission Gate

Runs before each tool's `Execute` function. Can skip execution entirely. The `info.Ctx` carries the tool execution context (with tool call ID), useful for spawning child agents or passing tracing context.

```go
result, _ := goai.GenerateText(ctx, model,
    goai.WithPrompt("Delete the temp files and read the config"),
    goai.WithMaxSteps(5),
    goai.WithTools(readFile, deleteFile),
    goai.WithOnBeforeToolExecute(func(info goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
        // Block dangerous tools
        if info.ToolName == "delete_file" {
            return goai.BeforeToolExecuteResult{
                Skip:   true,
                Result: "Permission denied: delete operations are not allowed.",
            }
        }
        return goai.BeforeToolExecuteResult{} // allow
    }),
)
```

**Skip behavior:**
- `Skip: true, Result: "..."` --synthetic result sent to LLM
- `Skip: true, Error: err` --error message sent to LLM (Result is ignored)
- `Skip: true` (no Result or Error) --empty string sent to LLM
- Not called for unknown tools (which fail with `ErrUnknownTool`)

The hook can also override the context and input passed to `Execute` via the result's `Ctx` and `Input` fields (nil = no override). This is useful for injecting tracing context or rewriting tool arguments before execution.

**Observability:** When a tool is skipped, `OnToolCall` still fires with `Skipped: true` so observers can track it. `OnAfterToolExecute` does NOT fire for skipped tools.

### OnAfterToolExecute --Output Transformation

Runs after each tool's `Execute` function, before the result reaches the LLM. Can modify the output. The `info.Ctx` carries the same tool execution context as `OnBeforeToolExecute`.

```go
goai.WithOnAfterToolExecute(func(info goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
    if info.Error != nil {
        return goai.AfterToolExecuteResult{} // don't modify errors
    }
    // Redact secrets from tool output
    redacted := secretScanner.Scan(info.Output)
    if redacted != info.Output {
        return goai.AfterToolExecuteResult{Output: redacted}
    }
    return goai.AfterToolExecuteResult{} // no change
}),
```

**Modification rules:**
- `Output: "..."` --replaces the tool's output
- `Output: ""` --preserves original output (use `" "` to force empty)
- `Error: err` --replaces the tool's error (nil preserves original; cannot clear an error to nil)
- Not called for skipped or unknown tools

### OnBeforeStep --Loop Control

Runs before each LLM call in a multi-step tool loop (step 2+ only, not step 1). Can inject messages or stop the loop. The `info.Ctx` carries the generation context for cancellation checks or external calls.

```go
goai.WithOnBeforeStep(func(info goai.BeforeStepInfo) goai.BeforeStepResult {
    // Inject context from external sources between steps
    if hasNewMessages() {
        return goai.BeforeStepResult{
            ExtraMessages: []provider.Message{
                goai.UserMessage("[System] New context available: " + getContext()),
            },
        }
    }
    // Or stop the loop early
    if shouldStop() {
        return goai.BeforeStepResult{Stop: true}
    }
    return goai.BeforeStepResult{}
}),
```

**Behavior:**
- `Stop: true` --terminates the loop; accumulated result returned as-is
- `ExtraMessages: [...]` --appended to conversation before the next LLM call
- When `Stop` is true, `ExtraMessages` are ignored (Stop takes precedence)
- `Messages` field is a shallow clone --safe to read, do not mutate Content directly

## Observability Hooks

These hooks support multiple callbacks (each `WithOn*` call appends, not replaces).

### OnRequest / OnResponse

```go
goai.WithOnRequest(func(info goai.RequestInfo) {
    log.Printf("LLM call: model=%s messages=%d tools=%d", info.Model, info.MessageCount, info.ToolCount)
}),
goai.WithOnResponse(func(info goai.ResponseInfo) {
    log.Printf("LLM response: latency=%s tokens=%d finish=%s", info.Latency, info.Usage.InputTokens, info.FinishReason)
}),
```

### OnToolCallStart / OnToolCall

```go
goai.WithOnToolCallStart(func(info goai.ToolCallStartInfo) {
    log.Printf("Tool starting: %s (call %s)", info.ToolName, info.ToolCallID)
}),
goai.WithOnToolCall(func(info goai.ToolCallInfo) {
    status := "OK"
    if info.Skipped {
        status = "SKIPPED"
    } else if info.Error != nil {
        status = "ERROR"
    }
    log.Printf("Tool done: %s %s (%s)", info.ToolName, status, info.Duration)
}),
```

### OnStepFinish

```go
goai.WithOnStepFinish(func(step goai.StepResult) {
    log.Printf("Step %d: %s (tools=%d, tokens=%d)",
        step.Number, step.FinishReason, len(step.ToolCalls), step.Usage.InputTokens)
}),
```

## Panic Recovery

Hook panics are recovered in most paths. "Propagates" = crashes the caller. "Recovered" = caught, logged to stderr, execution continues.

| Hook | GenerateText | StreamText (step 1)* | StreamText (step 2+) | GenerateObject |
|------|-------------|---------------------|---------------------|----------------|
| OnRequest | Propagates | Propagates | Recovered | Propagates |
| OnResponse | Recovered | Propagates (error) | Recovered | Propagates |
| OnToolCallStart | Recovered | Recovered | Recovered | Recovered |
| OnToolCall | Recovered | Recovered | Recovered | Recovered |
| OnStepFinish | Recovered | Recovered | Recovered | Recovered |
| OnBeforeToolExecute | Recovered (skips tool) | Recovered (skips tool) | Recovered (skips tool) | Recovered (skips tool) |
| OnAfterToolExecute | Recovered (preserves result) | Recovered (preserves result) | Recovered (preserves result) | Recovered (preserves result) |
| OnBeforeStep | Recovered (proceeds) | N/A | Recovered (proceeds) | Recovered (proceeds) |

*\* StreamText step 1 runs synchronously in the caller's goroutine (before the background goroutine starts). Step 2+ runs in a background goroutine where all panics are recovered. StreamObject OnRequest propagates like GenerateObject. StreamObject OnResponse propagates on the error path but is recovered on the success path (consume goroutine).*

## Complete Example

See [`examples/hooks/main.go`](https://github.com/zendev-sh/goai/blob/main/examples/hooks/main.go) for a runnable example combining permission gates, secret scanning, and loop control.

## Integration with Observability Providers

GoAI's [OpenTelemetry](observability.md#opentelemetry) and [Langfuse](observability.md#langfuse) integrations use the observability hooks internally. The interceptor hooks are not yet integrated into these providers --use them directly in your application code.
