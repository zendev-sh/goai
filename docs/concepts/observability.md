---
title: Observability
description: "Trace LLM calls, tool executions, and agent runs in GoAI. Plug-in observability providers via lifecycle hooks. Ships with Langfuse support."
---

# Observability

GoAI's observability is built on five lifecycle hooks: `OnRequest`, `OnResponse`, `OnToolCallStart`, `OnToolCall`, and `OnStepFinish`. Any observability provider can plug into these hooks to trace LLM calls, tool executions, and multi-step agent runs.

## How It Works

Observability providers return `[]goai.Option` containing hook closures. Append them to your call:

```go
result, err := goai.GenerateText(ctx, model,
    append(provider.Run(),
        goai.WithPrompt("Hello"),
    )...,
)
```

Each hook fires at a specific point in the request lifecycle:

| Hook | When | Data |
|------|------|------|
| `OnRequest` | Before each LLM call | Model, full message history, tool count |
| `OnResponse` | After each LLM call | Latency, token usage, finish reason, errors |
| `OnToolCallStart` | Before each tool execution | Tool call ID, tool name, step, input |
| `OnToolCall` | After each tool execution | Tool name, input/output, duration, errors |
| `OnStepFinish` | After each step completes | Step number, finish reason, tool calls |

This design means observability never touches the core SDK. Providers are optional imports with zero impact on non-instrumented code.

## Available Providers

| Provider | Package | Status |
|----------|---------|--------|
| [Langfuse](#langfuse) | `observability/langfuse` | Shipped |
| OpenTelemetry | `observability/otel` | Planned |

### Custom Provider

Any observability backend can be integrated by implementing the hooks pattern:

```go
func MyTracer() []goai.Option {
    return []goai.Option{
        goai.WithOnRequest(func(info goai.RequestInfo) {
            // start span
        }),
        goai.WithOnResponse(func(info goai.ResponseInfo) {
            // record latency, usage, errors
        }),
        goai.WithOnToolCall(func(info goai.ToolCallInfo) {
            // record tool execution
        }),
        goai.WithOnStepFinish(func(step goai.StepResult) {
            // close span, flush
        }),
    }
}
```

---

## Langfuse

[Langfuse](https://langfuse.com) is an open-source LLM observability platform. The `observability/langfuse` package provides zero-dependency tracing with automatic trace hierarchy, tool spans, and token usage tracking.

### Setup

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
```

```go
import "github.com/zendev-sh/goai/observability/langfuse"

lf := langfuse.New(langfuse.Config{TraceName: "my-agent"})
```

Credentials can also be passed directly via `Config.PublicKey`, `Config.SecretKey`, `Config.Host`.

### Trace Hierarchy

Each run creates a structured trace in Langfuse:

```
Trace
└── Span("agent")            - wraps the entire run
    ├── Generation("step-1") - LLM call with input/output, model, usage
    ├── Span("tool-name")    - tool execution with input/output, duration
    └── Generation("step-2") - final LLM call
```

Generations include model name, token usage (input/output/reasoning/cache), and finish reason. Tool spans include the tool name, input arguments, output, and duration.

### Usage Patterns

**One-off calls** with `Run()`:

```go
result, err := goai.GenerateText(ctx, model,
    append(lf.Run(),
        goai.WithSystem("You are a helpful assistant."),
        goai.WithPrompt("What is Go?"),
    )...,
)
```

**Reusable factory** with `With()` for running the same agent multiple times:

```go
runAgent := lf.With(
    goai.WithSystem("You are a weather assistant."),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
)

for _, city := range cities {
    result, err := goai.GenerateText(ctx, model,
        append(runAgent(), goai.WithPrompt("Weather in "+city))...,
    )
}
```

**Structured output** with tools:

```go
result, err := goai.GenerateObject[WeatherReport](ctx, model,
    append(lf.Run(),
        goai.WithSystem("You are a weather assistant."),
        goai.WithPrompt("What's the weather in Tokyo?"),
        goai.WithTools(weatherTool),
        goai.WithMaxSteps(3),
    )...,
)
```

### Config

```go
langfuse.Config{
    // Credentials (override env vars)
    PublicKey string   // LANGFUSE_PUBLIC_KEY
    SecretKey string   // LANGFUSE_SECRET_KEY
    Host      string   // LANGFUSE_HOST or LANGFUSE_BASE_URL

    // Trace metadata
    TraceName   string   // name in Langfuse UI (default: "agent")
    UserID      string   // associate traces with a user
    SessionID   string   // group traces into a session
    Tags        []string // searchable tags
    Metadata    any      // arbitrary metadata on the trace
    Release     string   // app release version
    Version     string   // trace schema version
    Environment string   // falls back to LANGFUSE_ENV

    // Prompt management
    PromptName    string // link to a Langfuse prompt
    PromptVersion int

    // Error handling
    OnFlushError func(error) // called on HTTP flush failure (nil = silent)
}
```

### Error Handling

Tracing is best-effort. A Langfuse outage never crashes your app:

- `OnFlushError` receives flush errors if set; otherwise they are silently discarded
- If the LLM call fails, a partial trace is flushed with `level: ERROR`

### Concurrency

The HTTP client is shared across all runs. Each `Run()` or `With()()` call creates isolated per-run state. Concurrent runs are safe with no additional synchronization needed.

### Example

See the [full runnable example](https://github.com/zendev-sh/goai/tree/main/examples/langfuse) for a multi-step agent with tool calls and structured output.
