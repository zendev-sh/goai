---
title: Observability
description: "Trace LLM calls, tool executions, and agent runs in GoAI. Plug-in observability providers via lifecycle hooks. Ships with Langfuse support."
---

# Observability

GoAI's observability is built on five lifecycle hooks: `OnRequest`, `OnResponse`, `OnToolCallStart`, `OnToolCall`, and `OnStepFinish`. Any observability provider can plug into these hooks to trace LLM calls, tool executions, and multi-step agent runs.

## How It Works

Observability integrations typically expose helper options. For Langfuse, the primary API is `langfuse.WithTracing(...)`:

```go
result, err := goai.GenerateText(ctx, model,
	langfuse.WithTracing(langfuse.TraceName("my-agent")),
	goai.WithPrompt("Hello"),
)
_ = result
_ = err
```

Each hook fires at a specific point in the request lifecycle:

| Hook              | When                       | Data                                        |
| ----------------- | -------------------------- | ------------------------------------------- |
| `OnRequest`       | Before each LLM call       | Model, full message history, tool count     |
| `OnResponse`      | After each LLM call        | Latency, token usage, finish reason, errors |
| `OnToolCallStart` | Before each tool execution | Tool call ID, tool name, step, input        |
| `OnToolCall`      | After each tool execution  | Tool name, input/output, duration, errors   |
| `OnStepFinish`    | After each step completes  | Step number, finish reason, tool calls      |

This design means observability never touches the core SDK. Providers are optional imports with zero impact on non-instrumented code.

## Available Providers

| Provider              | Package                  | Status  |
| --------------------- | ------------------------ | ------- |
| [Langfuse](#langfuse) | `observability/langfuse` | Shipped |
| OpenTelemetry         | `observability/otel`     | Planned |

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
        goai.WithOnToolCallStart(func(info goai.ToolCallStartInfo) {
            // record tool execution start
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
```

Credentials can be read from env vars or overridden via tracing options.

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

**One-off calls** with `WithTracing()`:

```go
result, err := goai.GenerateText(ctx, model,
    langfuse.WithTracing(
        langfuse.TraceName("my-agent"),
        langfuse.UserID("user-42"),
        langfuse.Tags("prod", "demo"),
    ),
    goai.WithSystem("You are a helpful assistant."),
    goai.WithPrompt("What is Go?"),
)
_ = result
_ = err
```

**Structured output** with tools:

```go
result, err := goai.GenerateObject[WeatherReport](ctx, model,
    langfuse.WithTracing(
        langfuse.TraceName("weather-agent"),
        langfuse.SessionID("session-abc123"),
        langfuse.Version("1.0.0"),
    ),
    goai.WithSystem("You are a weather assistant."),
    goai.WithPrompt("What's the weather in Tokyo?"),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(3),
)
_ = result
_ = err
```

**Deprecated compatibility API** (`langfuse.New(...).Run()/With()`) remains available but is superseded by `WithTracing`.

### Config

`WithTracing` accepts option functions for the same fields previously set via `Config`:

```go
langfuse.WithTracing(
    langfuse.TraceName("my-agent"),
    langfuse.UserID("user-42"),
    langfuse.SessionID("session-abc123"),
    langfuse.Tags("prod", "v1"),
    langfuse.Release("2026.04.03"),
    langfuse.Environment("production"),
    langfuse.OnFlushError(func(err error) { log.Printf("langfuse flush error: %v", err) }),
)
```

### Error Handling

Tracing is best-effort. A Langfuse outage never crashes your app:

- `OnFlushError` receives flush errors if set; otherwise they are silently discarded
- If the LLM call fails, a partial trace is flushed with `level: ERROR`

### Concurrency

Each traced call has isolated state and is safe to run concurrently.

### Example

See the [full runnable example](https://github.com/zendev-sh/goai/tree/main/examples/langfuse) for a multi-step agent with tool calls and structured output.
