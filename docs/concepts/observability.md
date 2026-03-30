---
title: Observability
description: "Trace LLM calls with Langfuse in GoAI. Zero-dependency tracing with automatic trace hierarchy, tool spans, and token usage tracking."
---

# Observability

GoAI provides built-in observability via the `observability/langfuse` package. Every LLM call, tool execution, and agent run is traced automatically.

```go
import "github.com/zendev-sh/goai/observability/langfuse"

lf := langfuse.New(langfuse.Config{TraceName: "my-agent"})

result, err := goai.GenerateText(ctx, model,
    append(lf.Run(),
        goai.WithPrompt("Hello"),
    )...,
)
```

## Setup

Set environment variables for Langfuse credentials:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
```

Or pass them directly:

```go
lf := langfuse.New(langfuse.Config{
    PublicKey: "pk-lf-...",
    SecretKey: "sk-lf-...",
    Host:      "https://cloud.langfuse.com",
    TraceName: "my-agent",
})
```

## Trace Hierarchy

Each run creates a structured trace:

```
Trace
└── Span("agent")            - wraps the entire run
    ├── Generation("step-1") - LLM call with input/output, model, usage
    ├── Span("tool-name")    - tool execution with input/output, duration
    └── Generation("step-2") - final LLM call
```

Generations include model name, token usage (input/output/reasoning/cache), and finish reason. Tool spans include the tool name, input arguments, output, and duration.

## Usage Patterns

### One-off calls with `Run()`

`Run()` returns a fresh set of options for a single call:

```go
lf := langfuse.New(langfuse.Config{TraceName: "assistant"})

result, err := goai.GenerateText(ctx, model,
    append(lf.Run(),
        goai.WithSystem("You are a helpful assistant."),
        goai.WithPrompt("What is Go?"),
    )...,
)
```

### Reusable factory with `With()`

`With()` bakes in base options and returns a factory. Call it per run:

```go
runAgent := lf.With(
    goai.WithSystem("You are a weather assistant."),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
)

// Each call gets a separate trace
for _, city := range cities {
    result, err := goai.GenerateText(ctx, model,
        append(runAgent(), goai.WithPrompt("Weather in "+city))...,
    )
}
```

### Structured output with tools

Works with `GenerateObject` and multi-step tool loops:

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

## Config Options

```go
langfuse.Config{
    // Credentials (override env vars)
    PublicKey string   // LANGFUSE_PUBLIC_KEY
    SecretKey string   // LANGFUSE_SECRET_KEY
    Host      string   // LANGFUSE_HOST or LANGFUSE_BASE_URL

    // Trace metadata
    TraceName   string   // name shown in Langfuse UI (default: "agent")
    UserID      string   // associate traces with a user
    SessionID   string   // group traces into a session
    Tags        []string // searchable tags
    Metadata    any      // arbitrary metadata attached to the trace
    Release     string   // app release version
    Version     string   // trace schema version
    Environment string   // "production", "staging", etc. (falls back to LANGFUSE_ENV)

    // Prompt management
    PromptName    string // link to a Langfuse prompt
    PromptVersion int    // prompt version number

    // Error handling
    OnFlushError func(error) // called on HTTP flush failure (nil = silent)
}
```

## Error Handling

Tracing is best-effort. A Langfuse outage or flush failure never crashes your app:

- If `OnFlushError` is set, it receives the error
- If `OnFlushError` is nil, flush errors are silently discarded
- If the LLM call itself fails, a partial trace is still flushed with `level: ERROR`

```go
lf := langfuse.New(langfuse.Config{
    TraceName: "my-agent",
    OnFlushError: func(err error) {
        log.Printf("langfuse flush error: %v", err)
    },
})
```

## Concurrency

The HTTP client is shared across all runs. Each `Run()` or `With()()` call creates isolated per-run state with no locking needed within a run. Concurrent runs are safe:

```go
var wg sync.WaitGroup
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        goai.GenerateText(ctx, model, append(lf.Run(), goai.WithPrompt("hi"))...)
    }()
}
wg.Wait()
```

## What Gets Traced

| Hook | Data captured |
|------|--------------|
| `OnRequest` | Model name, full message history (system + user + tool results) |
| `OnResponse` | Latency, token usage (input/output/reasoning/cache), finish reason |
| `OnToolCall` | Tool name, input args, output, duration, errors |
| `OnStepFinish` | Step number, finish reason, triggers trace flush on final step |

## Example

See the [full runnable example](https://github.com/zendev-sh/goai/tree/main/examples/langfuse) for a complete multi-step agent with tool calls and structured output.
