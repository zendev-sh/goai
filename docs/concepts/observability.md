---
title: Observability
description: "Trace LLM calls, tool executions, and agent runs in GoAI. Plug-in observability providers via lifecycle hooks. Ships with Langfuse and OpenTelemetry support."
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
| [OpenTelemetry](#opentelemetry) | `observability/otel`     | Shipped |

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

---

## OpenTelemetry

The `observability/otel` package provides OpenTelemetry tracing and metrics for GoAI calls. It emits spans following [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) and works with any OTel-compatible backend (Jaeger, Datadog, Honeycomb, Grafana Tempo, etc.).

### Setup

```bash
go get github.com/zendev-sh/goai/observability/otel
# Plus your preferred exporter, e.g.:
go get go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp
go get go.opentelemetry.io/otel/sdk/trace
```

```go
import goaiotel "github.com/zendev-sh/goai/observability/otel"
```

By default, `WithTracing()` uses the global `TracerProvider` and `MeterProvider` registered via `otel.SetTracerProvider` / `otel.SetMeterProvider`. Override with explicit options if needed.

**Production setup** with OTLP exporter:

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

exporter, err := otlptracehttp.New(ctx,
    otlptracehttp.WithEndpoint("tempo.example.com:4318"),
    otlptracehttp.WithHeaders(map[string]string{
        "Authorization": "Bearer <API_KEY>",
    }),
)
tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter))
defer tp.Shutdown(ctx)
otel.SetTracerProvider(tp)
```

Or use standard OTel environment variables (the SDK reads them automatically):

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://tempo.example.com:4318
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <API_KEY>"
```

### Span Hierarchy

Each run creates a structured trace:

```
chat (root)
├── chat {model}      — LLM API call (step 1) with model, usage, finish reason
├── execute_tool {tool} — tool execution with duration
└── chat {model}      — LLM API call (step 2)
```

### Usage Patterns

**Basic** -- uses the global TracerProvider:

```go
result, err := goai.GenerateText(ctx, model,
    goaiotel.WithTracing(),
    goai.WithPrompt("What is Go?"),
)
_ = result
_ = err
```

**With custom providers and span name**:

```go
result, err := goai.GenerateText(ctx, model,
    goaiotel.WithTracing(
        goaiotel.WithTracerProvider(tp),
        goaiotel.WithMeterProvider(mp),
        goaiotel.WithSpanName("my-agent"),
    ),
    goai.WithPrompt("What is Go?"),
)
_ = result
_ = err
```

**With custom attributes and message recording**:

```go
result, err := goai.GenerateObject[WeatherReport](ctx, model,
    goaiotel.WithTracing(
        goaiotel.WithSpanName("weather-agent"),
        goaiotel.WithAttributes(
            attribute.String("user.id", "user-42"),
            attribute.String("session.id", "session-abc123"),
        ),
        goaiotel.RecordInputMessages(true),
        goaiotel.RecordOutputMessages(true),
    ),
    goai.WithSystem("You are a weather assistant."),
    goai.WithPrompt("What's the weather in Tokyo?"),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(3),
)
_ = result
_ = err
```

### Config

| Option | Description |
| --- | --- |
| `WithTracerProvider(tp)` | Use a specific `trace.TracerProvider` instead of the global one |
| `WithMeterProvider(mp)` | Use a specific `metric.MeterProvider` instead of the global one |
| `WithSpanName(name)` | Set the root span name (default: `"chat"`) |
| `WithAttributes(attrs...)` | Attach custom `attribute.KeyValue` pairs to the root span |
| `RecordInputMessages(bool)` | Record full input messages as span events (default: `false`) |
| `RecordOutputMessages(bool)` | Record full output messages as span events (default: `false`) |

### Semantic Conventions

Spans are annotated with `gen_ai.*` attributes following the OpenTelemetry GenAI semantic conventions:

| Attribute | Description |
| --- | --- |
| `gen_ai.system` | Always `"goai"` |
| `gen_ai.operation.name` | Operation name (e.g. `"chat"`) |
| `gen_ai.request.model` | Model ID |
| `gen_ai.response.finish_reasons` | Finish reason(s) |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.usage.total_tokens` | Total token count (only if > 0) |
| `goai.usage.reasoning_tokens` | Reasoning/thinking token count (only if > 0) |
| `gen_ai.usage.cache_read.input_tokens` | Prompt cache read tokens (only if > 0) |
| `gen_ai.usage.cache_creation.input_tokens` | Prompt cache write tokens (only if > 0) |
| `goai.step` | 1-based step index (on LLM call and tool spans) |
| `gen_ai.tool.name` | Tool name (on tool spans) |
| `gen_ai.tool.call.id` | Tool call ID (on tool spans) |
| `http.response.status_code` | HTTP status code (only if > 0) |

### Metrics

When a `MeterProvider` is configured, the following metrics are recorded:

| Metric | Type | Description |
| --- | --- | --- |
| `gen_ai.client.token.usage` | Int64 Histogram | Token counts per type (`gen_ai.token.type` = `"input"` / `"output"`) |
| `gen_ai.client.operation.duration` | Float64 Histogram | LLM call duration in seconds |
| `goai.tool.duration` | Float64 Histogram | Tool execution duration in seconds (includes `gen_ai.tool.name` attribute) |

### Error Handling

Tracing is best-effort. An OTel collector outage never crashes your app:

- Failed spans are recorded with `codes.Error` status and the error message
- If the LLM call fails, the span is ended with error status and the error is recorded

### Concurrency

Each traced call has isolated state and is safe to run concurrently. The underlying `TracerProvider` and `MeterProvider` handle their own thread safety.

### Example

See the [full runnable example](https://github.com/zendev-sh/goai/tree/main/examples/otel) for a multi-step agent with tool calls and structured output.
