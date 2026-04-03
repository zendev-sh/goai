---
title: Streaming
description: "Stream LLM responses in real time with GoAI's TextStream API. Choose from text-only, full chunk, or blocking consumption modes."
---

# Streaming

`StreamText` returns a `*TextStream` that provides three ways to consume the response. Choose one based on what level of detail you need.

```go
stream, err := goai.StreamText(ctx, model,
    goai.WithPrompt("Write a haiku about Go."),
)
if err != nil {
    log.Fatal(err)
}
```

## Consumption Modes

### TextStream() - text deltas only

Returns a `<-chan string` that emits text content as it arrives. This is the simplest mode for displaying text to a user.

```go
for chunk := range stream.TextStream() {
    fmt.Print(chunk)
}
fmt.Println()
```

### Stream() - full stream chunks

Returns a `<-chan provider.StreamChunk` with all chunk types: text, reasoning, tool calls, finish events, usage data.

```go
for chunk := range stream.Stream() {
    switch chunk.Type {
    case provider.ChunkText:
        fmt.Print(chunk.Text)
    case provider.ChunkReasoning:
        fmt.Print(chunk.Text) // Extended thinking
    case provider.ChunkToolCall:
        fmt.Printf("Tool call: %s(%s)\n", chunk.ToolName, chunk.ToolInput)
    case provider.ChunkFinish:
        fmt.Printf("Done. Tokens: %d\n", chunk.Usage.TotalTokens)
    }
}
```

### Chunk Types

| Type                       | Fields Used                                     | Description                          |
| -------------------------- | ----------------------------------------------- | ------------------------------------ |
| `ChunkText`                | `Text`, `Metadata`                              | Generated text content               |
| `ChunkReasoning`           | `Text`, `Metadata`                              | Extended thinking / chain-of-thought |
| `ChunkToolCall`            | `ToolCallID`, `ToolName`, `ToolInput`           | Complete tool call                   |
| `ChunkToolCallStreamStart` | `ToolCallID`, `ToolName`                        | Start of a streaming tool call       |
| `ChunkToolCallDelta`       | `ToolInput`                                     | Incremental tool call input          |
| `ChunkToolResult`          | `Text`                                          | Tool execution result                |
| `ChunkStepFinish`          | `FinishReason`, `Usage`, `Response`, `Metadata` | End of a tool-loop step              |
| `ChunkFinish`              | `FinishReason`, `Usage`, `Response`, `Metadata` | End of generation                    |
| `ChunkError`               | `Error`                                         | Stream error                         |

`Metadata` carries provider-specific data: `ChunkFinish` may include `"providerMetadata"` (OpenAI Chat Completions via openaicompat) or flat keys like `"iterations"` (Anthropic) and `"cacheWriteInputTokens"` (Bedrock). These are propagated into `TextResult.ProviderMetadata` and `Response.ProviderMetadata` respectively. Note: the OpenAI Responses API streaming path does not populate `ProviderMetadata` on the finish chunk — logprobs are only available via `GenerateText`. Reasoning summaries are delivered as `ChunkReasoning` chunks during streaming rather than appearing in `ProviderMetadata`.

### Result() - block for final result

Blocks until the stream completes, then returns a `*TextResult` with accumulated text, tool calls, usage, and metadata.

```go
result := stream.Result()
fmt.Println(result.Text)
fmt.Printf("Tokens: %d\n", result.TotalUsage.TotalTokens)
```

## Goroutine Lifetime and Leaks

Both `TextStream` and `ObjectStream[T]` start a background goroutine when created.
This goroutine runs until the stream is fully consumed or the context is cancelled.

**Always do one of the following:**

- Consume the full stream via `Stream()`, `TextStream()`, or `PartialObjectStream()` channels
- Call `Result()` which drains the stream internally
- Cancel the context passed to `StreamText`/`StreamObject`

Discarding a stream without consuming it or cancelling the context will leak the goroutine.
The leaked goroutine will be blocked writing to its output channel until the process exits.

## Mutual Exclusivity

`TextStream()` and `Stream()` are mutually exclusive - call only one of them. Only the first call starts the background consumer goroutine; calling the second one after the first returns a closed channel.

`Result()` can always be called, including after `TextStream()` or `Stream()`. If called after a streaming method, it waits for the stream to finish and returns the accumulated data.

After consuming the stream, call `Err()` to check for any errors that occurred during streaming:

```go
for text := range stream.TextStream() {
    fmt.Print(text)
}
if err := stream.Err(); err != nil {
    log.Fatal("stream error:", err)
}
```

A common pattern is to stream text to the user, then inspect the final result:

```go
stream, err := goai.StreamText(ctx, model,
    goai.WithPrompt("Explain monads."),
)
if err != nil {
    log.Fatal(err)
}

for chunk := range stream.TextStream() {
    fmt.Print(chunk)
}
fmt.Println()

result := stream.Result()
fmt.Printf("Finish reason: %s, tokens: %d\n",
    result.FinishReason, result.TotalUsage.TotalTokens)
```

## Streaming Tool Loops

`StreamText` supports the same auto tool loop as `GenerateText`. Pass `WithMaxSteps` and tools with `Execute` functions. All steps stream through a single unified channel, with `ChunkStepFinish` marking step boundaries.

```go
stream, err := goai.StreamText(ctx, model,
    goai.WithPrompt("What's the weather in Tokyo and London?"),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
)
if err != nil {
    log.Fatal(err)
}

for chunk := range stream.Stream() {
    switch chunk.Type {
    case provider.ChunkText:
        fmt.Print(chunk.Text)
    case provider.ChunkToolCall:
        fmt.Printf("\n[tool call: %s]\n", chunk.ToolName)
    case provider.ChunkStepFinish:
        fmt.Printf("[step done, reason=%s]\n", chunk.FinishReason)
    case provider.ChunkFinish:
        fmt.Printf("\n[finished, tokens=%d]\n", chunk.Usage.TotalTokens)
    }
}

if err := stream.Err(); err != nil {
    log.Fatal("stream error:", err)
}

result := stream.Result()
fmt.Printf("Completed in %d steps\n", len(result.Steps))
```

## Non-Streaming Alternative

`GenerateText` blocks until the full response is ready. No stream, no channels - just the result.

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What is 2+2?"),
)
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

## Streaming Objects

`StreamObject[T]` returns an `*ObjectStream[T]` which provides both incremental partial objects and a final typed result:

```go
stream, err := goai.StreamObject[MyStruct](ctx, model,
    goai.WithPrompt("Generate a struct"),
)
if err != nil {
    log.Fatal(err)
}

// PartialObjectStream() returns a channel of partial typed objects
// Use this for progressive UI updates as fields become available
for partial := range stream.PartialObjectStream() {
    fmt.Printf("title=%q sections=%d\n", partial.Title, len(partial.Sections))
}

// Result() blocks until the stream completes, then returns the final object
result, err := stream.Result()
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Parsed: %+v\n", result.Object)

// Err() checks for streaming errors
if err := stream.Err(); err != nil {
    log.Fatal("stream error:", err)
}
```

### ObjectStream vs PartialObjectStream

| Method                  | Use Case                  | Returns                            |
| ----------------------- | ------------------------- | ---------------------------------- |
| `PartialObjectStream()` | Progressive typed updates | `*T` partial objects               |
| `Result()`              | Final typed result        | `*ObjectResult[T]` with `Object T` |

`ObjectStream` does not expose a raw `Stream()` method; use `PartialObjectStream()` for incremental updates and `Result()` for the final value.

## Hooks on Streaming Calls

`OnRequest` and `OnResponse` hooks fire on streaming paths similarly to `GenerateText`:

- `OnRequest` fires before the stream begins (before the HTTP request is issued).
- `OnResponse` fires after each step is drained (including intermediate tool-call steps). For non-tool-loop streaming it fires once. If the initial `StreamText` call itself returns an error (before any chunks are read), `OnResponse` fires immediately with that error.

```go
stream, err := goai.StreamText(ctx, model,
    goai.WithPrompt("Summarize the Go spec."),
    goai.WithOnRequest(func(info goai.RequestInfo) {
        log.Printf("request: model=%s messages=%d", info.Model, info.MessageCount)
    }),
    goai.WithOnResponse(func(info goai.ResponseInfo) {
        log.Printf("response: latency=%v tokens=%d err=%v",
            info.Latency, info.Usage.TotalTokens, info.Error)
    }),
)
```

`OnToolCall` and `OnStepFinish` also fire during streaming tool loops, with the same semantics as in `GenerateText`.

> **Note:** In streaming tool loops, provider-specific metadata is attached to each `ChunkStepFinish.Metadata` under keys like `"providerMetadata"`. The final `ChunkFinish` includes aggregated usage, finish reason, and final response metadata.

## TextResult Fields

Both `StreamText` (via `Result()`) and `GenerateText` return a `*TextResult`:

| Field              | Type                        | Description                                   |
| ------------------ | --------------------------- | --------------------------------------------- |
| `Text`             | `string`                    | Accumulated generated text                    |
| `ToolCalls`        | `[]provider.ToolCall`       | Tool calls from the final step                |
| `Steps`            | `[]goai.StepResult`         | Per-step results (for multi-step tool loops)  |
| `TotalUsage`       | `provider.Usage`            | Aggregated token usage across all steps       |
| `FinishReason`     | `provider.FinishReason`     | Why generation stopped                        |
| `Response`         | `provider.ResponseMetadata` | Provider metadata (response ID, actual model) |
| `ProviderMetadata` | `map[string]map[string]any` | Provider-specific response data               |
| `Sources`          | `[]provider.Source`         | Citations/references from the response        |
