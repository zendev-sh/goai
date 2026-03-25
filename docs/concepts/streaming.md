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

Returns a `<-chan provider.StreamChunk` with all chunk types: text, tool calls, finish events, usage data.

```go
for chunk := range stream.Stream() {
    switch chunk.Type {
    case provider.ChunkText:
        fmt.Print(chunk.Text)
    case provider.ChunkToolCall:
        fmt.Printf("Tool call: %s(%s)\n", chunk.ToolName, chunk.ToolInput)
    case provider.ChunkFinish:
        fmt.Printf("Done. Tokens: %d\n", chunk.Usage.TotalTokens)
    }
}
```

### Result() - block for final result

Blocks until the stream completes, then returns a `*TextResult` with accumulated text, tool calls, usage, and metadata.

```go
result := stream.Result()
fmt.Println(result.Text)
fmt.Printf("Tokens: %d\n", result.TotalUsage.TotalTokens)
```

## Mutual Exclusivity

`TextStream()` and `Stream()` are mutually exclusive - call only one of them. Both start a background goroutine that consumes from the provider's stream. Calling the second one after the first returns a closed channel.

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

## TextResult Fields

Both `StreamText` (via `Result()`) and `GenerateText` return a `*TextResult`:

| Field          | Type                       | Description                                  |
|----------------|---------------------------|-----------------------------------------------|
| `Text`         | `string`                   | Accumulated generated text                    |
| `ToolCalls`    | `[]provider.ToolCall`      | Tool calls from the final step                |
| `Steps`        | `[]StepResult`             | Per-step results (for multi-step tool loops)  |
| `TotalUsage`   | `provider.Usage`           | Aggregated token usage across all steps       |
| `FinishReason` | `provider.FinishReason`    | Why generation stopped                        |
| `Response`     | `provider.ResponseMetadata`| Provider metadata (response ID, actual model) |
| `Sources`      | `[]provider.Source`        | Citations/references from the response        |
