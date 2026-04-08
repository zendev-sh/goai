---
title: Multi-Turn Conversations
description: "Build multi-turn conversations with GoAI. Use ResponseMessages to preserve tool call history across turns."
---

# Multi-Turn Conversations

GoAI supports multi-turn conversations through `WithMessages`. After each generation call, `ResponseMessages` provides the full assistant and tool message history, ready to append to your conversation.

## The Problem

When using tools with `WithMaxSteps`, GoAI runs an automatic tool loop: generate, execute tools, re-generate. The result contains the final text, but your conversation history needs the intermediate tool call and tool result messages too. Without them, the model may re-execute tools on the next turn.

## ResponseMessages

`ResponseMessages` on `TextResult` (and `ObjectResult`) contains all assistant and tool messages produced during a generation call. Append it directly to your message history:

```go
var messages []provider.Message

// Turn 1
messages = append(messages, goai.UserMessage("What's the weather in Tokyo?"))
result, err := goai.GenerateText(ctx, model,
    goai.WithMessages(messages...),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
)
if err != nil {
    log.Fatal(err)
}

// Append the full response history (assistant + tool messages).
messages = append(messages, result.ResponseMessages...)

// Turn 2 - model sees the tool was already called, won't repeat it.
messages = append(messages, goai.UserMessage("Is that warm enough for a picnic?"))
result, err = goai.GenerateText(ctx, model,
    goai.WithMessages(messages...),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
)
```

## What ResponseMessages Contains

The contents depend on what happened during generation:

| Scenario | ResponseMessages |
|----------|-----------------|
| Simple text (no tools) | 1 assistant message with text |
| Tool loop (2 steps) | assistant (tool calls) + tool (results) + assistant (final text) |
| Parallel tool calls | Tool results merged into a single message per round-trip |
| MaxSteps exhausted | All completed round-trips (no duplicate final message) |
| Streaming with reasoning | Reasoning consolidated into a single `PartReasoning` with metadata |

## Streaming

`ResponseMessages` works the same way with `StreamText`. Call `Result()` after consuming the stream:

```go
stream, err := goai.StreamText(ctx, model,
    goai.WithMessages(messages...),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
)
if err != nil {
    log.Fatal(err)
}
// Consume the stream.
for chunk := range stream.Stream() {
    fmt.Print(chunk.Text)
}
// Check for errors before using ResponseMessages.
if err := stream.Err(); err != nil {
    log.Fatal(err)
}
// Append to history.
messages = append(messages, stream.Result().ResponseMessages...)
```

On stream errors, `ResponseMessages` may be partial. Always check `Err()` first.

## Structured Output

`GenerateObject` and `StreamObject` also populate `ResponseMessages`:

```go
result, err := goai.GenerateObject[MyStruct](ctx, model,
    goai.WithMessages(messages...),
    goai.WithTools(helperTool),
    goai.WithMaxSteps(5),
)
messages = append(messages, result.ResponseMessages...)
```

## Without ResponseMessages

If you prefer manual control, you can build messages from `result.Steps`. But you need to reconstruct the tool call parts and capture tool outputs via `WithOnToolCall`. `ResponseMessages` handles this automatically.

::: tip
See [`examples/multi-turn/`](https://github.com/zendev-sh/goai/tree/main/examples/multi-turn) for a runnable example.
:::
