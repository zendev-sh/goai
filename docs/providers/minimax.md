---
title: MiniMax Provider
description: "Use MiniMax M2.7, M2.5, M2.1, and M2 models in Go with GoAI via MiniMax's Anthropic-compatible API. Supports thinking/reasoning, tool calling, and streaming."
---

# MiniMax

[MiniMax](https://www.minimax.io/) provider using the Anthropic-compatible API (recommended by MiniMax). Delegates to GoAI's Anthropic provider with MiniMax's endpoint.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/minimax"
```

Set the `MINIMAX_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

| Model | Speed | Context | Notes |
|-------|-------|---------|-------|
| `MiniMax-M2.7` | ~60 tps | 204,800 | Latest, self-evolving architecture |
| `MiniMax-M2.7-highspeed` | ~100 tps | 204,800 | Fast variant (requires highspeed plan) |
| `MiniMax-M2.5` | ~60 tps | 204,800 | Coding SOTA |
| `MiniMax-M2.5-highspeed` | ~100 tps | 204,800 | Fast variant (requires highspeed plan) |
| `MiniMax-M2.1` | ~60 tps | 204,800 | MoE 230B |
| `MiniMax-M2.1-highspeed` | ~100 tps | 204,800 | Fast variant (requires highspeed plan) |
| `MiniMax-M2` | standard | 204,800 | First gen, agentic |

## Tested Models

**E2E tested** (real API calls, 2026-03-27): `MiniMax-M2.7`, `MiniMax-M2.5`, `MiniMax-M2.1`, `MiniMax-M2`

Generate, stream, thinking, and thinking stream verified on all 4 models. Tool calling verified on M2.7, M2.5, M2 (M2.1 requires system prompt for reliable tool output).

## Usage

### Basic

```go
model := minimax.Chat("MiniMax-M2.7")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Explain attention mechanisms"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

### Streaming

```go
model := minimax.Chat("MiniMax-M2.7")

stream, err := goai.StreamText(ctx, model, goai.WithPrompt("Write a haiku"))
if err != nil {
    log.Fatal(err)
}
for chunk := range stream.TextStream() {
    fmt.Print(chunk)
}
```

### Tool Calling

```go
model := minimax.Chat("MiniMax-M2.7")

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What is the weather in Tokyo?"),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(3),
)
```

### Thinking / Reasoning

MiniMax supports native Anthropic thinking blocks for step-by-step reasoning:

```go
model := minimax.Chat("MiniMax-M2.7")

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Solve step by step: what is 23 * 47?"),
    goai.WithProviderOptions(map[string]any{
        "thinking": map[string]any{
            "type":         "enabled",
            "budgetTokens": 1000,
        },
    }),
)
// Reasoning content in result.ProviderMetadata["anthropic"]["reasoning"]
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source |
| `WithBaseURL(url)` | `string` | Override the default `https://api.minimax.io/anthropic` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Uses MiniMax's **Anthropic-compatible API** (recommended by MiniMax over OpenAI-compatible).
- Text-only input/output. **Image and document input not supported.**
- Supports **thinking/reasoning** via Anthropic's native thinking blocks (`ProviderOptions["thinking"]`).
- Environment variable `MINIMAX_BASE_URL` can override the default endpoint (e.g., for China: `https://api.minimaxi.com/anthropic`).
- M2.1 tool calling requires a system prompt for reliable output.
- Highspeed model variants require a specific MiniMax token plan.
