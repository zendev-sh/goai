---
title: Quick Start
description: "Get started with GoAI in minutes. Generate and stream text with OpenAI, Anthropic, Google, and 21+ LLM providers using a unified Go API."
---

# Quick Start

## Generate Text

The simplest call - send a prompt, get a response:

```go
package main

import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

func main() {
    model := openai.Chat("gpt-4o")

    result, err := goai.GenerateText(context.Background(), model,
        goai.WithPrompt("What is the capital of France?"),
    )
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Text)
    fmt.Printf("Tokens: %d in, %d out\n",
        result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
}
```

`GenerateText` waits for the full response before returning. Use `WithSystem` to set a system prompt:

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithSystem("You are a helpful assistant that responds in haiku."),
    goai.WithPrompt("Tell me about Go."),
)
```

## Stream Text

For real-time output, use `StreamText`. It returns a `TextStream` with channel-based consumption:

```go
stream, err := goai.StreamText(ctx, model,
    goai.WithPrompt("Write a short story about a robot."),
)
if err != nil {
    panic(err)
}

for text := range stream.TextStream() {
    fmt.Print(text)
}
fmt.Println()

// Get the final accumulated result after streaming completes.
result := stream.Result()
fmt.Printf("Tokens: %d in, %d out\n",
    result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
```

`TextStream()` returns a `<-chan string` that emits text fragments as they arrive. The channel closes when the stream ends. Call `Result()` afterward to access usage data and metadata.

## Switching Providers

Every provider follows the same pattern - import the provider package, create a model, pass it to any GoAI function. The rest of your code stays the same.

### OpenAI

```go
import "github.com/zendev-sh/goai/provider/openai"

model := openai.Chat("gpt-4o")
```

Environment variable: `OPENAI_API_KEY`

### Anthropic

```go
import "github.com/zendev-sh/goai/provider/anthropic"

model := anthropic.Chat("claude-sonnet-4-20250514")
```

Environment variable: `ANTHROPIC_API_KEY`

### Google Gemini

```go
import "github.com/zendev-sh/goai/provider/google"

model := google.Chat("gemini-2.5-flash")
```

Environment variable: `GOOGLE_GENERATIVE_AI_API_KEY` or `GEMINI_API_KEY`

### Others

All 21+ providers work the same way:

```go
import "github.com/zendev-sh/goai/provider/groq"
import "github.com/zendev-sh/goai/provider/mistral"
import "github.com/zendev-sh/goai/provider/deepseek"

groqModel := groq.Chat("llama-3.3-70b-versatile")
mistralModel := mistral.Chat("mistral-large-latest")
deepseekModel := deepseek.Chat("deepseek-chat")
```

See the [Providers & Models](/concepts/providers-and-models) page for the full provider table with supported models and environment variables.

## Authentication

Providers auto-resolve API keys from environment variables. No explicit key setup is needed if the correct variable is set.

To override the environment variable, pass the key explicitly:

```go
model := openai.Chat("gpt-4o", openai.WithAPIKey("sk-..."))
```

For OAuth, rotating keys, or cloud IAM, use `TokenSource`:

```go
import (
    "context"

    "github.com/zendev-sh/goai/provider"
    "github.com/zendev-sh/goai/provider/openai"
)

ts := provider.CachedTokenSource(func(ctx context.Context) (*provider.Token, error) {
    tok, err := fetchOAuthToken(ctx)
    return &provider.Token{
        Value:     tok.AccessToken,
        ExpiresAt: tok.Expiry,
    }, err
})

model := openai.Chat("gpt-4o", openai.WithTokenSource(ts))
```

## Common Options

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithSystem("You are a concise assistant."),
    goai.WithPrompt("Explain goroutines."),
    goai.WithMaxOutputTokens(200),
    goai.WithTemperature(0.7),
)
```

| Option | Description |
|--------|-------------|
| `WithSystem(s)` | System prompt |
| `WithPrompt(s)` | User message |
| `WithMaxOutputTokens(n)` | Limit response length |
| `WithTemperature(t)` | Randomness (0.0 - 2.0) |
| `WithMaxRetries(n)` | Retries on 429/5xx (default 2) |
| `WithTimeout(d)` | Overall timeout |

## Next Steps

- [Structured Output](structured-output.md) - type-safe responses with `GenerateObject[T]`
- [Concepts](/concepts/providers-and-models) - tools, embeddings, image generation, provider-defined tools, and more
