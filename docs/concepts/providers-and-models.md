---
title: Providers and Models
description: "Understand GoAI's provider architecture, model interfaces, capabilities system, and the full list of 20+ supported LLM providers."
---

# Providers and Models

A **provider** is an AI service (OpenAI, Anthropic, Google, etc.). A **model** is a specific model offered by that provider (gpt-4o, claude-sonnet-4-20250514, gemini-2.0-flash).

GoAI uses a factory pattern: each provider package exports a `Chat()` function that returns a model implementing the `provider.LanguageModel` interface.

```go
import "github.com/zendev-sh/goai/provider/openai"

model := openai.Chat("gpt-4o")
```

The model is then passed to GoAI functions like `StreamText` or `GenerateText`:

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Explain quantum computing in one sentence."),
)
```

## Model Interfaces

GoAI defines three model interfaces in the `provider` package.

### LanguageModel

Text generation and tool calling.

```go
type LanguageModel interface {
    ModelID() string
    DoGenerate(ctx context.Context, params GenerateParams) (*GenerateResult, error)
    DoStream(ctx context.Context, params GenerateParams) (*StreamResult, error)
}
```

Used with `GenerateText`, `StreamText`, `GenerateObject`, and `StreamObject`.

Models may also implement the optional `CapableModel` interface to declare capabilities. Use `provider.ModelCapabilitiesOf(model)` to query safely.

### EmbeddingModel

Vector embeddings from text.

```go
type EmbeddingModel interface {
    ModelID() string
    DoEmbed(ctx context.Context, values []string, params EmbedParams) (*EmbedResult, error)
    MaxValuesPerCall() int
}
```

Used with `Embed` and `EmbedMany`.

### ImageModel

Image generation from text prompts.

```go
type ImageModel interface {
    ModelID() string
    DoGenerate(ctx context.Context, params ImageParams) (*ImageResult, error)
}
```

Used with `GenerateImage`.

## Capabilities

Models that implement the optional `CapableModel` interface expose their capabilities. Use `provider.ModelCapabilitiesOf` to query safely (returns zero-value if not implemented):

```go
caps := provider.ModelCapabilitiesOf(model)
if caps.ToolCall {
    // safe to pass tools
}
if caps.Reasoning {
    // model supports extended thinking
}
```

The `ModelCapabilities` struct includes:

| Field              | Type          | Description                               |
| ------------------ | ------------- | ----------------------------------------- |
| `Temperature`      | `bool`        | Accepts temperature parameter             |
| `Reasoning`        | `bool`        | Extended thinking / chain-of-thought      |
| `Attachment`       | `bool`        | File attachment support                   |
| `ToolCall`         | `bool`        | Tool / function calling                   |
| `InputModalities`  | `ModalitySet` | Supported input types (text, image, etc.) |
| `OutputModalities` | `ModalitySet` | Supported output types                    |

## Provider Configuration

Every provider accepts options for authentication and customization:

```go
// Explicit API key
model := openai.Chat("gpt-4o", openai.WithAPIKey("sk-..."))

// Dynamic token source (OAuth, service accounts)
model := openai.Chat("gpt-4o", openai.WithTokenSource(myTokenSource))

// Custom HTTP client (proxies, logging, URL rewrite)
model := openai.Chat("gpt-4o", openai.WithHTTPClient(myClient))
```

If no API key or token source is provided, providers read from environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Provider List

| Provider   | Import Path                                     | Factory Functions            |
| ---------- | ----------------------------------------------- | ---------------------------- |
| OpenAI     | `github.com/zendev-sh/goai/provider/openai`     | `Chat`, `Embedding`, `Image` |
| Anthropic  | `github.com/zendev-sh/goai/provider/anthropic`  | `Chat`                       |
| Google     | `github.com/zendev-sh/goai/provider/google`     | `Chat`, `Embedding`, `Image` |
| Azure      | `github.com/zendev-sh/goai/provider/azure`      | `Chat`, `Image`              |
| Vertex AI  | `github.com/zendev-sh/goai/provider/vertex`     | `Chat`, `Embedding`, `Image` |
| Bedrock    | `github.com/zendev-sh/goai/provider/bedrock`    | `Chat`                       |
| Mistral    | `github.com/zendev-sh/goai/provider/mistral`    | `Chat`                       |
| xAI        | `github.com/zendev-sh/goai/provider/xai`        | `Chat`                       |
| Groq       | `github.com/zendev-sh/goai/provider/groq`       | `Chat`                       |
| DeepInfra  | `github.com/zendev-sh/goai/provider/deepinfra`  | `Chat`                       |
| OpenRouter | `github.com/zendev-sh/goai/provider/openrouter` | `Chat`                       |
| DeepSeek   | `github.com/zendev-sh/goai/provider/deepseek`   | `Chat`                       |
| Fireworks  | `github.com/zendev-sh/goai/provider/fireworks`  | `Chat`                       |
| Together   | `github.com/zendev-sh/goai/provider/together`   | `Chat`                       |
| Cohere     | `github.com/zendev-sh/goai/provider/cohere`     | `Chat`, `Embedding`          |
| Cerebras   | `github.com/zendev-sh/goai/provider/cerebras`   | `Chat`                       |
| Perplexity | `github.com/zendev-sh/goai/provider/perplexity` | `Chat`                       |
| Ollama     | `github.com/zendev-sh/goai/provider/ollama`     | `Chat`, `Embedding`          |
| vLLM       | `github.com/zendev-sh/goai/provider/vllm`       | `Chat`, `Embedding`          |
| Compat     | `github.com/zendev-sh/goai/provider/compat`     | `Chat`, `Embedding`          |

The `compat` provider works with any OpenAI-compatible API. Pass a custom base URL:

```go
import "github.com/zendev-sh/goai/provider/compat"

model := compat.Chat("my-model",
    compat.WithBaseURL("https://my-api.example.com/v1"),
    compat.WithAPIKey("my-key"),
)
```
