---
title: OpenAI Provider
description: "Use OpenAI GPT-4o, o3, and GPT-5 models in Go with GoAI. Supports Chat Completions and Responses API, embeddings, and image generation."
---

# OpenAI

OpenAI provider for GoAI - supports Chat Completions API and Responses API with automatic routing.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

Set the `OPENAI_API_KEY` environment variable, or pass it explicitly:

```go
import "github.com/zendev-sh/goai/provider/openai"

model := openai.Chat("gpt-4o", openai.WithAPIKey("sk-..."))
```

The provider also reads `OPENAI_BASE_URL` from the environment when no explicit base URL is set.

## Models

| Model ID | Type | Notes |
|----------|------|-------|
| `gpt-4o` | Chat | Responses API by default |
| `gpt-4o-mini` | Chat | Responses API by default |
| `gpt-5` | Chat | Reasoning model, Responses API by default |
| `gpt-5-mini` | Chat | Reasoning model, Responses API by default |
| `o3` | Chat | Reasoning model, Responses API by default |
| `o4-mini` | Chat | Reasoning model, Responses API by default |
| `codex-*` | Chat | Reasoning model, Responses API by default |
| `text-embedding-3-small` | Embedding | 1536 dimensions |
| `text-embedding-3-large` | Embedding | 3072 dimensions |
| `text-embedding-ada-002` | Embedding | Legacy |
| `dall-e-3` | Image | Legacy image generation |
| `gpt-image-1` | Image | Defaults to b64_json |

## Tested Models

Unit tested with mock HTTP server (100% coverage). Last run: 2026-03-15.

| Model | Generate | Stream | Status |
|-------|----------|--------|--------|
| `gpt-4o` | PASS | PASS | Stable |
| `o3` | PASS | PASS | Stable |
| `text-embedding-3-small` | PASS | N/A | Stable |
| `dall-e-3` | PASS | N/A | Stable |
| `gpt-image-1` | PASS | N/A | Stable |

OpenAI models are also E2E tested via the [Azure provider](azure.md) (21 models PASS including gpt-4.1, gpt-5, gpt-5.1, o3).

## Usage

### Chat

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

func main() {
    model := openai.Chat("gpt-4o")
    result, err := goai.GenerateText(context.Background(), model,
        goai.WithPrompt("Explain Go interfaces in one paragraph."),
    )
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Text)
}
```

### Streaming

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider"
    "github.com/zendev-sh/goai/provider/openai"
)

model := openai.Chat("gpt-4o")
stream, err := goai.StreamText(context.Background(), model,
    goai.WithPrompt("Write a haiku about Go."),
)
if err != nil {
    panic(err)
}
for chunk := range stream.Stream() {
    if chunk.Type == provider.ChunkText {
        fmt.Print(chunk.Text)
    }
}
```

### Embedding

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

model := openai.Embedding("text-embedding-3-small")
result, err := goai.Embed(context.Background(), model, "Hello world")
if err != nil {
    panic(err)
}
fmt.Println(len(result.Embedding)) // 1536
```

### Image Generation

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

model := openai.Image("gpt-image-1")
result, err := goai.GenerateImage(context.Background(), model,
    goai.WithImagePrompt("A futuristic city skyline"),
)
if err != nil {
    panic(err)
}
fmt.Printf("Generated %d bytes\n", len(result.Images[0].Data))
```

## API Routing

All models use the Responses API (`/v1/responses`) by default. To force Chat Completions:

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Hello"),
    goai.WithProviderOptions(map[string]any{
        "useResponsesAPI": false,
    }),
)
```

**Reasoning model detection** is automatic based on model ID:
- `o1`, `o3`, `o4` (o-series) - reasoning enabled, temperature disabled
- `gpt-5*` (except `gpt-5-chat`) - reasoning enabled
- `codex-*` - reasoning enabled

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Static API key. Falls back to `OPENAI_API_KEY` env var. |
| `WithTokenSource(ts)` | `provider.TokenSource` | Dynamic token resolution (OAuth, service accounts). |
| `WithBaseURL(url)` | `string` | Override base URL. Falls back to `OPENAI_BASE_URL` env var. |
| `WithHeaders(h)` | `map[string]string` | Additional HTTP headers on every request. |
| `WithHTTPClient(c)` | `*http.Client` | Custom HTTP client for proxies, logging, URL rewriting. |

### Provider Options (via `goai.WithProviderOptions`)

| Key | Type | Description |
|-----|------|-------------|
| `useResponsesAPI` | `bool` | Force Responses API (`true`) or Chat Completions (`false`). Default: `true`. |
| `store` | `bool` | Persist responses for later retrieval. |
| `serviceTier` | `string` | Service tier (e.g., `"auto"`, `"flex"`). |
| `parallelToolCalls` | `bool` | Allow parallel tool calls. |
| `reasoning_effort` | `string` | Reasoning effort level (`"low"`, `"medium"`, `"high"`). |
| `reasoning_summary` | `string` | Reasoning summary mode. |
| `text_verbosity` | `string` | Text verbosity level. |
| `metadata` | `map[string]any` | Request metadata. |
| `logprobs` | `bool` or `int` | Enable log probabilities. |
| `truncation` | `string` | Truncation strategy for Responses API. |
| `include` | `[]string` | Additional data to include in Responses API response. |
| `user` | `string` | End-user identifier for abuse monitoring. |
| `instructions` | `string` | System instructions (Responses API alternative to system message). |
| `previousResponseId` | `string` | Chain responses (Responses API). |
| `maxToolCalls` | `int` | Maximum number of tool calls per turn. |
| `conversation` | `map[string]any` | Conversation context (Responses API). |
| `prompt_cache_key` | `string` | Cache key for prompt caching (Responses API). |
| `safetyIdentifier` | `string` | Safety identifier for content filtering. |

## Provider Tools

Four built-in tools are available via `openai.Tools`. These require the Responses API.

| Tool | Description |
|------|-------------|
| `openai.Tools.WebSearch()` | Server-side web search. Model decides when to search. |
| `openai.Tools.CodeInterpreter()` | Server-side Python code execution in a sandbox. |
| `openai.Tools.FileSearch()` | Semantic/keyword search over uploaded files via vector stores. |
| `openai.Tools.ImageGeneration()` | Generate images within a conversation using GPT Image. |

### WebSearch

```go
def := openai.Tools.WebSearch(
    openai.WithSearchContextSize("medium"),
    openai.WithUserLocation(openai.WebSearchLocation{
        Country: "US",
        City:    "San Francisco",
    }),
)
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What happened in tech news today?"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

Options: `WithSearchContextSize("low"|"medium"|"high")`, `WithUserLocation(...)`, `WithSearchFilters(...)`, `WithExternalWebAccess(bool)`.

### CodeInterpreter

```go
def := openai.Tools.CodeInterpreter()
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Calculate the first 20 Fibonacci numbers."),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

Options: `WithContainerID(containerID)` for an existing container, or `WithContainerFiles(&openai.CodeInterpreterContainer{FileIDs: [...]})` for auto-provisioned.

### FileSearch

```go
def := openai.Tools.FileSearch(
    openai.WithVectorStoreIDs("vs_abc123"),
    openai.WithMaxNumResults(5),
)
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Find information about error handling."),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

Options: `WithVectorStoreIDs(...)`, `WithMaxNumResults(n)`, `WithRanking(...)`, `WithFileSearchFilters(...)`.

### ImageGeneration

```go
def := openai.Tools.ImageGeneration(
    openai.WithImageQuality("high"),
    openai.WithImageSize("1024x1024"),
)
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Generate an image of a sunset over mountains."),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

Options: `WithBackground("auto"|"opaque"|"transparent")`, `WithInputFidelity("low"|"high")`, `WithImageModel("gpt-image-1")`, `WithOutputFormat("png"|"jpeg"|"webp")`, `WithImageQuality("auto"|"low"|"medium"|"high")`, `WithImageSize("auto"|"1024x1024"|"1024x1536"|"1536x1024")`, `WithOutputCompression(0-100)`, `WithPartialImages(0-3)`, `WithModeration("auto"|"low")`, `WithInputImageMask(ImageGenerationMask{...})`.

## Custom Routing with WithHTTPClient

Use `WithHTTPClient` to inject a custom `http.Client` for proxy support, request logging, or custom auth flows:

```go
// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
    return f(req)
}

// Example: route through a proxy with custom auth
transport := &http.Transport{}
client := &http.Client{
    Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
        req.URL.Host = "my-proxy.example.com"
        req.Header.Set("X-Custom-Auth", "token-here")
        return transport.RoundTrip(req)
    }),
}

model := openai.Chat("gpt-4o", openai.WithHTTPClient(client))
```

This pattern supports Copilot (URL rewrite + OAuth token swap) and Codex (URL rewrite + session headers) without separate provider implementations.

## Notes

- The embedding model supports up to 2048 values per batch call via `MaxValuesPerCall()`. `goai.EmbedMany` auto-chunks larger batches.
- Image models `gpt-image-1`, `gpt-image-1-mini`, and `gpt-image-1.5` default to `b64_json` response format and reject an explicit `response_format` parameter. For older models like `dall-e-3`, the provider automatically sets `response_format` to `b64_json`.
- Structured output uses `response_format` with `json_schema` for both APIs. The Responses API places it under `text.format`.
- Per-request headers can be injected via `goai.WithHeaders(map[string]string{...})` for features like Codex session tracking.
