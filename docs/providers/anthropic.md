---
title: Anthropic Provider
description: "Use Anthropic Claude models in Go with GoAI. Supports the Messages API, extended thinking, prompt caching, and 10 provider-defined tools."
---

# Anthropic

[Anthropic](https://www.anthropic.com/) provider for GoAI - Claude models via the Messages API with native SSE streaming.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

Set the `ANTHROPIC_API_KEY` environment variable, or pass it explicitly:

```go
import "github.com/zendev-sh/goai/provider/anthropic"

model := anthropic.Chat("claude-sonnet-4-20250514", anthropic.WithAPIKey("sk-ant-..."))
```

The provider also reads `ANTHROPIC_BASE_URL` from the environment when no explicit base URL is set.

## Models

| Model ID | Type | Notes |
|----------|------|-------|
| `claude-opus-4-6-20260310` | Chat | Latest Opus, supports native output_format |
| `claude-sonnet-4-6-20260310` | Chat | Latest Sonnet, supports native output_format |
| `claude-opus-4-5-20250520` | Chat | Supports native output_format |
| `claude-sonnet-4-5-20241022` | Chat | Supports native output_format |
| `claude-opus-4-1-20250805` | Chat | Supports native output_format |
| `claude-sonnet-4-20250514` | Chat | Standard Sonnet |
| `claude-haiku-3-5-20241022` | Chat | Fastest Claude model |

## Tested Models

Unit tested with mock HTTP server (100% coverage). Last run: 2026-03-15.

| Model | Generate | Stream | Status |
|-------|----------|--------|--------|
| `claude-sonnet-4-20250514` | PASS | PASS | Stable |
| `claude-sonnet-4-5-20241022` | PASS | PASS | Stable |
| `claude-sonnet-4-6-20260310` | PASS | PASS | Via Bedrock/Azure E2E |

Claude models are also E2E tested via [Azure](azure.md) (claude-opus-4-6, claude-sonnet-4-6) and [Bedrock](bedrock.md) (4 Claude cross-region variants: us, eu, global, bare).

## Usage

### Chat

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/anthropic"
)

func main() {
    model := anthropic.Chat("claude-sonnet-4-20250514")
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
    "github.com/zendev-sh/goai/provider/anthropic"
)

model := anthropic.Chat("claude-sonnet-4-20250514")
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

### Extended Thinking

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Solve this step by step: What is 127 * 43?"),
    goai.WithProviderOptions(map[string]any{
        "thinking": map[string]any{
            "type":         "enabled",
            "budgetTokens": 8192,
        },
    }),
)
```

Thinking types: `"enabled"` (with `budgetTokens`), `"adaptive"`, `"disabled"`.

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Static API key. Falls back to `ANTHROPIC_API_KEY` env var. |
| `WithTokenSource(ts)` | `provider.TokenSource` | Dynamic token resolution. |
| `WithBaseURL(url)` | `string` | Override base URL. Falls back to `ANTHROPIC_BASE_URL` env var. |
| `WithHeaders(h)` | `map[string]string` | Additional HTTP headers on every request. |
| `WithHTTPClient(c)` | `*http.Client` | Custom HTTP client for proxies, logging, URL rewriting. |

### Provider Options (via `goai.WithProviderOptions`)

| Key | Type | Description |
|-----|------|-------------|
| `thinking` | `map[string]any` | Extended thinking config: `{type, budgetTokens}`. |
| `disableParallelToolUse` | `bool` | Disable parallel tool calls. |
| `effort` | `string` | Output quality level: `"low"`, `"medium"`, `"high"`, `"max"`. |
| `speed` | `string` | Inference speed: `"fast"`, `"standard"`. |
| `structuredOutputMode` | `string` | `"outputFormat"` for native JSON Schema, `"auto"` for auto-detect. |
| `container` | `map[string]any` | Code execution container: `{id, skills}`. |
| `contextManagement` | `map[string]any` | Automatic context window management edits. |

## Provider Tools

Ten built-in tools are available via `anthropic.Tools`. These use Anthropic's native tool types.

| Tool | Version | Description | Execution |
|------|---------|-------------|-----------|
| `anthropic.Tools.Computer()` | 20250124 | Screenshot, mouse/keyboard control | Client-side |
| `anthropic.Tools.Computer_20251124()` | 20251124 | Computer use with zoom support | Client-side |
| `anthropic.Tools.Bash()` | 20250124 | Execute bash commands | Client-side |
| `anthropic.Tools.TextEditor()` | 20250429 | View/edit files (str_replace) | Client-side |
| `anthropic.Tools.TextEditor_20250728()` | 20250728 | Text editor with maxCharacters | Client-side |
| `anthropic.Tools.WebSearch()` | 20250305 | Web search via Brave Search | Server-side |
| `anthropic.Tools.WebSearch_20260209()` | 20260209 | Updated web search | Server-side |
| `anthropic.Tools.WebFetch()` | 20260209 | Fetch URL content | Server-side |
| `anthropic.Tools.CodeExecution()` | 20260120 | Server-side Python execution | Server-side |
| `anthropic.Tools.CodeExecution_20250825()` | 20250825 | Earlier code execution version | Server-side |

### Computer Use

Computer use tools require client-side execution. The model generates actions (click, type, screenshot), and your application executes them and returns results.

```go
def := anthropic.Tools.Computer(anthropic.ComputerToolOptions{
    DisplayWidthPx:  1920,
    DisplayHeightPx: 1080,
})
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Open the browser and navigate to example.com"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
// Handle tool calls in result.ToolCalls - execute actions and return results
```

The `Computer_20251124` version adds zoom support for Opus 4.5+:

```go
def := anthropic.Tools.Computer_20251124(anthropic.Computer20251124Options{
    DisplayWidthPx:  1920,
    DisplayHeightPx: 1080,
    EnableZoom:      true,
})
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Take a zoomed screenshot of the top-left corner"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

### Bash and TextEditor

```go
bashDef := anthropic.Tools.Bash()
editorDef := anthropic.Tools.TextEditor()
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("List files in the current directory"),
    goai.WithTools(
        goai.Tool{Name: bashDef.Name, ProviderDefinedType: bashDef.ProviderDefinedType, ProviderDefinedOptions: bashDef.ProviderDefinedOptions},
        goai.Tool{Name: editorDef.Name, ProviderDefinedType: editorDef.ProviderDefinedType, ProviderDefinedOptions: editorDef.ProviderDefinedOptions},
    ),
)
```

`TextEditor_20250728` supports an optional character limit:

```go
def := anthropic.Tools.TextEditor_20250728(anthropic.WithMaxCharacters(100000))
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Open main.go and fix the syntax error"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

### WebSearch

```go
def := anthropic.Tools.WebSearch(
    anthropic.WithMaxUses(5),
    anthropic.WithAllowedDomains("go.dev", "github.com"),
)
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What are the latest developments in Go 1.25?"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

Options: `WithMaxUses(n)`, `WithAllowedDomains(...)`, `WithBlockedDomains(...)`, `WithWebSearchUserLocation(...)`.

### WebFetch

```go
def := anthropic.Tools.WebFetch(
    anthropic.WithCitations(true),
    anthropic.WithMaxContentTokens(4096),
)
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Summarize the content at https://go.dev/blog"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

Options: `WithWebFetchMaxUses(n)`, `WithWebFetchAllowedDomains(...)`, `WithWebFetchBlockedDomains(...)`, `WithCitations(bool)`, `WithMaxContentTokens(n)`.

### CodeExecution

```go
def := anthropic.Tools.CodeExecution()
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Calculate the standard deviation of [4, 8, 15, 16, 23, 42]"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

## Notes

- **Prompt caching**: When `goai.WithPromptCaching(true)` is set, the system prompt receives `cache_control: {type: "ephemeral"}`. Message-level caching is controlled via `ProviderOptions["anthropic"]["cacheControl"]` on the message, or via `Part.CacheControl` on individual content parts.
- **Structured output**: Anthropic does not natively support JSON Schema `response_format` on all models. GoAI uses a synthetic tool injection pattern - a hidden tool with the schema is injected, and the model is forced to call it. For Claude Opus 4.1+, Sonnet 4.5+, Opus 4.5+, Sonnet 4.6, and Opus 4.6, set `structuredOutputMode: "outputFormat"` to use native `output_format` instead.
- **Beta headers**: Provider-defined tools automatically add the required `anthropic-beta` header values. The base beta features (`claude-code-20250219`, `interleaved-thinking-2025-05-14`) are always included.
- **Default max tokens**: 16384 when not explicitly set via `goai.WithMaxOutputTokens()`.
- **Auth header**: Uses `x-api-key` (not `Authorization: Bearer`), matching Anthropic's API convention.
- **Input modalities**: Supports text, images (base64), and PDF documents (base64).
