---
title: xAI Provider
description: "Use xAI Grok models in Go with GoAI. Supports Grok-3 chat, vision capabilities, and provider-defined web search and X search tools."
---

# xAI (Grok)

xAI provider for Grok models, using the OpenAI-compatible Chat Completions API. Includes provider-defined tools for web and X (Twitter) search.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/xai"
```

Set the `XAI_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

- `grok-3`, `grok-3-mini`
- `grok-2`, `grok-2-vision`

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `grok-3`

## Usage

```go
model := xai.Chat("grok-3")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("What is Grok?"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

## Provider Tools

xAI provides two built-in tool definitions for search capabilities:

### WebSearch

```go
def := xai.Tools.WebSearch(
    xai.WithAllowedDomains("wikipedia.org", "arxiv.org"),  // max 5
    xai.WithExcludedDomains("example.com"),                // max 5
    xai.WithWebSearchImageUnderstanding(true),
)

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What are the latest Go release notes?"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

### XSearch

```go
def := xai.Tools.XSearch(
    xai.WithAllowedXHandles("@elonmusk"),                  // max 10
    xai.WithExcludedXHandles("@spambot"),                  // max 10
    xai.WithXSearchDateRange("2025-01-01", "2025-12-31"),  // ISO 8601
    xai.WithXSearchImageUnderstanding(true),
    xai.WithXSearchVideoUnderstanding(true),
)

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What is trending on X about Go programming?"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

**Status**: These tool definitions are currently blocked on the Chat Completions API. They require the xAI Responses API, which is not yet supported. The tool definitions are available for forward compatibility.

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source |
| `WithBaseURL(url)` | `string` | Override the default `https://api.x.ai/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Supports image inputs (vision models).
- Environment variable `XAI_BASE_URL` can override the default endpoint.
