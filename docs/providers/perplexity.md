---
title: Perplexity Provider
description: "Use Perplexity Sonar models for search-augmented generation in Go with GoAI. Built-in web search with automatic citation support."
---

# Perplexity

[Perplexity AI](https://www.perplexity.ai/) provider using the OpenAI-compatible Chat Completions API. Perplexity models specialize in search-augmented generation with built-in citations.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/perplexity"
```

Set the `PERPLEXITY_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

- `sonar-pro` - large search model
- `sonar` - standard search model
- `sonar-reasoning-pro` - reasoning with search
- `sonar-reasoning` - reasoning with search (smaller)

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `sonar-pro`

## Usage

```go
model := perplexity.Chat("sonar-pro")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("What happened in tech news today?"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source |
| `WithBaseURL(url)` | `string` | Override the default `https://api.perplexity.ai` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Perplexity models perform live web searches and include citations/sources in responses. Citation URLs are returned in `result.Sources` as `provider.Source` entries with `Type: "url"`.
- Environment variable `PERPLEXITY_BASE_URL` can override the default endpoint.
