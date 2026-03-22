---
title: OpenRouter Provider
description: "Access models from multiple AI providers through OpenRouter's unified gateway in Go with GoAI. One API key for OpenAI, Anthropic, Google, and more."
---

# OpenRouter

OpenRouter multi-provider routing gateway using the OpenAI-compatible Chat Completions API. Access models from multiple providers through a single API key.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/openrouter"
```

Set the `OPENROUTER_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

Models use a `provider/model` prefix format:

- `anthropic/claude-sonnet-4`
- `openai/gpt-4o`
- `google/gemini-2.5-pro`
- `meta-llama/llama-3.3-70b-instruct`

See [openrouter.ai/models](https://openrouter.ai/models) for the full catalog.

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `anthropic/claude-sonnet-4`

## Usage

```go
model := openrouter.Chat("anthropic/claude-sonnet-4")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
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
| `WithBaseURL(url)` | `string` | Override the default `https://openrouter.ai/api/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Supports image inputs (depends on the underlying model).
- Automatically sends `HTTP-Referer` and `X-Title` headers as recommended by OpenRouter's API.
- Usage reporting is enabled by default (`usage: {include: true}` in request body).
- Environment variable `OPENROUTER_BASE_URL` can override the default endpoint.
