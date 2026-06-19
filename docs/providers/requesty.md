---
title: Requesty Provider
description: "Access models from multiple AI providers through Requesty's OpenAI-compatible gateway in Go with GoAI. One API key for OpenAI, Anthropic, Google, and more, with caching, failover, and cost controls."
---

# Requesty

[Requesty](https://requesty.ai/) is a unified, OpenAI-compatible LLM gateway. Access models from multiple providers through a single API key, with caching, failover, and cost optimization.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/requesty"
```

Set the `REQUESTY_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

Models use a `provider/model` prefix format:

- `openai/gpt-4o-mini`
- `anthropic/claude-sonnet-4-5`
- `google/gemini-2.5-flash`
- `deepseek/deepseek-chat`

See [app.requesty.ai/router/list](https://app.requesty.ai/router/list) for the full catalog.

## Tested Models

**Unit tested** (mock HTTP server): `anthropic/claude-sonnet-4`

**Live tested** against `https://router.requesty.ai/v1`: `openai/gpt-4o-mini`, `google/gemini-2.5-flash`

## Usage

```go
model := requesty.Chat("openai/gpt-4o-mini")

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
| `WithBaseURL(url)` | `string` | Override the default `https://router.requesty.ai/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Supports image inputs (depends on the underlying model).
- Sends optional `HTTP-Referer` and `X-Title` analytics headers, as documented by Requesty.
- Usage reporting is enabled by default (`usage: {include: true}` in request body).
- Environment variable `REQUESTY_BASE_URL` can override the default endpoint (for example, the EU endpoint `https://router.eu.requesty.ai/v1`).
