---
title: Standard Compute Provider
description: "Use routed or pinned models through the Standard Compute OpenAI-compatible gateway in Go with GoAI, with chat, streaming and tool calling."
---

# Standard Compute

[Standard Compute](https://standardcompute.com) is an OpenAI-compatible LLM gateway. One API key gives access to models from several labs, and the routed `StandardCompute` model sends each request to the cheapest enabled model that can handle it, with automatic fallback.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/standardcompute"
```

Set the `STANDARDCOMPUTE_API_KEY` environment variable, or pass `WithAPIKey()` directly. Create a key at [standardcompute.com](https://standardcompute.com).

## Models

- `StandardCompute`: routed. Each request goes to the cheapest enabled model that can handle it, with automatic fallback.

Pinned model IDs use a `lab/model` form, for example:

- `anthropic/claude-sonnet-5.5`
- `openai/gpt-5.6-terra`
- `deepseek/deepseek-v4-flash`
- `z-ai/glm-5.3`
- `moonshotai/kimi-k3`

`GET https://api.stdcmpt.com/v1/models` lists the full catalog and needs no API key.

## Tested Models

**Unit tested** (mock HTTP server): `StandardCompute`

## Usage

```go
model := standardcompute.Chat("StandardCompute")

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
| `WithBaseURL(url)` | `string` | Override the default `https://api.stdcmpt.com/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Declares text-only chat capability with tool calling. Image parts are sent in the standard `image_url` format for models that accept them.
- Streaming requests set `stream_options.include_usage` so the final chunk carries token usage.
- The package provides `Chat` only.
- Environment variable `STANDARDCOMPUTE_BASE_URL` can override the default endpoint.
