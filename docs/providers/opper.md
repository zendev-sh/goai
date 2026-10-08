---
title: Opper Provider
description: "Access models from many AI providers through Opper's EU-hosted, OpenAI-compatible gateway in Go with GoAI. One API key for 700+ models, with chat, streaming and tool calling."
---

# Opper

[Opper](https://opper.ai/) is an EU-hosted AI gateway: 700+ models from 50+ providers behind one OpenAI-compatible API and one API key. Token rates are the model providers' rates with no markup.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/opper"
```

Set the `OPPER_API_KEY` environment variable, or pass `WithAPIKey()` directly. Create a key at [platform.opper.ai](https://platform.opper.ai).

## Models

Model IDs are pool names. A pool is every provider serving that model, and Opper picks the route per request:

- `claude-sonnet-4-6`
- `gpt-5.5`
- `gpt-5.4-mini`
- `gemini-3.8-flash`
- `deepseek-v4-pro`

A `provider/model` ID such as `anthropic/claude-sonnet-4-6` pins one route. See [opper.ai/models](https://opper.ai/models) for the full catalog.

## Tested Models

**Unit tested** (mock HTTP server): `claude-sonnet-4-6`, `anthropic/claude-sonnet-4-6`

## Usage

```go
model := opper.Chat("claude-sonnet-4-6")

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
| `WithBaseURL(url)` | `string` | Override the default `https://api.opper.ai/v3/compat` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Declares text-only chat capability. Image parts are sent in the standard `image_url` format for models that accept them.
- The package provides `Chat` only. Opper also serves `/embeddings`, which `provider/compat` can reach with the same base URL.
- Environment variable `OPPER_BASE_URL` can override the default endpoint.
