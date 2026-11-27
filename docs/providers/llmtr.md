---
title: LLMTR Provider
description: "Access models from several labs through the LLMTR OpenAI-compatible gateway in Go with GoAI, with chat, streaming and tool calling."
---

# LLMTR

[LLMTR](https://llmtr.com/) is a prepaid OpenAI-compatible LLM gateway. One API key gives access to models from several labs.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/llmtr"
```

Set the `LLMTR_API_KEY` environment variable, or pass `WithAPIKey()` directly. Create a key at [llmtr.com/dashboard/keys](https://llmtr.com/dashboard/keys).

## Models

Models use a `vendor/model` prefix format:

- `qwen/qwen3.8-27b-free`

`GET https://llmtr.com/v1/models` lists the full catalog and needs no API key.

## Tested Models

**Unit tested** (mock HTTP server): `qwen/qwen3.8-27b-free`

## Usage

```go
model := llmtr.Chat("qwen/qwen3.8-27b-free")

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
| `WithBaseURL(url)` | `string` | Override the default `https://llmtr.com/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Declares text-only chat capability with tool calling. Image parts are sent in the standard `image_url` format for models that accept them.
- Streaming requests set `stream_options.include_usage` so the final chunk carries token usage.
- The package provides `Chat` only.
- Environment variable `LLMTR_BASE_URL` can override the default endpoint.
