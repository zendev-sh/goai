---
title: cheapaimodel Provider
description: "Use DeepSeek, GLM, Kimi and Grok models through the cheapaimodel OpenAI-compatible gateway in Go with GoAI, with chat, streaming, tool calling and structured output."
---

# cheapaimodel

[cheapaimodel](https://cheapaimodel.ai/) is an OpenAI-compatible LLM gateway. One API key gives access to models from DeepSeek, Z.ai (GLM), Moonshot (Kimi) and xAI (Grok), billed per token from a prepaid USD balance.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/cheapaimodel"
```

Set the `CHEAPAIMODEL_API_KEY` environment variable, or pass `WithAPIKey()` directly. Create a key at [console.cheapaimodel.ai/keys](https://console.cheapaimodel.ai/keys).

## Models

Model IDs are bare names:

| Model ID | Context | Image input |
|----------|---------|-------------|
| `deepseek-v4-1-flash` | 1M | Yes |
| `glm-5.3` | 1M | No |
| `glm-5.3-flash` | 1M | Yes |
| `grok-4.7` | 500K | Yes |
| `kimi-k3` | 1M | Yes |

`GET https://cheapaimodel.ai/api/models` lists the current catalog with prices and needs no API key. See [cheapaimodel.ai/models](https://cheapaimodel.ai/models) and [cheapaimodel.ai/docs](https://cheapaimodel.ai/docs) for details.

## Tested Models

**Unit tested** (mock HTTP server): `deepseek-v4-1-flash`, `glm-5.3`, `kimi-k3`

**Live tested** (real API, 2026-11-03): all five models above with `GenerateText`, `StreamText` and a tool loop (`GenerateText` and `StreamText` with `MaxSteps`), image input on the four models that accept it, and `GenerateObject` (see the `glm-5.3` note below).

## Usage

```go
model := cheapaimodel.Chat("deepseek-v4-1-flash")

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
| `WithBaseURL(url)` | `string` | Override the default `https://api.cheapaimodel.ai/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Declares text-only chat capability with tool calling. Image parts are sent in the standard `image_url` format for models that accept them.
- The models can return reasoning text, surfaced as `result.Reasoning`. Reasoning from earlier assistant turns is sent back as `reasoning_content`, and streaming requests set `stream_options.include_usage` so the final chunk carries token usage.
- `kimi-k3` only accepts `temperature` 1 or 0.6 and `top_p` 0.95. Other values return HTTP 400 `invalid_parameter_error`; leave them unset or use those values.
- `glm-5.3` often answers in plain text instead of following the JSON schema that `GenerateObject` sends (1 of 5 attempts parsed in testing; `glm-5.3-flash` 4 of 5, the other three models 5 of 5).
- `result.Response.Model` can be the upstream model name (for example `deepseek-v4.1-flash-1017`) rather than the ID you passed.
- The package provides `Chat` only. cheapaimodel has no embeddings models.
- Environment variable `CHEAPAIMODEL_BASE_URL` can override the default endpoint.
