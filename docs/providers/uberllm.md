---
title: UberLLM Provider
description: "Use OpenAI, Anthropic, Google, DeepSeek, Qwen, GLM and Kimi models through the UberLLM OpenAI-compatible API in Go with GoAI, with chat, streaming and tool calling."
---

# UberLLM

[UberLLM](https://uberllm.dev/) provides an OpenAI-compatible Chat Completions API at `https://api.uberllm.dev/v1`. One API key gives access to models from OpenAI, Anthropic, Google, DeepSeek, Qwen, GLM, Kimi and others, usually below list price. Live per-model prices: [uberllm.dev/models](https://uberllm.dev/models).

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/uberllm"
```

Set the `UBERLLM_API_KEY` environment variable, or pass `WithAPIKey()` directly. API docs: [uberllm.dev/docs/quickstart](https://uberllm.dev/docs/quickstart). Keys: [uberllm.dev/signup](https://uberllm.dev/signup).

## Models

Model IDs carry a vendor prefix:

- `deepseek/deepseek-v4.1-flash`
- `openai/gpt-5.4-mini`
- `anthropic/claude-sonnet-4.6`
- `qwen/qwen3.7-plus`
- `z-ai/glm-5.3-flash`
- `moonshotai/kimi-k3`

`GET https://api.uberllm.dev/v1/models` lists the full catalog and needs no API key.

## Tested Models

**Unit tested** (mock HTTP server): `deepseek/deepseek-v4.1-flash`, `z-ai/glm-5.3-flash`

## Usage

```go
model := uberllm.Chat("deepseek/deepseek-v4.1-flash")

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
| `WithBaseURL(url)` | `string` | Override the default `https://api.uberllm.dev/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Declares text-only chat capability with tool calling.
- Reasoning from earlier assistant turns is sent back as `reasoning_content`, and streaming requests set `stream_options.include_usage` so the final chunk carries token usage.
- The package provides `Chat` only. The catalog also lists image, video, music and speech models, which this provider does not wrap.
- Environment variable `UBERLLM_BASE_URL` can override the default endpoint.
