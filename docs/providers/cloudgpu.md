---
title: CloudGPU Provider
description: "Use DeepSeek, Kimi, GLM and MiniMax models through the CloudGPU OpenAI-compatible API in Go with GoAI, with chat, streaming and tool calling."
---

# CloudGPU

[CloudGPU](https://cloudgpu.app/) provides an OpenAI-compatible Chat Completions API at `https://cloudgpu.app/v1`. One API key gives access to models such as DeepSeek, Kimi, GLM and MiniMax.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/cloudgpu"
```

Set the `CLOUDGPU_API_KEY` environment variable, or pass `WithAPIKey()` directly. API docs and key setup: [cloudgpu.app/api/docs](https://cloudgpu.app/api/docs).

## Models

Model IDs are bare names:

- `deepseek-v4.1-flash`
- `deepseek-v4-pro`
- `kimi-k3`
- `glm-5.3-flash`
- `minimax-m3`

`GET https://cloudgpu.app/v1/models` lists the full catalog and needs no API key.

## Tested Models

**Unit tested** (mock HTTP server): `deepseek-v4.1-flash`, `glm-5.3-flash`

## Usage

```go
model := cloudgpu.Chat("deepseek-v4.1-flash")

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
| `WithBaseURL(url)` | `string` | Override the default `https://cloudgpu.app/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Declares text-only chat capability with tool calling.
- Reasoning from earlier assistant turns is sent back as `reasoning_content`, and streaming requests set `stream_options.include_usage` so the final chunk carries token usage.
- The package provides `Chat` only. The catalog also lists image, speech and transcription models, which this provider does not wrap.
- Environment variable `CLOUDGPU_BASE_URL` can override the default endpoint.
