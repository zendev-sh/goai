---
title: DeepInfra Provider
description: "Use DeepInfra serverless inference in Go with GoAI. Deploy and run Llama, Mixtral, and Qwen models via the OpenAI-compatible API."
---

# DeepInfra

[DeepInfra](https://deepinfra.com/) inference provider using the OpenAI-compatible Chat Completions API.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/deepinfra"
```

Set the `DEEPINFRA_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

- `meta-llama/Llama-3.3-70B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `Qwen/Qwen2.5-72B-Instruct`

Model IDs use the `org/model` prefix format.

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `meta-llama/Llama-3.3-70B-Instruct`

## Usage

```go
model := deepinfra.Chat("meta-llama/Llama-3.3-70B-Instruct")

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
| `WithBaseURL(url)` | `string` | Override the default `https://api.deepinfra.com/v1/openai` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Environment variable `DEEPINFRA_BASE_URL` can override the default endpoint.
