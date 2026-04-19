---
title: Together AI Provider
description: "Use Together AI inference in Go with GoAI. Run open-source Llama, Mixtral, and Qwen models via the OpenAI-compatible API."
---

# Together AI

[Together AI](https://www.together.ai/) inference provider using the OpenAI-compatible Chat Completions API.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/together"
```

Set `TOGETHER_AI_API_KEY` (or fallback `TOGETHER_API_KEY`) in the environment, or pass `WithAPIKey()` directly.

## Models

- `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `Qwen/Qwen2.5-72B-Instruct-Turbo`

Model IDs use the `org/model` prefix format.

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `meta-llama/Llama-3.3-70B-Instruct-Turbo`

## Usage

```go
model := together.Chat("meta-llama/Llama-3.3-70B-Instruct-Turbo")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

## Options

| Option                | Type                   | Description                                                 |
| --------------------- | ---------------------- | ----------------------------------------------------------- |
| `WithAPIKey(key)`     | `string`               | Set a static API key                                        |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source                                  |
| `WithBaseURL(url)`    | `string`               | Override the default `https://api.together.xyz/v1` endpoint |
| `WithHeaders(h)`      | `map[string]string`    | Set additional HTTP headers                                 |
| `WithHTTPClient(c)`   | `*http.Client`         | Set a custom `*http.Client`                                 |

## Notes

- Environment variable `TOGETHER_AI_BASE_URL` can override the default endpoint.
