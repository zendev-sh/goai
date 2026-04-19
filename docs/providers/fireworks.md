---
title: Fireworks AI Provider
description: "Use Fireworks AI inference in Go with GoAI. Run Llama, Mixtral, and Qwen models via the OpenAI-compatible Chat Completions API."
---

# Fireworks AI

[Fireworks AI](https://fireworks.ai/) inference provider using the OpenAI-compatible Chat Completions API.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/fireworks"
```

Set the `FIREWORKS_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

- `accounts/fireworks/models/llama-v3p3-70b-instruct`
- `accounts/fireworks/models/mixtral-8x7b-instruct`
- `accounts/fireworks/models/qwen2p5-72b-instruct`

Model IDs use the full `accounts/fireworks/models/` prefix format.

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `accounts/fireworks/models/llama-v3p3-70b-instruct`

## Usage

```go
model := fireworks.Chat("accounts/fireworks/models/llama-v3p3-70b-instruct")

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
| `WithBaseURL(url)` | `string` | Override the default `https://api.fireworks.ai/inference/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Environment variable `FIREWORKS_BASE_URL` can override the default endpoint.
