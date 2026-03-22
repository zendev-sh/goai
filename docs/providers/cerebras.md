---
title: Cerebras Provider
description: "Use Cerebras wafer-scale inference in Go with GoAI. Run Llama models with ultra-fast inference via the OpenAI-compatible API."
---

# Cerebras

Cerebras inference provider using the OpenAI-compatible Chat Completions API. Known for fast inference on Cerebras wafer-scale hardware.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/cerebras"
```

Set the `CEREBRAS_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

- `llama3.1-8b`, `llama3.1-70b`
- `llama-3.3-70b`

## Tested Models

**E2E tested** (real API calls, 2026-03-15): `llama3.1-8b`

**Unit tested** (mock HTTP server): `llama-3.3-70b`

## Usage

```go
model := cerebras.Chat("llama3.1-8b")

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
| `WithBaseURL(url)` | `string` | Override the default `https://api.cerebras.ai/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Environment variable `CEREBRAS_BASE_URL` can override the default endpoint.
