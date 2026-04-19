---
title: DeepSeek Provider
description: "Use DeepSeek-V3 and DeepSeek-R1 reasoning models in Go with GoAI via the OpenAI-compatible Chat Completions API."
---

# DeepSeek

[DeepSeek](https://www.deepseek.com/) provider using the OpenAI-compatible Chat Completions API.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/deepseek"
```

Set the `DEEPSEEK_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

- `deepseek-chat` (DeepSeek-V3)
- `deepseek-reasoner` (DeepSeek-R1)

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `deepseek-chat`

## Usage

```go
model := deepseek.Chat("deepseek-chat")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Explain attention mechanisms"))
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
| `WithBaseURL(url)` | `string` | Override the default `https://api.deepseek.com` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Supports reasoning capability (DeepSeek-R1 family).
- Environment variable `DEEPSEEK_BASE_URL` can override the default endpoint.
