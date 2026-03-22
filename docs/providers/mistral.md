---
title: Mistral Provider
description: "Use Mistral AI models in Go with GoAI. Supports Mistral Large, Codestral, and Magistral via the OpenAI-compatible Chat Completions API."
---

# Mistral

Mistral AI provider using the OpenAI-compatible Chat Completions API.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/mistral"
```

Set the `MISTRAL_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

- `mistral-large-latest`, `mistral-small-latest`
- `magistral-medium-latest`
- `codestral-latest`, `devstral-small-2507`

## Tested Models

**E2E tested** (real API calls, 2026-03-15): `mistral-small-latest`, `mistral-large-latest`, `devstral-small-2507`, `codestral-latest`, `magistral-medium-latest`

**Unit tested** (mock HTTP server): `mistral-large-latest`

## Usage

```go
model := mistral.Chat("mistral-large-latest")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Explain transformers"))
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
| `WithBaseURL(url)` | `string` | Override the default `https://api.mistral.ai/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Mistral requires 9-character alphanumeric tool call IDs. If tool calls fail, ensure tool call IDs conform to this format.
- Environment variable `MISTRAL_BASE_URL` can override the default endpoint.
