---
title: Groq Provider
description: "Use Groq's fast LPU inference in Go with GoAI. Run Llama, Mixtral, and Gemma models with low-latency via the OpenAI-compatible API."
---

# Groq

Groq inference provider using the OpenAI-compatible Chat Completions API. Known for fast inference on custom LPU hardware.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/groq"
```

Set the `GROQ_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

- `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`
- `mixtral-8x7b-32768`
- `gemma2-9b-it`

## Tested Models

**E2E tested** (real API calls, 2026-03-15): `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`

**Unit tested** (mock HTTP server): `llama-3.3-70b-versatile`

## Usage

```go
model := groq.Chat("llama-3.3-70b-versatile")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Explain LPUs"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

## Provider Tools

### BrowserSearch

Groq provides a browser search tool for interactive web search. Supported on `openai/gpt-oss-20b` and `openai/gpt-oss-120b` models.

```go
def := groq.Tools.BrowserSearch()

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Find the latest Go release notes"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source |
| `WithBaseURL(url)` | `string` | Override the default `https://api.groq.com/openai/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Environment variable `GROQ_BASE_URL` can override the default endpoint.
