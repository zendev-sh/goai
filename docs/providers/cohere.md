---
title: Cohere Provider
description: "Use Cohere Command and Embed models in Go with GoAI. Native Chat v2 API integration with reasoning, citations, and embedding support."
---

# Cohere

Cohere provider using the native Chat v2 and Embed APIs. This is not an OpenAI-compatible provider - it uses Cohere's own wire format.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/cohere"
```

Set the `COHERE_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

**Chat**: `command-r-plus`, `command-r`, `command-a-reasoning`

**Embeddings**: `embed-v4.0`, `embed-english-v3.0`, `embed-multilingual-v3.0`

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `command-r-plus`, `command-a-reasoning`, `embed-v4.0`

## Usage

### Chat

```go
model := cohere.Chat("command-r-plus")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Explain RAG"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

### Reasoning Models

The `command-a-reasoning` model supports thinking/reasoning via provider options:

```go
model := cohere.Chat("command-a-reasoning")

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Solve this step by step: what is 127 * 53?"),
    goai.WithProviderOptions(map[string]any{
        "thinking": map[string]any{
            "type":        "enabled",
            "tokenBudget": 4096,
        },
    }),
)
```

When streaming, reasoning chunks arrive as `provider.ChunkReasoning` type. In non-streaming mode (`GenerateText`), reasoning content is not surfaced in `TextResult`; use streaming to access it.

### Embeddings

```go
embedModel := cohere.Embedding("embed-v4.0")

result, err := goai.Embed(ctx, embedModel, "hello world")
```

Embedding provider options (pass in `EmbedParams.ProviderOptions`):

| Option | Type | Description |
|--------|------|-------------|
| `inputType` | string | `search_document` (default), `search_query`, `classification`, `clustering` |
| `truncate` | string | `NONE`, `START`, `END` |

### Citations

Cohere models can return citations with their responses. These are available in the result's `Sources` field:

```go
for _, source := range result.Sources {
    fmt.Printf("Citation: %s (chars %d-%d)\n", source.Title, source.StartIndex, source.EndIndex)
}
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source |
| `WithBaseURL(url)` | `string` | Override the default `https://api.cohere.com/v2` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Uses Cohere's native API format, not OpenAI-compatible. Request/response mapping is handled internally.
- Max embedding batch size: 96 values per call.
- Streaming uses Cohere-specific SSE event types (`content-delta`, `tool-call-start`, `tool-call-delta`, `tool-call-end`, `message-end`, `citation-start`).
- Environment variable `COHERE_BASE_URL` can override the default endpoint.
