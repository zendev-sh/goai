---
title: Generic OpenAI-Compatible Provider
description: "Connect to any OpenAI-compatible API with GoAI's generic compat provider. Works with LiteLLM, LocalAI, and custom LLM deployments."
---

# Generic OpenAI-Compatible

Generic provider for any OpenAI-compatible API endpoint. Use this when connecting to services like LiteLLM, LocalAI, or custom deployments that follow the OpenAI Chat Completions format.

For Ollama and vLLM specifically, see the dedicated [ollama](ollama.md) and [vllm](vllm.md) providers.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/compat"
```

## Usage

### Chat

`WithBaseURL` is required. Authentication is optional.

```go
model := compat.Chat("my-model",
    compat.WithBaseURL("https://my-api.example.com/v1"),
    compat.WithAPIKey("optional-api-key"),
)

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

### Embeddings

```go
embedModel := compat.Embedding("my-embed-model",
    compat.WithBaseURL("https://my-api.example.com/v1"),
)

result, err := goai.Embed(ctx, embedModel, "hello world")
```

Embedding provider options (pass in `EmbedParams.ProviderOptions`):

| Option | Type | Description |
|--------|------|-------------|
| `dimensions` | int | Output embedding dimensions |
| `user` | string | User identifier for tracking |

### Dynamic Authentication

```go
model := compat.Chat("my-model",
    compat.WithBaseURL("https://my-api.example.com/v1"),
    compat.WithTokenSource(myTokenSource),
)
```

### Custom HTTP Client

```go
model := compat.Chat("my-model",
    compat.WithBaseURL("https://my-api.example.com/v1"),
    compat.WithHTTPClient(&http.Client{
        Timeout: 30 * time.Second,
    }),
)
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key (optional, omits Authorization header when not set) |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source |
| `WithBaseURL(url)` | `string` | **Required.** Set the API base URL |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |
| `WithProviderID(id)` | `string` | Set a custom provider identifier |

## Notes

- Unlike other providers, `compat` has no default base URL. Calls will fail with a clear error message if `WithBaseURL` is not provided.
- When no API key or token source is configured, the Authorization header is omitted entirely. This is useful for local servers that do not require authentication.
- Max embedding batch size: 2048 values per call.
- The `compat` provider is used internally by the `ollama` and `vllm` providers.
