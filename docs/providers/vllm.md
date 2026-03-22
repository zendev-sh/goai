---
title: vLLM Provider
description: "Connect to self-hosted vLLM inference servers with GoAI. Supports chat and embeddings via the OpenAI-compatible API with optional auth."
---

# vLLM

vLLM provider for local or self-hosted model inference. vLLM exposes an OpenAI-compatible API and authentication is optional.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/vllm"
```

Start a vLLM server:

```bash
vllm serve meta-llama/Llama-3-8b --port 8000
```

## Models

Any model served by the vLLM instance, using the HuggingFace model ID format:

- `meta-llama/Llama-3-8b`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `meta-llama/Llama-3-8b`

## Usage

### Chat

```go
model := vllm.Chat("meta-llama/Llama-3-8b")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

### Embeddings

```go
embedModel := vllm.Embedding("BAAI/bge-small-en-v1.5")

result, err := goai.Embed(ctx, embedModel, "hello world")
```

### Authenticated vLLM Server

```go
model := vllm.Chat("meta-llama/Llama-3-8b",
    vllm.WithAPIKey("your-api-key"),
    vllm.WithBaseURL("https://my-vllm-server.example.com/v1"),
)
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key (optional, omits Authorization header when not set) |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source |
| `WithBaseURL(url)` | `string` | Override the default `http://localhost:8000/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- This is a convenience wrapper around the generic [`compat`](compat.md) provider.
- Default endpoint: `http://localhost:8000/v1`.
- Authentication is optional. When no API key or token source is configured, the Authorization header is omitted entirely.
