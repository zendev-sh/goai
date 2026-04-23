---
title: NVIDIA NIM Provider
description: "Use NVIDIA NIM (NVIDIA Inference Manager) in Go with GoAI. Access 100+ models including Llama, Nemotron, and more via the OpenAI-compatible API."
---

# NVIDIA NIM

[NVIDIA NIM](https://www.nvidia.com/en-us/ai/) provides optimized inference APIs for leading AI models. GoAI supports the OpenAI-compatible Chat Completions and Embeddings APIs.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/nvidia"
```

Set the `NVIDIA_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

NVIDIA NIM offers 100+ models. Some popular ones:

**Chat:**
- `meta/llama-3.3-70b-instruct`
- `nvidia/llama-3.1-nemotron-70b-instruct`
- `nvidia/nemotron-4-340b-instruct`
- `nvidia/nemotron-3-super-120b-a12b`
- `mistralai/mistral-large`
- `deepseek-ai/deepseek-v3`

**Embeddings:**
- `nvidia/embed-qa-4`
- `nvidia/nv-embed-v1`
- `nvidia/llama-nemotron-embed-1b-v2`

[View all models →](https://build.nvidia.com/explore)

## Tested Models

**E2E tested** (real API calls, 2026-04-23): `meta/llama-3.3-70b-instruct`

**Unit tested** (mock HTTP server): `meta/llama-3.3-70b-instruct`, `nvidia/nv-embed-qa-1`

## Usage

### Chat

```go
model := nvidia.Chat("meta/llama-3.3-70b-instruct")

result, err := goai.GenerateText(ctx, model,
    goai.WithSystem("You are a helpful assistant."),
    goai.WithPrompt("What is the capital of France?"),
)
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

### Streaming

```go
stream, err := goai.StreamText(ctx, model,
    goai.WithPrompt("Count from 1 to 5."),
)
for text := range stream.TextStream() {
    fmt.Print(text)
}
```

### Embeddings

```go
model := nvidia.Embedding("nvidia/embed-qa-4")

result, err := goai.Embed(ctx, model, "Hello world")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Dimensions: %d\n", len(result.Embedding))
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source |
| `WithBaseURL(url)` | `string` | Override the default `https://integrate.api.nvidia.com/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Environment variable `NVIDIA_BASE_URL` can override the default endpoint.
- NVIDIA NIM supports self-hosted deployments, making it suitable for enterprise use cases.
- All models are accessed via the OpenAI-compatible API format.