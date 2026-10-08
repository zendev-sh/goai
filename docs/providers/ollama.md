---
title: Ollama Provider
description: "Run local LLMs with Ollama and GoAI. Use Llama, Mistral, and other open models for chat and embeddings with no API key required."
---

# Ollama

[Ollama](https://ollama.com/) provider for local model inference. This GoAI provider uses Ollama's native `/api/chat` and `/api/embed` endpoints and requires no authentication by default.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/ollama"
```

Ollama must be running locally. Install from [ollama.com](https://ollama.com) and pull a model:

```bash
ollama pull llama3
ollama serve
```

No API key is required.

## Models

Any model available in Ollama. Common examples:

- `llama3`, `llama3.2:1b`, `llama3.2:3b`
- `mistral`, `mixtral`
- `codellama`
- `nomic-embed-text` (embeddings)

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `llama3`, `nomic-embed-text`

## Usage

### Chat

```go
model := ollama.Chat("llama3")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

### Embeddings

```go
embedModel := ollama.Embedding("nomic-embed-text")

result, err := goai.Embed(ctx, embedModel, "hello world")
```

### Remote Ollama Server

```go
model := ollama.Chat("llama3",
    ollama.WithBaseURL("http://192.168.1.100:11434"),
)
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithBaseURL(url)` | `string` | Override the default `http://localhost:11434` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- This is a native Ollama implementation using the Ollama API endpoints.
- Default endpoint: `http://localhost:11434`; chat and embeddings use `/api/chat` and `/api/embed`.
- No authentication options (`WithAPIKey`, `WithTokenSource`) since Ollama does not require auth by default. For authenticated setups, use the [`compat`](compat.md) provider directly.
