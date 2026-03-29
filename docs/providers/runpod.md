---
title: RunPod Provider
description: "Use RunPod serverless vLLM endpoints in Go with GoAI. Run any model on RunPod via the OpenAI-compatible Chat Completions API."
---

# RunPod

RunPod serverless vLLM inference provider using the OpenAI-compatible Chat Completions API.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/runpod"
```

Set the `RUNPOD_API_KEY` environment variable, or pass `WithAPIKey()` directly.

## Models

Model IDs depend on your RunPod endpoint configuration. Common examples:

- `meta-llama/Llama-3.3-70B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-72B-Instruct`

## Usage

RunPod requires **two arguments**: an endpoint ID and a model ID. This differs from most other providers.

```go
model := runpod.Chat("your-endpoint-id", "meta-llama/Llama-3.3-70B-Instruct")

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
| `WithBaseURL(url)` | `string` | Override the default `https://api.runpod.ai/v2/{endpointID}/openai/v1` endpoint |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- Environment variable `RUNPOD_BASE_URL` can override the default endpoint.
- The default base URL is constructed from the endpoint ID: `https://api.runpod.ai/v2/{endpointID}/openai/v1`.
- Chat only — no embedding or image generation support.
