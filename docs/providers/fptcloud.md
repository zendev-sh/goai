---
title: FPT Smart Cloud Provider
description: "Use FPT AI Marketplace in Go with GoAI. Run Qwen, Llama, GLM, gpt-oss, gemma, and embedding models via the OpenAI-compatible API. Global and Japan regions supported."
---

# FPT Smart Cloud

[FPT AI Marketplace](https://marketplace.fptcloud.com/) ([FPT Smart Cloud](https://fptsmartcloud.com/)) provider using the OpenAI-compatible Chat Completions and Embeddings endpoints. Runs models across Global and Japan regions.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/fptcloud"
```

Environment variables (or pass the equivalent `With*` options):

| Env Var         | Description                                                                       |
| --------------- | --------------------------------------------------------------------------------- |
| `FPT_API_KEY`   | API key from [marketplace.fptcloud.com](https://marketplace.fptcloud.com/)        |
| `FPT_REGION`    | Optional. `global` (default) or `jp`                                              |
| `FPT_BASE_URL`  | Optional. Full base URL override (takes precedence over `FPT_REGION`)             |

## Regions

| Region   | Base URL                        |
| -------- | ------------------------------- |
| `global` | `https://mkp-api.fptcloud.com/v1` |
| `jp`     | `https://mkp-api.fptcloud.jp/v1`  |

Each region has its own model catalog — some models are only available in specific regions. Select the region that matches where you created your API key.

## Models

Text generation models (availability varies by region):

- `Qwen3-32B`
- `Llama-3.3-70B-Instruct`
- `gpt-oss-120b`, `gpt-oss-20b`
- `GLM-4.7`
- `gemma-3-27b-it`
- `Nemotron-3-Super-120B-A12B`
- `DeepSeek-V3.2` (varies by region)

Embedding / retrieval models:

- `bge-m3`
- `bge-reranker-v2-m3`
- `gte-multilingual-base`
- `multilingual-e5-large`

Vision / document models:

- `Qwen2.5-VL-7B-Instruct`
- `FCI-Document-Parsing-V1.0`

List available models in your region:

```bash
curl -H "Authorization: Bearer $FPT_API_KEY" https://mkp-api.fptcloud.jp/v1/models | jq '.data[].id'
```

## Tested Models

**Unit tested** (mock HTTP server): `Qwen3-32B`

**E2E verified** (real API, generate + stream, JP region): `Qwen3-32B`

## Usage

```go
model := fptcloud.Chat("Qwen3-32B")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

Japan region:

```go
model := fptcloud.Chat("Qwen3-32B", fptcloud.WithRegion("jp"))
```

Embeddings:

```go
emb := fptcloud.Embedding("bge-m3")
result, err := goai.Embed(ctx, emb, []string{"hello", "world"})
```

## Options

| Option                | Type                    | Description                                                                  |
| --------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| `WithAPIKey(key)`     | `string`                | Set a static API key                                                         |
| `WithTokenSource(ts)` | `provider.TokenSource`  | Set a dynamic token source                                                   |
| `WithRegion(region)`  | `string`                | `"global"` (default) or `"jp"`. Unknown values fall back to `"global"`       |
| `WithBaseURL(url)`    | `string`                | Full base URL override (takes precedence over `WithRegion`)                  |
| `WithHeaders(h)`      | `map[string]string`     | Set additional HTTP headers                                                  |
| `WithHTTPClient(c)`   | `*http.Client`          | Set a custom `*http.Client`                                                  |

## Notes

- Reasoning models like `Qwen3-32B` emit `<think>...</think>` blocks before the final answer; the full output (including the thinking section) is returned in `result.Text`.
- API key scopes may restrict which models you can call — create keys scoped to the models you need in the FPT Marketplace console.
- Both regions expose an OpenAI-compatible `/models` endpoint for model discovery.
