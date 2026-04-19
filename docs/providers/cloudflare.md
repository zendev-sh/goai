---
title: Cloudflare Workers AI Provider
description: "Use Cloudflare Workers AI in Go with GoAI. Run Llama, Qwen, OpenAI gpt-oss, and BGE embeddings on Cloudflare's edge network via the OpenAI-compatible API."
---

# Cloudflare Workers AI

[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) provider using the [OpenAI-compatible](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/) Chat Completions and Embeddings endpoints. Runs inference on Cloudflare's global edge network.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/cloudflare"
```

Required environment variables (or pass `WithAPIKey` / `WithAccountID`):

| Env Var                 | Description                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `CLOUDFLARE_API_TOKEN`  | API token with `Workers AI` permission. Create at dash.cloudflare.com/profile/api-tokens |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID (used to build the API URL)                                        |
| `CLOUDFLARE_BASE_URL`   | Optional. Full base URL override (for AI Gateway or proxy)                               |

## Models

Popular text generation models:

- `@cf/meta/llama-3.1-8b-instruct`
- `@cf/meta/llama-3.3-70b-instruct-fp8-fast`
- `@cf/openai/gpt-oss-120b`
- `@cf/openai/gpt-oss-20b`
- `@cf/qwen/qwen2.5-coder-32b-instruct`
- `@cf/mistralai/mistral-small-3.1-24b-instruct`
- `@cf/deepseek-ai/deepseek-r1-distill-qwen-32b`

Embedding models:

- `@cf/baai/bge-base-en-v1.5`
- `@cf/baai/bge-large-en-v1.5`
- `@cf/baai/bge-m3`

Browse the full model catalog at [developers.cloudflare.com/workers-ai/models](https://developers.cloudflare.com/workers-ai/models/).

## Tested Models

**Unit tested** (mock HTTP server): `@cf/meta/llama-3.1-8b-instruct`, `@cf/baai/bge-base-en-v1.5`

**E2E verified** (real API, generate + stream): `@cf/meta/llama-3.1-8b-instruct`

## Usage

```go
model := cloudflare.Chat("@cf/meta/llama-3.1-8b-instruct")

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

Embeddings:

```go
emb := cloudflare.Embedding("@cf/baai/bge-base-en-v1.5")
result, err := goai.Embed(ctx, emb, []string{"hello", "world"})
```

## Options

| Option                | Type                    | Description                                                                                     |
| --------------------- | ----------------------- | ----------------------------------------------------------------------------------------------- |
| `WithAPIKey(key)`     | `string`                | Set a static API token                                                                          |
| `WithTokenSource(ts)` | `provider.TokenSource`  | Set a dynamic token source                                                                      |
| `WithAccountID(id)`   | `string`                | Cloudflare account ID. Used to build `…/accounts/{id}/ai/v1` URL                                |
| `WithBaseURL(url)`    | `string`                | Full base URL override (bypasses account-ID-derived URL). Useful for AI Gateway                 |
| `WithHeaders(h)`      | `map[string]string`     | Set additional HTTP headers                                                                     |
| `WithHTTPClient(c)`   | `*http.Client`          | Set a custom `*http.Client`                                                                     |

## Free Tier

Cloudflare Workers AI includes **10,000 Neurons/day free** (resets at 00:00 UTC). Beyond that, the Workers Paid plan charges $0.011 per 1,000 Neurons. See [Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/).

## Notes

- The provider automatically constructs the URL `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions` (and `/embeddings`) from `CLOUDFLARE_ACCOUNT_ID`.
- To route through [AI Gateway](https://developers.cloudflare.com/ai-gateway/), set `CLOUDFLARE_BASE_URL` or use `WithBaseURL`.
- Tool calling is supported by most chat models; the provider forwards tools in the OpenAI-compatible format.
