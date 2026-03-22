---
title: AWS Bedrock Provider
description: "Access 60+ models from 15 vendors on AWS Bedrock with GoAI. Pure Go SigV4 signing, EventStream streaming, and cross-region inference support."
---

# Bedrock

AWS Bedrock provider for GoAI - multi-model access via the native Converse API with SigV4 signing. No AWS SDK dependency.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

Set AWS credentials via environment variables, or pass them explicitly:

```go
import "github.com/zendev-sh/goai/provider/bedrock"

model := bedrock.Chat("anthropic.claude-sonnet-4-20250514-v1:0",
    bedrock.WithAccessKey("AKIA..."),
    bedrock.WithSecretKey("..."),
    bedrock.WithRegion("us-east-1"),
)
```

The provider self-resolves credentials from standard AWS environment variables when not explicitly configured.

## Models

Bedrock supports models from multiple providers. Use the Bedrock model ID format.

| Provider | Model ID Example | Notes |
|----------|-----------------|-------|
| Anthropic | `anthropic.claude-sonnet-4-20250514-v1:0` | Claude models |
| Amazon | `amazon.nova-pro-v1:0` | Nova models |
| Meta | `meta.llama4-scout-17b-instruct-v1:0` | Llama models |
| Mistral | `mistral.mistral-large-2407-v1:0` | Mistral models |
| DeepSeek | `deepseek.deepseek-v3-0324-v1:0` | DeepSeek models |
| AI21 | `ai21.jamba-1-5-mini-v1:0` | Jamba models |
| Cohere | `cohere.command-r-plus-v1:0` | Command models |
| Others | Various | Writer, ZAI, NVIDIA, Qwen, etc. |

### Cross-Region Inference

Bedrock supports cross-region inference profiles. Prefix the model ID with a geographic identifier:

```go
// US cross-region inference
model := bedrock.Chat("us.anthropic.claude-sonnet-4-6")

// EU cross-region inference
model := bedrock.Chat("eu.anthropic.claude-sonnet-4-6")

// Global cross-region inference
model := bedrock.Chat("global.anthropic.claude-sonnet-4-6")
```

The provider auto-detects the geographic prefix and routes to the correct regional endpoint. If a bare model ID fails, the provider automatically retries with a `us.` prefix.

## Tested Models

E2E tested with real AWS credentials. Last run: 2026-03-15 -- 61 models tested across 15 vendors.

| Vendor | Models | Generate | Stream | Status |
|--------|--------|----------|--------|--------|
| Meta (11) | llama4-scout, llama4-maverick, llama3-3-70b, llama3-2-90b, llama3-2-11b, llama3-2-3b, llama3-2-1b, llama3-1-70b, llama3-1-8b, llama3-70b, llama3-8b | PASS | PASS | Stable |
| Anthropic (10) | claude-sonnet-4-6, claude-sonnet-4-5, claude-sonnet-4, claude-opus-4-6, claude-opus-4-5, claude-opus-4-1, claude-haiku-4-5, claude-3-5-sonnet, claude-3-5-haiku, claude-3-haiku | PASS | PASS | Stable |
| Mistral (7) | mistral-large, mixtral-8x7b, mistral-7b, ministral-3-14b, ministral-3-8b, voxtral-mini, voxtral-small | PASS | PASS | Stable |
| Amazon (5) | nova-micro, nova-lite, nova-pro, nova-premier, nova-2-lite | PASS | PASS | Stable |
| Qwen (5) | qwen3-32b, qwen3-235b, qwen3-coder-30b, qwen3-coder-480b, qwen3-next-80b | PASS | PASS | Stable |
| OpenAI (4) | gpt-oss-120b, gpt-oss-20b, gpt-oss-safeguard-120b, gpt-oss-safeguard-20b | PASS | PASS | Stable |
| Google Gemma (3) | gemma-3-4b, gemma-3-12b, gemma-3-27b | PASS | PASS | Stable |
| AI21 (2) | jamba-1-5-mini, jamba-1-5-large | PASS | PASS | Stable |
| Cohere (2) | command-r-plus, command-r | PASS | PASS | Stable |
| DeepSeek (2) | deepseek-v3, deepseek-r1 | PASS | PASS | Stable |
| MiniMax (2) | minimax-m2, minimax-m2.1 | PASS | PASS | Stable |
| Moonshot (2) | kimi-k2-thinking, kimi-k2.5 | PASS | PASS | Stable |
| NVIDIA (2) | nemotron-nano-12b, nemotron-nano-9b | PASS | PASS | Stable |
| Writer (2) | palmyra-x4, palmyra-x5 | PASS | PASS | Stable |
| ZhipuAI (2) | glm-4.7, glm-4.7-flash | PASS | PASS | Stable |

## Usage

### Chat

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/bedrock"
)

func main() {
    model := bedrock.Chat("anthropic.claude-sonnet-4-20250514-v1:0")
    result, err := goai.GenerateText(context.Background(), model,
        goai.WithPrompt("Explain Go interfaces in one paragraph."),
    )
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Text)
}
```

### Streaming

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider"
    "github.com/zendev-sh/goai/provider/bedrock"
)

model := bedrock.Chat("anthropic.claude-sonnet-4-20250514-v1:0")
stream, err := goai.StreamText(context.Background(), model,
    goai.WithPrompt("Write a haiku about Go."),
)
if err != nil {
    panic(err)
}
for chunk := range stream.Stream() {
    if chunk.Type == provider.ChunkText {
        fmt.Print(chunk.Text)
    }
}
```

### Extended Thinking (Anthropic Models)

```go
model := bedrock.Chat("anthropic.claude-sonnet-4-20250514-v1:0",
    bedrock.WithReasoningConfig(bedrock.ReasoningConfig{
        Type:         bedrock.ReasoningEnabled,
        BudgetTokens: 8192,
    }),
)
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Solve this step by step: What is 127 * 43?"),
)
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithRegion(r)` | `string` | AWS region. Falls back to `AWS_REGION`, `AWS_DEFAULT_REGION`, then `us-east-1`. |
| `WithAccessKey(k)` | `string` | AWS access key ID. Falls back to `AWS_ACCESS_KEY_ID`. |
| `WithSecretKey(k)` | `string` | AWS secret access key. Falls back to `AWS_SECRET_ACCESS_KEY`. |
| `WithSessionToken(t)` | `string` | AWS session token for temporary credentials. Falls back to `AWS_SESSION_TOKEN`. |
| `WithBearerToken(t)` | `string` | Bearer token auth, bypasses SigV4. Falls back to `AWS_BEARER_TOKEN_BEDROCK`. |
| `WithBaseURL(url)` | `string` | Override endpoint URL. Falls back to `AWS_BEDROCK_BASE_URL`. |
| `WithHeaders(h)` | `map[string]string` | Additional HTTP headers. |
| `WithHTTPClient(c)` | `*http.Client` | Custom HTTP client. |
| `WithReasoningConfig(cfg)` | `ReasoningConfig` | Extended thinking configuration. |
| `WithAdditionalModelRequestFields(f)` | `map[string]any` | Extra fields merged into the request body. |
| `WithAnthropicBeta(betas)` | `[]string` | Anthropic beta features to enable. |

### ReasoningConfig

```go
type ReasoningConfig struct {
    Type               ReasoningType   // "enabled", "disabled", "adaptive"
    BudgetTokens       int             // Token budget (for Type "enabled")
    MaxReasoningEffort ReasoningEffort // "low", "medium", "high", "max"
}
```

For Anthropic models, `Type: "enabled"` maps to `thinking: {type: "enabled", budget_tokens: N}`. For non-Anthropic models, it maps to a `reasoningConfig` field.

ReasoningConfig can also be passed per-request via ProviderOptions:

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("..."),
    goai.WithProviderOptions(map[string]any{
        "reasoningConfig": map[string]any{
            "type":               "enabled",
            "budgetTokens":       4096,
            "maxReasoningEffort": "high",
        },
    }),
)
```

## Notes

- For direct Anthropic API access without AWS, see the [Anthropic provider](anthropic.md).
- **SigV4 signing**: Pure Go implementation with no AWS SDK dependency. Signs requests using HMAC-SHA256 with the standard `AWS4-HMAC-SHA256` algorithm.
- **EventStream protocol**: Streaming uses AWS EventStream binary framing (not SSE). The decoder validates CRC32 checksums on both the prelude and the full message.
- **Auto-fallback chain**: The provider automatically retries on several error conditions:
  1. Bare model ID fails with "invalid model identifier" or "on-demand throughput isn't supported" - retries with `us.` prefix
  2. `us.` prefix also fails - retries with the bare model ID from `us-east-1`
  3. `maxTokens` exceeds model limit - retries without maxTokens
  4. Tool calling not supported in streaming mode - retries without tools
- **Anthropic tools on Bedrock**: Bedrock's Converse API converts Anthropic server-side tools (web_search, web_fetch, code_execution) into standard function-call tools. The model generates tool_call requests, but these are NOT executed server-side. Provide Execute handlers to run them. Computer, bash, and text editor tools behave the same on both APIs - they always require client-side execution.
- **Prompt caching**: When enabled, a `cachePoint` block is appended to the system prompt.
- **Document handling**: PDF and other file attachments are sent as Bedrock document blocks. Filenames are sanitized to match AWS requirements (alphanumeric, hyphens, parentheses, brackets only; max 200 chars).
- **Thinking + maxTokens**: When thinking is enabled for Anthropic models, the provider automatically adds `budgetTokens` to `maxTokens` (Bedrock requires `maxTokens >= budget_tokens`) and removes `temperature`, `topP`, and `topK` from the inference config.
