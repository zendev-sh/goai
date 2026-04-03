---
title: Azure OpenAI Provider
description: "Use Azure OpenAI and Azure AI Services with GoAI. Auto-routes GPT, Claude, and third-party models with Managed Identity and API key auth."
---

# Azure

Azure OpenAI and Azure AI Services provider for GoAI - multi-model access through Azure's managed endpoints.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

Set environment variables, or pass options explicitly:

```go
import "github.com/zendev-sh/goai/provider/azure"

model := azure.Chat("gpt-4o",
    azure.WithAPIKey("your-api-key"),
    azure.WithEndpoint("https://your-resource.openai.azure.com"),
)
```

The provider self-resolves configuration from environment variables when not explicitly set.

## Auto-Routing

Azure hosts models from multiple providers. `azure.Chat()` automatically routes based on model ID:

| Model Type                         | Routing                            | Endpoint Format                                            |
| ---------------------------------- | ---------------------------------- | ---------------------------------------------------------- |
| OpenAI (GPT, o-series, codex)      | `openai.Chat()` + URL rewrite      | `{endpoint}/openai/v1{path}?api-version={version}`         |
| Claude                             | `anthropic.Chat()`                 | `{resource}.services.ai.azure.com/anthropic`               |
| All others (DeepSeek, Llama, etc.) | `openai.Chat()` + Chat Completions | `{resource}.services.ai.azure.com/models/chat/completions` |

```go
// OpenAI model - uses Responses API via azureRoundTripper
gpt := azure.Chat("gpt-4o")

// Claude model - routes to Anthropic Messages API on Azure
claude := azure.Chat("claude-sonnet-4-6")

// DeepSeek model - routes to Azure AI Services endpoint
deepseek := azure.Chat("DeepSeek-V3.2")
```

Detection matches model ID prefixes (case-insensitive, after stripping any provider prefix like `openai/`): `gpt-`, `o` followed by a digit (o1, o3, o4, etc.), `codex`, `chatgpt`, `computer-use`.

## Models

| Model ID            | Type  | Route                      |
| ------------------- | ----- | -------------------------- |
| `gpt-4o`            | Chat  | OpenAI endpoint            |
| `gpt-4.1`           | Chat  | OpenAI endpoint            |
| `gpt-5`             | Chat  | OpenAI endpoint, reasoning |
| `gpt-5.2`           | Chat  | OpenAI endpoint, reasoning |
| `o3`                | Chat  | OpenAI endpoint, reasoning |
| `claude-opus-4-6`   | Chat  | Anthropic endpoint         |
| `claude-sonnet-4-6` | Chat  | Anthropic endpoint         |
| `DeepSeek-V3.2`     | Chat  | AI Services endpoint       |
| `Kimi-K2.5`         | Chat  | AI Services endpoint       |
| `model-router`      | Chat  | AI Services endpoint       |
| `dall-e-3`          | Image | OpenAI endpoint            |

## Tested Models

E2E tested with real Azure credentials. Last run: 2026-03-15 - 21 models PASS.

| Model                | Generate | Stream | Status |
| -------------------- | -------- | ------ | ------ |
| `claude-opus-4-6`    | PASS     | PASS   | Stable |
| `claude-sonnet-4-6`  | PASS     | PASS   | Stable |
| `DeepSeek-V3.2`      | PASS     | PASS   | Stable |
| `gpt-4.1`            | PASS     | PASS   | Stable |
| `gpt-4.1-mini`       | PASS     | PASS   | Stable |
| `gpt-5`              | PASS     | PASS   | Stable |
| `gpt-5-codex`        | PASS     | PASS   | Stable |
| `gpt-5-mini`         | PASS     | PASS   | Stable |
| `gpt-5-pro`          | PASS     | PASS   | Stable |
| `gpt-5.1`            | PASS     | PASS   | Stable |
| `gpt-5.1-codex`      | PASS     | PASS   | Stable |
| `gpt-5.1-codex-max`  | PASS     | PASS   | Stable |
| `gpt-5.1-codex-mini` | PASS     | PASS   | Stable |
| `gpt-5.2`            | PASS     | PASS   | Stable |
| `gpt-5.2-codex`      | PASS     | PASS   | Stable |
| `gpt-5.3-codex`      | PASS     | PASS   | Stable |
| `gpt-5.4`            | PASS     | PASS   | Stable |
| `gpt-5.4-pro`        | PASS     | PASS   | Stable |
| `Kimi-K2.5`          | PASS     | PASS   | Stable |
| `model-router`       | PASS     | PASS   | Stable |
| `o3`                 | PASS     | PASS   | Stable |

Unit tested models: `gpt-4o`, `gpt-5.2-chat`, `dall-e-3`, `claude-sonnet-4-20250514`.

## Usage

### Chat (OpenAI Model)

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/azure"
)

func main() {
    model := azure.Chat("gpt-4o")
    result, err := goai.GenerateText(context.Background(), model,
        goai.WithPrompt("Explain Go interfaces in one paragraph."),
    )
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Text)
}
```

### Chat (Claude Model)

```go
model := azure.Chat("claude-sonnet-4-6",
    azure.WithAPIKey("your-api-key"),
    azure.WithEndpoint("https://your-resource.openai.azure.com"),
)
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Hello from Azure!"),
)
```

Claude models on Azure use the Anthropic Messages API protocol. All Anthropic provider options (thinking, effort, etc.) work transparently.

### Image Generation

```go
import "github.com/zendev-sh/goai/provider/azure"

model := azure.Image("dall-e-3",
    azure.WithAPIKey("your-api-key"),
    azure.WithEndpoint("https://your-resource.openai.azure.com"),
)
result, err := goai.GenerateImage(ctx, model,
    goai.WithImagePrompt("A mountain landscape"),
)
```

### Managed Identity (TokenSource)

```go
model := azure.Chat("gpt-4o",
    azure.WithTokenSource(myAzureTokenSource),
    azure.WithEndpoint("https://your-resource.openai.azure.com"),
)
```

When using `WithTokenSource`, the provider sends `Authorization: Bearer {token}` instead of `api-key`.

## Options

| Option                       | Type                   | Description                                                                                           |
| ---------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `WithAPIKey(key)`            | `string`               | Azure API key. Falls back to `AZURE_OPENAI_API_KEY`, then `AZURE_API_KEY`.                            |
| `WithTokenSource(ts)`        | `provider.TokenSource` | Dynamic token (Managed Identity, AAD).                                                                |
| `WithEndpoint(url)`          | `string`               | Azure endpoint URL. Falls back to `AZURE_OPENAI_ENDPOINT`, or constructed from `AZURE_RESOURCE_NAME`. |
| `WithHeaders(h)`             | `map[string]string`    | Additional HTTP headers.                                                                              |
| `WithHTTPClient(c)`          | `*http.Client`         | Custom HTTP client.                                                                                   |
| `WithAPIVersion(v)`          | `string`               | API version query parameter. Default: `"2025-03-01-preview"`.                                         |
| `WithDeploymentBasedURLs(b)` | `bool`                 | Use legacy deployment-based URL format. Default: `false`.                                             |

### Environment Variables

| Variable                | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| `AZURE_OPENAI_API_KEY`  | Primary API key                                            |
| `AZURE_API_KEY`         | Fallback API key                                           |
| `AZURE_OPENAI_ENDPOINT` | Endpoint URL (e.g., `https://myresource.openai.azure.com`) |
| `AZURE_RESOURCE_NAME`   | Resource name (auto-constructs endpoint URL)               |

## URL Format

**Default**:

```
{endpoint}/openai/v1{path}?api-version={version}
```

**Legacy deployment-based** (enabled with `WithDeploymentBasedURLs(true)`):

```
{endpoint}/openai/deployments/{model}{path}?api-version={version}
```

OpenAI models support both Chat Completions (`/chat/completions`) and Responses API (`/responses`) paths. Non-OpenAI models are forced to use Chat Completions via an internal wrapper.

## Notes

- **Transport-based architecture**: Azure does not implement a custom language model. For OpenAI models, it delegates to [`openai.Chat()`](openai.md) with a custom `http.RoundTripper` that rewrites URLs and injects Azure auth. This means all OpenAI provider features (Responses API, provider tools, structured output) work on Azure without extra code.
- **Claude on Azure**: Claude models are routed to `{resource}.services.ai.azure.com/anthropic`, which speaks the Anthropic Messages API protocol. The provider delegates to [`anthropic.Chat()`](anthropic.md) with the Azure endpoint and auth.
- **AI Services models**: Non-OpenAI, non-Claude models (DeepSeek, Llama, Phi, Grok, Kimi, etc.) route to `{resource}.services.ai.azure.com/models/chat/completions` using OpenAI Chat Completions format. These are forced to use Chat Completions (not Responses API). The AI Services API version is hardcoded to `2024-05-01-preview` and is not affected by `WithAPIVersion`.
- **Auth**: API key auth uses the `api-key` header (Azure convention). TokenSource auth uses `Authorization: Bearer` (for Managed Identity/AAD scenarios).
- **Resource name resolution**: The resource name is extracted from the endpoint URL (first segment before `.`) or from `AZURE_RESOURCE_NAME`. It is used to construct the AI Services and Anthropic endpoint URLs.
