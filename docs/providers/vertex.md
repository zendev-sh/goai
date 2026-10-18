---
title: Vertex AI Provider
description: "Use Google Vertex AI Gemini models in Go with GoAI. Supports ADC authentication, chat, embeddings, and Imagen image generation on GCP."
---

# Google Vertex AI

[Google Cloud Vertex AI](https://cloud.google.com/vertex-ai) provider with support for Gemini and Anthropic chat, embeddings, and image generation.

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

```go
import "github.com/zendev-sh/goai/provider/vertex"
```

### Authentication

Vertex AI supports three authentication methods, resolved in this priority order for the default transports:

1. **Explicit options** - `WithTokenSource(ts)` or `WithAPIKey(key)` passed directly take highest priority.
2. **Application Default Credentials (ADC)** - auto-detected when a GCP project is configured. Uses `gcloud` CLI credentials, service account JSON, or GCE metadata.
3. **API key from environment** - falls back to `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_GENERATIVE_AI_API_KEY` env vars. Routes requests to the Gemini API endpoint instead of Vertex AI.

Native Gemini chat (`WithNativeGemini`) is OAuth-only and rejects API keys. Use an explicit `TokenSource` or ADC with a configured project.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_VERTEX_PROJECT` | GCP project ID (also reads `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`) |
| `GOOGLE_VERTEX_LOCATION` | GCP region (also reads `GOOGLE_CLOUD_LOCATION`, defaults to `us-central1`) |
| `GOOGLE_VERTEX_BASE_URL` | Override the default/non-native endpoint |
| `GOOGLE_VERTEX_NATIVE_CHAT_BASE_URL` | Override the native Gemini chat models-collection URL |
| `GOOGLE_API_KEY` | API key for Gemini API fallback |
| `GEMINI_API_KEY` | API key for Gemini API fallback (alternative) |
| `GOOGLE_GENERATIVE_AI_API_KEY` | API key for Gemini API fallback (alternative) |

## Models

- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`
- `gemini-3-flash-preview`, `gemini-3-pro-preview`
- `text-embedding-004` (embeddings)
- `imagen-3.0-generate-002`, `imagen-4.0-fast-generate-001` (image generation)

## Tested Models

**Unit tested** (mock HTTP server, 2026-03-15): `gemini-2.5-pro`, `imagen-3.0-generate-002`, `text-embedding-004`

## Usage

### Chat

```go
model := vertex.Chat("gemini-2.5-pro",
    vertex.WithProject("my-gcp-project"),
    vertex.WithLocation("us-central1"),
)

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Explain quantum computing"))
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Text)
```

### Chat with API Key (Gemini API fallback)

When no project is configured, the provider automatically routes to the Gemini API if an API key is available:

```go
model := vertex.Chat("gemini-2.5-flash",
    vertex.WithAPIKey("your-api-key"),
)
```

### Native Gemini Chat

Use the native Vertex `generateContent` transport with OAuth/ADC:

```go
model := vertex.Chat("gemini-2.5-pro",
    vertex.WithProject("my-gcp-project"),
    vertex.WithLocation("global"),
    vertex.WithNativeGemini(),
)
```

`WithNativeChatBaseURL` optionally overrides the native models-collection URL. It requires `WithNativeGemini`; native chat rejects `WithBaseURL` and `WithAPIKey`.

### Anthropic Chat

```go
model := vertex.AnthropicChat("claude-sonnet-4-20250514",
    vertex.WithProject("my-gcp-project"),
    vertex.WithLocation("us-east5"),
)
```

`AnthropicChat` uses Vertex `rawPredict`/`streamRawPredict` with Anthropic's native Messages protocol, including prompt caching and extended thinking.

### Embeddings

```go
embedModel := vertex.Embedding("text-embedding-004",
    vertex.WithProject("my-gcp-project"),
)

result, err := goai.Embed(ctx, embedModel, "hello world")
```

Embedding provider options (pass under the `"vertex"` key in `EmbedParams.ProviderOptions`):

| Option | Type | Description |
|--------|------|-------------|
| `outputDimensionality` | int | Reduced dimension for output embedding |
| `taskType` | string | `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `QUESTION_ANSWERING`, `FACT_VERIFICATION`, `CODE_RETRIEVAL_QUERY` |
| `title` | string | Document title (only valid with `RETRIEVAL_DOCUMENT` task) |
| `autoTruncate` | bool | Truncate input text if too long (default: true) |

### Image Generation

```go
imgModel := vertex.Image("imagen-3.0-generate-002",
    vertex.WithProject("my-gcp-project"),
)

result, err := goai.GenerateImage(ctx, imgModel,
    goai.WithImagePrompt("A sunset over mountains"),
    goai.WithImageCount(1),
)
```

Image provider options (pass under the `"vertex"` key in `ImageParams.ProviderOptions`):

| Option | Type | Description |
|--------|------|-------------|
| `negativePrompt` | string | Text describing what to avoid |
| `personGeneration` | string | `dont_allow`, `allow_adult`, `allow_all` |
| `safetySetting` | string | `block_low_and_above`, `block_medium_and_above`, `block_only_high`, `block_none` |
| `addWatermark` | bool | Whether to add a watermark |
| `sampleImageSize` | string | `1K` or `2K` |
| `seed` | int | Seed for reproducible generation |

### Realtime Voice (Live API)

`Live` opens a Gemini Live [realtime session](../concepts/realtime.md) on Vertex AI. It resolves the project, location, and Application Default Credentials like `Chat`, connects to the regional `LlmBidiService` endpoint, and names the model as a Vertex publisher resource. A project is required. Experimental.

```go
model := vertex.Live("gemini-3.8-live",
    vertex.WithProject("my-gcp-project"),
    vertex.WithLocation("us-central1"),
)
session, err := goai.OpenRealtimeSession(ctx, model, goai.WithRealtimeTools(weatherTool))
```

The session protocol, audio formats, and options are the same as on the Gemini API; see the [Google provider](google.md#realtime-voice-live-api). `WithBaseURL` replaces the endpoint host.

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Set a static API key (routes to Gemini API endpoint) |
| `WithTokenSource(ts)` | `provider.TokenSource` | Set a dynamic token source (e.g., service account) |
| `WithProject(project)` | `string` | Set the GCP project ID |
| `WithLocation(location)` | `string` | Set the GCP region (default: `us-central1`) |
| `WithBaseURL(url)` | `string` | Override the endpoint used by the selected constructor; rejected by `Chat` with `WithNativeGemini` |
| `WithNativeGemini()` | — | Select native Gemini `generateContent` for `Chat` (OAuth-only) |
| `WithNativeChatBaseURL(url)` | `string` | Override native Gemini Chat's models-collection URL |
| `WithHeaders(h)` | `map[string]string` | Set additional HTTP headers |
| `WithHTTPClient(c)` | `*http.Client` | Set a custom `*http.Client` |

## Notes

- For direct API key access without GCP, see the [Google provider](google.md).
- With ADC or an explicit token source, `Chat` uses the Vertex AI OpenAI-compatible endpoint by default. `WithNativeGemini()` selects native `generateContent`; API keys are rejected in that mode. API-key fallback without native mode routes chat to `generativelanguage.googleapis.com`. Embeddings and image generation use native `:predict` endpoints (or the Gemini API equivalent with an API key).
- `WithNativeGemini` and `WithNativeChatBaseURL` are Chat-only and are rejected by `Embedding`, `Image`, and `AnthropicChat`.
- Tool schemas are automatically sanitized to comply with Gemini schema restrictions (no `additionalProperties`, enum values must be strings, etc.).
- Gemini-native provider options like `thinkingConfig` are translated to the OpenAI-compatible `reasoning_effort` knob (rather than dropped) before sending to the OpenAI-compatible endpoint.
- The `ADCTokenSource(ctx context.Context, scopes ...string)` function is exported for direct use with other providers that need GCP credentials. It takes a `context.Context` and optional OAuth scopes, and returns `(provider.TokenSource, error)`.
- Max embedding batch size: 250 values per call.
