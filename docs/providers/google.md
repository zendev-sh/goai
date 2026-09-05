---
title: Google Gemini Provider
description: "Use Google Gemini models in Go with GoAI. Supports chat, embeddings, image and Veo video generation, Google Search grounding, and code execution tools."
---

# Google

[Google Gemini](https://ai.google.dev/) provider for GoAI - Gemini models via the Google AI REST API.

For Vertex AI (GCP-managed Gemini), see the [Vertex provider](vertex.md).

## Setup

```bash
go get github.com/zendev-sh/goai@latest
```

Set the `GOOGLE_GENERATIVE_AI_API_KEY` (or `GEMINI_API_KEY`) environment variable, or pass it explicitly:

```go
import "github.com/zendev-sh/goai/provider/google"

model := google.Chat("gemini-2.5-flash", google.WithAPIKey("..."))
```

The provider also reads `GOOGLE_GENERATIVE_AI_BASE_URL` from the environment when no explicit base URL is set.

## Models

| Model ID | Type | Notes |
|----------|------|-------|
| `gemini-2.5-flash` | Chat | Fast, thinking-capable |
| `gemini-2.5-flash-lite` | Chat | Lightweight variant |
| `gemini-2.5-pro` | Chat | Most capable, thinking-capable |
| `gemini-3-flash-preview` | Chat | Next-gen, thinking-capable |
| `gemini-3-pro-preview` | Chat | Next-gen, thinking-capable |
| `gemini-3.1-pro-preview` | Chat | Next-gen, thinking-capable |
| `gemini-2.0-flash` | Chat | Previous gen, no thinking |
| `text-embedding-004` | Embedding | 768 dimensions |
| `imagen-4.0-generate-001` | Image | Imagen via :predict endpoint |
| `imagen-4.0-fast-generate-001` | Image | Imagen fast variant via :predict endpoint |
| `gemini-2.5-flash-image` | Image | Gemini image via generateContent |
| `veo-3.1-generate-preview` | Video | Veo via asynchronous predictLongRunning |

## Tested Models

E2E tested with real API calls. Last run: 2026-03-15.

| Model | Generate | Stream | Status |
|-------|----------|--------|--------|
| `gemini-2.5-flash` | PASS | PASS | Stable |
| `gemini-2.5-flash-lite` | PASS | PASS | Stable |
| `gemini-2.5-pro` | PASS | PASS | Stable |
| `gemini-3-flash-preview` | PASS | PASS | Stable |
| `gemini-3-pro-preview` | PASS | PASS | Stable |
| `gemini-3.1-pro-preview` | PASS | PASS | Stable |
| `gemini-2.0-flash` | PASS | PASS | Stable |
| `gemini-flash-latest` | PASS | PASS | Stable |
| `gemini-flash-lite-latest` | PASS | PASS | Stable |

Unit/integration tested models: `gemini-2.5-flash`, `gemini-2.5-flash-image`, `imagen-4.0-generate-001`, `imagen-4.0-fast-generate-001`, `text-embedding-004`, and `veo-3.1-generate-preview` (mock HTTP lifecycle). A credential-gated `e2e` test is available for a real Veo request; it is excluded from normal test runs to prevent accidental billing.

## Usage

### Chat

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/google"
)

func main() {
    model := google.Chat("gemini-2.5-flash")
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
    "github.com/zendev-sh/goai/provider/google"
)

model := google.Chat("gemini-2.5-flash")
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

### Embedding

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/google"
)

model := google.Embedding("text-embedding-004")
result, err := goai.Embed(context.Background(), model, "Hello world")
if err != nil {
    panic(err)
}
fmt.Println(len(result.Embedding)) // 768
```

Google-specific embedding options (under the `"google"` key in ProviderOptions):

```go
result, err := goai.Embed(ctx, model, "Hello world",
    goai.WithEmbeddingProviderOptions(map[string]any{
        "google": map[string]any{
            "taskType":             "RETRIEVAL_DOCUMENT",
            "title":                "My document title", // only valid for RETRIEVAL_DOCUMENT
            "outputDimensionality": 256,
        },
    }),
)
```

### Image Generation (Imagen)

```go
import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/google"
)

model := google.Image("imagen-4.0-fast-generate-001")
result, err := goai.GenerateImage(context.Background(), model,
    goai.WithImagePrompt("A serene mountain lake at sunset"),
)
if err != nil {
    panic(err)
}
fmt.Printf("Generated %d bytes\n", len(result.Images[0].Data))
```

### Image Generation (Gemini)

Gemini image models use the generateContent endpoint with `responseModalities: ["IMAGE"]`:

```go
model := google.Image("gemini-2.5-flash-image")
result, err := goai.GenerateImage(ctx, model,
    goai.WithImagePrompt("A cartoon cat programming in Go"),
)
```

### Video Generation (Veo)

Veo generation is asynchronous. GoAI starts the operation, polls it until completion, and downloads the authenticated video result.

```go
model := google.Video("veo-3.1-generate-preview")
result, err := goai.GenerateVideo(ctx, model,
    goai.WithVideoPrompt("A paper plane taking flight at sunrise"),
    goai.WithVideoAspectRatio("16:9"),
    goai.WithVideoResolution("720p"),
    goai.WithVideoDuration(8*time.Second),
    goai.WithVideoAudio(true),
)
if err != nil {
    panic(err)
}
fmt.Printf("Generated %d bytes\n", len(result.Video.Data))
```

Use `WithVideoImage` for image-to-video generation. `WithVideoFrameImages` supports first and last frames, while `WithVideoInputReferences` accepts image references; Veo does not allow combining those two modes. Common pixel resolutions are translated to Google's `720p`, `1080p`, and `4k` values. The Gemini Developer API supports `WithVideoSeed` but not the generic FPS option.

## Options

| Option | Type | Description |
|--------|------|-------------|
| `WithAPIKey(key)` | `string` | Static API key. Falls back to `GOOGLE_GENERATIVE_AI_API_KEY` or `GEMINI_API_KEY` env var. |
| `WithTokenSource(ts)` | `provider.TokenSource` | Dynamic token resolution. |
| `WithBaseURL(url)` | `string` | Override base URL. Falls back to `GOOGLE_GENERATIVE_AI_BASE_URL` env var. |
| `WithHeaders(h)` | `map[string]string` | Additional HTTP headers on every request. |
| `WithHTTPClient(c)` | `*http.Client` | Custom HTTP client for proxies, logging, URL rewriting. |

### Provider Options (via `goai.WithProviderOptions`)

Google-specific options are nested under the `"google"` key:

| Key | Type | Description |
|-----|------|-------------|
| `google.thinkingConfig` | `map[string]any` or `bool` | Override thinking config: `{includeThoughts, thinkingLevel}`. Set to `false` to disable. |
| `google.safetySettings` | `[]map[string]any` | Safety settings per category. |
| `google.cachedContent` | `string` | Cached content resource name. |
| `google.responseModalities` | `[]string` | Output modalities (e.g., `["TEXT", "IMAGE"]`). |
| `google.mediaResolution` | `string` | Media resolution for input images/videos. |
| `google.audioTimestamp` | `bool` | Enable audio timestamps in response. |
| `google.imageConfig` | `map[string]any` | Gemini image config: `{aspectRatio, imageSize}`. |
| `google.retrievalConfig` | `map[string]any` | Retrieval configuration for tool config. |
| `google.toolConfig.includeServerSideToolInvocations` | `bool` | Include Gemini server-side tool calls/responses; other raw `toolConfig` fields are ignored. |
| `google.google_search` | `map[string]any` | Legacy Google Search via ProviderOptions (prefer `google.Tools.GoogleSearch()`). |

## Provider Tools

Five built-in tools are available via `google.Tools`.

| Tool | Description | Execution |
|------|-------------|-----------|
| `google.Tools.GoogleSearch()` | Grounding with Google Search | Server-side |
| `google.Tools.URLContext()` | Fetch and process URL content | Server-side |
| `google.Tools.CodeExecution()` | Execute Python code in sandbox | Server-side |
| `google.Tools.ComputerUse(opts...)` | Computer use with configurable environment/excluded functions | Client-side |
| `google.Tools.FileSearch(opts...)` | File-based grounding (Vertex AI Search / uploaded files) | Server-side |

### GoogleSearch

```go
def := google.Tools.GoogleSearch()
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What are the latest Go releases?"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
// result.Sources contains grounding URLs
for _, src := range result.Sources {
    fmt.Printf("Source: %s - %s\n", src.Title, src.URL)
}
```

Options: `WithWebSearch()`, `WithImageSearch()`, `WithTimeRange(startRFC3339, endRFC3339)`.

### URLContext

```go
def := google.Tools.URLContext()
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Summarize the content at https://go.dev/doc/effective_go"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

No configuration options. The model uses URLs from the prompt to fetch and process content.

### CodeExecution

```go
def := google.Tools.CodeExecution()
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Calculate the mean and median of [3, 7, 8, 5, 12, 14, 21]"),
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

No configuration options. The model generates Python code, executes it server-side, and uses the output in its response.

### FileSearch

Grounds responses on a retrieval corpus (Vertex AI Search) or uploaded files.

```go
def := google.Tools.FileSearch(
    google.WithDynamicThreshold(0.6),
    google.WithCorpus("projects/.../corpora/my-corpus"),
)
```

Options: `WithDynamicThreshold(t)` (0.0–1.0) and `WithCorpus(id)`.

## Notes

- **Thinking/Reasoning**: Enabled by default for Gemini 2.5+ and 3.x models. Gemini 3.x models default to `thinkingLevel: "high"`. Gemma and Gemini 1.5/2.0 models do not support thinking. Set `google.thinkingConfig` to `false` in provider options to disable.
- **Schema sanitization**: Gemini has stricter JSON Schema requirements. GoAI auto-sanitizes schemas: enum values are coerced to strings, `additionalProperties` is removed, array items get a default type, and `required` fields are filtered to match `properties`.
- **Implicit caching**: The request body uses a struct with deterministic field ordering (system instruction and tools before contents) to maximize Gemini's implicit prompt cache hit rate.
- **Function call IDs**: Gemini does not provide tool call IDs. GoAI generates synthetic IDs in the format `call_{toolName}_{N}` with a counter to ensure uniqueness.
- **Role mapping**: `assistant` is mapped to `model`, and `tool` is mapped to `user` for function responses.
- **Embedding batch limit**: 100 values per call via `batchEmbedContents`. `goai.EmbedMany` auto-chunks larger batches.
- **Difference from Vertex**: This provider uses Google AI (`generativelanguage.googleapis.com`) with API-key auth. `vertex.Chat` uses Vertex's OpenAI-compatible transport by default and can opt into native Gemini `generateContent` with `vertex.WithNativeGemini()` plus OAuth/ADC. Native Vertex chat intentionally does not expose the Gemini Developer API file uploader.
- **File upload**: Google Gemini supports remote file upload via the Files API, using the two-phase resumable upload protocol (`X-Goog-Upload-*`). Use `model.FileUploader()` to get a `provider.FileUploader` for uploading and deleting files. Uploaded files are referenced via `Part.RemoteRef` in messages. Compat providers map file parts to native OpenAI shapes (`file` for PDFs, `input_audio` for audio).
