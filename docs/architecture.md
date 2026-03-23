---
title: Architecture
description: "GoAI SDK architecture overview — layers, data flow, provider system, and design principles."
---

# Architecture

GoAI is a Go SDK that provides one unified API across 20+ LLM providers. This document describes the overall architecture, key layers, data flow, and design decisions.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Application                         │
│  goai.GenerateText / StreamText / GenerateObject / StreamObject │
│  goai.Embed / EmbedMany / GenerateImage                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    options, retry, caching
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     Provider Interfaces                         │
│         LanguageModel · EmbeddingModel · ImageModel             │
└────┬────────────┬───────────┬───────────┬───────────┬───────────┘
     │            │           │           │           │
  ┌──▼───┐    ┌───▼───┐   ┌───▼───┐   ┌───▼────┐  ┌───▼────────┐
  │OpenAI│    │Anthro.│   │Google │   │Bedrock │  │13 compat   │
  │      │    │       │   │       │   │        │  │providers   │
  └──┬───┘    └───┬───┘   └───┬───┘   └───┬────┘  └───┬────────┘
     │            │           │           │           │
     │     Native API    Native API   Converse API    │
     │     (Responses    (Messages)   + SigV4 +       │
     │      + Chat                    EventStream     │
     │     Completions)                               │
     │                                                │
     └──────────────────────┬─────────────────────────┘
                            │
               ┌────────────▼─────────────┐
               │  internal/openaicompat   │
               │  Shared codec for 13+    │
               │  OpenAI-compatible APIs  │
               └────────────┬─────────────┘
                            │
               ┌────────────▼─────────────┐
               │     internal/sse         │
               │     SSE line parser      │
               ├──────────────────────────┤
               │     internal/httpc       │
               │     HTTP helpers         │
               └──────────────────────────┘
```

## Layers

GoAI is structured as three layers:

### 1. Core SDK (`goai` package)

The top-level `goai` package exposes 7 core functions — the only public API that users interact with:

| Function            | File          | Description                                                |
| ------------------- | ------------- | ---------------------------------------------------------- |
| `GenerateText`      | `generate.go` | Non-streaming text generation with auto tool loop          |
| `StreamText`        | `generate.go` | Streaming text generation via Go channels                  |
| `GenerateObject[T]` | `object.go`   | Type-safe structured output using generics                 |
| `StreamObject[T]`   | `object.go`   | Streaming partial objects with typed results               |
| `Embed`             | `embed.go`    | Single text embedding                                      |
| `EmbedMany`         | `embed.go`    | Batch embeddings with auto-chunking and parallel execution |
| `GenerateImage`     | `image.go`    | Image generation                                           |

Supporting modules in the core:

| File         | Responsibility                                                        |
| ------------ | --------------------------------------------------------------------- |
| `options.go` | Functional options pattern (`WithPrompt`, `WithTools`, etc.)          |
| `schema.go`  | `SchemaFrom[T]` — JSON Schema generation from Go types via reflection |
| `retry.go`   | Exponential backoff with jitter, Retry-After header support           |
| `caching.go` | Prompt cache control (copies messages, never mutates input)           |
| `errors.go`  | `APIError`, `ContextOverflowError`, overflow pattern detection        |

### 2. Provider Interfaces (`provider` package)

The `provider` package defines three model interfaces that all providers implement:

```go
type LanguageModel interface {
    ModelID() string
    DoGenerate(ctx context.Context, params GenerateParams) (*GenerateResult, error)
    DoStream(ctx context.Context, params GenerateParams) (*StreamResult, error)
}

type EmbeddingModel interface {
    ModelID() string
    DoEmbed(ctx context.Context, values []string, params EmbedParams) (*EmbedResult, error)
    MaxValuesPerCall() int
}

type ImageModel interface {
    ModelID() string
    DoGenerate(ctx context.Context, params ImageParams) (*ImageResult, error)
}
```

The provider package also defines all shared types: `Message`, `Part`, `StreamChunk`, `Usage`, `ToolDefinition`, `ToolCall`, `Source`, and `ResponseMetadata`.

An optional `CapableModel` interface allows providers to declare feature support (temperature, reasoning, tool calling, modalities).

### 3. Internal Packages (`internal/`)

| Package                 | Description                                                                                                                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `internal/openaicompat` | Shared request building and response parsing for all OpenAI-compatible APIs. `BuildRequest` constructs the wire format, `ParseStream`/`ParseResponse` decode responses into GoAI types. Used by 13+ providers. |
| `internal/sse`          | Minimal SSE (Server-Sent Events) scanner. Handles `data:` prefix, blank lines, `[DONE]` sentinel. JSON deserialization is left to the caller.                                                                  |
| `internal/httpc`        | HTTP utilities: `MustMarshalJSON`, `MustNewRequest`, `ParseDataURL`. Shared across all providers.                                                                                                              |

## Provider Architecture

Providers fall into two categories:

### Native Providers (custom API format)

These providers implement their own request/response codec because their APIs differ significantly from OpenAI:

| Provider      | API                              | Key Differences                                                                                                                                                                                                                         |
| ------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenAI**    | Chat Completions + Responses API | Dual API routing: Responses API (default for all models), Chat Completions (opt-in via `useResponsesAPI: false`). Provider-defined tools: web search, code interpreter, file search, image generation.                                  |
| **Anthropic** | Messages API                     | Custom SSE event format (`message_start`, `content_block_start`, `content_block_delta`, etc.). Structured output via tool trick or native `output_format`. Provider-defined tools: computer use, web search, code execution, web fetch. |
| **Google**    | Gemini REST API                  | Custom JSON format with `contents[]` array, `generationConfig`, `tools[]`. Provider-defined tools: Google Search grounding, URL context, code execution.                                                                                |
| **Bedrock**   | AWS Converse API                 | Binary EventStream protocol for streaming, SigV4 request signing (implemented without AWS SDK), cross-region inference fallback with `us.` prefix retry.                                                                                |
| **Cohere**    | Chat v2 + Embed API              | Custom message format, tool results as `tool_content` parts, native embed API.                                                                                                                                                          |

### OpenAI-Compatible Providers (shared codec)

13 providers use the `internal/openaicompat` shared codec. Each provider package is a thin wrapper that:

1. Sets the correct base URL and auth headers
2. Resolves credentials from environment variables
3. Delegates to `openaicompat.BuildRequest` / `ParseStream` / `ParseResponse`

```
openaicompat providers:
├── azure/       ← Also routes Claude models to Anthropic endpoint
├── vertex/      ← Uses OAuth2 ADC for auth
├── mistral/     ├── groq/       ├── xai/
├── deepseek/    ├── fireworks/  ├── together/
├── deepinfra/   ├── openrouter/ ├── perplexity/
├── cerebras/    ├── ollama/     ├── vllm/
└── compat/      ← Generic, user-configured endpoint
```

### Provider Contract

Every provider must:

- Have compile-time interface checks: `var _ provider.LanguageModel = (*chatModel)(nil)`
- Support: `WithAPIKey`, `WithTokenSource`, `WithBaseURL`, `WithHTTPClient`, `WithHeaders`
- Auto-resolve credentials from environment variables (e.g., `OPENAI_API_KEY`)
- Return `*goai.APIError` or `*goai.ContextOverflowError` for HTTP errors (via `goai.ParseHTTPErrorWithHeaders`)

## Data Flow

### GenerateText (non-streaming)

```
User calls goai.GenerateText(ctx, model, opts...)
  │
  ├─ applyOptions(opts...)          → merge functional options into options struct
  ├─ buildParams(options)           → convert to provider.GenerateParams
  ├─ buildToolMap(tools)            → create name→Tool lookup for auto loop
  │
  ├─ for step = 1..MaxSteps:
  │   ├─ OnRequest callback
  │   ├─ withRetry(fn):
  │   │   └─ model.DoGenerate(ctx, params)
  │   │       ├─ Build HTTP request (provider-specific)
  │   │       ├─ Send HTTP POST
  │   │       ├─ Parse response → provider.GenerateResult
  │   │       └─ Return {Text, ToolCalls, Usage, FinishReason, Sources}
  │   ├─ OnResponse callback
  │   ├─ if FinishReason == ToolCalls:
  │   │   ├─ executeTools(ctx, calls, toolMap, OnToolCall)
  │   │   └─ appendToolRoundTrip(msgs, result, toolMsgs)
  │   └─ else: return buildTextResult(steps, totalUsage)
  │
  └─ return TextResult{Text, Steps, TotalUsage, FinishReason, Sources}
```

### StreamText (streaming)

```
User calls goai.StreamText(ctx, model, opts...)
  │
  ├─ withRetry → model.DoStream(ctx, params)
  │   ├─ Build HTTP request (provider-specific)
  │   ├─ Send HTTP POST (streaming)
  │   ├─ Create channel (cap 64)
  │   ├─ Launch goroutine: parse SSE → emit StreamChunks
  │   └─ Return StreamResult{Stream: <-chan StreamChunk}
  │
  └─ Wrap in TextStream
      ├─ .TextStream() → <-chan string  (text-only)
      ├─ .Stream()     → <-chan StreamChunk (raw chunks)
      └─ .Result()     → *TextResult (blocks until done)
```

### GenerateObject[T] (structured output)

```
User calls goai.GenerateObject[Recipe](ctx, model, opts...)
  │
  ├─ SchemaFrom[Recipe]()           → JSON Schema via reflection
  ├─ params.ResponseFormat = {Name, Schema}
  │
  ├─ Provider applies ResponseFormat:
  │   ├─ OpenAI: json_schema response_format
  │   ├─ Anthropic: tool trick (injects synthetic tool) or native output_format
  │   └─ Google: responseMimeType + responseSchema
  │
  ├─ json.Unmarshal(result.Text, &obj)
  └─ return ObjectResult[Recipe]{Object, Usage, FinishReason}
```

## Auto Tool Loop

When tools with `Execute` handlers are provided and `MaxSteps > 1`, `GenerateText` automatically runs a tool loop:

```
Step 1: Generate → model returns ToolCalls
Step 2: Execute tools → collect results
Step 3: Append assistant message (with tool_use parts) + tool result messages
Step 4: Re-generate with updated conversation
... repeat until model stops requesting tools or MaxSteps reached
```

Each step fires `OnStepFinish` and `OnToolCall` callbacks. Usage is aggregated across all steps into `TotalUsage`.

## Streaming Architecture

Streaming uses Go channels (`<-chan StreamChunk`) with a buffer size of 64. Each provider launches a goroutine that:

1. Reads SSE events from the HTTP response body
2. Parses each event into `StreamChunk` structs
3. Sends chunks to the channel via `provider.TrySend` (context-aware, prevents goroutine leaks)
4. Closes the channel when the stream ends

`TextStream` wraps the raw channel and provides three consumption modes:

- `Stream()` — raw `StreamChunk` channel (for full control)
- `TextStream()` — text-only `string` channel (convenience)
- `Result()` — blocks until done, returns accumulated `TextResult`

These are mutually exclusive (enforced by `sync.Once`). The consume goroutine accumulates text, tool calls, sources, usage, and finish reason.

## Authentication

### TokenSource Interface

All providers accept either a static API key or a dynamic `TokenSource`:

```go
type TokenSource interface {
    Token(ctx context.Context) (string, error)
}
```

Implementations:

- **`StaticToken(key)`** — wraps a string, returns it unchanged
- **`CachedTokenSource(fetchFn)`** — lazy fetch with TTL-based caching

### CachedTokenSource

Thread-safe token caching with a critical design constraint: **never hold a mutex during network calls**.

```
Token(ctx) called
  ├─ Lock → check cache → if valid → unlock → return cached
  ├─ Unlock (before network!)
  ├─ fetchFn(ctx) → get fresh token (may be slow)
  ├─ Lock → store in cache → unlock
  └─ Return token
```

Brief double-fetch on concurrent expiry is acceptable. `CachedTokenSource` also implements `InvalidatingTokenSource` for retry-on-401 logic.

### AWS Bedrock Auth

Bedrock implements SigV4 signing without any AWS SDK dependency. Credentials are resolved from environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`). Cross-region fallback retries with `us.` prefix on model ID mismatch errors.

## Error Handling

All errors flow through typed error types:

```
HTTP error response
  │
  ├─ goai.ParseHTTPErrorWithHeaders(providerID, status, body, headers)
  │   ├─ extractErrorMessage() — tries {error.message}, {message}, {error}
  │   ├─ IsOverflow(message) — matches 14 regex patterns across providers
  │   │   └─ return *ContextOverflowError
  │   └─ return *APIError{StatusCode, IsRetryable, ResponseHeaders}
  │
  └─ Retry logic (in withRetry):
      ├─ errors.As(err, &apiErr) — always use errors.As, never type assertion
      ├─ apiErr.IsRetryable → 429, 503, 5xx
      ├─ Retry-After / Retry-After-ms header → exact delay
      └─ Exponential backoff: 2s × 2^attempt, jitter 0.5–1.5x, cap 60s
```

## Retry Mechanism

`withRetry[T]` is a generic retry wrapper used by all core functions:

- Attempts: `maxRetries + 1` (default `maxRetries = 2`, so 3 total attempts)
- Only retries on `APIError.IsRetryable` (429, 503, 5xx)
- Respects `Retry-After-ms` (OpenAI) and `Retry-After` (standard) headers
- Falls back to exponential backoff: base 2s, factor 2x, jitter 0.5–1.5x, cap 60s
- Context cancellation aborts retries immediately

## Schema Generation

`SchemaFrom[T]()` generates JSON Schema from Go types via reflection, compatible with OpenAI strict mode:

- All properties are `required` (pointer types become `nullable` via `type: ["<base>", "null"]`)
- `additionalProperties: false` on all objects
- Supports struct tags: `json:"name"` for naming, `jsonschema:"description=...,enum=a|b|c"` for metadata
- Flattens embedded structs
- Handles all Go primitive types, slices, maps, and nested structs

## Prompt Caching

When enabled via `WithPromptCaching(true)`, `applyCaching` copies messages (never mutates input) and sets `CacheControl: "ephemeral"` on strategic positions:

- Last part of the last user message
- Last part of the second-to-last user message (for multi-turn caching)
- System prompt (provider-specific: Anthropic adds `cache_control` to system content)

## Design Principles

1. **No external dependencies** — stdlib only, except `golang.org/x/oauth2` for Vertex AI ADC
2. **Vercel AI SDK is the reference** — wire formats, option names, and behaviors match Vercel
3. **No input mutation** — all functions copy slices/maps before modifying
4. **Lock-free network calls** — never hold a mutex during I/O
5. **`errors.As`, not type assertion** — always `errors.As(err, &target)`, never `err.(*Type)`
6. **Interface compliance checks** — every provider has `var _ provider.LanguageModel = (*chatModel)(nil)`
7. **Shared utilities in `internal/`** — common code lives once, not duplicated per provider
8. **90% test coverage** — mock HTTP servers, not internals; E2E tests for tier-1 providers

## Directory Structure

```
goai/
├── generate.go             # GenerateText, StreamText, TextStream, tool loop
├── object.go               # GenerateObject[T], StreamObject[T], ObjectStream
├── embed.go                # Embed, EmbedMany (auto-chunking, parallel)
├── image.go                # GenerateImage
├── options.go              # Functional options (WithPrompt, WithTools, etc.)
├── schema.go               # SchemaFrom[T] — JSON Schema from Go structs
├── errors.go               # APIError, ContextOverflowError, overflow detection
├── retry.go                # Exponential backoff with jitter
├── caching.go              # Prompt cache control
│
├── provider/
│   ├── provider.go         # LanguageModel, EmbeddingModel, ImageModel interfaces
│   ├── types.go            # Message, Part, StreamChunk, Usage, ToolDefinition, etc.
│   ├── token.go            # TokenSource, CachedTokenSource, StaticToken
│   │
│   ├── openai/             # OpenAI (Chat Completions + Responses API)
│   ├── anthropic/          # Anthropic (Messages API)
│   ├── google/             # Google Gemini (REST API)
│   ├── bedrock/            # AWS Bedrock (Converse API + SigV4 + EventStream)
│   ├── vertex/             # Vertex AI (OpenAI-compat + OAuth2 ADC)
│   ├── azure/              # Azure OpenAI (+ auto-routes Claude to Anthropic endpoint)
│   ├── cohere/             # Cohere (Chat v2 + native Embed)
│   ├── compat/             # Generic OpenAI-compatible
│   └── ...                 # mistral, groq, xai, deepseek, fireworks, together,
│                           # deepinfra, openrouter, perplexity, cerebras, ollama, vllm
│
├── internal/
│   ├── openaicompat/       # Shared codec: BuildRequest, ParseStream, ParseResponse
│   ├── sse/                # SSE line parser (data: prefix, [DONE] sentinel)
│   └── httpc/              # HTTP helpers (MustMarshalJSON, MustNewRequest, ParseDataURL)
│
├── examples/               # 16 runnable examples
└── bench/                  # Performance benchmarks (GoAI vs Vercel AI SDK)
```
