# GoAI Roadmap

> Last updated: 2026-04-19

## v0.4.4

### Core functions

- **GenerateText** / **StreamText** - Text generation with streaming (token-by-token, text-only, or blocking)
- **GenerateObject[T]** / **StreamObject[T]** - Type-safe structured output with auto JSON Schema from Go structs
- **Embed** / **EmbedMany** - Single and batch embeddings with auto-chunking + parallel execution
- **GenerateImage** - Text-to-image generation (OpenAI DALL-E, Google Imagen, Azure, Vertex AI)

### Providers (24+)

| Category         | Providers                                            |
| ---------------- | ---------------------------------------------------- |
| Flagship         | OpenAI, Anthropic, Google (Gemini + Imagen)          |
| Cloud platforms  | AWS Bedrock (SigV4), Azure OpenAI, Google Vertex AI  |
| Fast inference   | Groq, Cerebras, Fireworks, Together, DeepInfra       |
| Specialized      | Mistral, xAI, DeepSeek, Cohere, Perplexity, MiniMax  |
| Aggregators      | OpenRouter                                           |
| Edge / regional  | Cloudflare Workers AI, FPT Smart Cloud (Global + JP) |
| Local/Serverless | Ollama, vLLM, RunPod                                 |
| Bring your own   | `compat.Chat()` for any OpenAI-compatible endpoint   |

### SDK features

- **Tool system** - Define tools with JSON Schema, auto tool loop with `WithMaxSteps`
- **TokenSource** - Static keys, OAuth-refreshed, cached credentials (mutex-free during network I/O, TTL-based)
- **WithHTTPClient** - Custom transport for proxies, auth middleware, Codex/Copilot patterns
- **Prompt caching** - `WithPromptCaching(bool)` automatic `cache_control` on system messages (immutable, no input mutation)
- **Retry/backoff** - Exponential backoff on 429/5xx (+ OpenAI 404), `InvalidatingTokenSource` interface for token refresh on auth failures
- **Thread-safe** - All providers safe for concurrent use; Bedrock fallback uses RWMutex for cross-region retry
- **Telemetry hooks** - `WithOnRequest`, `WithOnResponse`, `WithOnToolCall`, `WithOnStepFinish`
- **SchemaFrom[T]** - Reflection-based JSON Schema generation, OpenAI strict mode compatible
- **Azure multi-model** - Auto-routing: OpenAI models use Responses API, Claude uses Anthropic endpoint, others use Chat Completions
- **Array content** - Handles response content as string or `[{type:"text",text:"..."}]` (Mistral magistral models)
- **Provider-defined tools**: 20 tools across 5 providers: Anthropic (10), OpenAI (4), Google (3), Groq (1), xAI (2). All E2E tested.
- **E2E validated** - 103 models across 7 providers tested with real API calls
- **Benchmarks** - Go wins 5/5 comparable categories (schema is a tie) vs Vercel AI SDK: streaming 1.1x, TTFC 1.3x, cold start 24.4x, memory 3.1x, GenerateText 1.4x
- **Documentation** - Full docs site, 24 provider pages, 26 runnable examples, API reference

---

## v0.5.1

| Feature        | Description                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| **MCP client** | 3 transports (stdio, HTTP, SSE), tools/prompts/resources, ConvertTools, 7 examples |
| xAI tools      | Provider-defined tools (web_search, x_search) via `/chat/completions`              |
| MiniMax        | M2.7, M2.5, M2.1, M2 models                                                        |

## v0.5.8

| Feature                | Description                                                                       |
| ---------------------- | --------------------------------------------------------------------------------- |
| **RunPod provider**    | Serverless vLLM endpoint support                                                  |
| **Bedrock embeddings** | Embedding support for all Bedrock text embedding models                           |
| **Docs / accuracy**    | Deep-review audit fixes: docs accuracy, streaming metadata, provider capabilities |

## v0.6.0

| Feature                    | Description                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------- |
| **OpenTelemetry tracing**  | Full tracing for generations, tool calls, and multi-step loops via `observability/otel` |
| **OpenTelemetry metrics**  | Token usage, request duration, and error rate metrics with GenAI semantic conventions   |
| **Context propagation**    | `RequestInfo.Ctx` carries trace context through provider calls                          |
| **Langfuse data race fix** | Fixed concurrent map access in Langfuse observability integration                       |

## v0.7.0

| Feature                   | Description                                                                                                                               |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Cloudflare Workers AI** | New provider with chat + embeddings. OpenAI-compatible endpoints, account-ID URL building, optional AI Gateway override via `WithBaseURL` |
| **FPT Smart Cloud**       | New provider (FPT AI Marketplace) with chat + embeddings. `WithRegion("global"/"jp")` for Japan / Global routing                          |

## v0.7.1

| Feature                            | Description                                                                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **OpenAI-compat factory refactor** | `openaicompat.NewChatModel` / `NewEmbeddingModel` absorb HTTP dispatch, token resolution, env-var plumbing from 14 provider packages. ~1,400 LOC removed. Public API unchanged |

## v0.7.2

| Feature        | Description                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------------- |
| **NVIDIA NIM** | New provider (OpenAI-compatible, chat + embeddings). E2E tested with `meta/llama-3.3-70b-instruct` |

## v0.7.3 - Current release

| Feature                       | Description                                                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reasoning text on results** | New `Reasoning` field on `TextResult`, `StepResult`, and `provider.GenerateResult`. Populated for both `GenerateText` and `StreamText` across Bedrock, Anthropic, OpenAI Responses, Google, Cohere, and OpenAI-compat (DeepSeek). |

### Planned

| Feature         | Description                                                                                   |
| --------------- | --------------------------------------------------------------------------------------------- |
| Output.array    | Stream validated array elements incrementally                                                 |
| Output.choice   | Convenience enum selection wrapper                                                            |
| ~~`goai/otel`~~ | ~~Pre-built OpenTelemetry integration (optional import)~~ **Shipped** in `observability/otel` |

### v1.0.0 - Stable API

GoAI reaches v1.0 when the API is complete enough that most Go+AI applications can be built without workarounds:

- **Stable interfaces** - `LanguageModel`, `EmbeddingModel`, `ImageModel` finalized with no planned breaking changes
- **Full provider coverage** - Every major AI provider works out of the box, including auth flows and regional endpoints
- **Production observability** - ~~First-class OpenTelemetry integration~~ (shipped), structured logging hooks, usage tracking
- **Comprehensive documentation** - Every exported type and function documented with examples, migration guides for common patterns

### Future

| Feature   | Description                                                     |
| --------- | --------------------------------------------------------------- |
| Agent     | Multi-step agent abstraction with built-in tool loop and memory |
| Reranking | `goai.Rerank()` for search and retrieval pipelines              |
| Speech    | Server-side audio generation and transcription                  |

---

## Design principles

1. **Go-native API** - Functional options, interfaces, composition. No TypeScript transliterations.
2. **Minimal required dependencies** - Core GoAI depends on stdlib + `golang.org/x/oauth2`. Optional submodules (e.g. `observability/otel`) have their own `go.mod` and are not pulled unless imported.
3. **Provider-agnostic** - Same code works across all providers. Switch models by changing one line.
4. **Consumer flexibility** - `WithHTTPClient` + `TokenSource` let consumers handle auth, proxies, and custom endpoints without GoAI needing to know.
5. **No middleware, no registry** - Go's interface composition is sufficient. We don't add abstractions until proven necessary.

---

## Contributing

Have a feature request? Open an issue on [GitHub](https://github.com/zendev-sh/goai/issues). PRs welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
