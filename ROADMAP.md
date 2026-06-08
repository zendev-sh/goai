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

## v0.7.3

| Feature                       | Description                                                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reasoning text on results** | New `Reasoning` field on `TextResult`, `StepResult`, and `provider.GenerateResult`. Populated for both `GenerateText` and `StreamText` across Bedrock, Anthropic, OpenAI Responses, Google, Cohere, and OpenAI-compat (DeepSeek). |

## v0.7.4

| Feature                       | Description                                                                                                                                                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Azure v1 GA path fix**      | Drop the `api-version` query parameter on the v1 GA path (`/openai/v1{path}`) per Azure spec. Spec-strict resources rejected it with `"API version not supported"` (observed on gpt-5.5). Legacy deployment-based path still uses `api-version`; opt in via `WithDeploymentBasedURLs(true)` when an explicit dated version is required. |

## v0.7.5

| Feature                                | Description                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Streaming tool-call delta forward**  | `provider/openai` (Responses API) and `provider/cohere` now emit `ChunkToolCallDelta` for each upstream argument fragment instead of buffering until the accumulated args parse as JSON. Enables progressive rendering of large tool inputs (e.g. `write.content`, `edit.edits[].newText`). Final `ChunkToolCall` semantics unchanged. (#56) |

## v0.7.6

| Feature                                  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Server-executed tool calls round-trip** | Anthropic (`web_search`, `code_execution`, `web_fetch`, `mcp`) and OpenAI Responses (`web_search_call`, `file_search_call`, `code_interpreter_call`, `image_generation_call`, `local_shell_call`, `mcp_call`, `computer_call`) provider-executed tool results are now captured into `ToolCall.Metadata` and re-emitted verbatim when the assistant turn is serialized back, so multi-turn conversations no longer drop server search context or trip Anthropic's orphan-`tool_use` 400. New `bedrock.AnthropicChat` constructor routes Anthropic models through Bedrock InvokeModel / InvokeModelWithResponseStream so the anthropic provider's parsing applies. (#61) |

## v0.7.7

| Feature                                       | Description                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reasoning-model `max_completion_tokens` fix** | OpenAI-family reasoning models (o-series, gpt-5+, codex) now always get `max_completion_tokens` instead of `max_tokens`. The rename is keyed on the model id rather than on a `reasoning_effort` provider option being present, so `WithMaxOutputTokens` no longer trips an `Unsupported parameter: 'max_tokens'` rejection (observed on Azure gpt-5). (#69) |

## v0.7.8

| Feature                               | Description                                                                                                                                                                                                                                                                                                                |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SSE scanner accepts large lines**   | `internal/sse` swaps `bufio.Scanner` (1 MiB cap) for a `bufio.Reader`-based scanner so long tool-call argument deltas and reasoning blocks no longer fail with `bufio.Scanner: token too long`. A 16 MiB `MaxLineSize` cap prevents unbounded allocation from a hostile stream. (#73)                                       |
| **DeepSeek thinking-mode round-trip** | `internal/openaicompat` echoes `reasoning_content` back on assistant messages so DeepSeek thinking-mode survives multi-turn conversations instead of dropping the prior chain-of-thought. (#72)                                                                                                                            |

## v0.7.9

| Feature                                       | Description                                                                                                                                                                                                                                                                                                          |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenAI Responses API large SSE lines**     | `provider/openai` Responses streaming swaps its local `bufio.Scanner` (1 MiB cap) for the `internal/sse` scanner via a new `NextLine` method, so very long `output_text.delta` and reasoning events no longer fail with `bufio.Scanner: token too long`. Completes the fix from #73 for the Responses code path. (#75) |

## v0.7.11

| Feature                          | Description                                                                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **llama.cpp server provider**    | New `provider/llamacpp` package for connecting to llama.cpp server (OpenAI-compatible API at `http://localhost:8080`). Chat + Embedding support. (#77) |

## v0.7.10

| Feature                                          | Description                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP HTTPTransport: POST-only Streamable HTTP** | `mcp.HTTPTransport.Start` now treats `405` (or `404`) on the optional GET-for-SSE channel as "no server-initiated stream" per the MCP Streamable HTTP spec (2025-03-26), so POST-only servers (Zoho MCP and others) work instead of failing with `mcp: SSE connection failed: HTTP 405`. Inline JSON-RPC and `text/event-stream` POST responses are still dispatched as before. (#76) |

## v0.8.3 - Current release

| Feature                           | Description                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Native Ollama provider**        | `provider/ollama` now talks to Ollama's native `/api/chat` and `/api/embed` endpoints directly over `net/http` instead of the OpenAI-compat layer. Adds think (extended reasoning) mode via `ProviderOptions["think"]`, JSON mode via the native `format` field, and assistant reasoning round-tripped through the native `thinking` field. No third-party Ollama SDK dependency. Contributed by @bclermont. (#92) |
| **Duplicate tool name detection** | `GenerateText`, `StreamText`, `GenerateObject` and `StreamObject` now return an error when two tools share a name, validated across all tools to match the Vercel AI SDK (whose tool set is keyed by name), instead of silently letting the last one win. Contributed by @bclermont. (#91)                                                                                                                  |

## v0.8.2

| Feature                                       | Description                                                                                                                                                                                                                                                                                                                          |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Tool name on synthetic orphan results**     | `provider.NormalizeToolMessages` now copies `ToolName` onto the synthetic "Tool execution aborted" results it injects for orphaned tool calls, so consumers that normalize before the Google provider no longer hit `functionResponse.name: Name cannot be empty`. The normalization is now copy-on-input (no longer mutates the caller's messages or `Content`/`ProviderOptions`). Contributed by @nehmeroumani. (#90) |

## v0.8.1

| Feature                                       | Description                                                                                                                                                                                                                                                                                                                          |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **OpenAI Responses API rate-limit errors**    | Streaming `error` events from the Responses API are now parsed for both the documented flat shape and the nested `error.{code,message}` shape OpenAI returns in production, with per-field fallback so partial nested objects do not drop the flat detail. Rate-limit (`rate_limit_exceeded`) errors are classified as retryable 429s so the provider's retry logic kicks in instead of surfacing a generic non-retryable error. Applies to OpenAI Responses mode and Azure. Contributed by @nehmeroumani. (#88, #89) |

## v0.8.0

| Feature                              | Description                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **MCP OAuth 2.1 + PKCE**             | Native OAuth 2.1 authorization with PKCE for remote MCP servers: `mcp.NewOAuthTokenSource`, `mcp.NewOAuthHTTPClient`, and a pluggable `TokenStore` (`mcp.NewMemoryTokenStore`) handle the authorization-code + refresh flow so HTTP/SSE transports can talk to auth-protected MCP servers. See `examples/mcp-oauth`. Contributed by @DevLumuz. (#80) |
| **Hook & predicate panics surface (breaking)** | Panics in lifecycle hooks (`OnRequest`, `OnResponse`, `OnStepFinish`, `OnFinish`, `OnBeforeStep`) and the `StopWhen` predicate are no longer swallowed to stderr. They are surfaced as a `*PanicError` (returned by `GenerateText`/`GenerateObject`, or via `stream.Err()` for `StreamText`/`StreamObject`) and reported to a new `WithOnPanic` observability hook. The tool path (tool `Execute`, tool hooks) stays resilient, converting panics to tool errors. Code relying on the old swallow-and-continue behavior must register `OnPanic` or stop panicking in hooks. (#82) |
| **`NewTool` typed constructor**      | `goai.NewTool[In](name, description, execute)` builds a `Tool` from a typed input struct: the JSON Schema is generated via `SchemaFrom` and the model's arguments are unmarshaled into `In` before `execute` runs, so callers no longer hand-write JSON Schema or unmarshal input. The raw `goai.Tool` struct form remains for hand-written schemas and provider-defined tools. (#83) |
| **Smaller module**                   | Removed compiled example binaries (~34 MB) accidentally committed to the repo root; a pre-commit hook now blocks committing compiled executables. `go get` no longer downloads them. (#81) |

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
