# CLAUDE.md - GoAI

Instructions for Claude Code when working on the GoAI codebase.

## Commands

```bash
go build ./...          # Build
go test ./...           # Test all packages
go test -cover ./...    # Test with coverage
golangci-lint run       # Lint
go test ./provider/openai/  # Test single package
```

## Architecture

GoAI is a Go SDK for AI applications - one API across 21+ LLM providers. Inspired by Vercel AI SDK, adapted to Go idioms.

```
goai/
├── generate.go             # GenerateText, StreamText
├── object.go               # GenerateObject[T], StreamObject[T]
├── embed.go                # Embed, EmbedMany
├── image.go                # GenerateImage
├── options.go              # WithPrompt, WithTools, etc.
├── schema.go               # SchemaFrom[T] - JSON Schema from Go structs
├── hooks.go                # RequestInfo, ResponseInfo, ToolCallInfo, ToolCallStartInfo, hook options
├── errors.go               # APIError, ContextOverflowError
├── retry.go                # Exponential backoff (errors.As, not type assertion)
├── caching.go              # Prompt cache control (copies msgs, no mutation)
├── provider/
│   ├── provider.go         # LanguageModel, EmbeddingModel, ImageModel interfaces
│   ├── types.go            # Message, Part, Usage, StreamChunk
│   ├── token.go            # TokenSource, CachedTokenSource (lock-free fetch)
│   ├── openai/             # OpenAI (Chat Completions + Responses API)
│   ├── anthropic/          # Anthropic (Messages API)
│   ├── google/             # Google Gemini (REST)
│   ├── bedrock/            # AWS Bedrock (Converse API + SigV4 + EventStream, RWMutex for fallback; InvokeModel API for embeddings)
│   ├── vertex/             # Vertex AI
│   ├── azure/              # Azure OpenAI
│   ├── cohere/             # Cohere (Chat v2 + Embed)
│   ├── minimax/            # MiniMax (Anthropic-compat, delegates to anthropic/)
│   ├── compat/             # Generic OpenAI-compatible
│   └── <13 more>/          # OpenAI-compat via internal/openaicompat
│   # tools.go files: anthropic/ (10), openai/ (4), google/ (3), xai/ (2), groq/ (1)
├── internal/
│   ├── openaicompat/       # Shared codec for 13+ providers
│   ├── sse/                # SSE parser
│   └── httpc/              # HTTP helpers + ParseDataURL
├── examples/               # 16 runnable examples
└── bench/                  # Performance benchmarks (GoAI vs Vercel AI SDK)
```

## Key Rules

1. **No external dependencies** beyond `golang.org/x/oauth2` (Vertex ADC only)
2. **Vercel AI SDK is the reference** - check Vercel source before modifying provider behavior
3. **90% test coverage** per package - mock HTTP servers, not internals
4. **Interface compliance checks** - every provider struct must have `var _ provider.LanguageModel = (*chatModel)(nil)`
5. **errors.As, not type assertion** - always `errors.As(err, &apiErr)`, never `err.(*APIError)`
6. **No input mutation** - functions must copy slices/maps before modifying (see `applyCaching`)
7. **Lock-free network calls** - never hold a mutex during I/O (see `CachedTokenSource`)
8. **Shared utilities in internal/** - `parseDataURL` lives in `httpc`, not duplicated per provider

## Adding Providers

OpenAI-compatible providers use `internal/openaicompat`. Pattern:

```go
var _ provider.LanguageModel = (*chatModel)(nil)

func Chat(modelID string, opts ...Option) provider.LanguageModel { ... }
func (m *chatModel) DoGenerate(ctx, params) (*provider.GenerateResult, error) { ... }
func (m *chatModel) DoStream(ctx, params) (*provider.StreamResult, error) { ... }
```

Every provider must support: `WithAPIKey`, `WithTokenSource`, `WithBaseURL`, `WithHTTPClient`, `WithHeaders`.
