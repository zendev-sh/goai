# Contributing to GoAI

Contributions welcome. Here's how to get started.

## Getting Started

### Prerequisites

- Go 1.25+
- `golangci-lint` (for linting)

### Setup

```bash
git clone https://github.com/zendev-sh/goai.git
cd goai
go test ./...
```

### Project Structure

```
goai/
├── *.go                    # Core SDK (GenerateText, StreamText, etc.)
├── provider/
│   ├── provider.go         # LanguageModel, EmbeddingModel, ImageModel interfaces
│   ├── types.go            # Shared types (Message, Part, Usage, etc.)
│   ├── token.go            # TokenSource, CachedTokenSource
│   ├── openai/             # OpenAI provider
│   ├── anthropic/          # Anthropic provider
│   ├── google/             # Google Gemini provider
│   └── ...                 # Other providers
├── internal/
│   ├── openaicompat/       # Shared codec for OpenAI-compatible providers
│   ├── gemini/             # Schema sanitization (Vertex, Google)
│   ├── sse/                # SSE parser
│   └── httpc/              # HTTP utilities
└── examples/               # Usage examples
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feat/your-feature
```

### 2. Make Changes

- Write code and tests
- Run tests: `go test ./...`
- Run linter: `golangci-lint run`
- Build: `go build ./...`

### 3. Submit a PR

- Keep PRs focused - one feature or fix per PR
- Include tests for new functionality
- Update examples if the public API changes

## Code Style

- Standard `gofmt`/`goimports` formatting
- Follow existing patterns in the codebase
- Prefer Go idioms: small interfaces, composition, functional options
- Use `internal/` for implementation details that shouldn't be public API

## Adding a New Provider

Most providers are OpenAI-compatible. To add one:

1. Create `provider/<name>/<name>.go`
2. Use `internal/openaicompat` for request/response handling
3. Implement `Chat()` constructor returning `provider.LanguageModel`
4. Add tests with mock HTTP server (see existing providers for patterns)
5. Add an entry to the Provider Table in `README.md`

For non-OpenAI-compatible providers (like Anthropic, Google, Cohere), implement the full request/response mapping.

### Provider Checklist

- [ ] `Chat(modelID, ...Option) provider.LanguageModel`
- [ ] `WithAPIKey(key)` option
- [ ] `WithHTTPClient(c)` option
- [ ] `WithTokenSource(ts)` option
- [ ] Streaming support (`DoStream`)
- [ ] Error handling (`ParseHTTPError`)
- [ ] Per-request headers (`_headers` extraction in `doHTTP`)
- [ ] `var _ provider.LanguageModel = (*chatModel)(nil)` compile-time check
- [ ] Tests with mock HTTP server (90% coverage target)

## Testing

### Running Tests

```bash
# All tests
go test ./...

# Specific package
go test ./provider/openai/

# With verbose output
go test -v ./provider/anthropic/

# With coverage
go test -cover ./...
```

### Test Patterns

- Use `net/http/httptest` for mock HTTP servers
- Test both streaming and non-streaming paths
- Test error handling (4xx, 5xx, malformed responses)
- Test tool call parsing and finish reason mapping

## Reporting Issues

- Use [GitHub Issues](https://github.com/zendev-sh/goai/issues)
- Include: Go version, provider, minimal reproduction code
- For API errors, include the HTTP status code and error message (redact API keys)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
