<p align="center">
  <img src="goai.png" alt="GoAI" width="400">
</p>

<h1 align="center">GoAI</h1>

<p align="center"><em>AI SDK, the Go way.</em></p>
<p align="center">Go SDK for building AI applications. One SDK, 20+ providers, MCP support.</p>

<p align="center">
  <a href="bench/RESULTS.md"><img src="https://img.shields.io/badge/streaming-1.1x_faster-brightgreen" alt="Streaming"></a>
  <a href="bench/RESULTS.md"><img src="https://img.shields.io/badge/cold_start-24x_faster-brightgreen" alt="Cold Start"></a>
  <a href="bench/RESULTS.md"><img src="https://img.shields.io/badge/memory-3.1x_less-brightgreen" alt="Memory"></a>
</p>

<p align="center"><strong>1.1x faster streaming</strong>, <strong>24x faster cold start</strong>, <strong>3.1x less memory</strong> vs Vercel AI SDK (<a href="bench/RESULTS.md">benchmarks</a>)</p>

<p align="center">
  <a href="https://goai.sh">Website</a> &middot;
  <a href="https://goai.sh/getting-started/installation">Docs</a> &middot;
  <a href="https://goai.sh/architecture">Architecture</a> &middot;
  <a href="https://goai.sh/providers/">Providers</a> &middot;
  <a href="https://goai.sh/examples">Examples</a>
</p>

---

Inspired by the [Vercel AI SDK](https://sdk.vercel.ai). The same clean abstractions, idiomatically adapted for Go with generics, interfaces, and functional options.

## What's New

> **v0.5.0**  - MCP (Model Context Protocol) client. Connect to any MCP server with 3 transports (stdio, HTTP, SSE). Auto-convert MCP tools for use with `GenerateText`. [Docs →](https://goai.sh/concepts/mcp)
>
> **v0.4.4**  - Provider-defined tools: 20 tools across Anthropic, OpenAI, Google, Groq, xAI. Computer use, web search, code execution. [Changelog →](https://github.com/zendev-sh/goai/releases)

## Features

- **7 core functions**: `GenerateText`, `StreamText`, `GenerateObject[T]`, `StreamObject[T]`, `Embed`, `EmbedMany`, `GenerateImage`
- **20+ providers**: OpenAI, Anthropic, Google, Bedrock, Azure, Vertex, Mistral, xAI, Groq, Cohere, DeepSeek, Fireworks, Together, DeepInfra, OpenRouter, Perplexity, Cerebras, Ollama, vLLM, + generic OpenAI-compatible
- **Auto tool loop**: Define tools with `Execute` handlers, set `MaxSteps`, GoAI handles the loop
- **Structured output**: `GenerateObject[T]` auto-generates JSON Schema from Go types via reflection
- **Streaming**: Real-time text and partial object streaming via channels
- **Dynamic auth**: `TokenSource` interface for OAuth, rotating keys, cloud IAM, with `CachedTokenSource` for TTL-based caching
- **Prompt caching**: Automatic cache control for supported providers (Anthropic, Bedrock)
- **Citations/sources**: Grounding and inline citations from xAI, Perplexity, Google, OpenAI
- **Web search**: Built-in web search tools for OpenAI, Anthropic, Google, Groq. Model decides when to search
- **Code execution**: Server-side Python sandboxes via OpenAI, Anthropic, Google. No local setup
- **Computer use**: Anthropic computer, bash, text editor tools for autonomous desktop interaction
- **20 provider-defined tools**: Web fetch, file search, image generation, X search, and more - [full list](#provider-defined-tools)
- **MCP client**: Connect to any MCP server (stdio, HTTP, SSE), auto-convert tools for use with GoAI
- **Telemetry hooks**: `OnRequest`, `OnResponse`, `OnStepFinish`, `OnToolCall` callbacks
- **Retry/backoff**: Automatic retry with exponential backoff on 429/5xx errors
- **Minimal dependencies**: Core uses only stdlib; Vertex adds `golang.org/x/oauth2` for ADC

## Performance vs Vercel AI SDK

| Metric               | GoAI   | Vercel AI SDK | Improvement |
| -------------------- | ------ | ------------- | ----------- |
| Streaming throughput | 1.46ms | 1.62ms        | 1.1x faster |
| Cold start           | 569us  | 13.9ms        | 24x faster  |
| Memory (1 stream)    | 220KB  | 676KB         | 3.1x less   |
| GenerateText         | 56us   | 79us          | 1.4x faster |

> Mock HTTP server, identical SSE fixtures, Apple M2. [Full report](bench/RESULTS.md)

## Install

```bash
go get github.com/zendev-sh/goai@latest
```

Requires Go 1.25+.

## Quick Start

Providers auto-resolve API keys from environment variables. No explicit `WithAPIKey` needed:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider/openai"
)

func main() {
	// Reads OPENAI_API_KEY from environment automatically.
	model := openai.Chat("gpt-4o")

	result, err := goai.GenerateText(context.Background(), model,
		goai.WithPrompt("What is the capital of France?"),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Text)
}
```

## Streaming

```go
stream, err := goai.StreamText(ctx, model,
	goai.WithSystem("You are a helpful assistant."),
	goai.WithPrompt("Write a haiku about Go."),
)
if err != nil {
	log.Fatal(err)
}

for text := range stream.TextStream() {
	fmt.Print(text)
}

result := stream.Result()
if err := stream.Err(); err != nil {
	log.Fatal(err)
}
fmt.Printf("\nTokens: %d in, %d out\n",
	result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
```

## Structured Output

Auto-generates JSON Schema from Go types. Works with OpenAI, Anthropic, and Google.

```go
type Recipe struct {
	Name        string   `json:"name" jsonschema:"description=Recipe name"`
	Ingredients []string `json:"ingredients"`
	Steps       []string `json:"steps"`
	Difficulty  string   `json:"difficulty" jsonschema:"enum=easy|medium|hard"`
}

result, err := goai.GenerateObject[Recipe](ctx, model,
	goai.WithPrompt("Give me a recipe for chocolate chip cookies"),
)
if err != nil {
	log.Fatal(err)
}
fmt.Printf("Recipe: %s (%s)\n", result.Object.Name, result.Object.Difficulty)
```

Streaming partial objects:

```go
stream, err := goai.StreamObject[Recipe](ctx, model,
	goai.WithPrompt("Give me a recipe for pancakes"),
)
if err != nil {
	log.Fatal(err)
}
for partial := range stream.PartialObjectStream() {
	fmt.Printf("\r%s (%d ingredients so far)", partial.Name, len(partial.Ingredients))
}
final, err := stream.Result()
```

## Tools

Define tools with JSON Schema and an `Execute` handler. Set `MaxSteps` to enable the auto tool loop.

```go
weatherTool := goai.Tool{
	Name:        "get_weather",
	Description: "Get the current weather for a city.",
	InputSchema: json.RawMessage(`{
		"type": "object",
		"properties": {"city": {"type": "string", "description": "City name"}},
		"required": ["city"]
	}`),
	Execute: func(ctx context.Context, input json.RawMessage) (string, error) {
		var args struct{ City string `json:"city"` }
		if err := json.Unmarshal(input, &args); err != nil {
			return "", err
		}
		return fmt.Sprintf("22°C and sunny in %s", args.City), nil
	},
}

result, err := goai.GenerateText(ctx, model,
	goai.WithPrompt("What's the weather in Tokyo?"),
	goai.WithTools(weatherTool),
	goai.WithMaxSteps(3),
)
if err != nil {
	log.Fatal(err)
}
fmt.Println(result.Text) // "It's 22°C and sunny in Tokyo."
```

## MCP (Model Context Protocol)

Connect to any MCP server and use its tools with GoAI. Supports stdio, Streamable HTTP, and legacy SSE transports.

```go
import "github.com/zendev-sh/goai/mcp"

// Connect to any MCP server
transport := mcp.NewStdioTransport("npx", []string{"-y", "@modelcontextprotocol/server-filesystem", "."})
client := mcp.NewClient("my-app", "1.0", mcp.WithTransport(transport))
_ = client.Connect(ctx)
defer client.Close()

// Use MCP tools with GoAI
tools, _ := client.ListTools(ctx, nil)
goaiTools := mcp.ConvertTools(client, tools.Tools)

result, _ := goai.GenerateText(ctx, model,
    goai.WithTools(goaiTools...),
    goai.WithPrompt("List files in the current directory"),
    goai.WithMaxSteps(5),
)
```

See [examples/mcp-tools](examples/mcp-tools/) and the [MCP documentation](https://goai.sh/concepts/mcp) for more.

## Citations / Sources

Providers that support grounding (Google, xAI, Perplexity) or inline citations (OpenAI) return sources:

```go
result, err := goai.GenerateText(ctx, model,
	goai.WithPrompt("What were the major news events today?"),
)
if err != nil {
	log.Fatal(err)
}

if len(result.Sources) > 0 {
	for _, s := range result.Sources {
		fmt.Printf("[%s] %s - %s\n", s.Type, s.Title, s.URL)
	}
}

// Sources are also available per-step in multi-step tool loops.
for _, step := range result.Steps {
	for _, s := range step.Sources {
		fmt.Printf("  Step source: %s\n", s.URL)
	}
}
```

## Computer Use

See [Provider-Defined Tools > Computer Use](#computer-use-1) and [examples/computer-use](examples/computer-use/) for Anthropic computer, bash, and text editor tools. Works with both Anthropic direct API and Bedrock.

## Embeddings

```go
ctx := context.Background()
model := openai.Embedding("text-embedding-3-small")

// Single
result, err := goai.Embed(ctx, model, "Hello world")
if err != nil {
	log.Fatal(err)
}
fmt.Printf("Dimensions: %d\n", len(result.Embedding))

// Batch (auto-chunked, parallel)
many, err := goai.EmbedMany(ctx, model, []string{"foo", "bar", "baz"},
	goai.WithMaxParallelCalls(4),
)
if err != nil {
	log.Fatal(err)
}
```

## Image Generation

```go
ctx := context.Background()
model := openai.Image("gpt-image-1")

result, err := goai.GenerateImage(ctx, model,
	goai.WithImagePrompt("A sunset over mountains, oil painting style"),
	goai.WithImageSize("1024x1024"),
)
if err != nil {
	log.Fatal(err)
}
os.WriteFile("sunset.png", result.Images[0].Data, 0644)
```

Also supported: Google Imagen (`google.Image("imagen-4.0-generate-001")`) and Vertex AI (`vertex.Image(...)`).

## Providers

Every provider auto-resolves credentials from environment variables. Override with options when needed:

```go
// Auto-resolved: reads OPENAI_API_KEY from env
model := openai.Chat("gpt-4o")

// Explicit key (overrides env)
model := openai.Chat("gpt-4o", openai.WithAPIKey("sk-..."))

// Cloud IAM auth (Vertex, Bedrock)
model := vertex.Chat("gemini-2.5-pro",
	vertex.WithProject("my-project"),
	vertex.WithLocation("us-central1"),
)

// AWS Bedrock (reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION from env)
model := bedrock.Chat("anthropic.claude-sonnet-4-6-v1:0")

// Local (Ollama, vLLM)
model := ollama.Chat("llama3", ollama.WithBaseURL("http://localhost:11434/v1"))

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
```

### Provider Table

| Provider   | Chat                           | Embed                | Image         | Auth                                        | E2E  | Import                |
| ---------- | ------------------------------ | -------------------- | ------------- | ------------------------------------------- | ---- | --------------------- |
| OpenAI     | `gpt-4o`, `o3`, `codex-*`      | `text-embedding-3-*` | `gpt-image-1` | `OPENAI_API_KEY`, `OPENAI_BASE_URL`         | Full | `provider/openai`     |
| Anthropic  | `claude-*`                     | -                    | -             | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`   | Full | `provider/anthropic`  |
| Google     | `gemini-*`                     | `text-embedding-004` | `imagen-*`    | `GEMINI_API_KEY` / `GOOGLE_GENERATIVE_AI_API_KEY` | Full | `provider/google`     |
| Bedrock    | `anthropic.*`, `meta.*`        | -                    | -             | AWS keys, `AWS_BEARER_TOKEN_BEDROCK`, `AWS_BEDROCK_BASE_URL` | Full | `provider/bedrock`    |
| Vertex     | `gemini-*`                     | `text-embedding-004` | `imagen-*`    | TokenSource, ADC                            | Unit | `provider/vertex`     |
| Azure      | `gpt-4o`, `claude-*`           | -                    | via Azure     | `AZURE_OPENAI_API_KEY`, TokenSource         | Full | `provider/azure`      |
| OpenRouter | various                        | -                    | -             | `OPENROUTER_API_KEY`, TokenSource           | Unit | `provider/openrouter` |
| Mistral    | `mistral-large`, `magistral-*` | -                    | -             | `MISTRAL_API_KEY`, TokenSource              | Full | `provider/mistral`    |
| Groq       | `mixtral-*`, `llama-*`         | -                    | -             | `GROQ_API_KEY`, TokenSource                 | Full | `provider/groq`       |
| xAI        | `grok-*`                       | -                    | -             | `XAI_API_KEY`, TokenSource                  | Unit | `provider/xai`        |
| Cohere     | `command-r-*`                  | `embed-*`            | -             | `COHERE_API_KEY`, TokenSource               | Unit | `provider/cohere`     |
| DeepSeek   | `deepseek-*`                   | -                    | -             | `DEEPSEEK_API_KEY`, TokenSource             | Unit | `provider/deepseek`   |
| Fireworks  | various                        | -                    | -             | `FIREWORKS_API_KEY`, TokenSource            | Unit | `provider/fireworks`  |
| Together   | various                        | -                    | -             | `TOGETHER_AI_API_KEY`, TokenSource          | Unit | `provider/together`   |
| DeepInfra  | various                        | -                    | -             | `DEEPINFRA_API_KEY`, TokenSource            | Unit | `provider/deepinfra`  |
| Perplexity | `sonar-*`                      | -                    | -             | `PERPLEXITY_API_KEY`, TokenSource           | Unit | `provider/perplexity` |
| Cerebras   | `llama-*`                      | -                    | -             | `CEREBRAS_API_KEY`, TokenSource             | Unit | `provider/cerebras`   |
| Ollama     | local models                   | local models         | -             | none                                        | Unit | `provider/ollama`     |
| vLLM       | local models                   | local models         | -             | `API_KEY`, TokenSource                      | Unit | `provider/vllm`       |
| Compat     | any OpenAI-compatible          | any                  | -             | configurable                                | Unit | `provider/compat`     |

**E2E column**: "Full" = tested with real API calls. "Unit" = tested with mock HTTP servers (100% coverage).

### Tested Models

<details>
<summary><strong>E2E tested - 99 models across 6 providers</strong> (real API calls, click to expand)</summary>

Last run: 2026-03-15. 99 models tested (generate + stream).

| Provider     | Models E2E tested (generate + stream)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Google (9)   | `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro` (stream), `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-3.1-pro-preview`, `gemini-2.0-flash`, `gemini-flash-latest`, `gemini-flash-lite-latest`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Azure (21)   | `claude-opus-4-6`, `claude-sonnet-4-6`, `DeepSeek-V3.2`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-5`, `gpt-5-codex`, `gpt-5-mini`, `gpt-5-pro`, `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-max`, `gpt-5.1-codex-mini`, `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5.4`, `gpt-5.4-pro`, `Kimi-K2.5`, `model-router`, `o3`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Bedrock (61) | **Anthropic**: `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-sonnet-4`, `claude-opus-4-6-v1`, `claude-opus-4-5`, `claude-opus-4-1`, `claude-haiku-4-5`, `claude-3-5-sonnet`, `claude-3-5-haiku`, `claude-3-haiku` · **Amazon**: `nova-micro`, `nova-lite`, `nova-pro`, `nova-premier`, `nova-2-lite` · **Meta**: `llama4-scout`, `llama4-maverick`, `llama3-3-70b`, `llama3-2-{90,11,3,1}b`, `llama3-1-{70,8}b`, `llama3-{70,8}b` · **Mistral**: `mistral-large`, `mixtral-8x7b`, `mistral-7b`, `ministral-3-{14,8}b`, `voxtral-{mini,small}` · **Others**: `deepseek.v3`, `deepseek.r1`, `ai21.jamba-1-5-{mini,large}`, `cohere.command-r{-plus,}`, `google.gemma-3-{4,12,27}b`, `minimax.{m2,m2.1}`, `moonshotai.kimi-k2{-thinking,.5}`, `nvidia.nemotron-nano-{12,9}b`, `openai.gpt-oss-{120,20}b{,-safeguard}`, `qwen.qwen3-{32,235,coder-30,coder-480}b`, `qwen.qwen3-next-80b`, `writer.palmyra-{x4,x5}`, `zai.glm-4.7{,-flash}` |
| Groq (2)     | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Mistral (5)  | `mistral-small-latest`, `mistral-large-latest`, `devstral-small-2507`, `codestral-latest`, `magistral-medium-latest`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Cerebras (1) | `llama3.1-8b`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

</details>

<details>
<summary><strong>Unit tested</strong> (mock HTTP server, 100% coverage, click to expand)</summary>

| Provider   | Models in unit tests                                                                               |
| ---------- | -------------------------------------------------------------------------------------------------- |
| OpenAI     | `gpt-4o`, `o3`, `text-embedding-3-small`, `dall-e-3`, `gpt-image-1`                                |
| Anthropic  | `claude-sonnet-4-20250514`, `claude-sonnet-4-5-20241022`, `claude-sonnet-4-6-20260310`             |
| Google     | `gemini-2.5-flash`, `gemini-2.5-flash-image`, `imagen-4.0-fast-generate-001`, `text-embedding-004` |
| Bedrock    | `us.anthropic.claude-sonnet-4-6`, `anthropic.claude-sonnet-4-20250514-v1:0`, `meta.llama3-70b`     |
| Azure      | `gpt-4o`, `gpt-5.2-chat`, `dall-e-3`, `claude-sonnet-4-6`                                          |
| Vertex     | `gemini-2.5-pro`, `imagen-3.0-generate-002`, `text-embedding-004`                                  |
| Cohere     | `command-r-plus`, `command-a-reasoning`, `embed-v4.0`                                              |
| Mistral    | `mistral-large-latest`                                                                             |
| Groq       | `llama-3.3-70b-versatile`                                                                          |
| xAI        | `grok-3`                                                                                           |
| DeepSeek   | `deepseek-chat`                                                                                    |
| DeepInfra  | `meta-llama/Llama-3.3-70B-Instruct`                                                                |
| Fireworks  | `accounts/fireworks/models/llama-v3p3-70b-instruct`                                                |
| OpenRouter | `anthropic/claude-sonnet-4`                                                                        |
| Perplexity | `sonar-pro`                                                                                        |
| Together   | `meta-llama/Llama-3.3-70B-Instruct-Turbo`                                                          |
| Cerebras   | `llama-3.3-70b`                                                                                    |
| Ollama     | `llama3`, `llama3.2:1b`, `nomic-embed-text`                                                        |
| vLLM       | `meta-llama/Llama-3-8b`                                                                            |

</details>

### Custom / Self-Hosted

Use the `compat` provider for any OpenAI-compatible endpoint:

```go
model := compat.Chat("my-model",
	compat.WithBaseURL("https://my-api.example.com/v1"),
	compat.WithAPIKey("..."),
)
```

### Dynamic Auth with TokenSource

For OAuth, rotating keys, or cloud IAM:

```go
ts := provider.CachedTokenSource(func(ctx context.Context) (*provider.Token, error) {
	tok, err := fetchOAuthToken(ctx)
	return &provider.Token{
		Value:     tok.AccessToken,
		ExpiresAt: tok.Expiry,
	}, err
})

model := openai.Chat("gpt-4o", openai.WithTokenSource(ts))
```

`CachedTokenSource` handles TTL-based caching (zero ExpiresAt = cache forever), thread-safe refresh without holding locks during network calls, and manual token invalidation via the `InvalidatingTokenSource` interface.

### AWS Bedrock

Native Converse API with SigV4 signing (no AWS SDK dependency). Supports cross-region inference fallback, extended thinking, and image/document input:

```go
model := bedrock.Chat("anthropic.claude-sonnet-4-6-v1:0",
	bedrock.WithRegion("us-west-2"),
	bedrock.WithReasoningConfig(bedrock.ReasoningConfig{
		Type:         bedrock.ReasoningEnabled,
		BudgetTokens: 4096,
	}),
)
```

Auto-resolves `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` from environment. Cross-region fallback retries with `us.` prefix on model ID mismatch errors.

### Azure OpenAI

Supports both OpenAI models (GPT, o-series) and Claude models (routed to Azure Anthropic endpoint automatically):

```go
// OpenAI models
model := azure.Chat("gpt-4o",
	azure.WithEndpoint("https://my-resource.openai.azure.com"),
)

// Claude models (auto-routed to Anthropic endpoint)
model := azure.Chat("claude-sonnet-4-6",
	azure.WithEndpoint("https://my-resource.openai.azure.com"),
)
```

Auto-resolves `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (or `AZURE_RESOURCE_NAME`) from environment.

### Response Metadata

Every result includes provider response metadata:

```go
result, _ := goai.GenerateText(ctx, model, goai.WithPrompt("Hello"))
fmt.Printf("Request ID: %s\n", result.Response.ID)
fmt.Printf("Model used: %s\n", result.Response.Model)
```

## Options Reference

### Generation Options

| Option                    | Description                              | Default          |
| ------------------------- | ---------------------------------------- | ---------------- |
| `WithSystem(s)`           | System prompt                            | -                |
| `WithPrompt(s)`           | Single user message                      | -                |
| `WithMessages(...)`       | Conversation history                     | -                |
| `WithTools(...)`          | Available tools                          | -                |
| `WithMaxOutputTokens(n)`  | Response length limit                    | provider default |
| `WithTemperature(t)`      | Randomness (0.0-2.0)                     | provider default |
| `WithTopP(p)`             | Nucleus sampling                         | provider default |
| `WithTopK(k)`             | Top-K sampling                           | provider default |
| `WithFrequencyPenalty(p)` | Frequency penalty                        | provider default |
| `WithPresencePenalty(p)`  | Presence penalty                         | provider default |
| `WithSeed(s)`             | Deterministic generation                 | -                |
| `WithStopSequences(...)`  | Stop triggers                            | -                |
| `WithMaxSteps(n)`         | Tool loop iterations                     | 1 (no loop)      |
| `WithMaxRetries(n)`       | Retries on 429/5xx                       | 2                |
| `WithTimeout(d)`          | Overall timeout                          | none             |
| `WithHeaders(h)`          | Per-request HTTP headers                 | -                |
| `WithProviderOptions(m)`  | Provider-specific params                 | -                |
| `WithPromptCaching(b)`    | Enable prompt caching                    | false            |
| `WithToolChoice(tc)`      | "auto", "none", "required", or tool name | -                |

### Telemetry Hooks

| Option                 | Description                      |
| ---------------------- | -------------------------------- |
| `WithOnRequest(fn)`    | Called before each API call      |
| `WithOnResponse(fn)`   | Called after each API call       |
| `WithOnStepFinish(fn)` | Called after each tool loop step |
| `WithOnToolCall(fn)`   | Called after each tool execution |

### Structured Output Options

| Option                  | Description                                   |
| ----------------------- | --------------------------------------------- |
| `WithExplicitSchema(s)` | Override auto-generated JSON Schema           |
| `WithSchemaName(n)`     | Schema name for provider (default "response") |

### Embedding Options

| Option                            | Description               | Default |
| --------------------------------- | ------------------------- | ------- |
| `WithMaxParallelCalls(n)`         | Batch parallelism         | 4       |
| `WithEmbeddingProviderOptions(m)` | Embedding provider params | -       |

### Image Options

| Option                        | Description                    |
| ----------------------------- | ------------------------------ |
| `WithImagePrompt(s)`          | Text description               |
| `WithImageCount(n)`           | Number of images               |
| `WithImageSize(s)`            | Dimensions (e.g., "1024x1024") |
| `WithAspectRatio(s)`          | Aspect ratio (e.g., "16:9")    |
| `WithImageMaxRetries(n)`      | Retries on 429/5xx             |
| `WithImageTimeout(d)`         | Overall timeout                |
| `WithImageProviderOptions(m)` | Image provider params          |

## Error Handling

GoAI returns typed errors for actionable failure modes:

```go
result, err := goai.GenerateText(ctx, model, goai.WithPrompt("..."))
if err != nil {
	var overflow *goai.ContextOverflowError
	var apiErr *goai.APIError
	switch {
	case errors.As(err, &overflow):
		// Prompt too long - truncate and retry
	case errors.As(err, &apiErr):
		if apiErr.IsRetryable {
			// 429 rate limit, 503 - already retried MaxRetries times
		}
		fmt.Printf("API error %d: %s\n", apiErr.StatusCode, apiErr.Message)
		// apiErr.ResponseBody and apiErr.ResponseHeaders available for debugging
	default:
		// Network error, context cancelled, etc.
	}
}
```

Error types:

| Type | Fields | When |
|------|--------|------|
| `APIError` | `StatusCode`, `Message`, `IsRetryable`, `ResponseBody`, `ResponseHeaders` | Non-2xx API responses |
| `ContextOverflowError` | `Message`, `ResponseBody` | Prompt exceeds model context window |

Retry behavior: automatic exponential backoff on 429/5xx. Respects `Retry-After` header. OpenAI 404 treated as retryable (model propagation delay).

## Provider-Defined Tools

Providers expose built-in tools that the model can invoke server-side. GoAI supports 20 provider-defined tools across 5 providers:

| Provider  | Tools                                                                                                                                                                  | Import               |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Anthropic | `Computer`, `Computer_20251124`, `Bash`, `TextEditor`, `TextEditor_20250728`, `WebSearch`, `WebSearch_20260209`, `WebFetch`, `CodeExecution`, `CodeExecution_20250825` | `provider/anthropic` |
| OpenAI    | `WebSearch`, `CodeInterpreter`, `FileSearch`, `ImageGeneration`                                                                                                        | `provider/openai`    |
| Google    | `GoogleSearch`, `URLContext`, `CodeExecution`                                                                                                                          | `provider/google`    |
| xAI       | `WebSearch`, `XSearch`                                                                                                                                                 | `provider/xai`       |
| Groq      | `BrowserSearch`                                                                                                                                                        | `provider/groq`      |

All tools follow the same pattern: create a definition with `<provider>.Tools.ToolName()` (e.g., `openai.Tools`, `anthropic.Tools`), then pass it as a `goai.Tool`:

```go
// Example: def := openai.Tools.WebSearch(openai.WithSearchContextSize("medium"))
def := <provider>.Tools.ToolName(options...)
result, _ := goai.GenerateText(ctx, model,
    goai.WithTools(goai.Tool{
        Name:                   def.Name,
        ProviderDefinedType:    def.ProviderDefinedType,
        ProviderDefinedOptions: def.ProviderDefinedOptions,
    }),
)
```

### Web Search

The model searches the web and returns grounded responses. Available from OpenAI, Anthropic, Google, and Groq.

```go
// OpenAI (via Responses API) - also works via Azure
def := openai.Tools.WebSearch(openai.WithSearchContextSize("medium"))

// Anthropic (via Messages API) - also works via Bedrock
def := anthropic.Tools.WebSearch(anthropic.WithMaxUses(5))

// Google (grounding with Google Search) - returns Sources
def := google.Tools.GoogleSearch()
// result.Sources contains grounding URLs from Google Search

// Groq (interactive browser search)
def := groq.Tools.BrowserSearch()
```

### Code Execution

The model writes and runs code in a sandboxed environment. Server-side, no local setup needed.

```go
// OpenAI Code Interpreter - Python sandbox via Responses API
def := openai.Tools.CodeInterpreter()

// Anthropic Code Execution - Python sandbox via Messages API
def := anthropic.Tools.CodeExecution() // v20260120, GA, no beta needed

// Google Code Execution - Python sandbox via Gemini API
def := google.Tools.CodeExecution()
```

### Web Fetch

Claude fetches and processes content from specific URLs directly.

```go
def := anthropic.Tools.WebFetch(
    anthropic.WithWebFetchMaxUses(3),
    anthropic.WithCitations(true),
)
```

### File Search

Semantic search over uploaded files in vector stores (OpenAI Responses API).

```go
def := openai.Tools.FileSearch(
    openai.WithVectorStoreIDs("vs_abc123"),
    openai.WithMaxNumResults(5),
)
```

### Image Generation

LLM generates images inline during conversation (different from `goai.GenerateImage()` which calls the Images API directly).

```go
def := openai.Tools.ImageGeneration(
    openai.WithImageQuality("low"),
    openai.WithImageSize("1024x1024"),
)
// On Azure, also set: azure.WithHeaders(map[string]string{
//     "x-ms-oai-image-generation-deployment": "gpt-image-1.5",
// })
```

### Computer Use

Anthropic computer, bash, and text editor tools for autonomous desktop interaction. Client-side execution required.

```go
computerDef := anthropic.Tools.Computer(anthropic.ComputerToolOptions{
    DisplayWidthPx: 1920, DisplayHeightPx: 1080,
})
bashDef := anthropic.Tools.Bash()
textEditorDef := anthropic.Tools.TextEditor()
// Wrap each with an Execute handler for client-side execution
```

### URL Context

Gemini fetches and processes web content from URLs in the prompt.

```go
def := google.Tools.URLContext()
```

See [examples/](examples/) for complete runnable examples of each tool.

## Examples

See the [examples/](examples/) directory:

- [chat](examples/chat/) - Non-streaming generation
- [streaming](examples/streaming/) - Real-time text streaming
- [structured](examples/structured/) - Structured output with Go generics
- [tools](examples/tools/) - Single tool call
- [agent-loop](examples/agent-loop/) - Multi-step agent with callbacks
- [citations](examples/citations/) - Accessing sources and citations
- [computer-use](examples/computer-use/) - Anthropic computer, bash, and text editor tools
- [embedding](examples/embedding/) - Embeddings with similarity search
- [web-search](examples/web-search/) - Web search across providers (OpenAI, Anthropic, Google)
- [web-fetch](examples/web-fetch/) - Anthropic web fetch tool
- [code-execution](examples/code-execution/) - Anthropic code execution tool
- [code-interpreter](examples/code-interpreter/) - OpenAI code interpreter tool
- [google-search](examples/google-search/) - Google Search grounding with Gemini
- [google-code-execution](examples/google-code-execution/) - Google Gemini code execution tool
- [file-search](examples/file-search/) - OpenAI file search tool
- [image-generation](examples/image-generation/) - OpenAI image generation via Responses API
- [mcp-tools](examples/mcp-tools/) - MCP tools with GoAI LLM integration
- [mcp-filesystem](examples/mcp-filesystem/) - Filesystem MCP server via stdio
- [mcp-github](examples/mcp-github/) - GitHub MCP server via stdio
- [mcp-playwright](examples/mcp-playwright/) - Playwright MCP server for browser automation
- [mcp-remote](examples/mcp-remote/) - MCP over Streamable HTTP transport
- [mcp-sse](examples/mcp-sse/) - MCP over legacy SSE transport
- [mcp-local](examples/mcp-local/) - MCP client basics (no LLM needed)

## Project Structure

```
goai/                       # Core SDK
├── provider/               # Provider interface + shared types
│   ├── provider.go         # LanguageModel, EmbeddingModel, ImageModel interfaces
│   ├── types.go            # Message, Part, Usage, StreamChunk, etc.
│   ├── token.go            # TokenSource, CachedTokenSource
│   ├── openai/             # OpenAI (Chat Completions + Responses API)
│   ├── anthropic/          # Anthropic (Messages API)
│   ├── google/             # Google Gemini (REST API)
│   ├── bedrock/            # AWS Bedrock (Converse API + SigV4 + EventStream)
│   ├── vertex/             # Google Vertex AI (OpenAI-compat)
│   ├── azure/              # Azure OpenAI
│   ├── cohere/             # Cohere (Chat v2 + Embed)
│   ├── compat/             # Generic OpenAI-compatible
│   └── ...                 # 12 more OpenAI-compatible providers
├── internal/
│   ├── openaicompat/       # Shared codec for 13 OpenAI-compat providers
│   ├── gemini/             # Schema sanitization (Vertex, Google)
│   ├── sse/                # SSE line parser
│   └── httpc/              # HTTP utilities
├── examples/               # Usage examples
└── bench/                  # Performance benchmarks (GoAI vs Vercel AI SDK)
    ├── fixtures/           # Shared SSE test fixtures
    ├── go/                 # Go benchmarks (go test -bench)
    ├── ts/                 # TypeScript benchmarks (Bun + Tinybench)
    ├── collect.sh          # Result aggregation → report
    └── Makefile            # make bench-all
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
