<p align="center">
  <img src="goai.png" alt="GoAI" width="400">
</p>

<h1 align="center">GoAI</h1>

<p align="center"><em>AI SDK, the Go way.</em></p>
<p align="center">Go SDK for building AI applications. One SDK, 25+ providers, MCP support.</p>

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

> **v0.7.2** - New provider: [NVIDIA NIM](https://goai.sh/providers/nvidia) (OpenAI-compatible, chat + embeddings). E2E tested with `meta/llama-3.3-70b-instruct`. [Changelog →](https://github.com/zendev-sh/goai/releases)
>
> **v0.7.0** - New providers: [Cloudflare Workers AI](https://goai.sh/providers/cloudflare) and [FPT Smart Cloud](https://goai.sh/providers/fptcloud) (both OpenAI-compatible, chat + embeddings). [Changelog →](https://github.com/zendev-sh/goai/releases)
>
> **v0.6.0** - OpenTelemetry tracing + metrics, context propagation via RequestInfo.Ctx, Langfuse data race fix. [Changelog →](https://github.com/zendev-sh/goai/releases)
>
> **v0.5.8** - RunPod provider, Bedrock embeddings, and docs accuracy improvements. [Changelog →](https://github.com/zendev-sh/goai/releases)

## Features

- **7 core functions**: `GenerateText`, `StreamText`, `GenerateObject[T]`, `StreamObject[T]`, `Embed`, `EmbedMany`, `GenerateImage`
- **25+ providers**: OpenAI, Anthropic, Google, Bedrock, Azure, Vertex, Mistral, xAI, Groq, Cohere, DeepSeek, MiniMax, Fireworks, Together, DeepInfra, OpenRouter, Perplexity, Cerebras, Ollama, vLLM, RunPod, Cloudflare Workers AI, FPT Smart Cloud, NVIDIA NIM, + generic OpenAI-compatible
- **Auto tool loop**: Define tools with `Execute` handlers, set `MaxSteps` for `GenerateText` and `StreamText`
- **Structured output**: `GenerateObject[T]` auto-generates JSON Schema from Go types via reflection
- **Streaming**: Real-time text and partial object streaming via channels
- **Dynamic auth**: `TokenSource` interface for OAuth, rotating keys, cloud IAM, with `CachedTokenSource` for TTL-based caching
- **Prompt caching**: Automatic cache control for supported providers (Anthropic, Bedrock)
- **Citations/sources**: Grounding and inline citations from xAI, Perplexity, Google, OpenAI
- **Web search**: Built-in web search tools for OpenAI, Anthropic, Google, Groq, xAI. Model decides when to search
- **Code execution**: Server-side Python sandboxes via OpenAI, Anthropic, Google, xAI. No local setup
- **Computer use**: Anthropic computer, bash, text editor tools for autonomous desktop interaction
- **20 provider-defined tools**: Web fetch, file search, image generation, X search, and more - [full list](#provider-defined-tools)
- **MCP client**: Connect to any MCP server (stdio, HTTP, SSE), auto-convert tools for use with GoAI
- **Observability**: Built-in Langfuse and OpenTelemetry (OTel) integrations for tracing generations, tools, and multi-step loops
- **9 lifecycle hooks**: Observability (`OnRequest`, `OnResponse`, `OnToolCallStart`, `OnToolCall`, `OnStepFinish`, `OnFinish`) and interceptor (`OnBeforeToolExecute`, `OnAfterToolExecute`, `OnBeforeStep`) hooks for permission gates, secret scanning, output transformation, and loop control
- **Retry/backoff**: Automatic retry with exponential backoff on retryable HTTP errors (429/5xx)
- **Minimal dependencies**: Core depends on `golang.org/x/oauth2` + one indirect (`cloud.google.com/go/compute/metadata`). Optional `observability/otel` submodule uses separate `go.mod` with OTel SDK.

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

Most hosted providers auto-resolve API keys from environment variables. Local/custom providers may require explicit options:

```go
package main

import (
	"context"
	"fmt"
	"log"

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
ctx := context.Background()

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

Streaming with tools:

```go
import "github.com/zendev-sh/goai/provider"

stream, err := goai.StreamText(ctx, model,
	goai.WithPrompt("What's the weather in Tokyo?"),
	goai.WithTools(weatherTool),
	goai.WithMaxSteps(5),
)
for chunk := range stream.Stream() {
	switch chunk.Type {
	case provider.ChunkText:
		fmt.Print(chunk.Text)
	case provider.ChunkStepFinish:
		fmt.Println("\n[step complete]")
	}
}
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
import "encoding/json"

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

## Observability

Built-in [Langfuse](https://langfuse.com) and [OpenTelemetry](https://opentelemetry.io) integrations. Nine lifecycle hooks cover the full generation pipeline -- observability providers use them to trace LLM calls, tool executions, and multi-step agent loops:

```go
import "github.com/zendev-sh/goai/observability/langfuse"

// Credentials from env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Hello"),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
    langfuse.WithTracing(langfuse.TraceName("my-agent")),
)
```

Interceptor hooks let you control tool execution without modifying core code:

```go
// Permission gate: block dangerous tools
goai.WithOnBeforeToolExecute(func(info goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
    if info.ToolName == "delete_file" {
        return goai.BeforeToolExecuteResult{Skip: true, Result: "Permission denied."}
    }
    return goai.BeforeToolExecuteResult{}
}),

// Detect max-steps exhaustion
goai.WithOnFinish(func(info goai.FinishInfo) {
    if info.StepsExhausted {
        log.Printf("Loop exhausted after %d steps", info.TotalSteps)
    }
}),
```

See [examples/hooks](examples/hooks/), [examples/langfuse](examples/langfuse/), [examples/otel](examples/otel/), and the [observability docs](https://goai.sh/concepts/observability) for details.

## Providers

Many providers auto-resolve credentials from environment variables. Others (for example `ollama`, `vllm`, `compat`) use explicit options:

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

| Provider   | Chat                                                         | Embed                                                      | Image         | Auth                                                                                               | E2E  | Import                |
| ---------- | ------------------------------------------------------------ | ---------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------- | ---- | --------------------- |
| OpenAI     | `gpt-4o`, `o3`, `codex-*`                                    | `text-embedding-3-*`                                       | `gpt-image-1` | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, TokenSource                                                   | Full | `provider/openai`     |
| Anthropic  | `claude-*`                                                   | -                                                          | -             | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, TokenSource                                             | Full | `provider/anthropic`  |
| Google     | `gemini-*`                                                   | `text-embedding-004`                                       | `imagen-*`    | `GOOGLE_GENERATIVE_AI_API_KEY` / `GEMINI_API_KEY`, TokenSource                                     | Full | `provider/google`     |
| Bedrock    | `anthropic.*`, `meta.*`                                      | `titan-embed-*`, `cohere.embed-*`, `nova-2-*`, `marengo-*` | -             | AWS keys, `AWS_BEARER_TOKEN_BEDROCK`, `AWS_BEDROCK_BASE_URL`                                       | Full | `provider/bedrock`    |
| Vertex     | `gemini-*`                                                   | `text-embedding-004`                                       | `imagen-*`    | TokenSource, ADC, or `GOOGLE_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_GENERATIVE_AI_API_KEY` fallback | Unit | `provider/vertex`     |
| Azure      | `gpt-4o`, `claude-*`                                         | -                                                          | via Azure     | `AZURE_OPENAI_API_KEY`, TokenSource                                                                | Full | `provider/azure`      |
| OpenRouter | various                                                      | -                                                          | -             | `OPENROUTER_API_KEY`, TokenSource                                                                  | Unit | `provider/openrouter` |
| Mistral    | `mistral-large`, `magistral-*`                               | -                                                          | -             | `MISTRAL_API_KEY`, TokenSource                                                                     | Full | `provider/mistral`    |
| Groq       | `mixtral-*`, `llama-*`                                       | -                                                          | -             | `GROQ_API_KEY`, TokenSource                                                                        | Full | `provider/groq`       |
| xAI        | `grok-*`                                                     | -                                                          | -             | `XAI_API_KEY`, TokenSource                                                                         | Unit | `provider/xai`        |
| Cohere     | `command-r-*`                                                | `embed-*`                                                  | -             | `COHERE_API_KEY`, TokenSource                                                                      | Unit | `provider/cohere`     |
| DeepSeek   | `deepseek-*`                                                 | -                                                          | -             | `DEEPSEEK_API_KEY`, TokenSource                                                                    | Unit | `provider/deepseek`   |
| MiniMax    | `MiniMax-M2.7`, `MiniMax-M2.5`, `MiniMax-M2.1`, `MiniMax-M2` | -                                                          | -             | `MINIMAX_API_KEY`, `MINIMAX_BASE_URL`, TokenSource                                                 | Full | `provider/minimax`    |
| Fireworks  | various                                                      | -                                                          | -             | `FIREWORKS_API_KEY`, TokenSource                                                                   | Unit | `provider/fireworks`  |
| Together   | various                                                      | -                                                          | -             | `TOGETHER_AI_API_KEY` (or `TOGETHER_API_KEY`), TokenSource                                         | Unit | `provider/together`   |
| DeepInfra  | various                                                      | -                                                          | -             | `DEEPINFRA_API_KEY`, TokenSource                                                                   | Unit | `provider/deepinfra`  |
| Perplexity | `sonar-*`                                                    | -                                                          | -             | `PERPLEXITY_API_KEY`, TokenSource                                                                  | Unit | `provider/perplexity` |
| Cerebras   | `llama-*`                                                    | -                                                          | -             | `CEREBRAS_API_KEY`, TokenSource                                                                    | Unit | `provider/cerebras`   |
| Ollama     | local models                                                 | local models                                               | -             | none                                                                                               | Unit | `provider/ollama`     |
| vLLM       | local models                                                 | local models                                               | -             | Optional auth via `WithAPIKey` / `WithTokenSource`                                                 | Unit | `provider/vllm`       |
| RunPod     | any vLLM model                                               | -                                                          | -             | `RUNPOD_API_KEY`, TokenSource                                                                      | Unit | `provider/runpod`     |
| Cloudflare | `@cf/meta/*`, `@cf/openai/gpt-oss-*`, `@cf/qwen/*`           | `@cf/baai/bge-*`                                           | -             | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_BASE_URL`, TokenSource                | Unit | `provider/cloudflare` |
| FPT Cloud  | `Qwen3-*`, `Llama-*`, `gpt-oss-*`, `GLM-*`, `gemma-*`        | `bge-*`, `gte-*`, `multilingual-e5-*`                      | -             | `FPT_API_KEY`, `FPT_REGION` (`global`/`jp`), `FPT_BASE_URL`, TokenSource                           | Unit | `provider/fptcloud`   |
| NVIDIA NIM | `nvidia/llama-*`, `nvidia/nemotron-*`                        | `nvidia/nv-embed-*`                                        | -             | `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, TokenSource                                                   | Full | `provider/nvidia`     |
| Compat     | any OpenAI-compatible                                        | any                                                        | -             | configurable                                                                                       | Unit | `provider/compat`     |

**E2E column**: "Full" = tested with real API calls. "Unit" = tested with mock HTTP servers (100% coverage).

### Tested Models

<details>
<summary><strong>E2E tested - 104 models across 8 providers</strong> (real API calls, click to expand)</summary>

Last run: 2026-03-27. 104 models tested (generate + stream).

| Provider     | Models E2E tested (generate + stream)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Google (9)   | `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro` (stream), `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-3.1-pro-preview`, `gemini-2.0-flash`, `gemini-flash-latest`, `gemini-flash-lite-latest`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Azure (21)   | `claude-opus-4-6`, `claude-sonnet-4-6`, `DeepSeek-V3.2`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-5`, `gpt-5-codex`, `gpt-5-mini`, `gpt-5-pro`, `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-max`, `gpt-5.1-codex-mini`, `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5.4`, `gpt-5.4-pro`, `Kimi-K2.5`, `model-router`, `o3`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Bedrock (61) | **Anthropic**: `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-sonnet-4`, `claude-opus-4-6-v1`, `claude-opus-4-5`, `claude-opus-4-1`, `claude-haiku-4-5`, `claude-3-5-sonnet`, `claude-3-5-haiku`, `claude-3-haiku` · **Amazon**: `nova-micro`, `nova-lite`, `nova-pro`, `nova-premier`, `nova-2-lite` · **Meta**: `llama4-scout`, `llama4-maverick`, `llama3-3-70b`, `llama3-2-{90,11,3,1}b`, `llama3-1-{70,8}b`, `llama3-{70,8}b` · **Mistral**: `mistral-large`, `mixtral-8x7b`, `mistral-7b`, `ministral-3-{14,8}b`, `voxtral-{mini,small}` · **Others**: `deepseek.v3`, `deepseek.r1`, `ai21.jamba-1-5-{mini,large}`, `cohere.command-r{-plus,}`, `google.gemma-3-{4,12,27}b`, `minimax.{m2,m2.1}`, `moonshotai.kimi-k2{-thinking,.5}`, `nvidia.nemotron-nano-{12,9}b`, `openai.gpt-oss-{120,20}b{,-safeguard}`, `qwen.qwen3-{32,235,coder-30,coder-480}b`, `qwen.qwen3-next-80b`, `writer.palmyra-{x4,x5}`, `zai.glm-4.7{,-flash}` |
| Groq (2)     | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Mistral (5)  | `mistral-small-latest`, `mistral-large-latest`, `devstral-small-2507`, `codestral-latest`, `magistral-medium-latest`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Cerebras (1) | `llama3.1-8b`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| MiniMax (4)  | `MiniMax-M2.7`, `MiniMax-M2.5`, `MiniMax-M2.1`, `MiniMax-M2` (generate + stream + tools + thinking)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| NVIDIA (1)   | `meta/llama-3.3-70b-instruct`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

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
| NVIDIA     | `meta/llama-3.3-70b-instruct`                                                                      |
| Ollama     | `llama3`, `llama3.2:1b`, `nomic-embed-text`                                                        |
| vLLM       | `meta-llama/Llama-3-8b`                                                                            |
| RunPod     | `meta-llama/Llama-3.3-70B-Instruct`                                                                |
| Cloudflare | `@cf/meta/llama-3.1-8b-instruct`, `@cf/baai/bge-base-en-v1.5`                                      |
| FPT Cloud  | `Qwen3-32B`                                                                                        |

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

| Option                          | Description                              | Default          |
| ------------------------------- | ---------------------------------------- | ---------------- |
| `WithSystem(s)`                 | System prompt                            | -                |
| `WithPrompt(s)`                 | Single user message                      | -                |
| `WithMessages(...)`             | Conversation history                     | -                |
| `WithTools(...)`                | Available tools                          | -                |
| `WithMaxOutputTokens(n)`        | Response length limit                    | provider default |
| `WithTemperature(t)`            | Randomness (0.0-2.0)                     | provider default |
| `WithTopP(p)`                   | Nucleus sampling                         | provider default |
| `WithTopK(k)`                   | Top-K sampling                           | provider default |
| `WithFrequencyPenalty(p)`       | Frequency penalty                        | provider default |
| `WithPresencePenalty(p)`        | Presence penalty                         | provider default |
| `WithSeed(s)`                   | Deterministic generation                 | -                |
| `WithStopSequences(...)`        | Stop triggers                            | -                |
| `WithMaxSteps(n)`               | Tool loop iterations                     | 1 (no loop)      |
| `WithSequentialToolExecution()` | Execute tools one at a time              | parallel         |
| `WithMaxRetries(n)`             | Retries on 429/5xx                       | 2                |
| `WithTimeout(d)`                | Overall timeout                          | none             |
| `WithHeaders(h)`                | Per-request HTTP headers                 | -                |
| `WithProviderOptions(m)`        | Provider-specific params                 | -                |
| `WithPromptCaching(b)`          | Enable prompt caching                    | false            |
| `WithToolChoice(tc)`            | "auto", "none", "required", or tool name | -                |

### Lifecycle Hooks

| Option                        | Description                                                     |
| ----------------------------- | --------------------------------------------------------------- |
| `WithOnRequest(fn)`           | Called before each API call                                     |
| `WithOnResponse(fn)`          | Called after each API call                                      |
| `WithOnToolCallStart(fn)`     | Called before each tool execution begins                        |
| `WithOnToolCall(fn)`          | Called after each tool execution                                |
| `WithOnStepFinish(fn)`        | Called after each tool loop step                                |
| `WithOnFinish(fn)`            | Called once after all steps complete (carries `StepsExhausted`) |
| `WithOnBeforeToolExecute(fn)` | Intercept before tool Execute -- can skip, override ctx/input   |
| `WithOnAfterToolExecute(fn)`  | Intercept after tool Execute -- can modify output/error         |
| `WithOnBeforeStep(fn)`        | Intercept before step 2+ -- can inject messages or stop loop    |

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

GoAI generation and image APIs return typed errors for actionable failure modes (MCP client APIs return `*mcp.MCPError`):

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
		// HTTP API errors include ResponseBody and ResponseHeaders for debugging
	default:
		// Network error, context cancelled, etc.
	}
}
```

Error types:

| Type                   | Fields                                                                    | When                                |
| ---------------------- | ------------------------------------------------------------------------- | ----------------------------------- |
| `APIError`             | `StatusCode`, `Message`, `IsRetryable`, `ResponseBody`, `ResponseHeaders` | Non-2xx API responses               |
| `ContextOverflowError` | `Message`, `ResponseBody`                                                 | Prompt exceeds model context window |

Retry behavior: automatic exponential backoff on retryable HTTP errors (429/5xx, plus OpenAI 404 propagation). `retry-after-ms` and numeric `Retry-After` (seconds) are respected. Retries apply to request-level failures (including initial stream connection), not mid-stream error events.

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
- [streaming-tools](examples/streaming-tools/) - Streaming with multi-step tool loops
- [structured](examples/structured/) - Structured output with Go generics
- [tools](examples/tools/) - Single tool call
- [agent-loop](examples/agent-loop/) - Multi-step agent with callbacks
- [multi-turn](examples/multi-turn/) - Multi-turn conversation with ResponseMessages
- [citations](examples/citations/) - Accessing sources and citations
- [hooks](examples/hooks/) - Lifecycle hooks: permission gates, secret scanning, loop control, OnFinish
- [langfuse](examples/langfuse/) - Langfuse tracing integration
- [otel](examples/otel/) - OpenTelemetry tracing and metrics
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
│   ├── minimax/            # MiniMax (Anthropic-compatible API)
│   ├── compat/             # Generic OpenAI-compatible
│   	└── ...                 # 13 more OpenAI-compatible providers
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
