---
layout: home
title: GoAI SDK — Go SDK for AI Applications. One API, 25+ LLM Providers.
titleTemplate: false
description: "Open-source Go SDK for AI applications. One unified API for OpenAI, Anthropic, Google Gemini, AWS Bedrock, and 25+ LLM providers. Minimal dependencies."
---

<div class="vp-doc" style="max-width: 688px; margin: 0 auto; padding: 2rem 1.5rem;">

## What's New

> **v0.9.0** — Provider-neutral file upload and remote file references. `FileUploader` interface + `RemoteFileRef` for OpenAI, Anthropic, and Google Gemini. Compat providers map file parts to native OpenAI shapes. [Changelog →](https://github.com/zendev-sh/goai/releases)
>
> **v0.8.5** — New provider: [Requesty](/providers/requesty) (OpenAI-compatible LLM gateway). Anthropic native structured output, per-step `Reasoning`, Gemini schema sanitization. [Changelog →](https://github.com/zendev-sh/goai/releases)
>
> **v0.8.0** — MCP OAuth 2.1 + PKCE, `NewTool[In]` typed constructor, lifecycle hook panics surfaced as `*PanicError`. [Changelog →](https://github.com/zendev-sh/goai/releases)
>
> **v0.7.0** — New providers: [Cloudflare Workers AI](/providers/cloudflare) and [FPT Smart Cloud](/providers/fptcloud). [Changelog →](https://github.com/zendev-sh/goai/releases)

## What is GoAI SDK?

GoAI SDK is an open-source Go library for building AI applications. It provides one unified API across 25+ LLM providers — OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI, Groq, Mistral, Cohere, DeepSeek, Ollama, vLLM, NVIDIA NIM, and more.

Inspired by the [Vercel AI SDK](https://sdk.vercel.ai), GoAI is designed idiomatically for Go with generics, interfaces, and channels.

## Core Features

- **GenerateText** — non-streaming text generation across all providers
- **StreamText** — real-time streaming with auto tool loops via Go channels
- **GenerateObject[T]** — type-safe structured output using Go generics
- **StreamObject[T]** — partial object streaming with typed results
- **Embed / EmbedMany** — text embeddings with auto-chunking
- **GenerateImage** — image generation (OpenAI, Google, Azure, Vertex AI)
- **Tool Calling** — custom tools with auto tool loops (`MaxSteps`)
- **20 Provider-Defined Tools** — web search, code execution, computer use, file search
- **[MCP Client](/concepts/mcp)** — connect to any MCP server (stdio, HTTP, SSE), auto-convert tools for GoAI
- **Prompt Caching** — automatic cache control for Anthropic and OpenAI
- **[Observability](/concepts/observability)** — built-in Langfuse and OpenTelemetry integrations for tracing generations, tools, and multi-step loops
- **Multi-agent orchestration** - for declarative YAML workflows on top of GoAI, see [zenflow](https://zenflow.sh)

## Why GoAI?

- **One API, 25+ providers** — switch providers by changing one line of code
- **Minimal deps** — core requires only `golang.org/x/oauth2`; optional OTel integration in separate submodule
- **Go-native design** — generics for type safety, channels for streaming, interfaces for extensibility
- **24x faster cold start** than Vercel AI SDK (569μs vs 13.9ms)
- **3.1x less memory** per request (220KB vs 676KB)

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

func main() {
    result, _ := goai.GenerateText(context.Background(),
        openai.Chat("gpt-4o"),
        goai.WithPrompt("Explain Go interfaces in one sentence."),
    )
    fmt.Println(result.Text)
}
```

## MCP Support <Badge type="tip" text="v0.5.0" />

Connect to any [MCP server](https://modelcontextprotocol.io) and use its tools with GoAI:

```go
transport := mcp.NewStdioTransport("npx", []string{"-y", "@modelcontextprotocol/server-github"})
client := mcp.NewClient("my-app", "1.0", mcp.WithTransport(transport))
_ = client.Connect(ctx)
defer client.Close()

tools, _ := client.ListTools(ctx, nil)
goaiTools := mcp.ConvertTools(client, tools.Tools)

result, _ := goai.GenerateText(ctx, model,
    goai.WithTools(goaiTools...),
    goai.WithPrompt("Search for popular Go repositories"),
    goai.WithMaxSteps(5),
)
```

3 transports (stdio, HTTP, SSE), tools/prompts/resources, pagination, notifications. [Learn more →](/concepts/mcp)

## Companion Projects

**[zenflow](https://zenflow.sh)** - multi-agent orchestration engine for Go. Declarative YAML workflows, typed inter-agent mailboxes, single static binary. Built on GoAI for the LLM layer; reach for it when one `MaxSteps` loop is no longer enough.

## Supported Providers

OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI, Vertex AI, Cohere, Mistral, xAI (Grok), Groq, DeepSeek, Fireworks, Together AI, DeepInfra, OpenRouter, Perplexity, Cerebras, Cloudflare Workers AI, FPT Smart Cloud, NVIDIA NIM, Ollama, vLLM, RunPod, and any OpenAI-compatible endpoint.

[View all providers →](/providers/)

[Explore the architecture →](/architecture)

[Compare with other Go AI libraries →](/compare)

</div>
