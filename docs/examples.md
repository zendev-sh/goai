---
title: Examples
description: "Runnable GoAI code examples covering chat, streaming, structured output, tool calling, web search, code execution, and image generation."
---

# Examples

Runnable examples demonstrating GoAI features. Each example is a standalone `main.go` that you can run directly.

All examples live in [`goai/examples/`](https://github.com/zendev-sh/goai/tree/main/examples).

---

## Core

Basic text generation, streaming, and structured output.

### chat

Simple non-streaming text generation with `GenerateText`.

- **Provider:** Google Gemini
- **Features:** `GenerateText`, `WithSystem`, `WithPrompt`, token usage
- **Source:** [`examples/chat/`](https://github.com/zendev-sh/goai/tree/main/examples/chat)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run examples/chat/main.go
```

### streaming

Streaming text generation with real-time output via `TextStream()`.

- **Provider:** Google Gemini
- **Features:** `StreamText`, `TextStream()`, `Result()` for usage after streaming
- **Source:** [`examples/streaming/`](https://github.com/zendev-sh/goai/tree/main/examples/streaming)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run examples/streaming/main.go
```

### structured

Structured output with typed Go structs using `GenerateObject[T]` and `StreamObject[T]`.

- **Provider:** Google Gemini
- **Features:** `GenerateObject[T]`, `StreamObject[T]`, `PartialObjectStream()`, `SchemaFrom[T]`, struct tags (`jsonschema`)
- **Source:** [`examples/structured/`](https://github.com/zendev-sh/goai/tree/main/examples/structured)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run ./examples/structured/
```

### embedding

Text embeddings with `Embed` and `EmbedMany`, including cosine similarity comparison.

- **Provider:** Google Gemini (`text-embedding-004`)
- **Features:** `Embed`, `EmbedMany`, auto-chunking, pairwise cosine similarity
- **Source:** [`examples/embedding/`](https://github.com/zendev-sh/goai/tree/main/examples/embedding)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run ./examples/embedding/
```

### citations

Accessing `Sources` from grounded AI responses. The Sources API is provider-agnostic.

- **Provider:** Google Gemini
- **Features:** `TextResult.Sources`, per-step sources, `Source.URL`, `Source.Title`, text offsets
- **Source:** [`examples/citations/`](https://github.com/zendev-sh/goai/tree/main/examples/citations)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run examples/citations/main.go
```

---

## Tools

Custom tools, agent loops, and provider-defined tools.

### tools

Single-step tool call with a custom tool definition and `Execute` handler.

- **Provider:** Google Gemini
- **Features:** `goai.Tool`, `InputSchema` (JSON Schema), `Execute` function, `WithMaxSteps(2)`
- **Source:** [`examples/tools/`](https://github.com/zendev-sh/goai/tree/main/examples/tools)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run examples/tools/main.go
```

### agent-loop

Multi-step agent loop with multiple tools and step/tool-call callbacks.

- **Provider:** Google Gemini
- **Features:** Multiple tools, `WithMaxSteps(5)`, `WithOnStepFinish`, `WithOnToolCall`, step history
- **Source:** [`examples/agent-loop/`](https://github.com/zendev-sh/goai/tree/main/examples/agent-loop)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run examples/agent-loop/main.go
```

---

## Provider-Defined Tools

These examples use built-in tools provided by specific AI providers. The provider handles tool execution server-side.

### web-search

Web search across OpenAI, Anthropic, and Google using their respective provider-defined tools.

- **Providers:** OpenAI (direct + Azure), Anthropic (direct + Bedrock), Google
- **Features:** `openai.Tools.WebSearch()`, `anthropic.Tools.WebSearch()`, `google.Tools.GoogleSearch()`, source citations
- **Source:** [`examples/web-search/`](https://github.com/zendev-sh/goai/tree/main/examples/web-search)

```bash
export OPENAI_API_KEY=...
go run examples/web-search/main.go openai

export ANTHROPIC_API_KEY=...
go run examples/web-search/main.go anthropic

export GOOGLE_GENERATIVE_AI_API_KEY=...
go run examples/web-search/main.go google
```

### google-search

Google Search grounding with Gemini, returning grounded responses with source URLs.

- **Provider:** Google Gemini
- **Features:** `google.Tools.GoogleSearch()`, grounding sources
- **Source:** [`examples/google-search/`](https://github.com/zendev-sh/goai/tree/main/examples/google-search)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run examples/google-search/main.go
```

### web-fetch

Anthropic's web fetch tool for retrieving and processing content from specific URLs.

- **Providers:** Anthropic (direct + Bedrock)
- **Features:** `anthropic.Tools.WebFetch()`, citation support, URL content processing
- **Source:** [`examples/web-fetch/`](https://github.com/zendev-sh/goai/tree/main/examples/web-fetch)

```bash
export ANTHROPIC_API_KEY=...
go run examples/web-fetch/main.go anthropic
```

### computer-use

Anthropic's computer use tools: Computer (mouse/keyboard), Bash (shell), TextEditor (file editing).

- **Providers:** Anthropic (direct + Bedrock)
- **Features:** `anthropic.Tools.Computer()`, `anthropic.Tools.Bash()`, `anthropic.Tools.TextEditor()`, provider-defined tools with client-side `Execute` handlers
- **Source:** [`examples/computer-use/`](https://github.com/zendev-sh/goai/tree/main/examples/computer-use)

```bash
export ANTHROPIC_API_KEY=...
go run examples/computer-use/main.go anthropic
```

### code-execution

Anthropic's sandboxed Python code execution tool.

- **Providers:** Anthropic (direct + Bedrock)
- **Features:** `anthropic.Tools.CodeExecution()`, server-side Python execution
- **Source:** [`examples/code-execution/`](https://github.com/zendev-sh/goai/tree/main/examples/code-execution)

```bash
export ANTHROPIC_API_KEY=...
go run examples/code-execution/main.go anthropic
```

### code-interpreter

OpenAI's code interpreter tool for sandboxed Python execution.

- **Providers:** OpenAI (direct + Azure)
- **Features:** `openai.Tools.CodeInterpreter()`, server-side Python execution
- **Source:** [`examples/code-interpreter/`](https://github.com/zendev-sh/goai/tree/main/examples/code-interpreter)

```bash
export OPENAI_API_KEY=...
go run examples/code-interpreter/main.go openai
```

### google-code-execution

Google Gemini's sandboxed Python code execution tool.

- **Provider:** Google Gemini
- **Features:** `google.Tools.CodeExecution()`, server-side Python execution
- **Source:** [`examples/google-code-execution/`](https://github.com/zendev-sh/goai/tree/main/examples/google-code-execution)

```bash
export GOOGLE_GENERATIVE_AI_API_KEY=...
go run examples/google-code-execution/main.go
```

### image-generation

OpenAI's image generation tool via the Responses API. The LLM decides when to generate images during conversation.

- **Providers:** OpenAI (direct + Azure)
- **Features:** `openai.Tools.ImageGeneration()`, image quality/size options
- **Source:** [`examples/image-generation/`](https://github.com/zendev-sh/goai/tree/main/examples/image-generation)

```bash
export OPENAI_API_KEY=...
go run examples/image-generation/main.go openai
```

### file-search

OpenAI's file search tool for semantic/keyword search over vector stores.

- **Providers:** OpenAI (direct + Azure)
- **Features:** `openai.Tools.FileSearch()`, vector store integration
- **Source:** [`examples/file-search/`](https://github.com/zendev-sh/goai/tree/main/examples/file-search)

Requires a pre-created vector store with uploaded files.

```bash
export OPENAI_API_KEY=...
go run examples/file-search/main.go openai <vector-store-id>
```
