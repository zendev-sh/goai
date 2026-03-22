---
title: Installation
description: Install GoAI SDK with go get github.com/zendev-sh/goai. Requires Go 1.25+. Quick start guide with OpenAI, Anthropic, Google Gemini examples.
head:
  - - script
    - type: application/ld+json
    - |
      {"@context":"https://schema.org","@type":"FAQPage","mainEntity":[{"@type":"Question","name":"What is GoAI SDK?","acceptedAnswer":{"@type":"Answer","text":"GoAI SDK is an open-source Go library that provides one unified API for 20+ LLM providers including OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI, Groq, Mistral, Ollama and more. Inspired by the Vercel AI SDK, designed idiomatically for Go with generics and interfaces."}},{"@type":"Question","name":"How do I install GoAI SDK?","acceptedAnswer":{"@type":"Answer","text":"Run: go get github.com/zendev-sh/goai@latest. Requires Go 1.25 or later. API keys are auto-resolved from environment variables."}},{"@type":"Question","name":"How does GoAI SDK compare to LangChainGo?","acceptedAnswer":{"@type":"Answer","text":"GoAI SDK supports 20+ providers vs LangChainGo's fewer native integrations, uses Go generics for type-safe structured output (GenerateObject[T]), and has minimal dependencies (stdlib only, no heavy transitive deps). LangChainGo is a Python LangChain port; GoAI is designed from scratch for Go."}},{"@type":"Question","name":"Does GoAI SDK support streaming?","acceptedAnswer":{"@type":"Answer","text":"Yes. GoAI SDK supports real-time text streaming via StreamText and partial object streaming via StreamObject[T] using Go channels."}},{"@type":"Question","name":"Does GoAI SDK support tool calling?","acceptedAnswer":{"@type":"Answer","text":"Yes. Define tools with Execute handlers, set MaxSteps, and GoAI handles the auto tool loop automatically. Supports 20 provider-defined tools including web search, code execution, computer use, and file search."}}]}
---

# Installation

## Requirements

- Go 1.25 or later

## Install

```bash
go get github.com/zendev-sh/goai@latest
```

This installs the core SDK. Provider packages are included - no separate installs needed.

## Import

```go
import (
    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)
```

Each provider has its own sub-package under `provider/`. Import only the providers you use:

```go
import "github.com/zendev-sh/goai/provider/anthropic"
import "github.com/zendev-sh/goai/provider/google"
import "github.com/zendev-sh/goai/provider/bedrock"
```

## Verify

Create a file `main.go`:

```go
package main

import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

func main() {
    model := openai.Chat("gpt-4o")

    result, err := goai.GenerateText(context.Background(), model,
        goai.WithPrompt("Say hello in one sentence."),
    )
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Text)
}
```

Set your API key and run:

```bash
export OPENAI_API_KEY="sk-..."
go run main.go
```

If you see a response from the model, the installation is working.

## Dependencies

The only external dependency is `golang.org/x/oauth2`, used by the Vertex AI provider for Application Default Credentials. All other providers use the standard library.

## Frequently Asked Questions

### What Go version does GoAI SDK require?

GoAI SDK requires Go 1.25 or later due to its use of generics and latest standard library features.

### Do I need to configure API keys manually?

No. GoAI SDK auto-resolves API keys from environment variables. Set `OPENAI_API_KEY` in your environment and `openai.Chat("gpt-4o")` will use it automatically. No explicit `WithAPIKey` needed.

### Is GoAI SDK free for commercial use?

Yes. GoAI SDK is released under the MIT License, which allows free commercial use, modification, and distribution with no restrictions.

### How do I switch between LLM providers?

Change one line — the provider import and model initialization. All 7 core functions (`GenerateText`, `StreamText`, `GenerateObject`, `StreamObject`, `Embed`, `EmbedMany`, `GenerateImage`) work identically across all 20+ providers.

### Does GoAI SDK work with local models?

Yes. GoAI SDK supports Ollama and vLLM for local model serving. No API key required. You can also use the generic `compat` provider with any OpenAI-compatible endpoint.

### What makes GoAI different from LangChainGo?

GoAI SDK is designed from scratch for Go with generics and interfaces, while LangChainGo is a port of Python's LangChain. GoAI supports 20+ providers (vs ~10), uses Go generics for type-safe structured output (`GenerateObject[T]`), and has minimal dependencies (stdlib only). See the [comparison page](/compare) for details.
