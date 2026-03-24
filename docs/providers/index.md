---
title: Supported Providers
description: "GoAI SDK supports 20+ LLM providers: OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI, Groq, Mistral, Cohere, DeepSeek, Ollama, vLLM and more."
---

# Providers

GoAI SDK supports 20+ LLM providers with a single unified API. All providers auto-resolve credentials from environment variables. No explicit API key configuration needed for standard deployments.

All 7 core functions — `GenerateText`, `StreamText`, `GenerateObject[T]`, `StreamObject[T]`, `Embed`, `EmbedMany`, and `GenerateImage` — work identically across all providers.

## Tier 1

Dedicated implementations with extended API support.

| Provider                  | Chat                       | Embed                   | Image            | Provider Tools | Auth Env Var                          |
| ------------------------- | -------------------------- | ----------------------- | ---------------- | -------------- | ------------------------------------- |
| [OpenAI](openai.md)       | ✅ `gpt-4o`, `o3`          | ✅ `text-embedding-3-*` | ✅ `gpt-image-1` | 4 tools        | `OPENAI_API_KEY`                      |
| [Anthropic](anthropic.md) | ✅ `claude-*`              | —                       | —                | 10 tools       | `ANTHROPIC_API_KEY`                   |
| [Google](google.md)       | ✅ `gemini-*`              | ✅ `text-embedding-004` | ✅ `imagen-*`    | 3 tools        | `GOOGLE_GENERATIVE_AI_API_KEY` or `GEMINI_API_KEY` |
| [Bedrock](bedrock.md)     | ✅ `anthropic.*`, `meta.*` | —                       | —                | —              | `AWS_ACCESS_KEY_ID`                   |
| [Azure](azure.md)         | ✅ `gpt-4o`, `claude-*`    | —                       | ✅               | —              | `AZURE_OPENAI_API_KEY`                |
| [Vertex AI](vertex.md)    | ✅ `gemini-*`              | ✅                      | ✅               | —              | ADC (Application Default Credentials) |

## Tier 2

| Provider                | Chat                                    | Embed        | Image | Provider Tools | Auth Env Var       |
| ----------------------- | --------------------------------------- | ------------ | ----- | -------------- | ------------------ |
| [Cohere](cohere.md)     | ✅ `command-r-*`                        | ✅ `embed-*` | —     | —              | `COHERE_API_KEY`   |
| [Mistral](mistral.md)   | ✅ `mistral-large`, `magistral-*`       | —            | —     | —              | `MISTRAL_API_KEY`  |
| [xAI (Grok)](xai.md)    | ✅ `grok-*`                             | —            | —     | 2 tools        | `XAI_API_KEY`      |
| [Groq](groq.md)         | ✅ `llama-*`, `mixtral-*`               | —            | —     | 1 tool         | `GROQ_API_KEY`     |
| [DeepSeek](deepseek.md) | ✅ `deepseek-chat`, `deepseek-reasoner` | —            | —     | —              | `DEEPSEEK_API_KEY` |

## Tier 3

All use the shared `internal/openaicompat` codec.

| Provider                    | Endpoint            | Auth Env Var          |
| --------------------------- | ------------------- | --------------------- |
| [Fireworks](fireworks.md)   | `api.fireworks.ai`  | `FIREWORKS_API_KEY`   |
| [Together](together.md)     | `api.together.xyz`  | `TOGETHER_AI_API_KEY` |
| [DeepInfra](deepinfra.md)   | `api.deepinfra.com` | `DEEPINFRA_API_KEY`   |
| [OpenRouter](openrouter.md) | `openrouter.ai`     | `OPENROUTER_API_KEY`  |
| [Perplexity](perplexity.md) | `api.perplexity.ai` | `PERPLEXITY_API_KEY`  |
| [Cerebras](cerebras.md)     | `api.cerebras.ai`   | `CEREBRAS_API_KEY`    |

## Local / Custom

| Provider                        | Default Endpoint  | Auth          | Features                       |
| ------------------------------- | ----------------- | ------------- | ------------------------------ |
| [Ollama](ollama.md)             | `localhost:11434/v1` | None required | Embedding support              |
| [vLLM](vllm.md)                 | `localhost:8000/v1`  | Optional      | Embedding support              |
| [Generic Compatible](compat.md) | (required)        | Configurable  | Any OpenAI-compatible endpoint |

## Common Options

Most providers support these options (Bedrock uses AWS credential options; Azure uses `WithEndpoint`; Ollama requires no auth):

```go
openai.WithAPIKey(key)        // Static API key
openai.WithTokenSource(ts)    // Dynamic auth (OAuth, service accounts)
openai.WithBaseURL(url)       // Override endpoint
openai.WithHeaders(h)         // Custom HTTP headers
openai.WithHTTPClient(c)      // Custom HTTP transport
```

Each provider package exports its own `With*` functions with the same signatures.
