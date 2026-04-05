# GoAI SDK - AI Agent Skill

> Instructions for AI coding agents helping developers use the GoAI SDK.
> Load this file when a user's project imports `github.com/zendev-sh/goai`.

## Overview

GoAI is a Go SDK for AI applications. One unified API across 22+ LLM providers. Inspired by the Vercel AI SDK, adapted to Go idioms (generics, interfaces, channels).

- **Package**: `github.com/zendev-sh/goai`
- **Go version**: 1.25+
- **Dependencies**: stdlib + `golang.org/x/oauth2` for Vertex AI. Optional `observability/otel` submodule adds OTel SDK (separate go.mod, not pulled unless imported).
- **Docs**: https://goai.sh
- **GoDoc**: https://pkg.go.dev/github.com/zendev-sh/goai

## Quick Reference

```
go get github.com/zendev-sh/goai@latest
```

```go
import (
    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"     // or any provider
)
```

## Core API

### 7 Top-Level Functions

| Function                                      | Purpose                         | Returns                     |
| --------------------------------------------- | ------------------------------- | --------------------------- |
| `goai.GenerateText(ctx, model, opts...)`      | Non-streaming text generation   | `(*TextResult, error)`      |
| `goai.StreamText(ctx, model, opts...)`        | Streaming text via channels     | `(*TextStream, error)`      |
| `goai.GenerateObject[T](ctx, model, opts...)` | Typed structured output (JSON)  | `(*ObjectResult[T], error)` |
| `goai.StreamObject[T](ctx, model, opts...)`   | Streaming structured output     | `(*ObjectStream[T], error)` |
| `goai.Embed(ctx, model, text, opts...)`       | Single text embedding           | `(*EmbedResult, error)`     |
| `goai.EmbedMany(ctx, model, texts, opts...)`  | Batch embeddings (auto-chunked) | `(*EmbedManyResult, error)` |
| `goai.GenerateImage(ctx, model, imgOpts...)`  | Image generation                | `(*ImageResult, error)`     |

### Model Constructors

Each provider has `Chat()`, and optionally `Embedding()` and `Image()`:

```go
// Language models
openai.Chat("gpt-4o")
anthropic.Chat("claude-sonnet-4-20250514")
google.Chat("gemini-2.5-flash")
bedrock.Chat("anthropic.claude-sonnet-4-20250514-v1:0")
azure.Chat("gpt-4o", azure.WithEndpoint("https://my-resource.openai.azure.com"))
vertex.Chat("gemini-2.5-flash", vertex.WithProject("my-project"), vertex.WithLocation("us-central1"))
groq.Chat("llama-3.3-70b-versatile")
ollama.Chat("llama3.2")

// Embedding models
openai.Embedding("text-embedding-3-small")
google.Embedding("text-embedding-004")
cohere.Embedding("embed-english-v3.0")
ollama.Embedding("nomic-embed-text")

// Image models
openai.Image("gpt-image-1")
google.Image("imagen-4.0-generate-001")
```

### Auth - Auto-Resolved from Environment

Providers auto-read API keys from env vars. No explicit config needed:

| Provider  | Env Var                                                      |
| --------- | ------------------------------------------------------------ |
| OpenAI    | `OPENAI_API_KEY`                                             |
| Anthropic | `ANTHROPIC_API_KEY`                                          |
| Google    | `GOOGLE_GENERATIVE_AI_API_KEY` or `GEMINI_API_KEY`           |
| Bedrock   | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION` |
| Azure     | `AZURE_OPENAI_API_KEY`                                       |
| Vertex AI | Application Default Credentials (ADC)                        |
| xAI       | `XAI_API_KEY`                                                |
| Groq      | `GROQ_API_KEY`                                               |
| Cohere    | `COHERE_API_KEY`                                             |
| Mistral   | `MISTRAL_API_KEY`                                            |
| DeepSeek  | `DEEPSEEK_API_KEY`                                           |

Or set explicitly:

```go
model := openai.Chat("gpt-4o", openai.WithAPIKey("sk-..."))
```

Most providers support these options (Bedrock uses AWS credential options; Ollama requires no auth):

```go
provider.WithAPIKey(key)         // static API key
provider.WithTokenSource(ts)     // dynamic auth (OAuth, service accounts)
provider.WithBaseURL(url)        // override endpoint (Azure uses WithEndpoint)
provider.WithHeaders(h)          // custom HTTP headers
provider.WithHTTPClient(c)       // custom HTTP transport
```

---

## Patterns and Examples

### 1. Basic Text Generation

```go
result, err := goai.GenerateText(ctx, openai.Chat("gpt-4o"),
    goai.WithSystem("You are a helpful assistant."),
    goai.WithPrompt("What is Go?"),
)
if err != nil {
    return err
}
fmt.Println(result.Text)
```

### 2. Streaming

```go
stream, err := goai.StreamText(ctx, openai.Chat("gpt-4o"),
    goai.WithPrompt("Write a poem about Go."),
)
if err != nil {
    return err
}
// Option A: text-only channel
for text := range stream.TextStream() {
    fmt.Print(text)
}
// Option B: raw chunks (mutually exclusive with TextStream)
// for chunk := range stream.Stream() { ... }

// Always available after streaming completes:
result := stream.Result()
fmt.Printf("\nTokens: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
```

**Important**: `Stream()` and `TextStream()` are mutually exclusive. Only call one. `Result()` can always be called after either.

### 3. Structured Output (Generics)

```go
type Recipe struct {
    Name        string   `json:"name" jsonschema:"description=Recipe name"`
    Ingredients []string `json:"ingredients"`
    Steps       []string `json:"steps"`
    Difficulty  string   `json:"difficulty" jsonschema:"enum=easy|medium|hard"`
}

result, err := goai.GenerateObject[Recipe](ctx, openai.Chat("gpt-4o"),
    goai.WithPrompt("Chocolate chip cookies recipe"),
)
if err != nil {
    return err
}
fmt.Println(result.Object.Name)       // typed access
fmt.Println(result.Object.Difficulty)  // "easy", "medium", or "hard"
```

Schema is auto-generated from struct tags. Supported tags:

- `json:"name"` - property name
- `json:"-"` - exclude field
- `jsonschema:"description=..."` - adds description
- `jsonschema:"enum=a|b|c"` - restricts to enum values

Supported types: string, bool, int/uint (all sizes), float32/64, slices, maps (string keys), structs (embedded structs flattened), pointers (nullable).

### 4. Streaming Structured Output

```go
stream, err := goai.StreamObject[Recipe](ctx, model,
    goai.WithPrompt("Chocolate chip cookies recipe"),
)
if err != nil {
    return err
}
for partial := range stream.PartialObjectStream() {
    fmt.Printf("Name so far: %s\n", partial.Name)
}
result, err := stream.Result()
// result.Object is the final validated Recipe
```

### 5. Tool Calling

```go
weatherTool := goai.Tool{
    Name:        "get_weather",
    Description: "Get weather for a city.",
    InputSchema: json.RawMessage(`{
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }`),
    Execute: func(ctx context.Context, input json.RawMessage) (string, error) {
        var params struct {
            City string `json:"city"`
        }
        if err := json.Unmarshal(input, &params); err != nil {
            return "", err
        }
        return fmt.Sprintf("72F and sunny in %s", params.City), nil
    },
}

result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("What's the weather in Tokyo?"),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(3), // enable auto tool loop
)
```

**Key rules:**

- `WithMaxSteps(1)` (default) = no auto loop, tool calls returned in `result.ToolCalls`
- `WithMaxSteps(n)` where n > 1 = auto loop: generate -> execute tools -> re-generate -> repeat
- Tools without `Execute` are sent as definitions only (for manual handling)
- Tool errors are passed back to the model as `"error: ..."` messages

### 6. Conversation History

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithSystem("You are a helpful assistant."),
    goai.WithMessages(
        goai.UserMessage("Hi, my name is Alice."),
        goai.AssistantMessage("Hello Alice! How can I help you?"),
        goai.UserMessage("What's my name?"),
    ),
)
```

Message builders:

- `goai.SystemMessage(text)` - system message
- `goai.UserMessage(text)` - user message
- `goai.AssistantMessage(text)` - assistant message
- `goai.ToolMessage(toolCallID, toolName, output)` - tool result

### 7. Embeddings

```go
// Single embedding
model := openai.Embedding("text-embedding-3-small")
result, err := goai.Embed(ctx, model, "Hello world")
fmt.Printf("Dimensions: %d\n", len(result.Embedding))

// Batch embeddings (auto-chunked, parallel)
results, err := goai.EmbedMany(ctx, model, []string{"text1", "text2", "text3"},
    goai.WithMaxParallelCalls(4),
)
```

### 8. Image Generation

```go
result, err := goai.GenerateImage(ctx, openai.Image("gpt-image-1"),
    goai.WithImagePrompt("A gopher writing Go code"),
    goai.WithImageSize("1024x1024"),
    goai.WithImageCount(1),
)
// result.Images[0].Data = raw bytes, result.Images[0].MediaType = "image/png"
```

**Note**: `GenerateImage` uses `ImageOption` (not `Option`): `WithImagePrompt`, `WithImageCount`, `WithImageSize`, `WithAspectRatio`, `WithImageMaxRetries`, `WithImageTimeout`, `WithImageProviderOptions`.

### 9. Provider-Defined Tools (Server-Side)

Built-in tools executed by the provider, not your code:

```go
// OpenAI web search
result, err := goai.GenerateText(ctx, openai.Chat("gpt-4o"),
    goai.WithPrompt("Latest Go release?"),
    goai.WithTools(providerTool(openai.Tools.WebSearch())),
)

// Google Search grounding
result, err := goai.GenerateText(ctx, google.Chat("gemini-2.5-flash"),
    goai.WithPrompt("Latest news about Go"),
    goai.WithTools(providerTool(google.Tools.GoogleSearch())),
)

// Anthropic code execution
result, err := goai.GenerateText(ctx, anthropic.Chat("claude-sonnet-4-20250514"),
    goai.WithPrompt("Calculate fibonacci(10)"),
    goai.WithTools(providerTool(anthropic.Tools.CodeExecution())),
)

// Helper to convert provider.ToolDefinition → goai.Tool:
func providerTool(td provider.ToolDefinition) goai.Tool {
    return goai.Tool{
        Name: td.Name, ProviderDefinedType: td.ProviderDefinedType,
        ProviderDefinedOptions: td.ProviderDefinedOptions,
    }
}
```

Available provider tools:

- **OpenAI**: `WebSearch()`, `CodeInterpreter()`, `ImageGeneration()`, `FileSearch(opts...)`
- **Anthropic**: `WebSearch()`, `WebFetch()`, `Computer(opts)`, `Bash()`, `TextEditor()`, `CodeExecution()` (+ versioned variants)
- **Google**: `GoogleSearch()`, `URLContext()`, `CodeExecution()`
- **xAI**: `WebSearch()`, `XSearch()`
- **Groq**: `BrowserSearch()`

### 10. Observability Hooks

```go
result, err := goai.GenerateText(ctx, model,
    goai.WithPrompt("Hello"),
    goai.WithOnRequest(func(info goai.RequestInfo) {
        log.Printf("Calling %s with %d messages", info.Model, info.MessageCount)
    }),
    goai.WithOnResponse(func(info goai.ResponseInfo) {
        log.Printf("Response in %v, tokens: %d", info.Latency, info.Usage.TotalTokens)
    }),
    goai.WithOnStepFinish(func(step goai.StepResult) {
        log.Printf("Step %d: %s, tools: %d", step.Number, step.FinishReason, len(step.ToolCalls))
    }),
    goai.WithOnToolCall(func(info goai.ToolCallInfo) {
        log.Printf("Tool %s (step %d) took %v, input: %s", info.ToolName, info.Step, info.Duration, info.Input)
    }),
)
```

---

## Complete Options Reference

### Core Options (type `goai.Option`)

```go
goai.WithSystem(s string)                          // system prompt
goai.WithPrompt(s string)                          // single user message (shorthand)
goai.WithMessages(msgs ...provider.Message)        // conversation history
goai.WithPromptCaching(bool)                       // enable prompt caching

goai.WithTools(tools ...goai.Tool)                 // available tools
goai.WithMaxSteps(n int)                           // auto tool loop iterations (default: 1)
goai.WithToolChoice(tc string)                     // "auto" | "none" | "required" | "<tool_name>"

goai.WithMaxOutputTokens(n int)                    // response length limit
goai.WithTemperature(t float64)                    // randomness (0.0 - 2.0)
goai.WithTopP(p float64)                           // nucleus sampling
goai.WithTopK(k int)                               // top-K sampling
goai.WithFrequencyPenalty(p float64)               // frequency penalty
goai.WithPresencePenalty(p float64)                // presence penalty
goai.WithSeed(s int)                               // deterministic generation
goai.WithStopSequences(seqs ...string)             // stop sequences

goai.WithMaxRetries(n int)                         // retry count (default: 2)
goai.WithTimeout(d time.Duration)                  // request timeout
goai.WithHeaders(h map[string]string)              // additional HTTP headers
goai.WithProviderOptions(opts map[string]any)      // provider-specific params

goai.WithOnRequest(fn func(RequestInfo))           // before each API call
goai.WithOnResponse(fn func(ResponseInfo))         // after each API call
goai.WithOnStepFinish(fn func(StepResult))         // after each step
goai.WithOnToolCallStart(fn func(ToolCallStartInfo)) // before each tool execution
goai.WithOnToolCall(fn func(ToolCallInfo))         // after each tool execution

goai.WithExplicitSchema(schema json.RawMessage)    // override auto-generated schema
goai.WithSchemaName(name string)                   // schema name (default: "response")

goai.WithMaxParallelCalls(n int)                   // EmbedMany parallelism (default: 4)
goai.WithEmbeddingProviderOptions(opts map[string]any) // embedding-specific params
```

### Image Options (type `goai.ImageOption`)

```go
goai.WithImagePrompt(prompt string)                // text prompt
goai.WithImageCount(n int)                         // number of images (default: 1)
goai.WithImageSize(size string)                    // e.g. "1024x1024"
goai.WithAspectRatio(ratio string)                 // e.g. "16:9"
goai.WithImageMaxRetries(n int)                   // retries on 429/5xx
goai.WithImageTimeout(d time.Duration)            // overall timeout
goai.WithImageProviderOptions(opts map[string]any) // provider-specific params
```

---

## Result Types

### TextResult (from GenerateText / StreamText)

```go
type TextResult struct {
    Text             string                         // accumulated generated text
    ToolCalls        []provider.ToolCall            // tool calls from final step
    Steps            []StepResult                   // per-step results
    TotalUsage       provider.Usage                 // aggregated token usage
    FinishReason     provider.FinishReason          // "stop", "tool-calls", "length", etc.
    Response         provider.ResponseMetadata      // provider metadata (ID, Model)
    ProviderMetadata map[string]map[string]any      // provider-specific response data
    Sources          []provider.Source              // citations/references
}
```

### ObjectResult[T] (from GenerateObject / StreamObject)

```go
type ObjectResult[T any] struct {
    Object           T                              // the parsed typed object
    Usage            provider.Usage                 // total token consumption across all steps
    FinishReason     provider.FinishReason
    Response         provider.ResponseMetadata
    ProviderMetadata map[string]map[string]any      // provider-specific response data
    Steps            []StepResult                   // results from each generation step (multi-step tool loops)
}
```

### EmbedResult / EmbedManyResult

```go
type EmbedResult struct {
    Embedding []float64
    Usage     provider.Usage
}

type EmbedManyResult struct {
    Embeddings [][]float64
    Usage      provider.Usage
}
```

### ImageResult

```go
type ImageResult struct {
    Images []provider.ImageData  // .Data = []byte, .MediaType = "image/png"
}
```

### provider.Usage

```go
type Usage struct {
    InputTokens      int
    OutputTokens     int
    TotalTokens      int
    ReasoningTokens  int
    CacheReadTokens  int
    CacheWriteTokens int
}
```

---

## Error Handling

```go
import "errors"

result, err := goai.GenerateText(ctx, model, goai.WithPrompt("..."))
if err != nil {
    var apiErr *goai.APIError
    var overflowErr *goai.ContextOverflowError

    if errors.As(err, &overflowErr) {
        // Prompt too long for context window - truncate messages and retry
        log.Printf("Context overflow: %s", overflowErr.Message)
    } else if errors.As(err, &apiErr) {
        log.Printf("API error %d: %s (retryable: %v)", apiErr.StatusCode, apiErr.Message, apiErr.IsRetryable)
    } else {
        log.Printf("Error: %v", err)
    }
}
```

**Always use `errors.As()`, never type assertion.**

Built-in retry: transient errors (429, 5xx) are retried automatically up to `WithMaxRetries(2)` times with exponential backoff.

---

## TokenSource (Dynamic Auth)

For OAuth, service accounts, or rotating credentials:

```go
ts := provider.CachedTokenSource(func(ctx context.Context) (*provider.Token, error) {
    // Fetch token from your auth system
    token, err := myAuthClient.GetToken(ctx)
    if err != nil {
        return nil, err
    }
    return &provider.Token{
        Value:     token.AccessToken,
        ExpiresAt: token.ExpiresAt,
    }, nil
})

model := openai.Chat("gpt-4o", openai.WithTokenSource(ts))
```

- `provider.StaticToken(key)` - wraps a static string
- `provider.CachedTokenSource(fetchFn)` - caches until expiry, thread-safe

---

## Supported Providers

### Tier 1 (dedicated implementations)

| Provider      | Import               | Chat | Embed | Image | Provider Tools |
| ------------- | -------------------- | ---- | ----- | ----- | -------------- |
| OpenAI        | `provider/openai`    | Yes  | Yes   | Yes   | 4              |
| Anthropic     | `provider/anthropic` | Yes  | -     | -     | 10             |
| Google Gemini | `provider/google`    | Yes  | Yes   | Yes   | 3              |
| AWS Bedrock   | `provider/bedrock`   | Yes  | Yes   | -     | -              |
| Azure OpenAI  | `provider/azure`     | Yes  | -     | Yes   | -              |
| Vertex AI     | `provider/vertex`    | Yes  | Yes   | Yes   | -              |

### Tier 2

| Provider   | Import              | Chat | Embed | Provider Tools |
| ---------- | ------------------- | ---- | ----- | -------------- |
| Cohere     | `provider/cohere`   | Yes  | Yes   | -              |
| Mistral    | `provider/mistral`  | Yes  | -     | -              |
| xAI (Grok) | `provider/xai`      | Yes  | -     | 2              |
| Groq       | `provider/groq`     | Yes  | -     | 1              |
| DeepSeek   | `provider/deepseek` | Yes  | -     | -              |

### Tier 3 (OpenAI-compatible)

Fireworks, Together, DeepInfra, OpenRouter, Perplexity, Cerebras

### Local

| Provider | Import            | Chat | Embed | Default URL          |
| -------- | ----------------- | ---- | ----- | -------------------- |
| Ollama   | `provider/ollama` | Yes  | Yes   | `localhost:11434/v1` |
| vLLM     | `provider/vllm`   | Yes  | Yes   | `localhost:8000/v1`  |
| RunPod   | `provider/runpod` | Yes  | -     | `RUNPOD_ENDPOINT_ID` |
| Custom   | `provider/compat` | Yes  | Yes   | user-defined         |

---

## Common Mistakes to Avoid

1. **Don't use `WithPrompt` and `WithMessages` with user message together carelessly** - `WithPrompt("x")` is auto-wrapped as a `UserMessage` and prepended to `Messages`. Use one or the other for the initial user input.

2. **Don't call both `Stream()` and `TextStream()`** - they are mutually exclusive on `TextStream`. The second call returns a closed channel.

3. **Don't forget `WithMaxSteps` for tool loops** - default is 1 (no loop). Set `WithMaxSteps(n)` where n > 1 to enable automatic tool execution.

4. **Don't type-assert errors** - always use `errors.As(err, &target)`, never `err.(*goai.APIError)`.

5. **Don't mix `Option` and `ImageOption`** - `GenerateImage` uses `ImageOption`, all other functions use `Option`.

6. **Don't forget error handling on streaming** - always check the error from `StreamText`/`StreamObject` before consuming the stream.

7. **`Result()` on streams is blocking** - it waits for the entire stream to complete. Call it after consuming `TextStream()`/`Stream()`, or on its own if you don't need incremental output.
