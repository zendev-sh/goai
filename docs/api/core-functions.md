---
title: Core Functions
description: "API reference for GoAI's core functions including GenerateText, StreamText, GenerateObject, StreamObject, Embed, EmbedMany, and GenerateImage."
---

# Core Functions

This page documents every public function in the `goai` package.

Import path: `github.com/zendev-sh/goai`

---

## GenerateText

Performs a non-streaming text generation. When tools with `Execute` functions are provided and `MaxSteps > 1`, it automatically runs a tool loop: generate, execute tools, re-generate.

```go
func GenerateText(ctx context.Context, model provider.LanguageModel, opts ...Option) (*TextResult, error)
```

**Parameters:**

| Name    | Type                     | Description                                                   |
| ------- | ------------------------ | ------------------------------------------------------------- |
| `ctx`   | `context.Context`        | Request context for cancellation and deadlines.               |
| `model` | `provider.LanguageModel` | The language model to use.                                    |
| `opts`  | `...Option`              | Configuration options (system prompt, messages, tools, etc.). |

**Returns:** `*TextResult` containing generated text, tool calls, step history, and usage. Returns an error on API failure or context cancellation.

**Behavior:**

1. Builds request parameters from the provided options.
2. Calls `model.DoGenerate` with automatic retry on transient errors.
3. If the model returns tool calls and executable tools are registered, executes them and appends results to the conversation.
4. Repeats up to `MaxSteps` times (default 1 = single model step; if that step requests executable tools, they can still be executed before returning).
5. Fires telemetry hooks (`OnRequest`, `OnResponse`, `OnStepFinish`, `OnToolCall`) at appropriate points.

**Example:**

```go
model := google.Chat("gemini-2.0-flash") // auto-reads GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY

result, err := goai.GenerateText(context.Background(), model,
    goai.WithSystem("You are a helpful assistant. Be concise."),
    goai.WithPrompt("What is the capital of France?"),
)
if err != nil {
    log.Fatal(err)
}

fmt.Println(result.Text)
fmt.Printf("Tokens: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
```

---

## StreamText

Performs a streaming text generation, returning a `TextStream` for incremental consumption.

```go
func StreamText(ctx context.Context, model provider.LanguageModel, opts ...Option) (*TextStream, error)
```

**Parameters:** Same as `GenerateText`.

**Returns:** `*TextStream` for consuming the response incrementally. Returns an error if the initial API call fails.

**Tool Loop Support:**

When `WithMaxSteps` is set > 1 and tools with `Execute` functions are provided, `StreamText` runs an automatic tool loop. All steps stream through a single unified channel. Step boundaries are marked by `ChunkStepFinish` chunks. The initial `DoStream` failure returns `(nil, error)`. Subsequent step errors flow through the stream as `ChunkError` chunks — check `stream.Err()` after consuming.

**Example (basic):**

```go
stream, err := goai.StreamText(context.Background(), model,
    goai.WithPrompt("Write a haiku about Go programming."),
)
if err != nil {
    log.Fatal(err)
}

for text := range stream.TextStream() {
    fmt.Print(text)
}
fmt.Println()

result := stream.Result()
fmt.Printf("Tokens: %d in, %d out\n", result.TotalUsage.InputTokens, result.TotalUsage.OutputTokens)
```

**Example (streaming with tools):**

```go
stream, err := goai.StreamText(ctx, model,
    goai.WithPrompt("What's the weather in Tokyo?"),
    goai.WithTools(weatherTool),
    goai.WithMaxSteps(5),
)
if err != nil {
    log.Fatal(err)
}

for chunk := range stream.Stream() {
    switch chunk.Type {
    case provider.ChunkText:
        fmt.Print(chunk.Text)
    case provider.ChunkStepFinish:
        fmt.Println("\n[step complete]")
    }
}

if err := stream.Err(); err != nil {
    log.Fatal("stream error:", err)
}
```

### TextStream

`TextStream` provides three consumption modes. `Stream()` and `TextStream()` are mutually exclusive - only call one. `Result()` can always be called, including after a streaming method, to get the accumulated final result.

#### TextStream.Stream

Returns a channel that emits raw `provider.StreamChunk` values from the provider.

```go
func (ts *TextStream) Stream() <-chan provider.StreamChunk
```

Use this when you need access to all chunk types (text, tool calls, reasoning, finish events).

#### TextStream.TextStream

Returns a channel that emits only text content strings, filtering out non-text chunks.

```go
func (ts *TextStream) TextStream() <-chan string
```

Use this when you only need the generated text tokens.

#### TextStream.Result

Blocks until the stream completes and returns the final accumulated result.

```go
func (ts *TextStream) Result() *TextResult
```

Can be called after `Stream()` or `TextStream()` to get usage, finish reason, and the complete text. Can also be called without any streaming method to silently consume the stream and return the result.

#### TextStream.Err

Returns the first error encountered during streaming, or nil if the stream completed successfully. Follows the `bufio.Scanner.Err()` pattern: must be called after the stream is fully consumed (after `Result()`, or after the `Stream()`/`TextStream()` channel is drained). Blocks until the stream is done.

```go
func (ts *TextStream) Err() error
```

---

## GenerateObject

Performs a non-streaming structured output generation. The type parameter `T` determines the output schema.

```go
func GenerateObject[T any](ctx context.Context, model provider.LanguageModel, opts ...Option) (*ObjectResult[T], error)
```

**Parameters:** Same as `GenerateText`, plus structured output options (`WithExplicitSchema`, `WithSchemaName`).

**Returns:** `*ObjectResult[T]` containing the parsed object, usage, and metadata. Returns an error on API failure, JSON parse failure, or if `MaxSteps` is exhausted before a stop step occurs.

**Behavior:**

1. Auto-generates a JSON Schema from `T` using `SchemaFrom[T]()`, unless overridden with `WithExplicitSchema`.
2. Sets `ResponseFormat` on the provider request to enable native JSON mode.
3. Parses the model's JSON response into the target type `T`.

When tools with `Execute` functions are provided and `MaxSteps > 1`, `GenerateObject` runs a tool loop identical to `GenerateText`. `ResponseFormat` is set on every step, and the model decides when to call tools vs produce the final JSON output. Structured output is parsed from the step that returns `finishReason` `"stop"`.

**Example:**

```go
type Recipe struct {
    Name        string   `json:"name" jsonschema:"description=Recipe name"`
    Ingredients []string `json:"ingredients" jsonschema:"description=List of ingredients"`
    Steps       []string `json:"steps" jsonschema:"description=Cooking steps"`
    Difficulty  string   `json:"difficulty" jsonschema:"enum=easy|medium|hard"`
}

result, err := goai.GenerateObject[Recipe](ctx, model,
    goai.WithPrompt("Give me a recipe for chocolate chip cookies"),
)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Recipe: %s\n", result.Object.Name)
fmt.Printf("Difficulty: %s\n", result.Object.Difficulty)
```

---

## StreamObject

Performs a streaming structured output generation, returning partial objects as JSON accumulates.

```go
func StreamObject[T any](ctx context.Context, model provider.LanguageModel, opts ...Option) (*ObjectStream[T], error)
```

**Parameters:** Same as `GenerateObject[T]`.

**Returns:** `*ObjectStream[T]` for consuming partial and final results. Returns an error if the initial API call fails.

> **Note:** Unlike `GenerateObject`, `StreamObject` is intentionally single-step — it does not support tool loops. Use `GenerateObject` when you need tools and multi-step behaviour.

### ObjectStream.PartialObjectStream

Returns a channel that emits progressively populated partial objects as JSON tokens arrive.

```go
func (os *ObjectStream[T]) PartialObjectStream() <-chan *T
```

Each emitted value has more fields populated than the previous one. Fields not yet received have their zero values.

### ObjectStream.Result

Blocks until the stream completes and returns the final validated object.

```go
func (os *ObjectStream[T]) Result() (*ObjectResult[T], error)
```

Returns an error if JSON parsing of the accumulated text fails.

### ObjectStream.Err

Returns the first error encountered during streaming, or nil if the stream completed successfully. Follows the `bufio.Scanner.Err()` pattern: must be called after the stream is fully consumed (after `Result()`, or after the `PartialObjectStream()` channel is drained). Blocks until the stream is done.

```go
func (os *ObjectStream[T]) Err() error
```

**Example:**

```go
stream, err := goai.StreamObject[Recipe](ctx, model,
    goai.WithPrompt("Give me a recipe for pancakes"),
)
if err != nil {
    log.Fatal(err)
}

for partial := range stream.PartialObjectStream() {
    if partial.Name != "" {
        fmt.Printf("\rStreaming: %s (%d ingredients)", partial.Name, len(partial.Ingredients))
    }
}
fmt.Println()

final, err := stream.Result()
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Usage: %d input, %d output tokens\n", final.Usage.InputTokens, final.Usage.OutputTokens)
```

---

## Embed

Generates an embedding vector for a single text value.

```go
func Embed(ctx context.Context, model provider.EmbeddingModel, value string, opts ...Option) (*EmbedResult, error)
```

**Parameters:**

| Name    | Type                      | Description                                                                |
| ------- | ------------------------- | -------------------------------------------------------------------------- |
| `ctx`   | `context.Context`         | Request context.                                                           |
| `model` | `provider.EmbeddingModel` | The embedding model to use.                                                |
| `value` | `string`                  | The text to embed.                                                         |
| `opts`  | `...Option`               | Options (`WithTimeout`, `WithMaxRetries`, `WithEmbeddingProviderOptions`). |

**Returns:** `*EmbedResult` containing the embedding vector, usage, and provider response metadata (`Response.Model` when available).

**Example:**

```go
model := google.Embedding("text-embedding-004", google.WithAPIKey(apiKey))

result, err := goai.Embed(ctx, model, "The quick brown fox jumps over the lazy dog")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Dimensions: %d\n", len(result.Embedding))
```

---

## EmbedMany

Generates embedding vectors for multiple text values. Auto-chunks when values exceed the model's `MaxValuesPerCall` limit and processes chunks in parallel.

```go
func EmbedMany(ctx context.Context, model provider.EmbeddingModel, values []string, opts ...Option) (*EmbedManyResult, error)
```

**Parameters:**

| Name     | Type                      | Description                                                                                        |
| -------- | ------------------------- | -------------------------------------------------------------------------------------------------- |
| `ctx`    | `context.Context`         | Request context.                                                                                   |
| `model`  | `provider.EmbeddingModel` | The embedding model to use.                                                                        |
| `values` | `[]string`                | The texts to embed.                                                                                |
| `opts`   | `...Option`               | Options (`WithMaxParallelCalls`, `WithTimeout`, `WithMaxRetries`, `WithEmbeddingProviderOptions`). |

**Returns:** `*EmbedManyResult` containing one embedding vector per input value, aggregated usage, and provider response metadata (`Response.Model` when available). When `EmbedMany` auto-chunks a large batch, `ProviderMetadata` is merged from all chunks using last-write-wins semantics per namespace key; `Response` is taken from the first chunk.

**Example:**

```go
texts := []string{
    "I love programming",
    "Software engineering is great",
    "The weather is nice today",
}

result, err := goai.EmbedMany(ctx, model, texts)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Generated %d embeddings\n", len(result.Embeddings))
```

---

## GenerateImage

Generates images from a text prompt.

```go
func GenerateImage(ctx context.Context, model provider.ImageModel, opts ...ImageOption) (*ImageResult, error)
```

**Parameters:**

| Name    | Type                  | Description                                                                       |
| ------- | --------------------- | --------------------------------------------------------------------------------- |
| `ctx`   | `context.Context`     | Request context.                                                                  |
| `model` | `provider.ImageModel` | The image model to use.                                                           |
| `opts`  | `...ImageOption`      | Image-specific options (prompt, count, size, aspect ratio, max retries, timeout). |

Note: `GenerateImage` uses `ImageOption` instead of `Option`. See the [Options](options.md) page for image-specific options.

**Returns:** `*ImageResult` containing the generated images as byte data.

**Example:**

```go
result, err := goai.GenerateImage(ctx, model,
    goai.WithImagePrompt("A sunset over mountains, oil painting style"),
    goai.WithImageSize("1024x1024"),
    goai.WithImageCount(1),
)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Generated %d image(s)\n", len(result.Images))
```

---

## Message Builders

Convenience functions for constructing `provider.Message` values.

### SystemMessage

```go
func SystemMessage(text string) provider.Message
```

Creates a system message with text content (role: `system`).

### UserMessage

```go
func UserMessage(text string) provider.Message
```

Creates a user message with text content (role: `user`).

### AssistantMessage

```go
func AssistantMessage(text string) provider.Message
```

Creates an assistant message with text content (role: `assistant`).

### ToolMessage

```go
func ToolMessage(toolCallID, toolName, output string) provider.Message
```

Creates a tool result message (role: `tool`) containing the output of a tool execution.

**Example:**

```go
msgs := []provider.Message{
    goai.SystemMessage("You are a helpful assistant."),
    goai.UserMessage("Hello!"),
    goai.AssistantMessage("Hi there! How can I help?"),
    goai.UserMessage("What time is it?"),
}

result, err := goai.GenerateText(ctx, model,
    goai.WithMessages(msgs...),
)
```

---

## SchemaFrom

Generates a JSON Schema from a Go type using reflection. The schema is compatible with OpenAI strict mode: all properties are required, pointer types become nullable, and `additionalProperties` is set to `false` on all objects.

```go
func SchemaFrom[T any]() json.RawMessage
```

**Supported struct tags:**

| Tag                            | Example                                | Description                           |
| ------------------------------ | -------------------------------------- | ------------------------------------- |
| `json:"name"`                  | `json:"city"`                          | Sets the property name in the schema. |
| `json:"-"`                     | `json:"-"`                             | Excludes the field from the schema.   |
| `jsonschema:"description=..."` | `jsonschema:"description=City name"`   | Adds a description to the property.   |
| `jsonschema:"enum=a\|b\|c"`    | `jsonschema:"enum=easy\|medium\|hard"` | Restricts values to an enum.          |

**Supported types:** string, bool, int (all sizes), uint (all sizes), float32/64, slices, maps with string keys, structs, `time.Time` (string with `date-time` format), `json.RawMessage` (any type), and recursive types. Embedded structs are flattened. Pointer types produce nullable schemas (`type: ["<base>", "null"]`).

> **Limitation:** mutually recursive named slice types (for example `type A []B; type B []A`) are not supported and can overflow the stack. Use struct wrappers instead.

**Example:**

```go
type City struct {
    Name       string  `json:"name" jsonschema:"description=City name"`
    Population int     `json:"population"`
    Country    string  `json:"country" jsonschema:"enum=US|UK|FR|DE"`
    Nickname   *string `json:"nickname"` // nullable
}

schema := goai.SchemaFrom[City]()
// Produces: {"type":"object","properties":{...},"required":["name","population","country","nickname"],"additionalProperties":false}
```

`SchemaFrom` is called automatically by `GenerateObject` and `StreamObject`. Use it directly when you need the schema for other purposes (validation, documentation, passing to `WithExplicitSchema`).

---

## Error Utility Functions

See [Errors](errors.md) for `IsOverflow`, `ParseHTTPError`, `ParseHTTPErrorWithHeaders`, `ParseStreamError`, `ClassifyStreamError`, `ErrUnknownTool`, and error type documentation.

## Tool Call Context Utility

### ToolCallIDFromContext

Returns the provider tool call ID from a tool execution context.

```go
func ToolCallIDFromContext(ctx context.Context) string
```

This is useful inside a tool `Execute` function when you need to correlate logs or telemetry with the provider's tool call identifier.
