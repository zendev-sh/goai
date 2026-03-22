---
title: Options
description: "Reference for all GoAI option functions. Configure prompts, tools, temperature, retries, streaming, structured output, embeddings, and image generation."
---

# Options

All option functions for configuring GoAI calls. Options follow the functional options pattern - pass them as variadic arguments to `GenerateText`, `StreamText`, `GenerateObject`, `StreamObject`, `Embed`, or `EmbedMany`.

Import: `github.com/zendev-sh/goai`

---

## Core Options

### WithSystem

Sets the system prompt.

```go
func WithSystem(s string) Option
```

**Default:** empty (no system prompt).

### WithPrompt

Sets a shorthand single user message. When set, it is auto-wrapped as a `UserMessage` and prepended to `Messages`.

```go
func WithPrompt(s string) Option
```

**Default:** empty.

Use `WithPrompt` for simple single-turn requests. Use `WithMessages` for multi-turn conversations.

### WithMessages

Sets the conversation history.

```go
func WithMessages(msgs ...provider.Message) Option
```

**Default:** empty. Use with message builders (`UserMessage`, `AssistantMessage`, etc.) for multi-turn conversations.

### WithPromptCaching

Enables provider-specific prompt caching (e.g., Anthropic's cache_control).

```go
func WithPromptCaching(b bool) Option
```

**Default:** `false`.

---

## Tool Options

### WithTools

Sets the tools available to the model.

```go
func WithTools(tools ...Tool) Option
```

**Default:** no tools. Pass `goai.Tool` values with `Name`, `Description`, `InputSchema`, and optionally `Execute` for automatic tool loop execution.

### WithMaxSteps

Sets the maximum number of auto tool loop iterations. Step 1 is the initial generation. Each subsequent step executes tool calls and re-generates.

```go
func WithMaxSteps(n int) Option
```

**Default:** `1` (no tool loop - the model generates once and returns, even if it requests tool calls).

Set to 2 or higher to enable automatic tool execution. For example, `WithMaxSteps(2)` allows: step 1 = model calls tool, step 2 = model uses tool result to generate final answer.

### WithToolChoice

Controls how the model selects tools.

```go
func WithToolChoice(tc string) Option
```

**Values:**

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides whether to use tools (default when tools are present). |
| `"none"` | Model does not use tools. |
| `"required"` | Model must use at least one tool. |
| `"<tool_name>"` | Model must use the specified tool. |

**Default:** provider-dependent (typically `"auto"` when tools are present).

---

## Generation Options

### WithMaxOutputTokens

Limits the response length in tokens.

```go
func WithMaxOutputTokens(n int) Option
```

**Default:** `0` (provider default).

### WithTemperature

Controls randomness. Higher values produce more diverse output.

```go
func WithTemperature(t float64) Option
```

**Default:** `nil` (provider default, typically around 1.0).

### WithTopP

Controls nucleus sampling. Only tokens with cumulative probability up to `p` are considered.

```go
func WithTopP(p float64) Option
```

**Default:** `nil` (provider default).

### WithStopSequences

Causes generation to stop when any of the specified sequences is encountered.

```go
func WithStopSequences(seqs ...string) Option
```

**Default:** empty (no stop sequences).

---

## Infrastructure Options

### WithMaxRetries

Sets the retry count for transient errors (429, 503, 5xx).

```go
func WithMaxRetries(n int) Option
```

**Default:** `2`. Retries use exponential backoff. Only retryable errors trigger retry (see [Errors](errors.md)).

### WithTimeout

Sets the timeout for the entire generation call. For streaming calls, the timeout covers from the initial request until the stream is fully consumed.

```go
func WithTimeout(d time.Duration) Option
```

**Default:** no timeout (relies on the context).

### WithHeaders

Sets additional HTTP headers for this request.

```go
func WithHeaders(h map[string]string) Option
```

**Default:** empty. Useful for provider-specific headers (e.g., Azure deployment routing).

### WithProviderOptions

Sets provider-specific request parameters that do not map to standard GoAI options.

```go
func WithProviderOptions(opts map[string]any) Option
```

**Default:** empty. Consult provider documentation for supported keys.

---

## Telemetry Hooks

### WithOnRequest

Sets a callback invoked before each model API call.

```go
func WithOnRequest(fn func(RequestInfo)) Option
```

**Default:** `nil`.

### WithOnResponse

Sets a callback invoked after each model API call completes (success or failure).

```go
func WithOnResponse(fn func(ResponseInfo)) Option
```

**Default:** `nil`.

### WithOnStepFinish

Sets a callback invoked after each generation step completes, including after tool execution.

```go
func WithOnStepFinish(fn func(StepResult)) Option
```

**Default:** `nil`. Only relevant when `MaxSteps > 1`.

### WithOnToolCall

Sets a callback invoked after each tool execution.

```go
func WithOnToolCall(fn func(ToolCallInfo)) Option
```

**Default:** `nil`. Only relevant when tools with `Execute` are provided and `MaxSteps > 1`.

---

## Structured Output Options

These options apply to `GenerateObject` and `StreamObject`.

### WithExplicitSchema

Overrides the auto-generated JSON Schema. When set, `SchemaFrom[T]()` is not called.

```go
func WithExplicitSchema(schema json.RawMessage) Option
```

**Default:** `nil` (schema auto-generated from the type parameter `T`).

### WithSchemaName

Sets the schema name sent to providers. Used by OpenAI's `json_schema` response format.

```go
func WithSchemaName(name string) Option
```

**Default:** `"response"`.

---

## Embedding Options

These options apply to `Embed` and `EmbedMany`.

### WithMaxParallelCalls

Sets the maximum number of concurrent API calls when `EmbedMany` auto-chunks a large batch.

```go
func WithMaxParallelCalls(n int) Option
```

**Default:** `4`.

### WithEmbeddingProviderOptions

Sets provider-specific parameters for embedding requests.

```go
func WithEmbeddingProviderOptions(opts map[string]any) Option
```

**Default:** empty.

---

## Image Options

These options use the separate `ImageOption` type and apply only to `GenerateImage`.

### WithImagePrompt

Sets the text prompt for image generation.

```go
func WithImagePrompt(prompt string) ImageOption
```

### WithImageCount

Sets the number of images to generate.

```go
func WithImageCount(n int) ImageOption
```

**Default:** `1`.

### WithImageSize

Sets the image dimensions (e.g., `"1024x1024"`, `"512x512"`).

```go
func WithImageSize(size string) ImageOption
```

**Default:** provider default.

### WithAspectRatio

Sets the aspect ratio as an alternative to explicit size (e.g., `"16:9"`, `"1:1"`).

```go
func WithAspectRatio(ratio string) ImageOption
```

**Default:** provider default.

### WithImageProviderOptions

Sets provider-specific options for image generation.

```go
func WithImageProviderOptions(opts map[string]any) ImageOption
```

**Default:** empty.
