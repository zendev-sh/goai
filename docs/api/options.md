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

### WithOptions

Combines multiple `Option` values into a single `Option`. Useful for helper libraries that want to return one reusable option bundle.

```go
func WithOptions(opts ...Option) Option
```

**Default:** empty.

**Example:**

```go
base := goai.WithOptions(
    goai.WithSystem("You are concise."),
    goai.WithMaxOutputTokens(200),
)
```

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

**Default:** `1` (single model step).

Values below 1 are silently clamped to 1.

Set to 2 or higher to enable multi-step auto loops. For example, `WithMaxSteps(2)` allows: step 1 = model calls tool, step 2 = model uses tool result to generate final answer.

With `GenerateText`, tools with `Execute` can still be executed at step 1 if the model requests them. With `MaxSteps(1)`, GoAI executes those tools and returns without a follow-up generation step.

### WithSequentialToolExecution

Forces tool calls to execute one at a time instead of in parallel.

```go
func WithSequentialToolExecution() Option
```

**Default:** `false` (parallel execution).

Useful when tools share non-thread-safe resources or when execution order matters.

### WithToolChoice

Controls how the model selects tools.

```go
func WithToolChoice(tc string) Option
```

**Values:**

| Value           | Behavior                                                             |
| --------------- | -------------------------------------------------------------------- |
| `"auto"`        | Model decides whether to use tools (default when tools are present). |
| `"none"`        | Model does not use tools.                                            |
| `"required"`    | Model must use at least one tool.                                    |
| `"<tool_name>"` | Model must use the specified tool.                                   |

**Default:** provider-dependent (typically `"auto"` when tools are present).

**Constants:**

- `goai.ToolChoiceAuto`
- `goai.ToolChoiceNone`
- `goai.ToolChoiceRequired`

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

### WithTopK

Limits sampling to the top K tokens. Only available with some providers (e.g., Google, Anthropic).

```go
func WithTopK(k int) Option
```

**Default:** `nil` (provider default).

### WithFrequencyPenalty

Penalizes tokens based on how frequently they appear in the text so far.

```go
func WithFrequencyPenalty(p float64) Option
```

**Default:** `nil` (provider default).

### WithPresencePenalty

Penalizes tokens that have already appeared in the text so far.

```go
func WithPresencePenalty(p float64) Option
```

**Default:** `nil` (provider default).

### WithSeed

Sets the seed for deterministic generation. Not all providers support this.

```go
func WithSeed(s int) Option
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

> **Note:** `WithProviderOptions` validates values eagerly and panics for non-JSON-serializable values (for example: channels, functions, unsafe pointers, cyclic values, or excessive nesting).

---

## Telemetry Hooks

> **Panic handling:** Hooks use two panic strategies depending on execution context:
>
> - **Caller goroutine** (`OnRequest`): panics propagate in `GenerateText`, `GenerateObject`, and `StreamText` first step. In `StreamText` step 2+ (goroutine), each callback is individually recovered.
> - **Mixed** (`OnResponse`): recovered in `GenerateText`, `GenerateObject`, `StreamObject`, and background goroutines; propagates in `StreamText` first-step error path. See per-hook docs below.
> - **Worker goroutines** (`OnStepFinish`, `OnToolCallStart`, `OnToolCall`): panics are recovered, logged to stderr, and do not propagate.
> - **Interceptor hooks** (`OnBeforeToolExecute`, `OnAfterToolExecute`, `OnBeforeStep`): always panic-recovered with hook-specific behavior (skip tool, preserve result, or proceed normally).

### WithOnRequest

Sets a callback invoked before each model API call.

```go
func WithOnRequest(fn func(RequestInfo)) Option
```

**Default:** `nil`.

> **Panic behavior:** In `GenerateText`, `GenerateObject`, and `StreamText`'s first step, panics propagate to the caller. In `StreamText` step 2+ (goroutine), each callback is individually panic-recovered.

### WithOnResponse

Sets a callback invoked after each model API call completes (success or failure).

```go
func WithOnResponse(fn func(ResponseInfo)) Option
```

**Default:** `nil`.

> **Panic behavior:** In `GenerateText`, all `StreamText` success paths, `GenerateObject`, `StreamObject` (success path), and `StreamObject` (error path), panics are individually recovered and logged to stderr. In `StreamText`'s first-step error path, panics propagate to the caller.

### WithOnStepFinish

Sets a callback invoked after each generation step completes, including after tool execution.

```go
func WithOnStepFinish(fn func(StepResult)) Option
```

**Default:** `nil`. Only relevant when `MaxSteps > 1`.

> **Panic behavior:** Panics are recovered and logged to stderr.

### WithOnToolCallStart

Sets a callback invoked before each tool execution. Fires from the tool's goroutine in `executeToolsParallel`. Also fires during `StreamText` when tools are executed in multi-step loops.

```go
func WithOnToolCallStart(fn func(ToolCallStartInfo)) Option
```

**Default:** `nil`. Relevant when tools with `Execute` are provided.

> **Panic behavior:** If the callback panics, the tool does not execute and the panic is recovered and logged to stderr.

### WithOnToolCall

Sets a callback invoked after each tool execution.

```go
func WithOnToolCall(fn func(ToolCallInfo)) Option
```

**Default:** `nil`. Relevant when tools with `Execute` are provided.

> **Note:** When multiple tools execute in a single step, OnToolCall callbacks fire concurrently from separate goroutines. Order is non-deterministic.

> **Panic behavior:** Panics are recovered and logged to stderr.

### WithOnBeforeToolExecute

```go
goai.WithOnBeforeToolExecute(func(info goai.BeforeToolExecuteInfo) goai.BeforeToolExecuteResult {
    // Return Skip: true to prevent tool execution
    return goai.BeforeToolExecuteResult{}
})
```

Called before each tool's Execute function. Can skip execution for permission checks, rate limiting, or doom-loop detection. `info.Ctx` carries the tool execution context (with tool call ID injected). Only one callback supported (replaces previous). Panic-recovered: a panic skips the tool with an error result. The returned `Ctx` and `Input` fields can override the context and input passed to `Execute` (nil = no override).

### WithOnAfterToolExecute

```go
goai.WithOnAfterToolExecute(func(info goai.AfterToolExecuteInfo) goai.AfterToolExecuteResult {
    // Return modified Output to transform tool results
    return goai.AfterToolExecuteResult{}
})
```

Called after each tool's Execute function, before the result is sent to the LLM. Can modify output for secret scanning, truncation, or transformation. `info.Ctx` carries the same tool execution context as `OnBeforeToolExecute`. Only one callback supported. Panic-recovered: preserves original result. The returned `Metadata` field is passed through to `ToolCallInfo.Metadata` for downstream observability hooks.

### WithOnBeforeStep

```go
goai.WithOnBeforeStep(func(info goai.BeforeStepInfo) goai.BeforeStepResult {
    // Return ExtraMessages to inject, or Stop: true to end loop
    return goai.BeforeStepResult{}
})
```

Called before each LLM call in a multi-step tool loop (step 2+ only, not step 1). Can inject additional messages or stop the loop early. `info.Ctx` carries the generation context for cancellation checks or external calls. Only one callback supported. Panic-recovered: a panic is logged and the step proceeds normally.

> **Panic handling note:** Interceptor hooks (`OnBeforeToolExecute`, `OnAfterToolExecute`, `OnBeforeStep`): always panic-recovered with hook-specific behavior (skip tool, preserve result, or proceed normally).

### WithOnFinish

```go
goai.WithOnFinish(func(info goai.FinishInfo) {
    // info.StepsExhausted, info.TotalSteps, info.TotalUsage, info.FinishReason
})
```

Called once after all generation steps complete. Fires in all code paths: `GenerateText`, `StreamText`, `GenerateObject` (including max_steps error), and `StreamObject`. Does NOT fire when DoGenerate/DoStream returns a provider error. Multiple callbacks supported (append). Panic-recovered.

`FinishInfo.StepsExhausted` is the authoritative signal for max-steps exhaustion -- it is not available from any per-step hook.

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

**Default:** `4` (when `n <= 0`).

### WithEmbeddingProviderOptions

Sets provider-specific parameters for embedding requests.

```go
func WithEmbeddingProviderOptions(opts map[string]any) Option
```

**Default:** empty.

> **Note:** `WithEmbeddingProviderOptions` uses the same validation as `WithProviderOptions` and panics for non-JSON-serializable values.

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

### WithImageMaxRetries

Sets the retry count for transient errors during image generation (429, 503, 5xx).

```go
func WithImageMaxRetries(n int) ImageOption
```

**Default:** `2`. Retries use exponential backoff.

### WithImageTimeout

Sets the timeout for the image generation call.

```go
func WithImageTimeout(d time.Duration) ImageOption
```

**Default:** no timeout (relies on the context).

### WithImageProviderOptions

Sets provider-specific options for image generation.

```go
func WithImageProviderOptions(opts map[string]any) ImageOption
```

**Default:** empty.
