---
title: Types
description: "Complete type reference for GoAI and provider packages. Covers TextResult, Message, Part, StreamChunk, Usage, ToolCall, and all public types."
---

# Types

This page documents all public types in the `goai` and `goai/provider` packages.

---

## goai Package Types

Import: `github.com/zendev-sh/goai`

### TextResult

The final result of a text generation call (`GenerateText` or `TextStream.Result()`).

```go
type TextResult struct {
    Text         string                   // Accumulated generated text.
    ToolCalls    []provider.ToolCall       // Tool calls from the final step.
    Steps        []StepResult             // Results from each generation step.
    TotalUsage   provider.Usage           // Aggregated token usage across all steps.
    FinishReason provider.FinishReason    // Why generation stopped.
    Response     provider.ResponseMetadata // Provider metadata from the last step.
    Sources      []provider.Source        // Citations/references from all steps.
}
```

### StepResult

The result of a single generation step in a multi-step tool loop.

```go
type StepResult struct {
    Number       int                      // 1-based step index.
    Text         string                   // Text generated in this step.
    ToolCalls    []provider.ToolCall       // Tool calls requested in this step.
    FinishReason provider.FinishReason    // Finish reason for this step.
    Usage        provider.Usage           // Token usage for this step.
    Response     provider.ResponseMetadata // Provider metadata for this step.
    Sources      []provider.Source        // Citations from this step.
}
```

### TextStream

A streaming text generation response with three consumption modes.

| Method | Return Type | Description |
|--------|-------------|-------------|
| `Stream()` | `<-chan provider.StreamChunk` | Raw stream chunks (all types). |
| `TextStream()` | `<-chan string` | Text content only. |
| `Result()` | `*TextResult` | Blocks until complete, returns accumulated result. |

`Stream()` and `TextStream()` are mutually exclusive. `Result()` can be called after either.

### ObjectResult

The final result of a structured output generation.

```go
type ObjectResult[T any] struct {
    Object       T                        // The parsed structured output.
    Usage        provider.Usage           // Token consumption.
    FinishReason provider.FinishReason    // Why generation stopped.
    Response     provider.ResponseMetadata // Provider metadata.
}
```

### ObjectStream

A streaming structured output response.

| Method | Return Type | Description |
|--------|-------------|-------------|
| `PartialObjectStream()` | `<-chan *T` | Emits progressively populated partial objects. |
| `Result()` | `(*ObjectResult[T], error)` | Blocks until complete, returns final validated object. |

### EmbedResult

The result of a single embedding generation.

```go
type EmbedResult struct {
    Embedding []float64     // The generated vector.
    Usage     provider.Usage // Token consumption.
}
```

### EmbedManyResult

The result of a batch embedding generation.

```go
type EmbedManyResult struct {
    Embeddings [][]float64   // One vector per input value.
    Usage      provider.Usage // Aggregated token consumption.
}
```

### ImageResult

The result of image generation.

```go
type ImageResult struct {
    Images []provider.ImageData // Generated images.
}
```

### Tool

Defines a tool that can be called by the model during generation. Includes an optional `Execute` function for automatic tool loop execution.

```go
type Tool struct {
    Name                   string                                                    // Tool identifier.
    Description            string                                                    // What the tool does (used by the model).
    InputSchema            json.RawMessage                                           // JSON Schema for input parameters.
    ProviderDefinedType    string                                                    // Provider-defined tool type (e.g. "computer_20250124").
    ProviderDefinedOptions map[string]any                                            // Provider-specific tool configuration.
    Execute                func(ctx context.Context, input json.RawMessage) (string, error) // Tool implementation.
}
```

When `Execute` is non-nil and `MaxSteps > 1`, `GenerateText` automatically invokes the tool and feeds results back to the model.

When using provider-defined tools (web search, code execution, etc.), set `ProviderDefinedType` and leave `Execute` nil - the provider handles execution server-side.

### Option

A function that configures a generation call. See [Options](options.md) for all available option functions.

```go
type Option func(*options)
```

### ImageOption

A function that configures an image generation call.

```go
type ImageOption func(*imageOptions)
```

### RequestInfo

Passed to the `OnRequest` hook before a generation call.

```go
type RequestInfo struct {
    Provider     string    // Provider identifier (e.g. "openai", "anthropic").
    Model        string    // Model ID.
    MessageCount int       // Number of messages in the request.
    ToolCount    int       // Number of tools available.
    Timestamp    time.Time // When the request was initiated.
}
```

### ResponseInfo

Passed to the `OnResponse` hook after a generation call completes.

```go
type ResponseInfo struct {
    Latency      time.Duration         // Time from request to response.
    Usage        provider.Usage        // Token consumption for this call.
    FinishReason provider.FinishReason // Why generation stopped.
    Error        error                 // Non-nil if the call failed.
    StatusCode   int                   // HTTP status code (0 if not applicable).
}
```

### ToolCallInfo

Passed to the `OnToolCall` hook after a tool executes.

```go
type ToolCallInfo struct {
    ToolName  string        // Name of the tool called.
    InputSize int           // Byte length of the tool input JSON.
    Duration  time.Duration // How long execution took.
    Error     error         // Non-nil if execution failed.
}
```

### APIError

Represents a non-overflow API error. See [Errors](errors.md).

### ContextOverflowError

Indicates the prompt exceeded the model's context window. See [Errors](errors.md).

---

## provider Package Types

Import: `github.com/zendev-sh/goai/provider`

### LanguageModel

Interface for text generation models.

```go
type LanguageModel interface {
    ModelID() string
    DoGenerate(ctx context.Context, params GenerateParams) (*GenerateResult, error)
    DoStream(ctx context.Context, params GenerateParams) (*StreamResult, error)
    Capabilities() ModelCapabilities
}
```

### EmbeddingModel

Interface for embedding models.

```go
type EmbeddingModel interface {
    ModelID() string
    DoEmbed(ctx context.Context, values []string, params EmbedParams) (*EmbedResult, error)
    MaxValuesPerCall() int
}
```

`MaxValuesPerCall()` returns the maximum number of values that can be embedded in a single call. Returns 0 if there is no limit. `EmbedMany` uses this to auto-chunk large batches.

### ImageModel

Interface for image generation models.

```go
type ImageModel interface {
    ModelID() string
    DoGenerate(ctx context.Context, params ImageParams) (*ImageResult, error)
}
```

### GenerateParams

All parameters for a generation request. Constructed internally by GoAI from options - provider implementations receive this.

```go
type GenerateParams struct {
    Messages        []Message          // Conversation history.
    System          string             // System prompt.
    Tools           []ToolDefinition   // Available tools.
    MaxOutputTokens int                // Response length limit (0 = provider default).
    Temperature     *float64           // Randomness control (nil = provider default).
    TopP            *float64           // Nucleus sampling (nil = provider default).
    StopSequences   []string           // Stop generation when encountered.
    Headers         map[string]string  // Additional HTTP headers.
    ProviderOptions map[string]any     // Provider-specific parameters.
    PromptCaching   bool               // Enable prompt caching.
    ToolChoice      string             // Tool selection: "auto", "none", "required", or tool name.
    ResponseFormat  *ResponseFormat    // Structured JSON output schema.
}
```

### GenerateResult

Response from a non-streaming generation.

```go
type GenerateResult struct {
    Text             string                       // Generated text.
    ToolCalls        []ToolCall                    // Tool calls requested by the model.
    Sources          []Source                      // Citations from response annotations.
    FinishReason     FinishReason                  // Why generation stopped.
    Usage            Usage                         // Token consumption.
    Response         ResponseMetadata              // Provider metadata.
    ProviderMetadata map[string]map[string]any     // Provider-specific response data.
}
```

### StreamResult

Wraps a streaming response channel.

```go
type StreamResult struct {
    Stream <-chan StreamChunk // Emits chunks as they arrive. Closed when the stream ends.
}
```

### StreamChunk

A single event in a streaming response. The `Type` field determines which other fields are populated.

```go
type StreamChunk struct {
    Type         StreamChunkType    // Chunk kind.
    Text         string             // Content (for ChunkText, ChunkReasoning).
    ToolCallID   string             // Tool call fields (for ChunkToolCall).
    ToolName     string
    ToolInput    string
    FinishReason FinishReason       // For ChunkStepFinish, ChunkFinish.
    Usage        Usage              // For ChunkFinish.
    Error        error              // For ChunkError.
    Response     ResponseMetadata   // Populated on ChunkFinish.
    Metadata     map[string]any     // Provider-specific data.
}
```

### StreamChunkType

```go
type StreamChunkType string

const (
    ChunkText                StreamChunkType = "text"
    ChunkReasoning           StreamChunkType = "reasoning"
    ChunkToolCall            StreamChunkType = "tool_call"
    ChunkToolCallDelta       StreamChunkType = "tool_call_delta"
    ChunkToolCallStreamStart StreamChunkType = "tool_call_streaming_start"
    ChunkToolResult          StreamChunkType = "tool_result"
    ChunkStepFinish          StreamChunkType = "step_finish"
    ChunkFinish              StreamChunkType = "finish"
    ChunkError               StreamChunkType = "error"
)
```

### Message

A conversation message.

```go
type Message struct {
    Role            Role           // Sender (system, user, assistant, tool).
    Content         []Part         // Content parts.
    ProviderOptions map[string]any // Provider-specific message parameters.
}
```

### Part

A single content element within a message. The `Type` field determines which other fields are populated.

```go
type Part struct {
    Type            PartType        // Part kind.
    Text            string          // For PartText, PartReasoning.
    URL             string          // For images (data:image/... format).
    ToolCallID      string          // For PartToolCall.
    ToolName        string          // For PartToolCall.
    ToolInput       json.RawMessage // For PartToolCall.
    ToolOutput      string          // For PartToolResult.
    CacheControl    string          // Cache directive (e.g. "ephemeral").
    Detail          string          // Image detail level ("low", "high", "auto").
    MediaType       string          // Content type (for PartImage, PartFile).
    Filename        string          // For PartFile.
    ProviderOptions map[string]any  // Provider-specific part parameters.
}
```

### PartType

```go
type PartType string

const (
    PartText       PartType = "text"
    PartReasoning  PartType = "reasoning"
    PartImage      PartType = "image"
    PartToolCall   PartType = "tool-call"
    PartToolResult PartType = "tool-result"
    PartFile       PartType = "file"
)
```

### Role

```go
type Role string

const (
    RoleSystem    Role = "system"
    RoleUser      Role = "user"
    RoleAssistant Role = "assistant"
    RoleTool      Role = "tool"
)
```

### ToolDefinition

Wire-level tool schema sent to the provider. This is the provider-facing counterpart of `goai.Tool`.

```go
type ToolDefinition struct {
    Name                   string         // Tool identifier.
    Description            string         // What the tool does.
    InputSchema            json.RawMessage // JSON Schema for input parameters.
    ProviderDefinedType    string         // Provider-defined tool type.
    ProviderDefinedOptions map[string]any // Provider-specific tool configuration.
}
```

### ToolCall

The model's request to invoke a tool.

```go
type ToolCall struct {
    ID    string          // Unique identifier for this call.
    Name  string          // Tool to invoke.
    Input json.RawMessage // JSON-encoded arguments.
}
```

### Usage

Token consumption for a request.

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

### FinishReason

```go
type FinishReason string

const (
    FinishStop          FinishReason = "stop"           // Normal completion.
    FinishToolCalls     FinishReason = "tool_calls"     // Model wants to call tools.
    FinishLength        FinishReason = "length"         // Hit max output tokens.
    FinishContentFilter FinishReason = "content_filter" // Content policy triggered.
    FinishError         FinishReason = "error"          // Generation error.
    FinishOther         FinishReason = "other"          // Provider-specific reason.
)
```

### ModelCapabilities

Describes what features a model supports.

```go
type ModelCapabilities struct {
    Temperature      bool        // Accepts temperature parameter.
    Reasoning        bool        // Supports extended thinking/reasoning.
    Attachment       bool        // Supports file attachments.
    ToolCall         bool        // Supports tool/function calling.
    InputModalities  ModalitySet // Supported input types.
    OutputModalities ModalitySet // Supported output types.
}
```

### ModalitySet

```go
type ModalitySet struct {
    Text  bool
    Audio bool
    Image bool
    Video bool
    PDF   bool
}
```

### ResponseMetadata

Provider-specific response information.

```go
type ResponseMetadata struct {
    ID               string         // Provider's response identifier.
    Model            string         // Actual model used (may differ from requested).
    Headers          map[string]string // Selected response headers.
    ProviderMetadata map[string]any // Provider-specific metadata.
}
```

### Source

A citation or reference from the model's response.

```go
type Source struct {
    ID               string         // Source identifier.
    Type             string         // Source kind (e.g. "url", "document").
    URL              string         // Citation URL.
    Title            string         // Citation title.
    StartIndex       int            // Start character offset in the text.
    EndIndex         int            // End character offset in the text.
    ProviderMetadata map[string]any // Provider-specific source data.
}
```

### ImageParams

Parameters for image generation.

```go
type ImageParams struct {
    Prompt          string         // Image description.
    N               int            // Number of images to generate.
    Size            string         // Dimensions (e.g. "1024x1024").
    AspectRatio     string         // Alternative to Size (e.g. "16:9").
    ProviderOptions map[string]any // Provider-specific parameters.
}
```

### ImageData

A single generated image.

```go
type ImageData struct {
    Data      []byte // Raw image bytes.
    MediaType string // MIME type (e.g. "image/png").
}
```

### ImageResult (provider)

Response from image generation at the provider level.

```go
type ImageResult struct {
    Images           []ImageData                   // Generated images.
    ProviderMetadata map[string]map[string]any     // Provider-specific data.
}
```

### ResponseFormat

Requests structured JSON output matching a schema.

```go
type ResponseFormat struct {
    Name   string          // Schema name (used by OpenAI's json_schema mode).
    Schema json.RawMessage // JSON Schema the output must conform to.
}
```

### EmbedParams

Parameters for an embedding request.

```go
type EmbedParams struct {
    ProviderOptions map[string]any // Provider-specific parameters.
}
```

### EmbedResult (provider)

Response from embedding generation at the provider level.

```go
type EmbedResult struct {
    Embeddings [][]float64   // Generated vectors.
    Usage      Usage         // Token consumption.
}
```

### Token

An authentication token with optional expiry.

```go
type Token struct {
    Value     string    // Token string (API key, OAuth access token, etc.).
    ExpiresAt time.Time // When the token expires. Zero means no expiry.
}
```

### TokenSource

Interface for providing authentication tokens. See [provider/token.go](https://github.com/zendev-sh/goai/blob/main/provider/token.go) for built-in implementations (`StaticToken`, `CachedTokenSource`).

```go
type TokenSource interface {
    Token(ctx context.Context) (string, error)
}
```

### InvalidatingTokenSource

A `TokenSource` whose cached token can be cleared, forcing a fresh fetch. Used by retry-on-401 logic.

```go
type InvalidatingTokenSource interface {
    TokenSource
    Invalidate()
}
```

### StaticToken

Creates a `TokenSource` that always returns the given key. Use for simple API key authentication.

```go
func StaticToken(key string) TokenSource
```

### CachedTokenSource

Creates a `TokenSource` that caches tokens until expiry. The fetch function is called lazily on first use and again when the cached token expires. Safe for concurrent use. The returned value also implements `InvalidatingTokenSource`.

```go
func CachedTokenSource(fetchFn TokenFetchFunc) TokenSource
```

### TokenFetchFunc

```go
type TokenFetchFunc func(ctx context.Context) (*Token, error)
```
