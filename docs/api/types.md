---
title: Types
description: "Reference for commonly used GoAI and provider types, including TextResult, Message, Part, StreamChunk, Usage, and ToolCall."
---

# Types

This page documents the commonly used public types in the `goai` and `goai/provider` packages. Specialized state and hook types are covered in their dedicated concept/reference pages.

---

## goai Package Types

Import: `github.com/zendev-sh/goai`

### TextResult

The final result of a text generation call (`GenerateText` or `TextStream.Result()`).

```go
type TextResult struct {
    Text             string                       // Accumulated generated text (includes reasoning text when streaming).
    Reasoning        string                       // Accumulated reasoning text when provided by the model.
    ToolCalls        []provider.ToolCall           // Tool calls from the final step.
    Steps            []StepResult                 // Results from each generation step.
    TotalUsage       provider.Usage               // Aggregated token usage across all steps.
    FinishReason     provider.FinishReason        // Why generation stopped.
    Response         provider.ResponseMetadata    // Provider metadata from the last step.
    ProviderMetadata map[string]map[string]any    // Provider-specific response data.
    Sources          []provider.Source            // Citations/references from all steps.
    StepsExhausted   bool                         // True when MaxSteps was reached with tool calls still pending.
    ResponseMessages []provider.Message           // Assistant + tool messages for multi-turn continuation.
}
```

### StepResult

The result of a single generation step in a multi-step tool loop.

```go
type StepResult struct {
    Number       int                      // 1-based step index.
    Text         string                   // Text generated in this step (excludes reasoning text).
    Reasoning    string                   // Reasoning text for this step when provided by the model.
    ToolCalls    []provider.ToolCall       // Tool calls requested in this step.
    ToolResults  []provider.ToolResult     // Tool results for the requested calls.
    FinishReason provider.FinishReason    // Finish reason for this step.
    Usage        provider.Usage           // Token usage for this step.
    Response     provider.ResponseMetadata // Provider metadata for this step.
    ProviderMetadata map[string]map[string]any // Provider-specific response data.
    Sources      []provider.Source        // Citations from this step.
}
```

### TextStream

A streaming text generation response with three consumption modes.

| Method         | Return Type                   | Description                                        |
| -------------- | ----------------------------- | -------------------------------------------------- |
| `Stream()`     | `<-chan provider.StreamChunk` | Raw stream chunks (all types).                     |
| `TextStream()` | `<-chan string`               | Text content only.                                 |
| `Result()`     | `*TextResult`                 | Blocks until complete, returns accumulated result. |
| `Err()`        | `error`                       | Returns the first stream error encountered.        |

`Stream()` and `TextStream()` are mutually exclusive. `Result()` can be called after either.

### ObjectResult

The final result of a structured output generation.

```go
type ObjectResult[T any] struct {
    Object           T                        // The parsed structured output.
    Usage            provider.Usage           // Token consumption.
    FinishReason     provider.FinishReason    // Why generation stopped.
    ProviderMetadata map[string]map[string]any // Provider-specific response data.
    Response         provider.ResponseMetadata // Provider metadata.
    ResponseMessages []provider.Message       // Assistant + tool messages for multi-turn continuation.
    Steps            []StepResult             // Results from each generation step (for multi-step tool loops).
}
```

### ObjectStream

A streaming structured output response.

| Method                  | Return Type                 | Description                                            |
| ----------------------- | --------------------------- | ------------------------------------------------------ |
| `PartialObjectStream()` | `<-chan *T`                 | Emits progressively populated partial objects.         |
| `Result()`              | `(*ObjectResult[T], error)` | Blocks until complete, returns final validated object. |
| `Err()`                 | `error`                     | Returns the first stream error encountered.            |

### EmbedResult

The result of a single embedding generation.

```go
type EmbedResult struct {
    Embedding        []float64                 // The generated vector.
    Usage            provider.Usage            // Token consumption.
    ProviderMetadata map[string]map[string]any // Provider-specific response data.
    Response         provider.ResponseMetadata // Provider metadata.
}
```

### EmbedManyResult

The result of a batch embedding generation.

```go
type EmbedManyResult struct {
    Embeddings       [][]float64               // One vector per input value.
    Usage            provider.Usage            // Aggregated token consumption.
    ProviderMetadata map[string]map[string]any // Provider-specific response data.
    Response         provider.ResponseMetadata // Provider metadata.
}
```

### ImageResult

The result of image generation.

```go
type ImageResult struct {
    Images           []provider.ImageData      // Generated images.
    ProviderMetadata map[string]map[string]any // Provider-specific response data.
    Usage            provider.Usage            // Token consumption.
    Response         provider.ResponseMetadata // Provider metadata.
}
```

### VideoResult

The result of video generation. `Video` is the first generated video and `Videos` contains all results.

```go
type VideoResult struct {
    Video            provider.VideoData
    Videos           []provider.VideoData
    ProviderMetadata map[string]map[string]any
    Response         provider.ResponseMetadata
}
```

### SchemaFrom

Generates a JSON Schema from a Go struct type `T`. Used by `GenerateObject` and `StreamObject` to describe the expected output structure, and compatible with OpenAI strict-mode schemas.

```go
func SchemaFrom[T any]() json.RawMessage
```

**Supported struct tags:**

| Tag                            | Example                                     | Description                                            |
| ------------------------------ | ------------------------------------------- | ------------------------------------------------------ |
| `json:"name"`                  | `json:"firstName"`                          | Field name in JSON output; `json:"-"` to skip a field. |
| `jsonschema:"description=..."` | `jsonschema:"description=User's full name"` | Adds a description to the field in the schema.         |
| `jsonschema:"enum=a\|b\|c"`    | `jsonschema:"enum=easy\|medium\|hard"`      | Restricts the field to enumerated values.              |

**Example:**

```go
type Recipe struct {
    Name        string   `json:"name" jsonschema:"description=Recipe name"`
    Ingredients []string `json:"ingredients" jsonschema:"description=List of ingredients"`
    Steps       []string `json:"steps" jsonschema:"description=Cooking steps"`
    Difficulty  string   `json:"difficulty" jsonschema:"enum=easy|medium|hard"`
}

schema := goai.SchemaFrom[Recipe]()
```

**Edge cases:**

- `time.Time` is converted to `{"type": "string", "format": "date-time"}`.
- Self-referential named slice types (e.g. `type Foo []Foo`) are detected and produce a schema with `{"type": "array"}` (no items) to avoid infinite recursion.
- Mutually recursive named slice types (e.g. `type A []B; type B []A`) are not detected and will cause a stack overflow. Use struct wrappers instead of raw named-slice mutual recursion.
- Pointer fields are unwrapped and marked nullable; pointer layers on the top-level schema type are unwrapped without a nullable marker.

See [Structured Output](../getting-started/structured-output.md) for full usage.

### Tool

Defines a tool that can be called by the model during generation. Includes an optional `Execute` function for automatic tool loop execution.

```go
type Tool struct {
    Name                   string                                                    // Tool identifier.
    Description            string                                                    // What the tool does (used by the model).
    InputSchema            json.RawMessage                                           // JSON Schema for input parameters.
    ProviderDefinedType    string                                                    // Provider-defined tool type (e.g. "computer_20250124").
    ProviderDefinedOptions map[string]any                                            // Provider-specific tool configuration.
    DeferLoading           bool                                                      // Defer provider tool loading until selected.
    Execute                func(ctx context.Context, input json.RawMessage) (string, error) // Tool implementation.
}
```

When `Execute` is non-nil, `GenerateText` invokes requested tools even with the default single step; `MaxSteps > 1` enables a follow-up model call. `StreamText` requires `MaxSteps > 1` for automatic execution/looping.

When using provider-defined tools (web search, code execution, etc.), set `ProviderDefinedType` and leave `Execute` nil - the provider handles execution server-side.

### NewTool

Builds a `Tool` from a typed input struct and a typed execute function. The JSON Schema is generated from `In` via `SchemaFrom`, and the model's raw JSON arguments are unmarshaled into `In` before execute runs - so callers neither hand-write JSON Schema nor unmarshal input.

```go
func NewTool[In any](name, description string, execute func(ctx context.Context, input In) (string, error)) Tool
```

```go
weatherTool := goai.NewTool("get_weather", "Get the current weather for a city.",
    func(ctx context.Context, in struct {
        City string `json:"city" jsonschema:"description=City name"`
    }) (string, error) {
        return forecast(in.City), nil
    })
```

`In` is typically a struct using `json`/`jsonschema` tags (see [SchemaFrom](#schemafrom)); use `struct{}` for a no-parameter tool. If the model's arguments fail to unmarshal into `In`, execute is not called and the error is returned to the model as the tool result. For a hand-written schema or a provider-defined tool, build the `Tool` struct directly instead.

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

### VideoOption

A function that configures a video generation call.

```go
type VideoOption func(*videoOptions)
```

### RequestInfo

Passed to the `OnRequest` hook before a generation call.

```go
type RequestInfo struct {
    Ctx          context.Context    // Caller's context (for span parenting in observability hooks).
    Model        string             // Model ID.
    MessageCount int                // Number of messages in the request.
    ToolCount    int                // Number of tools available.
    Timestamp    time.Time          // When the request was initiated.
    Messages     []provider.Message // Full conversation history sent to the model for this call.
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
    ToolCallID   string          // Provider-assigned identifier for this tool call.
    ToolName     string          // Name of the tool called.
    Step         int             // 1-based index of the generation step in which this tool was called.
    Input        json.RawMessage // Raw JSON arguments passed to the tool.
    Output       string          // String result returned by the tool.
    OutputObject any             // Parsed JSON value of Output when the tool returned valid JSON; nil otherwise.
    StartTime    time.Time       // When tool execution began. Zero for unknown tools. For skipped tools, reflects skip decision time.
    Duration     time.Duration   // Time from before Execute to after OnAfterToolExecute (includes hook overhead). Zero when Skipped.
    Skipped      bool            // True when skipped by OnBeforeToolExecute. Duration is zero, StartTime reflects skip decision time.
    Error        error           // Non-nil if execution failed.
    Metadata     map[string]any  // Consumer metadata from OnAfterToolExecute (nil if not set).
}
```

### ToolCallStartInfo

Passed to the `OnToolCallStart` hook before a tool executes.

```go
type ToolCallStartInfo struct {
    ToolCallID string          // Provider-assigned identifier for this tool call.
    ToolName   string          // Name of the tool called.
    Step       int             // 1-based index of the generation step in which this tool was called.
    Input      json.RawMessage // Raw JSON arguments passed to the tool.
}
```

### BeforeToolExecuteInfo

Passed to the `OnBeforeToolExecute` hook before a tool's Execute function runs.

```go
type BeforeToolExecuteInfo struct {
    Ctx        context.Context // Tool execution context (with tool call ID).
    ToolCallID string          // Provider-assigned identifier.
    ToolName   string          // Name of the tool about to execute.
    Step       int             // 1-based step index.
    Input      json.RawMessage // Raw JSON arguments.
}
```

### BeforeToolExecuteResult

Controls what happens after `OnBeforeToolExecute` runs.

```go
type BeforeToolExecuteResult struct {
    Skip   bool            // Prevent Execute from running; use Result as synthetic output.
    Result string          // Synthetic tool output when Skip is true.
    Error  error           // Tool error when Skip is true.
    Ctx    context.Context // Replaces tool context (nil = use default).
    Input  json.RawMessage // Replaces tool input (nil = use original).
}
```

### AfterToolExecuteInfo

Passed to the `OnAfterToolExecute` hook after a tool's Execute function completes.

```go
type AfterToolExecuteInfo struct {
    Ctx        context.Context // Tool execution context (with tool call ID).
    ToolCallID string          // Provider-assigned identifier.
    ToolName   string          // Name of the tool that executed.
    Step       int             // 1-based step index.
    Input      json.RawMessage // Raw JSON arguments.
    Output     string          // Tool output.
    Error      error           // Non-nil if Execute failed.
}
```

### AfterToolExecuteResult

Controls tool output modification before it reaches the LLM.

```go
type AfterToolExecuteResult struct {
    Output   string         // Replaces tool output (empty = preserve original).
    Error    error          // Replaces tool error (nil = preserve original).
    Metadata map[string]any // Opaque consumer data, passed to ToolCallInfo.Metadata.
}
```

### BeforeStepInfo

Passed to the `OnBeforeStep` hook before each LLM call in a multi-step tool loop (step 2+).

```go
type BeforeStepInfo struct {
    Ctx      context.Context    // Generation context (for cancellation, tracing).
    Step     int                // 1-based step number.
    Messages []provider.Message // Current conversation history (shallow clone).
}
```

### BeforeStepResult

Controls behavior before the next LLM call.

```go
type BeforeStepResult struct {
    ExtraMessages []provider.Message // Appended before LLM call. Ignored when Stop is true.
    Stop          bool               // Terminate tool loop. ExtraMessages are ignored.
}
```

### FinishInfo

Passed to the `OnFinish` hook after all generation steps complete.

```go
type FinishInfo struct {
    StepsExhausted bool               // True when MaxSteps reached while model still wanted tools.
    TotalSteps     int                // Number of generation steps executed.
    TotalUsage     provider.Usage     // Aggregated token usage across all steps.
    FinishReason   provider.FinishReason // Finish reason from the last step.
    StoppedBy      provider.StopCause // Why generation terminated.
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
}
```

### CapableModel

Optional interface that `LanguageModel` implementations can satisfy to declare capabilities. Use `ModelCapabilitiesOf` to query safely.

```go
type CapableModel interface {
    Capabilities() ModelCapabilities
}
```

### ModelCapabilitiesOf

Returns the model's capabilities if it implements `CapableModel`, or a zero-value `ModelCapabilities` otherwise.

```go
func ModelCapabilitiesOf(m LanguageModel) ModelCapabilities
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

### VideoModel

Interface for video generation models.

```go
type VideoModel interface {
    ModelID() string
    DoGenerate(ctx context.Context, params VideoParams) (*VideoResult, error)
}
```

### GenerateParams

All parameters for a generation request. Constructed internally by GoAI from options - provider implementations receive this.

```go
type GenerateParams struct {
    Messages         []Message          // Conversation history.
    System           string             // System prompt.
    Tools            []ToolDefinition   // Available tools.
    MaxOutputTokens  int                // Response length limit (0 = provider default).
    Temperature      *float64           // Randomness control (nil = provider default).
    TopP             *float64           // Nucleus sampling (nil = provider default).
    TopK             *int               // Top-K sampling (nil = provider default).
    FrequencyPenalty *float64           // Frequency penalty (nil = provider default).
    PresencePenalty  *float64           // Presence penalty (nil = provider default).
    Seed             *int               // Deterministic generation (nil = provider default).
    StopSequences    []string           // Stop generation when encountered.
    Headers          map[string]string  // Additional HTTP headers.
    ProviderOptions  map[string]any     // Provider-specific parameters.
    PromptCaching    bool               // Enable prompt caching.
    CacheTTL         string             // Provider cache lifetime hint.
    ToolChoice       string             // Tool selection: "auto", "none", "required", or tool name.
    ResponseFormat   *ResponseFormat    // Structured JSON output schema.
}
```

### GenerateResult

Response from a non-streaming generation.

```go
type GenerateResult struct {
    Text             string                       // Generated text.
    Reasoning        string                       // Generated reasoning text.
    ReasoningParts   []Part                       // Reasoning blocks with preserved boundaries/metadata.
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
    ToolCallID   string             // Tool call fields (for ChunkToolCall, ChunkToolCallStreamStart).
    ToolName     string
    ToolInput    string
    FinishReason FinishReason       // For ChunkStepFinish, ChunkFinish.
    Usage        Usage              // For ChunkFinish (may also appear on ChunkStepFinish).
    Error        error              // For ChunkError.
    Response     ResponseMetadata   // Populated on ChunkFinish (may also appear on ChunkStepFinish).
    Metadata     map[string]any     // Provider-specific data.
    StoppedBy    StopCause          // How the tool loop terminated.
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
    ChunkToolResult          StreamChunkType = "tool_result" // reserved for future use; not currently emitted by any provider
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
    ToolCallID      string          // For PartToolCall and PartToolResult.
    ToolName        string          // For PartToolCall and PartToolResult.
    ToolInput       json.RawMessage // For PartToolCall.
    ToolOutput      string          // For PartToolResult.
    CacheControl    string          // Cache directive (e.g. "ephemeral").
    CacheControlTTL string          // Provider cache lifetime for this part.
    Detail          string          // Image detail level ("low", "high", "auto").
    MediaType       string          // Content type (for PartImage, PartFile).
    Filename        string          // For PartFile.
    RemoteRef       *RemoteFileRef  // Reference to an uploaded remote file (for PartFile, PartImage).
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
    DeferLoading           bool           // Defer provider tool loading until selected.
}
```

### ToolCall

The model's request to invoke a tool.

```go
type ToolCall struct {
    ID    string          // Unique identifier for this call.
    Name  string          // Tool to invoke.
    Input json.RawMessage // JSON-encoded arguments.

    // Provider-specific data that must be preserved across tool round-trips.
    Metadata map[string]any
}
```

### StopCause

```go
type StopCause string

const (
    StopCauseNatural    StopCause = "natural"     // Normal completion.
    StopCauseMaxSteps   StopCause = "max-steps"   // Hit MaxSteps limit.
    StopCausePredicate  StopCause = "predicate"   // Stopped by WithStopWhen.
    StopCauseBeforeStep StopCause = "before-step" // Stopped by OnBeforeStep.
    StopCauseAbort      StopCause = "abort"       // Terminated due to error.
    StopCauseEmpty      StopCause = "empty"       // Provider closed stream without chunks.
    StopCauseNoExecutableTools StopCause = "no-executable-tools" // Model requested tools with no executable handlers.
)
```

### ToolResult

The result of a tool execution sent back to the model.

```go
type ToolResult struct {
    ToolCallID string // ID of the originating ToolCall.
    ToolName   string // Name of the tool.
    Output     string // Stringified result.
    Error      error  // Error returned by the tool (if any).
    IsError    bool   // Convenience boolean for Error != nil.
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
    FinishToolCalls     FinishReason = "tool-calls"     // Model wants to call tools.
    FinishLength        FinishReason = "length"         // Hit max output tokens.
    FinishContentFilter FinishReason = "content-filter" // Content policy triggered.
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
    FileUpload       bool        // Supports remote file upload.
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
    Images           []ImageData                 // Generated images.
    ProviderMetadata map[string]map[string]any   // Provider-specific data.
    Usage            Usage                       // Token/operation usage.
    Response         ResponseMetadata            // Provider metadata.
}
```

### VideoParams

Provider-independent video generation parameters, including prompt, input media, output settings, and polling configuration.

```go
type VideoParams struct {
    Prompt          string
    Image           *MediaData
    N               int
    AspectRatio     string
    Resolution      string
    Duration        time.Duration
    FPS             int
    Seed            *int64
    FrameImages     []VideoFrame
    InputReferences []MediaData
    GenerateAudio   *bool
    ProviderOptions map[string]any
    MaxRetries      int
    PollInterval    time.Duration
    PollTimeout     time.Duration
}
```

### VideoData

```go
type VideoData struct {
    Data      []byte
    MediaType string
}
```

`MediaData` represents inline or remote media inputs. `VideoFrame` tags an image as `VideoFrameFirst` or `VideoFrameLast`.

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
    Embeddings       [][]float64              // Generated vectors.
    Usage            Usage                   // Token consumption.
    ProviderMetadata map[string]map[string]any // Provider-specific response data.
    Response         ResponseMetadata         // Provider metadata.
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

A `TokenSource` whose cached token can be cleared, forcing a fresh fetch. Supports application-level retry-on-401 logic.

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

### TrySend

Utility for provider implementors. Sends a chunk to a stream channel, returning `false` if the context is cancelled. Prevents goroutine leaks when the consumer stops reading.

```go
func TrySend(ctx context.Context, out chan<- StreamChunk, chunk StreamChunk) bool
```

### FileUpload

Describes a file to upload to a provider's remote storage.

```go
type FileUpload struct {
    Reader    io.Reader // File content to upload.
    Filename  string    // Name of the file.
    MediaType string    // MIME type (e.g. "application/pdf").
    Purpose   string    // Intended use (e.g. "assistants", "vision").
}
```

### RemoteFileRef

A reference to an uploaded remote file. Carries raw bytes for fallback on providers without native file APIs.

```go
type RemoteFileRef struct {
    Provider  string    // Provider that owns this file.
    ID        string    // Provider-specific file identifier.
    URI       string    // Provider-specific file URI (e.g. for Gemini).
    Filename  string    // Original file name.
    MediaType string    // MIME type.
    ExpiresAt time.Time // When the remote file expires (zero if unknown).
    Data      []byte    // Raw file bytes for compat fallback.
}
```

### FileUploader

Interface for uploading and deleting remote files. Providers that support file upload implement `FileUploadCapableModel`.

```go
type FileUploader interface {
    UploadFile(ctx context.Context, upload FileUpload) (*RemoteFileRef, error)
    DeleteFile(ctx context.Context, ref RemoteFileRef) error
}
```

### FileUploadCapableModel

Optional interface that `LanguageModel` implementations can satisfy to indicate they support remote file upload. Use `FileUploader()` to get the uploader.

```go
type FileUploadCapableModel interface {
    FileUploader() FileUploader
}
```

### ErrFileUploadUnsupported

Sentinel error returned by providers that do not support remote file upload. Callers can check for this error to fall back to inline data URIs.

```go
var ErrFileUploadUnsupported = errors.New("goai: file upload not supported by this provider")
```
