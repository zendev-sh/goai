package provider

import (
	"context"
	"encoding/json"
	"errors"
)

// TrySend sends a chunk to the output channel, returning false if the context
// is cancelled. This prevents goroutine leaks when the consumer stops reading
// and the channel buffer is full.
//
// Convention: callers MUST check the return value and return early when false,
// EXCEPT for terminal sends where the function exits immediately after the call.
// Terminal sends may leave the return value unchecked since no further work follows.
func TrySend(ctx context.Context, out chan<- StreamChunk, chunk StreamChunk) bool {
	select {
	case out <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}

// StreamChunkType identifies the kind of streaming chunk.
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

// FinishReason indicates why generation stopped.
type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishToolCalls     FinishReason = "tool-calls"
	FinishLength        FinishReason = "length"
	FinishContentFilter FinishReason = "content-filter"
	FinishError         FinishReason = "error"
	FinishOther         FinishReason = "other"
)

// Role identifies the sender of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// PartType identifies the kind of content in a message part.
type PartType string

const (
	PartText       PartType = "text"
	PartReasoning  PartType = "reasoning"
	PartImage      PartType = "image"
	PartToolCall   PartType = "tool-call"
	PartToolResult PartType = "tool-result"
	PartFile       PartType = "file"
)

// GenerateParams contains all parameters for a generation request.
type GenerateParams struct {
	// Messages is the conversation history.
	Messages []Message

	// System is the system prompt.
	System string

	// Tools available to the model.
	Tools []ToolDefinition

	// MaxOutputTokens limits the response length. 0 means provider default.
	MaxOutputTokens int

	// Temperature controls randomness. nil means provider default.
	Temperature *float64

	// TopP controls nucleus sampling. nil means provider default.
	TopP *float64

	// TopK limits sampling to the top K tokens. nil means provider default.
	TopK *int

	// FrequencyPenalty penalizes tokens based on frequency. nil means provider default.
	FrequencyPenalty *float64

	// PresencePenalty penalizes tokens that have appeared. nil means provider default.
	PresencePenalty *float64

	// Seed for deterministic generation. nil means provider default.
	Seed *int

	// StopSequences causes generation to stop when encountered.
	StopSequences []string

	// Headers are additional HTTP headers for this request.
	Headers map[string]string

	// ProviderOptions are provider-specific request parameters.
	ProviderOptions map[string]any

	// PromptCaching enables provider-specific prompt caching.
	PromptCaching bool

	// ToolChoice controls tool selection: "auto", "none", "required", or a specific tool name.
	ToolChoice string

	// ResponseFormat requests structured JSON output matching a schema.
	// When set, providers apply their native JSON mode (OpenAI json_schema,
	// Anthropic tool trick, Gemini responseMimeType+responseSchema).
	ResponseFormat *ResponseFormat
}

// GenerateResult is the response from a non-streaming generation.
type GenerateResult struct {
	// Text is the generated text content.
	Text string

	// Reasoning is the model's thinking/reasoning text (when extended
	// thinking is enabled). Excludes signatures and redacted blocks —
	// those remain in ProviderMetadata for replay. Empty when the
	// provider does not return reasoning or thinking is disabled.
	Reasoning string

	// ToolCalls requested by the model.
	ToolCalls []ToolCall

	// Sources extracted from response annotations (e.g. url_citation).
	Sources []Source

	// FinishReason indicates why generation stopped.
	FinishReason FinishReason

	// Usage tracks token consumption.
	Usage Usage

	// Response contains provider-specific metadata.
	Response ResponseMetadata

	// ProviderMetadata contains provider-specific response data
	// (e.g. logprobs, prediction tokens).
	ProviderMetadata map[string]map[string]any
}

// Source represents a citation or reference from the model's response.
// Matches Vercel AI SDK's LanguageModelV2Source.
type Source struct {
	// ID is the source identifier (provider-assigned or generated).
	ID string

	// Type identifies the source kind (e.g. "url", "document").
	// Maps to Vercel's sourceType field.
	Type string

	// URL is the citation URL.
	URL string

	// Title is the citation title.
	Title string

	// StartIndex is the start character offset in the text.
	StartIndex int

	// EndIndex is the end character offset in the text.
	EndIndex int

	// ProviderMetadata contains provider-specific source data.
	ProviderMetadata map[string]any
}

// StreamResult wraps a streaming response channel.
type StreamResult struct {
	// Stream emits chunks as they arrive. The channel is closed when the stream ends.
	Stream <-chan StreamChunk
}

// StreamChunk is a single event in a streaming response.
// The Type field determines which other fields are populated.
type StreamChunk struct {
	// Type identifies this chunk's kind.
	Type StreamChunkType

	// Text content (for ChunkText and ChunkReasoning).
	Text string

	// Tool call fields (for ChunkToolCall and ChunkToolCallStreamStart).
	ToolCallID string
	ToolName   string
	ToolInput  string

	// FinishReason (for ChunkStepFinish and ChunkFinish).
	FinishReason FinishReason

	// Usage (for ChunkFinish, may also appear on ChunkStepFinish).
	Usage Usage

	// Error carries the provider/goai error when Type == ChunkError.
	//   - Nil for all non-ChunkError chunk types.
	//   - May wrap APIError, NetworkError, or plain errors produced by the
	//     provider or goai's tool loop (e.g. tool execution failures that are
	//     surfaced as chunks rather than returned via Err()).
	//   - Providers set this field directly; consumers should use
	//     errors.Is / errors.As to branch on specific error categories rather
	//     than comparing error values.
	Error error

	// Response metadata (populated on ChunkFinish with ID, Model from the provider).
	Response ResponseMetadata

	// Metadata for provider-specific data (e.g. thoughtSignature).
	Metadata map[string]any

	// StoppedBy classifies how the tool loop terminated. Only meaningful on
	// the final ChunkFinish emitted by goai's tool loop. Values:
	//
	//	""                     - single-step provider chunk or unknown
	//	"natural"              - the loop ended because the model did not request
	//	                         additional tool calls (FinishStop or equivalent)
	//	"max-steps"            - MaxSteps was reached while tool calls were still
	//	                         pending (StepsExhausted)
	//	"predicate"            - WithStopWhen predicate returned true
	//	"before-step"          - OnBeforeStep hook returned Stop=true
	//	"abort"                - the stream terminated on an error path
	//	"empty"                - provider closed its stream with no chunks
	//	                         (streaming-only; see StopCauseEmpty godoc)
	//	"no-executable-tools"  - model returned tool calls but no tool in the
	//	                         configured set has an Execute function
	//	                         (sync-only; see StopCauseNoExecutableTools godoc)
	StoppedBy StopCause
}

// StopCause classifies how a multi-step tool loop terminated. It is carried
// on the final ChunkFinish emitted by goai's loop and on FinishInfo so sync
// consumers (OnFinish hook) and stream consumers both see a consistent
// signal. Provider-level chunks leave this empty.
type StopCause string

const (
	// StopCauseNatural indicates the loop ended because the model did not
	// request further tool calls.
	StopCauseNatural StopCause = "natural"
	// StopCauseMaxSteps indicates MaxSteps was reached with tool calls still pending.
	StopCauseMaxSteps StopCause = "max-steps"
	// StopCausePredicate indicates WithStopWhen returned true.
	StopCausePredicate StopCause = "predicate"
	// StopCauseBeforeStep indicates OnBeforeStep returned Stop=true.
	StopCauseBeforeStep StopCause = "before-step"
	// StopCauseAbort indicates the stream terminated on an error path.
	StopCauseAbort StopCause = "abort"
	// StopCauseEmpty indicates the provider closed its stream without sending
	// any meaningful chunks (no text, no tool calls, no finish reason). This
	// is not an error path; it is a distinct no-op response that consumers may
	// want to treat differently from both "natural" (model completed) and
	// "abort" (error).
	//
	// Emission scope (FIX 38 - narrowed): emitted only by the streaming
	// multi-step tool-loop path (`streamWithToolLoop`, entered when
	// MaxSteps>1 AND at least one configured tool has an Execute function).
	// Sync (GenerateText / DoGenerate) paths do not emit this cause.
	// Streaming single-shot paths (MaxSteps=1 or no executable tools, which
	// bypass streamWithToolLoop) ALSO do not emit this cause - they hardcode
	// StopCauseNatural regardless of chunk content, consistent with
	// Vercel's single-iteration streaming behavior. Consumers that need a
	// distinct "empty response" signal on single-shot streams should
	// inspect len(TextResult.Text) and len(TextResult.ToolCalls) directly.
	// A latent semantic question (should single-shot empty streams also
	// emit StopCauseEmpty?) is tracked at the fireOnFinish call site in
	// generate.go - see FIX 32 TODO.
	StopCauseEmpty StopCause = "empty"
	// StopCauseNoExecutableTools indicates the model returned tool calls but
	// no tool in the configured tool set has an Execute function. The loop
	// cannot proceed (there is nothing to execute) and exits cleanly. This is
	// semantically distinct from StopCauseNatural ("model stopped on its
	// own") because here the model still wants to continue but the consumer
	// has not provided executable tools.
	//
	// Emission scope: sync-only (GenerateText). StreamText does NOT emit this
	// cause because the streaming tool-loop path (streamWithToolLoop) is only
	// entered when at least one executable tool is configured; a streaming
	// call with zero executable tools takes the single-shot path and reports
	// StopCauseNatural. Consumers needing a uniform signal across sync and
	// stream can inspect ToolCalls on the last step and treat a non-empty
	// list with no corresponding Execute function as equivalent.
	StopCauseNoExecutableTools StopCause = "no-executable-tools"
)

// IsValid reports whether s is one of the StopCause constants declared in
// this package. The empty string ("") is NOT considered valid (it is used
// internally as a sentinel for "not yet classified" and to mark
// provider-level chunks that predate the loop classifier).
//
// Consumers may use IsValid to defend against typos or downstream code
// that constructs a StopCause from arbitrary text (StopCause is a string
// alias so construction of unknown values cannot be prevented by the type
// system alone).
func (s StopCause) IsValid() bool {
	switch s {
	case StopCauseNatural,
		StopCauseMaxSteps,
		StopCausePredicate,
		StopCauseBeforeStep,
		StopCauseAbort,
		StopCauseEmpty,
		StopCauseNoExecutableTools:
		return true
	}
	return false
}

// Message represents a conversation message.
type Message struct {
	// Role identifies the sender.
	Role Role

	// Content parts that make up this message.
	Content []Part

	// ProviderOptions are provider-specific message parameters.
	ProviderOptions map[string]any
}

// Part is a single content element within a message.
// The Type field determines which other fields are populated.
type Part struct {
	// Type identifies this part's kind.
	Type PartType

	// Text content (for PartText and PartReasoning).
	Text string

	// URL for images (data:image/png;base64,... format).
	URL string

	// Tool call fields (for PartToolCall and PartToolResult).
	ToolCallID string
	ToolName   string
	ToolInput  json.RawMessage

	// ToolOutput is the result text (for PartToolResult).
	ToolOutput string

	// CacheControl directive (e.g. "ephemeral") for prompt caching.
	CacheControl string

	// Detail level for image parts ("low", "high", "auto").
	Detail string

	// MediaType of the content (for PartImage, PartFile).
	MediaType string

	// Filename of the content (for PartFile).
	Filename string

	// ProviderOptions are provider-specific part parameters.
	ProviderOptions map[string]any
}

// ToolDefinition describes a tool available to the model.
type ToolDefinition struct {
	// Name is the tool's identifier.
	Name string

	// Description explains what the tool does (used by the model to decide when to call it).
	Description string

	// InputSchema is the JSON Schema for the tool's input parameters.
	InputSchema json.RawMessage

	// ProviderDefinedType, when non-empty, marks this as a provider-defined tool
	// (e.g. "computer_20250124", "bash_20250124", "text_editor_20250124").
	// Providers use this to emit the correct API type instead of "custom".
	ProviderDefinedType string

	// ProviderDefinedOptions holds provider-specific tool configuration
	// (e.g. displayWidthPx for computer use). Providers interpret these as needed.
	ProviderDefinedOptions map[string]any
}

// ToolCall represents the model's request to invoke a tool.
type ToolCall struct {
	// ID is a unique identifier for this tool call.
	ID string

	// Name of the tool to invoke.
	Name string

	// Input is the JSON-encoded arguments.
	Input json.RawMessage

	// Metadata carries provider-specific data that must be preserved across
	// tool round-trips (e.g., Google's thoughtSignature).
	Metadata map[string]any
}

// ToolResult represents the outcome of executing a single tool call.
//
// It is a structured companion to the tool_result message content: where
// tool-result messages carry the string payload destined for the LLM, a
// ToolResult exposes the same information to in-process consumers (e.g.
// StopCondition predicates) without forcing them to parse message parts.
//
// Output is the exact string sent back to the model (stringified JSON for
// structured results, or "error: <detail>" when the tool call failed).
// Error is non-nil iff the tool returned an error or panicked; IsError is
// true in the same condition and is provided for ergonomics when
// predicates only need the boolean signal.
type ToolResult struct {
	// ToolCallID matches the ID of the originating ToolCall.
	ToolCallID string

	// ToolName is the name of the tool that was invoked.
	ToolName string

	// Output is the stringified result sent to the model. For failed calls
	// this is "error: <message>" (matching what the tool-result message
	// carries); predicates should inspect Error or IsError to distinguish
	// errors from deliberately empty successful outputs.
	Output string

	// Error is the error returned by the tool's Execute function, or
	// goai.ErrUnknownTool when the model requested a tool not in the
	// configured set. Nil on success.
	//
	// The standard encoding/json package cannot marshal the error
	// interface directly (it would emit "{}"). ToolResult implements
	// MarshalJSON to surface Error as a string ("Error.Error()") in
	// JSON output so the field is useful in observability / logging.
	Error error

	// IsError is a convenience boolean equivalent to Error != nil.
	IsError bool
}

// MarshalJSON implements json.Marshaler so that ToolResult round-trips
// through encoding/json in a useful form. The Error field (an interface)
// would otherwise marshal as "{}"; we emit it as a string using
// err.Error(). All other fields use their natural JSON representation.
//
// FIX 39 - empty-string error fidelity. When r.Error is non-nil but
// r.Error.Error() == "" (e.g. errors.New("")), we still emit the Error
// key (as "") and IsError=true. Previously the Error field had
// `omitempty`, so an empty message was dropped and UnmarshalJSON
// resurrected a synthesized placeholder instead of honoring the empty
// message the producer sent. Emitting Error unconditionally when
// IsError is true preserves the exact string (including "") across
// round-trips. Producers who never use empty-message errors are
// unaffected: the wire form gains at most one `"Error":""` field.
//
// FIX 40 - sentinel identity is NOT preserved. If the producer set
// r.Error to a sentinel (e.g. ErrUnknownTool from this package),
// UnmarshalJSON reconstructs a plain errors.New(msg); callers doing
// `errors.Is(r.Error, ErrUnknownTool)` after a JSON round-trip will get
// false. Reconstructing sentinels from strings requires a registry and
// is out of scope. Consumers relying on sentinel-based dispatch should
// either avoid JSON round-trips for that data or add an out-of-band
// error-code field.
func (r ToolResult) MarshalJSON() ([]byte, error) {
	type alias struct {
		ToolCallID string `json:"ToolCallID"`
		ToolName   string `json:"ToolName"`
		Output     string `json:"Output"`
		// FIX 39: pointer alias with omitempty so "" round-trips faithfully.
		//   - nil *string    → field absent in wire form (when r.Error == nil)
		//   - non-nil *string → field emitted (even if the pointee is "")
		// omitempty is load-bearing: it drops the key when the pointer is nil.
		// MarshalJSON below always assigns a non-nil *string whenever
		// r.Error != nil (even Error.Error() == ""), so the "Error":"" case
		// is emitted intentionally and survives UnmarshalJSON.
		Error   *string `json:"Error,omitempty"`
		IsError bool    `json:"IsError"`
	}
	a := alias{
		ToolCallID: r.ToolCallID,
		ToolName:   r.ToolName,
		Output:     r.Output,
		IsError:    r.IsError,
	}
	if r.Error != nil {
		s := r.Error.Error()
		a.Error = &s
	}
	return json.Marshal(a)
}

// UnmarshalJSON implements json.Unmarshaler so that ToolResult round-trips
// through encoding/json. The Error field was marshaled as a string (see
// MarshalJSON); here we reconstruct it via errors.New when present and
// preserve the invariant IsError == (Error != nil). If the JSON explicitly
// sets IsError=true but has no Error key, we synthesize a placeholder
// error so the invariant is preserved; conversely if Error is set but
// IsError is false, we coerce IsError to true.
//
// FIX 39 uses a pointer-to-string alias so that a producer's empty
// Error value ("") is distinguishable from "key absent entirely"
// (nil pointer). Empty string preserves the producer's errors.New("")
// faithfully; nil + IsError=true falls back to the placeholder.
// FIX 40: sentinel identity is not recovered (see MarshalJSON godoc).
func (r *ToolResult) UnmarshalJSON(data []byte) error {
	type alias struct {
		ToolCallID string  `json:"ToolCallID"`
		ToolName   string  `json:"ToolName"`
		Output     string  `json:"Output"`
		Error      *string `json:"Error,omitempty"`
		IsError    bool    `json:"IsError"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	r.ToolCallID = a.ToolCallID
	r.ToolName = a.ToolName
	r.Output = a.Output
	switch {
	case a.Error != nil:
		// FIX 39: empty-string error messages round-trip faithfully.
		// Previous code (non-pointer alias + omitempty) conflated
		// "absent" with "empty string", causing errors.New("") to
		// resurrect as a synthesized placeholder.
		r.Error = errors.New(*a.Error)
		r.IsError = true
	case a.IsError:
		// Explicit IsError=true with no Error key: synthesize placeholder.
		r.Error = errors.New("tool error (no message)")
		r.IsError = true
	default:
		r.Error = nil
		r.IsError = false
	}
	return nil
}

// Usage tracks token consumption for a request.
type Usage struct {
	InputTokens      int
	OutputTokens     int
	TotalTokens      int
	ReasoningTokens  int
	CacheReadTokens  int
	CacheWriteTokens int
}

// ModelCapabilities describes what features a model supports.
type ModelCapabilities struct {
	// Temperature indicates the model accepts a temperature parameter.
	Temperature bool

	// Reasoning indicates the model supports extended thinking/reasoning.
	Reasoning bool

	// Attachment indicates the model supports file attachments.
	Attachment bool

	// ToolCall indicates the model supports tool/function calling.
	ToolCall bool

	// InputModalities lists supported input types.
	InputModalities ModalitySet

	// OutputModalities lists supported output types.
	OutputModalities ModalitySet
}

// ModalitySet lists supported content modalities.
type ModalitySet struct {
	Text  bool
	Audio bool
	Image bool
	Video bool
	PDF   bool
}

// ResponseMetadata contains provider-specific response information.
type ResponseMetadata struct {
	// ID is the provider's response identifier.
	ID string

	// Model is the actual model used (may differ from requested).
	Model string

	// Headers are selected response headers.
	Headers map[string]string

	// ProviderMetadata contains provider-specific metadata (e.g. iterations,
	// context_management, container, citations, reasoning signatures).
	ProviderMetadata map[string]any
}

// ImageParams contains parameters for image generation.
type ImageParams struct {
	// Prompt describes the image to generate.
	Prompt string

	// N is the number of images to generate.
	N int

	// Size specifies dimensions (e.g. "1024x1024").
	Size string

	// AspectRatio (e.g. "16:9", "1:1"). Alternative to Size.
	AspectRatio string

	// ProviderOptions are provider-specific parameters.
	ProviderOptions map[string]any
}

// ImageResult is the response from image generation.
type ImageResult struct {
	Images []ImageData

	// ProviderMetadata contains provider-specific response data
	// (e.g. revisedPrompt).
	ProviderMetadata map[string]map[string]any

	// Usage tracks token or operation consumption (if reported by the provider).
	Usage Usage

	// Response contains provider-specific response metadata (ID, model, headers).
	Response ResponseMetadata
}

// ImageData contains a single generated image.
type ImageData struct {
	// Data is the raw image bytes.
	Data []byte

	// MediaType (e.g. "image/png").
	MediaType string
}

// ResponseFormat requests structured JSON output matching a schema.
// Used by GenerateObject/StreamObject to enable provider-specific JSON mode.
type ResponseFormat struct {
	// Name identifies the schema (used by OpenAI's json_schema mode).
	Name string

	// Schema is the JSON Schema that the output must conform to.
	Schema json.RawMessage
}

// EmbedParams contains parameters for an embedding request.
type EmbedParams struct {
	// ProviderOptions are provider-specific request parameters.
	ProviderOptions map[string]any
}

// EmbedResult is the response from embedding generation.
type EmbedResult struct {
	// Embeddings contains the generated vectors.
	Embeddings [][]float64

	// Usage tracks token consumption.
	Usage Usage

	// ProviderMetadata contains provider-specific response data.
	ProviderMetadata map[string]map[string]any

	// Response contains provider-specific response metadata (ID, model, headers).
	Response ResponseMetadata
}
