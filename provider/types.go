package provider

import (
	"context"
	"encoding/json"
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

	// Error (for ChunkError).
	Error error

	// Response metadata (populated on ChunkFinish with ID, Model from the provider).
	Response ResponseMetadata

	// Metadata for provider-specific data (e.g. thoughtSignature).
	Metadata map[string]any
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

	// Tool call fields (for PartToolCall).
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
