// Package openaicompat provides shared request building and response parsing
// for OpenAI-compatible API providers (OpenAI, OpenRouter, Groq, DeepInfra, etc.).
//
// Provider-specific packages call BuildRequest to construct the wire format,
// and ParseStream/ParseResponse to decode responses into GoAI provider types.
package openaicompat

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// RequestConfig holds provider-specific settings for building requests.
type RequestConfig struct {
	// IncludeStreamOptions adds stream_options.include_usage to the request.
	IncludeStreamOptions bool

	// ExtraBody contains provider-specific fields to merge into the request body.
	ExtraBody map[string]any
}

// BuildRequest creates a standard OpenAI chat/completions request body.
func BuildRequest(params provider.GenerateParams, modelID string, streaming bool, cfg RequestConfig) map[string]any {
	body := map[string]any{
		"model":  modelID,
		"stream": streaming,
	}

	if params.MaxOutputTokens > 0 {
		body["max_tokens"] = params.MaxOutputTokens
	}

	if streaming && cfg.IncludeStreamOptions {
		body["stream_options"] = map[string]any{
			"include_usage": true,
		}
	}

	// Messages
	body["messages"] = ConvertMessages(params.Messages, params.System)

	// Extract structuredOutputs and strictJsonSchema once -- used by both tools and response format.
	structuredOutputs := true // default true, matching Vercel
	strictJSON := false      // default false, matching Vercel
	if v, ok := params.ProviderOptions["structuredOutputs"]; ok {
		if b, ok := v.(bool); ok {
			structuredOutputs = b
		}
	}
	if v, ok := params.ProviderOptions["strictJsonSchema"]; ok {
		if b, ok := v.(bool); ok {
			strictJSON = b
		}
	}

	// Tools -- with optional strict JSON schema support via ProviderOptions.
	// Provider-defined tools (ProviderDefinedType != "") are sent as {"type": providerType, ...opts}
	// matching Groq's browser_search and similar provider tool formats.
	if len(params.Tools) > 0 {
		tools := make([]map[string]any, len(params.Tools))
		for i, t := range params.Tools {
			if t.ProviderDefinedType != "" {
				// Provider-defined tool -- send type + options as flat object.
				tool := map[string]any{
					"type": t.ProviderDefinedType,
				}
				for k, v := range t.ProviderDefinedOptions {
					tool[k] = v
				}
				tools[i] = tool
			} else {
				// Regular function tool.
				var schema any
				if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
					schema = map[string]any{}
				}
				fn := map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  schema,
				}
				if structuredOutputs {
					fn["strict"] = strictJSON
				}
				tools[i] = map[string]any{
					"type":     "function",
					"function": fn,
				}
			}
		}
		body["tools"] = tools
	}

	// Tool choice
	if params.ToolChoice != "" {
		switch params.ToolChoice {
		case "auto", "none", "required":
			body["tool_choice"] = params.ToolChoice
		default:
			// Specific tool name
			body["tool_choice"] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": params.ToolChoice,
				},
			}
		}
	}

	// Temperature
	if params.Temperature != nil {
		body["temperature"] = *params.Temperature
	}
	if params.TopP != nil {
		body["top_p"] = *params.TopP
	}
	if params.TopK != nil {
		body["top_k"] = *params.TopK
	}
	if params.FrequencyPenalty != nil {
		body["frequency_penalty"] = *params.FrequencyPenalty
	}
	if params.PresencePenalty != nil {
		body["presence_penalty"] = *params.PresencePenalty
	}
	if params.Seed != nil {
		body["seed"] = *params.Seed
	}

	// Stop sequences
	if len(params.StopSequences) > 0 {
		body["stop"] = params.StopSequences
	}

	// Extra body (provider-specific)
	for k, v := range cfg.ExtraBody {
		body[k] = v
	}

	// Provider options passthrough -- maps known provider options to wire format,
	// then passes through any remaining arbitrary fields.
	applyProviderOptions(body, params.ProviderOptions)

	// Response format (structured output / JSON mode).
	if params.ResponseFormat != nil {
		schemaSet := false
		if structuredOutputs && len(params.ResponseFormat.Schema) > 0 {
			var schema any
			if err := json.Unmarshal(params.ResponseFormat.Schema, &schema); err == nil {
				body["response_format"] = map[string]any{
					"type": "json_schema",
					"json_schema": map[string]any{
						"name":   params.ResponseFormat.Name,
						"schema": schema,
						"strict": strictJSON,
					},
				}
				schemaSet = true
			}
		}
		if !schemaSet {
			// Schema-less JSON mode (json_object) -- item 7.
			body["response_format"] = map[string]any{
				"type": "json_object",
			}
		}
	}

	// Per-request headers (extracted in doHTTP before marshaling).
	if len(params.Headers) > 0 {
		body["_headers"] = params.Headers
	}

	// Reasoning models use max_completion_tokens instead of max_tokens.
	// Follows Vercel AI SDK pattern: always set max_tokens first, then rename
	// if reasoning_effort is present (indicating a reasoning model).
	if _, hasReasoning := body["reasoning_effort"]; hasReasoning {
		if v, ok := body["max_tokens"]; ok {
			body["max_completion_tokens"] = v
			delete(body, "max_tokens")
		}
	}

	return body
}

// providerOptionsBaseKnown holds option keys that are always consumed by the
// provider layer and must never be forwarded to the wire format.
// Keys that are conditionally consumed (parallelToolCalls, user, logprobs, etc.)
// are tracked in a local copy inside applyProviderOptions.
var providerOptionsBaseKnown = map[string]bool{
	"structuredOutputs": true,
	"strictJsonSchema":  true,
	"useResponsesAPI":   true, // consumed by openai.shouldUseResponsesAPI
}

// providerOptionsProtected holds wire-format keys that must not be overwritten
// by arbitrary provider option pass-through.
var providerOptionsProtected = map[string]bool{
	"model": true, "stream": true, "messages": true,
	"max_tokens": true, "max_completion_tokens": true,
	"temperature": true, "top_p": true, "stop": true,
	"seed": true, "frequency_penalty": true, "presence_penalty": true,
	"tools": true, "tool_choice": true, "response_format": true,
}

// applyProviderOptions maps known provider options to their wire-format keys,
// then passes through any unknown keys directly.
func applyProviderOptions(body map[string]any, opts map[string]any) {
	if opts == nil {
		return
	}

	// Start with a local copy of the base known set so we can add keys that
	// are conditionally consumed below without mutating the package-level var.
	knownKeys := make(map[string]bool, len(providerOptionsBaseKnown)+8)
	for k, v := range providerOptionsBaseKnown {
		knownKeys[k] = v
	}

	if v, ok := opts["parallelToolCalls"]; ok {
		body["parallel_tool_calls"] = v
		knownKeys["parallelToolCalls"] = true
	}
	if v, ok := opts["user"]; ok {
		body["user"] = v
		knownKeys["user"] = true
	}
	if v, ok := opts["logprobs"]; ok {
		switch lp := v.(type) {
		case bool:
			if lp {
				body["logprobs"] = true
				body["top_logprobs"] = 0
			}
		case int:
			body["logprobs"] = true
			body["top_logprobs"] = lp
		case float64:
			body["logprobs"] = true
			body["top_logprobs"] = int(lp)
		}
		knownKeys["logprobs"] = true
	}
	if v, ok := opts["metadata"]; ok {
		body["metadata"] = v
		knownKeys["metadata"] = true
	}
	if v, ok := opts["safetyIdentifier"]; ok {
		body["safety_identifier"] = v
		knownKeys["safetyIdentifier"] = true
	}
	if v, ok := opts["store"]; ok {
		body["store"] = v
		knownKeys["store"] = true
	}
	if v, ok := opts["serviceTier"]; ok {
		body["service_tier"] = v
		knownKeys["serviceTier"] = true
	}

	// Pass through any remaining unknown keys.
	for k, v := range opts {
		if !knownKeys[k] && !providerOptionsProtected[k] {
			body[k] = v
		}
	}
}

// streamResponse is the JSON structure of an OpenAI SSE data line.
type streamResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`

	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role             string          `json:"role,omitempty"`
			Content          json.RawMessage `json:"content,omitempty"`
			ReasoningContent string          `json:"reasoning_content,omitempty"`
			ToolCalls        []struct {
				Index    int    `json:"index"`
				ID       string `json:"id,omitempty"`
				Type     string `json:"type,omitempty"`
				Function struct {
					Name      string `json:"name,omitempty"`
					Arguments string `json:"arguments,omitempty"`
				} `json:"function,omitempty"`
			} `json:"tool_calls,omitempty"`
			Annotations []annotation `json:"annotations,omitempty"`
		} `json:"delta"`
		Logprobs     *json.RawMessage `json:"logprobs,omitempty"`
		FinishReason string           `json:"finish_reason,omitempty"`
	} `json:"choices"`

	// Citations is a top-level URL array returned by xAI and Perplexity
	// (as opposed to per-annotation citations from OpenAI).
	Citations []string `json:"citations,omitempty"`

	Usage *streamUsage `json:"usage,omitempty"`
}

type streamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitempty"`

	CompletionTokensDetails *struct {
		ReasoningTokens          int `json:"reasoning_tokens"`
		AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details,omitempty"`
}

// annotation represents a url_citation annotation in a chat completions response.
type annotation struct {
	Type        string `json:"type"`
	URLCitation *struct {
		URL        string `json:"url"`
		Title      string `json:"title"`
		StartIndex int    `json:"start_index"`
		EndIndex   int    `json:"end_index"`
	} `json:"url_citation,omitempty"`
}

// ParseStream reads SSE from the scanner and sends StreamChunks on out.
// It handles tool call accumulation, usage normalization, and error detection.
// The out channel is closed when the stream ends.
func ParseStream(ctx context.Context, scanner *sse.Scanner, out chan<- provider.StreamChunk) {
	defer close(out)
	// Track active tool calls by index, accumulating argument fragments.
	type activeToolCall struct {
		id   string
		name string
		args strings.Builder
	}
	activeTools := make(map[int]*activeToolCall)
	var usage provider.Usage
	var responseMeta provider.ResponseMetadata
	providerMeta := map[string]any{}
	var citations []string

	for {
		data, ok := scanner.Next()
		if !ok {
			break
		}

		// Check for stream error events first (they are valid JSON but not chat responses).
		if streamErr := goai.ClassifyStreamError([]byte(data)); streamErr != nil {
			provider.TrySend(ctx, out, provider.StreamChunk{
				Type:  provider.ChunkError,
				Error: streamErr,
			})
			return
		}

		var resp streamResponse
		if err := json.Unmarshal([]byte(data), &resp); err != nil {
			continue
		}

		// Capture response ID and model from the first chunk that has them.
		if responseMeta.ID == "" && resp.ID != "" {
			responseMeta.ID = resp.ID
		}
		if responseMeta.Model == "" && resp.Model != "" {
			responseMeta.Model = resp.Model
		}

		// Handle usage -- normalize InputTokens to exclude cached tokens
		// (matching Anthropic convention). Item 1: compute TotalTokens correctly.
		if resp.Usage != nil {
			usage.InputTokens = resp.Usage.PromptTokens
			usage.OutputTokens = resp.Usage.CompletionTokens
			usage.TotalTokens = resp.Usage.TotalTokens
			if resp.Usage.PromptTokensDetails != nil {
				usage.CacheReadTokens = resp.Usage.PromptTokensDetails.CachedTokens
				usage.InputTokens -= usage.CacheReadTokens
				if usage.InputTokens < 0 {
					usage.InputTokens = 0
				}
			}
			// Item 10: extract prediction tokens + reasoning tokens from completion_tokens_details.
			if resp.Usage.CompletionTokensDetails != nil {
				usage.ReasoningTokens = resp.Usage.CompletionTokensDetails.ReasoningTokens
				if resp.Usage.CompletionTokensDetails.AcceptedPredictionTokens > 0 {
					providerMeta["acceptedPredictionTokens"] = resp.Usage.CompletionTokensDetails.AcceptedPredictionTokens
				}
				if resp.Usage.CompletionTokensDetails.RejectedPredictionTokens > 0 {
					providerMeta["rejectedPredictionTokens"] = resp.Usage.CompletionTokensDetails.RejectedPredictionTokens
				}
			}
		}

		// Capture top-level citations (xAI, Perplexity) -- accumulate across chunks.
		if len(resp.Citations) > 0 {
			citations = append(citations, resp.Citations...)
		}

		if len(resp.Choices) == 0 {
			continue
		}

		choice := resp.Choices[0]
		delta := choice.Delta

		// Text content -- handle both string and array [{type:"text",text:"..."}] formats.
		if text := extractTextContent(delta.Content); text != "" {
			if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: text}) {
				return
			}
		}

		// Reasoning content
		if delta.ReasoningContent != "" {
			if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkReasoning, Text: delta.ReasoningContent}) {
				return
			}
		}

		// Item 12: extract logprobs from stream.
		if choice.Logprobs != nil {
			var lp any
			if json.Unmarshal(*choice.Logprobs, &lp) == nil && lp != nil {
				providerMeta["logprobs"] = lp
			}
		}

		// Item 11: extract annotations (url_citation) from stream.
		for _, ann := range delta.Annotations {
			if ann.Type == "url_citation" && ann.URLCitation != nil {
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type: provider.ChunkText,
					Metadata: map[string]any{
						"source": provider.Source{
							Type:       "url",
							URL:        ann.URLCitation.URL,
							Title:      ann.URLCitation.Title,
							StartIndex: ann.URLCitation.StartIndex,
							EndIndex:   ann.URLCitation.EndIndex,
						},
					},
				}) {
					return
				}
			}
		}

		// Tool calls -- item 17: generate fallback ID if provider omits it.
		for _, tc := range delta.ToolCalls {
			tcID := tc.ID
			if tcID == "" && tc.Function.Name != "" {
				// First chunk for a tool call with no ID -- generate one.
				if active := activeTools[tc.Index]; active == nil {
					tcID = generateToolCallID()
				}
			}

			if tcID != "" {
				activeTools[tc.Index] = &activeToolCall{id: tcID, name: tc.Function.Name}
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type:       provider.ChunkToolCallStreamStart,
					ToolCallID: tcID,
					ToolName:   tc.Function.Name,
				}) {
					return
				}
			}

			if tc.Function.Arguments != "" {
				active := activeTools[tc.Index]
				if active == nil {
					active = &activeToolCall{}
					activeTools[tc.Index] = active
				}
				active.args.WriteString(tc.Function.Arguments)

				// Emit delta for UI streaming progress.
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type:       provider.ChunkToolCallDelta,
					ToolCallID: active.id,
					ToolName:   active.name,
					ToolInput:  tc.Function.Arguments,
				}) {
					return
				}

				// Emit ChunkToolCall when accumulated args form valid JSON.
				if accumulated := active.args.String(); json.Valid([]byte(accumulated)) {
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:       provider.ChunkToolCall,
						ToolCallID: active.id,
						ToolName:   active.name,
						ToolInput:  accumulated,
					}) {
						return
					}
					active.args.Reset()
				}
			}
		}

		// Finish reason -- flush remaining accumulated args.
		if choice.FinishReason != "" {
			if choice.FinishReason == "tool_calls" {
				for _, active := range activeTools {
					if remaining := active.args.String(); remaining != "" {
						if !provider.TrySend(ctx, out, provider.StreamChunk{
							Type:       provider.ChunkToolCall,
							ToolCallID: active.id,
							ToolName:   active.name,
							ToolInput:  remaining,
						}) {
							return
						}
						active.args.Reset()
					}
				}
			}
			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:         provider.ChunkStepFinish,
				FinishReason: mapFinishReason(choice.FinishReason),
			}) {
				return
			}
		}
	}

	// Stream ended (DONE or EOF).
	if err := scanner.Err(); err != nil {
		if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: fmt.Errorf("reading stream: %w", err)}) {
			return
		}
		return
	}

	chunk := provider.StreamChunk{
		Type:     provider.ChunkFinish,
		Usage:    usage,
		Response: responseMeta,
	}
	if len(providerMeta) > 0 || len(citations) > 0 {
		if chunk.Metadata == nil {
			chunk.Metadata = map[string]any{}
		}
		if len(providerMeta) > 0 {
			chunk.Metadata["providerMetadata"] = map[string]map[string]any{"openai": providerMeta}
		}
		if len(citations) > 0 {
			sources := make([]provider.Source, len(citations))
			for i, c := range citations {
				sources[i] = provider.Source{
					Type: "url",
					URL:  c,
					ID:   fmt.Sprintf("citation_%d", i),
				}
			}
			chunk.Metadata["sources"] = sources
		}
	}
	if !provider.TrySend(ctx, out, chunk) {
		return
	}
}

// chatResponse is the JSON structure of a non-streaming OpenAI response.
type chatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role        string          `json:"role"`
			Content     json.RawMessage `json:"content"`
			ToolCalls   []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
			Annotations []annotation `json:"annotations,omitempty"`
		} `json:"message"`
		Logprobs     *json.RawMessage `json:"logprobs,omitempty"`
		FinishReason string           `json:"finish_reason"`
	} `json:"choices"`

	// Citations is a top-level URL array returned by xAI and Perplexity.
	Citations []string `json:"citations,omitempty"`

	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details,omitempty"`
		CompletionTokensDetails *struct {
			ReasoningTokens          int `json:"reasoning_tokens"`
			AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
		} `json:"completion_tokens_details,omitempty"`
	} `json:"usage,omitempty"`
}

// ParseResponse parses a non-streaming chat/completions JSON response.
func ParseResponse(body []byte) (*provider.GenerateResult, error) {
	var resp chatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	result := &provider.GenerateResult{
		Response: provider.ResponseMetadata{
			ID:    resp.ID,
			Model: resp.Model,
		},
	}

	providerMeta := map[string]any{}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		result.Text = extractTextContent(choice.Message.Content)
		result.FinishReason = mapFinishReason(choice.FinishReason)

		for _, tc := range choice.Message.ToolCalls {
			id := tc.ID
			if id == "" {
				id = generateToolCallID() // Item 17: fallback ID.
			}
			result.ToolCalls = append(result.ToolCalls, provider.ToolCall{
				ID:    id,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			})
		}

		// Item 11: extract annotations (url_citation).
		for _, ann := range choice.Message.Annotations {
			if ann.Type == "url_citation" && ann.URLCitation != nil {
				result.Sources = append(result.Sources, provider.Source{
					Type:       "url",
					URL:        ann.URLCitation.URL,
					Title:      ann.URLCitation.Title,
					StartIndex: ann.URLCitation.StartIndex,
					EndIndex:   ann.URLCitation.EndIndex,
				})
			}
		}

		// Item 12: extract logprobs.
		if choice.Logprobs != nil {
			var lp any
			if json.Unmarshal(*choice.Logprobs, &lp) == nil && lp != nil {
				providerMeta["logprobs"] = lp
			}
		}
	}

	// Top-level citations (xAI, Perplexity) -- simple URL array.
	for i, c := range resp.Citations {
		result.Sources = append(result.Sources, provider.Source{
			Type: "url",
			URL:  c,
			ID:   fmt.Sprintf("citation_%d", i),
		})
	}

	// Item 1: compute TotalTokens correctly from the response.
	if resp.Usage != nil {
		result.Usage.InputTokens = resp.Usage.PromptTokens
		result.Usage.OutputTokens = resp.Usage.CompletionTokens
		result.Usage.TotalTokens = resp.Usage.TotalTokens
		if resp.Usage.PromptTokensDetails != nil {
			result.Usage.CacheReadTokens = resp.Usage.PromptTokensDetails.CachedTokens
			result.Usage.InputTokens -= result.Usage.CacheReadTokens
			if result.Usage.InputTokens < 0 {
				result.Usage.InputTokens = 0
			}
		}
		// Item 10: extract prediction tokens + reasoning tokens.
		if resp.Usage.CompletionTokensDetails != nil {
			result.Usage.ReasoningTokens = resp.Usage.CompletionTokensDetails.ReasoningTokens
			if resp.Usage.CompletionTokensDetails.AcceptedPredictionTokens > 0 {
				providerMeta["acceptedPredictionTokens"] = resp.Usage.CompletionTokensDetails.AcceptedPredictionTokens
			}
			if resp.Usage.CompletionTokensDetails.RejectedPredictionTokens > 0 {
				providerMeta["rejectedPredictionTokens"] = resp.Usage.CompletionTokensDetails.RejectedPredictionTokens
			}
		}
	}

	if len(providerMeta) > 0 {
		result.ProviderMetadata = map[string]map[string]any{"openai": providerMeta}
	}

	return result, nil
}

// extractTextContent extracts text from a content field that can be either a
// JSON string or an array of content objects [{type:"text",text:"..."}].
// Matches Vercel AI SDK's Mistral provider which handles union(string | array).
func extractTextContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	// Fast path: try string first (most common case).
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}

	// Array path: [{type:"text",text:"..."}, {type:"thinking",thinking:"..."}]
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "text" && p.Text != "" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}

	return ""
}

// mapFinishReason converts OpenAI wire format finish reasons to GoAI FinishReason.
func mapFinishReason(reason string) provider.FinishReason {
	switch reason {
	case "stop":
		return provider.FinishStop
	case "tool_calls":
		return provider.FinishToolCalls
	case "length":
		return provider.FinishLength
	case "content_filter":
		return provider.FinishContentFilter
	default:
		return provider.FinishOther
	}
}

// generateToolCallID generates a random tool call ID when the provider omits one.
// Follows the format "call_" + random hex string (item 17).
func generateToolCallID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "call_" + hex.EncodeToString(b)
}
