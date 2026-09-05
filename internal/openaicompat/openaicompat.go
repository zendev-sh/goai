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
	"sort"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/internal/sse"
	"github.com/zendev-sh/goai/provider"
)

// openAIKnownKeys lists provider option keys that applyProviderOptions handles
// explicitly and that must not be forwarded verbatim in the passthrough loop.
// Allocated once at package init to avoid per-request map allocation.
var openAIKnownKeys = map[string]bool{
	"structuredOutputs": true,
	"strictJsonSchema":  true,
	"useResponsesAPI":   true,
	"parallelToolCalls": true,
	"user":              true,
	"logprobs":          true,
	"metadata":          true,
	"safetyIdentifier":  true,
	"store":             true,
	"serviceTier":       true,
}

// openAIProtectedKeys lists wire-format keys that must not be overwritten by
// provider option passthrough. Allocated once at package init to avoid
// per-request map allocation.
var openAIProtectedKeys = map[string]bool{
	"model": true, "stream": true, "messages": true,
	"max_tokens": true, "max_completion_tokens": true,
	"temperature": true, "top_p": true, "stop": true,
	"seed": true, "frequency_penalty": true, "presence_penalty": true,
	"tools": true, "tool_choice": true, "response_format": true,
}

// Accumulation caps that bound worst-case memory from a malicious or
// misbehaving provider streaming unbounded data. Generous enough for
// legitimate large outputs (multi-hundred-thousand-token generations) while
// preventing client OOM. Declared as package vars so tests can lower them.
var (
	// maxToolCallArgsBytes caps the bytes buffered for a single streaming
	// tool call's argument fragments before they are finalized.
	maxToolCallArgsBytes int64 = 64 << 20 // 64 MiB

	// maxCitationsBytes caps the total bytes of top-level citation URLs
	// accumulated across a stream.
	maxCitationsBytes int64 = 64 << 20 // 64 MiB

	// maxResponseBodyBytes caps the non-streaming response body read in
	// DoGenerate / DoEmbed.
	maxResponseBodyBytes int64 = 64 << 20 // 64 MiB
)

// RequestConfig holds provider-specific settings for building requests.
type RequestConfig struct {
	// IncludeStreamOptions adds stream_options.include_usage to the request.
	IncludeStreamOptions bool

	// ExtraBody contains provider-specific fields to merge into the request body.
	ExtraBody map[string]any

	// IncludeReasoningContent controls serialization of the non-standard
	// reasoning_content field. Providers must opt in when their API requires it.
	IncludeReasoningContent bool

	// UseMaxCompletionTokens is a provider policy: when true the request body
	// always emits max_completion_tokens (renaming any max_tokens); when false
	// it always emits max_tokens. Nil keeps the modelID heuristic
	// (IsReasoningModel). It expresses a provider-wide wire-format preference,
	// NOT an inference about whether the model is a reasoning model. Providers
	// that accept both fields (e.g. Groq) force it true; routers to
	// heterogeneous upstreams (e.g. OpenRouter) leave it nil and let callers
	// opt in via a typed option.
	UseMaxCompletionTokens *bool

	// JsonSchemaAsJsonObject makes structured output emit
	// {"type":"json_object","schema":{...}} instead of
	// {"type":"json_schema","json_schema":{...}}. Fireworks only accepts the
	// former shape (item 44).
	JsonSchemaAsJsonObject bool

	// JsonObjectAsJsonSchema makes schema-less JSON mode emit a generic
	// {"type":"json_schema","json_schema":{...}} object instead of
	// {"type":"json_object"}. Perplexity Sonar only accepts text/json_schema
	// and rejects json_object (item 55).
	JsonObjectAsJsonSchema bool

	// SeedKey is the wire-format key used for the seed parameter. Defaults to
	// "seed"; Mistral uses "random_seed" (item 57).
	SeedKey string

	// FlatInputFile makes PDF parts serialize as the flat
	// {"type":"input_file","file_data":...} shape instead of the nested
	// {"type":"file","file":{...}} shape. Requesty requires the flat form (item 59).
	FlatInputFile bool
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
	body["messages"] = ConvertMessagesWithConfig(params.Messages, params.System, MessagesConfig{
		IncludeReasoningContent: cfg.IncludeReasoningContent,
		FlatInputFile:           cfg.FlatInputFile,
	})

	// Extract structuredOutputs and strictJsonSchema once -- used by both tools and response format.
	structuredOutputs := true // default true, matching Vercel
	strictJSON := false       // default false, matching Vercel
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
		seedKey := cfg.SeedKey
		if seedKey == "" {
			seedKey = "seed"
		}
		body[seedKey] = *params.Seed
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
				if cfg.JsonSchemaAsJsonObject {
					// Fireworks only accepts {"type":"json_object","schema":{...}}.
					body["response_format"] = map[string]any{
						"type":   "json_object",
						"schema": schema,
					}
				} else {
					body["response_format"] = map[string]any{
						"type": "json_schema",
						"json_schema": map[string]any{
							"name":   params.ResponseFormat.Name,
							"schema": schema,
							"strict": strictJSON,
						},
					}
				}
				schemaSet = true
			}
		}
		if !schemaSet {
			if cfg.JsonObjectAsJsonSchema {
				// Perplexity Sonar only accepts text/json_schema, not json_object.
				name := params.ResponseFormat.Name
				if name == "" {
					name = "json_schema"
				}
				body["response_format"] = map[string]any{
					"type": "json_schema",
					"json_schema": map[string]any{
						"name":   name,
						"schema": map[string]any{"type": "object"},
						"strict": false,
					},
				}
			} else {
				// Schema-less JSON mode (json_object) -- item 7.
				body["response_format"] = map[string]any{
					"type": "json_object",
				}
			}
		}
	}

	// Per-request headers (extracted in doHTTP before marshaling). The
	// Authorization header is reserved for the configured token source: a
	// caller-supplied Authorization is dropped so it can never strip or replace
	// the provider credential (Bedrock likewise sets auth last).
	if len(params.Headers) > 0 {
		body["_headers"] = sanitizeRequestHeaders(params.Headers)
	}

	// Reasoning models (o-series, gpt-5+, codex) require
	// max_completion_tokens instead of max_tokens. The rename is keyed
	// on the model id, not on reasoning_effort being present: a
	// reasoning model rejects max_tokens outright (Azure gpt-5 returns
	// `Unsupported parameter: 'max_tokens'`) whether or not the caller
	// passed a reasoning_effort.
	// The caller may know what the model id cannot tell (Azure: the id is the
	// deployment name); nil falls back to the id heuristic.
	useMaxCompletionTokens := IsReasoningModel(modelID)
	if cfg.UseMaxCompletionTokens != nil {
		useMaxCompletionTokens = *cfg.UseMaxCompletionTokens
	}
	if useMaxCompletionTokens {
		if v, ok := body["max_tokens"]; ok {
			body["max_completion_tokens"] = v
			delete(body, "max_tokens")
		}
	}

	return body
}

// IsReasoningModel reports whether modelID is an OpenAI-family reasoning
// model (o-series, gpt-5+, codex). Reasoning models require
// max_completion_tokens in place of max_tokens, do not accept a
// temperature, and support reasoning. provider/openai consumes this
// predicate for capability detection.
func IsReasoningModel(modelID string) bool {
	id := strings.ToLower(modelID)
	// o-series reasoning models (o1, o3, o4, ...).
	if len(id) >= 2 && id[0] == 'o' && id[1] >= '0' && id[1] <= '9' {
		return true
	}
	// GPT-5+ models (gpt-5-chat is NOT a reasoning model, per Vercel).
	if strings.HasPrefix(id, "gpt-5") && !strings.HasPrefix(id, "gpt-5-chat") {
		return true
	}
	// codex- prefix models.
	return strings.HasPrefix(id, "codex-")
}

// sanitizeRequestHeaders removes the Authorization header (case-insensitively)
// from per-request headers so a caller cannot override the configured token.
// Returns a fresh map; the input is never mutated.
func sanitizeRequestHeaders(headers map[string]string) map[string]string {
	out := make(map[string]string, len(headers))
	for k, v := range headers {
		if strings.EqualFold(k, "Authorization") {
			continue
		}
		out[k] = v
	}
	return out
}

// applyProviderOptions maps known provider options to their wire-format keys,
// then passes through any unknown keys directly.
func applyProviderOptions(body map[string]any, opts map[string]any) {
	if opts == nil {
		return
	}

	if v, ok := opts["parallelToolCalls"]; ok {
		body["parallel_tool_calls"] = v
	}
	if v, ok := opts["user"]; ok {
		body["user"] = v
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
	}
	if v, ok := opts["metadata"]; ok {
		body["metadata"] = v
	}
	if v, ok := opts["safetyIdentifier"]; ok {
		body["safety_identifier"] = v
	}
	if v, ok := opts["store"]; ok {
		body["store"] = v
	}
	if v, ok := opts["serviceTier"]; ok {
		body["service_tier"] = v
	}

	// Pass through any remaining unknown keys (not handled above, not protected).
	for k, v := range opts {
		if !openAIKnownKeys[k] && !openAIProtectedKeys[k] {
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
			Reasoning        string          `json:"reasoning,omitempty"`
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

// promptTokensDetails carries the standard prompt_tokens_details.cached_tokens
// field, plus the top-level cache counters some providers emit instead (item 46).
type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type streamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	// Top-level cache counters emitted by Together/DeepInfra/DeepSeek
	// (prompt_cache_hit_tokens / prompt_cache_miss_tokens) and OpenRouter
	// (cached_tokens) when prompt_tokens_details.cached_tokens is absent (item 46).
	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
	CachedTokens          int `json:"cached_tokens"`

	PromptTokensDetails *promptTokensDetails `json:"prompt_tokens_details,omitempty"`

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
		id      string
		name    string
		args    strings.Builder
		size    int64
		started bool
	}
	activeTools := make(map[int]*activeToolCall)
	var usage provider.Usage
	var responseMeta provider.ResponseMetadata
	providerMeta := map[string]any{}
	var citations []string
	var citationsSize int64

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
			// Item 1: when a provider omits total_tokens, fall back to the
			// input+output sum instead of leaving TotalTokens at 0.
			if usage.TotalTokens <= 0 && usage.InputTokens+usage.OutputTokens > 0 {
				usage.TotalTokens = usage.InputTokens + usage.OutputTokens
			}
			usage.CacheReadTokens = cacheReadTokens(resp.Usage.PromptTokensDetails, resp.Usage.PromptCacheHitTokens, resp.Usage.CachedTokens)
			usage.InputTokens -= usage.CacheReadTokens
			if usage.InputTokens < 0 {
				usage.InputTokens = 0
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
			for _, c := range resp.Citations {
				if citationsSize+int64(len(c)) > maxCitationsBytes {
					provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: fmt.Errorf("stream citations exceed %d byte limit", maxCitationsBytes),
					})
					return
				}
				citationsSize += int64(len(c))
				citations = append(citations, c)
			}
		}

		if len(resp.Choices) == 0 {
			continue
		}

		choice := resp.Choices[0]
		delta := choice.Delta

		// Reasoning content -- emitted before text: a delta can carry both
		// fields at the reasoning→answer transition, and reasoning always
		// precedes the answer. Prefer reasoning_content (DeepSeek native),
		// fall back to reasoning (OpenRouter).
		if delta.ReasoningContent != "" {
			if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkReasoning, Text: delta.ReasoningContent}) {
				return
			}
		} else if delta.Reasoning != "" {
			if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkReasoning, Text: delta.Reasoning}) {
				return
			}
		}

		// Text content -- handle both string and array [{type:"text",text:"..."}] formats.
		if text := extractTextContent(delta.Content); text != "" {
			if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: text}) {
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
			active := activeTools[tc.Index]
			if active == nil {
				active = &activeToolCall{}
				activeTools[tc.Index] = active
			}

			if tc.ID != "" && !active.started {
				active.id = tc.ID
			} else if active.id == "" && tc.Function.Name != "" {
				// First chunk for a tool call with no ID -- generate one.
				active.id = generateToolCallID()
			}
			if tc.Function.Name != "" {
				active.name = tc.Function.Name
			}

			if active.id != "" && active.name != "" && !active.started {
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type:       provider.ChunkToolCallStreamStart,
					ToolCallID: active.id,
					ToolName:   active.name,
				}) {
					return
				}
				active.started = true

				// Flush any arguments that arrived before the name/id resolved
				// as a delta (streaming progress). Do NOT finalize a
				// ChunkToolCall here: more fragments may still arrive and the
				// call is only complete at the finish reason (item 4).
				if pending := active.args.String(); pending != "" {
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:       provider.ChunkToolCallDelta,
						ToolCallID: active.id,
						ToolName:   active.name,
						ToolInput:  pending,
					}) {
						return
					}
				}
			}

			if tc.Function.Arguments != "" {
				if active.size+int64(len(tc.Function.Arguments)) > maxToolCallArgsBytes {
					provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: fmt.Errorf("tool call arguments exceed %d byte limit", maxToolCallArgsBytes),
					})
					return
				}
				active.size += int64(len(tc.Function.Arguments))
				active.args.WriteString(tc.Function.Arguments)

				// Emit delta for UI streaming progress.
				if active.started {
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:       provider.ChunkToolCallDelta,
						ToolCallID: active.id,
						ToolName:   active.name,
						ToolInput:  tc.Function.Arguments,
					}) {
						return
					}
				}
			}
		}

		// Finish reason -- flush remaining accumulated args.
		if choice.FinishReason != "" {
			if choice.FinishReason == "tool_calls" {
				// Iterate in ascending index order so ChunkToolCall events for
				// parallel tool calls are deterministic and in index order.
				indices := make([]int, 0, len(activeTools))
				for idx := range activeTools {
					indices = append(indices, idx)
				}
				sort.Ints(indices)
				for _, idx := range indices {
					active := activeTools[idx]
					if !active.started {
						continue
					}
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:       provider.ChunkToolCall,
						ToolCallID: active.id,
						ToolName:   active.name,
						ToolInput:  active.args.String(),
					}) {
						return
					}
					// Finalized at the finish reason; the EOF finalize pass
					// must not re-emit it.
					delete(activeTools, idx)
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

	// Finalize any tool call that never saw a finish_reason (item 4): the
	// accumulated arguments are emitted once as a complete ChunkToolCall.
	// All started (name/id resolved) calls are flushed first, in ascending
	// index order, before a single ChunkError is surfaced for any tool whose
	// name/id never resolved -- emitting the error must not drop the fully
	// resolved sibling calls. A started call with no remaining arguments is a
	// valid zero-argument tool call, not an error.
	indices := make([]int, 0, len(activeTools))
	for idx := range activeTools {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	unresolved := false
	for _, idx := range indices {
		active := activeTools[idx]
		if !active.started {
			// A tool call that received at least one delta fragment but whose
			// name/id never resolved is malformed: surface a single error
			// instead of silently dropping it (matches Vercel AI SDK, which
			// throws in this case).
			unresolved = true
			continue
		}
		if !provider.TrySend(ctx, out, provider.StreamChunk{
			Type:       provider.ChunkToolCall,
			ToolCallID: active.id,
			ToolName:   active.name,
			ToolInput:  active.args.String(),
		}) {
			return
		}
		active.args.Reset()
	}
	if unresolved {
		if !provider.TrySend(ctx, out, provider.StreamChunk{
			Type:  provider.ChunkError,
			Error: fmt.Errorf("tool call arguments received but tool name/id never resolved"),
		}) {
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
			Role             string          `json:"role"`
			Content          json.RawMessage `json:"content"`
			ReasoningContent string          `json:"reasoning_content,omitempty"`
			Reasoning        string          `json:"reasoning,omitempty"`
			ToolCalls        []struct {
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
		PromptTokens            int                  `json:"prompt_tokens"`
		CompletionTokens        int                  `json:"completion_tokens"`
		TotalTokens             int                  `json:"total_tokens"`
		PromptCacheHitTokens    int                  `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens   int                  `json:"prompt_cache_miss_tokens"`
		CachedTokens            int                  `json:"cached_tokens"`
		PromptTokensDetails     *promptTokensDetails `json:"prompt_tokens_details,omitempty"`
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
		result.Reasoning = choice.Message.ReasoningContent
		if result.Reasoning == "" {
			result.Reasoning = choice.Message.Reasoning
		}
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
		// Item 1: when a provider omits total_tokens, fall back to the
		// input+output sum instead of leaving TotalTokens at 0.
		if result.Usage.TotalTokens <= 0 && result.Usage.InputTokens+result.Usage.OutputTokens > 0 {
			result.Usage.TotalTokens = result.Usage.InputTokens + result.Usage.OutputTokens
		}
		result.Usage.CacheReadTokens = cacheReadTokens(resp.Usage.PromptTokensDetails, resp.Usage.PromptCacheHitTokens, resp.Usage.CachedTokens)
		result.Usage.InputTokens -= result.Usage.CacheReadTokens
		if result.Usage.InputTokens < 0 {
			result.Usage.InputTokens = 0
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

// cacheReadTokens resolves the number of prompt tokens served from cache. It
// prefers prompt_tokens_details.cached_tokens, then falls back to the top-level
// prompt_cache_hit_tokens (Together/DeepInfra/DeepSeek) and finally OpenRouter's
// top-level cached_tokens (item 46). prompt_cache_miss_tokens is intentionally
// not counted here: miss tokens are computed, not read from cache.
func cacheReadTokens(details *promptTokensDetails, hitTokens, openRouterCached int) int {
	if details != nil && details.CachedTokens > 0 {
		return details.CachedTokens
	}
	if hitTokens > 0 {
		return hitTokens
	}
	return openRouterCached
}
