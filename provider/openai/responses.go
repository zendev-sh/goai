package openai

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/zendev-sh/goai"
	"github.com/zendev-sh/goai/provider"
)

// buildResponsesRequest creates an OpenAI Responses API request body.
func buildResponsesRequest(params provider.GenerateParams, modelID string, streaming bool) map[string]any {
	body := map[string]any{
		"model":  modelID,
		"stream": streaming,
	}

	// System prompt goes in "instructions" field.
	if params.System != "" {
		body["instructions"] = params.System
	}

	// Messages → Responses API "input" format.
	body["input"] = convertToResponsesInput(params.Messages)

	if params.MaxOutputTokens > 0 {
		body["max_output_tokens"] = params.MaxOutputTokens
	}

	// Tools use flat format in Responses API.
	if len(params.Tools) > 0 {
		tools := make([]map[string]any, len(params.Tools))
		for i, t := range params.Tools {
			if t.ProviderDefinedType != "" {
				// Provider-defined tool (web_search, etc.) -- type is the tool type.
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
				tools[i] = map[string]any{
					"type":        "function",
					"name":        t.Name,
					"description": t.Description,
					"parameters":  schema,
				}
			}
		}
		body["tools"] = tools
	}

	// Tool choice.
	if params.ToolChoice != "" {
		switch params.ToolChoice {
		case "auto", "none", "required":
			body["tool_choice"] = params.ToolChoice
		default:
			body["tool_choice"] = map[string]any{
				"type": "function",
				"name": params.ToolChoice,
			}
		}
	}

	// Temperature/TopP (only if explicitly set -- reasoning models typically omit these).
	if params.Temperature != nil {
		body["temperature"] = *params.Temperature
	}
	if params.TopP != nil {
		body["top_p"] = *params.TopP
	}

	// Stop sequences.
	if len(params.StopSequences) > 0 {
		body["stop"] = params.StopSequences
	}

	// Response format (structured output / JSON mode).
	// Items 7, 8, 9: support json_object, structuredOutputs toggle, strictJsonSchema.
	if params.ResponseFormat != nil {
		structuredOutputs := true
		strictJSON := false
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

		text := getOrCreateMap(body, "text")
		schemaSet := false
		if structuredOutputs && len(params.ResponseFormat.Schema) > 0 {
			var schema any
			if err := json.Unmarshal(params.ResponseFormat.Schema, &schema); err == nil {
				text["format"] = map[string]any{
					"type":   "json_schema",
					"name":   params.ResponseFormat.Name,
					"schema": schema,
					"strict": strictJSON,
				}
				schemaSet = true
			}
		}
		if !schemaSet {
			// Schema-less JSON mode (json_object) -- item 7.
			text["format"] = map[string]any{
				"type": "json_object",
			}
		}
		body["text"] = text
	}

	// Provider options passthrough.
	applyResponsesProviderOptions(body, params.ProviderOptions)

	// Per-request headers (extracted in doHTTP before marshaling).
	if len(params.Headers) > 0 {
		body["_headers"] = params.Headers
	}

	return body
}

// applyResponsesProviderOptions applies provider-specific options to a Responses API body.
// Follows Vercel AI SDK pattern: maps flat keys to the nested Responses API format.
// Items 2, 4: add metadata, logprobs, user, safetyIdentifier, maxToolCalls,
// conversation, instructions, previousResponseId, strictJsonSchema, store.
func applyResponsesProviderOptions(body map[string]any, opts map[string]any) {
	if opts == nil {
		return
	}

	// Known options that should NOT be passed through to the body directly.
	consumed := map[string]bool{
		"structuredOutputs": true,
		"strictJsonSchema":  true,
		"useResponsesAPI":   true,
	}

	// Item 2: store from ProviderOptions (no longer hardcoded false).
	if v, ok := opts["store"]; ok {
		body["store"] = v
		consumed["store"] = true
	}
	if v, ok := opts["serviceTier"]; ok {
		body["service_tier"] = v
		consumed["serviceTier"] = true
	}
	if v, ok := opts["parallelToolCalls"]; ok {
		body["parallel_tool_calls"] = v
		consumed["parallelToolCalls"] = true
	}
	if v, ok := opts["truncation"]; ok {
		body["truncation"] = v
		consumed["truncation"] = true
	}
	if v, ok := opts["include"]; ok {
		body["include"] = v
		consumed["include"] = true
	}
	if v, ok := opts["prompt_cache_key"]; ok {
		body["prompt_cache_key"] = v
		consumed["prompt_cache_key"] = true
	}

	// Item 4: missing options.
	if v, ok := opts["metadata"]; ok {
		body["metadata"] = v
		consumed["metadata"] = true
	}
	if v, ok := opts["user"]; ok {
		body["user"] = v
		consumed["user"] = true
	}
	if v, ok := opts["safetyIdentifier"]; ok {
		body["safety_identifier"] = v
		consumed["safetyIdentifier"] = true
	}
	if v, ok := opts["maxToolCalls"]; ok {
		body["max_tool_calls"] = v
		consumed["maxToolCalls"] = true
	}
	if v, ok := opts["conversation"]; ok {
		body["conversation"] = v
		consumed["conversation"] = true
	}
	if v, ok := opts["instructions"]; ok {
		body["instructions"] = v
		consumed["instructions"] = true
	}
	if v, ok := opts["previousResponseId"]; ok {
		body["previous_response_id"] = v
		consumed["previousResponseId"] = true
	}

	// Logprobs -- item 4.
	if v, ok := opts["logprobs"]; ok {
		consumed["logprobs"] = true
		switch lp := v.(type) {
		case bool:
			if lp {
				body["top_logprobs"] = 20 // TOP_LOGPROBS_MAX per Vercel
				addIncludeKey(body, "message.output_text.logprobs")
			}
		case int:
			body["top_logprobs"] = lp
			addIncludeKey(body, "message.output_text.logprobs")
		case float64:
			body["top_logprobs"] = int(lp)
			addIncludeKey(body, "message.output_text.logprobs")
		}
	}

	// Reasoning: {effort, summary} -- Vercel wraps into nested "reasoning" object.
	reasoning := getOrCreateMap(body, "reasoning")
	if v, ok := opts["reasoning_effort"]; ok {
		reasoning["effort"] = v
		consumed["reasoning_effort"] = true
	}
	if v, ok := opts["reasoning_summary"]; ok {
		reasoning["summary"] = v
		consumed["reasoning_summary"] = true
	}
	if len(reasoning) > 0 {
		body["reasoning"] = reasoning
	}

	// text_verbosity → text.verbosity (Vercel: text: {verbosity: ...}).
	if v, ok := opts["text_verbosity"]; ok {
		text := getOrCreateMap(body, "text")
		text["verbosity"] = v
		body["text"] = text
		consumed["text_verbosity"] = true
	}

	// Auto-include reasoning.encrypted_content when store=false and reasoning is set.
	// Follows Vercel: if (store === false && isReasoningModel) addInclude("reasoning.encrypted_content")
	if body["store"] == false && len(reasoning) > 0 {
		addIncludeKey(body, "reasoning.encrypted_content")
	}

	// Protected keys that must not be overwritten by provider options.
	protectedKeys := map[string]bool{
		"model": true, "stream": true, "input": true,
		"instructions": true, "max_output_tokens": true,
		"temperature": true, "top_p": true, "stop": true,
		"tools": true, "tool_choice": true,
	}

	// Pass through any remaining unknown keys.
	for k, v := range opts {
		if !consumed[k] && !protectedKeys[k] {
			body[k] = v
		}
	}
}

// addIncludeKey adds a key to the "include" array in the body, avoiding duplicates.
func addIncludeKey(body map[string]any, key string) {
	var includes []string
	switch v := body["include"].(type) {
	case []string:
		includes = v
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				includes = append(includes, s)
			}
		}
	}
	if !slices.Contains(includes, key) {
		includes = append(includes, key)
	}
	body["include"] = includes
}

// convertToResponsesInput converts provider.Message slice to Responses API input format.
func convertToResponsesInput(msgs []provider.Message) []map[string]any {
	result := make([]map[string]any, 0, len(msgs))

	for _, msg := range msgs {
		switch msg.Role {
		case provider.RoleSystem:
			result = append(result, map[string]any{
				"role":    "developer",
				"content": partsToText(msg.Content),
			})

		case provider.RoleTool:
			for _, part := range msg.Content {
				if part.Type == provider.PartToolResult {
					result = append(result, map[string]any{
						"type":    "function_call_output",
						"call_id": part.ToolCallID,
						"output":  part.ToolOutput,
					})
				}
			}

		case provider.RoleAssistant:
			var items []map[string]any
			var textParts []string

			for _, part := range msg.Content {
				switch part.Type {
				case provider.PartText:
					textParts = append(textParts, part.Text)
				case provider.PartReasoning:
					if part.Text != "" {
						textParts = append(textParts, part.Text)
					}
				case provider.PartToolCall:
					items = append(items, map[string]any{
						"type":      "function_call",
						"call_id":   part.ToolCallID,
						"name":      part.ToolName,
						"arguments": string(part.ToolInput),
					})
				}
			}

			if len(textParts) > 0 {
				items = append([]map[string]any{{
					"type": "message",
					"role": "assistant",
					"content": []map[string]any{{
						"type": "output_text",
						"text": strings.Join(textParts, "\n"),
					}},
				}}, items...)
			}

			result = append(result, items...)

		case provider.RoleUser:
			var contentItems []map[string]any
			for _, part := range msg.Content {
				switch part.Type {
				case provider.PartText:
					if part.Text != "" {
						contentItems = append(contentItems, map[string]any{
							"type": "input_text",
							"text": part.Text,
						})
					}
				case provider.PartImage:
					contentItems = append(contentItems, map[string]any{
						"type":      "input_image",
						"image_url": part.URL,
					})
				case provider.PartFile:
					item := map[string]any{
						"type":      "input_file",
						"file_data": part.URL,
					}
					if part.Filename != "" {
						item["filename"] = part.Filename
					}
					contentItems = append(contentItems, item)
				}
			}
			if len(contentItems) == 0 {
				contentItems = []map[string]any{{
					"type": "input_text",
					"text": partsToText(msg.Content),
				}}
			}
			result = append(result, map[string]any{
				"role":    "user",
				"content": contentItems,
			})
		}
	}

	return result
}

func partsToText(parts []provider.Part) string {
	var texts []string
	for _, p := range parts {
		if p.Type == provider.PartText && p.Text != "" {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// --- Responses API SSE streaming ---

// responsesToolCall tracks an in-flight tool call by output_index.
type responsesToolCall struct {
	id   string
	name string
	args strings.Builder
}

// responsesReasoning tracks an in-flight reasoning block by output_index.
// Copilot rotates item_id mid-stream, so we track by output_index
// and use the canonical ID from output_item.added.
type responsesReasoning struct {
	canonicalID string
}

// streamResponses parses SSE from the OpenAI Responses API.
// Uses bufio.Scanner directly because Responses API has event-typed SSE
// (event: + data: pairs), unlike Chat Completions (data: only).
func streamResponses(ctx context.Context, body io.ReadCloser, out chan<- provider.StreamChunk) {
	defer close(out)
	defer func() { _ = body.Close() }()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var usage provider.Usage
	var eventType string
	var hasFunctionCall bool

	activeTools := make(map[int]*responsesToolCall)
	activeReasoning := make(map[int]*responsesReasoning)
	currentReasoningIdx := -1

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkFinish, Usage: usage}) {
				return
			}
			return
		}

		switch eventType {
		case "response.output_text.delta":
			var ev struct {
				Delta string `json:"delta"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil && ev.Delta != "" {
				if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: ev.Delta}) {
					return
				}
			}

		case "response.refusal.delta":
			var ev struct {
				Delta string `json:"delta"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil && ev.Delta != "" {
				if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkText, Text: ev.Delta}) {
					return
				}
			}

		case "response.reasoning_summary_text.delta":
			var ev struct {
				ItemID       string `json:"item_id"`
				SummaryIndex int    `json:"summary_index"`
				Delta        string `json:"delta"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil && ev.Delta != "" {
				// Use canonical ID from activeReasoning if available.
				id := ev.ItemID
				if currentReasoningIdx >= 0 {
					if ar := activeReasoning[currentReasoningIdx]; ar != nil {
						id = ar.canonicalID
					}
				}
				if !provider.TrySend(ctx, out, provider.StreamChunk{
					Type: provider.ChunkReasoning,
					Text: ev.Delta,
					Metadata: map[string]any{
						"reasoningId": fmt.Sprintf("%s:%d", id, ev.SummaryIndex),
					},
				}) {
					return
				}
			}

		case "response.reasoning_summary_part.added":
			// New summary segment within same reasoning item -- no-op for chunk emission
			// but tracked for canonical ID resolution.

		case "response.output_item.added":
			var ev struct {
				OutputIndex int `json:"output_index"`
				Item        struct {
					Type   string `json:"type"`
					ID     string `json:"id"`
					CallID string `json:"call_id"`
					Name   string `json:"name"`
				} `json:"item"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil {
				switch ev.Item.Type {
				case "function_call":
					hasFunctionCall = true
					activeTools[ev.OutputIndex] = &responsesToolCall{
						id:   ev.Item.CallID,
						name: ev.Item.Name,
					}
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:       provider.ChunkToolCallStreamStart,
						ToolCallID: ev.Item.CallID,
						ToolName:   ev.Item.Name,
					}) {
						return
					}
				case "reasoning":
					activeReasoning[ev.OutputIndex] = &responsesReasoning{
						canonicalID: ev.Item.ID,
					}
					currentReasoningIdx = ev.OutputIndex
				}
			}

		case "response.function_call_arguments.delta":
			var ev struct {
				OutputIndex int    `json:"output_index"`
				Delta       string `json:"delta"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil && ev.Delta != "" {
				active := activeTools[ev.OutputIndex]
				if active == nil {
					active = &responsesToolCall{}
					activeTools[ev.OutputIndex] = active
				}
				active.args.WriteString(ev.Delta)

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

		case "response.function_call_arguments.done":
			var ev struct {
				OutputIndex int `json:"output_index"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil {
				if active := activeTools[ev.OutputIndex]; active != nil {
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

		case "response.output_item.done":
			var ev struct {
				OutputIndex int `json:"output_index"`
				Item        struct {
					Type string `json:"type"`
				} `json:"item"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil {
				switch ev.Item.Type {
				case "reasoning":
					delete(activeReasoning, ev.OutputIndex)
					if currentReasoningIdx == ev.OutputIndex {
						currentReasoningIdx = -1
					}
				default:
					delete(activeTools, ev.OutputIndex)
				}
			}

		case "response.completed", "response.incomplete":
			var ev struct {
				Response struct {
					ID    string `json:"id"`
					Model string `json:"model"`
					IncompleteDetails *struct {
						Reason string `json:"reason"`
					} `json:"incomplete_details"`
					Usage struct {
						InputTokens  int `json:"input_tokens"`
						OutputTokens int `json:"output_tokens"`
						OutputTokensDetails *struct {
							ReasoningTokens int `json:"reasoning_tokens"`
						} `json:"output_tokens_details"`
						InputTokensDetails *struct {
							CachedTokens int `json:"cached_tokens"`
						} `json:"input_tokens_details"`
					} `json:"usage"`
				} `json:"response"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil {
				usage.InputTokens = ev.Response.Usage.InputTokens
				usage.OutputTokens = ev.Response.Usage.OutputTokens
				usage.TotalTokens = ev.Response.Usage.InputTokens + ev.Response.Usage.OutputTokens
				if ev.Response.Usage.OutputTokensDetails != nil {
					usage.ReasoningTokens = ev.Response.Usage.OutputTokensDetails.ReasoningTokens
				}
				if ev.Response.Usage.InputTokensDetails != nil {
					usage.CacheReadTokens = ev.Response.Usage.InputTokensDetails.CachedTokens
				}
				usage.InputTokens -= usage.CacheReadTokens
			}

			// Flush remaining tool call args.
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
				}
			}

			var incompleteReason string
			if ev.Response.IncompleteDetails != nil {
				incompleteReason = ev.Response.IncompleteDetails.Reason
			}
			finishReason := mapResponsesFinishReason(eventType, incompleteReason, hasFunctionCall)
			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:         provider.ChunkStepFinish,
				FinishReason: finishReason,
			}) {
				return
			}
			if !provider.TrySend(ctx, out, provider.StreamChunk{
				Type:     provider.ChunkFinish,
				Usage:    usage,
				Response: provider.ResponseMetadata{ID: ev.Response.ID, Model: ev.Response.Model},
			}) {
				return
			}
			return

		case "response.failed":
			var ev struct {
				Response struct {
					Error struct {
						Message string `json:"message"`
						Code    string `json:"code"`
					} `json:"error"`
				} `json:"response"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil {
				msg := cmp.Or(ev.Response.Error.Message, "response failed")
				code := ev.Response.Error.Code
				switch code {
				case "context_length_exceeded":
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: &goai.ContextOverflowError{Message: msg, ResponseBody: data},
					}) {
						return
					}
				case "insufficient_quota":
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: &goai.APIError{Message: "Quota exceeded. Check your plan and billing details.", IsRetryable: false},
					}) {
						return
					}
				case "usage_not_included":
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: &goai.APIError{Message: "Usage not included in response. Check your plan supports usage reporting for this model.", IsRetryable: false},
					}) {
						return
					}
				case "invalid_prompt":
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: &goai.APIError{Message: msg, IsRetryable: false},
					}) {
						return
					}
				default:
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: &goai.APIError{Message: msg},
					}) {
						return
					}
				}
			}
			return

		case "error":
			var ev struct {
				Message string `json:"message"`
				Code    string `json:"code"`
			}
			if json.Unmarshal([]byte(data), &ev) == nil {
				msg := cmp.Or(ev.Message, "stream error")
				if ev.Code == "context_overflow" || ev.Code == "max_tokens" ||
					ev.Code == "context_length_exceeded" {
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: &goai.ContextOverflowError{Message: msg, ResponseBody: data},
					}) {
						return
					}
				} else {
					if !provider.TrySend(ctx, out, provider.StreamChunk{
						Type:  provider.ChunkError,
						Error: &goai.APIError{Message: msg},
					}) {
						return
					}
				}
			}
			return
		}

		eventType = ""
	}

	if err := scanner.Err(); err != nil {
		if !provider.TrySend(ctx, out, provider.StreamChunk{Type: provider.ChunkError, Error: fmt.Errorf("reading stream: %w", err)}) {
			return
		}
	}
}

// mapResponsesFinishReason maps Responses API completion status to a FinishReason.
func mapResponsesFinishReason(eventType string, incompleteReason string, hasFunctionCall bool) provider.FinishReason {
	if hasFunctionCall {
		return provider.FinishToolCalls
	}
	if eventType == "response.incomplete" {
		switch incompleteReason {
		case "max_output_tokens":
			return provider.FinishLength
		case "content_filter":
			return provider.FinishContentFilter
		default:
			return provider.FinishOther
		}
	}
	return provider.FinishStop
}

// --- Responses API non-streaming ---

// responsesResult is the JSON structure of a non-streaming Responses API result.
type responsesResult struct {
	ID     string `json:"id"`
	Model  string `json:"model"`
	Status string `json:"status"`

	Output []struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type        string              `json:"type"`
			Text        string              `json:"text"`
			Annotations []responsesAnnotation `json:"annotations,omitempty"`
			Logprobs    *json.RawMessage    `json:"logprobs,omitempty"`
		} `json:"content,omitempty"`

		// function_call fields
		CallID    string `json:"call_id,omitempty"`
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`

		// reasoning fields
		ID      string `json:"id,omitempty"`
		Summary []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"summary,omitempty"`
	} `json:"output"`

	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		OutputTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"output_tokens_details,omitempty"`
		InputTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"input_tokens_details,omitempty"`
	} `json:"usage,omitempty"`

	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`

	Error *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

// responsesAnnotation represents a citation in a Responses API content part.
type responsesAnnotation struct {
	Type        string `json:"type"`
	URLCitation *struct {
		URL        string `json:"url"`
		Title      string `json:"title"`
		StartIndex int    `json:"start_index"`
		EndIndex   int    `json:"end_index"`
	} `json:"url_citation,omitempty"`
}

// parseResponsesResult parses a non-streaming Responses API JSON response.
func parseResponsesResult(body []byte) (*provider.GenerateResult, error) {
	var resp responsesResult
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("openai: parsing responses result: %w", err)
	}

	if resp.Error != nil {
		if resp.Error.Code == "context_length_exceeded" {
			return nil, &goai.ContextOverflowError{Message: resp.Error.Message, ResponseBody: string(body)}
		}
		return nil, &goai.APIError{Message: resp.Error.Message, ResponseBody: string(body)}
	}

	result := &provider.GenerateResult{
		Response: provider.ResponseMetadata{
			ID:    resp.ID,
			Model: resp.Model,
		},
	}

	// Extract text, tool calls, sources, logprobs from output.
	var textParts []string
	var hasFunctionCall bool
	providerMeta := map[string]any{}
	var allLogprobs []any

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" && c.Text != "" {
					textParts = append(textParts, c.Text)
				}
				// Item 11: extract annotations (url_citation).
				for _, ann := range c.Annotations {
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
				if c.Logprobs != nil {
					var lp any
					if json.Unmarshal(*c.Logprobs, &lp) == nil && lp != nil {
						allLogprobs = append(allLogprobs, lp)
					}
				}
			}
		case "function_call":
			hasFunctionCall = true
			result.ToolCalls = append(result.ToolCalls, provider.ToolCall{
				ID:    item.CallID,
				Name:  item.Name,
				Input: json.RawMessage(item.Arguments),
			})
		case "reasoning":
			for _, s := range item.Summary {
				if s.Text != "" {
					reasoning, _ := providerMeta["reasoning"].([]map[string]any)
					providerMeta["reasoning"] = append(reasoning, map[string]any{
						"type": s.Type,
						"text": s.Text,
					})
				}
			}
		}
	}

	result.Text = strings.Join(textParts, "")

	if len(allLogprobs) > 0 {
		providerMeta["logprobs"] = allLogprobs
	}

	// Finish reason.
	if hasFunctionCall {
		result.FinishReason = provider.FinishToolCalls
	} else if resp.Status == "incomplete" {
		reason := ""
		if resp.IncompleteDetails != nil {
			reason = resp.IncompleteDetails.Reason
		}
		switch reason {
		case "max_output_tokens":
			result.FinishReason = provider.FinishLength
		case "content_filter":
			result.FinishReason = provider.FinishContentFilter
		default:
			result.FinishReason = provider.FinishOther
		}
	} else {
		result.FinishReason = provider.FinishStop
	}

	// Usage -- Item 1: compute TotalTokens.
	if resp.Usage != nil {
		result.Usage.InputTokens = resp.Usage.InputTokens
		result.Usage.OutputTokens = resp.Usage.OutputTokens
		result.Usage.TotalTokens = resp.Usage.InputTokens + resp.Usage.OutputTokens
		if resp.Usage.OutputTokensDetails != nil {
			result.Usage.ReasoningTokens = resp.Usage.OutputTokensDetails.ReasoningTokens
		}
		if resp.Usage.InputTokensDetails != nil {
			result.Usage.CacheReadTokens = resp.Usage.InputTokensDetails.CachedTokens
		}
		result.Usage.InputTokens -= result.Usage.CacheReadTokens
	}

	if len(providerMeta) > 0 {
		result.ProviderMetadata = map[string]map[string]any{"openai": providerMeta}
	}

	return result, nil
}

// getOrCreateMap returns the existing map[string]any at body[key], or a new empty map.
func getOrCreateMap(body map[string]any, key string) map[string]any {
	if m, ok := body[key].(map[string]any); ok {
		return m
	}
	return map[string]any{}
}
